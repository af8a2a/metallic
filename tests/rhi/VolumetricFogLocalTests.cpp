#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"
#include "stb/stb_image_write.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <numbers>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

template<typename T>
T checked(Result<T> result, const char* operation)
{
    if (!result) { throw std::runtime_error(std::string(operation) + ": " + resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

constexpr uint32_t kWidth = 128, kHeight = 96;
constexpr std::array<std::array<uint32_t, 2>, 5> kPixels{{{64, 48}, {56, 40}, {78, 48}, {32, 24}, {6, 8}}};
using Samples = std::array<float, kPixels.size()>;

// Actual UE 5.7.4 D3D12 SceneCapture2D scene-color readback, linear HDR.
// +X Unreal maps to -Z Metallic. The white unlit plane is 40 m from the eye.
struct Probe {
    std::string_view name;
    Samples reference;
    bool capturedInUnreal = true;
};
constexpr Samples kGlobal{0.402099609375f, 0.402099609375f, 0.3935546875f, 0.34765625f, 0.2685546875f};
constexpr std::array kProbes{
    Probe{"clear", {1, 1, 1, 1, 1}}, Probe{"base", kGlobal},
    Probe{"point", {2.759765625f, 2.814453125f, 2.37890625f, 1.216796875f, 0.4375f}},
    Probe{"pointOff", kGlobal}, Probe{"pointFar", kGlobal},
    Probe{"spot", {0.595703125f, 0.5986328125f, 0.5078125f, 0.34765625f, 0.2685546875f}},
    Probe{"local", {0.011688232421875f, 0.01177978515625f, 0.1617431640625f, 0.34765625f, 0.2685546875f}},
    Probe{"emissive", {1.107421875f, 1.1064453125f, 0.81201171875f, 0.34765625f, 0.2685546875f}},
    Probe{"pointUnbounded", {}, false}, Probe{"pointLongRange", {}, false}, Probe{"pointOverflow", {}, false}
};

std::filesystem::path writeWhiteWall(const std::filesystem::path& directory)
{
    std::filesystem::create_directories(directory);
    constexpr std::array<float, 9> vertices{-120, -100, -40, 120, -100, -40, 0, 150, -40};
    std::ofstream binary(directory / "wall.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(vertices.data()), sizeof(vertices));
    std::ofstream gltf(directory / "wall.gltf");
    gltf << R"json({
        "asset":{"version":"2.0"},"extensionsUsed":["KHR_materials_unlit"],
        "scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0}]}],
        "materials":[{"doubleSided":true,"pbrMetallicRoughness":{"baseColorFactor":[1,1,1,1]},
            "extensions":{"KHR_materials_unlit":{}}}],
        "buffers":[{"uri":"wall.bin","byteLength":36}],"bufferViews":[{"buffer":0,"byteLength":36}],
        "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",
            "min":[-120,-100,-40],"max":[120,150,-40]}]
    })json";
    const auto path = directory / "probe.metallic_scene.json";
    std::ofstream document(path);
    document << R"json({"version":4,"nodes":[],"sources":[{"id":"wall","path":"wall.gltf"}],
        "world":{"environment":{"enabled":false,"visible":false,"sun":{"enabled":false},"moon":{"enabled":false}},
            "lighting":{"autoExposure":{"enabled":false},"exposureEV100":0,"lights":[]}}})json";
    if (!binary || !gltf || !document) { throw std::runtime_error("Could not write fog test scene"); }
    return path;
}

// Match UE's unlit SceneCapture wall independently of surface shading. Raster
// visibility still supplies the measured depth, camera and GPUScene light grid.
class WhiteFogProbeWall final : public UnsafePass
{
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection result;
        result.addTextureOutput("color", "Uniform unlit radiance").transferWrite()
            .transient(RenderGraphInitialization::FullOverwrite).format = Format::RGBA32Sfloat;
        return result;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const float radiance = properties().value("radiance", 1.0f);
        return context.commandBuffer().clearColorTexture(*context.outputTexture("color").texture(),
            TextureLayout::TransferDestination, {radiance, radiance, radiance, 1});
    }
};

RenderGraph makeFogGraph(const RenderView& view, bool worldShadows = false)
{
    static const bool registered = registerRenderGraphPassType("WhiteFogProbeWall", "Unlit white fog probe wall",
        [] { return std::make_unique<WhiteFogProbeWall>(); });
    if (!registered) { throw std::runtime_error("Cannot register white wall fixture"); }
    RenderGraph graph;
    graph.setViewProperties({{"camera", view.cameraProperties()}, {"renderResolution", view.renderResolutionProperties()},
        {"temporalJitter", false}});
    graph.addNode("VisibilityBufferPass", "VBuffer", {{"initialLoad", true}, {"enableClusterRtx", false},
        {"autoLod", false}, {"meshletNormalConeCull", false}, {"camera", view.cameraProperties()}});
    graph.addNode("WhiteFogProbeWall", "Wall");
    graph.addNode("VolumetricFogPass", "Fog", {{"gridPixelSize", worldShadows ? 4 : 16}, {"gridSizeZ", 64},
        {"conservativeDepth", false}, {"temporalReprojection", false}, {"jitter", false}, {"localFogVolumes", true}});
    graph.addNode("VolumetricFogCompositePass", "Composite");
    graph.addEdge("VBuffer.depth", "Fog.depth");
    graph.addEdge("VBuffer.rasterInfo", "Fog.rasterInfo");
    graph.addEdge("Wall.color", "Composite.sceneColor");
    graph.addEdge("VBuffer.depth", "Composite.depth");
    graph.addEdge("Fog.integrated", "Composite.integrated");
    graph.addEdge("Fog.view", "Composite.view");
    graph.markOutput("Composite.color");
    graph.markOutput("VBuffer.depth");
    if (worldShadows) {
        // One coarse page per face. No surface receiver requests a fine page for the air under test.
        graph.addNode("VirtualShadowMapLightsPass", "Shadows", {{"publishWorldShadows", true},
            {"compactOutputs", true}, {"debugOutput", false}, {"lightCapacity", 4},
            {"physicalPageBudget", 7}, {"maxWorkItems", 4096}});
        graph.addEdge("VBuffer.depth", "Shadows.depth");
        graph.addEdge("VBuffer.rasterInfo", "Shadows.rasterInfo");
        graph.addEdge("Shadows.worldShadowFaces", "Fog.worldShadowFaces");
        graph.markOutput("Fog.scattering");
    }
    return graph;
}

void configureProbe(RenderWorld& world, std::string_view name)
{
    auto env = world.worldEnvironment();
    auto& fog = env.volumetricFog;
    fog.enabled = name != "clear";
    fog.density = 0.45f;
    fog.heightFalloff = 0;
    fog.scatteringDistribution = 0;
    fog.distanceMeters = 60;
    fog.localVolumes.clear();
    if (name == "local" || name == "emissive") {
        environment::LocalFogVolume local;
        local.centerMeters = {0, 0, -30};
        local.radiusMeters = 10;
        local.heightFalloff = 1;
        local.albedo = {0, 0, 0};
        if (name == "emissive") { local.emissive = {2, 2, 2}; }
        fog.localVolumes.push_back(local);
    }
    world.setWorldEnvironment(std::move(env));

    scene::LightingSettings lights;
    lights.autoExposure.enabled = false;
    if (name.starts_with("point") || name == "spot") {
        scene::PunctualLight light;
        light.properties.type = name == "spot" ? "spot" : "point";
        light.properties.intensity = 8000;
        light.properties.intensityUnit = scene::LightUnit::Candela;
        light.properties.range = 18;
        if (name == "pointUnbounded") { light.properties.range = 0; }
        if (name == "pointLongRange") { light.properties.range = 1.0e6; }
        light.properties.innerConeAngle = std::numbers::pi / 12.0;
        light.properties.outerConeAngle = std::numbers::pi / 6.0;
        light.properties.volumetricScatteringIntensity = name == "pointOff" ? 0.0 : 1.0;
        light.direction = float3(0, 0, -1);
        light.position = float3(name == "pointFar" ? 1000.0f : 0.0f, 0, name == "spot" ? -10.0f : -20.0f);
        if (name == "pointOverflow") {
            // Put the sole contributing light after the cluster's 64-entry limit.
            auto filler = light;
            filler.properties.volumetricScatteringIntensity = 0;
            lights.lights.assign(65, filler);
        }
        lights.lights.push_back(light);
    }
    if (!world.setLighting(std::move(lights))) { throw std::runtime_error("Invalid test lights"); }
}

class VolumetricFogLocalTest final : public RHITest
{
public:
    VolumetricFogLocalTest()
    {
        name = "volumetric_fog_local_ue_reference";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        const auto& caps = context.device.capabilities();
        if (!caps.bindlessDescriptorHeap || !caps.meshShader || !caps.taskShader || !caps.geometryShader) {
            return RHITestResult::skip("Fog scene rendering needs realtime raster device features");
        }
        try {
            const auto directory = context.outputDirectory / "local-fog-reference";
            scene::SceneDocument document;
            if (!document.load(writeWhiteWall(directory))) {
                return RHITestResult::fail(document.documentWarning() + document.lastLoadResult().error);
            }
            RenderWorld world;
            world.setScene(&document);
            RenderView view;
            if (!view.setRenderResolution(kWidth, kHeight)) { return RHITestResult::fail("Invalid fog probe render resolution"); }
            if (!view.setCameraProperties({{"eye", {0, 0, 0}}, {"center", {0, 0, -1}}, {"up", {0, 1, 0}},
                {"fovDegrees", 2.0 * std::atan(0.75) * 180.0 / std::numbers::pi}, {"znear", 0.05}, {"zfar", 1000.0}})) {
                return RHITestResult::fail("Invalid fog probe camera");
            }
            view.setTemporalJitter(false);
            const auto graph = makeFogGraph(view);
            RenderGraphExecutor executor;
            executor.bindRenderWorld(&world);
            executor.bindRenderView(&view);
            std::string log;
            if (!executor.compile(context.device, graph, kWidth, kHeight, log)) { return RHITestResult::fail(log); }
            const auto& frame = executor.outputResource("VBuffer.depth")->desc;
            const auto& volume = executor.outputResource("Fog.integrated")->desc;
            if (frame.width != kWidth || frame.height != kHeight || volume.width != 8 || volume.height != 6 || volume.depth != 64) {
                return RHITestResult::fail("Fog reference requires a 128x96 native raster and 8x6x64 volume");
            }
            constexpr uint64_t imageBytes = kWidth * kHeight * sizeof(float);
            auto buffer = checked(context.device.createBuffer({.size = 5 * imageBytes,
                .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}), "readback");
            nlohmann::json measured;
            Samples globalOnly{}, bounded{}, unbounded{};
            float maxError = 0;
            std::string failure;
            for (const auto& probe : kProbes) {
                configureProbe(world, probe.name);
                checked(executor.execute({.graphicsQueue = &context.graphicsQueue}), "render fog scene");
                checked(executor.waitForSubmittedWork(), "wait for fog scene");
                bench::GPUCommands copy(context.graphicsQueue);
                checked(copy.initialize(context.device), "begin texture read");
                checked(executor.transitionOutput(*copy.commands, "Composite.color", ResourceState::TransferSource), "transition output");
                checked(copy.commands->copyTextureToBuffer({.texture = executor.outputResource("Composite.color")->texture,
                    .buffer = checked(buffer->slice({0, 4 * imageBytes}), "readback range"), .width = kWidth, .height = kHeight}), "read fog output");
                if (probe.name == "clear") {
                    checked(executor.transitionOutput(*copy.commands, "VBuffer.depth", ResourceState::TransferSource), "transition depth");
                    checked(copy.commands->copyTextureToBuffer({.texture = executor.outputResource("VBuffer.depth")->texture,
                        .buffer = checked(buffer->slice({4 * imageBytes, imageBytes}), "depth range"), .width = kWidth, .height = kHeight}), "read wall depth");
                }
                const BufferBarrierDesc host{.buffer = buffer.get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::Host, AccessBits::HostRead}};
                checked(copy.commands->synchronize({.buffers = {&host, 1}}), "publish readback");
                checked(copy.submitAndWait(), "submit readback");
                buffer->invalidate();
                const auto* pixels = static_cast<const float*>(buffer->map());
                if (!pixels) { throw std::runtime_error("Cannot map fog readback"); }
                if (probe.name == "clear") {
                    auto* info = executor.outputResource("VBuffer.rasterInfo")->buffer;
                    auto* camera = static_cast<const VisibilityBufferFrameInfo*>(info->map());
                    if (!camera) { throw std::runtime_error("Cannot map raster camera"); }
                    measured["eyeMeters"] = {camera->eye[0], camera->eye[1], camera->eye[2]};
                    const float nearZ = camera->clipOrtho[0], farZ = camera->clipOrtho[1];
                    info->unmap();
                    for (auto pixel : kPixels) {
                        const float deviceZ = pixels[4 * kWidth * kHeight + pixel[1] * kWidth + pixel[0]];
                        const float depth = nearZ * farZ / (nearZ + deviceZ * (farZ - nearZ));
                        measured["depthMeters"].push_back(depth);
                        if (std::abs(depth - 40) > 0.01f && failure.empty()) { failure = "Wall raster depth " + std::to_string(depth); }
                    }
                }
                Samples actual{};
                for (size_t i = 0; i < kPixels.size(); ++i) {
                    const auto* rgb = pixels + (kPixels[i][1] * kWidth + kPixels[i][0]) * 4;
                    actual[i] = rgb[0];
                    for (uint32_t channel = 0; channel < 3; ++channel) {
                        const float error = std::abs(rgb[channel] - probe.reference[i]);
                        if (probe.capturedInUnreal) { maxError = std::max(maxError, error); }
                        // UE's scene-color target is half precision. Allow two half ULPs plus one interpolation ULP.
                        if ((!std::isfinite(error) || (probe.capturedInUnreal && error > 0.0002f + 0.002f * probe.reference[i])) &&
                            failure.empty()) {
                            failure = std::string(probe.name) + " sample " + std::to_string(i) + " expected " +
                                std::to_string(probe.reference[i]) + ", read " + std::to_string(rgb[channel]);
                        }
                    }
                }
                buffer->unmap();
                if (probe.name == "base") { globalOnly = actual; }
                if (probe.name == "point") { bounded = actual; }
                if (probe.name == "pointUnbounded") { unbounded = actual; }
                for (size_t i = 0; i < kPixels.size(); ++i) {
                    const bool inactive = probe.name == "pointOff" || probe.name == "pointFar";
                    const bool outside = i >= 3 && (probe.name == "spot" || probe.name == "local" || probe.name == "emissive");
                    if ((inactive || outside) && actual[i] != globalOnly[i] && failure.empty()) {
                        failure = std::string(probe.name) + " changed an unaffected pixel " + std::to_string(i);
                    }
                    if (probe.name == "pointLongRange" && (actual[i] != unbounded[i] || actual[i] <= bounded[i]) && failure.empty()) {
                        failure = "Unbounded and long-range lights disagree or lost radiance at pixel " + std::to_string(i);
                    }
                    if (probe.name == "pointOverflow" && actual[i] != bounded[i] && failure.empty()) {
                        failure = "Overflow cluster lost or duplicated the active light at pixel " + std::to_string(i);
                    }
                }
                auto& row = measured[std::string(probe.name)];
                row["metallic"] = actual;
                if (probe.capturedInUnreal) { row["unreal"] = probe.reference; }
            }
            measured["maxAbsoluteError"] = maxError;
            measured["pixels"] = kPixels;
            std::ofstream(directory / "comparison.json") << measured.dump(2);
            if (!failure.empty()) { return RHITestResult::fail(failure + ", max error " + std::to_string(maxError)); }
            return RHITestResult::pass("40 UE/Metallic values plus unbounded and overflow light lists, max reference error " +
                std::to_string(maxError));
        } catch (const std::exception& e) { return RHITestResult::fail(e.what()); }
    }
};

METALLIC_REGISTER_RHI_TEST(VolumetricFogLocalTest);

std::filesystem::path writeShadowDoors(const std::filesystem::path& directory)
{
    const auto document = writeWhiteWall(directory);
    const auto path = directory / "wall.gltf";
    auto asset = nlohmann::json::parse(std::ifstream(path));
    // The 16 m wide orthographic view sees only the backdrop. Put two walls
    // outside it at x = +/-10 m, with doorways at different depths.
    constexpr float quad[]{0,-1,-1, 0,1,-1, 0,1,1, 0,-1,-1, 0,1,1, 0,-1,1};
    std::ofstream binary(directory / "panel.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(quad), sizeof(quad));
    asset["buffers"].push_back({{"uri", "panel.bin"}, {"byteLength", sizeof(quad)}});
    asset["bufferViews"].push_back({{"buffer", 1}, {"byteLength", sizeof(quad)}});
    asset["accessors"].push_back({{"bufferView", 1}, {"componentType", 5126}, {"count", 6},
        {"type", "VEC3"}, {"min", {0,-1,-1}}, {"max", {0,1,1}}});
    asset["meshes"].push_back(asset["meshes"][0]);
    asset["meshes"][1]["primitives"][0]["attributes"]["POSITION"] = 1;
    auto panel = [&](float x, float yMin, float yMax, float zMin, float zMax) {
        asset["scenes"][0]["nodes"].push_back(asset["nodes"].size());
        asset["nodes"].push_back({{"mesh", 1}, {"translation", {x, 0.5f*(yMin+yMax), 0.5f*(zMin+zMax)}},
            {"scale", {1, 0.5f*(yMax-yMin), 0.5f*(zMax-zMin)}}});
    };
    for (float x : {-10.0f, 10.0f}) {
        const float door = x < 0 ? -15.0f : -20.0f;
        panel(x, -12, 12, -39, door - 2);
        panel(x, -12, 12, door + 2, -1);
        panel(x, 1, 12, door - 2, door + 2);
    }
    std::ofstream gltf(path);
    gltf << asset.dump();
    if (!binary || !gltf) { throw std::runtime_error("Could not write offscreen shadow doors"); }
    return document;
}

class VolumetricFogPunctualShadowTest final : public RHITest
{
public:
    VolumetricFogPunctualShadowTest()
    {
        name = "volumetric_fog_punctual_vsm_shadows";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        const auto& caps = context.device.capabilities();
        if (!caps.bindlessDescriptorHeap || !caps.meshShader || !caps.taskShader || !caps.geometryShader) {
            return RHITestResult::skip("Fog VSM scene requires --rhi-realtime");
        }
        try {
            const auto directory = context.outputDirectory / "fog-vsm-doors";
            scene::SceneDocument document;
            if (!document.load(writeShadowDoors(directory))) {
                return RHITestResult::fail(document.documentWarning() + document.lastLoadResult().error);
            }
            RenderWorld world;
            world.setScene(&document);
            configureProbe(world, "base");
            scene::LightingSettings lighting;
            for (bool spot : {false, true}) {
                scene::PunctualLight light;
                light.properties.type = spot ? "spot" : "point";
                light.position = {spot ? 20.0f : -20.0f, 0, -15};
                light.direction = {spot ? -1.0f : 1.0f, 0, 0};
                light.properties.intensity = 8000;
                light.properties.intensityUnit = scene::LightUnit::Candela;
                light.properties.range = 60;
                light.properties.innerConeAngle = 0.75;
                light.properties.outerConeAngle = 1.0;
                lighting.lights.push_back(light);
            }
            RenderView view;
            view.setRenderResolution(kWidth, kHeight);
            view.setCamera({.eye = {0, 0, 0}, .center = {0, 0, -1}, .farPlane = 100,
                .orthoHeight = 12, .orthographic = true});
            RenderGraphExecutor executor;
            executor.bindRenderWorld(&world);
            executor.bindRenderView(&view);
            std::string log;
            const auto errors = context.validationMessageCount ? context.validationMessageCount->load() : 0;
            const auto graph = makeFogGraph(view, true);
            auto mismatched = graph;
            mismatched.addNode("VisibilityBufferPass", "OtherView");
            const auto edge = std::find_if(mismatched.edges().begin(), mismatched.edges().end(),
                [](const auto& candidate) { return candidate.dstPass == "Fog" && candidate.dstField == "rasterInfo"; });
            mismatched.removeEdge(edge->id);
            mismatched.addEdge("OtherView.rasterInfo", "Fog.rasterInfo");
            if (mismatched.validate(log) || log.find("share the same rasterInfo") == std::string::npos) {
                return RHITestResult::fail("Cross-view world shadow resources were not rejected: " + log);
            }
            if (!executor.compile(context.device, graph, kWidth, kHeight, log)) {
                return RHITestResult::fail(log);
            }
            constexpr uint32_t nx = kWidth / 4, ny = kHeight / 4, nz = 64;
            constexpr uint64_t volumeBytes = nx * ny * nz * 8, colorBytes = kWidth * kHeight * 16;
            constexpr uint64_t depthBytes = kWidth * kHeight * 4;
            const auto& volume = executor.outputResource("Fog.scattering")->desc;
            if (volume.width != nx || volume.height != ny || volume.depth != nz) {
                return RHITestResult::fail("Unexpected fog shadow probe grid");
            }
            auto readback = checked(context.device.createBuffer({.size = volumeBytes + colorBytes + depthBytes,
                .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}), "readback");
            // The center froxel column is at x = 0.25 m. At z slices 13, 25 and 37 its depths are
            // 6.92, 14.86 and 25.15 m. The latter two see only the left and right doorway, respectively.
            constexpr std::array<uint32_t, 4> slices{13, 25, 37, 0};
            constexpr std::array<std::array<uint32_t, 2>, 6> referencePixels{{
                {64,48}, {64,0}, {32,48}, {96,48}, {32,12}, {96,12}}};
            nlohmann::json sceneColor{{"pixels", referencePixels}};
            nlohmann::json observations;
            std::string failure;
            for (size_t lamp = 0; lamp < lighting.lights.size(); ++lamp) {
                const auto& type = lighting.lights[lamp].properties.type;
                lighting.lights[1 - lamp].properties.volumetricScatteringIntensity = 0;
                lighting.lights[lamp].properties.volumetricScatteringIntensity = 1;
                std::array<uint16_t, 4> baseline{};
                for (int frame = 0; frame < 3; ++frame) {
                    const bool shadowed = frame == 1;
                    lighting.lights[lamp].properties.castVolumetricShadow = shadowed;
                    if (!world.setLighting(lighting)) { return RHITestResult::fail("Invalid shadow probe lighting"); }
                    checked(executor.execute({.graphicsQueue = &context.graphicsQueue}), "render shadow doors");
                    checked(executor.waitForSubmittedWork(), "wait for shadow doors");
                    bench::GPUCommands capture(context.graphicsQueue);
                    checked(capture.initialize(context.device), "begin shadow readback");
                    const auto copy = [&](const char* name, uint64_t offset, uint64_t size) {
                        const auto& resource = *executor.outputResource(name);
                        checked(executor.transitionOutput(*capture.commands, name, ResourceState::TransferSource), "transition capture");
                        checked(capture.commands->copyTextureToBuffer({.texture = resource.texture,
                            .buffer = checked(readback->slice({offset, size}), "capture slice"), .width = resource.desc.width,
                            .height = resource.desc.height, .depth = resource.desc.depth}), "copy output");
                    };
                    copy("Fog.scattering", 0, volumeBytes);
                    copy("Composite.color", volumeBytes, colorBytes);
                    copy("VBuffer.depth", volumeBytes + colorBytes, depthBytes);
                    const BufferBarrierDesc host{.buffer = readback.get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
                    checked(capture.commands->synchronize({.buffers = {&host, 1}}), "publish capture");
                    checked(capture.submitAndWait(), "submit capture");
                    readback->invalidate();
                    const auto* bytes = static_cast<const std::byte*>(readback->map());
                    if (!bytes) { throw std::runtime_error("Cannot map shadow door capture"); }
                    const auto* scattering = reinterpret_cast<const uint16_t*>(bytes);
                    const auto* color = reinterpret_cast<const float*>(bytes + volumeBytes);
                    const auto* depth = reinterpret_cast<const float*>(bytes + volumeBytes + colorBytes);
                    const std::string captureName = type + (shadowed ? "-shadowed" : "-unshadowed");
                    if (frame < 2) {
                        for (const auto& [x, y] : referencePixels) {
                            const auto* rgb = color + (y * kWidth + x) * 4;
                            sceneColor["cases"][captureName].push_back({rgb[0], rgb[1], rgb[2]});
                        }
                    }
                    std::array<uint16_t, 4> actual{};
                    for (size_t i = 0; i < actual.size(); ++i) {
                        const bool lintel = i == 3;
                        const uint32_t z = lintel ? slices[lamp + 1] : slices[i];
                        const uint32_t y = lintel ? 0 : ny / 2;
                        actual[i] = scattering[4 * ((z * ny + y) * nx + nx / 2)];
                        if (frame == 0) { baseline[i] = actual[i]; }
                        const bool visible = !shadowed || i == lamp + 1;
                        // Visibility is binary inside these regions. Allow a few FP16 ULPs for the UE
                        // shadowed-light color path; the ordinary light grid quantizes RGB to 10 bits.
                        const bool correct = visible ? actual[i] > 0 && actual[i] < 0x7c00 &&
                            std::abs(int(actual[i]) - int(baseline[i])) <= 4 : actual[i] == 0;
                        if (!correct && failure.empty()) {
                            failure = type + " frame " + std::to_string(frame) + " sample " + std::to_string(i) +
                                " expected " + (visible ? "lit" : "shadowed") + ", FP16 " + std::to_string(actual[i]);
                        }
                    }
                    observations[type].push_back(actual);
                    std::vector<uint8_t> image(kWidth * kHeight * 4);
                    for (uint32_t i = 0; i < kWidth * kHeight; ++i) {
                        const float distance = 100.0f - depth[i] * (100.0f - 0.05f);
                        if (std::abs(distance - 40.0f) > 0.02f && failure.empty()) {
                            failure = "An offscreen door entered the primary raster";
                        }
                        for (int c = 0; c < 3; ++c) {
                            const float l = std::max(0.0f, color[i * 4 + c]);
                            image[i * 4 + c] = uint8_t(255.0f * std::sqrt(l / (1.0f + l)));
                        }
                        image[i * 4 + 3] = 255;
                    }
                    readback->unmap();
                    if (frame < 2 && !saveRgba8Png(directory / (captureName + ".png"),
                        image.data(), kWidth, kHeight, log)) { return RHITestResult::fail(log); }
                }
            }
            std::ofstream(directory / "scattering-fp16.json") << observations.dump(2);
            std::ofstream(directory / "metallic-scene-color.json") << sceneColor.dump(2);
            if (!failure.empty()) { return RHITestResult::fail(failure); }
            if (context.validationMessageCount && errors != context.validationMessageCount->load()) {
                return RHITestResult::fail("Vulkan validation reported a shadow door graph error");
            }
            return RHITestResult::pass("Point and spot doors block distinct depths of the same camera ray, including offscreen casters");
        } catch (const std::exception& e) { return RHITestResult::fail(e.what()); }
    }
};

METALLIC_REGISTER_RHI_TEST(VolumetricFogPunctualShadowTest);

struct FogGIMeasurement {
    std::array<double, 3> scatter{};
    std::array<double, 3> hdr{};
    double extinction = 0;
};

float decodeFogHalf(uint16_t bits)
{
    const int exponent = (bits >> 10) & 31;
    const int mantissa = bits & 1023;
    if (exponent == 31) { throw std::runtime_error("Nonfinite scattering in fog GI capture"); }
    return std::ldexp(float(exponent ? 1024 + mantissa : mantissa), exponent ? exponent - 25 : -24) *
        (bits & 0x8000u ? -1.0f : 1.0f);
}

FogGIMeasurement readFogGI(RHITestContext& context, RenderGraphExecutor& executor,
    const std::filesystem::path& directory, const std::string& name)
{
    const auto& volume = executor.outputResource("Fog.scattering")->desc;
    const uint64_t scatterBytes = uint64_t(volume.width) * volume.height * volume.depth * 8;
    constexpr uint64_t colorBytes = kWidth * kHeight * 16;
    auto buffer = checked(context.device.createBuffer({.size = scatterBytes + colorBytes,
        .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}), "fog GI readback");
    bench::GPUCommands copy(context.graphicsQueue);
    checked(copy.initialize(context.device), "initialize fog GI readback");
    const auto read = [&](const char* output, uint64_t offset, uint64_t size) {
        const auto& resource = *executor.outputResource(output);
        checked(executor.transitionOutput(*copy.commands, output, ResourceState::TransferSource), "transition fog GI output");
        checked(copy.commands->copyTextureToBuffer({.texture = resource.texture,
            .buffer = checked(buffer->slice({offset, size}), "fog GI buffer range"),
            .width = resource.desc.width, .height = resource.desc.height, .depth = resource.desc.depth}), "read fog GI output");
    };
    read("Fog.scattering", 0, scatterBytes);
    read("Composite.color", scatterBytes, colorBytes);
    const BufferBarrierDesc barrier{.buffer = buffer.get(), .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
    checked(copy.commands->synchronize({.buffers = {&barrier, 1}}), "publish fog GI readback");
    checked(copy.submitAndWait(), "submit fog GI readback");
    buffer->invalidate();
    const auto* bytes = static_cast<const std::byte*>(buffer->map());
    if (!bytes) { throw std::runtime_error("Cannot map fog GI readback"); }
    const auto* volumePixels = reinterpret_cast<const uint16_t*>(bytes);
    const auto* pixels = reinterpret_cast<const float*>(bytes + scatterBytes);
    FogGIMeasurement result;
    uint32_t samples = 0;
    // These central cells are in air, between the eye and the back wall in every scene.
    for (uint32_t z = 12; z <= 34; ++z) {
        for (uint32_t y = volume.height / 3; y < volume.height * 2 / 3; ++y) {
            for (uint32_t x = volume.width / 4; x < volume.width / 2; ++x) {
                const auto* cell = volumePixels + 4 * ((z * volume.height + y) * volume.width + x);
                for (uint32_t c = 0; c < 3; ++c) { result.scatter[c] += decodeFogHalf(cell[c]); }
                result.extinction += decodeFogHalf(cell[3]);
                ++samples;
            }
        }
    }
    for (double& c : result.scatter) { c /= samples; }
    result.extinction /= samples;
    samples = 0;
    std::vector<uint8_t> image(kWidth * kHeight * 4, 255);
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            const bool sample = x >= kWidth / 4 && x < kWidth / 2 && y >= kHeight / 3 && y < kHeight * 2 / 3;
            for (uint32_t c = 0; c < 3; ++c) {
                const auto i = 4 * (y * kWidth + x) + c;
                const float v = pixels[i];
                if (!std::isfinite(v) || v < 0) { throw std::runtime_error("Invalid scene color in " + name); }
                if (sample) { result.hdr[c] += v; }
                image[i] = uint8_t(255.0f * std::sqrt(v / (1.0f + v)));
            }
            if (sample) { ++samples; }
        }
    }
    for (double& c : result.hdr) { c /= samples; }
    buffer->unmap();
    std::string error;
    if (!saveRgba8Png(directory / (name + ".png"), image.data(), kWidth, kHeight, error)) {
        throw std::runtime_error(error);
    }
    return result;
}

class VolumetricFogLumenTest final : public RHITest
{
public:
    VolumetricFogLumenTest()
    {
        name = "volumetric_fog_lumen_wall_bounce_and_sky_visibility";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        const auto& caps = context.device.capabilities();
        if (!caps.bindlessDescriptorHeap || !caps.meshShader || !caps.taskShader || !caps.geometryShader) {
            return RHITestResult::skip("Lumen fog scene requires --rhi-realtime");
        }
        try {
            const auto directory = std::filesystem::absolute(context.outputDirectory / "fog-lumen");
            std::filesystem::create_directories(directory);
            const auto root = std::filesystem::path(PROJECT_SOURCE_DIR);
            scene::SceneDocument document;
            if (!document.load(root / "Asset/Lumen/CornellBox.metallic_scene.json")) {
                return RHITestResult::fail(document.lastLoadResult().error);
            }
            for (int node : {5, 6, 7}) { document.setObjectVisible(document.objectForNode(node).entity(), false); }
            RenderWorld world;
            world.setScene(&document);
            world.setEnvironment(document.environment());
            auto env = document.worldEnvironment();
            env.volumetricFog.enabled = true;
            env.volumetricFog.density = 5.0f;
            env.volumetricFog.heightFalloff = 0;
            env.volumetricFog.scatteringDistribution = 0;
            env.volumetricFog.distanceMeters = 4;
            world.setWorldEnvironment(env);
            auto lights = document.lighting();
            scene::PunctualLight light;
            light.properties.type = "spot";
            light.properties.intensity = 60;
            light.properties.intensityUnit = scene::LightUnit::Candela;
            light.properties.range = 3;
            light.properties.outerConeAngle = 1.05;
            light.properties.innerConeAngle = 0.65;
            light.properties.volumetricScatteringIntensity = 0;
            light.direction = float3(-1, 0, 0);
            light.position = float3(-0.25f, 1, 0);
            lights.lights = {light};
            if (!world.setLighting(lights)) { return RHITestResult::fail("Invalid wall spotlight"); }
            RenderView view;
            view.setRenderResolution(kWidth, kHeight);
            view.setCameraProperties({{"eye", {0, 1, 0.9}}, {"center", {0, 1, 0}}, {"up", {0, 1, 0}},
                {"fovDegrees", 60}, {"znear", 0.05}, {"zfar", 10}});
            view.setTemporalJitter(false);
            auto graph = makeFogGraph(view);
            // A black background isolates integrated in-scattering while preserving actual surface depth.
            graph.setNodeProperties(graph.findNode("Wall")->id, {{"radiance", 0.0f}});
            graph.markOutput("Fog.scattering");
            RenderGraphExecutor executor;
            executor.bindRenderWorld(&world);
            executor.bindRenderView(&view);
            std::string error;
            if (!executor.subsystemHost()->configure<EnvironmentLightingSubsystem>(
                    {.initialDecodeTimeoutMilliseconds = 10000}, error)) { return RHITestResult::fail(error); }
            const auto compile = [&] {
                if (!executor.compile(context.device, graph, kWidth, kHeight, error)) { throw std::runtime_error(error); }
            };
            nlohmann::json measurements;
            const auto capture = [&](const char* name) {
                checked(executor.execute({.graphicsQueue = &context.graphicsQueue}), "render Cornell fog");
                checked(executor.waitForSubmittedWork(), "wait for Cornell fog");
                const auto sample = readFogGI(context, executor, directory, name);
                measurements[name] = {{"scattering", sample.scatter}, {"linearHDR", sample.hdr}, {"extinction", sample.extinction}};
                return sample;
            };
            compile();
            const auto baseline = capture("unconnected");
            RenderGraph cornell;
            if (!loadRenderGraphFromFile(root / "Pipelines/Samples/lumen_cornell_box.metallic_graph.json", cornell, error)) {
                return RHITestResult::fail(error);
            }
            for (const auto* name : {"GlobalDistanceFieldPages", "SurfaceCache"}) {
                const auto* node = cornell.findNode(name);
                graph.addNode(node->type, name, node->properties);
            }
            graph.addNode("LumenTranslucencyVolumePass", "Volume", {{"gridPixelSize", 8}, {"jitter", false}});
            for (const std::string field : {"depth", "rasterInfo"}) { graph.addEdge("VBuffer." + field, "Volume." + field); }
            for (const std::string field : {"pageTable", "pageAtlas", "coarseMip", "domain"}) {
                graph.addEdge("GlobalDistanceFieldPages." + field, "SurfaceCache." + field);
                graph.addEdge("GlobalDistanceFieldPages." + field, "Volume." + field);
            }
            for (const std::string field : {"objectGrid", "objectGridScene"}) {
                graph.addEdge("GlobalDistanceFieldPages." + field, "Volume." + field);
            }
            graph.addEdge("SurfaceCache.cards", "Volume.cards");
            graph.addEdge("SurfaceCache.scene", "Volume.cardScene");
            graph.addEdge("SurfaceCache.depth", "Volume.cardDepth");
            const uint32_t directLightingEdge = graph.addEdge("SurfaceCache.lighting", "Volume.cardLighting")->id;
            for (const std::string field : {"giAmbient", "giDirectional", "giGrid"}) {
                graph.addEdge("Volume." + field, "Fog." + field);
            }
            compile();
            const auto red = capture("red-wall");
            const auto& materials = document.materials();
            const auto left = std::ranges::find(materials, "Left Wall Red", &scene::RenderMaterial::name);
            if (left == materials.end()) { return RHITestResult::fail("Missing left wall material"); }
            auto gray = *left;
            gray.baseColorFactor = float4(0.58f, 0.58f, 0.58f, 1);
            if (!document.setMaterialProperties(int32_t(left - materials.begin()), gray)) {
                return RHITestResult::fail("Could not edit Cornell wall material");
            }
            world.notifySceneChanged(RenderChangeBits::Material);
            const auto neutral = capture("neutral-wall");

            lights.lights.clear();
            if (!world.setLighting(lights)) { return RHITestResult::fail("Could not disable spotlight"); }
            const auto dark = capture("light-off");
            auto sky = document.environment();
            sky.enabled = true;
            sky.path = directory / "white-sky.hdr";
            sky.visible = false;
            const std::vector<float> white(16 * 8 * 3, 1.0f);
            if (!stbi_write_hdr(sky.path.string().c_str(), 16, 8, 3, white.data())) { return RHITestResult::fail("Write white sky"); }
            world.setEnvironment(sky);
            const auto* skyProvider = executor.subsystemHost()->get<EnvironmentLightingSubsystem>();
            for (int frame = 0; frame < 32 && !skyProvider->snapshot().mapAvailable; ++frame) {
                checked(executor.execute({.graphicsQueue = &context.graphicsQueue}), "publish Cornell sky");
                checked(executor.waitForSubmittedWork(), "wait for Cornell sky");
            }
            if (!skyProvider->snapshot().mapAvailable) { return RHITestResult::fail("Cornell HDRI did not become resident"); }
            const auto closed = capture("roof-closed");
            if (!document.setObjectVisible(document.objectForNode(1).entity(), false)) {
                return RHITestResult::fail("Could not remove Cornell roof");
            }
            world.notifySceneChanged(RenderChangeBits::Geometry);
            const auto open = capture("roof-open");
            // With white diffuse surfaces, the roof both occludes the sky and reflects light into the air.
            const auto right = std::ranges::find(materials, "Right Wall Green", &scene::RenderMaterial::name);
            if (right == materials.end()) { return RHITestResult::fail("Missing right wall material"); }
            auto plaster = *right;
            plaster.baseColorFactor = float4(0.73f, 0.73f, 0.73f, 1);
            if (!document.setMaterialProperties(int32_t(right - materials.begin()), plaster)) {
                return RHITestResult::fail("Could not neutralize right wall");
            }
            world.notifySceneChanged(RenderChangeBits::Material);
            const auto* radiosity = cornell.findNode("Radiosity");
            graph.addNode(radiosity->type, radiosity->name, radiosity->properties);
            for (const auto& edge : cornell.edges()) {
                if (edge.dstPass == radiosity->name) {
                    graph.addEdge(edge.srcPass + "." + edge.srcField, edge.dstPass + "." + edge.dstField);
                }
            }
            graph.removeEdge(directLightingEdge);
            graph.addEdge("Radiosity.lighting", "Volume.cardLighting");
            compile();
            const auto measureRadiosity = [&](bool roofVisible, const char* name) {
                if (roofVisible && !document.setObjectVisible(document.objectForNode(1).entity(), true)) {
                    throw std::runtime_error("Could not restore Cornell roof");
                }
                if (roofVisible) { world.notifySceneChanged(RenderChangeBits::Geometry); }
                for (int frame = 0; frame < 31; ++frame) {
                    checked(executor.execute({.graphicsQueue = &context.graphicsQueue}), "accumulate Cornell radiosity");
                    checked(executor.waitForSubmittedWork(), "wait for Cornell radiosity");
                }
                return capture(name);
            };
            const auto bouncedOpen = measureRadiosity(false, "diffuse-roof-open");
            const auto bouncedClosed = measureRadiosity(true, "diffuse-roof-closed");
            std::ofstream(directory / "linear-measurements.json") << measurements.dump(2);
            const auto redder = [](const auto& a, const auto& b) { return a[0] * b[1] > a[1] * b[0]; };
            if (baseline.scatter != std::array<double, 3>{} || dark.scatter != baseline.scatter ||
                baseline.extinction <= 0 || red.extinction != baseline.extinction || open.extinction != baseline.extinction) {
                return RHITestResult::fail("Disconnected or unlit GI changed fog scattering or extinction: " + measurements.dump());
            }
            if (red.scatter[0] <= 0 || neutral.scatter[1] <= red.scatter[1] ||
                !redder(red.scatter, neutral.scatter) || !redder(red.hdr, neutral.hdr)) {
                return RHITestResult::fail("The wall failed to tint in-air scattering and HDR: " + measurements.dump());
            }
            if (open.scatter[1] <= closed.scatter[1] * 1.05 || open.hdr[1] <= closed.hdr[1]) {
                return RHITestResult::fail("Removing the roof failed to admit skylight: " + measurements.dump());
            }
            // UE 5.7.4 SM6 SceneCapture, same 32x32 pixels, unit pre-exposure and RGBA16F Lumen.
            // For the sky probes UE used black walls with independently baked 210x220x10 cm roof geometry.
            constexpr double neutralUE = 0.36837816, roofClosedUE = 0.07413197, roofOpenUE = 0.11758447;
            if (std::abs(neutral.hdr[1] / neutralUE - 1.0) > 0.03 ||
                std::abs(closed.hdr[1] / roofClosedUE - 1.0) > 0.03 ||
                std::abs(open.hdr[1] / roofOpenUE - 1.0) > 0.03) {
                return RHITestResult::fail("Cornell fog lighting disagrees with Unreal reference HDR: " + measurements.dump());
            }
            if (bouncedOpen.scatter[1] <= open.scatter[1] || bouncedClosed.scatter[1] <= closed.scatter[1] ||
                bouncedOpen.hdr[1] <= open.hdr[1] || bouncedClosed.hdr[1] <= closed.hdr[1]) {
                return RHITestResult::fail("Skylit surface cards failed to illuminate the air: " + measurements.dump());
            }
            return RHITestResult::pass("Wall color and roof visibility change in-air scattering and composited HDR");
        } catch (const std::exception& e) { return RHITestResult::fail(e.what()); }
    }
};

METALLIC_REGISTER_RHI_TEST(VolumetricFogLumenTest);
} // namespace
} // namespace metallic::tests
