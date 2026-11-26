#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/SceneDocument.h"

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
        result.addTextureOutput("color", "Unlit white radiance").transferWrite()
            .transient(RenderGraphInitialization::FullOverwrite).format = Format::RGBA32Sfloat;
        return result;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        return context.commandBuffer().clearColorTexture(*context.outputTexture("color").texture(),
            TextureLayout::TransferDestination, {1, 1, 1, 1});
    }
};

RenderGraph makeFogGraph(const RenderView& view)
{
    RenderGraph graph;
    graph.setViewProperties({{"camera", view.cameraProperties()}, {"renderResolution", view.renderResolutionProperties()},
        {"temporalJitter", false}});
    graph.addNode("VisibilityBufferPass", "VBuffer", {{"initialLoad", true}, {"enableClusterRtx", false},
        {"autoLod", false}, {"meshletNormalConeCull", false}, {"camera", view.cameraProperties()}});
    graph.addNode("WhiteFogProbeWall", "Wall");
    graph.addNode("VolumetricFogPass", "Fog", {{"gridPixelSize", 16}, {"gridSizeZ", 64},
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
            static const bool registered = registerRenderGraphPassType("WhiteFogProbeWall", "Unlit white fog probe wall",
                [] { return std::make_unique<WhiteFogProbeWall>(); });
            if (!registered) { return RHITestResult::fail("Cannot register white wall fixture"); }
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
} // namespace
} // namespace metallic::tests
