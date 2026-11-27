#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

constexpr uint32_t kWidth = 65, kHeight = 49;
constexpr std::array<std::string_view, 6> kOutputs{
    "Cloud.scattering", "Cloud.transmittance", "Cloud.cloudDepth", "Cloud.color", "Deferred.color", "VBuffer.depth"};

template<typename T>
T checked(Result<T> result, std::string_view operation)
{
    if (!result) { throw std::runtime_error(std::string(operation) + ": " + resultToString(result)); }
    if constexpr (!std::is_void_v<T>) { return std::move(*result); }
}

struct CloudPixel {
    std::array<float, 4> scattering, color, background;
    float transmittance, cloudDepthMeters, sceneDepth;
};

CloudPixel capture(RHITestContext& context, RenderGraphExecutor& executor, HistoryResourceManager& history, Buffer& readback,
    std::string_view name)
{
    checked(executor.execute({.graphicsQueue = &context.graphicsQueue, .historyResources = &history}), "render cloud view");
    checked(executor.waitForSubmittedWork(), std::string("wait for cloud view ") + std::string(name));
    bench::GPUCommands copy(context.graphicsQueue);
    checked(copy.initialize(context.device), "begin cloud readback");
    std::array<std::array<float, 4>, kOutputs.size()> pixel{};
    constexpr std::array<Format, kOutputs.size()> formats{Format::RGBA32Sfloat, Format::R32Sfloat,
        Format::R32Sfloat, Format::RGBA32Sfloat, Format::RGBA32Sfloat, Format::D32Sfloat};
    for (size_t i = 0; i < kOutputs.size(); ++i) {
        auto* output = executor.outputResource(kOutputs[i]);
        if (!output || output->desc.width != executor.width() || output->desc.height != executor.height() || output->desc.format != formats[i]) {
            throw std::runtime_error("Unexpected cloud view resource: " + std::string(kOutputs[i]));
        }
        checked(executor.transitionOutput(*copy.commands, kOutputs[i], ResourceState::TransferSource), "transition cloud readback");
        checked(copy.commands->copyTextureToBuffer({.texture = output->texture,
            .buffer = checked(readback.slice({i * sizeof(pixel[0]), sizeof(pixel[0])}), "cloud pixel slice"),
            .textureOffsetX = int32_t(output->desc.width / 2), .textureOffsetY = int32_t(output->desc.height / 2),
            .width = 1, .height = 1}), "copy center pixel");
    }
    const BufferBarrierDesc host{.buffer = &readback, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
    checked(copy.commands->synchronize({.buffers = {&host, 1}}), "make cloud readback visible");
    checked(copy.submitAndWait(), "submit cloud readback");
    readback.invalidate();
    const void* data = readback.map();
    if (!data) { throw std::runtime_error("Cannot map cloud readback"); }
    std::memcpy(pixel.data(), data, sizeof(pixel));
    readback.unmap();
    return {pixel[0], pixel[3], pixel[4], pixel[1][0], pixel[2][0], pixel[5][0]};
}

class VolumetricCloudViewTest final : public RHITest {
public:
    VolumetricCloudViewTest() { name = "volumetric_cloud_view_depth_and_composite"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        const auto& caps = context.device.capabilities();
        if (!caps.bindlessDescriptorHeap || !caps.meshShader || !caps.taskShader || !caps.geometryShader) {
            return RHITestResult::skip("Volumetric cloud foreground test requires --rhi-realtime");
        }
        try {
            scene::SceneDocument scene;
            if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Lumen/CornellBox.metallic_scene.json")) {
                return RHITestResult::fail(scene.lastLoadResult().error);
            }
            // The center ray encounters only the ceiling. An orthographic view keeps the
            // same mesh resolvable after moving it 2 km away without changing the ray.
            for (int node : {5, 6, 7}) { scene.setObjectVisible(scene.objectForNode(node).entity(), false); }
            RenderWorld world;
            world.setScene(&scene);
            world.setEnvironment({.enabled = true, .visible = true});
            environment::WorldEnvironment env;
            env.source = environment::EnvironmentSource::PhysicalAtmosphere;
            env.sun.enabled = true;
            env.sun.direction = float3(-0.6f, -0.8f, 0.0f);
            env.weather.cloudCoverage = 1.0f;
            env.weather.cloudDensity = 0.2f;
            env.weather.cloudBaseAltitudeKm = 1.0f;
            env.weather.cloudTopAltitudeKm = 3.0f;
            env.weather.noiseSeed = 27;
            env.weather.windSpeed = 10.0f;
            if (!world.setWorldEnvironment(env)) { return RHITestResult::fail("Invalid physical cloud fixture"); }
            RenderView view;
            view.setRenderResolution(kWidth, kHeight);
            ViewCamera camera{.eye = {0, 1, 0}, .center = {0, 2, 0}, .up = {0, 0, -1},
                .farPlane = 10000.0f, .orthoHeight = 0.8f, .orthographic = true};
            if (!view.setCamera(camera)) { return RHITestResult::fail("Invalid zenith camera"); }
            RenderGraph graph;
            graph.addNode("VisibilityBufferPass", "VBuffer", {{"initialLoad", true}, {"autoLod", false},
                {"enableClusterRtx", false}, {"meshletNormalConeCull", false}, {"visualization", "none"}});
            graph.addNode("VisibilityBufferDeferredPass", "Deferred", {{"outputLinear", true}, {"independentViewClouds", true}});
            graph.addNode("VolumetricCloudViewPass", "Cloud");
            for (const std::string pin : {"visibility", "depth", "domain", "rasterInfo"}) {
                graph.addEdge("VBuffer." + pin, "Deferred." + pin);
            }
            graph.addEdge("VBuffer.depth", "Cloud.depth");
            graph.addEdge("VBuffer.rasterInfo", "Cloud.rasterInfo");
            graph.addEdge("Deferred.color", "Cloud.sceneColor");
            graph.addEdge("Deferred.cameraAerial", "Cloud.cameraAerial");
            for (auto output : kOutputs) { graph.markOutput(std::string(output)); }
            HistoryResourceManager history;
            checked(history.initialize(context.device), "initialize cloud history");
            RenderGraphExecutor executor;
            executor.bindRenderWorld(&world);
            executor.bindRenderView(&view);
            std::string log;
            if (!executor.compile(context.device, graph, kWidth, kHeight, log)) { return RHITestResult::fail(log); }
            auto readback = checked(context.device.createBuffer({.size = kOutputs.size() * sizeof(float) * 4,
                .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}), "cloud readback");
            nlohmann::json cases;
            const auto measure = [&](const std::string& name, std::string_view expectedStage = {}) {
                const auto sample = capture(context, executor, history, *readback, name);
                auto frames = checked(executor.collectCompletedGpuExecutionStats(), "read cloud reconstruction stages");
                if (!expectedStage.empty()) {
                    bool found = false;
                    for (const auto& frame : frames) {
                        for (const auto& node : frame.nodes) {
                            if (node.name != "Cloud") { continue; }
                            for (const auto& section : node.sections) { found |= section.name == expectedStage; }
                        }
                    }
                    if (!found) { throw std::runtime_error(name + " missing " + std::string(expectedStage)); }
                }
                cases[name] = {{"scattering", sample.scattering}, {"transmittance", sample.transmittance},
                    {"cloudDepthMeters", sample.cloudDepthMeters}, {"sceneDepth", sample.sceneDepth},
                    {"background", sample.background}, {"color", sample.color}};
                for (unsigned i = 0; i < 3; ++i) {
                    const float expected = sample.scattering[i] + sample.transmittance * sample.background[i];
                    if (!std::isfinite(expected) || !std::isfinite(sample.color[i]) ||
                        std::abs(sample.color[i] - expected) > 1e-5f * std::max(std::abs(expected), 1.0f)) {
                        throw std::runtime_error("Cloud scene-linear composite mismatch: " + cases.dump());
                    }
                }
                if (std::abs(sample.scattering[3] - (1.0f - sample.transmittance)) > 1e-6f) {
                    throw std::runtime_error("Cloud coverage mismatch: " + cases.dump());
                }
                return sample;
            };
            const auto clear = measure("disabled");
            env.weather.cloudEnabled = true;
            world.setWorldEnvironment(env);
            const auto blocked = measure("ceiling-before-layer");
            float4x4 inLayer;
            inLayer.SetupByTranslation(float3(0.0f, 2000.0f, -0.05f));
            if (!scene.setNodeLocalMatrix(1, inLayer)) { return RHITestResult::fail("Could not move ceiling into clouds"); }
            world.notifySceneChanged(RenderChangeBits::Geometry);
            const auto clipped = measure("ceiling-in-layer");
            if (!scene.setObjectVisible(scene.objectForNode(1).entity(), false)) { return RHITestResult::fail("Could not open roof"); }
            world.notifySceneChanged(RenderChangeBits::Geometry);
            const auto open = measure("open-orthographic");
            const uint32_t cloudId = graph.findNode("Cloud")->id;
            const auto setSampleCount = [&](const char* key, uint32_t count) {
                if (!graph.setNodeRuntimeProperty(cloudId, key, count) || !executor.syncRuntimeProperties(graph)) {
                    throw std::runtime_error(std::string("Could not update live cloud setting ") + key);
                }
            };
            // Orthographic tracing uses a fixed noise phase, so all three samples use the same ray.
            setSampleCount("shadowViewRaySampleMaxCount", 2);
            const auto lowShadow = measure("orthographic-two-shadow-samples");
            setSampleCount("viewRaySampleMaxCount", 2);
            const auto lowView = measure("orthographic-two-view-samples");
            if (open.transmittance != lowShadow.transmittance ||
                std::abs(open.scattering[1] - lowShadow.scattering[1]) <= 1e-5f ||
                std::abs(lowView.transmittance - lowShadow.transmittance) <= 1e-5f) {
                throw std::runtime_error("Cloud view or self-shadow samples did not affect the expected integral: " + cases.dump());
            }
            if (!graph.setNodeRuntimeProperties(cloudId, RenderGraphProperties::object()) || !executor.syncRuntimeProperties(graph)) {
                return RHITestResult::fail("Could not restore default cloud sampling");
            }
            camera.orthographic = false;
            camera.eye[1] = 1800.0f;
            camera.center[1] = 1801.0f;
            view.setCamera(camera);
            constexpr std::string_view spatial = "Cloud spatial reconstruction";
            constexpr std::string_view temporal = "Cloud temporal reconstruction";
            const auto inside = measure("inside-perspective", spatial);
            for (unsigned i = 0; i < 4; ++i) {
                const auto sample = measure("static-phase-" + std::to_string(i), temporal);
                if (!(sample.transmittance < 1.0f && sample.cloudDepthMeters > 0.0f)) {
                    throw std::runtime_error("Interleaved cloud sample disappeared: " + cases.dump());
                }
            }
            camera.eye[0] += 0.25f;
            camera.center[0] += 0.25f;
            view.setCamera(camera);
            measure("translated", temporal);
            world.setEnvironmentElapsedSeconds(1.0 / 60.0);
            measure("wind-advanced", temporal);

            // Move a small foreground ceiling into the view. Then move only the
            // camera across its edge, revealing and covering the same cloud pixel.
            float4x4 nearby;
            nearby.SetupByTranslation(float3(0.0f, 1830.0f, 0.0f));
            if (!scene.setNodeLocalMatrix(1, nearby) || !scene.setObjectVisible(scene.objectForNode(1).entity(), true)) {
                return RHITestResult::fail("Could not place the perspective foreground occluder");
            }
            world.notifySceneChanged(RenderChangeBits::Geometry);
            const auto foreground = measure("perspective-foreground");
            camera.eye[0] += 3.0f;
            camera.center[0] += 3.0f;
            view.setCamera(camera);
            const auto revealed = measure("revealed-cloud", temporal);
            camera.eye[0] -= 3.0f;
            camera.center[0] -= 3.0f;
            view.setCamera(camera);
            const auto covered = measure("covered-cloud", temporal);
            ViewCamera below = camera;
            below.eye[1] = 400.0f;
            below.center[1] = 399.0f;
            view.setCamera(below);
            view.cameraCut();
            const auto cut = measure("cut-below-layer", spatial);
            view.setCamera(camera);
            view.cameraCut();
            measure("cut-back", spatial);
            env.weather.cloudEnabled = false;
            world.setWorldEnvironment(env);
            const auto clearSky = measure("disabled-open-sky");
            env.weather.cloudEnabled = true;
            world.setWorldEnvironment(env);
            measure("re-enabled", spatial);
            view.setRenderResolution(kWidth + 2, kHeight + 2);
            checked(executor.compile(context.device, graph, kWidth + 2, kHeight + 2, log), "resize cloud targets");
            measure("resized", spatial);
            measure("resized-history", temporal);
            setSampleCount("viewRaySampleMaxCount", 64);
            measure("view-samples-edited", spatial);
            measure("view-samples-steady", temporal);
            setSampleCount("shadowViewRaySampleMaxCount", 4);
            measure("shadow-samples-edited", spatial);
            measure("shadow-samples-steady", temporal);
            std::ofstream(context.outputDirectory / "VolumetricCloudView.json") << cases.dump(2);
            const auto identity = [](const CloudPixel& p) {
                return p.transmittance == 1 && p.cloudDepthMeters == 0 && p.scattering == std::array<float, 4>{} && p.color == p.background;
            };
            if (!identity(clear) || !identity(clearSky) || !identity(blocked) || !identity(cut) ||
                !identity(foreground) || !identity(covered) || !(foreground.sceneDepth > 0 && covered.sceneDepth > 0 &&
                    revealed.sceneDepth == 0 && revealed.transmittance < 1.0f) || !(blocked.sceneDepth > clipped.sceneDepth &&
                    clipped.sceneDepth > 0 && open.sceneDepth == 0 && inside.sceneDepth == 0)) {
                return RHITestResult::fail("Cloud bypass or foreground depth clipping failed: " + cases.dump());
            }
            if (!(clipped.cloudDepthMeters >= 998 && clipped.cloudDepthMeters < 2000 &&
                    open.cloudDepthMeters >= 998 && open.cloudDepthMeters < 3000 &&
                    clipped.transmittance < 1 && clipped.transmittance > open.transmittance + 0.01f &&
                    open.transmittance >= 0 && open.scattering[1] > 0 &&
                    inside.cloudDepthMeters > 0 && inside.cloudDepthMeters <= 1201 && inside.transmittance < 1 &&
                    inside.transmittance >= 0 && inside.scattering[1] > 0)) {
                return RHITestResult::fail("Cloud transmission, in-scattering or representative depth outside layer: " + cases.dump());
            }
            return RHITestResult::pass("Verified cloud L/T/depth through open, obstructed and in-layer perspective/orthographic views");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(VolumetricCloudViewTest);

} // namespace
} // namespace metallic::tests
