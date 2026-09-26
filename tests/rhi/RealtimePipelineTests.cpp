#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/SceneDocument.h"
#include "stb/stb_image_write.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <stdexcept>

namespace metallic::tests {
namespace {

RhiTestResult realtimeFailure(std::string message)
{
    spdlog::error("Realtime regression: {}", message);
    return RhiTestResult::fail(std::move(message));
}

class EnvironmentPrefilterProbePass final : public render::ComputePass {
public:
    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array ids{render::EnvironmentLightingSubsystem::kSubsystemId};
        return ids;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data").buffer(8 * 16, 16).storageReadWrite();
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "RealtimeGuideProbe",
            .entryPointName = "environmentPrefilterProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {{.binding = 2}, {.binding = 3}};
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * 4,
            .bindings = bindings, .bindingCount = 2, .requiresRayQuery = false}, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 2, .buffer = context.outputBuffer("data").buffer()},
            {.binding = 3, .buffer = context.subsystem<render::EnvironmentLightingSubsystem>()->snapshot().prefilteredSpecularBuffer}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings, .bindingCount = 2});
    }
private:
    render::ComputeProgram program_;
};

class EnvironmentPrefilterTest final : public RhiTest {
public:
    EnvironmentPrefilterTest() { type = RhiTestType::Rendering; name = "realtime_environment_prefilter_energy"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        const auto path = std::filesystem::absolute(context.outputDirectory / "RealtimeConstant.hdr");
        std::filesystem::create_directories(context.outputDirectory);
        std::vector<float> pixels(32 * 16 * 3);
        for (size_t i = 0; i < pixels.size(); i += 3) { pixels[i] = 2.0f; pixels[i + 1] = 1.0f; pixels[i + 2] = 0.5f; }
        if (!stbi_write_hdr(path.string().c_str(), 32, 16, 3, pixels.data())) { return realtimeFailure("HDR fixture write failed"); }
        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({.applicationName = "Environment prefilter energy",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("Requires bindless descriptors"); }
        if (!result) { return realtimeFailure("Prefilter device creation failed"); }
        render::RenderWorld world;
        world.setEnvironment({.enabled = true, .path = path});
        render::registerRenderGraphPassType("EnvironmentPrefilterProbePass", "IBL energy probe",
            [] { return std::make_unique<EnvironmentPrefilterProbePass>(); });
        render::RenderGraph graph;
        graph.addNode("EnvironmentPrefilterProbePass", "Probe");
        graph.markOutput("Probe.data");
        render::RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        std::string log;
        if (!executor.compile(*device, graph, 1, 1, log)) { return realtimeFailure(log); }
        bool ready = false;
        for (uint32_t frame = 0; frame < 60 && !ready; ++frame) {
            if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
                !executor.waitForSubmittedWork()) { return realtimeFailure("Prefilter execution failed"); }
            ready = executor.subsystemHost()->get<render::EnvironmentLightingSubsystem>()->snapshot().mapAvailable;
            if (!ready) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
        }
        if (!ready) { return realtimeFailure("HDR publication timed out"); }
        auto* buffer = executor.outputResource("Probe.data")->buffer;
        buffer->invalidate();
        const auto* values = static_cast<const std::array<float, 4>*>(buffer->map());
        if (values == nullptr) { return realtimeFailure("Prefilter readback failed"); }
        bool valid = true;
        const std::array expected{2.0f, 1.0f, 0.5f};
        for (size_t i = 0; i < 8; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                valid &= std::isfinite(values[i][c]) && std::abs(values[i][c] - expected[c]) < 0.002f;
            }
        }
        buffer->unmap();
        return valid ? RhiTestResult::pass("GGX preserves constant radiance at every roughness, poles and wrap seam")
            : realtimeFailure("GGX filtering changed constant environment energy");
    }
};

class RealtimeReadbackPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("color").transferRead().format = render::Format::Rgba8Unorm;
        reflection.addTextureInput("motion").sampledRead().format = render::Format::Rg16Sfloat;
        reflection.addTextureInput("depth").sampledRead().format = render::Format::R32Sfloat;
        reflection.addBufferOutput("pixels").buffer(uint64_t(context.width) * context.height * 4).transferWrite();
        reflection.addBufferOutput("guides").buffer(uint64_t(context.width) * context.height * 16, 16).storageReadWrite();
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "RealtimeGuideProbe",
            .entryPointName = "realtimeGuideProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::SampledImage},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::SampledImage},
            {.binding = 2, .kind = render::ComputeResourceBindingKind::StorageBuffer}};
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * 4, .bindings = bindings, .bindingCount = 3,
            .requiresRayQuery = false}, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        context.commandBuffer().copyTextureToBuffer({.texture = context.inputTexture("color").texture(),
            .buffer = context.outputBuffer("pixels").buffer(), .bufferRowPitch = context.width() * 4,
            .bufferSlicePitch = context.width() * context.height() * 4,
            .width = context.width(), .height = context.height()});
        auto* motion = context.inputTexture("motion").view();
        auto* depth = context.inputTexture("depth").view();
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureViews = &motion, .textureViewCount = 1},
            {.binding = 1, .textureViews = &depth, .textureViewCount = 1},
            {.binding = 2, .buffer = context.outputBuffer("guides").buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings, .bindingCount = 3,
            .groupCountX = (context.width() + 7) / 8, .groupCountY = (context.height() + 7) / 8});
    }
private:
    render::ComputeProgram program_;
};

class RealtimePipelineTest : public RhiTest {
public:
    explicit RealtimePipelineTest(bool sponza = false) : sponza_(sponza)
    {
        type = RhiTestType::Rendering;
        name = sponza ? "gpu_driven_sponza_realtime_pipeline" : "realtime_clustered_dlss_pipeline";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("realtime-lighting", sample, log) ||
            !sample.desc.requiresStreamline) {
            return realtimeFailure("Realtime sample metadata: " + log);
        }
        if (sponza_ && !render::setRenderSampleScenePath(sample, "Asset/Sponza/glTF/Sponza.gltf", log)) { return realtimeFailure(log); }
        for (const auto& node : sample.graph.nodes()) {
            if (node.type.find("PathTrace") != std::string::npos || node.type.find("Nrd") != std::string::npos ||
                node.type == "StreamlineDlssRrPass" || node.type == "SceneRealtimeLightingPass") {
                return realtimeFailure("Realtime pipeline contains a reference/path-tracing pass");
            }
        }
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return realtimeFailure(scene.lastLoadResult().error);
        }
        // Streamline owns process-wide Vulkan state: use one device for the test.
        auto* device = &context.device;
        if (!device->capabilities().streamlineDlssSr || !device->capabilities().meshShader) {
            return RhiTestResult::skip("Requires --rhi-realtime and supported mesh shaders/DLSS-SR");
        }
        const uint32_t initialValidationCount = context.validationMessageCount != nullptr
            ? context.validationMessageCount->load() : 0;
        render::Result<> result;
        render::RenderWorld world;
        world.setScene(&scene);
        world.setEnvironment({.enabled = true, .path = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.environment->path});
        auto lighting = scene.lighting();
        lighting.autoExposure.enabled = true;
        auto& light = lighting.lights.emplace_back();
        light.properties.type = "point";
        light.properties.intensity = 50.0f;
        light.properties.range = 8.0f;
        light.position = float3(1.0f, 1.0f, 2.0f);
        world.setLighting(lighting);
        render::registerRenderGraphPassType("RealtimeReadbackPass", "Realtime GPU regression readback",
            [] { return std::make_unique<RealtimeReadbackPass>(); });
        sample.graph.addNode("RealtimeReadbackPass", "Readback");
        sample.graph.addEdge("FinalBlit.color", "Readback.color");
        sample.graph.addEdge("DlssSr.motionVectors", "Readback.motion");
        sample.graph.addEdge("DlssSr.depth", "Readback.depth");
        sample.graph.markOutput("Readback.pixels");
        sample.graph.markOutput("Readback.guides");
        render::RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        for (uint32_t variant = 0; variant < 3; ++variant) {
            const uint32_t width = variant == 2 ? 321 : 512, height = variant == 2 ? 217 : 384;
            // Keep the Sponza regression on GPUDrivenSample's default NR-off
            // configuration. The original realtime test covers optional NR.
            if (variant == 1 && !sponza_) {
                sample.graph.setNodeRuntimeProperty(sample.graph.findNode("DlssNr")->id, "enabled", true);
            }
            result = executor.compile(*device, sample.graph, width, height, log);
            if (!result) { return realtimeFailure(log); }
            if (variant == 2) {
                auto camera = executor.renderView()->camera();
                camera.reversedZ = false;
                executor.renderView()->setCamera(camera);
            }
            for (uint32_t frame = 0; frame < 12; ++frame) {
                const bool captureFrame = sponza_ && context.nsightCapture != nullptr && variant == 0 && frame == 4;
                if (captureFrame) {
                    if (!context.nsightCapture->requestCapture({.explicitFrameBoundaries = true}, log) ||
                        !context.nsightCapture->frameBoundary(context.graphicsQueue, nullptr, log)) {
                        return realtimeFailure("Start Nsight frame capture: " + log);
                    }
                }
                if (frame == 8) {
                    // One view update drives raster, lighting, guides and SR.
                    auto camera = executor.renderView()->camera();
                    camera.eye[0] = 0.12f * float(variant + 1);
                    executor.renderView()->setCamera(camera);
                }
                if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
                    !executor.waitForSubmittedWork()) { return realtimeFailure("Realtime frame execution failed"); }
                if (captureFrame) {
                    if (!context.nsightCapture->frameBoundary(context.graphicsQueue,
                            executor.outputResource("FinalBlit.color")->texture, log)) {
                        return realtimeFailure("End Nsight frame capture: " + log);
                    }
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                    auto capture = context.nsightCapture->poll();
                    while (capture.state == render::profiling::NsightGraphicsCaptureState::CapturePending &&
                        std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        capture = context.nsightCapture->poll();
                    }
                    if (capture.state != render::profiling::NsightGraphicsCaptureState::CaptureCompleted ||
                        !std::filesystem::is_regular_file(capture.capturePath)) {
                        return realtimeFailure("Nsight frame export failed or timed out: " + capture.message);
                    }
                    spdlog::info("Nsight replay regression capture: {}", capture.capturePath.string());
                }
                auto* buffer = executor.outputResource("Readback.guides")->buffer;
                buffer->invalidate();
                const auto* data = static_cast<const std::array<float, 4>*>(buffer->map());
                if (data == nullptr) { return realtimeFailure("Guide readback failed"); }
                float maxMotion = 0.0f;
                size_t surfacePixels = 0;
                bool valid = true;
                for (uint32_t i = 0; i < width * height; ++i) {
                    valid &= std::isfinite(data[i][0]) && std::isfinite(data[i][1]) &&
                        std::isfinite(data[i][2]) && data[i][2] >= 0.0f && data[i][2] <= 1.0f;
                    maxMotion = std::max(maxMotion, std::max(std::abs(data[i][0]), std::abs(data[i][1])));
                    surfacePixels += data[i][2] < 0.9999f;
                }
                buffer->unmap();
                if (!valid || surfacePixels < 100 || (frame >= 3 && frame != 8 && maxMotion > 0.0001f)) {
                    return realtimeFailure("Invalid depth/static jitter motion: " + std::to_string(maxMotion) + " variant=" + std::to_string(variant) + " frame=" + std::to_string(frame));
                }
                if (frame == 8 && maxMotion < 0.0001f) { return realtimeFailure("Camera movement produced no motion vectors"); }
            }
            auto* buffer = executor.outputResource("Readback.pixels")->buffer;
            buffer->invalidate();
            const auto* pixels = static_cast<const uint8_t*>(buffer->map());
            if (pixels == nullptr) { return realtimeFailure("Color readback failed"); }
            bool saved = saveRgba8Png(context.outputDirectory / ("RealtimePipeline" + std::to_string(variant) + ".png"),
                pixels, width, height, log);
            buffer->unmap();
            if (!saved) { return realtimeFailure(log); }
        }
        executor = render::RenderGraphExecutor{};
        if (context.validationMessageCount != nullptr && context.validationMessageCount->load() != initialValidationCount) {
            return realtimeFailure("Vulkan validation messages: " +
                std::to_string(context.validationMessageCount->load() - initialValidationCount));
        }
        return RhiTestResult::pass("Raster/LightGrid/SH/HDRI/auto exposure/SR, optional NR, camera jitter and odd-size resize");
    }
private:
    bool sponza_ = false;
};

class GpuDrivenSponzaRealtimePipelineTest final : public RealtimePipelineTest {
public:
    GpuDrivenSponzaRealtimePipelineTest() : RealtimePipelineTest(true) {}
};

METALLIC_REGISTER_RHI_TEST(RealtimePipelineTest);
METALLIC_REGISTER_RHI_TEST(GpuDrivenSponzaRealtimePipelineTest);
METALLIC_REGISTER_RHI_TEST(EnvironmentPrefilterTest);

class RealtimeShadowTest final : public RhiTest {
public:
    RealtimeShadowTest() { type = RhiTestType::Rendering; name = "realtime_ray_traced_sigma_shadows"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("realtime-lighting", sample, log) ||
            !render::setRenderSampleScenePath(sample, "Asset/LookDev/OpenPbrDefault/OpenPbrDefault.gltf", log)) {
            return realtimeFailure(log);
        }
        sample.graph.removeNode(sample.graph.findNode("DlssSr")->id);
        sample.graph.removeNode(sample.graph.findNode("DlssNr")->id);
        sample.graph.addEdge("Deferred.color", "AutoExposure.source");
        sample.graph.addEdge("AutoExposure.color", "FinalBlit.source");
        sample.graph.setViewProperties({{"camera", {{"eye", {0.0, 1.4, 3.65}}, {"center", {0.0, 0.8, 0.0}},
            {"fovDegrees", 50.0}, {"znear", 0.05}, {"zfar", 100.0}, {"reversedZ", true}}}, {"temporalJitter", true}});
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return realtimeFailure(scene.lastLoadResult().error);
        }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        auto result = preview.initialize(context.enableValidation, true);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders"); }
        if (!result) { return realtimeFailure("Initialize realtime shadow preview"); }
        preview.setEnvironment({.enabled = false});
        scene::LightingSettings lighting;
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 4.0f;
        scene::PunctualLight sun;
        sun.properties.type = "directional";
        sun.properties.intensity = 30;
        sun.direction = float3(0.6f, -1.0f, -0.3f);
        // A disabled prefix and an active local light must not compact the sun's source slot.
        scene::PunctualLight inactive;
        inactive.enabled = false;
        lighting.lights.push_back(inactive);
        auto point = inactive;
        point.enabled = true;
        point.properties.intensity = 0.001;
        lighting.lights.push_back(point);
        lighting.lights.push_back(sun);
        preview.setLighting(lighting);
        const auto* shadowNode = sample.graph.findNode("Shadows");
        if (shadowNode == nullptr || shadowNode->type != "RayTracedShadowPass") {
            return realtimeFailure("Realtime pipeline is missing the explicit shadow stage");
        }
        const auto shadows = shadowNode->id;
        const auto deferred = sample.graph.findNode("Deferred")->id;
        sample.graph.setNodeRuntimeProperty(shadows, "shadowAngularRadius", 1.5f);
        sample.graph.setNodeRuntimeProperty(shadows, "shadowRayLength", 100000.0f);
        std::vector<uint32_t> unshadowed;
        for (uint32_t mode = 0; mode < 4; ++mode) {
            sample.graph.setNodeRuntimeProperty(shadows, "rayTracedShadows", true);
            sample.graph.setNodeRuntimeProperty(deferred, "debugDisableShadows", mode == 0);
            sample.graph.setNodeRuntimeProperty(shadows, "sigmaDenoise", mode >= 2);
            sample.graph.setNodeRuntimeProperty(shadows, "shadowDebug", mode == 3);
            for (uint32_t frame = 0; frame < 12; ++frame) {
                if (!preview.render(sample.graph, 513, 385)) { return realtimeFailure(preview.lastLog()); }
            }
            const auto& pixels = preview.pixels();
            if (mode == 0) { unshadowed = pixels; }
            if (mode == 2) {
                size_t darker = 0;
                for (size_t i = 0; i < pixels.size(); ++i) {
                    darker += int(unshadowed[i] & 255u) - int(pixels[i] & 255u) > 8;
                }
                if (darker < 100) { return realtimeFailure("SIGMA shadow did not attenuate direct lighting"); }
            }
            const char* files[] = {"ShadowDisabled.png", "ShadowRaw.png", "ShadowSigma.png", "ShadowVisibility.png"};
            if (!saveRgba8Png(context.outputDirectory / files[mode], reinterpret_cast<const uint8_t*>(pixels.data()),
                preview.width(), preview.height(), log)) { return realtimeFailure(log); }
        }
        sample.graph.setNodeRuntimeProperty(shadows, "shadowDebug", false);
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", false);
        for (uint32_t frame = 0; frame < 8; ++frame) {
            if (!preview.render(sample.graph, 513, 385)) { return realtimeFailure(preview.lastLog()); }
        }
        size_t darker = 0;
        for (size_t i = 0; i < preview.pixels().size(); ++i) {
            darker += int(unshadowed[i] & 255u) - int(preview.pixels()[i] & 255u) > 8;
        }
        if (darker < 100) { return realtimeFailure("Unbinned deferred resolve did not consume the graph shadow"); }
        if (!preview.render(sample.graph, 321, 217)) { return realtimeFailure(preview.lastLog()); }
        // Deterministic comparisons exercise live properties without rebuilding the graph.
        auto view = sample.graph.viewProperties();
        view["temporalJitter"] = false;
        sample.graph.setViewProperties(view);
        sample.graph.setNodeRuntimeProperty(shadows, "sigmaDenoise", false);
        sample.graph.setNodeRuntimeProperty(shadows, "shadowAngularRadius", 0.0f);
        sample.graph.setNodeRuntimeProperty(shadows, "shadowDebug", true);
        sample.graph.setNodeRuntimeProperty(shadows, "shadowLightIndex", -1);
        const auto draw = [&]() {
            // Let SIGMA converge; zero-radius raw shadows remain deterministic.
            for (uint32_t frame = 0; frame < 8; ++frame) {
                if (!preview.render(sample.graph, 321, 217)) { throw std::runtime_error(preview.lastLog()); }
            }
            return preview.pixels();
        };
        const auto automatic = draw();
        sample.graph.setNodeRuntimeProperty(shadows, "shadowLightIndex", int(scene.lights().size()) + 2);
        if (draw() != automatic) { return realtimeFailure("Explicit sun slot and Auto produce different shadow signals"); }
        sample.graph.setNodeRuntimeProperty(shadows, "shadowLightIndex", 1412);
        if (draw() != automatic) { return realtimeFailure("Invalid slot did not fall back to Auto"); }
        // Old screen-space settings must never affect TLAS visibility.
        sample.graph.setNodeRuntimeProperty(shadows, "shadowDistance", 0.01f);
        sample.graph.setNodeRuntimeProperty(shadows, "shadowThickness", 10.0f);
        sample.graph.setNodeRuntimeProperty(shadows, "shadowSteps", 8);
        sample.graph.setNodeRuntimeProperty(shadows, "preserveGeometryShadows", false);
        if (draw() != automatic) { return realtimeFailure("Legacy screen-space controls altered full ray tracing"); }
        sample.graph.setNodeRuntimeProperty(shadows, "shadowRayLength", 0.01f);
        const auto shortTrace = draw();
        size_t changed = 0;
        for (size_t i = 0; i < automatic.size(); ++i) { changed += automatic[i] != shortTrace[i]; }
        if (changed < 100) { return realtimeFailure("Live ray length did not update TLAS visibility"); }
        sample.graph.setNodeRuntimeProperty(shadows, "shadowDebug", false);
        const auto shortLighting = draw();
        sample.graph.setNodeRuntimeProperty(shadows, "shadowRayLength", 100000.0f);
        const auto longLighting = draw();
        changed = 0;
        for (size_t i = 0; i < shortLighting.size(); ++i) {
            changed += int(shortLighting[i] & 255u) - int(longLighting[i] & 255u) > 8;
        }
        if (changed < 30) { return realtimeFailure("Trace settings did not affect the selected stable LightGrid sun slot"); }
        const auto hardLighting = longLighting;
        // Live source-radius edits must widen the full geometry penumbra and
        // lighten pixels inside the old hard shadow in final deferred lighting.
        sample.graph.setNodeRuntimeProperty(shadows, "shadowDebug", true);
        const auto hardVisibility = draw();
        sample.graph.setNodeRuntimeProperty(shadows, "sigmaDenoise", true);
        std::array<size_t, 3> softened{};
        const float angles[] = {0.0f, 1.0f, 8.0f};
        const char* captures[] = {"ShadowAngle0.png", "ShadowAngle1.png", "ShadowAngle8.png"};
        uint32_t fullyLit = 0;
        for (auto pixel : hardVisibility) { fullyLit = std::max(fullyLit, pixel & 255u); }
        for (size_t angle = 0; angle < 3; ++angle) {
            sample.graph.setNodeRuntimeProperty(shadows, "shadowAngularRadius", angles[angle]);
            const auto visibility = draw();
            for (size_t i = 0; i < visibility.size(); ++i) {
                const auto value = visibility[i] & 255u;
                softened[angle] += (hardVisibility[i] & 255u) < 3u && value > 12u && value + 8u < fullyLit;
            }
            if (!saveRgba8Png(context.outputDirectory / captures[angle],
                reinterpret_cast<const uint8_t*>(visibility.data()), preview.width(), preview.height(), log)) {
                return realtimeFailure(log);
            }
        }
        if (softened[2] < softened[0] + 30 || softened[2] < softened[1] + 20) {
            return realtimeFailure("Angular radius did not widen the geometry penumbra: " +
                std::to_string(softened[0]) + ", " + std::to_string(softened[1]) + ", " + std::to_string(softened[2]));
        }
        sample.graph.setNodeRuntimeProperty(shadows, "shadowDebug", false);
        const auto softLighting = draw();
        size_t lightened = 0;
        for (size_t i = 0; i < hardLighting.size(); ++i) {
            lightened += (hardVisibility[i] & 255u) < 3u &&
                int(softLighting[i] & 255u) - int(hardLighting[i] & 255u) > 8;
        }
        if (lightened < 30) { return realtimeFailure("Deferred hard-shadow composition erased the SIGMA penumbra"); }
        if (!saveRgba8Png(context.outputDirectory / "ShadowAngle8Lighting.png",
            reinterpret_cast<const uint8_t*>(softLighting.data()), preview.width(), preview.height(), log)) {
            return realtimeFailure(log);
        }
        return RhiTestResult::pass("Full TLAS + SIGMA, live light selection, legacy settings, resize and both resolve paths; "
            "penumbra pixels at 0/1/8 degrees: " + std::to_string(softened[0]) + "/" +
            std::to_string(softened[1]) + "/" + std::to_string(softened[2]) +
            "; final lighting softened pixels: " + std::to_string(lightened));
    }
};

METALLIC_REGISTER_RHI_TEST(RealtimeShadowTest);

class StreamedRealtimeTest : public RhiTest {
public:
    explicit StreamedRealtimeTest(bool miniZorah = false) : miniZorah_(miniZorah)
    {
        type = RhiTestType::Rendering;
        name = miniZorah ? "minizorah_realtime_pipeline" : "streamed_realtime_pipeline";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        const auto require = [](bool condition, const std::string& message) {
            if (!condition) { throw std::runtime_error(message); }
        };
        if (miniZorah_ && std::getenv("METALLIC_TEST_MINIZORAH") == nullptr) {
            return RhiTestResult::skip("Set METALLIC_TEST_MINIZORAH=1 for the default full scene");
        }
        if (!context.device.capabilities().meshShader || !context.device.capabilities().clusterAccelerationStructure ||
            (miniZorah_ && !context.device.capabilities().streamlineDlssSr)) {
            return RhiTestResult::skip("Requires --rhi-realtime with mesh shaders, CLAS and DLSS-SR");
        }
        try {
            std::string log;
            RenderSampleLoadResult sample;
            require(loadBuiltInRenderSample(kDefaultGPUDrivenSampleId, sample, log), log);
            auto& graph = sample.graph;
            if (!miniZorah_) {
                const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
                const auto cache = std::filesystem::absolute(context.outputDirectory / "RealtimeBunny.meshstream.bin");
                require(scene::buildMeshletStreamAssetOffline({.sourcePath = source, .outputPath = cache,
                    .meshletOptions = {.maxWorkers = 1}}, log), log);
                require(setRenderSampleScenePath(sample, source.generic_string(), log), log);
                auto& props = graph.findNode("VBuffer")->properties;
                props["streamAssetPath"] = cache.generic_string();
                props["maxResidentPages"] = 64; props["maxResidentBytes"] = 16777216;
                props["maxLockedFallbackPages"] = 64; props["maxActiveGroups"] = 1024;
                props["maxClasBytes"] = 16777216; props["maxClasBuildClusters"] = 2048;
                props["maxGpuPageRequests"] = 1024; props["maxGpuPageUnloadRequests"] = 1024;
                props["maxTraversalWorkers"] = 32; props["maxTraversalWorkItems"] = 2048;
                props["autoLod"] = false; props["lodLevel"] = 0;
                graph.setViewProperties({{"camera", {{"eye", {0, .2, 2.5}}, {"center", {0, 0, 0}},
                    {"znear", .01}, {"zfar", 100}, {"fovDegrees", 50}, {"reversedZ", true}}}, {"temporalJitter", false}});
                scene::Scene fixture;
                require(fixture.loadStreamMetadata(source), fixture.lastLoadResult().error);
                const auto center = fixture.bounds().center();
                const auto radius = fixture.bounds().radius();
                auto view = graph.viewProperties();
                view["camera"]["eye"] = {center.x, center.y + radius * .3f, center.z + radius * 3};
                view["camera"]["center"] = {center.x, center.y, center.z};
                graph.setViewProperties(view);
                graph.removeNode(graph.findNode("DlssSr")->id);
                graph.removeNode(graph.findNode("DlssNr")->id);
                graph.addEdge("Deferred.color", "AutoExposure.source");
                graph.addEdge("AutoExposure.color", "FinalBlit.source");
                graph.findNode("Shadows")->properties["sigmaDenoise"] = false;
                graph.findNode("Shadows")->properties["shadowAngularRadius"] = 0.0;
            }
            registerRenderGraphPassType("RealtimeReadbackPass", "Realtime GPU regression readback",
                [] { return std::make_unique<RealtimeReadbackPass>(); });
            graph.addNode("RealtimeReadbackPass", "Readback");
            graph.addEdge("FinalBlit.color", "Readback.color");
            graph.addEdge(miniZorah_ ? "DlssSr.motionVectors" : "Deferred.motionVectors", "Readback.motion");
            graph.addEdge(miniZorah_ ? "DlssSr.depth" : "Deferred.deviceDepth", "Readback.depth");
            graph.markOutput("Readback.pixels"); graph.markOutput("Readback.guides");
            RenderWorld world;
            // The compact-cook fixture must produce lit geometry without an HDRI
            // that could hide a failed visibility-to-surface decode.
            world.setEnvironment({.enabled = miniZorah_, .path = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.environment->path});
            scene::LightingSettings lighting;
            lighting.autoExposure.enabled = miniZorah_;
            lighting.exposureEV100 = 2;
            scene::PunctualLight sun;
            sun.properties.type = "directional"; sun.properties.intensity = 10;
            sun.direction = float3(.6f, -1, -.3f);
            lighting.lights.push_back(sun);
            world.setLighting(lighting);
            RenderGraphExecutor executor;
            executor.bindRenderWorld(&world);
            uint32_t width = 512, height = 320;
            require(bool(executor.compile(context.device, graph, width, height, log)), log);
            auto* streamer = executor.subsystemHost()->get<StreamerSubsystem>();
            require(streamer != nullptr && streamer->streamCount() == 1, "Streamer must own exactly one raster session");
            require(!streamer->sceneReadiness().ready && streamer->sceneReadiness().requiredPages != 0,
                "Scene became presentable before fallback uploads");
            const scene::Scene* metadata = nullptr;
            require(bool(streamer->manager().resolveScene(graph.findNode("VBuffer")->properties, nullptr, metadata, log)), log);
            require(metadata && metadata->hasStreamGeometry(), "Default producer must resolve streaming metadata");
            for (const auto& primitive : metadata->renderPrimitives()) {
                require(primitive.positions.empty() && primitive.indices.empty(), "Full source geometry became resident");
            }
            std::shared_ptr<SceneResourceSnapshot> materials;
            require(bool(streamer->manager().acquire(context.device, context.graphicsQueue,
                graph.findNode("Deferred")->properties, metadata, SceneResourceFeatureBits::Materials, materials, log)), log);
            require(materials->pathTraceResources->valid() && materials->pathTraceResources->materialBuffer() != nullptr &&
                materials->pathTraceResources->shadingVertexBuffer() == nullptr &&
                materials->pathTraceResources->indexBuffer() == nullptr &&
                !materials->pathTraceResources->accelerationStructure().valid(), "Material acquisition imported resident geometry/RTAS");
            std::shared_ptr<SceneResourceSnapshot> rejected;
            require(!streamer->manager().acquire(context.device, context.graphicsQueue,
                graph.findNode("Deferred")->properties, metadata, SceneResourceFeatureBits::Geometry, rejected, log),
                "Metadata must not silently fall back to resident import");
            const auto draw = [&]() {
                require(bool(executor.execute({.graphicsQueue = &context.graphicsQueue,
                    .computeQueue = context.device.getQueue(QueueType::Compute)})), "Streamed realtime execution failed");
                if (miniZorah_) { require(bool(vulkan::notifyStreamlineOffscreenFrame()), "Offscreen Streamline frame failed"); }
                require(bool(executor.waitForSubmittedWork()), "Streamed realtime submission failed");
            };
            const auto rebuildAndDraw = [&](uint32_t newWidth, uint32_t newHeight) {
                require(executor.executionStats().streaming.size() == 1, "Missing stream continuity telemetry");
                const auto before = executor.executionStats().streaming.front();
                const auto begin = std::chrono::steady_clock::now();
                width = newWidth; height = newHeight;
                require(bool(executor.compile(context.device, graph, width, height, log)), log);
                const auto compileMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                draw();
                require(executor.executionStats().streaming.size() == 1, "Stream missing after graph rebuild");
                const auto& after = executor.executionStats().streaming.front();
                require(after.generation == before.generation && after.frameIndex == before.frameIndex + 1 &&
                    after.totalUploadBytes >= before.totalUploadBytes,
                    "Viewport graph rebuild restarted streaming and discarded loaded pages");
                require(streamer->streamCount() == 1, "Viewport graph rebuild allocated another streaming session");
                spdlog::info("Stream preserved across {}x{} rebuild: generation={} frame={} compile={:.3f} ms",
                    width, height, after.generation, after.frameIndex, compileMs);
            };
            uint32_t sceneReadyFrame = 0;
            for (uint32_t frame = 0; frame < (miniZorah_ ? 180u : 48u); ++frame) {
                // Dock layout settles after the first visible frames, while pages
                // are still loading. Exercise that resize, then a resource-only
                // rebuild (DLSS render dimensions differ from graph dimensions).
                if (frame == 2) { rebuildAndDraw(481, 301); }
                else if (frame == 4) { rebuildAndDraw(512, 320); }
                else if (frame == 6) { rebuildAndDraw(width, height); }
                else { draw(); }
                const auto readiness = streamer->sceneReadiness();
                if (sceneReadyFrame != 0) { require(readiness.ready, "Scene readiness regressed during streaming/resize"); }
                if (sceneReadyFrame == 0 && readiness.ready) {
                    sceneReadyFrame = frame + 1;
                    spdlog::info("Complete fallback coverage at frame {} ({} / {} geometry + CLAS pages)",
                        sceneReadyFrame, readiness.completedPages, readiness.requiredPages);
                }
            }
            const auto pixels = [&]() {
                auto* buffer = executor.outputResource("Readback.pixels")->buffer;
                buffer->invalidate(); const auto* bytes = static_cast<const uint32_t*>(buffer->map());
                require(bytes != nullptr, "Color readback failed");
                std::vector<uint32_t> result(bytes, bytes + size_t(width) * height); buffer->unmap(); return result;
            };
            const auto shaded = pixels();
            require(std::count_if(shaded.begin(), shaded.end(), [&](uint32_t p) { return p != shaded[0]; }) > 100,
                "Streamed deferred output has no useful image");
            const auto* infoData = executor.outputResource("VBuffer.rasterInfo")->buffer->map();
            VisibilityBufferFrameInfo info; std::memcpy(&info, infoData, sizeof(info));
            executor.outputResource("VBuffer.rasterInfo")->buffer->unmap();
            auto* gpuScene = executor.subsystemHost()->get<GPUSceneSubsystem>();
            const auto* stream = gpuScene->visibilityStream({info.lightGridViewIndex, info.lightGridViewGeneration}, info.frameIndex, info.sceneIdentity);
            require(stream && stream->accelerationStructure && info.hasStreamGeometry && info.residentRecordCount == 0,
                "Streamed raster did not publish its TLAS and visibility namespace");
            require(gpuScene->globalBufferViews().vertices.buffer == nullptr, "GPUScene uploaded resident vertices");
            require(saveRgba8Png(context.outputDirectory / (miniZorah_ ? "MiniZorahRealtime.png" : "StreamedRealtime.png"),
                reinterpret_cast<const uint8_t*>(shaded.data()), width, height, log), log);
            if (!miniZorah_) {
                graph.setNodeRuntimeProperty(graph.findNode("Deferred")->id, "materialBinning", false);
                require(bool(executor.compile(context.device, graph, width, height, log)), log);
                draw();
                const auto unbinned = pixels();
                require(shaded == unbinned, "Streamed material classification differs from unbinned deferred shading");
                graph.setNodeRuntimeProperty(graph.findNode("Deferred")->id, "debugDisableShadows", true);
                require(bool(executor.compile(context.device, graph, width, height, log)), log); draw();
                const auto unshadowed = pixels();
                size_t shadowed = 0;
                for (size_t i = 0; i < shaded.size(); ++i) {
                    shadowed += int(unshadowed[i] & 255u) - int(shaded[i] & 255u) > 8;
                }
                require(shadowed > 100, "Streamed TLAS did not attenuate direct lighting");
            }
            auto camera = executor.renderView()->camera(); camera.eye[0] += .12f;
            executor.renderView()->setCamera(camera); draw();
            auto* guides = executor.outputResource("Readback.guides")->buffer;
            guides->invalidate(); const auto* values = static_cast<const std::array<float, 4>*>(guides->map());
            size_t covered = 0, moved = 0;
            for (size_t i = 0; i < size_t(width) * height; ++i) {
                require(std::isfinite(values[i][0]) && std::isfinite(values[i][1]) && std::isfinite(values[i][2]), "Nonfinite guides");
                covered += values[i][2] < .99999f;
                moved += std::abs(values[i][0]) + std::abs(values[i][1]) > .0001f;
            }
            guides->unmap(); require(covered > 100 && moved > 100, "Camera change did not reach streamed geometry/upscaler guides");
            require(streamer->sceneReadiness().ready, "Fallback coverage never became presentable");
            rebuildAndDraw(321, 217);
            if (!miniZorah_) {
                require(bool(executor.reloadShaders(log)), "Streamed shader reload: " + log);
                for (uint32_t frame = 0; frame < 32; ++frame) { draw(); }
                streamer->collectReleasedStreams();
                require(streamer->streamCount() == 1, "Shader reload leaked the previous streaming session");
                require(streamer->sceneReadiness().ready, "Reloaded streaming shaders lost fallback coverage");
                const auto reloaded = pixels();
                require(std::count_if(reloaded.begin(), reloaded.end(), [&](uint32_t p) { return p != reloaded[0]; }) > 100,
                    "Reloaded streaming session produced an empty image");
            }
            RenderGraph empty; empty.addNode("FinalBlitPass", "Empty"); empty.markOutput("Empty.color");
            require(bool(executor.compile(context.device, empty, width, height, log)), log);
            for (uint32_t frame = 0; frame <= executor.subsystemHost()->frameSlotCount(); ++frame) { draw(); }
            streamer->collectReleasedStreams();
            require(streamer->streamCount() == 0, "Streamer retained an unused raster session after graph removal");
            return RhiTestResult::pass("Stream-only materials, unified lighting/TLAS, guides, resize and Streamer retirement verified");
        } catch (const std::exception& error) { return realtimeFailure(error.what()); }
    }
private:
    bool miniZorah_;
};
class MiniZorahRealtimeTest final : public StreamedRealtimeTest {
public:
    MiniZorahRealtimeTest() : StreamedRealtimeTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(StreamedRealtimeTest);
METALLIC_REGISTER_RHI_TEST(MiniZorahRealtimeTest);
} // namespace
} // namespace metallic::tests
