#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "TestResourceLayouts.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "RHITest.h"
#include "RenderGraphViewerTestUI.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Core/SlangCompiler.h"
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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <thread>
#include <stdexcept>

namespace metallic::tests {
namespace {

RHITestResult realtimeFailure(std::string message)
{
    spdlog::error("Realtime regression: {}", message);
    return RHITestResult::fail(std::move(message));
}

void saveRealtimeExecutionCapture(render::RenderGraphExecutor& executor, const RHITestContext& context,
    bool miniZorah)
{
    using namespace render;
    const auto require = [](bool condition, const char* message) {
        if (!condition) { throw std::runtime_error(message); }
    };
    const auto snapshot = executor.executionSnapshot();
    require(snapshot && snapshot->success && snapshot->status == RenderGraphExecutionSnapshotStatus::Submitted &&
        !snapshot->externalRecording, "Production execution capture has no successful managed submission");
    require(snapshot->passes.size() == executor.executionStats().nodes.size() && !snapshot->resources.empty() &&
        !snapshot->queues.empty() && !snapshot->batches.empty(), "Production capture omitted executed graph metadata");
    std::set<uint64_t> resourceIds;
    uint64_t textures = 0, buffers = 0, privateResources = 0, aliases = 0, stages = 0, uses = 0;
    uint64_t barriers = 0, memoryBarriers = 0, imageTransitions = 0, nativeCalls = 0, knownAllocations = 0;
    uint64_t observedAllocationBytes = 0, crossQueueWaits = 0;
    nlohmann::json resources = nlohmann::json::array(), passes = nlohmann::json::array();
    for (const auto& resource : snapshot->resources) {
        require(resource.id && resourceIds.insert(resource.id).second, "Production capture duplicated a canonical allocation");
        textures += resource.type == RenderGraphResourceType::Texture2D;
        buffers += resource.type == RenderGraphResourceType::Buffer;
        privateResources += resource.privateResource;
        aliases += resource.aliases.size();
        if (resource.memory.known) {
            require(resource.memory.allocationId == resource.id && resource.memory.sizeBytes != 0,
                "Production capture contains invalid native allocation metadata");
            ++knownAllocations;
            observedAllocationBytes += resource.memory.sizeBytes;
        }
        resources.push_back({{"id", resource.id}, {"name", resource.name}, {"aliases", resource.aliases},
            {"private", resource.privateResource}, {"type", uint32_t(resource.type)},
            {"memoryKnown", resource.memory.known}, {"memoryBlockId", std::to_string(resource.memory.memoryBlockId)},
            {"offsetBytes", resource.memory.offsetBytes}, {"sizeBytes", resource.memory.sizeBytes}});
    }
    const auto checkUses = [&](const auto& values) {
        uses += values.size();
        for (const auto& use : values) {
            require(resourceIds.contains(use.resourceId), "Production pass/stage refers to an uncaptured allocation");
        }
    };
    const auto checkBarriers = [&](const auto& values) {
        barriers += values.size();
        for (const auto& barrier : values) {
            require(resourceIds.contains(barrier.resourceId), "Production barrier refers to an uncaptured allocation");
        }
    };
    const auto countNative = [&](const SynchronizationStats& stats) {
        nativeCalls += stats.calls;
        memoryBarriers += stats.memoryBarriers;
        imageTransitions += stats.imageTransitions;
    };
    for (const auto& pass : snapshot->passes) {
        require(pass.recorded && std::any_of(snapshot->queues.begin(), snapshot->queues.end(),
            [&](const auto& queue) { return queue.id == pass.actualQueueId; }),
            "Production pass has no actual recorded queue");
        checkUses(pass.uses);
        checkBarriers(pass.barriers);
        countNative(pass.synchronization);
        nlohmann::json stageNames = nlohmann::json::array();
        for (const auto& stage : pass.stages) {
            require(stage.recorded, "Successful production capture has an unrecorded internal stage");
            ++stages;
            checkUses(stage.uses);
            checkBarriers(stage.barriers);
            countNative(stage.synchronization);
            stageNames.push_back({{"name", stage.name}, {"uses", stage.uses.size()}, {"barriers", stage.barriers.size()},
                {"restoreBoundary", stage.restoreBoundary}});
        }
        passes.push_back({{"id", pass.id}, {"name", pass.name}, {"type", pass.type}, {"queue", pass.actualQueueId},
            {"logicalQueue", uint32_t(pass.logicalQueue)}, {"uses", pass.uses.size()}, {"barriers", pass.barriers.size()},
            {"stages", std::move(stageNames)}});
    }
    for (const auto& batch : snapshot->batches) {
        require(batch.accepted, "Production capture contains an unaccepted submitted batch");
        crossQueueWaits += batch.waitPredecessors.size();
    }
    require(aliases && stages && barriers && knownAllocations, "Production capture lacks resource aliases, stages or synchronization evidence");
    const auto counts = nlohmann::json{{"passes", snapshot->passes.size()}, {"resources", snapshot->resources.size()},
        {"textures", textures}, {"buffers", buffers}, {"privateResources", privateResources}, {"aliases", aliases},
        {"queues", snapshot->queues.size()}, {"segments", snapshot->segments.size()}, {"batches", snapshot->batches.size()},
        {"crossQueueWaits", crossQueueWaits}, {"stages", stages}, {"uses", uses}, {"plannedBarriers", barriers},
        {"nativePlannerCalls", nativeCalls}, {"nativeMemoryBarriers", memoryBarriers}, {"nativeImageTransitions", imageTransitions},
        {"knownAllocations", knownAllocations}, {"observedAllocationBytes", observedAllocationBytes}};
    const std::string label = miniZorah ? "MiniZorahExecution" : "StreamedExecution";
    std::filesystem::create_directories(context.outputDirectory);
    std::ofstream report(context.outputDirectory / (label + ".json"));
    report << nlohmann::json{{"graph", snapshot->graphName}, {"generation", snapshot->graphGeneration},
        {"execution", snapshot->executionId}, {"counts", counts}, {"resources", std::move(resources)},
        {"passes", std::move(passes)}}.dump(2) << '\n';
    require(report.good(), "Could not save production execution metadata");
    spdlog::info("[Graph capture] {} {}", label, counts.dump());
    editor::RenderGraphExecutionViewer viewer;
    viewer.update(snapshot);
    viewer.setLive(false);
    ViewerUIContext ui;
    using Tab = editor::RenderGraphExecutionViewer::Tab;
    for (const auto& [tab, name] : std::array{std::pair{Tab::Resources, "resources"},
            std::pair{Tab::Queues, "queues"}, std::pair{Tab::Memory, "memory"}}) {
        const auto failure = ui.save(viewer, tab, context.outputDirectory / (label + "-" + name + ".png"));
        if (!failure.empty()) { throw std::runtime_error(failure); }
    }
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
            .entryPointName = "environmentPrefilterProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeResourceBindingDesc bindings[] = {{.binding = 2}, {.binding = 3}};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .bindings = {bindings, 2},
            .requiresRayQuery = false,
            .resourceParameters = metallic::tests::kRealtimeGuideProbeLayout,
        }, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 2, .buffer = context.outputBuffer("data").buffer()},
            {.binding = 3, .buffer = context.subsystem<render::EnvironmentLightingSubsystem>()->snapshot().prefilteredSpecularBuffer}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = {bindings, 2}});
    }
private:
    render::ComputeProgram program_;
};

class EnvironmentPrefilterTest final : public RHITest {
public:
    EnvironmentPrefilterTest() { type = RHITestType::Rendering; name = "realtime_environment_prefilter_energy"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto path = std::filesystem::absolute(context.outputDirectory / "RealtimeConstant.hdr");
        std::filesystem::create_directories(context.outputDirectory);
        std::vector<float> pixels(32 * 16 * 3);
        for (size_t i = 0; i < pixels.size(); i += 3) { pixels[i] = 2.0f; pixels[i + 1] = 1.0f; pixels[i + 2] = 0.5f; }
        if (!stbi_write_hdr(path.string().c_str(), 32, 16, 3, pixels.data())) { return realtimeFailure("HDR fixture write failed"); }
        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({.applicationName = "Environment prefilter energy",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("Requires bindless descriptors"); }
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
        const auto expected = render::color::fromLinearRec709({2.0f, 1.0f, 0.5f});
        for (size_t i = 0; i < 8; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                valid &= std::isfinite(values[i][c]) && std::abs(values[i][c] - expected[c]) < 0.002f;
            }
        }
        buffer->unmap();
        return valid ? RHITestResult::pass("GGX preserves constant radiance at every roughness, poles and wrap seam")
            : realtimeFailure("GGX filtering changed constant environment energy");
    }
};

class RealtimeReadbackPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("color").transferRead().format = render::Format::RGBA8Unorm;
        reflection.addTextureInput("motion").sampledRead().format = render::Format::RG16Sfloat;
        reflection.addTextureInput("depth").sampledRead().format = render::Format::R32Sfloat;
        reflection.addBufferOutput("pixels").buffer(uint64_t(context.width) * context.height * 4).transferWrite();
        reflection.addBufferOutput("guides").buffer(uint64_t(context.width) * context.height * 16, 16).storageReadWrite();
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "RealtimeGuideProbe",
            .entryPointName = "realtimeGuideProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeResourceBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::SampledImage},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::SampledImage},
            {.binding = 2, .kind = render::ComputeResourceBindingKind::StorageBuffer}};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .bindings = {bindings, 3},
            .requiresRayQuery = false,
            .resourceParameters = metallic::tests::kRealtimeGuideProbeLayout,
        }, log);
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        if (auto commandResult = (context.outputBuffer("pixels").buffer())->slice().and_then([&](const auto& bufferSlice) { return context.commandBuffer().copyTextureToBuffer({.texture = context.inputTexture("color").texture(),
            .buffer = bufferSlice, .bufferRowPitch = context.width() * 4,
            .bufferSlicePitch = context.width() * context.height() * 4,
            .width = context.width(), .height = context.height()}); }); !commandResult) { return commandResult; }
        auto* motion = context.inputTexture("motion").view();
        auto* depth = context.inputTexture("depth").view();
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureViews = {&motion, 1}},
            {.binding = 1, .textureViews = {&depth, 1}},
            {.binding = 2, .buffer = context.outputBuffer("guides").buffer()}};
        return program_.dispatch({
            .commandBuffer = &context.commandBuffer(),
            .bindings = {bindings, 3},
            .groupCountX = (context.width() + 7) / 8,
            .groupCountY = (context.height() + 7) / 8,
        });
    }
private:
    render::ComputeProgram program_;
};

class RealtimePipelineTest : public RHITest {
public:
    explicit RealtimePipelineTest(bool sponza = false) : sponza_(sponza)
    {
        type = RHITestType::Rendering;
        name = sponza ? "gpu_driven_sponza_realtime_pipeline" : "realtime_clustered_dlss_pipeline";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("realtime-lighting", sample, log) ||
            !sample.desc.requiresStreamline) {
            return realtimeFailure("Realtime sample metadata: " + log);
        }
        if (sponza_ && !render::setRenderSampleScenePath(sample, "Asset/Sponza/glTF/Sponza.gltf", log)) { return realtimeFailure(log); }
        for (const auto& node : sample.graph.nodes()) {
            if (node.type.find("PathTrace") != std::string::npos || node.type.find("NRD") != std::string::npos ||
                node.type == "StreamlineDLSSRRPass" || node.type == "SceneRealtimeLightingPass") {
                return realtimeFailure("Realtime pipeline contains a reference/path-tracing pass");
            }
        }
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return realtimeFailure(scene.lastLoadResult().error);
        }
        // Streamline owns process-wide Vulkan state: use one device for the test.
        auto* device = &context.device;
        if (!metallic::render::vulkan::deviceCapabilities(*device).streamlineDlssSr || !device->capabilities().meshShader) {
            return RHITestResult::skip("Requires --rhi-realtime and supported mesh shaders/DLSS-SR");
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
        sample.graph.addEdge("DLSSSR.motionVectors", "Readback.motion");
        sample.graph.addEdge("DLSSSR.depth", "Readback.depth");
        sample.graph.markOutput("Readback.pixels");
        sample.graph.markOutput("Readback.guides");
        render::RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        for (uint32_t variant = 0; variant < 3; ++variant) {
            const uint32_t width = variant == 2 ? 321 : 512, height = variant == 2 ? 217 : 384;
            // Keep the Sponza regression on GPUDrivenSample's default NR-off
            // configuration. The original realtime test covers optional NR.
            if (variant == 1 && !sponza_) {
                sample.graph.setNodeRuntimeProperty(sample.graph.findNode("DLSSNR")->id, "enabled", true);
            }
            result = executor.compile(*device, sample.graph, width, height, log);
            if (!result) { return realtimeFailure(log); }
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
        return RHITestResult::pass("Raster/LightGrid/SH/HDRI/auto exposure/SR, optional NR, camera jitter and odd-size resize");
    }
private:
    bool sponza_ = false;
};

class WorkingColorDisplayReadbackPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("color").transferRead().format = render::Format::RGBA8Unorm;
        reflection.addBufferOutput("pixels").buffer(uint64_t(context.width) * context.height * 4).transferWrite();
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        return context.outputBuffer("pixels").buffer()->slice().and_then([&](const auto& slice) {
            return context.commandBuffer().copyTextureToBuffer({.texture = context.inputTexture("color").texture(),
                .buffer = slice, .bufferRowPitch = context.width() * 4,
                .bufferSlicePitch = context.width() * context.height() * 4,
                .width = context.width(), .height = context.height()});
        });
    }
};

class WorkingColorDLSSRRTest final : public RHITest {
public:
    WorkingColorDLSSRRTest() { type = RHITestType::Rendering; name = "working_color_dlss_rr_history"; }
    RHITestResult run(RHITestContext& context) override
    {
        if (!render::vulkan::deviceCapabilities(context.device).streamlineDlssRr) {
            return RHITestResult::skip("Requires --rhi-streamline and DLSS-RR support");
        }
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("pathtracing-sample-dlss-rr", sample, log)) { return realtimeFailure(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return realtimeFailure(document.lastLoadResult().error);
        }
        // Legacy RR assets use per-pass cameras. Bind one shared view for the
        // camera-motion portion of this test, as the realtime sample already does.
        sample.graph.setViewProperties({{"camera", sample.graph.findNode("PathTrace")->properties.at("camera")}});
        sample.graph.findNode("PathTrace")->properties.erase("camera");
        sample.graph.findNode("DLSSRR")->properties.erase("camera");
        render::RenderWorld world;
        world.setScene(&document);
        if (sample.desc.environment) {
            world.setEnvironment({.path = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.environment->path});
        }
        render::registerRenderGraphPassType("WorkingColorDisplayReadbackPass", "SDK color regression readback",
            [] { return std::make_unique<WorkingColorDisplayReadbackPass>(); });
        sample.graph.addNode("WorkingColorDisplayReadbackPass", "Readback");
        sample.graph.addEdge("FinalBlit.color", "Readback.color");

        sample.graph.markOutput("Readback.pixels");

        render::RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        const uint32_t initialValidationCount = context.validationMessageCount ? context.validationMessageCount->load() : 0;
        for (uint32_t variant = 0; variant < 2; ++variant) {
            const uint32_t width = variant == 0 ? 384 : 321, height = variant == 0 ? 256 : 217;
            if (!executor.compile(context.device, sample.graph, width, height, log)) { return realtimeFailure(log); }
            for (uint32_t frame = 0; frame < 12; ++frame) {
                if (frame == 6) {
                    auto camera = executor.renderView()->camera();
                    camera.eye[0] += 0.05f;
                    executor.renderView()->setCamera(camera);
                }
                if (!executor.execute({.graphicsQueue = &context.graphicsQueue}) || !executor.waitForSubmittedWork()) {
                    return realtimeFailure("ACEScg DLSS-RR frame/history execution failed");
                }

            }
            auto* buffer = executor.outputResource("Readback.pixels")->buffer;
            buffer->invalidate();
            const auto* pixels = static_cast<const uint8_t*>(buffer->map());
            if (!pixels) { return realtimeFailure("RR color readback failed"); }
            uint64_t energy = 0;
            for (size_t i = 0; i < size_t(width) * height; ++i) {
                for (size_t c = 0; c < 3; ++c) { energy += pixels[i * 4 + c]; }
            }
            const bool saved = saveRgba8Png(context.outputDirectory / ("DLSSRRWorkingColor" + std::to_string(variant) + ".png"),
                pixels, width, height, log);
            buffer->unmap();
            if (!saved) { return realtimeFailure(log); }
            if (energy == 0 || energy == uint64_t(width) * height * 3 * 255) { return realtimeFailure("RR color is entirely black or white"); }
        }
        executor = render::RenderGraphExecutor{};
        if (context.validationMessageCount && context.validationMessageCount->load() != initialValidationCount) {
            return realtimeFailure("RR produced Vulkan validation messages");
        }
        return RHITestResult::pass("DLSS-RR working HDR/albedo, 24 frames, camera change and odd-size resize");
    }
};
METALLIC_REGISTER_RHI_TEST(WorkingColorDLSSRRTest);

class WorkingColorDLSSNonBindlessCompileTest final : public RHITest {
public:
    WorkingColorDLSSNonBindlessCompileTest() { type = RHITestType::Resource; name = "working_color_dlss_non_bindless_compile"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        // Run this case in its own process without --rhi-streamline: Streamline
        // stores the active Vulkan device globally, so do not replace a live one.
        if (vulkan::deviceCapabilities(context.device).streamline) {
            return RHITestResult::skip("Run without --rhi-streamline to isolate the non-bindless SDK device");
        }
        std::unique_ptr<Device> device;
        auto created = createDevice({.applicationName = "Non-bindless DLSS compile contract",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = false,
            .enableStreamline = true})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Streamline device unavailable"); }
        if (!created) { return realtimeFailure("Non-bindless Streamline device creation failed"); }
        if (!vulkan::deviceCapabilities(*device).streamlineDlssRr || !vulkan::deviceCapabilities(*device).streamlineDlssSr) {
            return RHITestResult::skip("DLSS-RR/SR SDK feature unavailable");
        }
        if (device->capabilities().bindlessDescriptorHeap) { return realtimeFailure("Non-bindless test device unexpectedly enabled the heap"); }
        const RenderGraphCompileContext compileContext{.device = device.get(),
            .graphicsQueue = device->getQueue(QueueType::Graphics), .width = 321, .height = 217};
        for (const auto& [type, mode] : std::array{
                std::pair{"StreamlineDLSSRRPass", "Balanced"}, std::pair{"StreamlineDLSSRRPass", "Off"},
                std::pair{"StreamlineDLSSSRPass", "Off"}, std::pair{"StreamlineDLSSSRPass", "Balanced"}}) {
            auto pass = createRenderGraphPass(type);
            if (!pass) { return realtimeFailure("Cannot create non-bindless DLSS pass"); }
            pass->setProperties({{"mode", mode}});
            std::string log;
            auto prepared = pass->prepare(compileContext, log);
            if (!prepared) { return realtimeFailure(std::string(type) + " prepare failed: " + log); }
            auto compiled = pass->compile(compileContext, log);
            const bool requiresHeap = std::string_view(type) == "StreamlineDLSSSRPass" && std::string_view(mode) != "Off";
            if (requiresHeap ? !hasError(compiled, Error::Unsupported) : !compiled) {
                return realtimeFailure(std::string(type) + "/" + mode + " changed its original bindless requirement: " + log);
            }
        }
        return RHITestResult::pass("Non-bindless RR beauty and RR/SR Off compile; SR beauty retains its existing heap requirement");
    }
};
METALLIC_REGISTER_RHI_TEST(WorkingColorDLSSNonBindlessCompileTest);

class WorkingColorHDRReadbackPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        auto& source = reflection.addTextureInput("source").transferRead();
        source.format = render::Format::RGBA32Sfloat;
        source.matchOutputExtent = false;
        reflection.addBufferOutput("pixels").buffer(uint64_t(context.width) * context.height * 16).transferWrite();
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const auto source = context.inputTexture("source");
        const auto width = source.desc().width, height = source.desc().height;
        return context.outputBuffer("pixels").buffer()->slice().and_then([&](const auto& slice) {
            return context.commandBuffer().copyTextureToBuffer({.texture = source.texture(), .buffer = slice,
                .bufferRowPitch = width * 16, .bufferSlicePitch = width * height * 16, .width = width, .height = height});
        });
    }
};

class DLSSFloat32HDRContractTest final : public RHITest {
public:
    DLSSFloat32HDRContractTest() { type = RHITestType::Resource; name = "dlss_float32_hdr_contract"; }
    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
        for (const char* type : {"StreamlineDLSSSRPass", "StreamlineDLSSRRPass"}) {
            auto pass = createRenderGraphPass(type);
            if (!pass) { return realtimeFailure("Missing DLSS pass"); }
            const auto reflection = pass->reflect({});
            const auto* input = reflection.findField("inputColor", RenderGraphFieldVisibility::Input);
            const auto* output = reflection.findField("color", RenderGraphFieldVisibility::Output);
            if (!input || !output || input->format != Format::RGBA32Sfloat || output->format != Format::RGBA32Sfloat) {
                return realtimeFailure(std::string(type) + " must preserve unexposed scene color in RGBA32F");
            }
        }
        return RHITestResult::pass("SR/RR input and output HDR retain FP32 range before exposure");
    }
};
METALLIC_REGISTER_RHI_TEST(DLSSFloat32HDRContractTest);

class DLSSFloat32HDRFixturePass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        using namespace render;
        RenderPassReflection reflection;
        for (const auto& [name, format] : {
                std::pair{"color", Format::RGBA32Sfloat}, std::pair{"motionVectors", Format::RG16Sfloat},
                std::pair{"depth", Format::R32Sfloat}, std::pair{"albedo", Format::RGBA16Sfloat},
                std::pair{"specularAlbedo", Format::RGBA16Sfloat}, std::pair{"normalRoughness", Format::RGBA16Sfloat},
                std::pair{"specularHitDistance", Format::R32Sfloat}}) {
            auto& field = reflection.addTextureOutput(name).transferWrite();
            field.format = format;
            if (std::string_view(name) == "color") { field.colorEncoding = DisplayColorEncoding::SceneLinear; }
        }
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        for (const char* name : {"color", "motionVectors", "depth", "albedo", "specularAlbedo", "normalRoughness", "specularHitDistance"}) {
            const auto texture = context.outputTexture(name);
            if (!texture.valid()) { continue; }
            const std::array<float, 4> value = std::string_view(name) == "color"
                ? std::array<float, 4>{1e9f, 2e8f, 0.125f, 1.0f} : std::array<float, 4>{0.5f, 0.0f, 0.0f, 0.0f};
            auto result = context.commandBuffer().clearColorTexture(*texture.texture(),
                render::TextureLayout::TransferDestination, {value[0], value[1], value[2], value[3]});
            if (!result) { return result; }
        }
        return {};
    }
};

class DLSSFloat32HDROffCopyTest final : public RHITest {
public:
    DLSSFloat32HDROffCopyTest() { type = RHITestType::Rendering; name = "dlss_float32_hdr_off_copy"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto capabilities = vulkan::deviceCapabilities(context.device);
        if (!capabilities.streamlineDlssSr || !capabilities.streamlineDlssRr) {
            return RHITestResult::skip("Requires --rhi-streamline and SR/RR support");
        }
        registerRenderGraphPassType("DLSSFloat32HDRFixturePass", "FP32 solar-range DLSS input",
            [] { return std::make_unique<DLSSFloat32HDRFixturePass>(); });
        registerRenderGraphPassType("WorkingColorHDRReadbackPass", "FP32 HDR raw readback",
            [] { return std::make_unique<WorkingColorHDRReadbackPass>(); });
        constexpr uint32_t width = 38, height = 26;
        for (const char* type : {"StreamlineDLSSSRPass", "StreamlineDLSSRRPass"}) {
            const bool rr = std::string_view(type) == "StreamlineDLSSRRPass";
            RenderGraph graph;
            graph.addNode("DLSSFloat32HDRFixturePass", "Source");
            graph.addNode(type, "DLSS", {{"mode", "Off"}});
            graph.addNode("WorkingColorHDRReadbackPass", "InputReadback");
            graph.addNode("WorkingColorHDRReadbackPass", "OutputReadback");
            graph.addEdge("Source.color", "DLSS.inputColor");
            graph.addEdge("Source.motionVectors", "DLSS.motionVectors");
            graph.addEdge("Source.depth", rr ? "DLSS.linearDepth" : "DLSS.depth");
            if (rr) {
                for (const char* name : {"albedo", "specularAlbedo", "normalRoughness", "specularHitDistance"}) {
                    graph.addEdge(std::string("Source.") + name, std::string("DLSS.") + name);
                }
            }
            graph.addEdge("Source.color", "InputReadback.source");
            graph.addEdge("DLSS.color", "OutputReadback.source");
            graph.markOutput("InputReadback.pixels");
            graph.markOutput("OutputReadback.pixels");
            RenderGraphExecutor executor;
            std::string log;
            if (!executor.compile(context.device, graph, width, height, log) ||
                !executor.execute({.graphicsQueue = &context.graphicsQueue}) || !executor.waitForSubmittedWork()) {
                return realtimeFailure(std::string(type) + " FP32 Off graph failed: " + log);
            }
            auto* inputBuffer = executor.outputResource("InputReadback.pixels")->buffer;
            auto* outputBuffer = executor.outputResource("OutputReadback.pixels")->buffer;
            inputBuffer->invalidate();
            outputBuffer->invalidate();
            const auto* input = static_cast<const float*>(inputBuffer->map());
            const auto* output = static_cast<const float*>(outputBuffer->map());
            if (!input || !output) {
                if (input) { inputBuffer->unmap(); }
                if (output) { outputBuffer->unmap(); }
                return realtimeFailure("FP32 DLSS readback map failed");
            }
            bool valid = std::memcmp(input, output, size_t(width) * height * 4 * sizeof(float)) == 0;
            for (size_t pixel = 0; pixel < size_t(width) * height; ++pixel) {
                valid &= std::isfinite(output[pixel * 4]) && output[pixel * 4] == 1e9f &&
                    output[pixel * 4 + 1] == 2e8f && output[pixel * 4 + 2] == 0.125f;
            }
            inputBuffer->unmap();
            outputBuffer->unmap();
            if (!valid) { return realtimeFailure(std::string(type) + " clamped/quantized FP32 solar-range color in mode Off"); }
        }
        return RHITestResult::pass("SR/RR Off preserve 1e9 solar-range color and dark channels byte for byte in FP32");
    }
};
METALLIC_REGISTER_RHI_TEST(DLSSFloat32HDROffCopyTest);

class WorkingColorDLSSDebugBypassTest : public RHITest {
public:
    explicit WorkingColorDLSSDebugBypassTest(bool rayReconstruction = false) : rayReconstruction_(rayReconstruction)
    {
        type = RHITestType::Rendering;
        name = rayReconstruction ? "working_color_dlss_rr_debug_bypass" : "working_color_dlss_sr_debug_bypass";
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto& capabilities = vulkan::deviceCapabilities(context.device);
        if (rayReconstruction_ ? !capabilities.streamlineDlssRr : !capabilities.streamlineDlssSr) {
            return RHITestResult::skip("Requires --rhi-streamline and the selected DLSS feature");
        }
        try {
            const auto require = [](bool condition, const std::string& message) {
                if (!condition) { throw std::runtime_error(message); }
            };
            RenderSampleLoadResult sample;
            std::string log;
            require(loadBuiltInRenderSample(rayReconstruction_ ? "pathtracing-sample-dlss-rr" : "pathtracing-sample-dlss-sr", sample, log), log);
            scene::SceneDocument document;
            require(document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath), document.lastLoadResult().error);
            const std::string sdkName = rayReconstruction_ ? "DLSSRR" : "DLSSSR";
            const auto pathTraceId = sample.graph.findNode("PathTrace")->id;
            sample.graph.setViewProperties({{"camera", sample.graph.findNode("PathTrace")->properties.at("camera")}});
            sample.graph.findNode("PathTrace")->properties.erase("camera");
            sample.graph.findNode(sdkName)->properties.erase("camera");
            RenderWorld world;
            world.setScene(&document);
            if (sample.desc.environment) {
                world.setEnvironment({.path = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.environment->path});
            }
            registerRenderGraphPassType("WorkingColorHDRReadbackPass", "SDK debug raw readback",
                [] { return std::make_unique<WorkingColorHDRReadbackPass>(); });
            sample.graph.addNode("WorkingColorHDRReadbackPass", "InputReadback");
            sample.graph.addNode("WorkingColorHDRReadbackPass", "OutputReadback");
            sample.graph.addEdge("PathTrace.color", "InputReadback.source");
            sample.graph.addEdge(sdkName + ".color", "OutputReadback.source");
            sample.graph.markOutput("InputReadback.pixels");
            sample.graph.markOutput("OutputReadback.pixels");
            RenderGraphExecutor executor;
            executor.bindRenderWorld(&world);
            constexpr uint32_t outputWidth = 321, outputHeight = 217;
            require(bool(executor.compile(context.device, sample.graph, outputWidth, outputHeight, log)), log);
            executor.setExecutionCaptureEnabled(true);
            const auto sourceWidth = executor.outputResource("PathTrace.color")->desc.width;
            const auto sourceHeight = executor.outputResource("PathTrace.color")->desc.height;
            require(sourceWidth != outputWidth || sourceHeight != outputHeight, "Debug regression must exercise DLSS input/output resize");
            const auto readPixels = [&](const char* resource, uint32_t width, uint32_t height) {
                auto* buffer = executor.outputResource(resource)->buffer;
                buffer->invalidate();
                const auto* raw = static_cast<const std::array<float, 4>*>(buffer->map());
                require(raw != nullptr, "Debug raw readback failed");
                std::vector<std::array<float, 4>> pixels(raw, raw + size_t(width) * height);
                buffer->unmap();
                return pixels;
            };
            for (const char* debug : {"baseColor", "shadowTransmittance", "geometryNormal", "frontFace", "material", "final"}) {
                require(sample.graph.setNodeRuntimeProperty(pathTraceId, "debugView", debug), "Cannot set production debug mode");
                require(executor.syncRuntimeProperties(sample.graph), "Cannot sync production debug mode");
                require(bool(executor.execute({.graphicsQueue = &context.graphicsQueue})) && bool(executor.waitForSubmittedWork()),
                    "Production DLSS debug frame failed");
                const bool sceneColor = std::string_view(debug) == "final";
                const auto encoding = sceneColor ? DisplayColorEncoding::SceneLinear :
                    (std::string_view(debug) == "baseColor" || std::string_view(debug) == "shadowTransmittance"
                        ? DisplayColorEncoding::DisplayLinearRec709 : DisplayColorEncoding::sRGB);
                require(executor.outputResource("PathTrace.color")->colorEncoding == encoding &&
                    executor.outputResource(sdkName + ".color")->colorEncoding == encoding,
                    "DLSS debug input/output encoding changed");
                require(executor.outputResource("AutoExposure.color")->colorEncoding ==
                    (sceneColor ? DisplayColorEncoding::ExposedLinear : encoding), "AutoExposure lost DLSS debug encoding");
                const auto snapshot = executor.executionSnapshot();
                require(snapshot && snapshot->success, "Missing debug execution capture");
                const auto pass = std::find_if(snapshot->passes.begin(), snapshot->passes.end(),
                    [&](const auto& value) { return value.name == sdkName; });
                require(pass != snapshot->passes.end(), "Missing DLSS execution capture");
                const bool bypass = std::any_of(pass->stages.begin(), pass->stages.end(),
                    [](const auto& stage) { return stage.name == "DLSS display/data bypass"; });
                const bool sdk = std::any_of(pass->stages.begin(), pass->stages.end(),
                    [](const auto& stage) { return stage.name == "DLSS ray reconstruction" || stage.name == "DLSS super resolution"; });
                require(sceneColor ? sdk && !bypass : bypass && !sdk, "Display/debug pixels entered DLSS HDR SDK or scene path stayed bypassed");
                if (sceneColor) { continue; }
                const auto input = readPixels("InputReadback.pixels", sourceWidth, sourceHeight);
                const auto output = readPixels("OutputReadback.pixels", outputWidth, outputHeight);
                const auto load = [&](int x, int y, size_t channel) {
                    return input[size_t(std::clamp(y, 0, int(sourceHeight) - 1)) * sourceWidth + std::clamp(x, 0, int(sourceWidth) - 1)][channel];
                };
                for (uint32_t y = 0; y < outputHeight; ++y) { for (uint32_t x = 0; x < outputWidth; ++x) {
                    const float px = (float(x) + 0.5f) * sourceWidth / outputWidth - 0.5f;
                    const float py = (float(y) + 0.5f) * sourceHeight / outputHeight - 0.5f;
                    const int ix = int(std::floor(px)), iy = int(std::floor(py));
                    const float fx = px - ix, fy = py - iy;
                    for (size_t c = 0; c < 3; ++c) {
                        const float top = std::lerp(load(ix, iy, c), load(ix + 1, iy, c), fx);
                        const float bottom = std::lerp(load(ix, iy + 1, c), load(ix + 1, iy + 1, c), fx);
                        const float expected = std::lerp(top, bottom, fy);
                        const float actual = output[size_t(y) * outputWidth + x][c];
                        require(std::isfinite(actual) && std::abs(actual - expected) < 0.0012f,
                            std::string("DLSS debug resize changed color: ") + debug);
                    }
                }}
            }
            return RHITestResult::pass("Production DLSS graph: five display/data debug modes bypass SDK, preserve raw resized pixels/encoding, then resume scene HDR");
        } catch (const std::exception& error) { return realtimeFailure(error.what()); }
    }
private:
    bool rayReconstruction_;
};
class WorkingColorDLSSRRDebugBypassTest final : public WorkingColorDLSSDebugBypassTest {
public:
    WorkingColorDLSSRRDebugBypassTest() : WorkingColorDLSSDebugBypassTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(WorkingColorDLSSDebugBypassTest);
METALLIC_REGISTER_RHI_TEST(WorkingColorDLSSRRDebugBypassTest);

class GPUDrivenSponzaRealtimePipelineTest final : public RealtimePipelineTest {
public:
    GPUDrivenSponzaRealtimePipelineTest() : RealtimePipelineTest(true) {}
};

METALLIC_REGISTER_RHI_TEST(RealtimePipelineTest);
METALLIC_REGISTER_RHI_TEST(GPUDrivenSponzaRealtimePipelineTest);
METALLIC_REGISTER_RHI_TEST(EnvironmentPrefilterTest);

class RealtimeShadowTest final : public RHITest {
public:
    RealtimeShadowTest() { type = RHITestType::Rendering; name = "realtime_ray_traced_sigma_shadows"; }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("realtime-lighting", sample, log) ||
            !render::setRenderSampleScenePath(sample, "Asset/LookDev/OpenPBRDefault/OpenPbrDefault.gltf", log)) {
            return realtimeFailure(log);
        }
        sample.graph.removeNode(sample.graph.findNode("DLSSSR")->id);
        sample.graph.removeNode(sample.graph.findNode("DLSSNR")->id);
        sample.graph.addEdge("Deferred.color", "AutoExposure.source");
        sample.graph.addEdge("AutoExposure.color", "FinalBlit.source");
        sample.graph.setViewProperties({{"camera", {{"eye", {0.0, 1.4, 3.65}}, {"center", {0.0, 0.8, 0.0}},
            {"fovDegrees", 50.0}, {"znear", 0.05}, {"zfar", 100.0}}}, {"temporalJitter", true}});
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return realtimeFailure(scene.lastLoadResult().error);
        }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        auto result = preview.initialize(context.enableValidation, true);
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders"); }
        if (!result) { return realtimeFailure("Initialize realtime shadow preview"); }
        preview.setEnvironment({.enabled = false});
        scene::LightingSettings lighting;
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 4.0f;
        environment::WorldEnvironment environment;
        environment.sun = {.direction = float3(0.6f, -1.0f, -0.3f), .illuminance = 30.0f,
            .angularRadius = 1.5f * 0.01745329252f, .enabled = true};
        // Disabled and active local sources must not change the fixed Sun slot.
        scene::PunctualLight inactive;
        inactive.enabled = false;
        lighting.lights.push_back(inactive);
        auto point = inactive;
        point.enabled = true;
        point.properties.intensity = 0.001;
        lighting.lights.push_back(point);
        preview.setLighting(lighting);
        preview.setWorldEnvironment(environment);
        const auto* shadowNode = sample.graph.findNode("Shadows");
        if (shadowNode == nullptr || shadowNode->type != "RayTracedShadowPass") {
            return realtimeFailure("Realtime pipeline is missing the explicit shadow stage");
        }
        const auto shadows = shadowNode->id;
        const auto deferred = sample.graph.findNode("Deferred")->id;
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
        environment.sun.angularRadius = 0.0f;
        preview.setWorldEnvironment(environment);
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
        sample.graph.setNodeRuntimeProperty(shadows, "shadowLightIndex", 0);
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
        if (changed < 30) { return realtimeFailure("Trace settings did not affect the selected celestial Sun slot"); }
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
            environment.sun.angularRadius = angles[angle] * 0.01745329252f;
            preview.setWorldEnvironment(environment);
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
        return RHITestResult::pass("Full TLAS + SIGMA, live light selection, legacy settings, resize and both resolve paths; "
            "penumbra pixels at 0/1/8 degrees: " + std::to_string(softened[0]) + "/" +
            std::to_string(softened[1]) + "/" + std::to_string(softened[2]) +
            "; final lighting softened pixels: " + std::to_string(lightened));
    }
};

METALLIC_REGISTER_RHI_TEST(RealtimeShadowTest);

class StreamedRealtimeTest : public RHITest {
public:
    explicit StreamedRealtimeTest(bool miniZorah = false) : miniZorah_(miniZorah)
    {
        type = RHITestType::Rendering;
        name = miniZorah ? "minizorah_realtime_pipeline" : "streamed_realtime_pipeline";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto require = [](bool condition, const std::string& message) {
            if (!condition) { throw std::runtime_error(message); }
        };
        if (miniZorah_ && std::getenv("METALLIC_TEST_MINIZORAH") == nullptr) {
            return RHITestResult::skip("Set METALLIC_TEST_MINIZORAH=1 for the default full scene");
        }
        if (!context.device.capabilities().meshShader || !context.device.capabilities().clusterAccelerationStructure ||
            (miniZorah_ && !metallic::render::vulkan::deviceCapabilities(context.device).streamlineDlssSr)) {
            return RHITestResult::skip("Requires --rhi-realtime with mesh shaders, CLAS and DLSS-SR");
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
                    {"znear", .01}, {"zfar", 100}, {"fovDegrees", 50}}}, {"temporalJitter", false}});
                scene::Scene fixture;
                require(fixture.loadStreamMetadata(source), fixture.lastLoadResult().error);
                const auto center = fixture.bounds().center();
                const auto radius = fixture.bounds().radius();
                auto view = graph.viewProperties();
                view["camera"]["eye"] = {center.x, center.y + radius * .3f, center.z + radius * 3};
                view["camera"]["center"] = {center.x, center.y, center.z};
                graph.setViewProperties(view);
                graph.removeNode(graph.findNode("DLSSSR")->id);
                graph.removeNode(graph.findNode("DLSSNR")->id);
                graph.addEdge("Deferred.color", "AutoExposure.source");
                graph.addEdge("AutoExposure.color", "FinalBlit.source");
                graph.findNode("Shadows")->properties["sigmaDenoise"] = false;
            }
            registerRenderGraphPassType("RealtimeReadbackPass", "Realtime GPU regression readback",
                [] { return std::make_unique<RealtimeReadbackPass>(); });
            graph.addNode("RealtimeReadbackPass", "Readback");
            graph.addEdge("FinalBlit.color", "Readback.color");
            graph.addEdge(miniZorah_ ? "DLSSSR.motionVectors" : "Deferred.motionVectors", "Readback.motion");
            graph.addEdge(miniZorah_ ? "DLSSSR.depth" : "Deferred.deviceDepth", "Readback.depth");
            graph.markOutput("Readback.pixels"); graph.markOutput("Readback.guides");
            RenderWorld world;
            // The compact-cook fixture must produce lit geometry without an HDRI
            // that could hide a failed visibility-to-surface decode.
            world.setEnvironment({.enabled = miniZorah_, .path = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.environment->path});
            scene::LightingSettings lighting;
            lighting.autoExposure.enabled = miniZorah_;
            lighting.exposureEV100 = 2;
            environment::WorldEnvironment environment;
            environment.sun = {.direction = float3(.6f, -1, -.3f), .illuminance = 10.0f,
                .angularRadius = 0.0f, .enabled = true};
            world.setWorldEnvironment(environment);
            world.setLighting(lighting);
            RenderGraphExecutor executor;
            executor.bindRenderWorld(&world);
            uint32_t width = 512, height = 320;
            const char* aliasSetting = std::getenv("METALLIC_TEST_TEXTURE_ALIASING");
            RenderGraphCompileOptions compileOptions;
            compileOptions.enableTextureAliasing = aliasSetting && std::string_view(aliasSetting) == "1";
            const char* bufferAliasSetting = std::getenv("METALLIC_TEST_BUFFER_ALIASING");
            compileOptions.enableBufferAliasing = bufferAliasSetting && std::string_view(bufferAliasSetting) == "1";
            nlohmann::json memorySamples = nlohmann::json::array();
            bool memoryCorrectnessVerified = false;
            const std::string memoryLabel = std::string(miniZorah_ ? "MiniZorah" : "Streamed") +
                "TextureMemory-Aliasing" + (compileOptions.enableTextureAliasing ? "On" : "Off");
            const auto measureTextureMemory = [&](std::string_view phase) {
                const auto& stats = executor.textureMemoryStats();
                require(stats.complete, "Production texture allocation statistics are incomplete");
                require(stats.logicalBytes + stats.overheadBytes == stats.backingBytes + stats.savedBytes,
                    "Texture memory accounting does not balance");
                const auto& buffers = executor.bufferMemoryStats();
                require(buffers.complete, "Production buffer allocation statistics are incomplete");
                require(buffers.logicalBytes + buffers.overheadBytes == buffers.backingBytes + buffers.savedBytes,
                    "Buffer memory accounting does not balance");
                const auto budget = context.device.memoryBudget();
                const auto& frameResources = budget.domains[size_t(MemoryBudgetDomain::FrameResources)];
                nlohmann::json domains = nlohmann::json::array();
                constexpr std::array domainNames{"Other", "Geometry", "CLAS", "CLASScratch", "RayTracing",
                    "MaterialTextures", "FrameResources", "Upload"};
                static_assert(domainNames.size() == size_t(MemoryBudgetDomain::Count));
                for (size_t index = 0; index < budget.domains.size(); ++index) {
                    const auto& domain = budget.domains[index];
                    domains.push_back({{"name", domainNames[index]}, {"allocationBytes", domain.allocationBytes},
                        {"peakAllocationBytes", domain.peakAllocationBytes},
                        {"deviceLocalBytes", domain.deviceLocalBytes}, {"allocationCount", domain.allocationCount}});
                }
                nlohmann::json heaps = nlohmann::json::array();
                for (size_t index = 0; index < budget.heaps.size(); ++index) {
                    const auto& heap = budget.heaps[index];
                    heaps.push_back({{"index", index}, {"deviceLocal", heap.deviceLocal},
                        {"sizeBytes", heap.sizeBytes}, {"budgetBytes", heap.budgetBytes},
                        {"usageBytes", heap.usageBytes}, {"blockBytes", heap.blockBytes},
                        {"allocationBytes", heap.allocationBytes}, {"blockCount", heap.blockCount},
                        {"allocationCount", heap.allocationCount}});
                }
                nlohmann::json slots = nlohmann::json::array();
                for (const auto& slot : stats.slots) {
                    slots.push_back({{"complete", slot.complete}, {"backingAllocationId", slot.backingAllocationId},
                        {"logicalBytes", slot.logicalBytes}, {"backingBytes", slot.backingBytes},
                        {"savedBytes", slot.savedBytes}, {"overheadBytes", slot.overheadBytes},
                        {"resources", slot.resources}});
                }
                memorySamples.push_back({{"phase", phase}, {"width", width}, {"height", height},
                    {"textureMemory", {{"complete", stats.complete}, {"logicalBytes", stats.logicalBytes},
                        {"backingBytes", stats.backingBytes}, {"savedBytes", stats.savedBytes},
                        {"overheadBytes", stats.overheadBytes}, {"textureCount", stats.textureCount},
                        {"transientTextureCount", stats.transientTextureCount},
                        {"eligibleTextureCount", stats.eligibleTextureCount},
                        {"pinnedTextureCount", stats.pinnedTextureCount},
                        {"unknownTextureCount", stats.unknownTextureCount},
                        {"aliasedTextureCount", stats.aliasedTextureCount},
                        {"aliasSlotCount", stats.aliasSlotCount},
                        {"backingAllocationCount", stats.backingAllocationCount}, {"slots", std::move(slots)}}},
                    {"bufferMemory", {{"aliasingEnabled", buffers.aliasingEnabled}, {"complete", buffers.complete},
                        {"logicalBytes", buffers.logicalBytes}, {"backingBytes", buffers.backingBytes},
                        {"savedBytes", buffers.savedBytes}, {"overheadBytes", buffers.overheadBytes},
                        {"bufferCount", buffers.bufferCount}, {"transientBufferCount", buffers.transientBufferCount},
                        {"eligibleBufferCount", buffers.eligibleBufferCount}, {"pinnedBufferCount", buffers.pinnedBufferCount},
                        {"aliasedBufferCount", buffers.aliasedBufferCount}, {"aliasSlotCount", buffers.aliasSlotCount},
                        {"backingAllocationCount", buffers.backingAllocationCount}, {"unknownBufferCount", buffers.unknownBufferCount}}},
                    {"deviceTelemetry", {{"frameResourceAllocationBytes", frameResources.allocationBytes},
                        {"frameResourceDeviceLocalBytes", frameResources.deviceLocalBytes},
                        {"frameResourceAllocationCount", frameResources.allocationCount},
                        {"driverBudget", budget.driverBudget}, {"domains", std::move(domains)},
                        {"heaps", std::move(heaps)}}}});
                std::filesystem::create_directories(context.outputDirectory);
                std::ofstream report(context.outputDirectory / (memoryLabel + ".json"));
                report << nlohmann::json{{"schemaVersion", 1}, {"workload", name},
                    {"textureAliasingEnabled", compileOptions.enableTextureAliasing},
                    {"bufferAliasingEnabled", compileOptions.enableBufferAliasing},
                    {"correctnessVerified", memoryCorrectnessVerified},
                    {"scope", "Separate owned graph texture and buffer native capacities; buffer totals include independent host readbacks. logicalBytes is the independent allocation counterfactual; backingBytes counts each physical backing once. Device telemetry includes private resources, scene/SDK allocations and driver usage, and is not an alias savings metric."},
                    {"samples", memorySamples}}.dump(2) << '\n';
                require(report.good(), "Could not save production texture memory statistics");
                spdlog::info("[Texture memory] {} {} {}x{} logical={} backing={} saved={} overhead={} aliasTextures={} aliasSlots={} FrameResources={}",
                    memoryLabel, phase, width, height, stats.logicalBytes, stats.backingBytes, stats.savedBytes,
                    stats.overheadBytes, stats.aliasedTextureCount, stats.aliasSlotCount, frameResources.allocationBytes);
                spdlog::info("[Buffer memory] {} {} aliasing={} count={} eligible={} aliased={} slots={} logical={} backing={} saved={}",
                    memoryLabel, phase, buffers.aliasingEnabled, buffers.bufferCount, buffers.eligibleBufferCount,
                    buffers.aliasedBufferCount, buffers.aliasSlotCount, buffers.logicalBytes, buffers.backingBytes, buffers.savedBytes);
            };
            const auto compileGraph = [&](std::string_view phase) {
                const auto begin = std::chrono::steady_clock::now();
                require(bool(executor.compile(context.device, graph, width, height, compileOptions, log)), log);
                const auto compileMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                measureTextureMemory(phase);
                return compileMs;
            };
            compileGraph("initial-compiled");
            auto* streamer = executor.subsystemHost()->get<StreamerSubsystem>();
            require(streamer != nullptr && streamer->streamCount() == 1, "Streamer must own exactly one raster session");
            require(!streamer->sceneReadiness().ready && streamer->sceneReadiness().requiredPages != 0,
                "Scene became presentable before fallback uploads");
            const scene::Scene* metadata = nullptr;
            require(bool(streamer->manager().resolveScene(graph.findNode("VBuffer")->properties, nullptr, log).transform([&](auto value) { metadata = std::move(value); })), log);
            require(metadata && metadata->hasStreamGeometry(), "Default producer must resolve streaming metadata");
            for (const auto& primitive : metadata->renderPrimitives()) {
                require(primitive.positions.empty() && primitive.indices.empty(), "Full source geometry became resident");
            }
            std::shared_ptr<SceneResourceSnapshot> materials;
            require(bool(streamer->manager().acquire(context.device, context.graphicsQueue, graph.findNode("Deferred")->properties, metadata, SceneResourceFeatureBits::Materials, log).transform([&](auto value) { materials = std::move(value); })), log);
            require(materials->pathTraceResources->valid() && materials->pathTraceResources->materialBuffer() != nullptr &&
                materials->pathTraceResources->shadingVertexBuffer() == nullptr &&
                materials->pathTraceResources->indexBuffer() == nullptr &&
                !materials->pathTraceResources->accelerationStructure().valid(), "Material acquisition imported resident geometry/RTAS");
            std::shared_ptr<SceneResourceSnapshot> rejected;
            require(!streamer->manager().acquire(context.device, context.graphicsQueue, graph.findNode("Deferred")->properties, metadata, SceneResourceFeatureBits::Geometry, log).transform([&](auto value) { rejected = std::move(value); }),
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
                width = newWidth; height = newHeight;
                const auto compileMs = compileGraph("viewport-rebuild-compiled");
                draw();
                measureTextureMemory("viewport-rebuild-completed");
                require(executor.executionStats().streaming.size() == 1, "Stream missing after graph rebuild");
                const auto& after = executor.executionStats().streaming.front();
                require(after.generation == before.generation && after.frameIndex == before.frameIndex + 1 &&
                    after.totalUploadBytes >= before.totalUploadBytes,
                    "Viewport graph rebuild restarted streaming and discarded loaded pages");
                require(streamer->streamCount() == 1, "Viewport graph rebuild allocated another streaming session");
                spdlog::info("Stream preserved across {}x{} rebuild: generation={} frame={} compile={:.3f} ms",
                    width, height, after.generation, after.frameIndex, compileMs);
            };
            const char* captureSetting = std::getenv("METALLIC_TEST_GRAPH_CAPTURE");
            const bool captureGraph = captureSetting && std::string_view(captureSetting) == "1";
            const uint32_t frameCount = miniZorah_ ? 180u : 48u;
            uint32_t sceneReadyFrame = 0;
            for (uint32_t frame = 0; frame < frameCount; ++frame) {
                if (captureGraph && frame + 1 == frameCount) { executor.setExecutionCaptureEnabled(true); }
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
            measureTextureMemory(miniZorah_ ? "180-frames-completed" : "48-frames-completed");
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
            if (captureGraph) {
                executor.setExecutionCaptureEnabled(false);
                saveRealtimeExecutionCapture(executor, context, miniZorah_);
            }
            if (!miniZorah_) {
                graph.setNodeRuntimeProperty(graph.findNode("Deferred")->id, "materialBinning", false);
                compileGraph("unbinned-compiled");
                draw();
                const auto unbinned = pixels();
                require(shaded == unbinned, "Streamed material classification differs from unbinned deferred shading");
                graph.setNodeRuntimeProperty(graph.findNode("Deferred")->id, "debugDisableShadows", true);
                compileGraph("unshadowed-compiled"); draw();
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
                measureTextureMemory("shaders-reloaded");
                for (uint32_t frame = 0; frame < 32; ++frame) { draw(); }
                streamer->collectReleasedStreams();
                require(streamer->streamCount() == 1, "Shader reload leaked the previous streaming session");
                require(streamer->sceneReadiness().ready, "Reloaded streaming shaders lost fallback coverage");
                const auto reloaded = pixels();
                require(std::count_if(reloaded.begin(), reloaded.end(), [&](uint32_t p) { return p != reloaded[0]; }) > 100,
                    "Reloaded streaming session produced an empty image");
            }
            RenderGraph empty; empty.addNode("FinalBlitPass", "Empty"); empty.markOutput("Empty.color");
            require(bool(executor.compile(context.device, empty, width, height, compileOptions, log)), log);
            measureTextureMemory("raster-session-retired");
            for (uint32_t frame = 0; frame <= executor.subsystemHost()->frameSlotCount(); ++frame) { draw(); }
            streamer->collectReleasedStreams();
            require(streamer->streamCount() == 0, "Streamer retained an unused raster session after graph removal");
            memoryCorrectnessVerified = true;
            measureTextureMemory("all-checks-completed");
            return RHITestResult::pass("Stream-only materials, unified lighting/TLAS, guides, resize and Streamer retirement verified");
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
