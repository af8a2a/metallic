#include "Editor/EditorApplication.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include "imgui.h"
#include "imgui_internal.h"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <chrono>

namespace metallic {

bool EditorApplication::runNsightCaptureSmokeTest()
{
    using render::profiling::NsightGraphicsCaptureState;
    if (nsightGraphicsCapture_.state() != NsightGraphicsCaptureState::Ready) {
        spdlog::error("[Smoke Nsight] Requires Nsight SDK injection before graphics initialization");
        return false;
    }
    const auto renderCaptureFrame = [&]() {
        auto profileFrame = profiler_.beginFrame();
        if (!waitForFrameSlotBeforeInput()) { return false; }
        const render::vulkan::StreamlineFrameScope streamlineFrame;
        pollEvents();
        return renderFrame() && viewportPreviewValid_;
    };
    // Settle streaming, the editor layout and DLSS before using the same request
    // and Present boundaries as the capture button.
    for (uint32_t frame = 0; frame < 90; ++frame) {
        if (!renderCaptureFrame()) { return false; }
    }
    for (uint32_t capture = 0; capture < 3; ++capture) {
        if (capture > 0) {
            // Exercise DLSS/resource recreation as well as steady-state capture.
            if (!SDL_SetWindowSize(window_, capture == 1 ? 1280 : 1600, capture == 1 ? 720 : 900)) {
                return false;
            }
            for (uint32_t frame = 0; frame < 48; ++frame) {
                if (!renderCaptureFrame()) { return false; }
                SDL_Delay(10); // Allow the normal resize debounce to expire.
            }
        }
        if (viewportTextureWidth_ <= 256) {
            spdlog::error("[Smoke Nsight] Must exercise a full-size viewport, not the reduced smoke preview");
            return false;
        }
        requestNsightGraphicsCapture();
        if (!nsightGraphicsCapture_.hasOutstandingCapture()) { return false; }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while (nsightGraphicsCapture_.hasOutstandingCapture() &&
            std::chrono::steady_clock::now() < deadline) {
            if (!renderCaptureFrame()) { return false; }
            pollNsightGraphicsCapture();
        }
        std::error_code error;
        const auto& path = nsightGraphicsCapture_.capturePath();
        const auto bytes = std::filesystem::file_size(path, error);
        if (nsightGraphicsCapture_.state() != NsightGraphicsCaptureState::CaptureCompleted ||
            error || bytes == 0) {
            spdlog::error("[Smoke Nsight] Capture did not complete: {}", nsightGraphicsCapture_.statusText());
            return false;
        }
        spdlog::info("[Smoke Nsight] Capture {} viewport {}x{}: '{}' ({} bytes)",
            capture + 1, viewportTextureWidth_, viewportTextureHeight_, path.string(), bytes);
    }
    return true;
}

bool EditorApplication::runDlssCameraSmokeTest()
{
    const auto expect = [](bool condition, const char* message) {
        if (!condition) { spdlog::error("[Smoke DLSS Camera] {}", message); }
        return condition;
    };
    auto* dlssNode = renderGraph_.findNode("DlssRr");
    if (dlssNode == nullptr) { dlssNode = renderGraph_.findNode("DlssSr"); }
    if (!expect(dlssNode != nullptr && graphExecutor_->renderView() == &viewportView_, "DLSS graph is bound to the viewport view")) {
        return false;
    }
    const bool rasterCamera = dlssNode->type == "StreamlineDlssSrPass";
    const auto initialView = viewportCameraProperties();
    const std::string preview = activePreviewOutput_;
    for (const auto& node : renderGraph_.nodes()) {
        activePreviewOutput_ = node.name + ".color";
        if (!expect(viewportCameraProperties() == initialView,
                "Intermediate previews preserve the viewport view")) { return false; }
    }
    activePreviewOutput_ = preview;
    const auto nodeProperties = [&]() {
        render::RenderGraphProperties result = render::RenderGraphProperties::array();
        for (const auto& node : renderGraph_.nodes()) { result.push_back(node.runtimeProperties); }
        return result;
    };
    const uint32_t dlssId = dlssNode->id;
    auto properties = viewportCameraProperties();
    uint32_t resetSerial = 7;
    renderGraph_.setNodeRuntimeProperty(dlssId, "resetSerial", resetSerial);
    const auto renderDlssFrame = [&]() {
        if (!waitForFrameSlotBeforeInput()) { return false; }
        const render::vulkan::StreamlineFrameScope streamlineFrame;
        return renderFrame();
    };
    // Let ImGui's initial docking layout and debounced viewport resize settle.
    for (uint32_t frame = 0; frame < 8; ++frame) {
        if (!renderDlssFrame() || !expect(viewportPreviewValid_, "Initial DLSS frame renders")) { return false; }
    }
    const uint64_t historyRevision = historyResources_.invalidationRevision();
    const uint64_t reprojectionRevision = historyResources_.reprojectionInvalidationRevision();
    for (uint32_t frame = 0; frame < 16; ++frame) {
        auto profileFrame = profiler_.beginFrame();
        // Exercise both translation and rotation through the viewport's actual
        // camera update path. An explicit user reset must remain independent.
        auto& camera = properties["camera"];
        if (frame < 8) {
            camera["eye"][0] = camera["eye"][0].get<float>() + 0.002f;
        }
        camera["center"][0] = camera["center"][0].get<float>() + 0.002f;
        if (frame == 8) {
            ++resetSerial;
            renderGraph_.setNodeRuntimeProperty(dlssId, "resetSerial", resetSerial);
        }
        const auto before = nodeProperties();
        applyViewportCameraProperties(properties, "Smoke DLSS camera motion");
        const auto* updatedDlss = renderGraph_.findNode(dlssId);
        if (!expect(updatedDlss->runtimeProperties.value("resetSerial", 0u) == resetSerial,
                "Camera motion preserves the DLSS reset counter") ||
            !expect(nodeProperties() == before, "Viewport motion never mutates pass properties") ||
            !expect(!renderGraph_.dirty(), "Camera motion does not rebuild the graph")) {
            return false;
        }
        if (!renderDlssFrame() || !expect(viewportPreviewValid_, "Moving DLSS frame renders")) { return false; }
        if (rasterCamera) {
            const auto* metadata = graphExecutor_->outputResource("VBuffer.rasterInfo");
            if (!expect(metadata != nullptr && metadata->buffer != nullptr, "Raster metadata is available")) { return false; }
            const void* mapped = metadata->buffer->map();
            if (!expect(mapped != nullptr, "Read the camera used for rasterization")) { return false; }
            render::VisibilityBufferFrameInfo info;
            std::memcpy(&info, mapped, sizeof(info));
            metadata->buffer->unmap();
            for (uint32_t axis = 0; axis < 3; ++axis) {
                if (!expect(std::abs(info.eye[axis] - camera["eye"][axis].get<float>()) < 0.00001f &&
                        std::abs(info.center[axis] - camera["center"][axis].get<float>()) < 0.00001f,
                        "Viewport translation and rotation reach the rendered raster camera")) { return false; }
            }
        }
    }
    if (!expect(historyResources_.reprojectionInvalidationRevision() == reprojectionRevision,
            "Camera movement preserves reprojection history") ||
        !expect(historyResources_.invalidationRevision() > historyRevision,
            "Camera movement still invalidates non-reprojected accumulation")) { return false; }
    spdlog::info("[Smoke DLSS Camera] Passed 16 moving {} frames, shared view and explicit reset",
        rasterCamera ? "SR" : "RR");
    return true;
}

bool EditorApplication::runVisibilityPreviewSmokeTest()
{
    const auto expect = [](bool condition, const char* message) {
        if (!condition) { spdlog::error("[Smoke VBuffer Preview] {}", message); }
        return condition;
    };
    const auto renderFrames = [&](uint32_t count) {
        for (uint32_t i = 0; i < count; ++i) {
            auto profileFrame = profiler_.beginFrame();
            if (!waitForFrameSlotBeforeInput()) { return false; }
            const render::vulkan::StreamlineFrameScope streamlineFrame;
            if (!renderFrame() || !expect(viewportPreviewValid_, "Preview renders after switching")) { return false; }
        }
        return true;
    };
    auto* node = renderGraph_.findNode("VBuffer");
    if (!expect(node != nullptr, "Graph has a VBuffer pass")) { return false; }
    const uint32_t nodeId = node->id;
    const std::string presentation = renderGraph_.presentationOutputName();
    const size_t edgeCount = renderGraph_.edges().size();
    viewportView_.setTemporalJitter(true);
    if (!renderFrames(8)) { return false; }
    const auto camera = viewportCameraProperties();
    const auto select = [&](const char* mode) {
        renderGraph_.setNodeRuntimeProperty(nodeId, "visualization", mode);
        pendingVisibilityPreviewNodeId_ = nodeId;
        // Switching away from DLSS also changes the viewport toolbar height;
        // wait through the normal resize debounce before comparing images.
        return renderFrames(8);
    };
    const auto readPixels = [&](std::vector<uint32_t>& pixels, const char* mode) {
        if (!frameSubmissions_.wait() || !graphExecutor_->waitForSubmittedWork()) { return false; }
        auto* output = graphExecutor_->outputResource(activePreviewOutput_);
        if (!expect(output != nullptr && output->desc.format == render::Format::Rgba8Unorm,
                "Diagnostic is an RGBA8 output")) { return false; }
        pixels.resize(size_t(output->desc.width) * output->desc.height);
        std::unique_ptr<render::Buffer> readback;
        render::RenderFrameContext frame;
        render::QueueSubmissionTracker tracker;
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        if (!device_->createBuffer({.size = pixels.size() * sizeof(uint32_t),
                .usage = render::BufferUsageBits::TransferDestination, .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }) ||
            !device_->createCommandPool(*graphicsQueue_).transform([&](auto rhiValue) { pool = std::move(rhiValue); }) || !pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }) ||
            !tracker.initialize(*device_, *graphicsQueue_) || !frame.begin(0) || !commands->begin(&frame) ||
            !graphExecutor_->transitionOutput(*commands, activePreviewOutput_, render::ResourceState::TransferSource)) { return false; }
        commands->copyTextureToBuffer({.texture = output->texture, .buffer = readback.get(),
            .width = output->desc.width, .height = output->desc.height});
        if (!graphExecutor_->transitionOutput(*commands, activePreviewOutput_, render::ResourceState::ShaderRead) ||
            !commands->end()) { return false; }
        render::CommandBuffer* buffers[] = {commands.get()};
        if (!tracker.submit({.commandBuffers = buffers, .commandBufferCount = 1}, frame) || !frame.wait()) { return false; }
        readback->invalidate();
        const void* mapped = readback->map();
        if (!mapped) { return false; }
        std::memcpy(pixels.data(), mapped, pixels.size() * sizeof(uint32_t));
        readback->unmap();
        if (const char* directory = std::getenv("METALLIC_SMOKE_TEST_OUTPUT_DIR")) {
            std::filesystem::create_directories(directory);
            std::ofstream file(std::filesystem::path(directory) / (std::string(mode) + ".ppm"), std::ios::binary);
            file << "P6\n" << output->desc.width << ' ' << output->desc.height << "\n255\n";
            for (uint32_t pixel : pixels) { file.write(reinterpret_cast<const char*>(&pixel), 3); }
        }
        return true;
    };
    std::vector<std::vector<uint32_t>> serialImages;
    for (bool async : {false, true}) {
        renderGraph_.setNodeRuntimeProperty(nodeId, "asyncSoftwareRaster", async);
        size_t modeIndex = 0;
        for (const char* mode : {"coverage", "meshlet", "triangle", "depth"}) {
            if (!select(mode) || !expect(activePreviewOutput_ == "VBuffer.color", "Visualization selects its diagnostic output")) { return false; }
            std::vector<uint32_t> pixels;
            if (!expect(readPixels(pixels, mode), "Read diagnostic pixels")) { return false; }
            if (!expect(viewportView_.temporalJitter() && viewportView_.constants(0, 128, 128, 128, 128).frame[2] == 0,
                    "Raw preview suppresses sampling jitter without changing the user's setting")) { return false; }
            if (!renderFrames(1)) { return false; }
            std::vector<uint32_t> nextPixels;
            if (!readPixels(nextPixels, mode) || !expect(nextPixels == pixels, "Static raw preview is stable across temporal samples")) { return false; }
            if (modeIndex == 1) {
                auto rotated = camera;
                rotated["camera"]["center"][0] = rotated["camera"]["center"][0].get<float>() + .15f;
                applyViewportCameraProperties(rotated, "Smoke unjittered debug rotation");
                if (!renderFrames(2) || !readPixels(nextPixels, mode) ||
                    !expect(nextPixels != pixels, "Camera rotation reaches the debug image")) { return false; }
                const auto rotatedPixels = nextPixels;
                if (!renderFrames(1) || !readPixels(nextPixels, mode) ||
                    !expect(nextPixels == rotatedPixels, "Rotated debug view remains stable across samples")) { return false; }
                applyViewportCameraProperties(camera, "Restore debug camera");
                if (!renderFrames(2)) { return false; }
            }
            if (modeIndex == 0) {
                const size_t covered = std::count(pixels.begin(), pixels.end(), 0xffffffffu);
                if (!expect(covered > 50, "Coverage displays white geometry")) { return false; }
            } else if (!expect(pixels != serialImages[0], "ID/depth modes differ from coverage")) { return false; }
            if (async) {
                if (!expect(pixels == serialImages[modeIndex], "Serial and async diagnostics match")) { return false; }
            } else { serialImages.push_back(std::move(pixels)); }
            ++modeIndex;
        }
        if (!select("none") || !expect(activePreviewOutput_ == presentation, "Off restores the presentation output")) { return false; }
        if (!expect(viewportView_.constants(0, 128, 128, 128, 128).frame[2] == 1,
                "Presentation restores temporal sampling")) { return false; }
    }
    setActivePreviewOutput("Deferred.color");
    if (!select("coverage") || !select("none") ||
        !expect(activePreviewOutput_ == "Deferred.color", "Off restores a custom intermediate preview")) { return false; }
    if (!select("meshlet")) { return false; }
    setActivePreviewOutput(presentation);
    if (!select("none") || !expect(activePreviewOutput_ == presentation, "Off preserves an explicit output selection") ||
        !expect(viewportCameraProperties() == camera && renderGraph_.edges().size() == edgeCount &&
            renderGraph_.presentationOutputName() == presentation, "Preview changes preserve camera and graph wiring")) { return false; }
    spdlog::info("[Smoke VBuffer Preview] Passed four diagnostic images, serial/async equivalence, Off restore and manual output selection");
    return true;
}

bool EditorApplication::runSceneSwitchSmokeTest()
{
    if (std::getenv("METALLIC_SMOKE_TEST_ZORAH_FULL_SWITCH")) {
        const auto render = [&]() {
            auto profileFrame = profiler_.beginFrame();
            const render::vulkan::StreamlineFrameScope streamlineFrame;
            return waitForFrameSlotBeforeInput() && renderFrame() && viewportPreviewValid_;
        };
        for (uint32_t cycle = 0; cycle < 2; ++cycle) {
            loadBuiltInSample(render::kDefaultGPUDrivenSampleId);
            for (uint32_t frame = 0; frame < 16; ++frame) {
                if (!render()) { return false; }
            }
            loadBuiltInSample(render::kGPUDrivenZorahFullSampleId);
            auto environment = renderWorld_.environment();
            environment.visible = false; // A sky image must not satisfy geometry coverage.
            renderWorld_.setEnvironment(std::move(environment));
            uint32_t readyFrame = 0;
            for (uint32_t frame = 1; frame <= 1200; ++frame) {
                if (!render()) { return false; }
                const auto* streamer = subsystemHost_.get<render::StreamerSubsystem>();
                const auto readiness = streamer->sceneReadiness();
                if (!readyFrame && readiness.ready) {
                    readyFrame = frame;
                    spdlog::info("[Smoke Full Switch] cycle={} readyFrame={} pages={}/{}",
                        cycle, frame, readiness.completedPages, readiness.requiredPages);
                    device_->logMemoryBudget("Full editor switch ready");
                }
                if (readyFrame && frame >= readyFrame + 60) { break; }
            }
            if (!readyFrame) {
                spdlog::error("[Smoke Full Switch] Full root pages did not become ready");
                return false;
            }
            const auto* source = subsystemHost_.get<render::GPUSceneSubsystem>()->sourceOverride();
            auto* streamer = subsystemHost_.get<render::StreamerSubsystem>();
            std::shared_ptr<render::SceneResourceSnapshot> materials;
            std::string textureLog;
            if (!streamer->manager().acquire(*device_, *graphicsQueue_, renderGraph_.findNode("Deferred")->properties,
                source, render::SceneResourceFeatureBits::Materials, materials, textureLog)) { return false; }
            const auto textures = materials->pathTraceResources->textureStats();
            spdlog::info("[Smoke Full Switch] Texture streaming cycle={} resident={} pending={} retired={} upgrades={} feedback={}",
                cycle,textures.residentAllocationBytes,textures.pendingAllocationBytes,textures.retiredAllocationBytes,
                textures.upgrades,textures.feedbackFrames);
            if (!textures.streamingEnabled || !textures.upgrades || !textures.feedbackFrames ||
                textures.peakLiveAllocationBytes > textures.budgetBytes) { return false; }
        }
        // Read the actual DLSS presentation output with the environment hidden.
        // Stream readiness alone cannot prove that the deferred decoder shaded it.
        if (!frameSubmissions_.wait() || !graphExecutor_->waitForSubmittedWork()) { return false; }
        const auto* source = subsystemHost_.get<render::GPUSceneSubsystem>()->sourceOverride();
        if (!source || source->renderNodes().size() != 43068) {
            spdlog::error("[Smoke Full Switch] Unexpected stream metadata: nodes={}", source ? source->renderNodes().size() : 0);
            return false;
        }
        auto* output = graphExecutor_->outputResource(activePreviewOutput_);
        if (!output || (output->desc.format != render::Format::Rgba8Unorm && output->desc.format != render::Format::Rgba16Sfloat)) {
            spdlog::error("[Smoke Full Switch] Unsupported readback output '{}' format={}", activePreviewOutput_, output ? int(output->desc.format) : -1);
            return false;
        }
        const bool hdr = output->desc.format == render::Format::Rgba16Sfloat;
        const size_t pixelCount = size_t(output->desc.width) * output->desc.height;
        std::unique_ptr<render::Buffer> readback;
        render::RenderFrameContext frame;
        render::QueueSubmissionTracker tracker;
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        if (!device_->createBuffer({.size = pixelCount * (hdr ? 8u : 4u),
                .usage = render::BufferUsageBits::TransferDestination, .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }) ||
            !device_->createCommandPool(*graphicsQueue_).transform([&](auto rhiValue) { pool = std::move(rhiValue); }) || !pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }) ||
            !tracker.initialize(*device_, *graphicsQueue_) || !frame.begin(0) || !commands->begin(&frame) ||
            !graphExecutor_->transitionOutput(*commands, activePreviewOutput_, render::ResourceState::TransferSource)) { return false; }
        commands->copyTextureToBuffer({.texture = output->texture, .buffer = readback.get(),
            .width = output->desc.width, .height = output->desc.height});
        if (!graphExecutor_->transitionOutput(*commands, activePreviewOutput_, render::ResourceState::ShaderRead) ||
            !commands->end()) { return false; }
        render::CommandBuffer* buffers[] = {commands.get()};
        if (!tracker.submit({.commandBuffers = buffers, .commandBufferCount = 1}, frame) || !frame.wait()) { return false; }
        readback->invalidate();
        const auto* pixels = static_cast<const uint8_t*>(readback->map());
        if (!pixels) { return false; }
        size_t shadedPixels = 0;
        for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
            bool shaded = false;
            for (size_t component = 0; component < 3; ++component) {
                if (hdr) {
                    uint16_t half;
                    std::memcpy(&half, pixels + (pixel * 4 + component) * 2, sizeof(half));
                    // Positive finite half values above 0.03125; alpha never counts.
                    shaded |= half > 0x2800u && half < 0x7c00u;
                } else {
                    shaded |= pixels[pixel * 4 + component] > 8u;
                }
            }
            shadedPixels += shaded;
        }
        readback->unmap();
        spdlog::info("[Smoke Full Switch] DLSS shaded geometry pixels={}/{} (environment hidden)", shadedPixels, pixelCount);
        if (shadedPixels < 10000) { return false; }
        // Release executable references before the retry test rebuilds the graph.
        commands.reset();
        if (!frame.reset()) { return false; }
        // A failed budget admission must pause automatic retries. Restoring the
        // budget alone must not compile again until the user requests a retry.
        const auto policy = device_->memoryBudget().policy;
        auto deniedPolicy = policy;
        deniedPolicy.graphReserveBytes = UINT64_MAX;
        device_->setMemoryBudgetPolicy(deniedPolicy);
        renderGraph_.markDirty();
        const bool denied = !updateViewportPreview(1404, 674) && viewportCompileFailed_ && !renderGraph_.dirty();
        device_->setMemoryBudgetPolicy(policy);
        if (!denied || updateViewportPreview(1404, 674) || !viewportCompileFailed_) {
            spdlog::error("[Smoke Full Switch] Failed compile was retried automatically");
            return false;
        }
        renderGraph_.markDirty();
        if (!updateViewportPreview(1404, 674) || viewportCompileFailed_ || !render()) {
            spdlog::error("[Smoke Full Switch] Explicit retry failed");
            return false;
        }
        spdlog::info("[Smoke Full Switch] Passed two MiniZorah-to-Full DLSS switches and failed-compile retry gate");
        return true;
    }
    if (std::getenv("METALLIC_SMOKE_TEST_MINIZORAH_SWITCH")) {
        const char* cycleCount = std::getenv("METALLIC_SMOKE_TEST_SWITCH_CYCLES");
        const char* frameCount = std::getenv("METALLIC_SMOKE_TEST_ROAM_FRAMES");
        const uint32_t cycles = cycleCount ? uint32_t(std::clamp(std::atoi(cycleCount), 1, 32)) : 4;
        const uint32_t frames = frameCount ? uint32_t(std::clamp(std::atoi(frameCount), 16, 2400)) : 1200;
        for (uint32_t cycle = 0; cycle < cycles; ++cycle) {
            if (cycle != 0) {
                loadBuiltInSample("realtime-lighting");
                if (!waitForPendingSceneLoad(30000)) { return false; }
            }
            for (uint32_t frame = 0; frame < 60; ++frame) {
                auto profileFrame = profiler_.beginFrame();
                const render::vulkan::StreamlineFrameScope streamlineFrame;
                if (!waitForFrameSlotBeforeInput() || !renderFrame() || !viewportPreviewValid_) { return false; }
            }
            loadBuiltInSample("gpu-driven-minizorah-vbuffer");
            if (std::getenv("METALLIC_TEST_CLAS_OFF")) {
                renderGraph_.findNode("GPUDriven")->properties["enableClas"] = false;
            }
            if (std::getenv("METALLIC_TEST_CLAS_LEGACY")) {
                renderGraph_.findNode("GPUDriven")->properties["compactClas"] = false;
            }
            const auto original = viewportCameraProperties();
            for (uint32_t frame = 0; frame < frames; ++frame) {
                auto profileFrame = profiler_.beginFrame();
                auto camera = original;
                auto& position = camera["camera"];
                const float angle = frame < 180 ? 0.f : std::sin(float(frame - 180) * .021f) * 2.7f;
                const float x = original["camera"]["center"][0].get<float>() - original["camera"]["eye"][0].get<float>();
                const float z = original["camera"]["center"][2].get<float>() - original["camera"]["eye"][2].get<float>();
                position["center"][0] = position["eye"][0].get<float>() + std::cos(angle) * x + std::sin(angle) * z;
                position["center"][2] = position["eye"][2].get<float>() - std::sin(angle) * x + std::cos(angle) * z;
                applyViewportCameraProperties(camera, "Smoke MiniZorah roam");
                const render::vulkan::StreamlineFrameScope streamlineFrame;
                if (!waitForFrameSlotBeforeInput() || !renderFrame() || !viewportPreviewValid_) { return false; }
            }
            if (!frameSubmissions_.wait() || !graphExecutor_->waitForSubmittedWork()) { return false; }
            spdlog::info("[Smoke MiniZorah Switch] Cycle {} passed realtime -> MiniZorah and {} frames", cycle, frames);
        }
        return true;
    }
    const auto expect = [](bool condition, const std::string& message) {
        if (!condition) { spdlog::error("[Smoke Scene Switch] {}", message); }
        return condition;
    };
    const auto renderFrames = [&]() {
        for (int frame = 0; frame < 3; ++frame) {
            auto profileFrame = profiler_.beginFrame();
            if (!waitForFrameSlotBeforeInput() || !renderFrame() ||
                !expect(viewportPreviewValid_, "Switched scene renders")) { return false; }
        }
        return true;
    };
    if (!renderFrames()) { return false; }
    const auto originalPath = scene_.sourcePath();
    auto* deferred = renderGraph_.findNode("Deferred");
    if (!expect(deferred != nullptr, "Comparison has a Deferred pass")) { return false; }
    const uint32_t deferredId = deferred->id;
    const std::filesystem::path meetMat = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/meet_mat.glb";
    for (const auto& target : {meetMat, meetMat, originalPath}) {
        const uint64_t previousIdentity = scene_.resourceIdentity();
        // Use the same asynchronous request, resource preparation and commit path
        // as the file picker. Never rewrite the graph directly in this test.
        SDL_strlcpy(sceneFilePath_, target.string().c_str(), sizeof(sceneFilePath_));
        loadScene();
        if (!expect(waitForPendingSceneLoad(30000), "Scene load completes: " + sceneStatus_) ||
            !expect(scene_.resourceIdentity() != previousIdentity &&
                std::filesystem::equivalent(scene_.sourcePath(), target), "New document becomes the runtime scene")) {
            return false;
        }
        for (const char* name : {"Reference", "VBuffer", "Deferred"}) {
            const auto* node = renderGraph_.findNode(name);
            if (!expect(node != nullptr, std::string("Scene consumer exists: ") + name)) { return false; }
            auto properties = node->properties;
            properties.merge_patch(node->runtimeProperties);
            std::filesystem::path actual = properties.value("path", "");
            if (actual.is_relative()) { actual = std::filesystem::path(PROJECT_SOURCE_DIR) / actual; }
            if (!expect(!actual.empty() && std::filesystem::equivalent(actual, target),
                    std::string(name) + " retained scene path '" + actual.string() + "' instead of '" + target.string() + "'")) {
                return false;
            }
        }
        if (!renderFrames()) { return false; }
    }
    for (bool classified : {false, true}) {
        renderGraph_.setNodeRuntimeProperty(deferredId, "materialBinning", classified);
        if (!renderFrames()) { return false; }
    }
    spdlog::info("[Smoke Scene Switch] Passed meet_mat switch, same-path reload, return to original scene and classification toggle");
    return true;
}

bool EditorApplication::runSliderDebugSmokeTest()
{
    const auto expect = [](bool condition, const char* message) {
        if (!condition) { spdlog::error("[Smoke Slider] {}", message); }
        return condition;
    };
    if (!renderFrame() || !expect(viewportPreviewValid_, "Comparison viewport renders")) { return false; }
    auto* slider = viewportSliderDebugNode();
    if (!expect(slider != nullptr, "Find comparison through exposure and presentation")) { return false; }
    const uint32_t sliderId = slider->id;
    const uint64_t historyRevision = historyResources_.invalidationRevision();
    const auto split = [&] { return renderGraph_.findNode(sliderId)->runtimeProperties.value("splitPosition", 0.5f); };
    const bool nrComparison = slider->type == "DlssNrPass";
    std::string rawOutput;
    for (const auto& edge : renderGraph_.edges()) {
        if (edge.dstPass == slider->name) { rawOutput = edge.srcPass + "." + edge.srcField; break; }
    }
    if (!expect(!rawOutput.empty(), "Comparison has an input preview")) { return false; }
    if (nrComparison) { setSliderDebugProperty(sliderId, "sliderDebug", true); }
    // Drive real ImGui input transitions through the overlay without depending on
    // the desktop's saved docking layout or moving the user's physical cursor.
    auto overlayFrame = [&](float x, float y, bool down, bool alt = false) {
        auto& io = ImGui::GetIO();
        io.AddMousePosEvent(x, y);
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, down);
        io.AddKeyEvent(ImGuiMod_Alt, alt);
        ImGui::NewFrame();
        ImGui::Begin("Slider interaction test");
        viewportInteractionEnabled_ = true;
        viewportHovered_ = true;
        const bool captured = drawSliderDebugOverlay(ImVec2(100, 100), ImVec2(500, 400));
        ImGui::End();
        ImGui::Render();
        ImGui::UpdatePlatformWindows();
        return captured;
    };
    overlayFrame(300, 250, false);
    if (!expect(overlayFrame(300, 250, true), "Grab vertical divider") ||
        !expect(overlayFrame(400, 250, true) && std::abs(split() - 0.75f) < 0.001f, "Drag without moving the camera") ||
        !expect(overlayFrame(650, 250, true) && split() == 1.0f, "Clamp drag outside image") ||
        !expect(overlayFrame(650, 250, false) && sliderDragNodeId_ == 0, "Release drag without selecting geometry")) {
        return false;
    }
    setSliderDebugProperty(sliderId, "splitPosition", 0.5f);
    setSliderDebugProperty(sliderId, "orientation", "horizontal");
    setSliderDebugProperty(sliderId, "swapSides", true);
    overlayFrame(300, 250, false);
    if (!expect(overlayFrame(300, 250, true), "Grab horizontal divider") ||
        !expect(overlayFrame(300, 175, true) && std::abs(split() - 0.25f) < 0.001f, "Drag horizontal divider")) { return false; }
    overlayFrame(300, 175, false);
    overlayFrame(300, 175, false, true);
    if (!expect(!overlayFrame(300, 175, true, true) && sliderDragNodeId_ == 0, "Alt-orbit bypasses divider")) { return false; }
    overlayFrame(300, 175, false, false);
    if (!expect(historyResources_.invalidationRevision() == historyRevision && !renderGraph_.dirty(),
            "Slider controls preserve accumulation and compiled graph")) { return false; }

    if (nrComparison) {
        if (!expect(renderFrame(), "NR comparison renders after dragging")) { return false; }
        setSliderDebugProperty(sliderId, "sliderDebug", false);
        if (!expect(!overlayFrame(300, 175, true) && sliderDragNodeId_ == 0,
                "Disabled NR comparison does not capture the mouse") ||
            !expect(historyResources_.invalidationRevision() == historyRevision && !renderGraph_.dirty(),
                "NR comparison toggles preserve history")) { return false; }
        overlayFrame(300, 175, false);
        activePreviewOutput_ = rawOutput;
        if (!expect(viewportSliderDebugNode() == nullptr, "NR input preview has no comparison controls")) { return false; }
        spdlog::info("[Smoke Slider] Passed DLSS-NR toggle, GPU viewport, drag, axes, camera gestures and history retention");
        return true;
    }

    auto cameraProperties = viewportCameraProperties();
    cameraProperties["camera"]["eye"][0] = cameraProperties["camera"]["eye"][0].get<float>() + 0.5f;
    applyViewportCameraProperties(cameraProperties, "Smoke camera");
    if (!expect(viewportView_.cameraProperties() == cameraProperties["camera"],
            "Comparison uses the shared viewport camera")) { return false; }
    if (!expect(historyResources_.invalidationRevision() > historyRevision, "Camera movement resets accumulation")) { return false; }
    activePreviewOutput_ = rawOutput;
    if (!expect(viewportSliderDebugNode() == nullptr && !overlayFrame(300, 175, true),
            "Raw producer preview has no comparison interaction")) { return false; }
    overlayFrame(300, 175, false);
    spdlog::info("[Smoke Slider] Passed GPU viewport, drag, endpoints, axes, camera gestures, shared view and history retention");
    return true;
}

bool EditorApplication::runMultiViewportSmokeTest()
{
    auto expect = [](bool condition, const char* message) {
        if (!condition) {
            spdlog::error("[Smoke Viewports] {} (SDL: {})", message, SDL_GetError());
        }
        return condition;
    };
    auto graphViewport = []() -> ImGuiViewport* {
        const ImGuiWindow* graph = ImGui::FindWindowByName("Render Graph Editor");
        return graph != nullptr && graph->ViewportOwned ? graph->Viewport : nullptr;
    };
    auto graphWindow = [&]() -> SDL_Window* {
        const ImGuiViewport* viewport = graphViewport();
        return viewport != nullptr
            ? SDL_GetWindowFromID(static_cast<SDL_WindowID>(
                reinterpret_cast<uintptr_t>(viewport->PlatformHandle)))
            : nullptr;
    };

    const bool testFinalBlit = std::getenv("METALLIC_SMOKE_TEST_FINAL_BLIT") != nullptr;
    std::string sourceOutput = activePreviewOutput_;
    std::string finalInput;
    uint32_t finalId = 0;
    uint32_t finalEdgeId = 0;
    if (testFinalBlit) {
        // Samples already contain FinalBlit. Replace it to exercise adding a
        // presentation node, while retaining the original scene color source.
        if (const render::RenderGraphNode* existing = renderGraph_.findNode("FinalBlit");
            existing != nullptr && existing->type == "FinalBlitPass") {
            sourceOutput.clear();
            for (const render::RenderGraphEdge& edge : renderGraph_.edges()) {
                if (edge.dstPass == existing->name && edge.dstField == "source") {
                    sourceOutput = render::makeRenderGraphFieldName(edge.srcPass, edge.srcField);
                    break;
                }
            }
            if (!expect(!sourceOutput.empty(), "Sample FinalBlit has a connected color source")) {
                return false;
            }
            renderGraph_.removeNode(existing->id);
        }
        renderGraph_.clearOutputs();
        addRenderGraphNode("FinalBlitPass", ImVec2(-1.0f, -1.0f));
        finalId = static_cast<uint32_t>(selectedGraphNodeId_);
        const render::RenderGraphNode* node = renderGraph_.findNode(finalId);
        if (!expect(node != nullptr && activePreviewOutput_ == renderGraph_.presentationOutputName(),
                "Adding FinalBlit selects its automatic output")) {
            return false;
        }
        finalInput = render::makeRenderGraphFieldName(node->name, "source");
    }
    renderGraphEditorOpen_ = true;
    SDL_WindowID closedWindowId = 0;
    for (uint32_t index = 0; index < 16; ++index) {
        auto profileFrame = profiler_.beginFrame();
        if (testFinalBlit && index == 2) {
            const render::RenderGraphEdge* edge = renderGraph_.addEdge(sourceOutput, finalInput);
            if (!expect(edge != nullptr, "Connect scene output to FinalBlit")) {
                return false;
            }
            finalEdgeId = edge->id;
            viewportPreviewValid_ = false;
            if (!expect(activePreviewRenderGraphNode() != renderGraph_.findNode(finalId),
                    "Source runtime settings remain available through FinalBlit")) {
                return false;
            }
        } else if (testFinalBlit && index == 8) {
            renderGraph_.removeEdge(finalEdgeId);
            viewportPreviewValid_ = false;
        } else if (testFinalBlit && index == 12) {
            renderGraph_.renameNode(finalId, "FinalPresentation");
            viewportPreviewValid_ = false;
        }
        if (index == 4) {
            SDL_Window* graph = graphWindow();
            if (!expect(graph != nullptr && SDL_SetWindowSize(graph, 960, 640) &&
                    SDL_SyncWindow(graph), "Resize independent graph window")) {
                return false;
            }
        } else if (index == 6) {
            if (!expect(SDL_MinimizeWindow(window_) && SDL_SyncWindow(window_), "Minimize main window")) {
                return false;
            }
        } else if (index == 8) {
            if (!expect(SDL_RestoreWindow(window_) && SDL_SyncWindow(window_), "Restore main window")) {
                return false;
            }
        } else if (index == 10) {
            SDL_Window* graph = graphWindow();
            if (!expect(graph != nullptr, "Graph window exists before closing")) {
                return false;
            }
            closedWindowId = SDL_GetWindowID(graph);
            SDL_Event closeEvent{};
            closeEvent.type = SDL_EVENT_WINDOW_CLOSE_REQUESTED;
            closeEvent.window.windowID = closedWindowId;
            if (!expect(SDL_PushEvent(&closeEvent), "Send native graph close request")) {
                return false;
            }
        } else if (index == 12) {
            renderGraphEditorOpen_ = true;
        } else if (index == 14) {
            // Exercise resource retirement after the additional backend draws.
            destroyViewportTexture();
        }

        const render::vulkan::StreamlineFrameScope streamlineFrame(
            (SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) == 0 &&
            ImGui::GetPlatformIO().Viewports.Size <= 1);
        pollEvents();
        const uint64_t before = submittedFrameIndex_;
        if (!renderFrame() || !expect(running_ && submittedFrameIndex_ == before + 1,
                "Editor keeps submitting frames throughout the window lifecycle")) {
            return false;
        }

        if (testFinalBlit && !expect(viewportPreviewValid_ &&
                activePreviewOutput_ == renderGraph_.presentationOutputName() &&
                graphExecutor_->outputResource(activePreviewOutput_) != nullptr &&
                renderGraph_.outputs().empty(),
                "FinalBlit presents without manually marked outputs")) {
            return false;
        }

        if (index == 10 || index == 11) {
            if (!expect(!renderGraphEditorOpen_, "Native close only closes the graph editor")) {
                return false;
            }
            if (index == 11 && !expect(SDL_GetWindowFromID(closedWindowId) == nullptr,
                    "Closed native graph window is destroyed")) {
                return false;
            }
        } else if (index > 0) {
            SDL_Window* graph = graphWindow();
            const ImGuiViewport* viewport = graphViewport();
            if (!expect(graph != nullptr && graph != window_ && SDL_GetWindowParent(graph) == nullptr,
                    "Graph editor owns an independent SDL window") ||
                !expect((SDL_GetWindowFlags(graph) & SDL_WINDOW_BORDERLESS) == 0,
                    "Graph window uses native decorations") ||
                !expect(viewport->RendererUserData != nullptr && viewport->DrawData != nullptr &&
                    viewport->DrawData->TotalVtxCount > 0, "Independent viewport has renderer and UI draw data")) {
                return false;
            }
            if (index == 5) {
                int width = 0;
                int height = 0;
                if (!expect(SDL_GetWindowSize(graph, &width, &height) && width == 960 && height == 640,
                        "Graph window keeps the requested native size")) {
                    return false;
                }
            }
            if (index == 6 || index == 7) {
                if (!expect((SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) != 0 &&
                        (SDL_GetWindowFlags(graph) & SDL_WINDOW_MINIMIZED) == 0,
                        "Graph keeps rendering while the main window is minimized")) {
                    return false;
                }
            }
        }
        SDL_Delay(10);
    }
    if (testFinalBlit) {
        spdlog::info("[Smoke FinalBlit] Passed automatic output, connection, disconnection and rename");
    }
    spdlog::info("[Smoke Viewports] Passed open, resize, minimize, restore, close, reopen and resource retirement");
    return true;
}

} // namespace metallic
