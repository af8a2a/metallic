#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Editor/EditorApplication.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include "imgui.h"
#include "imgui_internal.h"
#include "ImGuizmo.h"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <chrono>

namespace metallic {

bool EditorApplication::runTerrainPlacementSmokeTest()
{
    const auto expect = [](bool condition, const char* message) {
        if (!condition) { spdlog::error("[Smoke Terrain Placement] {}", message); }
        return condition;
    };
    const char* fixture = std::getenv("METALLIC_TERRAIN_PLACEMENT_SMOKE_FIXTURE");
    if (!expect(fixture && *fixture, "Run the smoke test with its disposable fixture")) { return false; }
    const std::filesystem::path fixtureRoot(fixture);
    const auto base = fixtureRoot / "base";
    const auto source = base / "terrain.gltf";
    std::error_code error;
    if (!expect(std::filesystem::equivalent(scene_.sourcePath(), source, error) && !error &&
            scene_.terrainHeightField() && scene_.terrainPlacements().empty(),
            "Load the original test EXR height field with no placements")) { return false; }

    auto& input = *gizmoSmokeInput_;
    const auto render = [&] {
        auto profileFrame = profiler_.beginFrame();
        return waitForFrameSlotBeforeInput() && renderFrame();
    };
    const auto waitForViewport = [&] {
        for (int attempt = 0; attempt < 180; ++attempt) {
            if (!render()) { return false; }
            if (viewportPreviewValid_ && viewportInteractionEnabled_ && input.viewportImage) { return true; }
        }
        return expect(false, "The terrain viewport became interactive");
    };
    const auto clickGround = [&](float worldX, float worldZ) {
        auto camera = viewportView_.camera();
        camera.eye = {worldX, 28.0f, worldZ};
        camera.center = {worldX, 0.0f, worldZ};
        camera.up = {0.0f, 0.0f, -1.0f};
        if (!viewportView_.setCamera(camera) || !waitForViewport()) { return false; }
        const auto [left, top, right, bottom] = *input.viewportImage;
        input.pointer = std::array{(left + right) * 0.5f, (top + bottom) * 0.5f};
        if (!render() || !expect(viewportHovered_, "Inject the pointer into the rendered terrain viewport")) {
            return false;
        }
        input.mouseDown = true;
        const bool pressed = render();
        input.mouseDown = false;
        return pressed && render();
    };

    struct Sample {
        scene::TerrainPlacementKind kind;
        float3 position;
    };
    const std::array samples{
        Sample{scene::TerrainPlacementKind::Rock, float3(-3.0f, 1.0f, -2.0f)},
        Sample{scene::TerrainPlacementKind::Vegetation, float3(3.0f, 2.875f, 1.0f)},
    };
    const auto expectedNormal = normalize(float3(-0.25f, 1.0f, -0.125f));
    const auto checkPlacement = [&](uint64_t id, const Sample& sample, float heightDelta) {
        const auto& placements = scene_.terrainPlacements();
        const auto placement = std::ranges::find(placements, id, &scene::TerrainPlacement::id);
        // ImGui floors the pointer to a pixel. Check the target neighborhood,
        // then verify the exact height at the resulting world-space anchor.
        if (placement == placements.end() || !placement->enabled || placement->kind != sample.kind ||
            std::abs(placement->worldX - sample.position.x) > 0.08f ||
            std::abs(placement->worldZ - sample.position.z) > 0.08f) {
            return false;
        }
        const auto object = scene_.sceneGraph().object(scene_.terrainPlacementObject(id));
        if (!object || !object.hasComponent<scene::TransformComponent>() ||
            !object.getComponent<scene::VisibilityComponent>().worldVisible) { return false; }
        const auto& world = object.getComponent<scene::TransformComponent>().worldMatrix;
        const auto position = float3(world.a03, world.a13, world.a23);
        const auto up = normalize(float3(world.a01, world.a11, world.a21));
        const float height = sample.position.y + heightDelta +
            (placement->worldX - sample.position.x) * 0.25f + (placement->worldZ - sample.position.z) * 0.125f;
        return length(position - float3(placement->worldX, height, placement->worldZ)) < 0.001f &&
            length(up - expectedNormal) < 0.001f;
    };

    std::array<uint64_t, samples.size()> ids{};
    std::array<float2, samples.size()> anchors{};
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        terrainPlacementKind_ = sample.kind;
        if (!clickGround(sample.position.x, sample.position.z) ||
            !expect(scene_.terrainPlacements().size() == i + 1, "Create exactly one placement per click")) {
            return false;
        }
        const auto& placed = scene_.terrainPlacements().back();
        ids[i] = placed.id;
        anchors[i] = float2(placed.worldX, placed.worldZ);
        if (!expect(checkPlacement(ids[i], sample, 0.0f),
                "Place at the exact source height and align local +Y to the terrain normal")) { return false; }
    }
    terrainPlacementKind_.reset();
    undoTransform();
    const auto undone = scene_.sceneGraph().object(scene_.terrainPlacementObject(ids.back()));
    if (!expect(!scene_.terrainPlacements().back().enabled && undone &&
            !undone.getComponent<scene::VisibilityComponent>().worldVisible,
            "Undo hides the last placed object and disables its anchor")) { return false; }
    redoTransform();
    if (!expect(checkPlacement(ids.back(), samples.back(), 0.0f), "Redo restores the same terrain anchor")) {
        return false;
    }
    if (!clickGround(samples.front().position.x, samples.front().position.z) ||
        !expect(scene_.terrainPlacements().size() == samples.size(), "Selection mode does not place objects")) {
        return false;
    }
    saveScene();
    if (!expect(!scene_.dirty(), "Save the two world-space anchors")) { return false; }

    // Leave the saved document and mapped stream asset in place. The regular
    // scene reload detects the changed source and cooks its replacement stream.
    const auto raised = fixtureRoot / "raised";
    for (const char* name : {"height.exr", "terrain.mthf", "terrain_height.bin", "terrain.gltf"}) {
        error.clear();
        if (!std::filesystem::copy_file(raised / name, base / name,
                std::filesystem::copy_options::overwrite_existing, error)) {
            spdlog::error("[Smoke Terrain Placement] Cannot replace fixture source '{}': {}", name, error.message());
            return false;
        }
    }
    loadDroppedScene(source);
    if (!waitForPendingSceneLoad(30000) || !waitForViewport() ||
        !expect(scene_.terrainPlacements().size() == samples.size(), "Reload both saved terrain anchors")) {
        return false;
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto placement = std::ranges::find(scene_.terrainPlacements(), ids[i], &scene::TerrainPlacement::id);
        if (!expect(checkPlacement(ids[i], samples[i], 3.0f) &&
                placement->worldX == anchors[i].x && placement->worldZ == anchors[i].y,
                "Keep the same ID and world XZ after reimport and recompute height and alignment")) { return false; }
    }
    auto overview = viewportView_.camera();
    overview.eye = {10.0f, 16.0f, 15.0f};
    overview.center = {0.0f, 4.5f, 0.0f};
    overview.up = {0.0f, 1.0f, 0.0f};
    if (!viewportView_.setCamera(overview) || !waitForViewport() ||
        !expect(exportSmokeMainViewport(fixtureRoot / "reimported.ppm"), "Capture the placed terrain objects")) {
        return false;
    }
    spdlog::info("[Smoke Terrain Placement] Passed viewport Rock/Vegetation clicks, source height/normal, undo/redo, and same-path +3 height reimport");
    return true;
}

bool EditorApplication::runExitSmokeTest()
{
    const auto render = [&] {
        auto profileFrame = profiler_.beginFrame();
        return renderFrame();
    };
    const auto requestClose = [&] {
        SDL_Event event{};
        event.type = SDL_EVENT_WINDOW_CLOSE_REQUESTED;
        event.window.windowID = SDL_GetWindowID(window_);
        if (!SDL_PushEvent(&event)) { return false; }
        pollEvents();
        return running_ && pendingSceneAction_ == PendingSceneAction::Exit;
    };

    if (!render() || !viewportPreviewValid_) { return false; }
    scene_.setDirty(true);
    if (!requestClose() || !render()) { return false; }

    ImGuiWindow* modal = ImGui::FindWindowByName("Unsaved Scene Changes");
    if (!modal) { return false; }
    ImGui::ActivateItemByID(modal->GetID("Cancel"));
    if (!render() || !running_ || !scene_.dirty() || pendingSceneAction_ != PendingSceneAction::None) {
        spdlog::error("[Smoke Exit] Cancel did not keep the dirty scene open");
        return false;
    }

    if (!requestClose() || !render()) { return false; }
    const uint64_t framesBeforeDiscard = submittedFrameIndex_;
    ImGui::ActivateItemByID(modal->GetID("Discard"));
    if (!render() || running_ || !scene_.dirty() || submittedFrameIndex_ != framesBeforeDiscard ||
        frameSlots_[currentFrameSlot_].context.recording()) {
        spdlog::error("[Smoke Exit] Discard did not stop without reloading the scene or submitting a frame");
        return false;
    }
    spdlog::info("[Smoke Exit] Cancel kept the scene open and Discard stopped before rendering another frame");
    return true;
}

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
    // and Present boundaries as the capture button. ZorahFull needs more than
    // 90 frames to publish all root CLAS; a valid preview can still be loading.
    bool sceneReady = false;
    for (uint32_t frame = 1; frame <= 1200; ++frame) {
        if (!renderCaptureFrame()) { return false; }
        const auto* streamer = subsystemHost_.get<render::StreamerSubsystem>();
        const auto readiness = streamer ? streamer->sceneReadiness() : render::StreamSceneReadiness{};
        if (frame >= 90 && readiness.ready) {
            sceneReady = true;
            spdlog::info("[Smoke Nsight] Scene ready at frame {} pages={}/{}",
                frame, readiness.completedPages, readiness.requiredPages);
            break;
        }
    }
    if (!sceneReady) {
        spdlog::error("[Smoke Nsight] Streamed scene did not become ready before capture");
        return false;
    }
    int initialWidth = 0, initialHeight = 0;
    if (!SDL_GetWindowSize(window_, &initialWidth, &initialHeight)) { return false; }
    std::filesystem::path previousPath;
    for (uint32_t capture = 0; capture < 3; ++capture) {
        if (capture > 0) {
            // Exercise DLSS/resource recreation as well as steady-state capture.
            // Grow then restore the settled layout. Shrinking to a fixed window
            // size can leave a 30-pixel viewport with the user's saved dock sizes,
            // which tests unsupported DLSS dimensions instead of capture resize.
            if (!SDL_SetWindowSize(window_, initialWidth + (capture == 1 ? 160 : 0),
                    initialHeight + (capture == 1 ? 90 : 0))) {
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
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
        while ((nsightGraphicsCapture_.hasOutstandingCapture() || nsightGraphicsCapture_.replayTracePending()) &&
            std::chrono::steady_clock::now() < deadline) {
            if (!renderCaptureFrame()) { return false; }
            pollNsightGraphicsCapture();
        }
        std::error_code error;
        const auto& path = nsightGraphicsCapture_.capturePath();
        const auto bytes = std::filesystem::file_size(path, error);
        if (nsightGraphicsCapture_.state() != NsightGraphicsCaptureState::CaptureCompleted ||
            error || bytes == 0 || path == previousPath ||
            path.extension() != (nsightMode_ == render::profiling::NsightCaptureMode::GPUTrace
                ? ".ngfx-gputrace" : ".ngfx-capture")) {
            spdlog::error("[Smoke Nsight] Capture did not complete: {}", nsightGraphicsCapture_.statusText());
            return false;
        }
        previousPath = path;
        if (nsightMode_ == render::profiling::NsightCaptureMode::GraphicsCapture) {
            const auto traceBytes = std::filesystem::file_size(nsightGraphicsCapture_.replayTracePath(), error);
            if (nsightGraphicsCapture_.replayTracePending() || error || traceBytes == 0 ||
                !nsightGraphicsCapture_.replayTraceError().empty()) {
                spdlog::error("[Smoke Nsight] Replay GPU Trace failed: {}", nsightGraphicsCapture_.replayTraceError());
                return false;
            }
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
    auto* dlssNode = renderGraph_.findNode("DLSSRR");
    if (dlssNode == nullptr) { dlssNode = renderGraph_.findNode("DLSSSR"); }
    if (!expect(dlssNode != nullptr && graphExecutor_->renderView() == &viewportView_, "DLSS graph is bound to the viewport view")) {
        return false;
    }
    const bool rasterCamera = dlssNode->type == "StreamlineDLSSSRPass";
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
        if (!expect(output != nullptr && output->desc.format == render::Format::RGBA8Unorm,
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
            !tracker.initialize(*device_, *graphicsQueue_) || !frame.begin(0) || !commands->begin(frame.submissionContext()) ||
            !graphExecutor_->transitionOutput(*commands, activePreviewOutput_, render::ResourceState::TransferSource)) { return false; }
        if (auto commandResult = (readback.get())->slice().and_then([&](const auto& bufferSlice) { return commands->copyTextureToBuffer({.texture = output->texture, .buffer = bufferSlice,
            .width = output->desc.width, .height = output->desc.height}); }); !commandResult) { return false; }
        if (!graphExecutor_->transitionOutput(*commands, activePreviewOutput_, render::ResourceState::ShaderRead) ||
            !commands->end()) { return false; }
        render::CommandBuffer* buffers[] = {commands.get()};
        if (!tracker.submit({.commandBuffers = {buffers, 1}}, frame) || !frame.wait()) { return false; }
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
    if (std::getenv("METALLIC_SMOKE_TEST_LOOKDEV_PATHS")) {
        using render::LookDevRenderPath;
        if (!lookDevComparisonGraph_) { return false; }
        const auto renderFrames = [&]() {
            for (uint32_t frame = 0; frame < 8; ++frame) {
                auto profileFrame = profiler_.beginFrame();
                const render::vulkan::StreamlineFrameScope streamlineFrame;
                if (!waitForFrameSlotBeforeInput()) { return false; }
                pollEvents();
                if (!renderFrame()) { return false; }
            }
            return viewportPreviewValid_ && graphExecutor_ && graphExecutor_->compiled();
        };
        if (!renderFrames()) { return false; }
        const auto source = scene_.sourcePath();
        auto camera = viewportCameraProperties();
        camera["camera"]["eye"][0] = camera["camera"]["eye"][0].get<float>() + 0.05f;
        applyViewportCameraProperties(camera, "LookDev path switch camera");
        camera = viewportCameraProperties();
        const auto materialCount = scene_.materials().size();
        scene_.setDirty(true);
        for (const auto path : {LookDevRenderPath::PathTraceOnly, LookDevRenderPath::DeferredOnly,
                LookDevRenderPath::Comparison, LookDevRenderPath::DeferredOnly, LookDevRenderPath::PathTraceOnly}) {
            setLookDevRenderPath(path);
            if (!renderFrames() || viewportCameraProperties() != camera || scene_.sourcePath() != source ||
                !scene_.dirty() || scene_.materials().size() != materialCount || activePreviewOutput_ != "FinalBlit.color") {
                spdlog::error("[Smoke LookDev Paths] Scene, camera, edits or display lost");
                return false;
            }
            bool pt = false, deferred = false, vbuffer = false, slider = false;
            for (const auto& node : graphExecutor_->executionStats().nodes) {
                pt |= node.name == "Reference";
                deferred |= node.name == "Deferred";
                vbuffer |= node.name == "VBuffer";
                slider |= node.name == "Slider";
            }
            if (pt != (path != LookDevRenderPath::DeferredOnly) ||
                deferred != (path != LookDevRenderPath::PathTraceOnly) || vbuffer != deferred ||
                slider != (path == LookDevRenderPath::Comparison)) {
                spdlog::error("[Smoke LookDev Paths] Inactive pass executed");
                return false;
            }
        }
        scene_.setDirty(false);
        loadBuiltInSample("lookdev-vbuffer");
        if (!renderFrames() || renderGraph_.findNode("Deferred") || !renderGraph_.findNode("Reference")) {
            spdlog::error("[Smoke LookDev Paths] Render path preference lost on sample load");
            return false;
        }
        spdlog::info("[Smoke LookDev Paths] Passed live mode switches, active GPU passes, camera/unsaved scene retention and sample reload");
        return true;
    }
    const bool studioSwitch = std::getenv("METALLIC_SMOKE_TEST_STUDIO_SWITCH") != nullptr;
    if (std::getenv("METALLIC_SMOKE_TEST_PAINTER_SWITCH") || studioSwitch) {
        const char* label = studioSwitch ? "Studio" : "Painter";
        const std::vector<const char*> ids = studioSwitch
            ? std::vector<const char*>{"studio-white-overview", "studio-white-chart", "studio-white-M03_CoatedPaint-textured", "studio-white-overview"}
            : std::vector<const char*>{"painter-M01_NeutralDielectric-uniform", "painter-M03_CoatedPaint-textured",
                "painter-M08_MaskedCard-textured", "painter-H01_HDREmission-textured", "painter-M01_NeutralDielectric-uniform"};
        for (const char* id : ids) {
            // A complete studio preset must reset previously edited environment settings.
            if (studioSwitch) { environmentUserEdited_ = true; }
            loadBuiltInSample(id);
            render::RenderSampleLoadResult expected;
            std::string message;
            if (!render::loadBuiltInRenderSample(id, expected, message)) { return false; }
            bool ready = false;
            for (uint32_t frame = 0; frame < 180; ++frame) {
                auto profileFrame = profiler_.beginFrame();
                const render::vulkan::StreamlineFrameScope streamlineFrame;
                if (!waitForFrameSlotBeforeInput()) { return false; }
                pollEvents();
                if (!renderFrame()) { return false; }
                if (viewportPreviewValid_ && scene_.sourcePath() == std::filesystem::path(expected.desc.scenePath)) {
                    ready = true;
                    if (frame > 4) { break; }
                }
            }
            if (!ready || !scene_.documentWarning().empty()) {
                spdlog::error("[Smoke {} Switch] Failed {}: {}", label, id, scene_.documentWarning());
                return false;
            }
            if (studioSwitch && (renderWorld_.environment().path.filename() != "WhiteStudio02.hdr" ||
                !renderWorld_.environment().hasExplicitSourceColorSpace())) { return false; }
            spdlog::info("[Smoke {} Switch] Rendered {} with {} materials and {} textures", label, id,
                scene_.materials().size(), scene_.textures().size());
        }
        const auto source = scene_.sourcePath();
        scene_.setDirty(true);
        loadBuiltInSample(ids[1]);
        const bool protectedEdits = scene_.sourcePath() == source && pendingSceneAction_ == PendingSceneAction::LoadSample;
        scene_.setDirty(false);
        pendingSceneAction_ = PendingSceneAction::None;
        pendingSceneValue_.clear();
        if (!protectedEdits) { return false; }
        spdlog::info("[Smoke {} Switch] Passed full scene changes, return to baseline and unsaved-edit protection", label);
        return true;
    }
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
            if (!streamer->manager().acquire(*device_, *graphicsQueue_, renderGraph_.findNode("Deferred")->properties, source, render::SceneResourceFeatureBits::Materials, textureLog).transform([&](auto value) { materials = std::move(value); })) { return false; }
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
        if (!output || (output->desc.format != render::Format::RGBA8Unorm && output->desc.format != render::Format::RGBA16Sfloat)) {
            spdlog::error("[Smoke Full Switch] Unsupported readback output '{}' format={}", activePreviewOutput_, output ? int(output->desc.format) : -1);
            return false;
        }
        const bool hdr = output->desc.format == render::Format::RGBA16Sfloat;
        const size_t pixelCount = size_t(output->desc.width) * output->desc.height;
        std::unique_ptr<render::Buffer> readback;
        render::RenderFrameContext frame;
        render::QueueSubmissionTracker tracker;
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        if (!device_->createBuffer({.size = pixelCount * (hdr ? 8u : 4u),
                .usage = render::BufferUsageBits::TransferDestination, .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }) ||
            !device_->createCommandPool(*graphicsQueue_).transform([&](auto rhiValue) { pool = std::move(rhiValue); }) || !pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }) ||
            !tracker.initialize(*device_, *graphicsQueue_) || !frame.begin(0) || !commands->begin(frame.submissionContext()) ||
            !graphExecutor_->transitionOutput(*commands, activePreviewOutput_, render::ResourceState::TransferSource)) { return false; }
        if (auto commandResult = (readback.get())->slice().and_then([&](const auto& bufferSlice) { return commands->copyTextureToBuffer({.texture = output->texture, .buffer = bufferSlice,
            .width = output->desc.width, .height = output->desc.height}); }); !commandResult) { return false; }
        if (!graphExecutor_->transitionOutput(*commands, activePreviewOutput_, render::ResourceState::ShaderRead) ||
            !commands->end()) { return false; }
        render::CommandBuffer* buffers[] = {commands.get()};
        if (!tracker.submit({.commandBuffers = {buffers, 1}}, frame) || !frame.wait()) { return false; }
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
    const bool nrComparison = slider->type == "DLSSNRPass";
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

bool EditorApplication::runGizmoSmokeTest()
{
    const auto expect = [](bool condition, const char* message) {
        if (!condition) { spdlog::error("[Smoke Gizmo] {}", message); }
        return condition;
    };
    const auto render = [&] {
        auto profileFrame = profiler_.beginFrame();
        return waitForFrameSlotBeforeInput() && renderFrame() && viewportPreviewValid_;
    };
    auto& input = *gizmoSmokeInput_;
    const auto pressAlignView = [&](SDL_Keymod modifiers = SDL_KMOD_LCTRL | SDL_KMOD_LSHIFT) {
        const auto sendKey = [&](bool down) {
            SDL_Event event{};
            event.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
            event.key.windowID = SDL_GetWindowID(window_);
            event.key.scancode = SDL_SCANCODE_F;
            event.key.key = SDLK_F;
            event.key.mod = down ? modifiers : SDL_KMOD_NONE;
            event.key.down = down;
            if (!expect(SDL_PushEvent(&event), "Send alignment key through SDL")) { return false; }
            pollEvents();
            return running_ && render();
        };
        return sendKey(true) && sendKey(false);
    };
    const auto cameraPose = [](const render::ViewCamera& camera) {
        const float3 eye(camera.eye[0], camera.eye[1], camera.eye[2]);
        const float3 center(camera.center[0], camera.center[1], camera.center[2]);
        const float3 up(camera.up[0], camera.up[1], camera.up[2]);
        const float3 forward = normalize(center - eye);
        const float3 right = normalize(cross(forward, up));
        return float4x4(float4(right, 0.0f), float4(cross(right, forward), 0.0f),
            float4(-forward, 0.0f), float4(eye, 1.0f));
    };
    const auto boundsEdgesDrawn = [&](std::optional<size_t> edgeCount = std::nullopt) {
        return !input.boundsEdges.empty() && (!edgeCount || input.boundsEdges.size() == *edgeCount) &&
            std::ranges::all_of(input.boundsEdges, [](const auto& edge) {
                return std::ranges::all_of(edge, [](float coordinate) { return std::isfinite(coordinate); }) &&
                    (edge[0] != edge[2] || edge[1] != edge[3]);
            });
    };
    const auto boundsMatch = [&](const auto& expected) {
        return std::ranges::equal(input.boundsEdges, expected, [](const auto& actual, const auto& reference) {
            return std::ranges::equal(actual, reference, [](float a, float b) { return std::abs(a - b) < 0.05f; });
        });
    };
    const auto dragLinearGizmo = [&](GizmoOperation operation, bool checkBounds) {
        gizmoOperation_ = operation;
        resetTransformHistory();
        input.pointer.reset();
        if (!render() || !expect(input.center && input.drewGizmo, "Draw the rotate or scale gizmo")) { return false; }
        const auto center = *input.center;
        std::array<float, 2> start = center;
        std::array<float, 2> end{center[0] + 25.0f, center[1]};
        if (operation == GizmoOperation::Rotate) {
            // ImGuizmo's screen ring includes the projected extents of its axis rings.
            const float nominalRadius = 0.06f * input.gizmoExtent;
            bool hovered = false;
            for (const float offset : {0.0f, -6.0f, 6.0f, -12.0f, 12.0f}) {
                const float radius = nominalRadius + offset;
                start = {center[0] + radius, center[1]};
                input.pointer = start;
                if (!render()) { return false; }
                if (viewportGizmoCapturingMouse_ && ImGuizmo::IsOver(ImGuizmo::ROTATE_SCREEN)) {
                    const float diagonal = radius * 0.70710678f;
                    end = {center[0] + diagonal, center[1] + diagonal};
                    hovered = true;
                    break;
                }
            }
            if (!expect(hovered, "Hover the screen rotation ring")) { return false; }
        } else {
            input.pointer = start;
            if (!render() || !expect(viewportGizmoCapturingMouse_, "Hover the uniform scale handle")) { return false; }
        }
        input.mouseDown = true;
        if (!render() || !expect(gizmoWasUsing_, "Start the rotate or scale gesture")) { return false; }
        for (const float step : {0.5f, 1.0f}) {
            const auto previousBounds = input.boundsEdges;
            input.pointer = std::array{std::lerp(start[0], end[0], step), std::lerp(start[1], end[1], step)};
            if (!render() || !expect(gizmoWasUsing_ && (!checkBounds || (boundsEdgesDrawn() && !boundsMatch(previousBounds))),
                    "Rotated or scaled bounds follow the drag in the same frame")) { return false; }
        }
        input.mouseDown = false;
        return render() && expect(!gizmoWasUsing_ && transformCommands_.size() == 1 && scene_.dirty(),
            "Rotate or scale commits one undoable transform");
    };
    const auto treeFrame = [&](bool down, const std::optional<std::array<float, 2>>& pointer = std::nullopt) {
        auto& io = ImGui::GetIO();
        if (pointer) { io.AddMousePosEvent((*pointer)[0], (*pointer)[1]); }
        io.AddMouseButtonEvent(ImGuiMouseButton_Left, down);
        input.sceneTreeRow.reset();
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(40.0f, 40.0f), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(650.0f, 580.0f), ImGuiCond_Always);
        ImGui::SetNextWindowFocus();
        ImGui::Begin("Scene Graph interaction test");
        drawSceneGraphTab();
        ImGui::End();
        ImGui::Render();
    };
    const auto nodeDepth = [&](const scene::ConstSceneObject& object) {
        const auto& graph = scene_.sceneGraph();
        int depth = 1;
        for (auto parent = object.getComponent<scene::RelationshipComponent>().parent;
            parent != scene::kNullSceneEntity; parent = graph.object(parent).getComponent<scene::RelationshipComponent>().parent) {
            ++depth;
        }
        return depth;
    };
    const auto clickTreeAt = [&](const std::array<float, 2>& position) {
        treeFrame(false, position);
        treeFrame(true, position);
        treeFrame(false, position);
    };
    const auto treeArrow = [](const std::array<float, 2>& row) {
        return std::array{row[0] - 2.0f - ImGui::GetTreeNodeToLabelSpacing() * 0.5f, row[1]};
    };
    const auto selectTreeRow = [&](const scene::ConstSceneObject& object, int32_t renderNodeIndex) {
        const int expectedDepth = nodeDepth(object) + (renderNodeIndex == scene::kInvalidSceneIndex ? 0 : 1);
        input.sceneTreeProbe = object.entity();
        input.sceneTreePrimitiveProbe = renderNodeIndex;
        const auto selection = sceneSelection_;
        sceneSelection_ = {};
        treeFrame(false);
        sceneSelection_ = selection;
        treeFrame(false);
        if (!expect(input.sceneTreeRow && input.sceneTreeDepth == expectedDepth,
                "Scene Graph reveals the selection at its hierarchy depth")) { return false; }

        const auto row = *input.sceneTreeRow;
        sceneSelection_ = {};
        clickTreeAt(row);
        input.sceneTreeProbe = scene::kNullSceneEntity;
        input.sceneTreePrimitiveProbe = scene::kInvalidSceneIndex;
        return expect(selectedSceneObject().entity() == object.entity() && sceneSelection_.renderNodeIndex == renderNodeIndex,
            "Click the node or primitive in the Scene Graph");
    };
    const auto selectShowcaseDecal = [&](const scene::ConstSceneObject& group, uint32_t decalId) {
        input.sceneTreeProbe = group.entity();
        const auto selection = sceneSelection_;
        sceneSelection_ = {};
        treeFrame(false);
        sceneSelection_ = selection;
        treeFrame(false);
        if (!expect(input.sceneTreeRow && input.sceneTreeDepth == nodeDepth(group),
                "Find the Floor Decals parent row")) { return false; }

        const auto arrow = treeArrow(*input.sceneTreeRow);
        input.sceneTreeProbe = scene::kNullSceneEntity;
        input.sceneTreeDecalProbe = decalId;
        treeFrame(false, arrow);
        if (!expect(!input.sceneTreeRow, "Collapsed Floor Decals hides its child rows")) { return false; }
        treeFrame(true, arrow);
        treeFrame(false, arrow);

        for (const uint32_t id : {1u, 2u, 3u}) {
            input.sceneTreeDecalProbe = id;
            treeFrame(false);
            if (!expect(input.sceneTreeRow && input.sceneTreeDepth == nodeDepth(group) + 1,
                    "Expanding Floor Decals reveals each of its three child decals")) { return false; }
        }
        input.sceneTreeDecalProbe = decalId;
        treeFrame(false);
        const auto row = *input.sceneTreeRow;
        sceneSelection_ = {};
        clickTreeAt(row);
        input.sceneTreeDecalProbe.reset();
        const auto* selected = selectedDecal();
        return expect(selected && selected->id == decalId, "Click the Floor Medallion in the Scene Graph");
    };
    const auto sceneBrowserHasInput = [](ImGuiWindow* browser) {
        const auto& context = *ImGui::GetCurrentContext();
        return context.NavWindow && ImGui::IsWindowChildOf(context.NavWindow, browser, false, false) &&
            context.HoveredWindow && ImGui::IsWindowChildOf(context.HoveredWindow, browser, false, false);
    };
    const auto clickSceneBrowserWorldItem = [&](scene::WorldItemId item, ImGuiWindow* browser) {
        ImGui::FocusWindow(browser);
        // Expand the item's ancestry, then clear selection so the click must select the actual row.
        sceneSelection_ = SceneSelection{
            .type = SceneSelectionType::WorldItem,
            .sceneLifetimeRevision = scene_.sceneGraph().lifetimeRevision(),
            .worldItem = item,
        };
        sceneTreeSelection_ = scene::kNullSceneEntity;
        if (!render()) { return false; }
        ImGui::SetScrollY(browser, browser->ScrollMax.y);
        input.sceneTreeRow.reset();
        input.sceneTreeWorldItemProbe = item;
        if (!render() || !expect(input.sceneTreeRow.has_value(), "Find the Arcade spotlight row in the Scene Browser")) {
            return false;
        }
        input.pointer = *input.sceneTreeRow;
        input.sceneTreeWorldItemProbe.reset();
        sceneSelection_ = {};
        if (!render()) { return false; }
        input.mouseDown = true;
        if (!render()) { return false; }
        input.mouseDown = false;
        return render() && expect(selectedWorldItem() == item && sceneBrowserHasInput(browser),
            "Click Arcade spotlight while leaving the Scene Browser focused and hovered");
    };
    const auto checkShowcaseWorldItems = [&] {
        using scene::WorldItemId;
        using scene::WorldItemType;
        const auto& cameras = scene_.cameras();
        const auto camera = std::ranges::find(cameras, "Fog & Rect Lights", &scene::RenderCamera::name);
        if (!expect(camera != cameras.end(), "Find the Fog & Rect Lights showcase camera")) { return false; }
        const scene::ConstSceneObject group = scene_.objectForNode(camera->nodeIndex);
        const auto groupWorldMatrix = group.getComponent<scene::TransformComponent>().worldMatrix;

        std::vector<WorldItemId> children;
        const auto appendChildren = [&](WorldItemType type, const auto& items) {
            for (const auto& item : items) {
                const WorldItemId id{.type = type, .id = item.stableId};
                if (scene_.worldItemSceneGraphParent(id) == group.entity()) { children.push_back(id); }
            }
        };
        appendChildren(WorldItemType::LocalFogVolume, scene_.worldEnvironment().volumetricFog.localVolumes);
        appendChildren(WorldItemType::PunctualLight, scene_.lighting().lights);
        appendChildren(WorldItemType::RectLight, scene_.lighting().rectLights);
        const auto hasGroupedLight = [&](std::string_view type, bool hasProfile) {
            return std::ranges::any_of(scene_.lighting().lights, [&](const scene::PunctualLight& light) {
                const WorldItemId item{WorldItemType::PunctualLight, light.stableId};
                return light.properties.type == type && static_cast<bool>(light.iesProfile) == hasProfile &&
                    scene_.worldItemSceneGraphParent(item) == group.entity();
            });
        };
        if (!expect(children.size() == 6 &&
                std::ranges::count(children, WorldItemType::LocalFogVolume, &WorldItemId::type) == 1 &&
                std::ranges::count(children, WorldItemType::PunctualLight, &WorldItemId::type) == 3 &&
                std::ranges::count(children, WorldItemType::RectLight, &WorldItemId::type) == 2 &&
                hasGroupedLight("point", false) && hasGroupedLight("spot", false) && hasGroupedLight("spot", true),
                "The showcase groups fog, a point light, two spotlights including one with IES, and two rect lights")) { return false; }
        const auto captureItem = *std::ranges::find(children, WorldItemType::RectLight, &WorldItemId::type);
        const auto iesSpot = std::ranges::find_if(scene_.lighting().lights, [](const scene::PunctualLight& light) {
            return light.properties.type == "spot" && light.iesProfile;
        });
        const WorldItemId iesSpotItem{WorldItemType::PunctualLight, iesSpot->stableId};
        const int childDepth = nodeDepth(group) + 1;
        input.sceneTreeProbe = group.entity();
        sceneSelection_ = {};
        treeFrame(false);
        if (!expect(input.sceneTreeRow && input.sceneTreeDepth == childDepth - 1,
                "Find the Fog & Rect Lights row at its hierarchy depth")) { return false; }
        const auto arrow = treeArrow(*input.sceneTreeRow);
        input.sceneTreeProbe = scene::kNullSceneEntity;
        input.sceneTreeWorldItemProbe = children.front();
        treeFrame(false, arrow);
        if (!expect(!input.sceneTreeRow, "Collapsed Fog & Rect Lights hides its child rows")) { return false; }
        clickTreeAt(arrow);

        const auto positionMatches = [](const std::optional<float3>& actual, const float3& expected) {
            return actual && length(*actual - expected) < 0.000001f;
        };
        std::vector<float3> originalPositions;
        for (const auto item : children) {
            const auto position = worldItemPosition(item);
            if (!expect(position.has_value(), "Get the child world-space position")) { return false; }
            originalPositions.push_back(*position);
        }
        constexpr int kDragFrames = 4;
        for (size_t i = 0; i < children.size(); ++i) {
            const auto item = children[i];
            input.sceneTreeWorldItemProbe = item;
            treeFrame(false);
            if (!expect(input.sceneTreeRow && input.sceneTreeDepth == childDepth,
                    "Expanding Fog & Rect Lights reveals each world item")) { return false; }
            const auto row = *input.sceneTreeRow;
            clickTreeAt(row);
            input.sceneTreeWorldItemProbe.reset();
            if (!expect(selectedWorldItem() == item, "Click the individual fog volume or light row")) { return false; }

            resetTransformHistory();
            const auto position = originalPositions[i];
            applySceneCamera(scene::RenderCamera{
                .eye = position + float3(7.2f, 3.0f, 1.2f),
                .center = position,
                .yfov = 1.047197551196598,
                .znear = 0.05,
                .zfar = 100.0,
            });
            input.pointer.reset();
            if (!render() || !expect(input.center && input.drewGizmo && boundsEdgesDrawn(),
                    "Each world item has its own gizmo and visible bounds")) { return false; }
            const auto beforeBounds = input.boundsEdges;
            const auto cameraBefore = viewportView_.camera();
            const auto [centerX, centerY] = *input.center;
            input.pointer = std::array{centerX, centerY};
            if (!render() || !expect(viewportGizmoCapturingMouse_, "Hover the world-item translation handle")) {
                return false;
            }
            input.mouseDown = true;
            if (!render() || !expect(gizmoWasUsing_, "Start the world-item drag")) { return false; }
            auto moved = position;
            for (int frame = 1; frame <= kDragFrames; ++frame) {
                const auto previousBounds = input.boundsEdges;
                input.pointer = std::array{centerX + 2.0f * float(frame), centerY};
                if (!render()) { return false; }
                const auto current = worldItemPosition(item);
                if (!expect(gizmoWasUsing_ && current && !positionMatches(current, moved) &&
                        boundsEdgesDrawn() && !boundsMatch(previousBounds),
                        "World-item bounds follow the gizmo in the same frame")) { return false; }
                moved = *current;
            }
            input.mouseDown = false;
            if (!render() || !expect(!gizmoWasUsing_ && transformCommands_.size() == 1 && scene_.dirty() &&
                    viewportView_.camera() == cameraBefore && scene::matrixNearlyEqual(groupWorldMatrix,
                        group.getComponent<scene::TransformComponent>().worldMatrix),
                    "World-item drag commits once and leaves both cameras in place")) { return false; }
            for (size_t sibling = 0; sibling < children.size(); ++sibling) {
                if (sibling != i && !expect(positionMatches(worldItemPosition(children[sibling]), originalPositions[sibling]),
                        "World-item drag leaves sibling items in place")) { return false; }
            }
            const auto afterBounds = input.boundsEdges;
            undoTransform();
            if (!render() || !expect(positionMatches(worldItemPosition(item), position) && !scene_.dirty() &&
                    boundsMatch(beforeBounds), "Undo restores the world item and its bounds")) { return false; }
            redoTransform();
            if (!render() || !expect(positionMatches(worldItemPosition(item), moved) && scene_.dirty() && boundsMatch(afterBounds),
                    "Redo restores the moved world item and its bounds")) { return false; }
            if (const char* path = std::getenv("METALLIC_FOG_LIGHT_GIZMO_SMOKE_CAPTURE");
                item == captureItem && path && *path &&
                !expect(exportSmokeMainViewport(path), "Export the rect light bounds frame as a PPM image")) { return false; }
            undoTransform();
            const auto name = scene_.worldItemName(item);
            spdlog::info("[Smoke Gizmo] worldItem='{}' frames={} boundEdges={}", name, kDragFrames, afterBounds.size());
            if (name == "Arcade spotlight") {
                resetTransformHistory();
                auto* browser = ImGui::FindWindowByName("Scene Browser");
                if (!expect(browser != nullptr, "Find the Scene Browser") ||
                    !clickSceneBrowserWorldItem(item, browser)) { return false; }
                const auto before = *worldItemTransform(item);
                if (!pressAlignView(SDL_KMOD_LSHIFT) ||
                    !expect(editor::worldItemGizmoTransformsEqual(*worldItemTransform(item), before, GizmoOperation::AlignPose) &&
                        transformCommands_.empty(), "Shift+F alone leaves the selected spotlight in place")) { return false; }
                if (!pressAlignView()) { return false; }
                const auto after = *worldItemTransform(item);
                const auto expected = editor::editWorldItemGizmoTransform(before, cameraPose(cameraBefore), GizmoOperation::AlignPose);
                if (!expect(expected && editor::worldItemGizmoTransformsEqual(after, *expected, GizmoOperation::AlignPose) &&
                        viewportView_.camera() == cameraBefore && sceneBrowserHasInput(browser) &&
                        transformCommands_.size() == 1 && scene_.dirty(),
                        "Ctrl+Shift+F aligns Arcade spotlight with Scene Browser focused and hovered")) { return false; }
                undoTransform();
                if (!expect(editor::worldItemGizmoTransformsEqual(*worldItemTransform(item), before, GizmoOperation::AlignPose) &&
                        !scene_.dirty(), "One undo restores Arcade spotlight")) { return false; }
                spdlog::info("[Smoke Gizmo] Ctrl+Shift+F aligns Arcade spotlight after a Scene Browser click");
            }
            if (item != iesSpotItem) { continue; }

            const auto readSpotlight = [&] {
                return *std::ranges::find(scene_.lighting().lights, item.id, &scene::PunctualLight::stableId);
            };
            const auto samePose = [](const scene::PunctualLight& a, const scene::PunctualLight& b) {
                return length(a.position - b.position) < 0.0001f && length(a.direction - b.direction) < 0.0001f &&
                    std::abs(a.iesRollDegrees - b.iesRollDegrees) < 0.001f &&
                    std::abs(a.properties.range - b.properties.range) < 0.0001;
            };
            for (const auto operation : {GizmoOperation::Rotate, GizmoOperation::Scale}) {
                const auto before = readSpotlight();
                if (!dragLinearGizmo(operation, true)) { return false; }
                const auto after = readSpotlight();
                const auto lightBounds = input.boundsEdges;
                const bool rotate = operation == GizmoOperation::Rotate;
                const bool edited = rotate
                    ? length(after.direction - before.direction) > 0.1f &&
                        std::abs(after.iesRollDegrees - before.iesRollDegrees) > 0.1f &&
                        after.properties.range == before.properties.range
                    : length(after.direction - before.direction) < 0.0001f && after.iesRollDegrees == before.iesRollDegrees &&
                        std::abs(after.properties.range - before.properties.range * 1.25) < 0.001;
                if (!expect(edited && length(after.position - before.position) < 0.000001f && after.iesProfile == before.iesProfile &&
                        after.properties.innerConeAngle == before.properties.innerConeAngle &&
                        after.properties.outerConeAngle == before.properties.outerConeAngle,
                        "IES spotlight rotation edits direction and roll, and uniform scale edits only range")) { return false; }
                undoTransform();
                if (!render() || !expect(samePose(readSpotlight(), before) && !scene_.dirty() && boundsMatch(beforeBounds),
                        "Undo restores the IES spotlight pose and bounds")) { return false; }
                redoTransform();
                if (!render() || !expect(samePose(readSpotlight(), after) && scene_.dirty() && boundsMatch(lightBounds),
                        "Redo restores the IES spotlight pose and bounds")) { return false; }
                if (const char* path = std::getenv("METALLIC_LIGHT_TRS_GIZMO_SMOKE_CAPTURE"); rotate && path && *path &&
                    !expect(exportSmokeMainViewport(path), "Export the rotated IES spotlight and bounds")) { return false; }
                undoTransform();
            }
            gizmoOperation_ = GizmoOperation::Translate;
            spdlog::info("[Smoke Gizmo] IES spotlight rotate and scale preserve projector parameters and support undo and redo");

            resetTransformHistory();
            auto alignCamera = viewportView_.camera();
            alignCamera.eye = {4.6f, 2.65f, 0.1f};
            alignCamera.center = {1.72f, 0.05f, -0.15f};
            alignCamera.up = {0.0f, 1.0f, -0.3f};
            if (!expect(viewportView_.setCamera(alignCamera), "Set a rolled view for IES spotlight alignment")) { return false; }
            const auto beforeAlign = readSpotlight();
            if (!pressAlignView()) { return false; }
            const auto afterAlign = readSpotlight();
            if (!expect(scene::matrixNearlyEqual(editor::worldItemGizmoMatrix(editor::captureWorldItemGizmoTransform(afterAlign)),
                    cameraPose(alignCamera), 0.0001f) && viewportView_.camera() == alignCamera &&
                    afterAlign.iesProfile == beforeAlign.iesProfile &&
                    scene::lightPropertiesNearlyEqual(afterAlign.properties, beforeAlign.properties) &&
                    transformCommands_.size() == 1 && scene_.dirty(),
                    "Ctrl+Shift+F aligns the IES spotlight position, direction and roll in one edit")) { return false; }
            if (const char* path = std::getenv("METALLIC_ALIGN_VIEW_GIZMO_SMOKE_CAPTURE"); path && *path &&
                !expect(exportSmokeMainViewport(path), "Export the view-aligned IES spotlight")) { return false; }
            undoTransform();
            if (!expect(samePose(readSpotlight(), beforeAlign) && !scene_.dirty(),
                    "Undo restores the IES spotlight pose")) { return false; }
            redoTransform();
            if (!expect(samePose(readSpotlight(), afterAlign) && scene_.dirty(),
                    "Redo restores the view-aligned IES spotlight pose")) { return false; }
            undoTransform();
            spdlog::info("[Smoke Gizmo] Ctrl+Shift+F aligns the IES spotlight to the camera and supports undo and redo");
        }
        return true;
    };
    const auto drag = [&](int32_t nodeIndex, std::string name, bool geometry,
                          int32_t renderNodeIndex = scene::kInvalidSceneIndex,
                          std::optional<uint32_t> decalId = std::nullopt) {
        const scene::ConstSceneObject object = scene_.objectForNode(nodeIndex);
        if (!expect(object && object.hasComponent<scene::TransformComponent>(), "Find editable Sponza node")) {
            return false;
        }
        resetTransformHistory();
        sceneSelection_ = SceneSelection{
            .type = SceneSelectionType::Node,
            .object = object.entity(),
            .sceneLifetimeRevision = scene_.sceneGraph().lifetimeRevision(),
            .index = nodeIndex,
            .nodeIndex = nodeIndex,
        };
        const auto& transform = object.getComponent<scene::TransformComponent>();
        const auto* instance = renderNodeIndex != scene::kInvalidSceneIndex ? &scene_.renderNodes()[renderNodeIndex] : nullptr;
        const bool checkBounds = decalId || (instance && startupSampleId_ == "sponza-technology-showcase" && startupScenePath_.empty());
        const auto& decals = scene_.decals();
        const auto decalIt = decalId ? std::ranges::find(decals, *decalId, &scene::RenderDecal::id) : decals.end();
        if (decalId && !expect(decalIt != decals.end(), "Find the Floor Medallion decal")) { return false; }
        const auto originalDecals = decalId ? decals : std::vector<scene::RenderDecal>{};
        const auto ownerLocal = transform.localMatrix;
        const auto ownerWorld = transform.worldMatrix;
        const scene::RenderNode* sibling = nullptr;
        float4x4 siblingWorld;
        float3 position(ownerWorld.a03, ownerWorld.a13, ownerWorld.a23);
        if (instance) {
            const auto& sections = object.getComponent<scene::MeshComponent>().renderNodeIndices;
            sibling = &scene_.renderNodes()[sections[0] == renderNodeIndex ? sections[1] : sections[0]];
            siblingWorld = sibling->worldMatrix;
            const auto& primitive = scene_.renderPrimitives()[instance->renderPrimitiveIndex];
            const auto center = instance->worldMatrix * float4(primitive.localBounds.center(), 1.0f);
            position = float3(center.x, center.y, center.z);
            sceneSelection_.type = SceneSelectionType::RenderPrimitive;
            sceneSelection_.index = instance->renderPrimitiveIndex;
            sceneSelection_.renderNodeIndex = renderNodeIndex;
        }
        if (decalId) {
            const auto& decalWorld = decalIt->worldFromDecal;
            position = float3(decalWorld.a03, decalWorld.a13, decalWorld.a23);
        }
        const auto editedMatrix = [&]() -> const float4x4& {
            if (decalId) { return decalIt->worldFromDecal; }
            return instance ? instance->worldMatrix : transform.localMatrix;
        };
        applySceneCamera(scene::RenderCamera{
            .eye = position + (instance ? float3(3.2f, 1.5f, 1.6f) :
                decalId ? float3(3.6f, 2.1f, 0.25f) : float3(7.2f, 2.65f, 0.35f)),
            .center = position + ((instance || decalId) ? float3(0.0f, 0.0f, 0.0f) : float3(0.0f, 0.75f, 0.0f)),
            .yfov = 1.047197551196598,
            .znear = 0.05,
            .zfar = 100.0,
        });
        input.pointer.reset();
        if (!render() || !(decalId ? selectShowcaseDecal(object, *decalId) : selectTreeRow(object, renderNodeIndex))) {
            return false;
        }
        const auto parent = scene_.sceneGraph().object(object.getComponent<scene::RelationshipComponent>().parent);
        const auto parentWorld = parent ? parent.getComponent<scene::TransformComponent>().worldMatrix : float4x4::Identity();
        constexpr uint32_t kReadyFrameLimit = 120;
        constexpr uint32_t kWarmupFrames = 12;
        uint32_t readyFrames = 0;
        for (uint32_t frame = 0; frame < kReadyFrameLimit && readyFrames < kWarmupFrames; ++frame) {
            if (!render()) { return false; }
            readyFrames = input.center ? readyFrames + 1 : 0;
        }
        if (!expect(readyFrames == kWarmupFrames, "Viewport gizmo becomes available after scene streaming")) {
            return false;
        }
        if (checkBounds && !expect(boundsEdgesDrawn(12), "Selected decal or primitive displays all bounding box edges")) {
            return false;
        }
        const auto beforeBounds = checkBounds ? input.boundsEdges : std::vector<std::array<float, 4>>{};

        const auto [centerX, centerY] = *input.center;
        const auto before = editedMatrix();
        input.pointer = std::array{centerX, centerY};
        if (!render() || !expect(viewportGizmoCapturingMouse_, "Pointer hovers the translation handle")) {
            return false;
        }
        input.mouseDown = true;
        if (!render() || !expect(gizmoWasUsing_, "Mouse press starts the translation gesture")) { return false; }

        constexpr uint32_t kDragFrames = 16;
        double frameMilliseconds = 0.0;
        double bindingMilliseconds = 0.0;
        uint64_t recompiledPasses = 0;
        const auto geometryBefore = scene_.geometryTransformRevision();
        const auto decalRevisionBefore = scene_.decalRevision();
        const auto cameraBefore = viewportView_.camera();
        const auto graphGeneration = graphExecutor_->executionStats().graphGeneration;
        for (uint32_t frame = 1; frame <= kDragFrames; ++frame) {
            const auto previous = editedMatrix();
            const auto previousBounds = checkBounds ? input.boundsEdges : std::vector<std::array<float, 4>>{};
            input.pointer = std::array{centerX + 2.0f * float(frame), centerY};
            const auto begin = std::chrono::steady_clock::now();
            if (!render() || !expect(gizmoWasUsing_ && !scene::matrixNearlyEqual(editedMatrix(), previous),
                    "Each drag frame updates the selected transform")) {
                return false;
            }
            if (checkBounds && !expect(boundsEdgesDrawn(12) && !boundsMatch(previousBounds),
                    "Bounding box follows the selected transform in the same drag frame")) {
                return false;
            }
            frameMilliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            const auto& stats = graphExecutor_->executionStats();
            const auto refresh = std::ranges::find(stats.preparation, "Refresh scene bindings",
                &render::RenderGraphProfileSection::name);
            if (!expect(stats.graphGeneration == graphGeneration && refresh != stats.preparation.end(),
                    "RenderGraph executes the drag frame without rebuilding the graph")) {
                return false;
            }
            bindingMilliseconds += refresh->cpuMilliseconds;
            const auto recompiles = std::ranges::find(refresh->cpuWork, "recompiledPasses",
                &render::CPUProfileWorkCounter::name);
            if (recompiles != refresh->cpuWork.end()) { recompiledPasses += recompiles->value; }
        }
        input.mouseDown = false;
        if (!render() || !expect(!gizmoWasUsing_ && transformCommands_.size() == 1 && scene_.dirty(),
                "Mouse release commits one undoable transform")) {
            return false;
        }
        if (!expect((scene_.geometryTransformRevision() != geometryBefore) == geometry,
                "Mesh motion updates the geometry revision and camera motion preserves it")) {
            return false;
        }
        if (parent && !expect(scene::matrixNearlyEqual(parent.getComponent<scene::TransformComponent>().worldMatrix, parentWorld) &&
                scene::matrixNearlyEqual(transform.worldMatrix, parentWorld * transform.localMatrix),
                "Child gizmo updates local transform independently of its parent")) {
            return false;
        }
        if (instance && !expect(scene::matrixNearlyEqual(transform.localMatrix, ownerLocal) &&
                scene::matrixNearlyEqual(transform.worldMatrix, ownerWorld) &&
                scene::matrixNearlyEqual(sibling->worldMatrix, siblingWorld),
                "Primitive drag leaves the mesh node and sibling instance in place")) {
            return false;
        }
        if (decalId && !expect(scene_.decalRevision() > decalRevisionBefore &&
                scene::matrixNearlyEqual(transform.localMatrix, ownerLocal) &&
                scene::matrixNearlyEqual(transform.worldMatrix, ownerWorld) && viewportView_.camera() == cameraBefore &&
                std::ranges::equal(decals, originalDecals, [&](const auto& current, const auto& original) {
                    return current.id == original.id &&
                        (current.id == *decalId || scene::matrixNearlyEqual(current.worldFromDecal, original.worldFromDecal));
                }), "Decal drag updates only the selected projector")) { return false; }
        const auto after = editedMatrix();
        const auto afterBounds = checkBounds ? input.boundsEdges : std::vector<std::array<float, 4>>{};
        undoTransform();
        if (!expect(scene::matrixNearlyEqual(editedMatrix(), before) && !scene_.dirty(),
                "Undo restores the original transform") || !render()) {
            return false;
        }
        if (checkBounds && !expect(boundsMatch(beforeBounds), "Undo restores the original bounding box")) { return false; }
        redoTransform();
        if (!expect(scene::matrixNearlyEqual(editedMatrix(), after) && scene_.dirty(),
                "Redo restores the completed drag") || !render()) {
            return false;
        }
        if (checkBounds && !expect(boundsMatch(afterBounds), "Redo restores the moved bounding box")) { return false; }
        spdlog::info("[Smoke Gizmo] node='{}' frames={} meanFrameMs={:.3f} meanBindingRefreshMs={:.3f} recompiledPasses={}",
            name, kDragFrames, frameMilliseconds / kDragFrames, bindingMilliseconds / kDragFrames, recompiledPasses);
        return true;
    };

    viewportView_.setRenderResolution(640, 360);
    const auto& cameras = scene_.cameras();
    const auto camera = std::ranges::find(cameras, "Floor Decals", &scene::RenderCamera::name);
    if (camera != cameras.end() && !drag(camera->nodeIndex, camera->name, false)) { return false; }
    if (startupSampleId_ == "sponza-technology-showcase" && startupScenePath_.empty()) {
        if (!expect(camera != cameras.end(), "The showcase includes the Floor Decals camera and decal group") ||
            !drag(camera->nodeIndex, "Floor Medallion decal", false, scene::kInvalidSceneIndex, 1u)) { return false; }
        if (const char* path = std::getenv("METALLIC_DECAL_GIZMO_SMOKE_CAPTURE"); path && *path) {
            if (!expect(exportSmokeMainViewport(path), "Export the decal gizmo frame as a PPM image")) { return false; }
        }
        gizmoBoundsVisible_ = false;
        if (!render() || !expect(input.boundsEdges.empty() && input.drewGizmo,
                "Bounds toggle hides the box and leaves the gizmo visible")) { return false; }
        gizmoBoundsVisible_ = true;

        const auto perspective = viewportView_.camera();
        const float3 viewOffset(perspective.eye[0] - perspective.center[0],
            perspective.eye[1] - perspective.center[1], perspective.eye[2] - perspective.center[2]);
        auto clipped = perspective;
        clipped.nearPlane = length(viewOffset) + 0.2f;
        if (!viewportView_.setCamera(clipped) || !render() || !expect(!input.drewGizmo && boundsEdgesDrawn(8),
                "The near plane clips four decal edges and preserves the eight visible segments")) {
            return false;
        }
        if (!viewportView_.setCamera(perspective)) { return false; }
        undoTransform();
        spdlog::info("[Smoke Gizmo] Decal bounds follow drag, undo and redo, and respect the toggle and near-plane clipping");

        resetTransformHistory();
        const auto beforeAlign = *selectedDecal();
        auto expected = cameraPose(perspective);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const auto& column = beforeAlign.worldFromDecal[axis];
            expected[axis] = expected[axis] * length(float3(column.x, column.y, column.z));
        }
        if (!pressAlignView()) { return false; }
        const auto afterAlign = *selectedDecal();
        if (!expect(scene::matrixNearlyEqual(afterAlign.worldFromDecal, expected, 0.0001f) &&
                afterAlign.baseColorTexture == beforeAlign.baseColorTexture && viewportView_.camera() == perspective &&
                transformCommands_.size() == 1 && scene_.dirty(),
                "Ctrl+Shift+F aligns the decal to the camera and preserves its projector dimensions")) { return false; }
        undoTransform();
        if (!expect(scene::matrixNearlyEqual(selectedDecal()->worldFromDecal, beforeAlign.worldFromDecal) && !scene_.dirty(),
                "Undo restores the decal projector pose")) { return false; }
        redoTransform();
        if (!expect(scene::matrixNearlyEqual(selectedDecal()->worldFromDecal, afterAlign.worldFromDecal) && scene_.dirty(),
                "Redo restores the view-aligned decal projector pose")) { return false; }
        undoTransform();
        spdlog::info("[Smoke Gizmo] Ctrl+Shift+F aligns the decal to the camera and supports undo and redo");
        if (!checkShowcaseWorldItems()) { return false; }
    }
    const auto& nodes = scene_.nodes();
    const auto mesh = std::ranges::find(nodes, "Sponza Opaque", &scene::SceneNode::name);
    if (!expect(mesh != nodes.end(), "Sponza Opaque is available in the chosen sample")) { return false; }
    const auto meshNode = static_cast<int32_t>(mesh - nodes.begin());
    const auto meshName = mesh->name;
    const auto& sections = scene_.objectForNode(meshNode).getComponent<scene::MeshComponent>().renderNodeIndices;
    if (!expect(sections.size() >= 2, "Sponza contains multiple primitive instances") ||
        !drag(meshNode, meshName + " primitive", true, sections.front())) { return false; }
    if (const char* path = std::getenv("METALLIC_GIZMO_SMOKE_CAPTURE"); path && *path) {
        if (!expect(exportSmokeMainViewport(path), "Export the primitive gizmo frame as a PPM image")) { return false; }
    }
    undoTransform();
    const auto& primitiveInstance = scene_.renderNodes()[sections.front()];
    const auto& primitiveSibling = scene_.renderNodes()[sections[1]];
    const auto owner = scene_.objectForNode(meshNode);
    const auto ownerWorld = owner.getComponent<scene::TransformComponent>().worldMatrix;
    const auto siblingWorld = primitiveSibling.worldMatrix;
    const auto primitiveCenter = scene_.renderPrimitives()[primitiveInstance.renderPrimitiveIndex].localBounds.center();
    const auto axisLengths = [](const float4x4& matrix) {
        return float3(length(float3(matrix.a00, matrix.a10, matrix.a20)),
            length(float3(matrix.a01, matrix.a11, matrix.a21)), length(float3(matrix.a02, matrix.a12, matrix.a22)));
    };
    for (const auto operation : {GizmoOperation::Rotate, GizmoOperation::Scale}) {
        const auto before = primitiveInstance.worldMatrix;
        const auto beforeLengths = axisLengths(before);
        const auto beforeCenter = before * float4(primitiveCenter, 1.0f);
        const auto beforeRevision = scene_.geometryTransformRevision();
        if (!dragLinearGizmo(operation, startupSampleId_ == "sponza-technology-showcase" && startupScenePath_.empty())) {
            return false;
        }
        const auto after = primitiveInstance.worldMatrix;
        const auto afterCenter = after * float4(primitiveCenter, 1.0f);
        const auto expectedLengths = operation == GizmoOperation::Rotate ? beforeLengths : beforeLengths * 1.25f;
        if (!expect(!scene::matrixNearlyEqual(before, after) && scene_.geometryTransformRevision() > beforeRevision &&
                length(axisLengths(after) - expectedLengths) < 0.0001f &&
                length(float3(afterCenter.x - beforeCenter.x, afterCenter.y - beforeCenter.y, afterCenter.z - beforeCenter.z)) < 0.0001f &&
                scene::matrixNearlyEqual(owner.getComponent<scene::TransformComponent>().worldMatrix, ownerWorld) &&
                scene::matrixNearlyEqual(primitiveSibling.worldMatrix, siblingWorld),
                "Rotating and scaling a primitive keeps its bounds center, mesh owner, and siblings in place")) { return false; }
        undoTransform();
        if (!expect(scene::matrixNearlyEqual(primitiveInstance.worldMatrix, before) && !scene_.dirty(),
                "Undo restores the primitive transform")) { return false; }
        redoTransform();
        if (!expect(scene::matrixNearlyEqual(primitiveInstance.worldMatrix, after) && scene_.dirty(),
                "Redo restores the primitive transform")) { return false; }
        undoTransform();
    }
    gizmoOperation_ = GizmoOperation::Translate;
    spdlog::info("[Smoke Gizmo] Primitive rotate and scale preserve pivot and support undo and redo");
    if (!drag(meshNode, meshName, true)) { return false; }

    input.pointer.reset();
    const auto& world = selectedSceneObject().getComponent<scene::TransformComponent>().worldMatrix;
    auto depthCamera = viewportView_.camera();
    depthCamera.up = {0.0f, 1.0f, 0.0f};
    for (float depth : {1.0f, 0.5f * depthCamera.nearPlane, -1.0f, depthCamera.farPlane + 1.0f, 1.0f}) {
        depthCamera.eye = {world.a03, world.a13, world.a23 + depth};
        depthCamera.center = {world.a03, world.a13, depthCamera.eye[2] - 1.0f};
        const bool visible = depth == 1.0f;
        if (!viewportView_.setCamera(depthCamera) || !render() ||
            !expect(input.drewGizmo == visible && input.center.has_value() == visible,
                "Gizmo follows the viewport depth range as the camera passes its pivot")) {
            return false;
        }
    }

    const auto [pivotX, pivotY] = *input.center;
    input.pointer = std::array{pivotX + 40.0f, pivotY};
    if (!render()) { return false; }
    input.rightMouseDown = true;
    if (!render() || !expect(viewportCameraDragButton_ == ImGuiMouseButton_Right && input.center && input.drewGizmo,
            "Start camera navigation with a visible gizmo")) { return false; }
    const auto centerBeforeLook = *input.center;
    const auto lookBefore = viewportView_.camera().center;
    (*input.pointer)[0] += 20.0f;
    if (!render() || !expect(input.center && input.drewGizmo && *input.center != centerBeforeLook &&
            viewportView_.camera().center != lookBefore, "Gizmo moves in the same frame as the camera")) {
        return false;
    }
    const auto centerAfterLook = *input.center;
    input.rightMouseDown = false;
    if (!render() || !expect(input.center == centerAfterLook, "Gizmo stays aligned when the camera stops")) { return false; }
    spdlog::info("[Smoke Gizmo] Gizmo respects camera clipping and tracks the current camera frame");

    // Exercise camera drags from a pitched pose whose authored up differs from world up.
    sceneSelection_ = {};
    viewportGizmoCapturingMouse_ = false;
    input.pointer.reset();
    auto initialCamera = viewportView_.camera();
    initialCamera.eye = {7.2f, 2.65f, 0.35f};
    initialCamera.center = {initialCamera.eye[0] - 0.8660254f, initialCamera.eye[1] - 0.5f, initialCamera.eye[2]};
    initialCamera.up = {-0.5f, 0.8660254f, 0.0f};
    for (const auto button : {ImGuiMouseButton_Right, ImGuiMouseButton_Left}) {
        const bool orbit = button == ImGuiMouseButton_Left;
        const auto cameraInputFrame = [&](float x, bool down, bool alt) {
            auto& io = ImGui::GetIO();
            io.AddMousePosEvent(x, 250.0f);
            io.AddKeyEvent(ImGuiMod_Alt, alt);
            io.AddKeyEvent(ImGuiKey_LeftAlt, alt);
            io.AddMouseButtonEvent(button, down);
            ImGui::NewFrame();
            ImGui::Begin("Camera interaction test");
            viewportInteractionEnabled_ = true;
            viewportHovered_ = true;
            handleViewportCameraControls(ImVec2(100.0f, 100.0f), ImVec2(500.0f, 400.0f));
            ImGui::End();
            ImGui::Render();
        };
        if (!expect(viewportView_.setCamera(initialCamera), "Set pitched viewport camera")) { return false; }
        cameraInputFrame(250.0f, false, orbit);
        cameraInputFrame(250.0f, true, orbit);
        cameraInputFrame(350.0f, true, orbit);
        cameraInputFrame(350.0f, false, orbit);
        cameraInputFrame(350.0f, false, false);

        const auto& moved = viewportView_.camera();
        const float3 forward(moved.center[0] - moved.eye[0], moved.center[1] - moved.eye[1], moved.center[2] - moved.eye[2]);
        const float3 up(moved.up[0], moved.up[1], moved.up[2]);
        if (!expect(std::abs(forward.x) < 0.0001f && std::abs(forward.y + 0.5f) < 0.0001f &&
                std::abs(forward.z + 0.8660254f) < 0.0001f &&
                std::abs(cross(forward, up).y) < 0.0001f,
                "Horizontal camera drag turns 90 degrees without changing pitch or adding roll") ||
            !expect(orbit ? moved.center == initialCamera.center : moved.eye == initialCamera.eye,
                "Camera gesture keeps its pivot fixed")) {
            return false;
        }
    }
    spdlog::info("[Smoke Gizmo] Pitched camera stays level through right-button look and Alt-left orbit");
    return render();
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
        if (testFinalBlit && index == 4) {
            viewportView_.setAdaptiveResolution(true);
            renderGraph_.markDirty();
        } else if (testFinalBlit && index == 7) {
            viewportView_.setRenderResolution(1920, 1080);
            renderGraph_.markDirty();
        }
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

        if (testFinalBlit && index >= 2 && index < 8) {
            const auto* source = graphExecutor_->outputResource(sourceOutput);
            const auto* final = graphExecutor_->outputResource(activePreviewOutput_);
            if (!expect(source != nullptr && final != nullptr &&
                    source->desc.width == viewportView_.renderWidth(graphExecutor_->width()) &&
                    source->desc.height == viewportView_.renderHeight(graphExecutor_->height()) &&
                    final->desc.width == graphExecutor_->width() && final->desc.height == graphExecutor_->height(),
                    "Fixed/adaptive scene extent and actual FinalBlit extent match the selected view mode")) {
                return false;
            }
            if (!viewportView_.adaptiveResolution()) {
                const uint64_t renderWidth = viewportView_.renderWidth();
                const uint64_t renderHeight = viewportView_.renderHeight();
                const uint64_t scaledWidth = final->desc.width * renderHeight;
                const uint64_t scaledHeight = final->desc.height * renderWidth;
                if (!expect(std::max(scaledWidth, scaledHeight) - std::min(scaledWidth, scaledHeight) <=
                        std::max(renderWidth, renderHeight),
                        "Fixed FinalBlit fits the viewport at the render aspect ratio")) {
                    return false;
                }
            }
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
