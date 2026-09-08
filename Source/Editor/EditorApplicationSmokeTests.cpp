#include "Editor/EditorApplication.h"

#include "imgui.h"
#include "imgui_internal.h"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <cmath>

namespace metallic {

bool EditorApplication::runSliderDebugSmokeTest()
{
    const auto expect = [](bool condition, const char* message) {
        if (!condition) { spdlog::error("[Smoke Slider] {}", message); }
        return condition;
    };
    if (!renderFrame() || !expect(viewportPreviewValid_, "Comparison viewport renders")) { return false; }
    auto* slider = viewportSliderDebugNode();
    auto* cameraNode = viewportCameraRenderGraphNode();
    if (!expect(slider != nullptr && cameraNode != nullptr, "Find comparison through exposure and presentation")) { return false; }
    const uint32_t sliderId = slider->id;
    const uint64_t historyRevision = historyResources_.invalidationRevision();
    const auto split = [&] { return renderGraph_.findNode(sliderId)->runtimeProperties.value("splitPosition", 0.5f); };
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

    auto cameraProperties = cameraNode->properties;
    cameraProperties.merge_patch(cameraNode->runtimeProperties);
    cameraProperties["camera"]["eye"][0] = cameraProperties["camera"]["eye"][0].get<float>() + 0.5f;
    applyBunnyCameraProperties(cameraProperties, "Smoke camera");
    uint32_t linkedCameras = 0;
    for (const auto& candidate : renderGraph_.nodes()) {
        if (candidate.properties.value("cameraSyncGroup", "") != "LookDevComparison") { continue; }
        ++linkedCameras;
        if (!expect(candidate.runtimeProperties["camera"] == cameraProperties["camera"],
                "Viewport camera synchronizes both BSDF paths")) { return false; }
    }
    if (!expect(linkedCameras >= 2, "Comparison contains linked cameras")) { return false; }
    if (!expect(historyResources_.invalidationRevision() > historyRevision, "Camera movement resets accumulation")) { return false; }
    activePreviewOutput_ = cameraNode->name + ".color";
    if (!expect(viewportSliderDebugNode() == nullptr && !overlayFrame(300, 175, true),
            "Raw producer preview has no comparison interaction")) { return false; }
    overlayFrame(300, 175, false);
    spdlog::info("[Smoke Slider] Passed GPU viewport, drag, endpoints, axes, camera gestures, linked cameras and history retention");
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
