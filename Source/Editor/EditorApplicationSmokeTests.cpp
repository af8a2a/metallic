#include "Editor/EditorApplication.h"

#include "imgui.h"
#include "imgui_internal.h"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <cstdlib>

namespace metallic {

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
    const std::string sourceOutput = activePreviewOutput_;
    std::string finalInput;
    uint32_t finalId = 0;
    uint32_t finalEdgeId = 0;
    if (testFinalBlit) {
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
