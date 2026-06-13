#pragma once

#include "Runtime/Render/RenderGraph/render_graph.h"

#include <cstdint>
#include <memory>
#include <string>

struct SDL_Renderer;
struct SDL_Texture;
struct SDL_Window;

namespace metallic {

class EditorApplication {
public:
    int run(bool smokeTest = false);

private:
    bool initialize();
    void shutdown();
    void pollEvents();
    void renderFrame();
    void drawDockspace();
    void drawPanels();
    void drawViewportPanel();
    void drawRenderGraphPanel();
    void drawRenderGraphInspector();
    void drawRenderGraphNode(const render::RenderGraphNode& node);
    void drawRenderGraphMenuBar();
    void resetDefaultRenderGraph();
    void saveRenderGraph();
    void loadRenderGraph();
    bool updateViewportPreview(uint32_t width, uint32_t height);
    void destroyViewportTexture();
    int graphInputAttributeId(const render::RenderGraphNode& node, uint32_t fieldIndex) const;
    int graphOutputAttributeId(const render::RenderGraphNode& node, uint32_t fieldIndex) const;

    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    SDL_Texture* viewportTexture_ = nullptr;
    std::unique_ptr<render::RenderGraphPreviewRenderer> graphPreviewRenderer_;
    render::RenderGraph renderGraph_;
    uint32_t viewportTextureWidth_ = 0;
    uint32_t viewportTextureHeight_ = 0;
    bool running_ = true;
    bool smokeTest_ = false;
    bool imguiContextCreated_ = false;
    bool imnodesContextCreated_ = false;
    bool imguiPlatformInitialized_ = false;
    bool imguiRendererInitialized_ = false;
    bool viewportPreviewValid_ = false;
    bool graphEditorPositionsInitialized_ = false;
    float mainScale_ = 1.0f;
    float clearColor_[4] = {0.07f, 0.08f, 0.10f, 1.0f};
    int selectedGraphNodeId_ = -1;
    int selectedGraphLinkId_ = -1;
    char graphFilePath_[260] = "Pipelines/default.metallic_graph.json";
    char graphNodeNameBuffer_[128] = {};
    std::string renderGraphStatus_;
};

} // namespace metallic
