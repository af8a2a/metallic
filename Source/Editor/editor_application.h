#pragma once

#include <cstdint>
#include <memory>

struct SDL_Renderer;
struct SDL_Texture;
struct SDL_Window;

namespace metallic {
namespace render {
class TrianglePreviewRenderer;
} // namespace render

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
    bool updateViewportPreview(uint32_t width, uint32_t height);
    void destroyViewportTexture();

    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    SDL_Texture* viewportTexture_ = nullptr;
    std::unique_ptr<render::TrianglePreviewRenderer> trianglePreviewRenderer_;
    uint32_t viewportTextureWidth_ = 0;
    uint32_t viewportTextureHeight_ = 0;
    bool running_ = true;
    bool smokeTest_ = false;
    bool imguiContextCreated_ = false;
    bool imguiPlatformInitialized_ = false;
    bool imguiRendererInitialized_ = false;
    bool viewportPreviewValid_ = false;
    float mainScale_ = 1.0f;
    float clearColor_[4] = {0.07f, 0.08f, 0.10f, 1.0f};
};

} // namespace metallic
