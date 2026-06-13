#pragma once

struct SDL_Renderer;
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

    SDL_Window* window_ = nullptr;
    SDL_Renderer* renderer_ = nullptr;
    bool running_ = true;
    bool smokeTest_ = false;
    bool imguiContextCreated_ = false;
    bool imguiPlatformInitialized_ = false;
    bool imguiRendererInitialized_ = false;
    float mainScale_ = 1.0f;
    float clearColor_[4] = {0.07f, 0.08f, 0.10f, 1.0f};
};

} // namespace metallic
