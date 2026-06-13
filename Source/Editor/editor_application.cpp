#include "Editor/editor_application.h"

#include "Runtime/Render/GAPI/rhi.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <iostream>

namespace metallic {
namespace {

constexpr int kBaseWindowWidth = 1600;
constexpr int kBaseWindowHeight = 900;
constexpr uint32_t kMaxViewportPreviewSize = 2048;
#if defined(SDL_PLATFORM_WINDOWS)
constexpr const char* kEditorRendererDrivers = "direct3d12,vulkan";
#else
constexpr const char* kEditorRendererDrivers = "vulkan";
#endif

float getMainDisplayScale()
{
    const SDL_DisplayID display = SDL_GetPrimaryDisplay();
    const float scale = display != 0 ? SDL_GetDisplayContentScale(display) : 1.0f;
    return std::max(scale, 1.0f);
}

} // namespace

int EditorApplication::run(bool smokeTest)
{
    smokeTest_ = smokeTest;

    if (!initialize()) {
        shutdown();
        return 1;
    }

    if (smokeTest) {
        pollEvents();
        renderFrame();
        shutdown();
        return 0;
    }

    while (running_) {
        pollEvents();

        if ((SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED) != 0) {
            SDL_Delay(10);
            continue;
        }

        renderFrame();
    }

    shutdown();
    return 0;
}

bool EditorApplication::initialize()
{
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        SDL_Log("SDL_Init failed: %s", SDL_GetError());
        return false;
    }

#ifdef SDL_HINT_IME_SHOW_UI
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#endif

    mainScale_ = getMainDisplayScale();
    const SDL_WindowFlags windowFlags =
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | SDL_WINDOW_HIGH_PIXEL_DENSITY;

    window_ = SDL_CreateWindow(
        "Metallic Engine Editor",
        static_cast<int>(kBaseWindowWidth * mainScale_),
        static_cast<int>(kBaseWindowHeight * mainScale_),
        windowFlags);
    if (window_ == nullptr) {
        SDL_Log("SDL_CreateWindow failed: %s", SDL_GetError());
        return false;
    }

    renderer_ = SDL_CreateRenderer(window_, kEditorRendererDrivers);
    if (renderer_ == nullptr) {
        SDL_Log("SDL_CreateRenderer failed for drivers '%s': %s", kEditorRendererDrivers, SDL_GetError());
        return false;
    }
    SDL_Log("SDL renderer backend: %s", SDL_GetRendererName(renderer_));

    SDL_SetRenderVSync(renderer_, 1);
    SDL_SetWindowPosition(window_, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
    SDL_ShowWindow(window_);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    imguiContextCreated_ = true;

    ImGuiIO& io = ImGui::GetIO();
    if (smokeTest_) {
        io.IniFilename = nullptr;
    }

    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.ScaleAllSizes(mainScale_);
    style.FontScaleDpi = mainScale_;

    imguiPlatformInitialized_ = ImGui_ImplSDL3_InitForSDLRenderer(window_, renderer_);
    if (!imguiPlatformInitialized_) {
        SDL_Log("ImGui SDL3 platform backend initialization failed");
        return false;
    }

    imguiRendererInitialized_ = ImGui_ImplSDLRenderer3_Init(renderer_);
    if (!imguiRendererInitialized_) {
        SDL_Log("ImGui SDL_Renderer3 backend initialization failed");
        return false;
    }

    trianglePreviewRenderer_ = std::make_unique<render::TrianglePreviewRenderer>();
    const render::Result previewResult = trianglePreviewRenderer_->initialize(false);
    if (previewResult != render::Result::Success) {
        std::cerr << "Triangle preview RHI initialization failed with Result "
                  << static_cast<int>(previewResult) << '\n';
        trianglePreviewRenderer_.reset();
    }

    return true;
}

void EditorApplication::shutdown()
{
    if (imguiRendererInitialized_) {
        ImGui_ImplSDLRenderer3_Shutdown();
        imguiRendererInitialized_ = false;
    }

    if (imguiPlatformInitialized_) {
        ImGui_ImplSDL3_Shutdown();
        imguiPlatformInitialized_ = false;
    }

    if (imguiContextCreated_) {
        ImGui::DestroyContext();
        imguiContextCreated_ = false;
    }

    destroyViewportTexture();
    trianglePreviewRenderer_.reset();

    if (renderer_ != nullptr) {
        SDL_DestroyRenderer(renderer_);
        renderer_ = nullptr;
    }

    if (window_ != nullptr) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }

    SDL_Quit();
}

void EditorApplication::pollEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        ImGui_ImplSDL3_ProcessEvent(&event);

        if (event.type == SDL_EVENT_QUIT) {
            running_ = false;
        }

        if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
            event.window.windowID == SDL_GetWindowID(window_)) {
            running_ = false;
        }
    }
}

void EditorApplication::renderFrame()
{
    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    drawDockspace();
    drawPanels();

    ImGui::Render();
    const ImGuiIO& io = ImGui::GetIO();

    SDL_SetRenderScale(renderer_, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
    SDL_SetRenderDrawColorFloat(
        renderer_,
        clearColor_[0],
        clearColor_[1],
        clearColor_[2],
        clearColor_[3]);
    SDL_RenderClear(renderer_);
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer_);
    SDL_RenderPresent(renderer_);
}

void EditorApplication::drawDockspace()
{
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_MenuBar |
        ImGuiWindowFlags_NoDocking |
        ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoBringToFrontOnFocus |
        ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));

    ImGui::Begin("MetallicEditorDockspace", nullptr, windowFlags);
    ImGui::PopStyleVar(3);

    const ImGuiID dockspaceId = ImGui::GetID("MetallicDockspace");
    ImGui::DockSpace(dockspaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit")) {
                running_ = false;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Window")) {
            ImGui::MenuItem("Scene", nullptr, false, false);
            ImGui::MenuItem("Viewport", nullptr, false, false);
            ImGui::MenuItem("Inspector", nullptr, false, false);
            ImGui::MenuItem("Assets", nullptr, false, false);
            ImGui::MenuItem("Console", nullptr, false, false);
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

    ImGui::End();
}

void EditorApplication::drawPanels()
{
    ImGui::Begin("Scene");
    ImGui::TextUnformatted("Scene graph");
    ImGui::Separator();
    ImGui::BulletText("Camera");
    ImGui::BulletText("Directional Light");
    ImGui::End();

    drawViewportPanel();

    ImGui::Begin("Inspector");
    ImGui::TextUnformatted("Selection");
    ImGui::Separator();
    ImGui::TextUnformatted("No entity selected");
    ImGui::End();

    ImGui::Begin("Assets");
    ImGui::TextUnformatted(PROJECT_SOURCE_DIR);
    ImGui::End();

    ImGui::Begin("Console");
    ImGui::TextUnformatted("Metallic editor initialized with SDL3.");
    ImGui::End();
}

void EditorApplication::drawViewportPanel()
{
    ImGui::Begin("Viewport");

    ImVec2 available = ImGui::GetContentRegionAvail();
    available.x = std::max(available.x, 1.0f);
    available.y = std::max(available.y, 1.0f);
    ImGui::InvisibleButton("ViewportCanvas", available);

    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max = ImGui::GetItemRectMax();
    const float width = max.x - min.x;
    const float height = max.y - min.y;
    const uint32_t previewWidth = std::clamp(
        static_cast<uint32_t>(std::ceil(width)),
        1u,
        kMaxViewportPreviewSize);
    const uint32_t previewHeight = std::clamp(
        static_cast<uint32_t>(std::ceil(height)),
        1u,
        kMaxViewportPreviewSize);
    const bool hasRhiPreview = updateViewportPreview(previewWidth, previewHeight);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRect(min, max, true);
    drawList->AddRectFilled(min, max, IM_COL32(16, 18, 22, 255));

    if (hasRhiPreview) {
        drawList->AddImage(
            static_cast<ImTextureID>(reinterpret_cast<std::uintptr_t>(viewportTexture_)),
            min,
            max,
            ImVec2(0.0f, 0.0f),
            ImVec2(1.0f, 1.0f));
    } else {
        const float gridStep = 32.0f * mainScale_;
        for (float x = min.x; x < max.x; x += gridStep) {
            drawList->AddLine(ImVec2(x, min.y), ImVec2(x, max.y), IM_COL32(32, 37, 45, 255));
        }
        for (float y = min.y; y < max.y; y += gridStep) {
            drawList->AddLine(ImVec2(min.x, y), ImVec2(max.x, y), IM_COL32(32, 37, 45, 255));
        }

        const ImVec2 center(min.x + width * 0.5f, min.y + height * 0.52f);
        const float radius = std::max(std::min(width, height) * 0.28f, 24.0f * mainScale_);
        const ImVec2 p0(center.x, center.y - radius);
        const ImVec2 p1(center.x - radius * 0.9f, center.y + radius * 0.72f);
        const ImVec2 p2(center.x + radius * 0.9f, center.y + radius * 0.72f);

        drawList->AddTriangleFilled(p0, p1, p2, IM_COL32(71, 140, 255, 255));
        drawList->AddTriangle(p0, p1, p2, IM_COL32(225, 237, 255, 255), 2.0f * mainScale_);
    }

    drawList->AddRect(min, max, IM_COL32(58, 67, 80, 255));
    drawList->PopClipRect();

    ImGui::End();
}

bool EditorApplication::updateViewportPreview(uint32_t width, uint32_t height)
{
    if (renderer_ == nullptr || trianglePreviewRenderer_ == nullptr) {
        return false;
    }

    if (viewportPreviewValid_ &&
        viewportTexture_ != nullptr &&
        viewportTextureWidth_ == width &&
        viewportTextureHeight_ == height) {
        return true;
    }

    const render::Result renderResult = trianglePreviewRenderer_->render(width, height);
    if (renderResult != render::Result::Success) {
        std::cerr << "Triangle preview render failed with Result "
                  << static_cast<int>(renderResult) << '\n';
        viewportPreviewValid_ = false;
        return false;
    }

    if (viewportTexture_ == nullptr ||
        viewportTextureWidth_ != width ||
        viewportTextureHeight_ != height) {
        destroyViewportTexture();
        viewportTexture_ = SDL_CreateTexture(
            renderer_,
            SDL_PIXELFORMAT_RGBA32,
            SDL_TEXTUREACCESS_STATIC,
            static_cast<int>(width),
            static_cast<int>(height));
        if (viewportTexture_ == nullptr) {
            std::cerr << "SDL_CreateTexture failed: " << SDL_GetError() << '\n';
            viewportPreviewValid_ = false;
            return false;
        }

        SDL_SetTextureBlendMode(viewportTexture_, SDL_BLENDMODE_NONE);
        SDL_SetTextureScaleMode(viewportTexture_, SDL_SCALEMODE_LINEAR);
        viewportTextureWidth_ = width;
        viewportTextureHeight_ = height;
    }

    const std::vector<uint32_t>& pixels = trianglePreviewRenderer_->pixels();
    if (pixels.size() < static_cast<size_t>(width) * static_cast<size_t>(height)) {
        viewportPreviewValid_ = false;
        return false;
    }

    if (!SDL_UpdateTexture(
            viewportTexture_,
            nullptr,
            pixels.data(),
            static_cast<int>(width * sizeof(uint32_t)))) {
        std::cerr << "SDL_UpdateTexture failed: " << SDL_GetError() << '\n';
        viewportPreviewValid_ = false;
        return false;
    }

    viewportPreviewValid_ = true;
    return true;
}

void EditorApplication::destroyViewportTexture()
{
    if (viewportTexture_ != nullptr) {
        SDL_DestroyTexture(viewportTexture_);
        viewportTexture_ = nullptr;
    }
    viewportTextureWidth_ = 0;
    viewportTextureHeight_ = 0;
    viewportPreviewValid_ = false;
}

} // namespace metallic
