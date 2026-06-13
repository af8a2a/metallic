#include "Editor/editor_application.h"

#include "Runtime/Render/GAPI/rhi.h"
#include "Runtime/Render/RenderGraph/render_graph.h"
#include "imnodes.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"
#include "imgui_impl_sdlrenderer3.h"

#include <SDL3/SDL.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

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

std::filesystem::path resolveGraphAssetPath(const char* path)
{
    std::filesystem::path assetPath(path == nullptr ? "" : path);
    if (assetPath.empty()) {
        assetPath = "Pipelines/default.metallic_graph.json";
    }
    if (assetPath.is_relative()) {
        assetPath = std::filesystem::path(PROJECT_SOURCE_DIR) / assetPath;
    }
    return assetPath;
}

std::string makeUniqueNodeName(const render::RenderGraph& graph, const std::string& type)
{
    std::string base = type;
    constexpr const char* suffix = "Pass";
    if (base.size() > std::strlen(suffix) &&
        base.compare(base.size() - std::strlen(suffix), std::strlen(suffix), suffix) == 0) {
        base.resize(base.size() - std::strlen(suffix));
    }
    if (base.empty()) {
        base = "Pass";
    }

    std::string name = base;
    uint32_t index = 1;
    while (graph.findNode(name) != nullptr) {
        name = base + std::to_string(++index);
    }
    return name;
}

render::RenderGraphProperties defaultPropertiesForPass(const std::string& type)
{
    if (type == "ClearColorPass") {
        return render::RenderGraphProperties{
            {"color", {0.04f, 0.06f, 0.09f, 1.0f}},
        };
    }
    return render::RenderGraphProperties::object();
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

    ImNodes::CreateContext();
    imnodesContextCreated_ = true;
    ImNodes::StyleColorsDark();

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

    resetDefaultRenderGraph();

    graphPreviewRenderer_ = std::make_unique<render::RenderGraphPreviewRenderer>();
    const render::Result previewResult = graphPreviewRenderer_->initialize(false);
    if (!previewResult) {
        std::cerr << "RenderGraph preview RHI initialization failed with Result "
                  << render::resultToString(previewResult) << '\n';
        graphPreviewRenderer_.reset();
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

    if (imnodesContextCreated_) {
        ImNodes::DestroyContext();
        imnodesContextCreated_ = false;
    }

    if (imguiContextCreated_) {
        ImGui::DestroyContext();
        imguiContextCreated_ = false;
    }

    destroyViewportTexture();
    graphPreviewRenderer_.reset();

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
            if (ImGui::MenuItem("New Render Graph")) {
                resetDefaultRenderGraph();
            }
            if (ImGui::MenuItem("Load Render Graph")) {
                loadRenderGraph();
            }
            if (ImGui::MenuItem("Save Render Graph")) {
                saveRenderGraph();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                running_ = false;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Window")) {
            ImGui::MenuItem("Scene", nullptr, false, false);
            ImGui::MenuItem("Viewport", nullptr, false, false);
            ImGui::MenuItem("Render Graph", nullptr, false, false);
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
    drawRenderGraphPanel();

    drawRenderGraphInspector();

    ImGui::Begin("Assets");
    ImGui::TextUnformatted(PROJECT_SOURCE_DIR);
    ImGui::End();

    ImGui::Begin("Console");
    ImGui::TextUnformatted("Metallic editor initialized with SDL3.");
    if (!renderGraphStatus_.empty()) {
        ImGui::Separator();
        ImGui::TextWrapped("%s", renderGraphStatus_.c_str());
    }
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
    if (renderer_ == nullptr || graphPreviewRenderer_ == nullptr) {
        return false;
    }

    if (viewportPreviewValid_ &&
        viewportTexture_ != nullptr &&
        viewportTextureWidth_ == width &&
        viewportTextureHeight_ == height &&
        !renderGraph_.dirty()) {
        return true;
    }

    const render::Result renderResult = graphPreviewRenderer_->render(renderGraph_, width, height);
    if (!renderResult) {
        std::cerr << "RenderGraph preview render failed with Result "
                  << render::resultToString(renderResult) << '\n';
        renderGraphStatus_ = graphPreviewRenderer_->lastLog();
        viewportPreviewValid_ = false;
        return false;
    }
    renderGraphStatus_ = graphPreviewRenderer_->lastLog();

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

    const std::vector<uint32_t>& pixels = graphPreviewRenderer_->pixels();
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

void EditorApplication::resetDefaultRenderGraph()
{
    renderGraph_ = render::RenderGraph::createDefaultTriangleGraph();
    graphEditorPositionsInitialized_ = false;
    selectedGraphNodeId_ = -1;
    selectedGraphLinkId_ = -1;
    viewportPreviewValid_ = false;
    renderGraphStatus_ = "Created default RenderGraph";
}

void EditorApplication::saveRenderGraph()
{
    std::string message;
    const std::filesystem::path path = resolveGraphAssetPath(graphFilePath_);
    if (!render::saveRenderGraphToFile(renderGraph_, path, message)) {
        renderGraphStatus_ = message;
        return;
    }
    renderGraphStatus_ = message;
}

void EditorApplication::loadRenderGraph()
{
    render::RenderGraph loadedGraph;
    std::string message;
    const std::filesystem::path path = resolveGraphAssetPath(graphFilePath_);
    if (!render::loadRenderGraphFromFile(path, loadedGraph, message)) {
        renderGraphStatus_ = message;
        return;
    }
    renderGraph_ = std::move(loadedGraph);
    graphEditorPositionsInitialized_ = false;
    selectedGraphNodeId_ = -1;
    selectedGraphLinkId_ = -1;
    viewportPreviewValid_ = false;
    renderGraphStatus_ = message;
}

void EditorApplication::drawRenderGraphMenuBar()
{
    ImGui::PushItemWidth(std::max(220.0f * mainScale_, ImGui::GetContentRegionAvail().x * 0.32f));
    ImGui::InputText("##GraphPath", graphFilePath_, sizeof(graphFilePath_));
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("New")) {
        resetDefaultRenderGraph();
    }
    ImGui::SameLine();
    if (ImGui::Button("Load")) {
        loadRenderGraph();
    }
    ImGui::SameLine();
    if (ImGui::Button("Save")) {
        saveRenderGraph();
    }
    ImGui::SameLine();
    if (ImGui::Button("Validate")) {
        std::string log;
        renderGraph_.validate(log);
        renderGraphStatus_ = log;
    }

    if (!renderGraphStatus_.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", renderGraphStatus_.c_str());
    }
}

int EditorApplication::graphInputAttributeId(const render::RenderGraphNode& node, uint32_t fieldIndex) const
{
    return static_cast<int>(node.id * 100u + 1u + fieldIndex);
}

int EditorApplication::graphOutputAttributeId(const render::RenderGraphNode& node, uint32_t fieldIndex) const
{
    return static_cast<int>(node.id * 100u + 51u + fieldIndex);
}

void EditorApplication::drawRenderGraphNode(const render::RenderGraphNode& node)
{
    std::unique_ptr<render::RenderGraphPass> pass = render::createRenderGraphPass(node.type);
    render::RenderPassReflection reflection;
    if (pass != nullptr) {
        pass->setProperties(node.properties);
        reflection = pass->reflect(render::RenderGraphCompileContext{});
    }

    ImNodes::BeginNode(static_cast<int>(node.id));
    ImNodes::BeginNodeTitleBar();
    ImGui::TextUnformatted(node.name.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("(%s)", node.type.c_str());
    ImNodes::EndNodeTitleBar();

    uint32_t inputIndex = 0;
    uint32_t outputIndex = 0;
    for (const render::RenderGraphField& field : reflection.fields()) {
        if (field.visibility == render::RenderGraphFieldVisibility::Input) {
            ImNodes::BeginInputAttribute(graphInputAttributeId(node, inputIndex++));
            ImGui::TextUnformatted(field.name.c_str());
            ImNodes::EndInputAttribute();
        }
    }

    if (node.type == "ClearColorPass" && node.properties.contains("color")) {
        ImNodes::BeginStaticAttribute(static_cast<int>(node.id * 100u + 90u));
        const auto& color = node.properties["color"];
        ImGui::ColorButton(
            "##ClearPreview",
            ImVec4(
                color.size() > 0 ? color[0].get<float>() : 0.0f,
                color.size() > 1 ? color[1].get<float>() : 0.0f,
                color.size() > 2 ? color[2].get<float>() : 0.0f,
                color.size() > 3 ? color[3].get<float>() : 1.0f),
            ImGuiColorEditFlags_NoTooltip,
            ImVec2(34.0f * mainScale_, 16.0f * mainScale_));
        ImNodes::EndStaticAttribute();
    }

    for (const render::RenderGraphField& field : reflection.fields()) {
        if (field.visibility == render::RenderGraphFieldVisibility::Output) {
            ImNodes::BeginOutputAttribute(graphOutputAttributeId(node, outputIndex++));
            const float textWidth = ImGui::CalcTextSize(field.name.c_str()).x;
            ImGui::Indent(std::max(90.0f * mainScale_ - textWidth, 0.0f));
            ImGui::TextUnformatted(field.name.c_str());
            ImNodes::EndOutputAttribute();
        }
    }

    ImNodes::EndNode();
}

void EditorApplication::drawRenderGraphPanel()
{
    ImGui::Begin("Render Graph");
    drawRenderGraphMenuBar();
    ImGui::Separator();

    const float paletteWidth = std::max(170.0f * mainScale_, 150.0f);
    ImGui::BeginChild("RenderGraphPassPalette", ImVec2(paletteWidth, 0.0f), true);
    ImGui::TextUnformatted("Passes");
    ImGui::Separator();
    for (const render::RenderGraphPassInfo& passInfo : render::listRenderGraphPassTypes()) {
        if (ImGui::Button(passInfo.type.c_str(), ImVec2(-1.0f, 0.0f))) {
            const std::string nodeName = makeUniqueNodeName(renderGraph_, passInfo.type);
            render::RenderGraphProperties properties = defaultPropertiesForPass(passInfo.type);
            renderGraph_.addNode(
                passInfo.type,
                nodeName,
                std::move(properties),
                60.0f * mainScale_,
                120.0f * mainScale_);
            graphEditorPositionsInitialized_ = false;
            viewportPreviewValid_ = false;
        }
        if (!passInfo.description.empty() && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", passInfo.description.c_str());
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();
    ImGui::BeginChild("RenderGraphCanvas", ImVec2(0.0f, 0.0f), true);

    struct AttributeInfo {
        std::string fullName;
        render::RenderGraphFieldVisibility visibility = render::RenderGraphFieldVisibility::Output;
    };
    std::unordered_map<int, AttributeInfo> attributes;
    std::unordered_map<std::string, int> inputAttributeIds;
    std::unordered_map<std::string, int> outputAttributeIds;

    for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
        std::unique_ptr<render::RenderGraphPass> pass = render::createRenderGraphPass(node.type);
        if (pass == nullptr) {
            continue;
        }
        const render::RenderPassReflection reflection = pass->reflect(render::RenderGraphCompileContext{});
        uint32_t inputIndex = 0;
        uint32_t outputIndex = 0;
        for (const render::RenderGraphField& field : reflection.fields()) {
            const std::string fullName = render::makeRenderGraphFieldName(node.name, field.name);
            if (field.visibility == render::RenderGraphFieldVisibility::Input) {
                const int attrId = graphInputAttributeId(node, inputIndex++);
                attributes.emplace(attrId, AttributeInfo{fullName, field.visibility});
                inputAttributeIds.emplace(fullName, attrId);
            } else {
                const int attrId = graphOutputAttributeId(node, outputIndex++);
                attributes.emplace(attrId, AttributeInfo{fullName, field.visibility});
                outputAttributeIds.emplace(fullName, attrId);
            }
        }
    }

    ImNodes::BeginNodeEditor();
    if (!graphEditorPositionsInitialized_) {
        for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
            ImNodes::SetNodeEditorSpacePos(
                static_cast<int>(node.id),
                ImVec2(node.uiX, node.uiY));
        }
        graphEditorPositionsInitialized_ = true;
    }

    for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
        drawRenderGraphNode(node);
    }

    for (const render::RenderGraphEdge& edge : renderGraph_.edges()) {
        const std::string src = render::makeRenderGraphFieldName(edge.srcPass, edge.srcField);
        const std::string dst = render::makeRenderGraphFieldName(edge.dstPass, edge.dstField);
        const auto srcAttr = outputAttributeIds.find(src);
        const auto dstAttr = inputAttributeIds.find(dst);
        if (srcAttr != outputAttributeIds.end() && dstAttr != inputAttributeIds.end()) {
            ImNodes::Link(static_cast<int>(edge.id), srcAttr->second, dstAttr->second);
        }
    }

    ImNodes::MiniMap(0.16f, ImNodesMiniMapLocation_BottomRight);
    ImNodes::EndNodeEditor();

    for (const render::RenderGraphNode& node : renderGraph_.nodes()) {
        const ImVec2 position = ImNodes::GetNodeEditorSpacePos(static_cast<int>(node.id));
        renderGraph_.setNodePosition(node.id, position.x, position.y);
    }

    int startedAttribute = 0;
    int endedAttribute = 0;
    if (ImNodes::IsLinkCreated(&startedAttribute, &endedAttribute)) {
        const auto started = attributes.find(startedAttribute);
        const auto ended = attributes.find(endedAttribute);
        if (started != attributes.end() && ended != attributes.end()) {
            const AttributeInfo* src = &started->second;
            const AttributeInfo* dst = &ended->second;
            if (src->visibility == render::RenderGraphFieldVisibility::Input &&
                dst->visibility == render::RenderGraphFieldVisibility::Output) {
                std::swap(src, dst);
            }
            if (src->visibility == render::RenderGraphFieldVisibility::Output &&
                dst->visibility == render::RenderGraphFieldVisibility::Input) {
                if (renderGraph_.addEdge(src->fullName, dst->fullName) == nullptr) {
                    renderGraphStatus_ = "Link already exists or endpoint is invalid";
                } else {
                    viewportPreviewValid_ = false;
                }
            }
        }
    }

    int destroyedLink = 0;
    if (ImNodes::IsLinkDestroyed(&destroyedLink)) {
        renderGraph_.removeEdge(static_cast<uint32_t>(destroyedLink));
        viewportPreviewValid_ = false;
    }

    const int selectedNodeCount = ImNodes::NumSelectedNodes();
    if (selectedNodeCount > 0) {
        std::vector<int> selectedNodes(static_cast<size_t>(selectedNodeCount));
        ImNodes::GetSelectedNodes(selectedNodes.data());
        selectedGraphNodeId_ = selectedNodes.front();
    }
    const int selectedLinkCount = ImNodes::NumSelectedLinks();
    if (selectedLinkCount > 0) {
        std::vector<int> selectedLinks(static_cast<size_t>(selectedLinkCount));
        ImNodes::GetSelectedLinks(selectedLinks.data());
        selectedGraphLinkId_ = selectedLinks.front();
    }

    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        ImGui::IsKeyPressed(ImGuiKey_Delete)) {
        if (selectedGraphLinkId_ >= 0 && renderGraph_.removeEdge(static_cast<uint32_t>(selectedGraphLinkId_))) {
            selectedGraphLinkId_ = -1;
            viewportPreviewValid_ = false;
        } else if (selectedGraphNodeId_ >= 0 &&
            renderGraph_.removeNode(static_cast<uint32_t>(selectedGraphNodeId_))) {
            selectedGraphNodeId_ = -1;
            viewportPreviewValid_ = false;
        }
    }

    if (ImGui::Button("Mark Selected color Output")) {
        const render::RenderGraphNode* node = renderGraph_.findNode(static_cast<uint32_t>(selectedGraphNodeId_));
        if (node != nullptr) {
            renderGraph_.clearOutputs();
            renderGraph_.markOutput(render::makeRenderGraphFieldName(node->name, "color"));
            viewportPreviewValid_ = false;
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Current output: %s", renderGraph_.firstOutputName().c_str());

    ImGui::EndChild();
    ImGui::End();
}

void EditorApplication::drawRenderGraphInspector()
{
    ImGui::Begin("Inspector");
    ImGui::TextUnformatted("Render Graph Selection");
    ImGui::Separator();

    render::RenderGraphNode* node = selectedGraphNodeId_ >= 0
        ? renderGraph_.findNode(static_cast<uint32_t>(selectedGraphNodeId_))
        : nullptr;
    if (node == nullptr) {
        ImGui::TextUnformatted("No graph node selected");
        ImGui::End();
        return;
    }

    static int editingNodeId = -1;
    if (editingNodeId != static_cast<int>(node->id)) {
        std::memset(graphNodeNameBuffer_, 0, sizeof(graphNodeNameBuffer_));
        const size_t copySize = std::min(node->name.size(), sizeof(graphNodeNameBuffer_) - 1);
        std::memcpy(graphNodeNameBuffer_, node->name.data(), copySize);
        editingNodeId = static_cast<int>(node->id);
    }

    ImGui::Text("Type: %s", node->type.c_str());
    ImGui::InputText("Name", graphNodeNameBuffer_, sizeof(graphNodeNameBuffer_));
    if (ImGui::IsItemDeactivatedAfterEdit() && std::strlen(graphNodeNameBuffer_) > 0) {
        if (!renderGraph_.renameNode(node->id, graphNodeNameBuffer_)) {
            renderGraphStatus_ = "Node rename failed";
        } else {
            viewportPreviewValid_ = false;
        }
    }

    if (node->type == "ClearColorPass") {
        render::RenderGraphProperties properties = node->properties;
        if (!properties.contains("color") || !properties["color"].is_array() || properties["color"].size() < 4) {
            properties["color"] = {0.04f, 0.06f, 0.09f, 1.0f};
        }
        float color[4] = {
            properties["color"][0].get<float>(),
            properties["color"][1].get<float>(),
            properties["color"][2].get<float>(),
            properties["color"][3].get<float>(),
        };
        if (ImGui::ColorEdit4("Color", color)) {
            properties["color"] = {color[0], color[1], color[2], color[3]};
            renderGraph_.setNodeProperties(node->id, std::move(properties));
            viewportPreviewValid_ = false;
        }
    } else if (!node->properties.empty()) {
        const std::string propertiesText = node->properties.dump(2);
        ImGui::TextWrapped("%s", propertiesText.c_str());
    }

    ImGui::Separator();
    if (ImGui::Button("Delete Node")) {
        renderGraph_.removeNode(node->id);
        selectedGraphNodeId_ = -1;
        viewportPreviewValid_ = false;
    }

    ImGui::End();
}

} // namespace metallic
