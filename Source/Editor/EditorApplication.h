#pragma once

#include "Editor/EditorProfiler.h"
#include "Editor/NvmlMonitor.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanSceneRtx.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Scene/Scene.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <volk.h>

struct SDL_Window;
struct ImVec2;

namespace metallic {

class EditorApplication {
public:
    int run(bool smokeTest = false, bool waitForGraphicsDebugger = false, const char* startupSampleId = nullptr);

private:
    bool initialize();
    void shutdown();
    void pollEvents();
    void renderFrame();
    void drawDockspace();
    void drawPanels();
    void drawScenePanel();
    void drawInspectorPanel();
    void drawStatisticsPanel();
    void drawCameraControls();
    void drawEnvironmentControls();
    void drawSceneNode(int32_t nodeIndex);
    void drawSceneGraphTab();
    void drawSceneListTab();
    void drawSceneListSelectable(const char* label, int32_t index, int32_t type);
    render::RenderGraphNode* activePreviewRenderGraphNode();
    bool drawRuntimeSettingsForNode(
        render::RenderGraphNode& node,
        bool hideCameraSettings,
        bool showEmptyMessage);
    void drawViewportPanel();
    void handleViewportCameraControls(const ImVec2& min, const ImVec2& max);
    void applyRuntimeNodeProperties(uint32_t nodeId, render::RenderGraphProperties properties, const char* status);
    void applyBunnyCameraProperties(render::RenderGraphProperties properties, const char* status);
    void drawRenderGraphEditorWindow();
    void drawRenderGraphPanel();
    void drawRenderGraphSettingsPanel();
    void drawRenderPassesPanel();
    void drawRenderGraphRenderUiPanel();
    void drawRenderGraphNode(const render::RenderGraphNode& node);
    void setupDefaultDockLayout();
    void loadBuiltInSample(const char* sampleId);
    void resetDefaultRenderGraph();
    void saveRenderGraph();
    void loadRenderGraph();
    void chooseSceneFile();
    void chooseEnvironmentFile();
    void loadScene();
    void loadDroppedScene(const std::filesystem::path& path);
    void loadDroppedRenderGraph(const std::filesystem::path& path);
    void addRecentScenePath(const std::filesystem::path& path);
    void applyLoadedSceneToRenderGraph(const std::filesystem::path& path);
    void applyLoadedSceneCamera();
    void applyEnvironmentToRenderGraph(const std::filesystem::path& path);
    void buildSceneRtx();
    void clearSceneRtx();
    void addRenderGraphNode(std::string type, ImVec2 screenPosition);
    void markRenderGraphOutput(std::string outputName);
    void setActivePreviewOutput(std::string outputName);
    bool bindViewportPreviewOutput(std::string_view outputName);
    bool initializeRhi();
    bool createOrResizeSwapchain(uint32_t width, uint32_t height);
    void destroySwapchainResources();
    bool initializeImGuiBackends();
    bool createViewportSampler();
    void destroyViewportDescriptor();
    bool updateViewportPreview(uint32_t width, uint32_t height);
    void destroyViewportTexture();
    bool renderGraphPreview();
    bool renderVulkanFrame();
    int graphInputAttributeId(const render::RenderGraphNode& node, uint32_t fieldIndex) const;
    int graphOutputAttributeId(const render::RenderGraphNode& node, uint32_t fieldIndex) const;

    enum class SceneSelectionType : int32_t {
        None,
        Node,
        Mesh,
        RenderPrimitive,
        Material,
        Camera,
        Light,
        Texture,
        Image,
    };

    struct SceneSelection {
        SceneSelectionType type = SceneSelectionType::None;
        int32_t index = scene::kInvalidSceneIndex;
        int32_t nodeIndex = scene::kInvalidSceneIndex;
        int32_t meshIndex = scene::kInvalidSceneIndex;
        int32_t primitiveIndex = scene::kInvalidSceneIndex;
    };

    SDL_Window* window_ = nullptr;
    std::unique_ptr<render::Device> device_;
    render::Queue* graphicsQueue_ = nullptr;
    std::unique_ptr<render::Swapchain> swapchain_;
    std::vector<std::unique_ptr<render::TextureView>> swapchainImageViews_;
    std::vector<render::ResourceState> swapchainImageStates_;
    std::unique_ptr<render::CommandPool> commandPool_;
    std::unique_ptr<render::CommandBuffer> commandBuffer_;
    std::unique_ptr<render::Fence> frameFence_;
    std::unique_ptr<render::SwapchainSemaphore> imageAvailableSemaphore_;
    std::vector<std::unique_ptr<render::SwapchainSemaphore>> renderFinishedSemaphores_;
    std::unique_ptr<render::RenderGraphExecutor> graphExecutor_;
    render::HistoryResourceManager historyResources_;
    std::unique_ptr<render::vulkan::SceneRtxBuilder> sceneRtx_;
    EditorProfiler profiler_;
    NvmlMonitor nvmlMonitor_;
    render::RenderGraph renderGraph_;
    scene::Scene scene_;
    VkSampler viewportSampler_ = VK_NULL_HANDLE;
    VkDescriptorSet viewportDescriptor_ = VK_NULL_HANDLE;
    uint32_t viewportTextureWidth_ = 0;
    uint32_t viewportTextureHeight_ = 0;
    uint32_t swapchainWidth_ = 0;
    uint32_t swapchainHeight_ = 0;
    uint32_t pendingViewportPreviewWidth_ = 0;
    uint32_t pendingViewportPreviewHeight_ = 0;
    uint32_t viewportResizeStableFrameCount_ = 0;
    uint64_t historyFrameIndex_ = 0;
    bool running_ = true;
    bool smokeTest_ = false;
    bool waitForGraphicsDebugger_ = false;
    bool imguiContextCreated_ = false;
    bool imnodesContextCreated_ = false;
    bool imguiPlatformInitialized_ = false;
    bool imguiRendererInitialized_ = false;
    bool viewportPreviewValid_ = false;
    bool viewportPreviewNeedsRender_ = false;
    bool swapchainOutOfDate_ = false;
    bool dockLayoutInitialized_ = false;
    bool renderGraphEditorOpen_ = false;
    bool profilerOpen_ = true;
    bool nvmlMonitorOpen_ = true;
    bool inspectorOpen_ = true;
    bool statisticsOpen_ = true;
    bool graphEditorPositionsInitialized_ = false;
    int viewportCameraDragButton_ = -1;
    float mainScale_ = 1.0f;
    float clearColor_[4] = {0.07f, 0.08f, 0.10f, 1.0f};
    int selectedGraphNodeId_ = -1;
    int selectedGraphLinkId_ = -1;
    char graphFilePath_[260] = "Pipelines/Samples/pathtracing_abeautiful_game_openpbr.metallic_graph.json";
    char sceneFilePath_[260] = "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf";
    char graphNodeNameBuffer_[128] = {};
    char graphOutputBuffer_[128] = "Bunny.color";
    char previewOutputBuffer_[128] = "Bunny.color";
    std::string activePreviewOutput_ = "Bunny.color";
    std::string renderGraphStatus_;
    std::string sceneStatus_ = "No scene loaded.";
    std::string sceneRtxStatus_ = "RTX AS not built.";
    std::string startupSampleId_;
    SceneSelection sceneSelection_;
    std::vector<std::filesystem::path> recentScenePaths_;
};

} // namespace metallic
