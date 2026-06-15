#pragma once

#include "Runtime/Render/RenderGraph/render_graph.h"
#include "Runtime/Render/GAPI/Vulkan/vulkan_scene_rtx.h"
#include "Runtime/Render/history_resources.h"
#include "Runtime/Scene/scene.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <volk.h>

struct SDL_Window;
struct ImVec2;

namespace metallic {

class EditorApplication {
public:
    int run(bool smokeTest = false, bool waitForGraphicsDebugger = false);

private:
    bool initialize();
    void shutdown();
    void pollEvents();
    void renderFrame();
    void drawDockspace();
    void drawPanels();
    void drawScenePanel();
    void drawEnvironmentControls();
    void drawCameraControls();
    void drawSceneNode(int32_t nodeIndex);
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
    void resetDefaultRenderGraph();
    void saveRenderGraph();
    void loadRenderGraph();
    void loadScene();
    void buildSceneRtx();
    void clearSceneRtx();
    void addRenderGraphNode(std::string type, ImVec2 screenPosition);
    void markRenderGraphOutput(std::string outputName);
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

    SDL_Window* window_ = nullptr;
    std::unique_ptr<render::Device> device_;
    render::Queue* graphicsQueue_ = nullptr;
    std::unique_ptr<render::Swapchain> swapchain_;
    std::vector<std::unique_ptr<render::TextureView>> swapchainImageViews_;
    std::vector<render::ResourceState> swapchainImageStates_;
    std::unique_ptr<render::CommandPool> commandPool_;
    std::unique_ptr<render::CommandBuffer> commandBuffer_;
    std::unique_ptr<render::Fence> frameFence_;
    std::unique_ptr<render::Semaphore> imageAvailableSemaphore_;
    std::unique_ptr<render::Semaphore> renderFinishedSemaphore_;
    std::unique_ptr<render::RenderGraphExecutor> graphExecutor_;
    render::HistoryResourceManager historyResources_;
    std::unique_ptr<render::vulkan::SceneRtxBuilder> sceneRtx_;
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
    bool graphEditorPositionsInitialized_ = false;
    bool viewportCameraDragging_ = false;
    float mainScale_ = 1.0f;
    float clearColor_[4] = {0.07f, 0.08f, 0.10f, 1.0f};
    int selectedGraphNodeId_ = -1;
    int selectedGraphLinkId_ = -1;
    char graphFilePath_[260] = "Pipelines/default.metallic_graph.json";
    char sceneFilePath_[260] = "Asset/meet_mat.glb";
    char graphNodeNameBuffer_[128] = {};
    char graphOutputBuffer_[128] = "Bunny.color";
    std::string renderGraphStatus_;
    std::string sceneStatus_ = "No scene loaded.";
    std::string sceneRtxStatus_ = "RTX AS not built.";
};

} // namespace metallic
