#pragma once

#include <volk.h>
#include <map>

struct ImDrawList;
struct ImDrawCmd;
struct ImGuiViewport;

namespace metallic {

// Pipelines compatible with ImGui's public Vulkan descriptor/push-constant ABI.
// The backend continues to own buffers, textures, draw submission and windows.
class EditorDisplayRenderer {
public:
    ~EditorDisplayRenderer();
    bool initialize(VkDevice device, VkFormat mainFormat, bool hdr, float paperWhiteNits);
    void shutdown();
    VkPipeline mainPipeline() const { return mainPipeline_; }
    void beginScRgbImage(ImDrawList& list, ImGuiViewport* viewport);

private:
    static void bindImagePipeline(const ImDrawList*, const ImDrawCmd* command);
    VkPipeline createPipeline(VkFormat format, bool hdr, bool scRgbImage, float paperWhiteNits);

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayouts_[2]{};
    VkPipeline mainPipeline_ = VK_NULL_HANDLE;
    VkPipeline mainImagePipeline_ = VK_NULL_HANDLE;
    std::map<VkFormat, VkPipeline> secondaryImagePipelines_;
    VkFormat mainFormat_ = VK_FORMAT_UNDEFINED;
    bool hdr_ = false;
    float paperWhiteNits_ = 203.0f;
    struct ImageCallbackData {
        EditorDisplayRenderer* renderer;
        ImGuiViewport* viewport;
    };
};

} // namespace metallic
