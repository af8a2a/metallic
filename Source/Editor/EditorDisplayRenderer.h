#pragma once

#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include <map>

struct ImDrawList;
struct ImDrawCmd;
struct ImGuiViewport;

namespace metallic {

// Pipelines compatible with ImGui's public Vulkan descriptor/push-constant ABI.
// The backend continues to own buffers, textures, draw submission and windows.
class EditorDisplayRenderer {
public:
    static bool loadBackendFunctions(render::vulkan::NativeDevice device);
    ~EditorDisplayRenderer();
    bool initialize(render::Device& device, VkFormat mainFormat, bool hdr, float paperWhiteNits,
        VkFormat pqOutputFormat = VK_FORMAT_UNDEFINED);
    void shutdown();
    VkPipeline mainPipeline() const { return mainPipeline_; }
    void beginScRgbImage(ImDrawList& list, ImGuiViewport* viewport);
    void encodeHDR10(VkCommandBuffer commands, VkDescriptorSet source, uint32_t width, uint32_t height);

private:
    static void bindImagePipeline(const ImDrawList*, const ImDrawCmd* command);
    VkPipeline createPipeline(VkFormat format, bool hdr, bool scRgbImage, float paperWhiteNits, bool encodePQ = false);

    VkDevice device_ = VK_NULL_HANDLE;
    render::Device* rhiDevice_ = nullptr;
    const VolkDeviceTable* functions_ = nullptr;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayouts_[2]{};
    VkPipeline mainPipeline_ = VK_NULL_HANDLE;
    VkPipeline mainImagePipeline_ = VK_NULL_HANDLE;
    VkPipeline pqPipeline_ = VK_NULL_HANDLE;
    VkFormat pqOutputFormat_ = VK_FORMAT_UNDEFINED;
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
