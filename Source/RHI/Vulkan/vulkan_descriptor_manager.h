#pragma once

#ifdef _WIN32

#include <vulkan/vulkan.h>
#include <array>
#include <vector>

struct VulkanTextureResource;

// Maximum binding slots matching Metal's argument-based binding model
constexpr uint32_t kMaxBufferBindings = 32;
constexpr uint32_t kMaxTextureBindings = 128;
constexpr uint32_t kMaxSamplerBindings = 16;

// Descriptor set layout:
//   Set 0: Storage buffers (binding 0..31)
//   Set 1: Sampled images (binding 0..127)
//   Set 2: Samplers (binding 0..15)
constexpr uint32_t kDescriptorSetBuffers  = 0;
constexpr uint32_t kDescriptorSetTextures = 1;
constexpr uint32_t kDescriptorSetSamplers = 2;
constexpr uint32_t kDescriptorSetCount    = 3;

struct PendingBufferBinding {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = VK_WHOLE_SIZE;
    bool dirty = false;
};

struct PendingTextureBinding {
    VulkanTextureResource* resource = nullptr;
    VkImageView imageView = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bool dirty = false;
    bool isStorage = false;
};

struct PendingSamplerBinding {
    VkSampler sampler = VK_NULL_HANDLE;
    bool dirty = false;
};

// Manages descriptor set layouts, pools, and per-draw allocation.
// Initial strategy: allocate a new descriptor set per draw/dispatch from a per-frame pool.
class VulkanDescriptorManager {
public:
    void init(VkDevice device);
    void destroy();

    // Call at the start of each frame to reset the pool
    void resetFrame();

    // Get the shared descriptor set layouts (for pipeline layout creation)
    const VkDescriptorSetLayout* layouts() const { return m_setLayouts.data(); }
    uint32_t layoutCount() const { return kDescriptorSetCount; }
    VkPipelineLayout pipelineLayout() const { return m_pipelineLayout; }

    // Allocate and write descriptor sets from pending bindings, bind them to the command buffer
    void flushAndBind(VkCommandBuffer cmd, VkPipelineBindPoint bindPoint,
                      const std::array<PendingBufferBinding, kMaxBufferBindings>& buffers,
                      const std::array<PendingTextureBinding, kMaxTextureBindings>& textures,
                      const std::array<PendingSamplerBinding, kMaxSamplerBindings>& samplers);

private:
    void createSetLayouts();
    void createPipelineLayout();
    void createPool();

    VkDevice m_device = VK_NULL_HANDLE;
    std::array<VkDescriptorSetLayout, kDescriptorSetCount> m_setLayouts{};
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
};

#endif // _WIN32
