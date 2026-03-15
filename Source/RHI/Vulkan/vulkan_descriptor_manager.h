#pragma once

#ifdef _WIN32

#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <vector>

struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;

struct VulkanPipelineResource;
struct VulkanTextureResource;

constexpr uint32_t kMaxBufferBindings = 32;
constexpr uint32_t kMaxTextureBindings = 128;
constexpr uint32_t kMaxSamplerBindings = 16;

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

class VulkanDescriptorManager {
public:
    void init(VkDevice device, VmaAllocator allocator);
    void destroy();
    void resetFrame();

    PendingBufferBinding createTransientUniformBuffer(const void* data, size_t size);

    void flushAndBind(VkCommandBuffer cmd,
                      VkPipelineBindPoint bindPoint,
                      const VulkanPipelineResource& pipeline,
                      const std::array<PendingBufferBinding, kMaxBufferBindings>& buffers,
                      const std::array<PendingTextureBinding, kMaxTextureBindings>& textures,
                      const std::array<PendingSamplerBinding, kMaxSamplerBindings>& samplers);

private:
    struct TransientBuffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        void* allocation = nullptr;
    };

    struct FrameState {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        std::vector<TransientBuffer> transientBuffers;
    };

    void createPools();
    void destroyFrameState(FrameState& frame);
    FrameState& currentFrame();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;
    std::array<FrameState, 2> m_frames{};
    uint32_t m_frameIndex = 1;
};

#endif // _WIN32
