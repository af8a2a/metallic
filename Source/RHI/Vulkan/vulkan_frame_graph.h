#pragma once

#ifdef _WIN32

#include "../rhi_backend.h"
#include "vulkan_descriptor_manager.h"
#include "vulkan_resource_state_tracker.h"
#include "vulkan_resource_handles.h"

#include <vulkan/vulkan.h>

// Forward declarations
class VulkanContext;
class VulkanOwnedTexture;

// Imported texture (wraps swapchain image, non-owning)
class VulkanImportedTexture final : public RhiTexture {
public:
    VulkanImportedTexture() = default;
    VulkanImportedTexture(VkImage image, VkImageView imageView, uint32_t w, uint32_t h)
    {
        set(image, imageView, w, h);
    }

    void set(VkImage image,
             VkImageView imageView,
             uint32_t w,
             uint32_t h,
             VkFormat format = VK_FORMAT_UNDEFINED,
             RhiTextureUsage usage = RhiTextureUsage::RenderTarget) {
        m_resource.image = image;
        m_resource.imageView = imageView;
        m_resource.width = w;
        m_resource.height = h;
        m_resource.format = format;
        m_resource.usage = usage;
        m_resource.ownsImage = false;
        m_resource.ownsImageView = false;
        m_resource.refCount = 1;
    }

    VkImage image() const { return m_resource.image; }
    VkImageView imageView() const { return m_resource.imageView; }
    void* nativeHandle() const override { return const_cast<VulkanTextureResource*>(&m_resource); }
    uint32_t width() const override { return m_resource.width; }
    uint32_t height() const override { return m_resource.height; }

private:
    VulkanTextureResource m_resource{};
};

class VulkanTransientPool;

// Frame graph backend for Vulkan
class VulkanFrameGraphBackend final : public RhiFrameGraphBackend {
public:
    VulkanFrameGraphBackend(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator);

    void setTransientPool(VulkanTransientPool* pool) { m_transientPool = pool; }

    std::unique_ptr<RhiTexture> createTexture(const RhiTextureDesc& desc) override;
    std::unique_ptr<RhiBuffer> createBuffer(const RhiBufferDesc& desc) override;

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;
    VulkanTransientPool* m_transientPool = nullptr;
};

// Command buffer abstraction
class VulkanCommandBuffer final : public RhiCommandBuffer {
public:
    VulkanCommandBuffer(VkCommandBuffer commandBuffer, VkDevice device,
                        VulkanDescriptorManager* descriptorManager,
                        VulkanResourceStateTracker* stateTracker,
                        VkCommandBuffer asyncComputeCommandBuffer = VK_NULL_HANDLE);

    std::unique_ptr<RhiRenderCommandEncoder> beginRenderPass(const RhiRenderPassDesc& desc) override;
    std::unique_ptr<RhiComputeCommandEncoder> beginComputePass(const RhiComputePassDesc& desc) override;
    std::unique_ptr<RhiBlitCommandEncoder> beginBlitPass(const RhiBlitPassDesc& desc) override;
    void prepareTextureForSampling(const RhiTexture* texture) override;
    void flushBarriers() override;
    void setNextPassQueueHint(RhiQueueHint hint) override;
    void transitionTexture(const RhiTexture* texture, VkImageLayout layout);

    // Returns true if any work was submitted to the async compute command buffer this frame.
    bool hadAsyncComputeWork() const { return m_hadAsyncComputeWork; }

private:
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkCommandBuffer m_asyncComputeCommandBuffer = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VulkanDescriptorManager* m_descriptorManager = nullptr;
    VulkanResourceStateTracker* m_stateTracker = nullptr;
    RhiQueueHint m_nextPassHint = RhiQueueHint::Auto;
    bool m_hadAsyncComputeWork = false;
};

// Load mesh shader extension functions (call once after device creation)
void vulkanLoadMeshShaderFunctions(VkDevice device);

// Helper functions for type conversions
VkFormat toVkFormat(RhiFormat format);
RhiFormat fromVkFormat(VkFormat format);
VkImageUsageFlags toVkImageUsage(RhiTextureUsage usage);

#endif // _WIN32
