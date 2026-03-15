#pragma once

#ifdef _WIN32

#include "../rhi_backend.h"
#include "vulkan_descriptor_manager.h"
#include "vulkan_image_tracker.h"
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

    void set(VkImage image, VkImageView imageView, uint32_t w, uint32_t h) {
        m_resource.image = image;
        m_resource.imageView = imageView;
        m_resource.width = w;
        m_resource.height = h;
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

// Frame graph backend for Vulkan
class VulkanFrameGraphBackend final : public RhiFrameGraphBackend {
public:
    VulkanFrameGraphBackend(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator);

    std::unique_ptr<RhiTexture> createTexture(const RhiTextureDesc& desc) override;
    std::unique_ptr<RhiBuffer> createBuffer(const RhiBufferDesc& desc) override;

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;
};

// Command buffer abstraction
class VulkanCommandBuffer final : public RhiCommandBuffer {
public:
    VulkanCommandBuffer(VkCommandBuffer commandBuffer, VkDevice device,
                        VulkanDescriptorManager* descriptorManager,
                        VulkanImageLayoutTracker* imageTracker);

    std::unique_ptr<RhiRenderCommandEncoder> beginRenderPass(const RhiRenderPassDesc& desc) override;
    std::unique_ptr<RhiComputeCommandEncoder> beginComputePass(const RhiComputePassDesc& desc) override;
    std::unique_ptr<RhiBlitCommandEncoder> beginBlitPass(const RhiBlitPassDesc& desc) override;

private:
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VulkanDescriptorManager* m_descriptorManager = nullptr;
    VulkanImageLayoutTracker* m_imageTracker = nullptr;
};

// Load mesh shader extension functions (call once after device creation)
void vulkanLoadMeshShaderFunctions(VkDevice device);

// Helper functions for type conversions
VkFormat toVkFormat(RhiFormat format);
RhiFormat fromVkFormat(VkFormat format);
VkImageUsageFlags toVkImageUsage(RhiTextureUsage usage);

#endif // _WIN32
