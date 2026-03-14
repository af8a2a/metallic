#pragma once

#ifdef _WIN32

#include "../rhi_backend.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <unordered_map>

// Forward declarations
class VulkanContext;

// Imported texture (wraps swapchain image, non-owning)
class VulkanImportedTexture final : public RhiTexture {
public:
    explicit VulkanImportedTexture(VkImage image = VK_NULL_HANDLE, uint32_t w = 0, uint32_t h = 0)
        : m_image(image), m_width(w), m_height(h) {}

    void setImage(VkImage image, uint32_t w, uint32_t h) {
        m_image = image;
        m_width = w;
        m_height = h;
    }

    VkImage image() const { return m_image; }
    void* nativeHandle() const override { return reinterpret_cast<void*>(m_image); }
    uint32_t width() const override { return m_width; }
    uint32_t height() const override { return m_height; }

private:
    VkImage m_image = VK_NULL_HANDLE;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
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
    VulkanCommandBuffer(VkCommandBuffer commandBuffer, VkDevice device);

    std::unique_ptr<RhiRenderCommandEncoder> beginRenderPass(const RhiRenderPassDesc& desc) override;
    std::unique_ptr<RhiComputeCommandEncoder> beginComputePass(const RhiComputePassDesc& desc) override;
    std::unique_ptr<RhiBlitCommandEncoder> beginBlitPass(const RhiBlitPassDesc& desc) override;

private:
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
};

// Helper functions for type conversions
VkFormat toVkFormat(RhiFormat format);
RhiFormat fromVkFormat(VkFormat format);
VkImageUsageFlags toVkImageUsage(RhiTextureUsage usage);
VkImageLayout toVkImageLayout(RhiTextureUsage usage);

#endif // _WIN32
