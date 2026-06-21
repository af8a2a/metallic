#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <volk.h>

namespace metallic::render::vulkan {

struct NativeDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t apiVersion = 0;
};

struct NativeQueue {
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t familyIndex = 0;
};

struct NativeBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    uint64_t size = 0;
};

struct NativeTexture {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t mipCount = 0;
    uint32_t layerCount = 0;
    VkImageCreateFlags flags = 0;
    VkImageUsageFlags usage = 0;
};

NativeDevice nativeDevice(Device& device);
NativeQueue nativeQueue(Queue& queue);
NativeBuffer nativeBuffer(Buffer& buffer);
NativeTexture nativeTexture(Texture& texture);
VkCommandBuffer nativeCommandBuffer(CommandBuffer& commandBuffer);
VkFormat nativeSwapchainFormat(Swapchain& swapchain);
VkImageView nativeImageView(TextureView& view);

} // namespace metallic::render::vulkan
