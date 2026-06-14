#pragma once

#include "Runtime/Render/GAPI/rhi.h"

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

NativeDevice nativeDevice(Device& device);
NativeQueue nativeQueue(Queue& queue);
VkCommandBuffer nativeCommandBuffer(CommandBuffer& commandBuffer);
VkFormat nativeSwapchainFormat(Swapchain& swapchain);
VkImageView nativeImageView(TextureView& view);

} // namespace metallic::render::vulkan
