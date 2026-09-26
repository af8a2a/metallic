#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <volk.h>

namespace metallic::render::vulkan {

struct NativeDevice {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t apiVersion = 0;
    bool descriptorHeapEnabled = false;
};

struct NativeQueue {
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t familyIndex = 0;
};

struct NativeBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    uint64_t size = 0;
    VkDevice device = VK_NULL_HANDLE;
};

struct NativePipeline {
    VkDevice device = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

struct NativeGraphicsShaders {
    VkDevice device = VK_NULL_HANDLE;
    VkShaderEXT vertex = VK_NULL_HANDLE;
    VkShaderEXT fragment = VK_NULL_HANDLE;
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
NativePipeline nativePipeline(ComputePipeline& pipeline);
NativePipeline nativePipeline(GraphicsPipeline& pipeline);
NativeGraphicsShaders nativeShaders(GraphicsShaderObjectProgram& program);
NativeTexture nativeTexture(Texture& texture);
VkCommandBuffer nativeCommandBuffer(CommandBuffer& commandBuffer);
VkDevice nativeCommandBufferDevice(CommandBuffer& commandBuffer);
// DGC leaves affected state undefined. Rebind pipeline/shaders, heap and push data afterwards.
void notifyGeneratedCommandsExecution(CommandBuffer& commandBuffer);
// Compatibility name: invalidates all tracked execution state after external commands.
void notifyExternalDescriptorSetBinding(CommandBuffer& commandBuffer);
VkFormat nativeSwapchainFormat(Swapchain& swapchain);
// Exporters use the backend policy, rather than hard-coding an optimal layout.
VkImageLayout nativeImageLayout(TextureView& view, ResourceState usage);
VkImageView nativeImageView(TextureView& view);
} // namespace metallic::render::vulkan
