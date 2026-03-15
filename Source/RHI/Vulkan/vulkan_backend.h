#pragma once

#include "../rhi_backend.h"

#include <vulkan/vulkan.h>

// Forward declare VmaAllocator to avoid including vk_mem_alloc.h in the header
// (VMA_IMPLEMENTATION must only be defined in one TU)
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;

std::unique_ptr<RhiContext> createVulkanContext(const RhiCreateInfo& createInfo,
                                                std::string& errorMessage);

// Access VMA allocator and Vulkan handles from the context (for frame graph backend, resource utils, etc.)
VmaAllocator getVulkanAllocator(RhiContext& context);
VkDevice getVulkanDevice(RhiContext& context);
VkPhysicalDevice getVulkanPhysicalDevice(RhiContext& context);
VkCommandBuffer getVulkanCurrentCommandBuffer(RhiContext& context);
VkQueue getVulkanGraphicsQueue(RhiContext& context);
uint32_t getVulkanGraphicsQueueFamily(RhiContext& context);
VkImage getVulkanCurrentBackbufferImage(RhiContext& context);
VkImageView getVulkanCurrentBackbufferImageView(RhiContext& context);
VkExtent2D getVulkanCurrentBackbufferExtent(RhiContext& context);
VkImageLayout getVulkanCurrentBackbufferLayout(RhiContext& context);
