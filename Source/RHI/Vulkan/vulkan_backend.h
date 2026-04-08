#pragma once

#include "rhi_backend.h"
#include "vulkan_diagnostics.h"

#include <vulkan/vulkan.h>

// Forward declare VmaAllocator to avoid including vk_mem_alloc.h in the header
// (VMA_IMPLEMENTATION must only be defined in one TU)
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;

std::unique_ptr<RhiContext> createVulkanContext(const RhiCreateInfo& createInfo,
                                                std::string& errorMessage);

bool vulkanHasExtension(RhiContext& context, const char* extensionName);

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

// Async compute queue support
VkQueue getVulkanComputeQueue(RhiContext& context);             // nullptr if unavailable
VkCommandBuffer getVulkanCurrentComputeCommandBuffer(RhiContext& context); // nullptr if unavailable

// End async compute command buffer and submit to the dedicated compute queue.
// Returns the timeline semaphore signal value (0 if no timeline semaphore or no compute queue).
uint64_t vulkanScheduleAsyncComputeSubmit(RhiContext& context);
void vulkanEnqueueGraphicsTimelineWait(RhiContext& context,
                                       VkSemaphore semaphore,
                                       uint64_t value,
                                       VkPipelineStageFlags2 stageMask);

// Returns the VkPipelineCache used for all pipeline compilations (VK_NULL_HANDLE if not loaded).
VkPipelineCache getVulkanPipelineCache(RhiContext& context);

struct VulkanPipelineCacheTelemetry {
    uint32_t graphicsPipelinesCompiled = 0;
    uint32_t computePipelinesCompiled = 0;
    double totalCompileMs = 0.0;
};

const VulkanGpuFrameDiagnostics& getVulkanLatestFrameDiagnostics(RhiContext& context);
const VulkanToolingInfo& getVulkanToolingInfo(RhiContext& context);
VulkanPipelineCacheTelemetry getVulkanPipelineCacheTelemetry(RhiContext& context);
bool vulkanIsDeviceLost(RhiContext& context);
const std::string& vulkanDeviceLostMessage(RhiContext& context);
VulkanGpuProfiler* getVulkanGpuProfiler(RhiContext& context);

// Returns VK_EXT_descriptor_buffer properties (zeroed if extension not enabled).
const VkPhysicalDeviceDescriptorBufferPropertiesEXT& getVulkanDescriptorBufferProperties(
    RhiContext& context);

// Transient memory subsystems (Phase 0.6)
class VulkanUploadRing;
class VulkanTransientPool;
class VulkanReadbackHeap;

VulkanUploadRing&    getVulkanUploadRing(RhiContext& context);
VulkanTransientPool& getVulkanTransientPool(RhiContext& context);
VulkanReadbackHeap&  getVulkanReadbackHeap(RhiContext& context);
