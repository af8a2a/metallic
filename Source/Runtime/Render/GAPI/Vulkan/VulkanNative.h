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

struct ClusterAccelerationStructureBuildSizes {
    uint64_t accelerationStructureSize = 0;
    uint64_t updateScratchSize = 0;
    uint64_t buildScratchSize = 0;
};

struct ClusterAccelerationStructureTriangleBuildSizesDesc {
    VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    uint32_t maxClusterTriangleCount = 0;
    uint32_t maxClusterVertexCount = 0;
    uint32_t maxClusterUniqueGeometryCount = 1;
    uint32_t maxGeometryIndexValue = 0;
    uint32_t minPositionTruncateBitCount = 0;
    uint32_t maxTotalTriangleCount = 0;
    uint32_t maxTotalVertexCount = 0;
    VkFormat vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    uint32_t maxAccelerationStructureCount = 1;
};

struct ClusterAccelerationStructureBottomLevelBuildSizesDesc {
    VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    uint32_t maxClusterCountPerAccelerationStructure = 0;
    uint32_t maxTotalClusterCount = 0;
    uint32_t maxAccelerationStructureCount = 1;
};

struct PartitionedAccelerationStructureBuildSizes {
    uint64_t accelerationStructureSize = 0;
    uint64_t updateScratchSize = 0;
    uint64_t buildScratchSize = 0;
    uint64_t operationInfoSize = 0;
    uint64_t operationCountSize = 0;
    uint64_t instanceWriteInfoSize = 0;
    uint64_t instanceUpdateInfoSize = 0;
    uint64_t partitionWriteInfoSize = 0;
};

struct PartitionedAccelerationStructureBuildSizesDesc {
    VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    uint32_t instanceCount = 0;
    uint32_t partitionCount = 1;
    uint32_t maxInstancePerPartitionCount = 0;
    uint32_t maxInstanceInGlobalPartitionCount = 0;
    uint32_t maxOperationCount = 1;
    bool allowInstanceUpdate = false;
    bool allowPartitionTranslation = false;
};

NativeDevice nativeDevice(Device& device);
NativeQueue nativeQueue(Queue& queue);
NativeBuffer nativeBuffer(Buffer& buffer);
NativeTexture nativeTexture(Texture& texture);
VkCommandBuffer nativeCommandBuffer(CommandBuffer& commandBuffer);
VkFormat nativeSwapchainFormat(Swapchain& swapchain);
VkImageView nativeImageView(TextureView& view);
Result queryClusterAccelerationStructureTriangleBuildSizes(
    Device& device,
    const ClusterAccelerationStructureTriangleBuildSizesDesc& desc,
    ClusterAccelerationStructureBuildSizes& outSizes);
Result queryClusterAccelerationStructureBottomLevelBuildSizes(
    Device& device,
    const ClusterAccelerationStructureBottomLevelBuildSizesDesc& desc,
    ClusterAccelerationStructureBuildSizes& outSizes);
Result queryPartitionedAccelerationStructureBuildSizes(
    Device& device,
    const PartitionedAccelerationStructureBuildSizesDesc& desc,
    PartitionedAccelerationStructureBuildSizes& outSizes);

} // namespace metallic::render::vulkan
