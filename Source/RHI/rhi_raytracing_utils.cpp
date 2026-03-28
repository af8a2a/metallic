#include "rhi_raytracing_utils.h"

#ifdef __APPLE__

#include "metal_raytracing_utils.h"

#include <vector>

namespace {

MetalRayTracingInstanceDesc toMetalInstanceDesc(const RhiRayTracingInstanceDesc& instance) {
    MetalRayTracingInstanceDesc metalInstance{};
    for (uint32_t index = 0; index < 12; ++index) {
        metalInstance.transform[index] = instance.transform[index];
    }
    metalInstance.accelerationStructureIndex = instance.accelerationStructureIndex;
    metalInstance.mask = instance.mask;
    metalInstance.opaque = instance.opaque;
    return metalInstance;
}

std::vector<const void*> toNativeAccelerationStructures(
    const RhiAccelerationStructure* const* referencedAccelerationStructures,
    uint32_t referencedAccelerationStructureCount) {
    std::vector<const void*> nativeHandles(referencedAccelerationStructureCount, nullptr);
    for (uint32_t index = 0; index < referencedAccelerationStructureCount; ++index) {
        if (referencedAccelerationStructures[index]) {
            nativeHandles[index] = referencedAccelerationStructures[index]->nativeHandle();
        }
    }
    return nativeHandles;
}

} // namespace

bool rhiBuildBottomLevelAccelerationStructure(const RhiDevice& device,
                                              const RhiCommandQueue& commandQueue,
                                              const RhiBuffer& positionBuffer,
                                              uint32_t positionStride,
                                              const RhiBuffer& indexBuffer,
                                              const RhiRayTracingGeometryRange* geometryRanges,
                                              uint32_t geometryCount,
                                              RhiAccelerationStructureHandle& outAccelerationStructure,
                                              std::string& errorMessage) {
    std::vector<MetalRayTracingGeometryRange> metalRanges(geometryCount);
    for (uint32_t index = 0; index < geometryCount; ++index) {
        metalRanges[index].indexOffset = geometryRanges[index].indexOffset;
        metalRanges[index].indexCount = geometryRanges[index].indexCount;
    }

    void* nativeAccelerationStructure = nullptr;
    if (!metalBuildBottomLevelAccelerationStructure(device.nativeHandle(),
                                                    commandQueue.nativeHandle(),
                                                    positionBuffer.nativeHandle(),
                                                    positionStride,
                                                    indexBuffer.nativeHandle(),
                                                    metalRanges.empty() ? nullptr : metalRanges.data(),
                                                    geometryCount,
                                                    nativeAccelerationStructure,
                                                    errorMessage)) {
        return false;
    }

    outAccelerationStructure.setNativeHandle(nativeAccelerationStructure);
    return true;
}

bool rhiBuildTopLevelAccelerationStructure(const RhiDevice& device,
                                           const RhiCommandQueue& commandQueue,
                                           const RhiAccelerationStructure* const* referencedAccelerationStructures,
                                           uint32_t referencedAccelerationStructureCount,
                                           const RhiRayTracingInstanceDesc* instances,
                                           uint32_t instanceCount,
                                           RhiAccelerationStructureHandle& outAccelerationStructure,
                                           RhiBufferHandle& outInstanceDescriptorBuffer,
                                           RhiBufferHandle& outScratchBuffer,
                                           std::string& errorMessage) {
    std::vector<const void*> nativeHandles = toNativeAccelerationStructures(
        referencedAccelerationStructures,
        referencedAccelerationStructureCount);
    std::vector<MetalRayTracingInstanceDesc> metalInstances(instanceCount);
    for (uint32_t index = 0; index < instanceCount; ++index) {
        metalInstances[index] = toMetalInstanceDesc(instances[index]);
    }

    void* nativeAccelerationStructure = nullptr;
    void* nativeInstanceDescriptorBuffer = nullptr;
    void* nativeScratchBuffer = nullptr;
    if (!metalBuildTopLevelAccelerationStructure(device.nativeHandle(),
                                                 commandQueue.nativeHandle(),
                                                 nativeHandles.empty() ? nullptr : nativeHandles.data(),
                                                 referencedAccelerationStructureCount,
                                                 metalInstances.empty() ? nullptr : metalInstances.data(),
                                                 instanceCount,
                                                 nativeAccelerationStructure,
                                                 nativeInstanceDescriptorBuffer,
                                                 nativeScratchBuffer,
                                                 errorMessage)) {
        return false;
    }

    outAccelerationStructure.setNativeHandle(nativeAccelerationStructure);
    outInstanceDescriptorBuffer.setNativeHandle(nativeInstanceDescriptorBuffer);
    outScratchBuffer.setNativeHandle(nativeScratchBuffer);
    return true;
}

bool rhiUpdateTopLevelAccelerationStructure(const RhiNativeCommandBuffer& commandBuffer,
                                            const RhiAccelerationStructure* const* referencedAccelerationStructures,
                                            uint32_t referencedAccelerationStructureCount,
                                            const RhiRayTracingInstanceDesc* instances,
                                            uint32_t instanceCount,
                                            const RhiAccelerationStructure& accelerationStructure,
                                            const RhiBuffer& instanceDescriptorBuffer,
                                            const RhiBuffer& scratchBuffer,
                                            std::string& errorMessage) {
    std::vector<const void*> nativeHandles = toNativeAccelerationStructures(
        referencedAccelerationStructures,
        referencedAccelerationStructureCount);
    std::vector<MetalRayTracingInstanceDesc> metalInstances(instanceCount);
    for (uint32_t index = 0; index < instanceCount; ++index) {
        metalInstances[index] = toMetalInstanceDesc(instances[index]);
    }

    return metalUpdateTopLevelAccelerationStructure(commandBuffer.nativeHandle(),
                                                    nativeHandles.empty() ? nullptr : nativeHandles.data(),
                                                    referencedAccelerationStructureCount,
                                                    metalInstances.empty() ? nullptr : metalInstances.data(),
                                                    instanceCount,
                                                    accelerationStructure.nativeHandle(),
                                                    instanceDescriptorBuffer.nativeHandle(),
                                                    scratchBuffer.nativeHandle(),
                                                    errorMessage);
}

#elif defined(_WIN32)

#include "rhi_resource_utils.h"
#include "vulkan_resource_handles.h"

#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

namespace {

struct VulkanRayTracingFunctions {
    PFN_vkCreateAccelerationStructureKHR createAccelerationStructure = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getAccelerationStructureDeviceAddress = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructures = nullptr;
    PFN_vkGetBufferDeviceAddress getBufferDeviceAddress = nullptr;

    bool valid() const {
        return createAccelerationStructure &&
               destroyAccelerationStructure &&
               getBuildSizes &&
               getAccelerationStructureDeviceAddress &&
               cmdBuildAccelerationStructures &&
               getBufferDeviceAddress;
    }
};

VulkanRayTracingFunctions loadRayTracingFunctions(VkDevice device) {
    VulkanRayTracingFunctions functions{};
    functions.createAccelerationStructure =
        reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    functions.destroyAccelerationStructure =
        reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
    functions.getBuildSizes =
        reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
            vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    functions.getAccelerationStructureDeviceAddress =
        reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
    functions.cmdBuildAccelerationStructures =
        reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
            vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
    functions.getBufferDeviceAddress =
        reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
            vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddress"));
    if (!functions.getBufferDeviceAddress) {
        functions.getBufferDeviceAddress =
            reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
                vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR"));
    }
    return functions;
}

bool ensureRayTracingContext(const VulkanResourceContextInfo& context,
                             std::string& errorMessage) {
    if (!context.initialized || context.device == VK_NULL_HANDLE || context.allocator == nullptr) {
        errorMessage = "Vulkan resource context is not initialized";
        return false;
    }
    if (!context.rayTracingEnabled) {
        errorMessage = "Vulkan ray tracing is not enabled on the active device";
        return false;
    }
    return true;
}

VkDeviceAddress queryBufferDeviceAddress(const VulkanRayTracingFunctions& functions,
                                         VkDevice device,
                                         VkBuffer buffer) {
    if (!functions.getBufferDeviceAddress || device == VK_NULL_HANDLE || buffer == VK_NULL_HANDLE) {
        return 0;
    }

    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = buffer;
    return functions.getBufferDeviceAddress(device, &addressInfo);
}

VkDeviceAddress getBufferDeviceAddress(const VulkanRayTracingFunctions& functions,
                                       const RhiBuffer& buffer) {
    auto* resource = getVulkanBufferResource(const_cast<RhiBuffer&>(buffer));
    if (!resource || resource->buffer == VK_NULL_HANDLE) {
        return 0;
    }
    if (resource->deviceAddress == 0) {
        resource->deviceAddress = queryBufferDeviceAddress(functions, resource->device, resource->buffer);
    }
    return resource->deviceAddress;
}

VkDeviceAddress getAccelerationStructureDeviceAddress(const VulkanRayTracingFunctions& functions,
                                                      const RhiAccelerationStructure& accelerationStructure) {
    auto* resource = getVulkanAccelerationStructureResource(&accelerationStructure);
    if (!resource || resource->accelerationStructure == VK_NULL_HANDLE) {
        return 0;
    }
    if (resource->deviceAddress == 0) {
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        addressInfo.accelerationStructure = resource->accelerationStructure;
        resource->deviceAddress =
            functions.getAccelerationStructureDeviceAddress(resource->device, &addressInfo);
    }
    return resource->deviceAddress;
}

bool createBufferHandle(const VulkanResourceContextInfo& context,
                        const VulkanRayTracingFunctions& /*functions*/,
                        VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        bool hostVisible,
                        const char* debugName,
                        RhiBufferHandle& outBuffer,
                        std::string& errorMessage) {
    if (size == 0) {
        errorMessage = "Cannot create a zero-sized Vulkan buffer for ray tracing";
        return false;
    }

    VmaBufferCreateInfo vmaInfo{};
    vmaInfo.device = context.device;
    vmaInfo.allocator = context.allocator;
    vmaInfo.size = size;
    vmaInfo.usage = vulkanEnableBufferDeviceAddress(usage, context.bufferDeviceAddressEnabled);
    vmaInfo.hostVisible = hostVisible;
    vmaInfo.debugName = debugName;

    auto resource = vmaCreateBufferResource(vmaInfo);
    if (!resource) {
        errorMessage = "Failed to create Vulkan ray tracing buffer";
        return false;
    }

    auto* res = new VulkanBufferResource(*resource);
    outBuffer = RhiBufferHandle(res, static_cast<size_t>(size));
    return true;
}

bool createAccelerationStructureHandle(const VulkanResourceContextInfo& context,
                                       const VulkanRayTracingFunctions& functions,
                                       VkAccelerationStructureTypeKHR type,
                                       VkDeviceSize size,
                                       RhiAccelerationStructureHandle& outAccelerationStructure,
                                       std::string& errorMessage) {
    if (size == 0) {
        errorMessage = "Cannot create a zero-sized Vulkan acceleration structure";
        return false;
    }

    auto* resource = new VulkanAccelerationStructureResource{};
    resource->device = context.device;
    resource->allocator = context.allocator;

    VmaBufferCreateInfo vmaInfo{};
    vmaInfo.device = context.device;
    vmaInfo.allocator = context.allocator;
    vmaInfo.size = size;
    vmaInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    vmaInfo.usage = vulkanEnableBufferDeviceAddress(vmaInfo.usage, context.bufferDeviceAddressEnabled);

    auto bufferResource = vmaCreateBufferResource(vmaInfo);
    if (!bufferResource) {
        errorMessage = "Failed to create Vulkan acceleration structure storage buffer";
        delete resource;
        return false;
    }

    resource->buffer = bufferResource->buffer;
    resource->allocation = bufferResource->allocation;
    resource->deviceAddress = bufferResource->deviceAddress;

    VkAccelerationStructureCreateInfoKHR createInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    createInfo.buffer = resource->buffer;
    createInfo.size = size;
    createInfo.type = type;

    const VkResult createResult =
        functions.createAccelerationStructure(context.device,
                                              &createInfo,
                                              nullptr,
                                              &resource->accelerationStructure);
    if (createResult != VK_SUCCESS) {
        errorMessage = "Failed to create Vulkan acceleration structure (VkResult: " +
            std::to_string(createResult) + ")";
        VulkanBufferResource asBuf{};
        asBuf.buffer = resource->buffer;
        asBuf.allocation = resource->allocation;
        asBuf.allocator = resource->allocator;
        vmaDestroyBufferResource(asBuf);
        delete resource;
        return false;
    }

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = resource->accelerationStructure;
    resource->deviceAddress =
        functions.getAccelerationStructureDeviceAddress(context.device, &addressInfo);

    outAccelerationStructure = RhiAccelerationStructureHandle(resource);
    return true;
}

template <typename RecordFn>
bool submitImmediateBuild(const VulkanResourceContextInfo& context,
                          const RhiCommandQueue& commandQueue,
                          RecordFn&& recordFn,
                          std::string& errorMessage) {
    VkQueue queue = static_cast<VkQueue>(commandQueue.nativeHandle());
    if (queue == VK_NULL_HANDLE) {
        queue = context.graphicsQueue;
    }
    if (queue == VK_NULL_HANDLE) {
        errorMessage = "Missing Vulkan graphics queue for acceleration structure build";
        return false;
    }

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = context.graphicsQueueFamily;
    VkResult result = vkCreateCommandPool(context.device, &poolInfo, nullptr, &commandPool);
    if (result != VK_SUCCESS) {
        errorMessage = "Failed to create Vulkan ray tracing command pool (VkResult: " +
            std::to_string(result) + ")";
        return false;
    }

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocateInfo.commandPool = commandPool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(context.device, &allocateInfo, &commandBuffer);
    if (result != VK_SUCCESS) {
        errorMessage = "Failed to allocate Vulkan ray tracing command buffer (VkResult: " +
            std::to_string(result) + ")";
        vkDestroyCommandPool(context.device, commandPool, nullptr);
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vulkanBeginCommandBufferHooked(commandBuffer, &beginInfo);
    if (result != VK_SUCCESS) {
        errorMessage = "Failed to begin Vulkan ray tracing command buffer (VkResult: " +
            std::to_string(result) + ")";
        vkDestroyCommandPool(context.device, commandPool, nullptr);
        return false;
    }

    recordFn(commandBuffer);

    result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        errorMessage = "Failed to end Vulkan ray tracing command buffer (VkResult: " +
            std::to_string(result) + ")";
        vkDestroyCommandPool(context.device, commandPool, nullptr);
        return false;
    }

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result == VK_SUCCESS) {
        result = vkQueueWaitIdle(queue);
    }
    vkDestroyCommandPool(context.device, commandPool, nullptr);

    if (result != VK_SUCCESS) {
        errorMessage = "Failed to submit Vulkan ray tracing build work (VkResult: " +
            std::to_string(result) + ")";
        return false;
    }

    return true;
}

VkTransformMatrixKHR toVkTransformMatrix(const float* transform) {
    VkTransformMatrixKHR matrix{};
    matrix.matrix[0][0] = transform[0];
    matrix.matrix[0][1] = transform[3];
    matrix.matrix[0][2] = transform[6];
    matrix.matrix[0][3] = transform[9];
    matrix.matrix[1][0] = transform[1];
    matrix.matrix[1][1] = transform[4];
    matrix.matrix[1][2] = transform[7];
    matrix.matrix[1][3] = transform[10];
    matrix.matrix[2][0] = transform[2];
    matrix.matrix[2][1] = transform[5];
    matrix.matrix[2][2] = transform[8];
    matrix.matrix[2][3] = transform[11];
    return matrix;
}

bool buildInstanceDescriptors(const VulkanRayTracingFunctions& functions,
                              const RhiAccelerationStructure* const* referencedAccelerationStructures,
                              uint32_t referencedAccelerationStructureCount,
                              const RhiRayTracingInstanceDesc* instances,
                              uint32_t instanceCount,
                              std::vector<VkAccelerationStructureInstanceKHR>& outInstances,
                              std::string& errorMessage) {
    outInstances.resize(instanceCount);
    for (uint32_t index = 0; index < instanceCount; ++index) {
        const RhiRayTracingInstanceDesc& instance = instances[index];
        if (instance.accelerationStructureIndex >= referencedAccelerationStructureCount) {
            errorMessage = "TLAS instance references an invalid BLAS index";
            return false;
        }

        const RhiAccelerationStructure* referencedAccelerationStructure =
            referencedAccelerationStructures[instance.accelerationStructureIndex];
        if (!referencedAccelerationStructure || !referencedAccelerationStructure->nativeHandle()) {
            errorMessage = "TLAS instance references a null BLAS handle";
            return false;
        }

        const VkDeviceAddress accelerationStructureAddress =
            getAccelerationStructureDeviceAddress(functions, *referencedAccelerationStructure);
        if (accelerationStructureAddress == 0) {
            errorMessage = "Failed to resolve a BLAS device address for TLAS build";
            return false;
        }

        VkAccelerationStructureInstanceKHR vkInstance{};
        vkInstance.transform = toVkTransformMatrix(instance.transform);
        vkInstance.instanceCustomIndex = index;
        vkInstance.mask = static_cast<uint8_t>(instance.mask & 0xFF);
        vkInstance.instanceShaderBindingTableRecordOffset = 0;
        vkInstance.flags = instance.opaque ? VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR : 0;
        vkInstance.accelerationStructureReference = accelerationStructureAddress;
        outInstances[index] = vkInstance;
    }

    return true;
}

bool uploadInstanceDescriptors(const VulkanResourceContextInfo& context,
                               const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                               VulkanBufferResource& instanceBuffer,
                               std::string& errorMessage) {
    const size_t uploadSize =
        instances.size() * sizeof(VkAccelerationStructureInstanceKHR);
    if (uploadSize == 0 || instanceBuffer.mappedData == nullptr || instanceBuffer.allocation == nullptr) {
        errorMessage = "Invalid Vulkan TLAS instance buffer";
        return false;
    }

    std::memcpy(instanceBuffer.mappedData, instances.data(), uploadSize);
    const VkResult flushResult =
        vmaFlushAllocation(context.allocator, instanceBuffer.allocation, 0, uploadSize);
    if (flushResult != VK_SUCCESS) {
        errorMessage = "Failed to flush Vulkan TLAS instance buffer (VkResult: " +
            std::to_string(flushResult) + ")";
        return false;
    }

    return true;
}

} // namespace

bool rhiBuildBottomLevelAccelerationStructure(const RhiDevice& /*device*/,
                                              const RhiCommandQueue& commandQueue,
                                              const RhiBuffer& positionBuffer,
                                              uint32_t positionStride,
                                              const RhiBuffer& indexBuffer,
                                              const RhiRayTracingGeometryRange* geometryRanges,
                                              uint32_t geometryCount,
                                              RhiAccelerationStructureHandle& outAccelerationStructure,
                                              std::string& errorMessage) {
    const VulkanResourceContextInfo& context = vulkanGetResourceContext();
    if (!ensureRayTracingContext(context, errorMessage)) {
        return false;
    }

    const VulkanRayTracingFunctions functions = loadRayTracingFunctions(context.device);
    if (!functions.valid()) {
        errorMessage = "Failed to load Vulkan ray tracing function pointers";
        return false;
    }

    if (!geometryRanges || geometryCount == 0) {
        errorMessage = "BLAS build requires at least one geometry range";
        return false;
    }

    const VkBuffer vertexBuffer = getVulkanBufferHandle(&positionBuffer);
    const VkBuffer triangleIndexBuffer = getVulkanBufferHandle(&indexBuffer);
    const VkDeviceAddress vertexAddress = getBufferDeviceAddress(functions, positionBuffer);
    const VkDeviceAddress indexAddress = getBufferDeviceAddress(functions, indexBuffer);
    if (vertexBuffer == VK_NULL_HANDLE || triangleIndexBuffer == VK_NULL_HANDLE ||
        vertexAddress == 0 || indexAddress == 0) {
        errorMessage = "BLAS build requires Vulkan buffers with device addresses";
        return false;
    }

    if (positionStride == 0 || positionBuffer.size() < positionStride) {
        errorMessage = "Invalid position buffer stride for BLAS build";
        return false;
    }

    const uint32_t maxVertex =
        static_cast<uint32_t>(positionBuffer.size() / positionStride);
    if (maxVertex == 0) {
        errorMessage = "Position buffer does not contain any vertices for BLAS build";
        return false;
    }

    std::vector<VkAccelerationStructureGeometryKHR> geometries(geometryCount);
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRanges(geometryCount);
    std::vector<uint32_t> primitiveCounts(geometryCount);
    for (uint32_t index = 0; index < geometryCount; ++index) {
        const RhiRayTracingGeometryRange& range = geometryRanges[index];
        if (range.indexCount == 0 || (range.indexCount % 3) != 0) {
            errorMessage = "BLAS geometry range index counts must be non-zero multiples of 3";
            return false;
        }

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = vertexAddress;
        triangles.vertexStride = positionStride;
        triangles.maxVertex = maxVertex - 1;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = indexAddress;

        VkAccelerationStructureGeometryKHR geometry{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometry.geometry.triangles = triangles;
        geometries[index] = geometry;

        VkAccelerationStructureBuildRangeInfoKHR buildRange{};
        buildRange.primitiveCount = range.indexCount / 3;
        buildRange.primitiveOffset = range.indexOffset * sizeof(uint32_t);
        buildRange.firstVertex = 0;
        buildRange.transformOffset = 0;
        buildRanges[index] = buildRange;
        primitiveCounts[index] = buildRange.primitiveCount;
    }

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = geometryCount;
    buildInfo.pGeometries = geometries.data();

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    functions.getBuildSizes(context.device,
                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                            &buildInfo,
                            primitiveCounts.data(),
                            &sizeInfo);

    RhiAccelerationStructureHandle accelerationStructure;
    if (!createAccelerationStructureHandle(context,
                                           functions,
                                           VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                           sizeInfo.accelerationStructureSize,
                                           accelerationStructure,
                                           errorMessage)) {
        return false;
    }

    RhiBufferHandle scratchBuffer;
    if (!createBufferHandle(context,
                            functions,
                            sizeInfo.buildScratchSize,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            false,
                            "Vulkan BLAS Scratch",
                            scratchBuffer,
                            errorMessage)) {
        rhiReleaseHandle(accelerationStructure);
        return false;
    }

    const VkDeviceAddress scratchAddress = getBufferDeviceAddress(functions, scratchBuffer);
    if (scratchAddress == 0) {
        errorMessage = "Failed to get a Vulkan scratch buffer device address for BLAS build";
        rhiReleaseHandle(scratchBuffer);
        rhiReleaseHandle(accelerationStructure);
        return false;
    }

    buildInfo.dstAccelerationStructure = getVulkanAccelerationStructureHandle(&accelerationStructure);
    buildInfo.scratchData.deviceAddress = scratchAddress;

    const VkAccelerationStructureBuildRangeInfoKHR* buildRangePointers[] = {buildRanges.data()};
    const bool buildSucceeded = submitImmediateBuild(
        context,
        commandQueue,
        [&](VkCommandBuffer commandBuffer) {
            functions.cmdBuildAccelerationStructures(commandBuffer,
                                                     1,
                                                     &buildInfo,
                                                     buildRangePointers);
        },
        errorMessage);

    rhiReleaseHandle(scratchBuffer);
    if (!buildSucceeded) {
        rhiReleaseHandle(accelerationStructure);
        return false;
    }

    outAccelerationStructure = accelerationStructure;
    return true;
}

bool rhiBuildTopLevelAccelerationStructure(const RhiDevice& /*device*/,
                                           const RhiCommandQueue& commandQueue,
                                           const RhiAccelerationStructure* const* referencedAccelerationStructures,
                                           uint32_t referencedAccelerationStructureCount,
                                           const RhiRayTracingInstanceDesc* instances,
                                           uint32_t instanceCount,
                                           RhiAccelerationStructureHandle& outAccelerationStructure,
                                           RhiBufferHandle& outInstanceDescriptorBuffer,
                                           RhiBufferHandle& outScratchBuffer,
                                           std::string& errorMessage) {
    const VulkanResourceContextInfo& context = vulkanGetResourceContext();
    if (!ensureRayTracingContext(context, errorMessage)) {
        return false;
    }

    const VulkanRayTracingFunctions functions = loadRayTracingFunctions(context.device);
    if (!functions.valid()) {
        errorMessage = "Failed to load Vulkan ray tracing function pointers";
        return false;
    }

    if (!referencedAccelerationStructures || referencedAccelerationStructureCount == 0) {
        errorMessage = "TLAS build requires at least one referenced BLAS";
        return false;
    }
    if (!instances || instanceCount == 0) {
        errorMessage = "TLAS build requires at least one instance";
        return false;
    }

    std::vector<VkAccelerationStructureInstanceKHR> vkInstances;
    if (!buildInstanceDescriptors(functions,
                                  referencedAccelerationStructures,
                                  referencedAccelerationStructureCount,
                                  instances,
                                  instanceCount,
                                  vkInstances,
                                  errorMessage)) {
        return false;
    }

    RhiBufferHandle instanceDescriptorBuffer;
    const VkDeviceSize instanceBufferSize =
        static_cast<VkDeviceSize>(vkInstances.size() * sizeof(VkAccelerationStructureInstanceKHR));
    if (!createBufferHandle(context,
                            functions,
                            instanceBufferSize,
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            true,
                            "Vulkan TLAS Instances",
                            instanceDescriptorBuffer,
                            errorMessage)) {
        return false;
    }

    auto* instanceBufferResource = getVulkanBufferResource(instanceDescriptorBuffer);
    if (!instanceBufferResource ||
        !uploadInstanceDescriptors(context, vkInstances, *instanceBufferResource, errorMessage)) {
        rhiReleaseHandle(instanceDescriptorBuffer);
        return false;
    }

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = getBufferDeviceAddress(functions, instanceDescriptorBuffer);

    VkAccelerationStructureGeometryKHR geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    const uint32_t primitiveCount = instanceCount;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    functions.getBuildSizes(context.device,
                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                            &buildInfo,
                            &primitiveCount,
                            &sizeInfo);

    RhiAccelerationStructureHandle accelerationStructure;
    if (!createAccelerationStructureHandle(context,
                                           functions,
                                           VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                           sizeInfo.accelerationStructureSize,
                                           accelerationStructure,
                                           errorMessage)) {
        rhiReleaseHandle(instanceDescriptorBuffer);
        return false;
    }

    RhiBufferHandle scratchBuffer;
    const VkDeviceSize scratchSize =
        std::max(sizeInfo.buildScratchSize, sizeInfo.updateScratchSize);
    if (!createBufferHandle(context,
                            functions,
                            scratchSize,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            false,
                            "Vulkan TLAS Scratch",
                            scratchBuffer,
                            errorMessage)) {
        rhiReleaseHandle(accelerationStructure);
        rhiReleaseHandle(instanceDescriptorBuffer);
        return false;
    }

    buildInfo.dstAccelerationStructure = getVulkanAccelerationStructureHandle(&accelerationStructure);
    buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(functions, scratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
    buildRangeInfo.primitiveCount = primitiveCount;
    buildRangeInfo.primitiveOffset = 0;
    buildRangeInfo.firstVertex = 0;
    buildRangeInfo.transformOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* buildRangePointers[] = {&buildRangeInfo};

    const bool buildSucceeded = submitImmediateBuild(
        context,
        commandQueue,
        [&](VkCommandBuffer commandBuffer) {
            functions.cmdBuildAccelerationStructures(commandBuffer,
                                                     1,
                                                     &buildInfo,
                                                     buildRangePointers);
        },
        errorMessage);

    if (!buildSucceeded) {
        rhiReleaseHandle(scratchBuffer);
        rhiReleaseHandle(accelerationStructure);
        rhiReleaseHandle(instanceDescriptorBuffer);
        return false;
    }

    outAccelerationStructure = accelerationStructure;
    outInstanceDescriptorBuffer = instanceDescriptorBuffer;
    outScratchBuffer = scratchBuffer;
    return true;
}

bool rhiUpdateTopLevelAccelerationStructure(const RhiNativeCommandBuffer& commandBuffer,
                                            const RhiAccelerationStructure* const* referencedAccelerationStructures,
                                            uint32_t referencedAccelerationStructureCount,
                                            const RhiRayTracingInstanceDesc* instances,
                                            uint32_t instanceCount,
                                            const RhiAccelerationStructure& accelerationStructure,
                                            const RhiBuffer& instanceDescriptorBuffer,
                                            const RhiBuffer& scratchBuffer,
                                            std::string& errorMessage) {
    const VulkanResourceContextInfo& context = vulkanGetResourceContext();
    if (!ensureRayTracingContext(context, errorMessage)) {
        return false;
    }

    const VulkanRayTracingFunctions functions = loadRayTracingFunctions(context.device);
    if (!functions.valid()) {
        errorMessage = "Failed to load Vulkan ray tracing function pointers";
        return false;
    }

    if (!referencedAccelerationStructures || referencedAccelerationStructureCount == 0) {
        errorMessage = "TLAS update requires referenced BLAS handles";
        return false;
    }
    if (!instances || instanceCount == 0) {
        errorMessage = "TLAS update requires at least one instance";
        return false;
    }

    auto* accelerationStructureResource =
        getVulkanAccelerationStructureResource(&accelerationStructure);
    auto* instanceBufferResource =
        getVulkanBufferResource(const_cast<RhiBuffer&>(instanceDescriptorBuffer));
    auto* scratchBufferResource = getVulkanBufferResource(scratchBuffer);
    if (!accelerationStructureResource ||
        !instanceBufferResource ||
        !scratchBufferResource ||
        accelerationStructureResource->accelerationStructure == VK_NULL_HANDLE) {
        errorMessage = "Invalid Vulkan TLAS resources for update";
        return false;
    }

    std::vector<VkAccelerationStructureInstanceKHR> vkInstances;
    if (!buildInstanceDescriptors(functions,
                                  referencedAccelerationStructures,
                                  referencedAccelerationStructureCount,
                                  instances,
                                  instanceCount,
                                  vkInstances,
                                  errorMessage)) {
        return false;
    }

    if (!uploadInstanceDescriptors(context, vkInstances, *instanceBufferResource, errorMessage)) {
        return false;
    }

    VkAccelerationStructureGeometryInstancesDataKHR instancesData{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instancesData.arrayOfPointers = VK_FALSE;
    instancesData.data.deviceAddress = getBufferDeviceAddress(functions, instanceDescriptorBuffer);

    VkAccelerationStructureGeometryKHR geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.srcAccelerationStructure = accelerationStructureResource->accelerationStructure;
    buildInfo.dstAccelerationStructure = accelerationStructureResource->accelerationStructure;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(functions, scratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
    buildRangeInfo.primitiveCount = instanceCount;
    buildRangeInfo.primitiveOffset = 0;
    buildRangeInfo.firstVertex = 0;
    buildRangeInfo.transformOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* buildRangePointers[] = {&buildRangeInfo};

    VkCommandBuffer vkCommandBuffer = static_cast<VkCommandBuffer>(commandBuffer.nativeHandle());
    if (vkCommandBuffer == VK_NULL_HANDLE) {
        errorMessage = "Invalid Vulkan command buffer for TLAS update";
        return false;
    }

    VkMemoryBarrier2 hostWriteBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    hostWriteBarrier.srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    hostWriteBarrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
    hostWriteBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    hostWriteBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo hostWriteDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    hostWriteDependency.memoryBarrierCount = 1;
    hostWriteDependency.pMemoryBarriers = &hostWriteBarrier;
    vkCmdPipelineBarrier2(vkCommandBuffer, &hostWriteDependency);

    functions.cmdBuildAccelerationStructures(vkCommandBuffer,
                                             1,
                                             &buildInfo,
                                             buildRangePointers);

    VkMemoryBarrier2 buildBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    buildBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    buildBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    buildBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    buildBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                 VK_ACCESS_2_SHADER_READ_BIT;

    VkDependencyInfo buildDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    buildDependency.memoryBarrierCount = 1;
    buildDependency.pMemoryBarriers = &buildBarrier;
    vkCmdPipelineBarrier2(vkCommandBuffer, &buildDependency);

    return true;
}

#endif // __APPLE__ / _WIN32
