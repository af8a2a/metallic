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

#endif // __APPLE__
