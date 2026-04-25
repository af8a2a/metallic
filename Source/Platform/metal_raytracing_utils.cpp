#include "metal_raytracing_utils.h"

#include "metal_resource_utils.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <Metal/MTL4AccelerationStructure.hpp>

#include <limits>
#include <vector>

namespace {

MTL::Device* metalDevice(void* handle) {
    return static_cast<MTL::Device*>(handle);
}

MTL4::CommandQueue* metalCommandQueue(void* handle) {
    return static_cast<MTL4::CommandQueue*>(handle);
}

MTL4::CommandBuffer* metalCommandBuffer(void* handle) {
    return static_cast<MTL4::CommandBuffer*>(handle);
}

MTL::Buffer* metalBuffer(void* handle) {
    return static_cast<MTL::Buffer*>(handle);
}

MTL::AccelerationStructure* metalAccelerationStructure(void* handle) {
    return static_cast<MTL::AccelerationStructure*>(handle);
}

std::string metalErrorMessage(NS::Error* error, const char* fallback) {
    if (error && error->localizedDescription()) {
        return error->localizedDescription()->utf8String();
    }
    return fallback ? fallback : "Unknown Metal error";
}

void fillTransform(MTL::PackedFloat4x3& destination,
                   const MetalRayTracingInstanceDesc& source) {
    destination.columns[0] = {source.transform[0], source.transform[1], source.transform[2]};
    destination.columns[1] = {source.transform[3], source.transform[4], source.transform[5]};
    destination.columns[2] = {source.transform[6], source.transform[7], source.transform[8]};
    destination.columns[3] = {source.transform[9], source.transform[10], source.transform[11]};
}

bool writeInstanceDescriptors(const void* const* referencedAccelerationStructures,
                              uint32_t referencedAccelerationStructureCount,
                              void* instanceDescriptorBufferHandle,
                              const MetalRayTracingInstanceDesc* instances,
                              uint32_t instanceCount,
                              std::string& errorMessage) {
    auto* descriptorBuffer = metalBuffer(instanceDescriptorBufferHandle);
    if (!descriptorBuffer) {
        errorMessage = "Missing instance descriptor buffer";
        return false;
    }

    auto* destination = static_cast<MTL::IndirectAccelerationStructureInstanceDescriptor*>(descriptorBuffer->contents());
    if (!destination && instanceCount > 0) {
        errorMessage = "Failed to map instance descriptor buffer";
        return false;
    }

    for (uint32_t index = 0; index < instanceCount; ++index) {
        const auto& source = instances[index];
        if (source.accelerationStructureIndex >= referencedAccelerationStructureCount ||
            !referencedAccelerationStructures[source.accelerationStructureIndex]) {
            errorMessage = "Invalid TLAS instance acceleration structure index";
            return false;
        }

        auto* referencedAs = metalAccelerationStructure(
            const_cast<void*>(referencedAccelerationStructures[source.accelerationStructureIndex]));

        auto& instance = destination[index];
        instance = {};
        fillTransform(instance.transformationMatrix, source);
        instance.options = source.opaque ? MTL::AccelerationStructureInstanceOptionOpaque
                                         : MTL::AccelerationStructureInstanceOptionNone;
        instance.mask = source.mask;
        instance.intersectionFunctionTableOffset = 0;
        instance.userID = index;
        instance.accelerationStructureID = referencedAs->gpuResourceID();
    }
    return true;
}

bool submitAndWait(MTL::Device* device,
                   MTL4::CommandQueue* commandQueue,
                   MTL4::CommandBuffer* commandBuffer,
                   std::string& errorMessage) {
    auto* event = device->newSharedEvent();
    if (!event) {
        errorMessage = "Failed to allocate Metal4 shared event";
        return false;
    }

    commandBuffer->endCommandBuffer();
    const MTL4::CommandBuffer* buffers[] = { commandBuffer };
    commandQueue->commit(buffers, 1);
    commandQueue->signalEvent(event, 1);
    const bool signaled = event->waitUntilSignaledValue(1, std::numeric_limits<uint64_t>::max());
    event->release();
    if (!signaled) {
        errorMessage = "Timed out waiting for Metal4 acceleration structure build";
    }
    return signaled;
}

MTL4::InstanceAccelerationStructureDescriptor* createTlasDescriptor(MTL::Buffer* instanceDescriptorBuffer,
                                                                    uint32_t instanceCount) {
    auto* accelerationDescriptor = MTL4::InstanceAccelerationStructureDescriptor::alloc()->init();
    accelerationDescriptor->setInstanceCount(instanceCount);
    accelerationDescriptor->setInstanceDescriptorBuffer(
        MTL4::BufferRange(instanceDescriptorBuffer->gpuAddress(), instanceDescriptorBuffer->length()));
    accelerationDescriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeIndirect);
    accelerationDescriptor->setInstanceDescriptorStride(sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor));
    return accelerationDescriptor;
}

} // namespace

bool metalBuildBottomLevelAccelerationStructure(void* deviceHandle,
                                                void* commandQueueHandle,
                                                void* positionBufferHandle,
                                                uint32_t positionStride,
                                                void* indexBufferHandle,
                                                const MetalRayTracingGeometryRange* geometryRanges,
                                                uint32_t geometryCount,
                                                void*& outAccelerationStructure,
                                                std::string& errorMessage) {
    auto* device = metalDevice(deviceHandle);
    auto* commandQueue = metalCommandQueue(commandQueueHandle);
    auto* positionBuffer = metalBuffer(positionBufferHandle);
    auto* indexBuffer = metalBuffer(indexBufferHandle);

    if (!device || !commandQueue || !positionBuffer || !indexBuffer) {
        errorMessage = "Missing Metal ray tracing build inputs";
        return false;
    }
    if (!geometryRanges || geometryCount == 0) {
        errorMessage = "No geometry ranges provided for BLAS build";
        return false;
    }

    std::vector<MTL4::AccelerationStructureTriangleGeometryDescriptor*> geometryDescriptors;
    geometryDescriptors.reserve(geometryCount);

    for (uint32_t index = 0; index < geometryCount; ++index) {
        const auto& geometryRange = geometryRanges[index];
        const uint64_t indexOffset = static_cast<uint64_t>(geometryRange.indexOffset) * sizeof(uint32_t);
        auto* descriptor = MTL4::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
        descriptor->setVertexBuffer(MTL4::BufferRange(positionBuffer->gpuAddress(), positionBuffer->length()));
        descriptor->setVertexStride(positionStride);
        descriptor->setVertexFormat(MTL::AttributeFormatFloat3);
        descriptor->setIndexBuffer(MTL4::BufferRange(indexBuffer->gpuAddress() + indexOffset,
                                                     indexBuffer->length() - indexOffset));
        descriptor->setIndexType(MTL::IndexTypeUInt32);
        descriptor->setTriangleCount(geometryRange.indexCount / 3);
        descriptor->setOpaque(true);
        geometryDescriptors.push_back(descriptor);
    }

    auto* descriptorArray = NS::Array::array(
        reinterpret_cast<const NS::Object* const*>(geometryDescriptors.data()),
        geometryDescriptors.size());

    auto* accelerationDescriptor = MTL4::PrimitiveAccelerationStructureDescriptor::alloc()->init();
    accelerationDescriptor->setGeometryDescriptors(descriptorArray);

    auto sizes = device->accelerationStructureSizes(accelerationDescriptor);
    auto* accelerationStructure = device->newAccelerationStructure(sizes.accelerationStructureSize);
    auto* scratchBuffer = device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    if (!accelerationStructure || !scratchBuffer) {
        errorMessage = "Failed to allocate BLAS resources";
        if (accelerationStructure) accelerationStructure->release();
        if (scratchBuffer) scratchBuffer->release();
        accelerationDescriptor->release();
        for (auto* descriptor : geometryDescriptors) {
            descriptor->release();
        }
        return false;
    }
    metalTrackAllocation(device, accelerationStructure);
    metalTrackAllocation(device, scratchBuffer);

    auto* allocator = device->newCommandAllocator();
    auto* commandBuffer = device->newCommandBuffer();
    if (!allocator || !commandBuffer) {
        errorMessage = "Failed to allocate BLAS command buffer";
        if (allocator) allocator->release();
        if (commandBuffer) commandBuffer->release();
        metalReleaseHandle(scratchBuffer);
        metalReleaseHandle(accelerationStructure);
        accelerationDescriptor->release();
        for (auto* descriptor : geometryDescriptors) {
            descriptor->release();
        }
        return false;
    }

    allocator->reset();
    commandBuffer->beginCommandBuffer(allocator);
    auto* encoder = commandBuffer->computeCommandEncoder();
    encoder->buildAccelerationStructure(accelerationStructure,
                                        accelerationDescriptor,
                                        MTL4::BufferRange(scratchBuffer->gpuAddress(), scratchBuffer->length()));
    encoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure,
                                       MTL::StageDispatch,
                                       MTL4::VisibilityOptionDevice);
    encoder->endEncoding();

    const bool submitted = submitAndWait(device, commandQueue, commandBuffer, errorMessage);

    commandBuffer->release();
    allocator->release();
    metalReleaseHandle(scratchBuffer);
    accelerationDescriptor->release();
    for (auto* descriptor : geometryDescriptors) {
        descriptor->release();
    }

    if (!submitted) {
        metalReleaseHandle(accelerationStructure);
        return false;
    }

    outAccelerationStructure = accelerationStructure;
    return true;
}

bool metalBuildTopLevelAccelerationStructure(void* deviceHandle,
                                             void* commandQueueHandle,
                                             const void* const* referencedAccelerationStructures,
                                             uint32_t referencedAccelerationStructureCount,
                                             const MetalRayTracingInstanceDesc* instances,
                                             uint32_t instanceCount,
                                             void*& outAccelerationStructure,
                                             void*& outInstanceDescriptorBuffer,
                                             void*& outScratchBuffer,
                                             std::string& errorMessage) {
    auto* device = metalDevice(deviceHandle);
    auto* commandQueue = metalCommandQueue(commandQueueHandle);
    if (!device || !commandQueue) {
        errorMessage = "Missing Metal device or command queue";
        return false;
    }
    if (!referencedAccelerationStructures || referencedAccelerationStructureCount == 0) {
        errorMessage = "No referenced BLAS provided for TLAS build";
        return false;
    }
    if (!instances || instanceCount == 0) {
        errorMessage = "No instances provided for TLAS build";
        return false;
    }

    outInstanceDescriptorBuffer = device->newBuffer(
        static_cast<uint64_t>(instanceCount) * sizeof(MTL::IndirectAccelerationStructureInstanceDescriptor),
        MTL::ResourceStorageModeShared);
    if (!outInstanceDescriptorBuffer) {
        errorMessage = "Failed to allocate TLAS instance descriptor buffer";
        return false;
    }
    metalTrackAllocation(device, outInstanceDescriptorBuffer);

    if (!writeInstanceDescriptors(referencedAccelerationStructures,
                                  referencedAccelerationStructureCount,
                                  outInstanceDescriptorBuffer,
                                  instances,
                                  instanceCount,
                                  errorMessage)) {
        metalReleaseHandle(outInstanceDescriptorBuffer);
        outInstanceDescriptorBuffer = nullptr;
        return false;
    }

    auto* accelerationDescriptor = createTlasDescriptor(metalBuffer(outInstanceDescriptorBuffer), instanceCount);

    auto sizes = device->accelerationStructureSizes(accelerationDescriptor);
    outAccelerationStructure = device->newAccelerationStructure(sizes.accelerationStructureSize);
    outScratchBuffer = device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    if (!outAccelerationStructure || !outScratchBuffer) {
        errorMessage = "Failed to allocate TLAS resources";
        if (outAccelerationStructure) {
            metalReleaseHandle(outAccelerationStructure);
            outAccelerationStructure = nullptr;
        }
        if (outScratchBuffer) {
            metalReleaseHandle(outScratchBuffer);
            outScratchBuffer = nullptr;
        }
        metalReleaseHandle(outInstanceDescriptorBuffer);
        outInstanceDescriptorBuffer = nullptr;
        accelerationDescriptor->release();
        return false;
    }
    metalTrackAllocation(device, outAccelerationStructure);
    metalTrackAllocation(device, outScratchBuffer);

    auto* allocator = device->newCommandAllocator();
    auto* commandBuffer = device->newCommandBuffer();
    if (!allocator || !commandBuffer) {
        errorMessage = "Failed to allocate TLAS command buffer";
        if (allocator) allocator->release();
        if (commandBuffer) commandBuffer->release();
        metalReleaseHandle(outAccelerationStructure);
        metalReleaseHandle(outScratchBuffer);
        metalReleaseHandle(outInstanceDescriptorBuffer);
        outAccelerationStructure = nullptr;
        outScratchBuffer = nullptr;
        outInstanceDescriptorBuffer = nullptr;
        accelerationDescriptor->release();
        return false;
    }

    allocator->reset();
    commandBuffer->beginCommandBuffer(allocator);
    auto* encoder = commandBuffer->computeCommandEncoder();
    encoder->buildAccelerationStructure(metalAccelerationStructure(outAccelerationStructure),
                                        accelerationDescriptor,
                                        MTL4::BufferRange(metalBuffer(outScratchBuffer)->gpuAddress(),
                                                          metalBuffer(outScratchBuffer)->length()));
    encoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure,
                                       MTL::StageDispatch,
                                       MTL4::VisibilityOptionDevice);
    encoder->endEncoding();

    const bool submitted = submitAndWait(device, commandQueue, commandBuffer, errorMessage);
    commandBuffer->release();
    allocator->release();
    accelerationDescriptor->release();

    if (!submitted) {
        metalReleaseHandle(outAccelerationStructure);
        metalReleaseHandle(outScratchBuffer);
        metalReleaseHandle(outInstanceDescriptorBuffer);
        outAccelerationStructure = nullptr;
        outScratchBuffer = nullptr;
        outInstanceDescriptorBuffer = nullptr;
        return false;
    }

    return true;
}

bool metalUpdateTopLevelAccelerationStructure(void* commandBufferHandle,
                                              const void* const* referencedAccelerationStructures,
                                              uint32_t referencedAccelerationStructureCount,
                                              const MetalRayTracingInstanceDesc* instances,
                                              uint32_t instanceCount,
                                              void* accelerationStructureHandle,
                                              void* instanceDescriptorBufferHandle,
                                              void* scratchBufferHandle,
                                              std::string& errorMessage) {
    auto* commandBuffer = metalCommandBuffer(commandBufferHandle);
    if (!commandBuffer || !accelerationStructureHandle || !instanceDescriptorBufferHandle || !scratchBufferHandle) {
        errorMessage = "Missing TLAS update inputs";
        return false;
    }
    if (!referencedAccelerationStructures || referencedAccelerationStructureCount == 0) {
        errorMessage = "No referenced BLAS provided for TLAS update";
        return false;
    }
    if (!instances || instanceCount == 0) {
        errorMessage = "No instances provided for TLAS update";
        return false;
    }

    if (!writeInstanceDescriptors(referencedAccelerationStructures,
                                  referencedAccelerationStructureCount,
                                  instanceDescriptorBufferHandle,
                                  instances,
                                  instanceCount,
                                  errorMessage)) {
        return false;
    }

    auto* accelerationDescriptor = createTlasDescriptor(metalBuffer(instanceDescriptorBufferHandle), instanceCount);

    auto* encoder = commandBuffer->computeCommandEncoder();
    encoder->buildAccelerationStructure(metalAccelerationStructure(accelerationStructureHandle),
                                        accelerationDescriptor,
                                        MTL4::BufferRange(metalBuffer(scratchBufferHandle)->gpuAddress(),
                                                          metalBuffer(scratchBufferHandle)->length()));
    encoder->barrierAfterEncoderStages(MTL::StageAccelerationStructure,
                                       MTL::StageDispatch,
                                       MTL4::VisibilityOptionDevice);
    encoder->endEncoding();
    accelerationDescriptor->release();
    return true;
}
