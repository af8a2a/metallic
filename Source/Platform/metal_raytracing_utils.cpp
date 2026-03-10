#include "metal_raytracing_utils.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <vector>

namespace {

MTL::Device* metalDevice(void* handle) {
    return static_cast<MTL::Device*>(handle);
}

MTL::CommandQueue* metalCommandQueue(void* handle) {
    return static_cast<MTL::CommandQueue*>(handle);
}

MTL::CommandBuffer* metalCommandBuffer(void* handle) {
    return static_cast<MTL::CommandBuffer*>(handle);
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

void fillInstanceDescriptor(MTL::AccelerationStructureInstanceDescriptor& destination,
                            const MetalRayTracingInstanceDesc& source) {
    destination = {};
    destination.transformationMatrix.columns[0] = {source.transform[0], source.transform[1], source.transform[2]};
    destination.transformationMatrix.columns[1] = {source.transform[3], source.transform[4], source.transform[5]};
    destination.transformationMatrix.columns[2] = {source.transform[6], source.transform[7], source.transform[8]};
    destination.transformationMatrix.columns[3] = {source.transform[9], source.transform[10], source.transform[11]};
    destination.options = source.opaque ? MTL::AccelerationStructureInstanceOptionOpaque
                                        : static_cast<MTL::AccelerationStructureInstanceOptions>(0);
    destination.mask = static_cast<uint8_t>(source.mask);
    destination.intersectionFunctionTableOffset = 0;
    destination.accelerationStructureIndex = source.accelerationStructureIndex;
}

bool writeInstanceDescriptors(void* instanceDescriptorBufferHandle,
                              const MetalRayTracingInstanceDesc* instances,
                              uint32_t instanceCount,
                              std::string& errorMessage) {
    auto* descriptorBuffer = metalBuffer(instanceDescriptorBufferHandle);
    if (!descriptorBuffer) {
        errorMessage = "Missing instance descriptor buffer";
        return false;
    }

    auto* destination = static_cast<MTL::AccelerationStructureInstanceDescriptor*>(descriptorBuffer->contents());
    if (!destination && instanceCount > 0) {
        errorMessage = "Failed to map instance descriptor buffer";
        return false;
    }

    for (uint32_t index = 0; index < instanceCount; ++index) {
        fillInstanceDescriptor(destination[index], instances[index]);
    }
    return true;
}

NS::Array* accelerationStructureArray(const void* const* handles, uint32_t count) {
    return NS::Array::array(
        reinterpret_cast<const NS::Object* const*>(handles),
        count);
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

    std::vector<MTL::AccelerationStructureTriangleGeometryDescriptor*> geometryDescriptors;
    geometryDescriptors.reserve(geometryCount);

    for (uint32_t index = 0; index < geometryCount; ++index) {
        const auto& geometryRange = geometryRanges[index];
        auto* descriptor = MTL::AccelerationStructureTriangleGeometryDescriptor::alloc()->init();
        descriptor->setVertexBuffer(positionBuffer);
        descriptor->setVertexBufferOffset(0);
        descriptor->setVertexStride(positionStride);
        descriptor->setVertexFormat(MTL::AttributeFormatFloat3);
        descriptor->setIndexBuffer(indexBuffer);
        descriptor->setIndexBufferOffset(static_cast<uint64_t>(geometryRange.indexOffset) * sizeof(uint32_t));
        descriptor->setIndexType(MTL::IndexTypeUInt32);
        descriptor->setTriangleCount(geometryRange.indexCount / 3);
        descriptor->setOpaque(true);
        geometryDescriptors.push_back(descriptor);
    }

    auto* descriptorArray = NS::Array::array(
        reinterpret_cast<const NS::Object* const*>(geometryDescriptors.data()),
        geometryDescriptors.size());

    auto* accelerationDescriptor = MTL::PrimitiveAccelerationStructureDescriptor::alloc()->init();
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

    auto* commandBuffer = commandQueue->commandBuffer();
    auto* encoder = commandBuffer->accelerationStructureCommandEncoder();
    encoder->buildAccelerationStructure(accelerationStructure, accelerationDescriptor, scratchBuffer, 0);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    scratchBuffer->release();
    accelerationDescriptor->release();
    for (auto* descriptor : geometryDescriptors) {
        descriptor->release();
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
        static_cast<uint64_t>(instanceCount) * sizeof(MTL::AccelerationStructureInstanceDescriptor),
        MTL::ResourceStorageModeShared);
    if (!outInstanceDescriptorBuffer) {
        errorMessage = "Failed to allocate TLAS instance descriptor buffer";
        return false;
    }
    if (!writeInstanceDescriptors(outInstanceDescriptorBuffer, instances, instanceCount, errorMessage)) {
        static_cast<MTL::Buffer*>(outInstanceDescriptorBuffer)->release();
        outInstanceDescriptorBuffer = nullptr;
        return false;
    }

    auto* accelerationDescriptor = MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
    accelerationDescriptor->setInstanceCount(instanceCount);
    accelerationDescriptor->setInstanceDescriptorBuffer(metalBuffer(outInstanceDescriptorBuffer));
    accelerationDescriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeDefault);
    accelerationDescriptor->setInstancedAccelerationStructures(
        accelerationStructureArray(referencedAccelerationStructures, referencedAccelerationStructureCount));

    auto sizes = device->accelerationStructureSizes(accelerationDescriptor);
    outAccelerationStructure = device->newAccelerationStructure(sizes.accelerationStructureSize);
    outScratchBuffer = device->newBuffer(sizes.buildScratchBufferSize, MTL::ResourceStorageModePrivate);
    if (!outAccelerationStructure || !outScratchBuffer) {
        errorMessage = "Failed to allocate TLAS resources";
        if (outAccelerationStructure) {
            metalAccelerationStructure(outAccelerationStructure)->release();
            outAccelerationStructure = nullptr;
        }
        if (outScratchBuffer) {
            metalBuffer(outScratchBuffer)->release();
            outScratchBuffer = nullptr;
        }
        metalBuffer(outInstanceDescriptorBuffer)->release();
        outInstanceDescriptorBuffer = nullptr;
        accelerationDescriptor->release();
        return false;
    }

    auto* commandBuffer = commandQueue->commandBuffer();
    auto* encoder = commandBuffer->accelerationStructureCommandEncoder();
    encoder->buildAccelerationStructure(metalAccelerationStructure(outAccelerationStructure),
                                        accelerationDescriptor,
                                        metalBuffer(outScratchBuffer),
                                        0);
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    accelerationDescriptor->release();
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

    if (!writeInstanceDescriptors(instanceDescriptorBufferHandle, instances, instanceCount, errorMessage)) {
        return false;
    }

    auto* accelerationDescriptor = MTL::InstanceAccelerationStructureDescriptor::alloc()->init();
    accelerationDescriptor->setInstanceCount(instanceCount);
    accelerationDescriptor->setInstanceDescriptorBuffer(metalBuffer(instanceDescriptorBufferHandle));
    accelerationDescriptor->setInstanceDescriptorType(MTL::AccelerationStructureInstanceDescriptorTypeDefault);
    accelerationDescriptor->setInstancedAccelerationStructures(
        accelerationStructureArray(referencedAccelerationStructures, referencedAccelerationStructureCount));

    auto* encoder = commandBuffer->accelerationStructureCommandEncoder();
    encoder->buildAccelerationStructure(metalAccelerationStructure(accelerationStructureHandle),
                                        accelerationDescriptor,
                                        metalBuffer(scratchBufferHandle),
                                        0);
    encoder->endEncoding();
    accelerationDescriptor->release();
    return true;
}
