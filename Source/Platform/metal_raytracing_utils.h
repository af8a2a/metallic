#pragma once

#include <cstdint>
#include <string>

struct MetalRayTracingGeometryRange {
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
};

struct MetalRayTracingInstanceDesc {
    float transform[12] = {};
    uint32_t accelerationStructureIndex = 0;
    uint32_t mask = 0xFF;
    bool opaque = true;
};

bool metalBuildBottomLevelAccelerationStructure(void* deviceHandle,
                                                void* commandQueueHandle,
                                                void* positionBufferHandle,
                                                uint32_t positionStride,
                                                void* indexBufferHandle,
                                                const MetalRayTracingGeometryRange* geometryRanges,
                                                uint32_t geometryCount,
                                                void*& outAccelerationStructure,
                                                std::string& errorMessage);

bool metalBuildTopLevelAccelerationStructure(void* deviceHandle,
                                             void* commandQueueHandle,
                                             const void* const* referencedAccelerationStructures,
                                             uint32_t referencedAccelerationStructureCount,
                                             const MetalRayTracingInstanceDesc* instances,
                                             uint32_t instanceCount,
                                             void*& outAccelerationStructure,
                                             void*& outInstanceDescriptorBuffer,
                                             void*& outScratchBuffer,
                                             std::string& errorMessage);

bool metalUpdateTopLevelAccelerationStructure(void* commandBufferHandle,
                                              const void* const* referencedAccelerationStructures,
                                              uint32_t referencedAccelerationStructureCount,
                                              const MetalRayTracingInstanceDesc* instances,
                                              uint32_t instanceCount,
                                              void* accelerationStructureHandle,
                                              void* instanceDescriptorBufferHandle,
                                              void* scratchBufferHandle,
                                              std::string& errorMessage);
