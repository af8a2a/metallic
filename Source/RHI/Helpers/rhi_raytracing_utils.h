#pragma once

#include <cstdint>
#include <string>

#include "rhi_backend.h"

struct RhiRayTracingGeometryRange {
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
};

struct RhiRayTracingInstanceDesc {
    float transform[12] = {};
    uint32_t accelerationStructureIndex = 0;
    uint32_t mask = 0xFF;
    bool opaque = true;
};

bool rhiBuildBottomLevelAccelerationStructure(const RhiDevice& device,
                                              const RhiCommandQueue& commandQueue,
                                              const RhiBuffer& positionBuffer,
                                              uint32_t positionStride,
                                              const RhiBuffer& indexBuffer,
                                              const RhiRayTracingGeometryRange* geometryRanges,
                                              uint32_t geometryCount,
                                              RhiAccelerationStructureHandle& outAccelerationStructure,
                                              std::string& errorMessage);

bool rhiBuildTopLevelAccelerationStructure(const RhiDevice& device,
                                           const RhiCommandQueue& commandQueue,
                                           const RhiAccelerationStructure* const* referencedAccelerationStructures,
                                           uint32_t referencedAccelerationStructureCount,
                                           const RhiRayTracingInstanceDesc* instances,
                                           uint32_t instanceCount,
                                           RhiAccelerationStructureHandle& outAccelerationStructure,
                                           RhiBufferHandle& outInstanceDescriptorBuffer,
                                           RhiBufferHandle& outScratchBuffer,
                                           std::string& errorMessage);

bool rhiUpdateTopLevelAccelerationStructure(const RhiNativeCommandBuffer& commandBuffer,
                                            const RhiAccelerationStructure* const* referencedAccelerationStructures,
                                            uint32_t referencedAccelerationStructureCount,
                                            const RhiRayTracingInstanceDesc* instances,
                                            uint32_t instanceCount,
                                            const RhiAccelerationStructure& accelerationStructure,
                                            const RhiBuffer& instanceDescriptorBuffer,
                                            const RhiBuffer& scratchBuffer,
                                            std::string& errorMessage);
