#pragma once

#include <Metal/Metal.hpp>
#include <vector>
#include <cstdint>

#include "rhi_backend.h"

struct LoadedMesh;
struct MeshletData;
class SceneGraph;

struct RaytracedShadowResources {
    std::vector<MTL::AccelerationStructure*> blasArray;
    std::vector<RhiAccelerationStructureHandle> blasHandles;
    MTL::AccelerationStructure* tlas = nullptr;
    RhiAccelerationStructureHandle tlasRhi;
    MTL::Buffer* instanceDescriptorBuffer = nullptr;
    RhiBufferHandle instanceDescriptorBufferRhi;
    MTL::Buffer* scratchBuffer = nullptr;
    RhiBufferHandle scratchBufferRhi;
    MTL::ComputePipelineState* pipeline = nullptr;
    RhiComputePipelineHandle pipelineRhi;
    MTL::Library* library = nullptr;

    // Cached for TLAS rebuild
    std::vector<MTL::AccelerationStructure*> referencedBLAS;
    uint32_t instanceCount = 0;

    void release();
};

bool buildAccelerationStructures(MTL::Device* device,
                                 MTL::CommandQueue* commandQueue,
                                 const LoadedMesh& mesh,
                                 const SceneGraph& sceneGraph,
                                 RaytracedShadowResources& out);

void updateTLAS(MTL::CommandBuffer* commandBuffer,
                const SceneGraph& sceneGraph,
                RaytracedShadowResources& res);

bool createShadowPipeline(MTL::Device* device,
                          RaytracedShadowResources& out,
                          const char* shaderBasePath = nullptr);

bool reloadShadowPipeline(MTL::Device* device,
                           RaytracedShadowResources& res,
                           const char* shaderBasePath = nullptr);
