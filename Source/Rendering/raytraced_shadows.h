#pragma once

#include <cstdint>
#include <vector>

#include "rhi_backend.h"

struct LoadedMesh;
class SceneGraph;

struct RaytracedShadowResources {
    std::vector<RhiAccelerationStructureHandle> blasArray;
    RhiAccelerationStructureHandle tlas;
    RhiBufferHandle instanceDescriptorBuffer;
    RhiBufferHandle scratchBuffer;
    RhiComputePipelineHandle pipeline;
    RhiShaderLibraryHandle library;

    std::vector<RhiAccelerationStructureHandle> referencedBlas;
    uint32_t instanceCount = 0;

    void release();
};

bool buildAccelerationStructures(const RhiDevice& device,
                                 const RhiCommandQueue& commandQueue,
                                 const LoadedMesh& mesh,
                                 const SceneGraph& sceneGraph,
                                 RaytracedShadowResources& out);

void updateTLAS(const RhiNativeCommandBuffer& commandBuffer,
                const SceneGraph& sceneGraph,
                RaytracedShadowResources& res);

bool createShadowPipeline(const RhiDevice& device,
                          RaytracedShadowResources& out,
                          const char* shaderBasePath = nullptr);

bool reloadShadowPipeline(const RhiDevice& device,
                          RaytracedShadowResources& res,
                          const char* shaderBasePath = nullptr);
