#pragma once

#include <cstdint>
#include <vector>

#include "rhi_backend.h"

struct LoadedMesh;
class SceneGraph;

struct RaytracedShadowResources {
    std::vector<void*> blasArray;
    std::vector<RhiAccelerationStructureHandle> blasHandles;
    void* tlas = nullptr;
    RhiAccelerationStructureHandle tlasRhi;
    void* instanceDescriptorBuffer = nullptr;
    RhiBufferHandle instanceDescriptorBufferRhi;
    void* scratchBuffer = nullptr;
    RhiBufferHandle scratchBufferRhi;
    void* pipeline = nullptr;
    RhiComputePipelineHandle pipelineRhi;
    void* library = nullptr;

    std::vector<void*> referencedBLAS;
    uint32_t instanceCount = 0;

    void release();
};

bool buildAccelerationStructures(void* deviceHandle,
                                 void* commandQueueHandle,
                                 const LoadedMesh& mesh,
                                 const SceneGraph& sceneGraph,
                                 RaytracedShadowResources& out);

void updateTLAS(void* commandBufferHandle,
                const SceneGraph& sceneGraph,
                RaytracedShadowResources& res);

bool createShadowPipeline(void* deviceHandle,
                          RaytracedShadowResources& out,
                          const char* shaderBasePath = nullptr);

bool reloadShadowPipeline(void* deviceHandle,
                          RaytracedShadowResources& res,
                          const char* shaderBasePath = nullptr);
