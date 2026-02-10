#pragma once

#include <Metal/Metal.hpp>
#include <vector>
#include <cstdint>

struct LoadedMesh;
struct MeshletData;
class SceneGraph;

struct RaytracedShadowResources {
    std::vector<MTL::AccelerationStructure*> blasArray;
    MTL::AccelerationStructure* tlas = nullptr;
    MTL::Buffer* instanceDescriptorBuffer = nullptr;
    MTL::Buffer* scratchBuffer = nullptr;
    MTL::ComputePipelineState* pipeline = nullptr;
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
                          RaytracedShadowResources& out);
