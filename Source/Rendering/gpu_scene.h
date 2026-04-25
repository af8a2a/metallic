#pragma once

#include <cstdint>
#include <vector>

#include "rhi_backend.h"

struct LoadedMesh;
struct MeshletData;
struct ClusterLODData;
class SceneGraph;

enum : uint32_t {
    kGpuSceneInstanceVisible = 1u << 0,
    kGpuSceneInstanceHasLod = 1u << 1,
};

struct GPUSceneGeometry {
    uint32_t meshletStart = 0;
    uint32_t meshletCount = 0;
    uint32_t primitiveGroupStart = 0;
    uint32_t primitiveGroupCount = 0;
    uint32_t indexStart = 0;
    uint32_t indexCount = 0;
    uint32_t materialIndex = UINT32_MAX;
    uint32_t lodRootNode = UINT32_MAX;
    float boundsCenterRadius[4] = {};
};
static_assert(sizeof(GPUSceneGeometry) == 48, "GPUSceneGeometry must match shader layout");

struct GPUSceneInstance {
    float worldMatrix[16] = {};
    float prevWorldMatrix[16] = {};
    uint32_t geometryIndex = UINT32_MAX;
    uint32_t dispatchStart = 0;
    uint32_t sceneNodeIndex = UINT32_MAX;
    uint32_t visibilityFlags = 0;
};
static_assert(sizeof(GPUSceneInstance) == 144, "GPUSceneInstance must match shader layout");

struct GpuSceneTables {
    std::vector<GPUSceneGeometry> geometries;
    std::vector<GPUSceneInstance> instances;
    std::vector<uint32_t> nodeToInstance;

    RhiBufferHandle geometryBuffer;
    RhiBufferHandle instanceBuffer;

    uint32_t geometryCount = 0;
    uint32_t instanceCount = 0;
    uint32_t totalMeshletDispatchCount = 0;
    uint32_t visibleInstanceCount = 0;
};

bool buildGpuSceneTables(const RhiDevice& device,
                         const LoadedMesh& mesh,
                         const MeshletData& meshletData,
                         const ClusterLODData* clusterLodData,
                         const SceneGraph& sceneGraph,
                         GpuSceneTables& out);
void updateGpuSceneTables(const SceneGraph& sceneGraph, GpuSceneTables& tables);
void releaseGpuSceneTables(GpuSceneTables& tables);
