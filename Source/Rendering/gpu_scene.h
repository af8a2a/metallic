#pragma once

#include <cstdint>
#include <vector>

#include "rhi_backend.h"
#include "cluster_types.h"

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
    uint32_t packedClusterStart = 0;
    uint32_t packedClusterCount = 0;
};
static_assert(sizeof(GPUSceneGeometry) == 56, "GPUSceneGeometry must match shader layout");

struct GPUSceneInstance {
    float worldMatrix[16] = {};
    float prevWorldMatrix[16] = {};
    uint32_t geometryIndex = UINT32_MAX;
    uint32_t dispatchStart = 0;
    uint32_t sceneNodeIndex = UINT32_MAX;
    uint32_t visibilityFlags = 0;
};
static_assert(sizeof(GPUSceneInstance) == 144, "GPUSceneInstance must match shader layout");

struct DagV1ValidationStats {
    bool available = false;
    bool passed = false;
    uint32_t geometryCount = 0;
    uint32_t checkedGeometryCount = 0;
    uint32_t skippedGeometryCount = 0;
    uint32_t validRootCount = 0;
    uint32_t invalidRootCount = 0;
    uint32_t traversalFailureCount = 0;
    uint32_t mismatchGeometryCount = 0;
    uint32_t expectedClusterCount = 0;
    uint32_t traversedClusterCount = 0;
    uint32_t missingClusterCount = 0;
    uint32_t unexpectedClusterCount = 0;
    uint32_t duplicateClusterCount = 0;
    uint32_t invalidNodeRefCount = 0;
    uint32_t invalidGroupRefCount = 0;
    uint32_t invalidClusterRefCount = 0;
    uint32_t maxDepth = 0;
};

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

    // Cluster visualization CPU worklist (Phase 1)
    std::vector<ClusterInfo> clusterVisWorklist;
    RhiBufferHandle clusterVisWorklistBuffer;
    uint32_t clusterVisWorklistCount = 0;

    DagV1ValidationStats dagV1Validation;
};

bool buildGpuSceneTables(const RhiDevice& device,
                         const LoadedMesh& mesh,
                         const MeshletData& meshletData,
                         const ClusterLODData* clusterLodData,
                         const SceneGraph& sceneGraph,
                         GpuSceneTables& out);
void updateGpuSceneTables(const SceneGraph& sceneGraph, GpuSceneTables& tables);
void releaseGpuSceneTables(GpuSceneTables& tables);
