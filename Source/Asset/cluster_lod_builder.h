#pragma once

#include <cstdint>
#include <cfloat>
#include <vector>

#include "rhi_backend.h"
#include "meshlet_builder.h"

// GPU cluster group (32 bytes, alignment-friendly)
struct GPUClusterGroup {
    float    center[3];         // bounding sphere center (object space)
    float    radius;            // bounding sphere radius
    float    error;             // max simplification error at this LOD
    float    parentError;       // parent group's error (coarser LOD), FLT_MAX for roots
    uint32_t clusterStart;      // first meshlet index in global array
    uint32_t clusterCount;      // number of meshlets in this group
};

static_assert(sizeof(GPUClusterGroup) == 32, "GPUClusterGroup must be 32 bytes");

// GPU LOD node for DAG traversal (32 bytes)
struct GPULodNode {
    float    center[3];         // bounding sphere center
    float    radius;            // bounding sphere radius
    float    maxError;          // max error in subtree
    uint32_t childOffset;       // first child index (node or group)
    uint32_t childCount;        // number of children
    uint32_t isLeaf;            // 1 = children are groups, 0 = children are nodes
};

static_assert(sizeof(GPULodNode) == 32, "GPULodNode must be 32 bytes");

// CPU-side per-level bookkeeping
struct ClusterLODLevel {
    uint32_t meshletStart;      // offset into allMeshlets
    uint32_t meshletCount;
    uint32_t groupStart;        // offset into allGroups
    uint32_t groupCount;
};

// Complete LOD hierarchy output
struct ClusterLODData {
    // All LOD levels' meshlet data (LOD 0 first)
    std::vector<GPUMeshlet>       allMeshlets;
    std::vector<unsigned int>     allMeshletVertices;
    std::vector<uint32_t>         allPackedTriangles;
    std::vector<GPUMeshletBounds> allBounds;
    std::vector<uint32_t>         allMaterialIDs;

    // LOD hierarchy
    std::vector<GPUClusterGroup>  groups;
    std::vector<GPULodNode>       nodes;
    std::vector<ClusterLODLevel>  levels;

    // GPU buffers (filled after upload)
    RhiBufferHandle meshletBuffer;
    RhiBufferHandle meshletVerticesBuffer;
    RhiBufferHandle meshletTrianglesBuffer;
    RhiBufferHandle boundsBuffer;
    RhiBufferHandle materialIDsBuffer;
    RhiBufferHandle groupBuffer;
    RhiBufferHandle nodeBuffer;

    uint32_t totalMeshletCount = 0;
    uint32_t totalGroupCount = 0;
    uint32_t totalNodeCount = 0;
    uint32_t lodLevelCount = 0;
};

struct LoadedMesh;

// Build cluster LOD hierarchy from existing meshlet data.
// mesh: source mesh with CPU positions/indices
// meshletData: LOD 0 meshlets (must have cpuMeshlets etc. populated)
// out: receives the full LOD hierarchy + GPU buffers
bool buildClusterLOD(const RhiDevice& device,
                     const LoadedMesh& mesh,
                     const MeshletData& meshletData,
                     ClusterLODData& out);

// Render LOD stats in ImGui
void drawClusterLODStats(const ClusterLODData& data);
