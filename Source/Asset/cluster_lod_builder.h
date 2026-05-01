#pragma once

#include <cfloat>
#include <cstdint>
#include <string>
#include <vector>

#include "rhi_backend.h"
#include "meshlet_builder.h"
#include "cluster_types.h"

// GPU cluster group (32 bytes, alignment-friendly)
struct GPUClusterGroup {
    float    center[3];         // bounding sphere center (object space)
    float    radius;            // bounding sphere radius
    float    error;             // error at this LOD
    float    parentError;       // coarser LOD transition threshold, FLT_MAX for terminal groups
    uint32_t clusterStart;      // offset into groupMeshletIndices
    uint32_t clusterCount;      // number of child meshlets in this group
};

static_assert(sizeof(GPUClusterGroup) == 32, "GPUClusterGroup must be 32 bytes");

// GPU LOD node for DAG traversal (48 bytes)
struct GPULodNode {
    float    center[3];         // bounding sphere center
    float    radius;            // bounding sphere radius
    float    maxError;          // max error in subtree
    uint32_t childOffset;       // first child index (node or group)
    uint32_t childCount;        // number of children
    uint32_t isLeaf;            // 1 = children are groups, 0 = children are nodes
    uint32_t representativeGroupStart; // offset into nodeRepresentativeGroupIndices
    uint32_t representativeGroupCount; // number of explicit representative groups
    uint32_t reserved0;
    uint32_t reserved1;
};

static_assert(sizeof(GPULodNode) == 48, "GPULodNode must be 48 bytes");

struct GPUClusterRefineInfo {
    uint32_t refineGroupStart;   // direct group range for the next finer LOD, UINT32_MAX if none
    uint32_t refineGroupCount;
};

static_assert(sizeof(GPUClusterRefineInfo) == 8, "GPUClusterRefineInfo must be 8 bytes");

// CPU-side per-level bookkeeping
struct ClusterLODLevel {
    uint32_t primitiveGroupIndex;
    uint32_t depth;
    uint32_t meshletStart;      // offset into allMeshlets
    uint32_t meshletCount;
    uint32_t groupStart;        // offset into allGroups
    uint32_t groupCount;
    uint32_t rootNode;          // root node for this LOD level
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
    std::vector<uint32_t>         groupMeshletIndices;
    std::vector<GPUClusterGroup>  groups;
    std::vector<GPULodNode>       nodes;
    std::vector<uint32_t>         nodeRepresentativeGroupIndices;
    std::vector<ClusterLODLevel>  levels;
    std::vector<uint32_t>         primitiveGroupLodRoots;
    std::vector<uint32_t>         primitiveGroupLod0Roots;

    // GPU buffers (filled after upload)
    RhiBufferHandle meshletBuffer;
    RhiBufferHandle meshletVerticesBuffer;
    RhiBufferHandle meshletTrianglesBuffer;
    RhiBufferHandle boundsBuffer;
    RhiBufferHandle materialIDsBuffer;
    RhiBufferHandle groupMeshletIndicesBuffer;
    RhiBufferHandle groupBuffer;
    RhiBufferHandle nodeBuffer;
    RhiBufferHandle nodeRepresentativeGroupIndexBuffer;
    RhiBufferHandle levelBuffer;

    // Packed cluster data (vk_lod_clusters-compatible)
    std::vector<PackedCluster>  packedClusters;
    std::vector<GPUClusterRefineInfo> clusterRefineInfos;
    std::vector<uint8_t>        clusterVertexData;
    std::vector<uint8_t>        clusterIndexData;
    RhiBufferHandle packedClusterBuffer;
    RhiBufferHandle clusterRefineInfoBuffer;
    RhiBufferHandle clusterVertexDataBuffer;
    RhiBufferHandle clusterIndexDataBuffer;

    uint32_t totalMeshletCount = 0;
    uint32_t totalGroupCount = 0;
    uint32_t totalNodeCount = 0;
    uint32_t lodLevelCount = 0;
    uint64_t sourceSceneSignature = 0u;
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
bool loadOrBuildClusterLOD(const RhiDevice& device,
                           const LoadedMesh& mesh,
                           const MeshletData& meshletData,
                           const std::string& sourcePath,
                           const std::string& cacheDirectory,
                           ClusterLODData& out);
void releaseClusterLOD(ClusterLODData& data);

// Render LOD stats in ImGui
void drawClusterLODStats(const ClusterLODData& data);
