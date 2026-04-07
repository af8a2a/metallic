#pragma once

#include <ml.h>
#include <cstdint>

// CPU-side payload structs for GPU-driven meshlet culling.

enum : uint32_t {
    kVisibleInstanceClassificationVisible = 1u << 0,
    kVisibleInstanceClassificationHasLod = 1u << 1,
    kMeshletDrawSourceScene = 0u,
    kMeshletDrawSourceClusterLod = 1u,
    kClusterLodGroupPageInvalidAddress = UINT32_MAX,
    kClusterLodGroupResidencyResident = 1u << 0,
    kClusterLodGroupResidencyRequested = 1u << 1,
    kClusterLodGroupResidencyAlwaysResident = 1u << 2,
    kClusterTraversalStatsHistogramSize = 8u,
};

struct VisibleInstanceInfo {
    uint32_t sceneInstanceID = UINT32_MAX;
    uint32_t geometryIndex = UINT32_MAX;
    uint32_t lodRootNode = UINT32_MAX;
    uint32_t classificationFlags = 0;
    float boundsCenterRadius[4] = {};
    // x = clip/view depth proxy, y = camera distance, z = uniform scale, w = projected scale
    float lodMetric[4] = {};
};
static_assert(sizeof(VisibleInstanceInfo) == 48, "VisibleInstanceInfo must match shader layout");

struct MeshletDrawInfo {
    uint32_t instanceID = UINT32_MAX;
    uint32_t globalMeshletID = UINT32_MAX;
    uint32_t meshletSource = kMeshletDrawSourceScene;
    uint32_t lodLevel = 0;
};
static_assert(sizeof(MeshletDrawInfo) == 16, "MeshletDrawInfo must match shader layout");

struct ClusterResidencyRequest {
    uint32_t lodRootNode = UINT32_MAX;
    uint32_t targetGroupIndex = UINT32_MAX;
    uint32_t lodLevel = 0;
    uint32_t sceneInstanceID = UINT32_MAX;
};
static_assert(sizeof(ClusterResidencyRequest) == 16,
              "ClusterResidencyRequest must match shader layout");

struct ClusterTraversalStats {
    uint32_t lodTraversalInstanceCount = 0;
    uint32_t fallbackInstanceCount = 0;
    uint32_t traversedNodeCount = 0;
    uint32_t occludedNodeCount = 0;
    uint32_t candidateGroupCount = 0;
    uint32_t selectedGroupCount = 0;
    uint32_t occludedGroupCount = 0;
    uint32_t candidateClusterMeshletCount = 0;
    uint32_t emittedClusterMeshletCount = 0;
    uint32_t candidateFallbackMeshletCount = 0;
    uint32_t emittedFallbackMeshletCount = 0;
    uint32_t maxSelectedLodLevel = 0;
    uint32_t selectedLodLevelHistogram[kClusterTraversalStatsHistogramSize] = {};
};
static_assert(sizeof(ClusterTraversalStats) ==
                  sizeof(uint32_t) * (12u + kClusterTraversalStatsHistogramSize),
              "ClusterTraversalStats must remain tightly packed");

struct InstanceClassifyUniforms {
    float4x4 viewProj;         // transposed for Slang
    float4x4 prevViewProj;     // transposed for Slang
    float4x4 prevView;         // transposed for Slang
    float4   cameraWorldPos;
    float4   prevCameraWorldPos;
    float2   prevProjScale;
    float2   hzbTextureSize;
    uint32_t instanceCount;
    uint32_t enableFrustumCull;
    uint32_t enableOcclusionCull;
    uint32_t hzbLevelCount;
    float    occlusionDepthBias;
    float    occlusionBoundsScale;
};

struct CullUniforms {
    float4x4 viewProj;         // transposed for Slang
    float4x4 prevViewProj;     // transposed for Slang
    float4x4 prevView;         // transposed for Slang
    float4   cameraWorldPos;
    float4   prevCameraWorldPos;
    float2   projScale;
    float2   renderTargetSize;
    float2   prevProjScale;
    float2   hzbTextureSize;
    uint32_t enableFrustumCull;
    uint32_t enableConeCull;
    uint32_t enableOcclusionCull;
    uint32_t hzbLevelCount;
    float    lodReferencePixels;
    float    occlusionDepthBias;
    float    occlusionBoundsScale;
    uint32_t clusterLodEnabled;
    uint32_t enableResidencyStreaming = 0;
};

struct StreamingAgeFilterUniforms {
    uint32_t groupCount = 0;
    uint32_t ageThreshold = 16;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};
