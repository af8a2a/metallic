#pragma once

#include <ml.h>
#include <cstdint>

// CPU-side payload structs for GPU-driven meshlet culling.

enum : uint32_t {
    kVisibleInstanceClassificationVisible = 1u << 0,
    kVisibleInstanceClassificationHasLod = 1u << 1,
    kMeshletDrawSourceScene = 0u,
    kMeshletDrawSourceClusterLod = 1u,
    kClusterLodGroupResidencyResident = 1u << 0,
    kClusterLodGroupResidencyRequested = 1u << 1,
    kClusterLodGroupResidencyAlwaysResident = 1u << 2,
    kClusterLodGroupResidencyTouched = 1u << 3,
    kClusterTraversalStatsHistogramSize = 8u,
};

constexpr uint64_t kClusterLodGroupPageInvalidAddressBit = 1ull << 63u;
constexpr uint64_t kClusterLodGroupPageInvalidAddressStart =
    kClusterLodGroupPageInvalidAddressBit;

constexpr uint64_t makeClusterLodGroupPageResidentAddress(uint32_t residentHeapOffset) {
    return uint64_t(residentHeapOffset);
}

constexpr uint64_t makeClusterLodGroupPageInvalidAddress(uint32_t frameIndex = 0u) {
    return kClusterLodGroupPageInvalidAddressStart | uint64_t(frameIndex);
}

constexpr bool isClusterLodGroupPageAddressValid(uint64_t address) {
    return (address & kClusterLodGroupPageInvalidAddressBit) == 0u;
}

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
    uint32_t requestFrameIndex = UINT32_MAX;
};
static_assert(sizeof(ClusterResidencyRequest) == 16,
              "ClusterResidencyRequest must match shader layout");

struct ClusterUnloadRequest {
    uint32_t targetGroupIndex = UINT32_MAX;
    uint32_t requestFrameIndex = UINT32_MAX;
};
static_assert(sizeof(ClusterUnloadRequest) == 8,
              "ClusterUnloadRequest must match shader layout");

struct ClusterStreamingGpuStats {
    uint32_t frameIndex = UINT32_MAX;
    uint32_t unloadRequestCount = 0;
    uint32_t unloadAgeSum = 0;
    uint32_t appliedPatchCount = 0;
    uint32_t copiedBytesLow = 0;
    uint32_t copiedBytesHigh = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};
static_assert(sizeof(ClusterStreamingGpuStats) == sizeof(uint32_t) * 8u,
              "ClusterStreamingGpuStats must match shader layout");

constexpr uint64_t clusterStreamingGpuStatsCopiedBytes(const ClusterStreamingGpuStats& stats) {
    return uint64_t(stats.copiedBytesLow) | (uint64_t(stats.copiedBytesHigh) << 32u);
}

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
    uint32_t residencyRequestFrameIndex = 0;
    uint32_t reserved0 = 0;
};

struct StreamingAgeFilterUniforms {
    uint32_t groupCount = 0;
    uint32_t ageThreshold = 16;
    uint32_t requestFrameIndex = 0;
    uint32_t reserved1 = 0;
};

struct StreamingPatch {
    uint32_t groupIndex = UINT32_MAX;
    uint64_t residentHeapOffset = makeClusterLodGroupPageInvalidAddress();
    uint32_t clusterStart = 0;
    uint32_t clusterCount = 0;
};
static_assert(sizeof(StreamingPatch) == sizeof(uint64_t) + sizeof(uint32_t) * 4u,
              "StreamingPatch must match shader layout");

struct StreamingUpdateUniforms {
    uint32_t patchCount = 0;
    uint32_t copySourceData = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
};
