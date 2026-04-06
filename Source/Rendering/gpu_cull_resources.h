#pragma once

#include <ml.h>
#include <cstdint>

// CPU-side payload structs for GPU-driven meshlet culling.

enum : uint32_t {
    kVisibleInstanceClassificationVisible = 1u << 0,
    kVisibleInstanceClassificationHasLod = 1u << 1,
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
    uint32_t instanceID;
    uint32_t globalMeshletID;
};

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
    float2   prevProjScale;
    float2   hzbTextureSize;
    uint32_t enableFrustumCull;
    uint32_t enableConeCull;
    uint32_t enableOcclusionCull;
    uint32_t hzbLevelCount;
    float    occlusionDepthBias;
    float    occlusionBoundsScale;
};
