#pragma once

#include <ml.h>
#include <cstdint>

// CPU-side payload structs for GPU-driven meshlet culling.

struct MeshletDrawInfo {
    uint32_t instanceID;
    uint32_t globalMeshletID;
};

struct GPUInstanceData {
    float4x4 mvp;          // pre-transposed for Slang
    float4x4 modelView;    // pre-transposed for Slang
    float4x4 worldMatrix;  // pre-transposed for Slang (for cull shader)
    uint32_t meshletStart;
    uint32_t meshletCount;
    uint32_t instanceID;
    uint32_t pad;
};

struct CullUniforms {
    float4x4 viewProj;         // transposed for Slang
    float4x4 prevViewProj;     // transposed for Slang
    float4x4 prevView;         // transposed for Slang
    float4   cameraWorldPos;
    float4   prevCameraWorldPos;
    float2   prevProjScale;
    float2   hzbTextureSize;
    uint32_t totalDispatchCount;
    uint32_t instanceCount;
    uint32_t enableFrustumCull;
    uint32_t enableConeCull;
    uint32_t enableOcclusionCull;
    uint32_t hzbLevelCount;
    float    occlusionDepthBias;
    float    occlusionBoundsScale;
};
