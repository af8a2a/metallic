#pragma once

#include <ml.h>
#include <cstdint>

// Shared structs for GPU-driven meshlet culling pipeline.
// Used by both CPU (MeshletCullPass) and GPU (meshlet_cull.slang, visibility_indirect.slang).

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
    uint32_t cullingPhase;     // 0 = phase1, 1 = phase2
    uint32_t _pad0;
};

// Counter buffer layout (32 bytes, StorageModeShared):
//   [0]:  phase1 visible counter (uint32)
//   [4]:  phase1 indirect args { x, y, z } (3 x uint32)
//   [16]: occlusion-failed counter (uint32)
//   [20]: occlusion-failed indirect args { x, y, z } (3 x uint32)
static constexpr uint32_t kCounterBufferSize = 32;
static constexpr uint32_t kIndirectArgsOffset = 4;
static constexpr uint32_t kOccFailedCounterOffset = 16;
static constexpr uint32_t kOccFailedIndirectOffset = 20;
