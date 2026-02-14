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
    float4   cameraWorldPos;
    uint32_t totalDispatchCount;
    uint32_t instanceCount;
    uint32_t enableFrustumCull;
    uint32_t enableConeCull;
};

// Counter buffer layout (16 bytes, MTL::ResourceStorageModePrivate):
//   offset 0:  atomic counter (uint32)
//   offset 4:  MTLDispatchThreadgroupsIndirectArguments { x, y, z } (3 x uint32)
static constexpr uint32_t kCounterBufferSize = 16;
static constexpr uint32_t kIndirectArgsOffset = 4;
