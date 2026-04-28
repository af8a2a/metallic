// GPU instance culling. The hard visibility flag is intentionally frustum-only
// until Phase 1 instance rechecks are wired; cluster-level HZB still handles
// occlusion conservatively via the two-phase cluster cull path.

#include "../Shared/cluster_scene.slang"
#include "../../Source/Rendering/hzb_constants.h"

struct InstanceData {
    float4x4 worldMatrix;
    float4x4 prevWorldMatrix;
    uint     geometryIndex;
    uint     dispatchStart;
    uint     sceneNodeIndex;
    uint     visibilityFlags;
};

struct GeometryData {
    uint meshletStart;
    uint meshletCount;
    uint primitiveGroupStart;
    uint primitiveGroupCount;
    uint indexStart;
    uint indexCount;
    uint materialIndex;
    uint lodRootNode;
    float4 boundsCenterRadius;
    uint packedClusterStart;
    uint packedClusterCount;
};

struct InstanceCullParams {
    float4x4 viewProj;
    uint instanceCount;
    uint phase;
    uint enableOcclusion;
    uint hzbValid;
    uint hzbWidth;
    uint hzbHeight;
    uint hzbMipCount;
    uint screenWidth;
    uint screenHeight;
    uint maxInstances;
    uint _pad0;
    uint _pad1;
};

[[vk::push_constant]] ConstantBuffer<InstanceCullParams> params;

[[vk::binding(0)]] StructuredBuffer<InstanceData>    instances;
[[vk::binding(1)]] StructuredBuffer<GeometryData>    geometries;
[[vk::binding(2)]] RWStructuredBuffer<uint>          visibleInstances;
[[vk::binding(3)]] RWByteAddressBuffer               counters;
[[vk::binding(4)]] RWByteAddressBuffer               indirectArgs;
[[vk::binding(5)]] RWStructuredBuffer<uint>          instanceVisibility;
[[vk::binding(6)]] Texture2D<float>                  hzbTextures[HZB_MAX_LEVELS];

// Counter layout: [0]=phase0 visible, [4]=phase0 rejected, [8]=phase1 visible
static const uint kCounterPhase0Visible  = 0u;
static const uint kCounterPhase0Rejected = 4u;
static const uint kCounterPhase1Visible  = 8u;
static const uint kInstanceVisibleFlag   = 1u;
static const float kEpsilon = 1.2e-7;
static const float kDepthNudge = 2.0 / float(1 << 24);
static const float kFrustumSlack = 1.0e-5;

uint visibleCounterOffset() {
    return params.phase == 0u ? kCounterPhase0Visible : kCounterPhase1Visible;
}

uint mipWidth(uint mip) {
    uint d = max(params.hzbWidth, 1u);
    for (uint i = 0u; i < mip; ++i) d = max(1u, (d + 1u) / 2u);
    return d;
}

uint mipHeight(uint mip) {
    uint d = max(params.hzbHeight, 1u);
    for (uint i = 0u; i < mip; ++i) d = max(1u, (d + 1u) / 2u);
    return d;
}

uint getCullBits(float4 hPos) {
    uint bits = 0u;
    bits |= hPos.x < -hPos.w ? 1u : 0u;
    bits |= hPos.x >  hPos.w ? 2u : 0u;
    bits |= hPos.y < -hPos.w ? 4u : 0u;
    bits |= hPos.y >  hPos.w ? 8u : 0u;
    bits |= hPos.z <  0.0    ? 16u : 0u;
    bits |= hPos.z >  hPos.w ? 32u : 0u;
    bits |= hPos.w <= 0.0    ? 64u : 0u;
    return bits;
}

float3 boxCorner(float3 center, float radius, uint corner) {
    return center + float3(
        (corner & 1u) != 0u ? radius : -radius,
        (corner & 2u) != 0u ? radius : -radius,
        (corner & 4u) != 0u ? radius : -radius);
}

bool projectBounds(float3 center, float radius, float4x4 worldMatrix,
                   out float4 clipMin, out float4 clipMax, out bool clipValid) {
    float4 hPos = mul(params.viewProj, mul(worldMatrix, float4(boxCorner(center, radius, 0u), 1.0)));
    uint commonBits = getCullBits(hPos);
    bool valid = hPos.w >= kEpsilon;
    float4 clip = float4(hPos.xyz / max(hPos.w, kEpsilon), hPos.w);
    clipMin = clip;
    clipMax = clip;
    clipValid = valid;

    [unroll]
    for (uint corner = 1u; corner < 8u; ++corner) {
        hPos = mul(params.viewProj, mul(worldMatrix, float4(boxCorner(center, radius, corner), 1.0)));
        commonBits &= getCullBits(hPos);
        valid = hPos.w >= kEpsilon;
        clip = float4(hPos.xyz / max(hPos.w, kEpsilon), hPos.w);
        clipMin = min(clipMin, clip);
        clipMax = max(clipMax, clip);
        clipValid = clipValid && valid;
    }

    clipMin = float4(clamp(clipMin.xy, float2(-1.0), float2(1.0)), clipMin.zw);
    clipMax = float4(clamp(clipMax.xy, float2(-1.0), float2(1.0)), clipMax.zw);
    return commonBits == 0u;
}

bool planeIntersectsSphere(float4x4 localToClip, float3 center, float radius, float4 clipPlane) {
    float4 localPlane = mul(clipPlane, localToClip);
    float centerDistance = dot(localPlane, float4(center, 1.0));
    float supportRadius = radius * length(localPlane.xyz);
    float slack = max(kFrustumSlack, (abs(centerDistance) + supportRadius) * kFrustumSlack);
    return centerDistance + supportRadius >= -slack;
}

bool sphereIntersectsFrustum(float3 center, float radius, float4x4 worldMatrix) {
    float4x4 localToClip = mul(params.viewProj, worldMatrix);

    // D3D/Metal/Vulkan clip volume: -w <= x,y <= w and 0 <= z <= w.
    return planeIntersectsSphere(localToClip, center, radius, float4( 1.0,  0.0,  0.0, 1.0)) &&
           planeIntersectsSphere(localToClip, center, radius, float4(-1.0,  0.0,  0.0, 1.0)) &&
           planeIntersectsSphere(localToClip, center, radius, float4( 0.0,  1.0,  0.0, 1.0)) &&
           planeIntersectsSphere(localToClip, center, radius, float4( 0.0, -1.0,  0.0, 1.0)) &&
           planeIntersectsSphere(localToClip, center, radius, float4( 0.0,  0.0,  1.0, 0.0)) &&
           planeIntersectsSphere(localToClip, center, radius, float4( 0.0,  0.0, -1.0, 1.0));
}

bool testHiz(float4 clipMin, float4 clipMax, bool clipValid) {
    if (params.enableOcclusion == 0u ||
        params.hzbValid == 0u ||
        params.hzbMipCount == 0u ||
        !clipValid) {
        return true;
    }

    float2 uvMin = float2(clipMin.x * 0.5 + 0.5, 0.5 - clipMax.y * 0.5);
    float2 uvMax = float2(clipMax.x * 0.5 + 0.5, 0.5 - clipMin.y * 0.5);
    uvMin = clamp(uvMin, float2(0.0), float2(1.0));
    uvMax = clamp(uvMax, float2(0.0), float2(1.0));

    float2 pixelSize = (uvMax - uvMin) * float2(float(params.screenWidth), float(params.screenHeight));
    float maxSize = max(pixelSize.x, pixelSize.y);
    if (maxSize < 1.0) return true;

    uint mip = min(uint(ceil(log2(max(maxSize * 0.5, 1.0)))), params.hzbMipCount - 1u);
    uint w = mipWidth(mip);
    uint h = mipHeight(mip);
    uint2 maxCoord = uint2(w - 1u, h - 1u);
    float2 mipSize = float2(float(w), float(h));
    uint2 rectMin = min(uint2(floor(uvMin * mipSize)), maxCoord);
    uint2 rectMax = min(uint2(max(ceil(uvMax * mipSize) - float2(1.0), float2(0.0))), maxCoord);
    rectMin = min(rectMin, rectMax);

    float hzbDepth = 1.0;
    [unroll]
    for (uint y = 0u; y < 4u; ++y) {
        uint sampleY = rectMin.y + y;
        if (sampleY > rectMax.y) continue;

        [unroll]
        for (uint x = 0u; x < 4u; ++x) {
            uint sampleX = rectMin.x + x;
            if (sampleX > rectMax.x) continue;
            hzbDepth = min(hzbDepth, hzbTextures[mip].Load(int3(uint2(sampleX, sampleY), 0)));
        }
    }

    return clipMax.z >= hzbDepth - kDepthNudge;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void resetMain() {
    counters.Store(kCounterPhase0Visible, 0u);
    counters.Store(kCounterPhase0Rejected, 0u);
    counters.Store(kCounterPhase1Visible, 0u);
    // Indirect args: (threadgroupsX, 1, 1)
    indirectArgs.Store(0u, 0u);
    indirectArgs.Store(4u, 1u);
    indirectArgs.Store(8u, 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint instIdx = dispatchThreadID.x;
    if (instIdx >= params.instanceCount) return;

    instanceVisibility[instIdx] = 0u;

    InstanceData inst = instances[instIdx];
    if ((inst.visibilityFlags & kInstanceVisibleFlag) == 0u) return;

    GeometryData geom = geometries[inst.geometryIndex];

    float3 center = geom.boundsCenterRadius.xyz;
    float radius = geom.boundsCenterRadius.w;
    if (radius <= 0.0) {
        uint degenerateIndex;
        counters.InterlockedAdd(visibleCounterOffset(), 1u, degenerateIndex);
        if (degenerateIndex < params.maxInstances) {
            visibleInstances[degenerateIndex] = instIdx;
            instanceVisibility[instIdx] = 1u;
        }
        return;
    }

    float4 clipMin, clipMax;
    bool clipValid;
    bool inFrustum = sphereIntersectsFrustum(center, radius, inst.worldMatrix);
    if (!inFrustum) {
        if (params.phase == 0u) {
            // Track rejected for Phase 1 recheck (future use)
            counters.InterlockedAdd(kCounterPhase0Rejected, 1u);
        }
        return;
    }
    projectBounds(center, radius, inst.worldMatrix, clipMin, clipMax, clipValid);

    uint index;
    counters.InterlockedAdd(visibleCounterOffset(), 1u, index);
    if (index < params.maxInstances) {
        visibleInstances[index] = instIdx;
        instanceVisibility[instIdx] = 1u;
    }

    bool hizVisible = testHiz(clipMin, clipMax, clipValid);
    if (!hizVisible && params.phase == 0u) {
        counters.InterlockedAdd(kCounterPhase0Rejected, 1u);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void finalizeMain() {
    uint count = min(counters.Load(visibleCounterOffset()), params.maxInstances);
    counters.Store(visibleCounterOffset(), count);
    // Generate indirect dispatch args: ceil(count / 64) threadgroups
    indirectArgs.Store(0u, (count + 63u) / 64u);
    indirectArgs.Store(4u, 1u);
    indirectArgs.Store(8u, 1u);
}
