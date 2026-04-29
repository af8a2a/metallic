// Two-phase cluster occlusion culling for the PackedCluster visualization path.

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

struct MeshletBounds {
    float4 center_radius;
    float4 cone_apex_pad;
    float4 cone_axis_cutoff;
};

struct ClusterCullParams {
    float4x4 viewProj;
    uint inputCount;
    uint dispatchCount;
    uint maxClusters;
    uint phase;
    uint enableOcclusion;
    uint hzbValid;
    uint hzbWidth;
    uint hzbHeight;
    uint hzbMipCount;
    uint screenWidth;
    uint screenHeight;
    uint instanceCount;
    uint useInstanceVisibility;
};

[[vk::push_constant]] ConstantBuffer<ClusterCullParams> params;

[[vk::binding(0)]] StructuredBuffer<ClusterInfo>     inputWorklist;
[[vk::binding(1)]] RWStructuredBuffer<ClusterInfo>   phase0Visible;
[[vk::binding(2)]] RWStructuredBuffer<ClusterInfo>   phase0Recheck;
[[vk::binding(3)]] RWStructuredBuffer<ClusterInfo>   phase1Visible;
[[vk::binding(4)]] StructuredBuffer<MeshletBounds>   clusterBounds;
[[vk::binding(5)]] StructuredBuffer<InstanceData>    instances;
[[vk::binding(6)]] RWByteAddressBuffer               counters;
[[vk::binding(7)]] RWByteAddressBuffer               indirectArgs;
[[vk::binding(8)]] StructuredBuffer<uint>            instanceVisibility;
[[vk::binding(9)]] Texture2D<float>                  hzbTextures[HZB_MAX_LEVELS];
[[vk::binding(10)]] StructuredBuffer<float4x4>       hzbViewProjBuf;

static const uint kCounterPhase0Visible = 0u;
static const uint kCounterPhase0Recheck = 4u;
static const uint kCounterPhase1Visible = 8u;
static const uint kIndirectPhase0 = 0u;
static const uint kIndirectPhase1 = 12u;
static const uint kInstanceVisibleFlag = 1u;
static const float kEpsilon = 1.2e-7;
static const float kDepthNudge = 2.0 / float(1 << 24);

uint inputCountForPhase() {
    if (params.phase == 0u) {
        return min(params.inputCount, params.maxClusters);
    }
    return min(counters.Load(kCounterPhase0Recheck), params.maxClusters);
}

uint visibleCounterOffset() {
    return params.phase == 0u ? kCounterPhase0Visible : kCounterPhase1Visible;
}

uint indirectOffset() {
    return params.phase == 0u ? kIndirectPhase0 : kIndirectPhase1;
}

bool instanceCanRender(uint instanceID) {
    if (instanceID >= params.instanceCount) {
        return false;
    }

    InstanceData inst = instances[instanceID];
    if ((inst.visibilityFlags & kInstanceVisibleFlag) == 0u) {
        return false;
    }

    return params.useInstanceVisibility == 0u || instanceVisibility[instanceID] != 0u;
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

bool projectBounds(ClusterInfo item, float4x4 vp, float4x4 worldMatrix,
                   out float4 clipMin, out float4 clipMax, out bool clipValid) {
    MeshletBounds bounds = clusterBounds[item.clusterID];
    float3 center = bounds.center_radius.xyz;
    float radius = bounds.center_radius.w;

    float4 hPos = mul(vp, mul(worldMatrix, float4(boxCorner(center, radius, 0u), 1.0)));
    uint commonBits = getCullBits(hPos);
    bool valid = hPos.w >= kEpsilon;
    float4 clip = float4(hPos.xyz / max(hPos.w, kEpsilon), hPos.w);
    clipMin = clip;
    clipMax = clip;
    clipValid = valid;

    [unroll]
    for (uint corner = 1u; corner < 8u; ++corner) {
        hPos = mul(vp, mul(worldMatrix, float4(boxCorner(center, radius, corner), 1.0)));
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

uint hzbMipDimension(uint baseDimension, uint mip) {
    uint dimension = max(baseDimension, 1u);
    for (uint level = 0u; level < mip; ++level) {
        dimension = max(1u, (dimension + 1u) / 2u);
    }
    return dimension;
}

int hzbMipLevelForRect(int4 pixelRect) {
    int2 rectSize = max(pixelRect.zw - pixelRect.xy, int2(0, 0));
    int2 dominantMip = int2(firstbithigh(rectSize.x), firstbithigh(rectSize.y));
    int mipLevel = max(max(dominantMip.x, dominantMip.y) - 1, 0);
    int2 shifted = pixelRect.xy >> mipLevel;
    int2 shiftedMax = pixelRect.zw >> mipLevel;
    mipLevel += any(shiftedMax - shifted > int2(3, 3)) ? 1 : 0;
    return mipLevel;
}

bool testHiz(float4 clipMin, float4 clipMax, bool clipValid) {
    if (params.enableOcclusion == 0u || params.hzbValid == 0u || params.hzbMipCount == 0u) {
        return true;
    }
    if (!clipValid) {
        return true;
    }

    float2 uvMin = float2(clipMin.x * 0.5 + 0.5, 0.5 - clipMax.y * 0.5);
    float2 uvMax = float2(clipMax.x * 0.5 + 0.5, 0.5 - clipMin.y * 0.5);
    uvMin = clamp(uvMin, float2(0.0), float2(1.0));
    uvMax = clamp(uvMax, float2(0.0), float2(1.0));

    float2 pixelSize = (uvMax - uvMin) * float2(float(params.screenWidth), float(params.screenHeight));
    if (max(pixelSize.x, pixelSize.y) < 1.0) {
        return true;
    }

    float4 uvRect = float4(uvMin.x, uvMin.y, uvMax.x, uvMax.y);
    float4 screenSize = float4(float(params.screenWidth), float(params.screenHeight),
                               float(params.screenWidth), float(params.screenHeight));
    int4 pixelsRect = int4(uvRect * screenSize + float4(0.5, 0.5, -0.5, -0.5));
    int2 pixelMin = min(pixelsRect.xy, pixelsRect.zw);
    int2 pixelMax = max(pixelsRect.xy, pixelsRect.zw);
    pixelsRect = int4(pixelMin, pixelMax);

    int hzbMip = min(hzbMipLevelForRect(pixelsRect), int(params.hzbMipCount) - 1);
    int4 mipRect = pixelsRect >> hzbMip;

    uint mipIdx = uint(hzbMip);
    int mipW = int(hzbMipDimension(params.hzbWidth, mipIdx));
    int mipH = int(hzbMipDimension(params.hzbHeight, mipIdx));

    int2 maxCoord = int2(mipW - 1, mipH - 1);
    int2 rectMin = clamp(mipRect.xy, int2(0, 0), maxCoord);
    int2 rectMax = clamp(mipRect.zw, int2(0, 0), maxCoord);
    rectMin = min(rectMin, rectMax);

    int4 sampleX = min(rectMin.x + int4(0, 1, 2, 3), int4(rectMax.x));
    int4 sampleY = min(rectMin.y + int4(0, 1, 2, 3), int4(rectMax.y));

    float4 row0 = float4(
        hzbTextures[mipIdx].Load(int3(sampleX.x, sampleY.x, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.y, sampleY.x, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.z, sampleY.x, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.w, sampleY.x, 0)));
    float4 row1 = float4(
        hzbTextures[mipIdx].Load(int3(sampleX.x, sampleY.y, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.y, sampleY.y, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.z, sampleY.y, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.w, sampleY.y, 0)));
    float4 row2 = float4(
        hzbTextures[mipIdx].Load(int3(sampleX.x, sampleY.z, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.y, sampleY.z, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.z, sampleY.z, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.w, sampleY.z, 0)));
    float4 row3 = float4(
        hzbTextures[mipIdx].Load(int3(sampleX.x, sampleY.w, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.y, sampleY.w, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.z, sampleY.w, 0)),
        hzbTextures[mipIdx].Load(int3(sampleX.w, sampleY.w, 0)));

    float4 colMin = min(min(row0, row1), min(row2, row3));
    float minDepth = min(min(colMin.x, colMin.y), min(colMin.z, colMin.w));

    return clipMax.z >= minDepth - kDepthNudge;
}

void appendVisible(ClusterInfo item) {
    uint index;
    counters.InterlockedAdd(visibleCounterOffset(), 1u, index);
    if (index >= params.maxClusters) {
        return;
    }

    if (params.phase == 0u) {
        phase0Visible[index] = item;
    } else {
        phase1Visible[index] = item;
    }
}

void appendRecheck(ClusterInfo item) {
    uint index;
    counters.InterlockedAdd(kCounterPhase0Recheck, 1u, index);
    if (index < params.maxClusters) {
        phase0Recheck[index] = item;
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void resetMain() {
    if (params.phase == 0u) {
        counters.Store(kCounterPhase0Visible, 0u);
        counters.Store(kCounterPhase0Recheck, 0u);
        counters.Store(kCounterPhase1Visible, 0u);
        indirectArgs.Store(kIndirectPhase0 + 0u, 0u);
        indirectArgs.Store(kIndirectPhase0 + 4u, 1u);
        indirectArgs.Store(kIndirectPhase0 + 8u, 1u);
        indirectArgs.Store(kIndirectPhase1 + 0u, 0u);
        indirectArgs.Store(kIndirectPhase1 + 4u, 1u);
        indirectArgs.Store(kIndirectPhase1 + 8u, 1u);
    } else {
        counters.Store(kCounterPhase1Visible, 0u);
        indirectArgs.Store(kIndirectPhase1 + 0u, 0u);
        indirectArgs.Store(kIndirectPhase1 + 4u, 1u);
        indirectArgs.Store(kIndirectPhase1 + 8u, 1u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint itemIndex = dispatchThreadID.x;
    uint count = inputCountForPhase();
    if (itemIndex >= count) {
        return;
    }

    ClusterInfo item = inputWorklist[itemIndex];
    if (!instanceCanRender(item.instanceID)) {
        return;
    }

    InstanceData inst = instances[item.instanceID];

    float4 clipMin, clipMax;
    bool clipValid;
    bool inFrustum = projectBounds(item, params.viewProj, inst.worldMatrix, clipMin, clipMax, clipValid);
    if (!inFrustum) {
        return;
    }

    // HZB test: phase0 uses history VP + prev world matrix; phase1 uses current VP + world matrix.
    float4x4 hzbVP = hzbViewProjBuf[0];
    float4x4 hzbWorld = (params.phase == 0u) ? inst.prevWorldMatrix : inst.worldMatrix;
    float4 hzbClipMin, hzbClipMax;
    bool hzbClipValid;
    projectBounds(item, hzbVP, hzbWorld, hzbClipMin, hzbClipMax, hzbClipValid);

    bool visible = testHiz(hzbClipMin, hzbClipMax, hzbClipValid);
    if (visible) {
        appendVisible(item);
    } else if (params.phase == 0u) {
        appendRecheck(item);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void finalizeMain() {
    uint counterOffset = visibleCounterOffset();
    uint count = min(counters.Load(counterOffset), params.maxClusters);
    counters.Store(counterOffset, count);

    uint offset = indirectOffset();
    indirectArgs.Store(offset + 0u, count);
    indirectArgs.Store(offset + 4u, 1u);
    indirectArgs.Store(offset + 8u, 1u);
}
