// Two-phase cluster occlusion culling for the PackedCluster visualization path.

#include "../Shared/cluster_scene.slang"
#include "../Shared/hzb_cull.slang"

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

// projectBounds: projects a cluster's sphere bounds into clip space.
// Frustum culling is always enabled for the flat cluster path (no enableFrustum flag).
bool projectBounds(ClusterInfo item, float4x4 vp, float4x4 worldMatrix,
                   out float4 clipMin, out float4 clipMax, out bool clipValid) {
    MeshletBounds bounds = clusterBounds[item.clusterID];
    return hzbProjectSphere(bounds.center_radius.xyz, bounds.center_radius.w,
                            worldMatrix, vp, /*frustumCull=*/true,
                            clipMin, clipMax, clipValid);
}

bool testHiz(float4 clipMin, float4 clipMax, bool clipValid) {
    return hzbTestVisibility(clipMin, clipMax, clipValid,
                             params.enableOcclusion != 0u,
                             params.hzbValid != 0u,
                             params.hzbMipCount,
                             params.screenWidth, params.screenHeight,
                             params.hzbWidth, params.hzbHeight,
                             hzbTextures);
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
