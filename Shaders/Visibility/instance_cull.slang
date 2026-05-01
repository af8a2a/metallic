// GPU instance culling. The hard visibility flag is intentionally frustum-only
// until Phase 1 instance rechecks are wired; cluster-level HZB still handles
// occlusion conservatively via the two-phase cluster cull path.

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

struct GeometryData {
    uint meshletStart;
    uint meshletCount;
    uint primitiveGroupStart;
    uint primitiveGroupCount;
    uint indexStart;
    uint indexCount;
    uint materialIndex;
    uint lodRootNode;
    uint lod0RootNode;
    float boundsCenterX;
    float boundsCenterY;
    float boundsCenterZ;
    float boundsRadius;
    uint packedClusterStart;
    uint packedClusterCount;
    uint reserved0;
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
static const float kFrustumSlack = 1.0e-5;

uint visibleCounterOffset() {
    return params.phase == 0u ? kCounterPhase0Visible : kCounterPhase1Visible;
}

// projectBounds: AABB-corner projection for the diagnostic HZB test.
// (Hard frustum reject uses sphereIntersectsFrustum below instead.)
bool projectBounds(float3 center, float radius, float4x4 worldMatrix,
                   out float4 clipMin, out float4 clipMax, out bool clipValid) {
    return hzbProjectSphere(center, radius, worldMatrix, params.viewProj,
                            /*frustumCull=*/false, clipMin, clipMax, clipValid);
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

// Diagnostic-only HZB test: result is observed but never used for hard rejection.
bool testHiz(float4 clipMin, float4 clipMax, bool clipValid) {
    return hzbTestVisibility(clipMin, clipMax, clipValid,
                             params.enableOcclusion != 0u,
                             params.hzbValid != 0u,
                             params.hzbMipCount,
                             params.screenWidth, params.screenHeight,
                             params.hzbWidth, params.hzbHeight,
                             hzbTextures);
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

    float3 center = float3(geom.boundsCenterX, geom.boundsCenterY, geom.boundsCenterZ);
    float radius = geom.boundsRadius;
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
