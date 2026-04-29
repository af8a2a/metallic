// GPU DAG traversal for the ClusterLOD visualization path.

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
    float boundsCenterX;
    float boundsCenterY;
    float boundsCenterZ;
    float boundsRadius;
    uint packedClusterStart;
    uint packedClusterCount;
};

struct MeshletBounds {
    float4 center_radius;
    float4 cone_apex_pad;
    float4 cone_axis_cutoff;
};

struct GPULodNode {
    float3 center;
    float  radius;
    float  maxError;
    uint   childOffset;
    uint   childCount;
    uint   isLeaf;
};

struct GPUClusterGroup {
    float3 center;
    float  radius;
    float  error;
    float  parentError;
    uint   clusterStart;
    uint   clusterCount;
};

struct ClusterInfo {
    uint instanceID;
    uint clusterID;
};

struct DagNodeTask {
    uint instanceID;
    uint nodeID;
};

struct DagCullParams {
    float4x4 viewProj;
    uint instanceCount;
    uint maxClusters;
    uint maxNodeTasks;
    uint phase;
    uint mode;
    uint iteration;
    uint enableOcclusion;
    uint hzbValid;
    uint hzbWidth;
    uint hzbHeight;
    uint hzbMipCount;
    uint screenWidth;
    uint screenHeight;
    uint useInstanceVisibility;
    uint enableFrustum;
    uint maxIterations;
};

[[vk::push_constant]] ConstantBuffer<DagCullParams> params;

[[vk::binding(0)]]  StructuredBuffer<InstanceData>       instances;
[[vk::binding(1)]]  StructuredBuffer<GeometryData>       geometries;
[[vk::binding(2)]]  StructuredBuffer<MeshletBounds>      clusterBounds;
[[vk::binding(3)]]  StructuredBuffer<GPULodNode>         lodNodes;
[[vk::binding(4)]]  StructuredBuffer<GPUClusterGroup>    lodGroups;
[[vk::binding(5)]]  StructuredBuffer<uint>               groupMeshletIndices;
[[vk::binding(6)]]  RWStructuredBuffer<ClusterInfo>      phase0Visible;
[[vk::binding(7)]]  RWStructuredBuffer<ClusterInfo>      phase0Recheck;
[[vk::binding(8)]]  RWStructuredBuffer<ClusterInfo>      phase1Visible;
[[vk::binding(9)]]  RWByteAddressBuffer                  counters;
[[vk::binding(10)]] RWByteAddressBuffer                  indirectArgs;
[[vk::binding(11)]] StructuredBuffer<uint>               instanceVisibility;
[[vk::binding(12)]] RWStructuredBuffer<DagNodeTask>      nodeQueue0;
[[vk::binding(13)]] RWStructuredBuffer<DagNodeTask>      nodeQueue1;
[[vk::binding(14)]] Texture2D<float>                     hzbTextures[HZB_MAX_LEVELS];

static const uint kCounterPhase0Visible = 0u;
static const uint kCounterPhase0Recheck = 4u;
static const uint kCounterPhase1Visible = 8u;
static const uint kDagQueueCount0 = 16u;
static const uint kDagQueueCount1 = 20u;
static const uint kDagNodeOverflow = 24u;
static const uint kDagNodeProcessed = 28u;
static const uint kDagClusterOverflow = 32u;
static const uint kDagSeededInstances = 36u;
static const uint kDagFrustumRejected = 40u;
static const uint kDagHzbRejected = 44u;
static const uint kDagHzbRecheck = 48u;
static const uint kDagInvalidRoot = 52u;
static const uint kDagMaxIterRemaining = 56u;
static const uint kDagLastActiveIteration = 60u;
static const uint kIndirectPhase0 = 0u;
static const uint kIndirectPhase1 = 12u;
static const uint kInstanceVisibleFlag = 1u;
static const uint kInvalidIndex = 0xFFFFFFFFu;
static const uint kModeSeedNodes = 1u;
static const uint kModeClearNextQueue = 2u;
static const uint kModeProcessNodes = 3u;
static const uint kModeProcessRecheck = 4u;
static const float kEpsilon = 1.2e-7;
static const float kDepthNudge = 2.0 / float(1 << 24);

uint visibleCounterOffset() {
    return params.phase == 0u ? kCounterPhase0Visible : kCounterPhase1Visible;
}

uint indirectOffset() {
    return params.phase == 0u ? kIndirectPhase0 : kIndirectPhase1;
}

uint queueCountOffset(uint queueIndex) {
    return (queueIndex & 1u) == 0u ? kDagQueueCount0 : kDagQueueCount1;
}

uint inputQueueIndex() {
    return params.iteration & 1u;
}

uint outputQueueIndex() {
    return (params.iteration + 1u) & 1u;
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

bool projectSphereBounds(float3 center,
                         float radius,
                         float4x4 worldMatrix,
                         out float4 clipMin,
                         out float4 clipMax,
                         out bool clipValid) {
    if (radius <= 0.0) {
        clipMin = float4(0.0, 0.0, 0.0, 0.0);
        clipMax = float4(0.0, 0.0, 0.0, 0.0);
        clipValid = false;
        return true;
    }

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
    return params.enableFrustum == 0u || commonBits == 0u;
}

// Returns the mip level whose texels cover a 4-texel footprint over the given
// pixel rect (already shifted right by 1 to match the HZB half-res base).
// Mirrors Nyx MipLevelForRect with HZB_TARGET_FOOTPRINT_PIXELS=4.
int hzbMipLevelForRect(int4 coarseRect) {
    // footprintMipBias = log2(4) - 1 = 1
    int2 rectSize = coarseRect.zw - coarseRect.xy;
    int2 dominantMip = int2(firstbithigh(rectSize.x), firstbithigh(rectSize.y));
    int mipLevel = max(max(dominantMip.x, dominantMip.y) - 1, 0);
    int2 shifted = coarseRect.xy >> mipLevel;
    int2 shiftedMax = coarseRect.zw >> mipLevel;
    // If the 4-texel span is exceeded, go one mip coarser.
    mipLevel += any(shiftedMax - shifted > int2(3, 3)) ? 1 : 0;
    return mipLevel;
}

bool testHiz(float4 clipMin, float4 clipMax, bool clipValid) {
    if (params.enableOcclusion == 0u || params.hzbValid == 0u || params.hzbMipCount == 0u) {
        return true;
    }
    // Near-plane straddle: conservative, treat as visible.
    if (!clipValid) {
        return true;
    }

    float sw = float(params.screenWidth);
    float sh = float(params.screenHeight);

    // Convert clip XY to UV, flipping Y (clip +Y=up, UV +Y=down).
    // saturate clamps to [0,1] before pixel conversion.
    float4 uvRect = saturate(float4(clipMin.x, clipMax.y, clipMax.x, clipMin.y) * 0.5 + 0.5);
    // uvRect = (minU, minV, maxU, maxV)

    // Pixel rect with rounding bias: min rounds up (+0.5), max rounds down (-0.5).
    int4 pixelsRect = int4(uvRect * float4(sw, sh, sw, sh) + float4(0.5, 0.5, -0.5, -0.5));
    pixelsRect.xy = min(pixelsRect.xy, pixelsRect.zw);
    pixelsRect.zw = max(pixelsRect.xy, pixelsRect.zw);

    // Small-bounds guard: sub-pixel footprint, skip HZB.
    if (pixelsRect.z < pixelsRect.x || pixelsRect.w < pixelsRect.y) {
        return true;
    }

    // HZB is half-res relative to the depth buffer.
    int4 coarseRect = pixelsRect >> 1;
    int hzbMip = min(hzbMipLevelForRect(coarseRect), int(params.hzbMipCount) - 1);
    coarseRect >>= hzbMip;

    uint mipIdx = uint(hzbMip);
    uint baseW = max(params.hzbWidth, 1u);
    uint baseH = max(params.hzbHeight, 1u);
    int mipW = int(max(baseW >> mipIdx, 1u));
    int mipH = int(max(baseH >> mipIdx, 1u));

    // 4x4 sample grid clamped to mip dimensions.
    int4 sampleX = clamp(coarseRect.x + int4(0, 1, 2, 3), int4(0), int4(min(coarseRect.z, mipW - 1)));
    int4 sampleY = clamp(coarseRect.y + int4(0, 1, 2, 3), int4(0), int4(min(coarseRect.w, mipH - 1)));

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

    // Reversed-Z: min = farthest sample = most conservative occluder.
    float4 colMin = min(min(row0, row1), min(row2, row3));
    float minDepth = min(min(colMin.x, colMin.y), min(colMin.z, colMin.w));

    // Visible if object's nearest depth (clipMax.z in reversed-Z) >= HZB min depth.
    return clipMax.z >= minDepth - kDepthNudge;
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

DagNodeTask loadTask(uint queueIndex, uint taskIndex) {
    return (queueIndex & 1u) == 0u ? nodeQueue0[taskIndex] : nodeQueue1[taskIndex];
}

void storeTask(uint queueIndex, uint taskIndex, DagNodeTask task) {
    if ((queueIndex & 1u) == 0u) {
        nodeQueue0[taskIndex] = task;
    } else {
        nodeQueue1[taskIndex] = task;
    }
}

void appendNode(uint queueIndex, DagNodeTask task) {
    uint index;
    counters.InterlockedAdd(queueCountOffset(queueIndex), 1u, index);
    if (index < params.maxNodeTasks) {
        storeTask(queueIndex, index, task);
    } else {
        uint ignored;
        counters.InterlockedAdd(kDagNodeOverflow, 1u, ignored);
    }
}

void appendVisible(ClusterInfo item) {
    uint index;
    counters.InterlockedAdd(visibleCounterOffset(), 1u, index);
    if (index >= params.maxClusters) {
        uint ignored;
        counters.InterlockedAdd(kDagClusterOverflow, 1u, ignored);
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
    } else {
        uint ignored;
        counters.InterlockedAdd(kDagClusterOverflow, 1u, ignored);
    }
}

void processCluster(uint instanceID, uint clusterID) {
    InstanceData inst = instances[instanceID];
    MeshletBounds bounds = clusterBounds[clusterID];

    float4 clipMin;
    float4 clipMax;
    bool clipValid;
    bool inFrustum = projectSphereBounds(
        bounds.center_radius.xyz,
        bounds.center_radius.w,
        inst.worldMatrix,
        clipMin,
        clipMax,
        clipValid);
    if (!inFrustum) {
        uint ignored;
        counters.InterlockedAdd(kDagFrustumRejected, 1u, ignored);
        return;
    }

    if (testHiz(clipMin, clipMax, clipValid)) {
        ClusterInfo item;
        item.instanceID = instanceID;
        item.clusterID = clusterID;
        appendVisible(item);
    } else if (params.phase == 0u) {
        ClusterInfo item;
        item.instanceID = instanceID;
        item.clusterID = clusterID;
        appendRecheck(item);
        uint ignored;
        counters.InterlockedAdd(kDagHzbRecheck, 1u, ignored);
    } else {
        uint ignored;
        counters.InterlockedAdd(kDagHzbRejected, 1u, ignored);
    }
}

void emitGroupClusters(uint instanceID, uint groupID) {
    GPUClusterGroup group = lodGroups[groupID];
    for (uint i = 0u; i < group.clusterCount; ++i) {
        processCluster(instanceID, groupMeshletIndices[group.clusterStart + i]);
    }
}

void processGroup(uint instanceID, uint groupID) {
    emitGroupClusters(instanceID, groupID);
}

void seedNodes(uint3 dispatchThreadID) {
    uint instanceID = dispatchThreadID.x;
    if (!instanceCanRender(instanceID)) {
        return;
    }

    InstanceData inst = instances[instanceID];
    GeometryData geom = geometries[inst.geometryIndex];
    if (geom.lodRootNode == kInvalidIndex) {
        uint ignored;
        counters.InterlockedAdd(kDagInvalidRoot, 1u, ignored);
        return;
    }

    DagNodeTask task;
    task.instanceID = instanceID;
    task.nodeID = geom.lodRootNode;
    appendNode(0u, task);

    uint ignored;
    counters.InterlockedAdd(kDagSeededInstances, 1u, ignored);
}

void clearNextQueue() {
    counters.Store(queueCountOffset(outputQueueIndex()), 0u);
}

void processNodes(uint3 dispatchThreadID) {
    uint taskIndex = dispatchThreadID.x;
    uint inQueue = inputQueueIndex();
    uint inputCount = min(counters.Load(queueCountOffset(inQueue)), params.maxNodeTasks);
    if (taskIndex >= inputCount) {
        return;
    }

    // Track the highest iteration that had live work, for maxIterRemaining.
    if (taskIndex == 0u) {
        uint ignored;
        counters.InterlockedMax(kDagLastActiveIteration, params.iteration + 1u, ignored);
    }

    DagNodeTask task = loadTask(inQueue, taskIndex);
    if (!instanceCanRender(task.instanceID)) {
        return;
    }

    GPULodNode node = lodNodes[task.nodeID];

    uint ignored;
    counters.InterlockedAdd(kDagNodeProcessed, 1u, ignored);

    if (node.isLeaf != 0u) {
        for (uint child = 0u; child < node.childCount; ++child) {
            processGroup(task.instanceID, node.childOffset + child);
        }
        return;
    }

    uint outQueue = outputQueueIndex();
    for (uint child = 0u; child < node.childCount; ++child) {
        DagNodeTask childTask;
        childTask.instanceID = task.instanceID;
        childTask.nodeID = node.childOffset + child;
        appendNode(outQueue, childTask);
    }
}

void processRecheck(uint3 dispatchThreadID) {
    uint itemIndex = dispatchThreadID.x;
    uint recheckCount = min(counters.Load(kCounterPhase0Recheck), params.maxClusters);
    if (itemIndex >= recheckCount) {
        return;
    }

    ClusterInfo item = phase0Recheck[itemIndex];
    if (!instanceCanRender(item.instanceID)) {
        return;
    }

    InstanceData inst = instances[item.instanceID];
    MeshletBounds bounds = clusterBounds[item.clusterID];

    float4 clipMin;
    float4 clipMax;
    bool clipValid;
    bool inFrustum = projectSphereBounds(
        bounds.center_radius.xyz,
        bounds.center_radius.w,
        inst.worldMatrix,
        clipMin,
        clipMax,
        clipValid);
    if (!inFrustum) {
        return;
    }

    if (testHiz(clipMin, clipMax, clipValid)) {
        appendVisible(item);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void resetMain() {
    if (params.phase == 0u) {
        counters.Store(kCounterPhase0Visible, 0u);
        counters.Store(kCounterPhase0Recheck, 0u);
        counters.Store(kCounterPhase1Visible, 0u);
        counters.Store(kDagQueueCount0, 0u);
        counters.Store(kDagQueueCount1, 0u);
        counters.Store(kDagNodeOverflow, 0u);
        counters.Store(kDagNodeProcessed, 0u);
        counters.Store(kDagClusterOverflow, 0u);
        counters.Store(kDagSeededInstances, 0u);
        counters.Store(kDagFrustumRejected, 0u);
        counters.Store(kDagHzbRejected, 0u);
        counters.Store(kDagHzbRecheck, 0u);
        counters.Store(kDagInvalidRoot, 0u);
        counters.Store(kDagMaxIterRemaining, 0u);
        counters.Store(kDagLastActiveIteration, 0u);
        indirectArgs.Store(kIndirectPhase0 + 0u, 0u);
        indirectArgs.Store(kIndirectPhase0 + 4u, 1u);
        indirectArgs.Store(kIndirectPhase0 + 8u, 1u);
        indirectArgs.Store(kIndirectPhase1 + 0u, 0u);
        indirectArgs.Store(kIndirectPhase1 + 4u, 1u);
        indirectArgs.Store(kIndirectPhase1 + 8u, 1u);
    } else {
        counters.Store(kCounterPhase1Visible, 0u);
        counters.Store(kDagHzbRejected, 0u);
        indirectArgs.Store(kIndirectPhase1 + 0u, 0u);
        indirectArgs.Store(kIndirectPhase1 + 4u, 1u);
        indirectArgs.Store(kIndirectPhase1 + 8u, 1u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    if (params.mode == kModeSeedNodes) {
        seedNodes(dispatchThreadID);
    } else if (params.mode == kModeClearNextQueue) {
        if (dispatchThreadID.x == 0u) {
            clearNextQueue();
        }
    } else if (params.mode == kModeProcessNodes) {
        processNodes(dispatchThreadID);
    } else if (params.mode == kModeProcessRecheck) {
        processRecheck(dispatchThreadID);
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

    if (params.phase == 0u) {
        uint lastActive = counters.Load(kDagLastActiveIteration);
        uint remaining = params.maxIterations > lastActive ? params.maxIterations - lastActive : 0u;
        counters.Store(kDagMaxIterRemaining, remaining);
    }
}
