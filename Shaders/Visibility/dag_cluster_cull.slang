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
    if (maxSize < 1.0) {
        return true;
    }

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
}
