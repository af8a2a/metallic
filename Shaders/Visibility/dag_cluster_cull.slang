// GPU DAG traversal for the ClusterLOD visualization path.

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
    // Packed flags: bit0=enableOcclusion, bit1=hzbValid, bit2=useInstanceVisibility, bit3=enableFrustum
    uint flags;
    uint packedHzbDims;    // hzbWidth<<16 | hzbHeight
    uint hzbMipCount;
    uint packedScreenDims; // screenWidth<<16 | screenHeight
    uint maxIterations;
    float3 cameraWorldPos;
    float cameraFovY;
    float lodErrorThreshold;
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
[[vk::binding(15)]] StructuredBuffer<float4x4>           hzbViewProjBuf;

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
static const uint kDagLodCulled = 64u;
static const uint kIndirectPhase0 = 0u;
static const uint kIndirectPhase1 = 12u;
static const uint kIndirectNodeDispatch = 24u;
static const uint kInstanceVisibleFlag = 1u;
static const uint kInvalidIndex = 0xFFFFFFFFu;
static const uint kModeSeedNodes = 1u;
static const uint kModeClearNextQueue = 2u;
static const uint kModeProcessNodes = 3u;
static const uint kModeProcessRecheck = 4u;
static const uint kModeWriteNodeIndirect = 5u;

bool flagEnabled(uint bit) { return (params.flags & (1u << bit)) != 0u; }
uint hzbWidth()       { return params.packedHzbDims >> 16u; }
uint hzbHeight()      { return params.packedHzbDims & 0xFFFFu; }
uint screenWidth()    { return params.packedScreenDims >> 16u; }
uint screenHeight()   { return params.packedScreenDims & 0xFFFFu; }

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

// Thin wrappers so call sites keep their original names.

bool projectSphereBounds(float3 center, float radius,
                         float4x4 worldMatrix, float4x4 vp,
                         out float4 clipMin, out float4 clipMax, out bool clipValid) {
    bool frustumCull = flagEnabled(3u);
    return hzbProjectSphere(center, radius, worldMatrix, vp, frustumCull,
                            clipMin, clipMax, clipValid);
}

bool testHiz(float4 clipMin, float4 clipMax, bool clipValid) {
    return hzbTestVisibility(clipMin, clipMax, clipValid,
                             flagEnabled(0u),
                             flagEnabled(1u),
                             params.hzbMipCount,
                             screenWidth(), screenHeight(),
                             hzbWidth(), hzbHeight(),
                             hzbTextures);
}

bool instanceCanRender(uint instanceID) {
    if (instanceID >= params.instanceCount) {
        return false;
    }

    InstanceData inst = instances[instanceID];
    if ((inst.visibilityFlags & kInstanceVisibleFlag) == 0u) {
        return false;
    }

    return !flagEnabled(2u) || instanceVisibility[instanceID] != 0u;
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
        params.viewProj,
        clipMin,
        clipMax,
        clipValid);
    if (!inFrustum) {
        uint ignored;
        counters.InterlockedAdd(kDagFrustumRejected, 1u, ignored);
        return;
    }

    // For HZB test: phase0 uses history VP + prev world matrix; phase1 uses current VP + world matrix.
    float4x4 hzbVP = hzbViewProjBuf[0];
    float4x4 hzbWorld = (params.phase == 0u) ? inst.prevWorldMatrix : inst.worldMatrix;
    float4 hzbClipMin, hzbClipMax;
    bool hzbClipValid;
    projectSphereBounds(
        bounds.center_radius.xyz,
        bounds.center_radius.w,
        hzbWorld,
        hzbVP,
        hzbClipMin,
        hzbClipMax,
        hzbClipValid);

    if (testHiz(hzbClipMin, hzbClipMax, hzbClipValid)) {
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

void writeNodeIndirect() {
    uint inQueue = inputQueueIndex();
    uint count = min(counters.Load(queueCountOffset(inQueue)), params.maxNodeTasks);
    indirectArgs.Store(kIndirectNodeDispatch + 0u, (count + 63u) / 64u);
    indirectArgs.Store(kIndirectNodeDispatch + 4u, 1u);
    indirectArgs.Store(kIndirectNodeDispatch + 8u, 1u);
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
        params.viewProj,
        clipMin,
        clipMax,
        clipValid);
    if (!inFrustum) {
        return;
    }

    // Phase1: HZB VP is current frame VP (same as frustum VP).
    float4x4 hzbVP = hzbViewProjBuf[0];
    float4 hzbClipMin, hzbClipMax;
    bool hzbClipValid;
    projectSphereBounds(
        bounds.center_radius.xyz,
        bounds.center_radius.w,
        inst.worldMatrix,
        hzbVP,
        hzbClipMin,
        hzbClipMax,
        hzbClipValid);

    if (testHiz(hzbClipMin, hzbClipMax, hzbClipValid)) {
        appendVisible(item);
    } else {
        uint ignored;
        counters.InterlockedAdd(kDagHzbRejected, 1u, ignored);
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
        counters.Store(kDagLodCulled, 0u);
        indirectArgs.Store(kIndirectPhase0 + 0u, 0u);
        indirectArgs.Store(kIndirectPhase0 + 4u, 1u);
        indirectArgs.Store(kIndirectPhase0 + 8u, 1u);
        indirectArgs.Store(kIndirectPhase1 + 0u, 0u);
        indirectArgs.Store(kIndirectPhase1 + 4u, 1u);
        indirectArgs.Store(kIndirectPhase1 + 8u, 1u);
        indirectArgs.Store(kIndirectNodeDispatch + 0u, 0u);
        indirectArgs.Store(kIndirectNodeDispatch + 4u, 1u);
        indirectArgs.Store(kIndirectNodeDispatch + 8u, 1u);
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
    } else if (params.mode == kModeWriteNodeIndirect) {
        if (dispatchThreadID.x == 0u) {
            writeNodeIndirect();
        }
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
