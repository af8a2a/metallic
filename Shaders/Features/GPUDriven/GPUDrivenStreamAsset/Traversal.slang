#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Persistent traversal workers and page-demand dispatch.
void runActiveTraversalWorker(
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader> traversalHeader,
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem> traversalWork,
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances,
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> pageWords,
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params)
{
    uint readIndex = 0xffffffffu;
    for (;;) {
        if (WaveActiveAllTrue(readIndex == 0xffffffffu)) {
            uint waveReadIndex = 0u;
            if (WaveIsFirstLane()) {
                InterlockedAdd(traversalHeader[0].readCounter, WaveGetLaneCount(), waveReadIndex);
            }
            waveReadIndex = WaveReadLaneFirst(waveReadIndex) + WaveGetLaneIndex();
            readIndex = waveReadIndex < params.traversalWorkCapacity
                ? waveReadIndex
                : 0xffffffffu;
            if (WaveActiveAllTrue(readIndex == 0xffffffffu)) {
                return;
            }
        }

        bool runnable = false;
        GPUDrivenStreamAssetTraversalWorkItem work;
        for (;;) {
            if (readIndex != 0xffffffffu) {
                uint readyFrame = 0u;
                InterlockedAdd(traversalWork[readIndex].readyFrame, 0u, readyFrame);
                runnable = readyFrame == params.frameIndex + 1u;
                if (runnable) {
                    DeviceMemoryBarrier();
                    work = traversalWork[readIndex];
                }
            }
            if (WaveActiveAnyTrue(runnable)) {
                break;
            }
            uint taskCount = 0u;
            InterlockedAdd(traversalHeader[0].taskCounter, 0u, taskCount);
            if (WaveActiveAnyTrue(taskCount == 0u)) {
                return;
            }
            DeviceMemoryBarrier();
        }

        bool validWork = runnable &&
            work.instanceIndex < params.sceneInstanceCount &&
            work.nodeIndex < params.sceneNodeCount;
        if (runnable && validWork) {
            GPUDrivenStreamAssetInstance instance = instances[work.instanceIndex];
            validWork = instance.visible != 0u && instance.primitiveIndex < params.scenePrimitiveCount;
            if (validWork) {
                GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
                validWork = primitive.nodeCount != 0u &&
                    primitive.nodeOffset < params.sceneNodeCount &&
                    primitive.nodeCount <= params.sceneNodeCount - primitive.nodeOffset &&
                    work.nodeIndex >= primitive.nodeOffset &&
                    work.nodeIndex < primitive.nodeOffset + primitive.nodeCount;
                if (validWork) {
                    GPUDrivenStreamAssetNode node = nodes[work.nodeIndex];
                    validWork = node.primitiveIndex == instance.primitiveIndex;
                    if (validWork && nodeNeedsTraversal(params, instance, primitive, node)) {
                        if (node.childCount != 0u) {
                            bool validChildren = node.childOffset >= primitive.nodeOffset &&
                                node.childOffset < primitive.nodeOffset + primitive.nodeCount &&
                                node.childCount <= primitive.nodeOffset + primitive.nodeCount - node.childOffset;
                            if (validChildren) {
                                for (uint child = node.childCount; child != 0u; --child) {
                                    enqueueActiveTraversalWork(
                                        traversalHeader,
                                        traversalWork,
                                        params,
                                        work.instanceIndex,
                                        node.childOffset + child - 1u);
                                }
                            } else {
                                validWork = false;
                            }
                        } else if (node.groupIndex < params.sceneGroupCount) {
                            bool fallbackGroup = node.groupIndex >= primitive.fallbackGroupOffset &&
                                node.groupIndex - primitive.fallbackGroupOffset < primitive.fallbackGroupCount;
                            GPUDrivenStreamAssetGroup group = groups[node.groupIndex];
                            bool validGroup = group.primitiveIndex == instance.primitiveIndex &&
                                group.clusterCount != 0u &&
                                group.clusterCount <= 32u &&
                                group.pageIndex < params.scenePageCount;
                            if (!validGroup) {
                                validWork = false;
                                reportInvalidTraversal(requests, params);
                            } else if (updateGroupPageDemand(
                                    pageTable,
                                    requests,
                                    params,
                                    instance,
                                    primitive,
                                    group) &&
                                !fallbackGroup &&
                                groupPageDrawable(pageTable, params, group)) {
                                uint clusterSelectionMask = buildGroupClusterSelectionMask(
                                    groups,
                                    pageWords,
                                    pageTable,
                                    params,
                                    instance,
                                    primitive,
                                    group);
                                if (clusterSelectionMask != 0u) {
                                    appendActivePage(
                                        activeGroups,
                                        activeHeader,
                                        pageTable,
                                        params,
                                        instance,
                                        work.instanceIndex,
                                        group,
                                        clusterSelectionMask);
                                }
                            }
                        } else {
                            validWork = false;
                        }
                    }
                }
            }
        }
        if (runnable) {
            if (!validWork) {
                InterlockedAdd(traversalHeader[0].overflowCount, 1u);
            }
            InterlockedAdd(traversalHeader[0].taskCounter, 0xffffffffu);
            readIndex = 0xffffffffu;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetTraversalMain(uint3 dispatchThreadID : SV_DispatchThreadID, uniform GPUDrivenStreamAssetPush push)
{
    uint rowBase = dispatchThreadID.y * (65535u * 64u);
    if (dispatchThreadID.x > 0xffffffffu - rowBase) { return; }
    uint threadIndex = dispatchThreadID.x + rowBase;
    StructuredBuffer<GPUDrivenStreamAssetInstance> instanceHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(push.instanceBuffer));
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitiveHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>(push.primitiveBuffer));
    StructuredBuffer<GPUDrivenStreamAssetLODLevel> lodLevelHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetLODLevel>>(push.lodLevelBuffer));
    StructuredBuffer<GPUDrivenStreamAssetGroup> groupHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>(push.groupBuffer));
    StructuredBuffer<GPUDrivenStreamAssetNode> nodeHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetNode>>(push.nodeBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTableHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(push.pageTableBuffer));
    RWStructuredBuffer<uint> requestHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.requestBuffer));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));

    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives = primitiveHandle;
    StructuredBuffer<GPUDrivenStreamAssetLODLevel> lodLevels = lodLevelHandle;
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups = groupHandle;
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes = nodeHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    RWStructuredBuffer<uint> requests = requestHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    if (push.traversalPhase == kTraversalPhaseClearPriority) {
        uint table = requests[kStreamRequestPriorityTableOffsetWord];
        if (table != 0u && threadIndex < params.scenePageCount) { requests[table + threadIndex] = 0u; }
        return;
    }
    if (push.traversalPhase == kTraversalPhaseGatherPriority) {
        uint table = requests[kStreamRequestPriorityTableOffsetWord];
        uint output = requests[kStreamRequestLoadPriorityOffsetWord];
        uint count = min(requests[kStreamRequestLoadCounterWord], requests[kStreamRequestMaxLoadsWord]);
        if (table != 0u && output != 0u && threadIndex < count) {
            uint page = requests[kStreamRequestHeaderWords + threadIndex];
            if (requests[kStreamRequestPrefetchLimitWord] != 0u) { page &= ~kStreamPrefetchPageTag; }
            requests[output + threadIndex] = page < params.scenePageCount ? requests[table + page] : 0u;
        }
        return;
    }
    if (push.traversalPhase == kTraversalPhaseUnload) {
        uint residentPageIndex = threadIndex;
        if (residentPageIndex >= push.activeBuildPhase || params.enableGpuUnloadRequests == 0u) {
            return;
        }
        StructuredBuffer<uint> residentPageHandle =
            resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.residentPageBuffer));
        StructuredBuffer<uint> residentPages = residentPageHandle;
        uint pageIndex = residentPages[residentPageIndex];
        if (pageIndex >= params.scenePageCount) {
            reportInvalidTraversal(requests, params);
            return;
        }
        GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[pageIndex];
        if (streamPageState(pageEntry) == kStreamStateResident &&
            pageEntry.lastRequestFrame != params.frameIndex) {
            requestStreamPageUnload(requests, params, pageIndex);
        }
        return;
    }

    if (params.sceneNodeCount != 0u) {
        uint workerCount = max(params.traversalWorkerCount, 1u);
        if (threadIndex >= workerCount) {
            return;
        }
        for (uint instanceIndex = threadIndex;
             instanceIndex < params.sceneInstanceCount;
             instanceIndex += workerCount) {
            GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
            if (instance.visible != 0u) {
                traverseInstancePageDemand(
                    primitives,
                    groups,
                    nodes,
                    pageTable,
                    requests,
                    params,
                    instance);
            }
        }
        return;
    }

    if (params.maxPrimitiveGroupCount == 0u) {
        return;
    }
    uint workItemIndex = threadIndex;
    uint instanceIndex = workItemIndex / params.maxPrimitiveGroupCount;
    uint localGroupIndex = workItemIndex - instanceIndex * params.maxPrimitiveGroupCount;
    if (instanceIndex >= params.sceneInstanceCount) {
        return;
    }

    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    if (instance.visible == 0u) {
        return;
    }
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        InterlockedAdd(requests[kStreamRequestInvalidPageCounterWord], 1u);
        InterlockedMax(requests[kStreamRequestLastInvalidPageFrameWord], params.frameIndex);
        return;
    }

    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    if (primitive.groupCount == 0u ||
        localGroupIndex >= primitive.groupCount ||
        primitive.groupOffset >= params.sceneGroupCount ||
        primitive.groupCount > params.sceneGroupCount - primitive.groupOffset) {
        if (localGroupIndex >= primitive.groupCount) {
            return;
        }
        InterlockedAdd(requests[kStreamRequestInvalidPageCounterWord], 1u);
        InterlockedMax(requests[kStreamRequestLastInvalidPageFrameWord], params.frameIndex);
        return;
    }

    uint groupIndex = primitive.groupOffset + localGroupIndex;
    GPUDrivenStreamAssetGroup group = groups[groupIndex];
    if (group.primitiveIndex != instance.primitiveIndex ||
        group.pageIndex >= params.scenePageCount) {
        InterlockedAdd(requests[kStreamRequestInvalidPageCounterWord], 1u);
        InterlockedMax(requests[kStreamRequestLastInvalidPageFrameWord], params.frameIndex);
        return;
    }
    if (!groupNeedsTraversal(params, instance, primitive, group)) {
        return;
    }
    uint state = streamPageState(pageTable[group.pageIndex]);
    if (streamPageAvailable(state)) {
        markStreamPageDemand(pageTable, params, group.pageIndex);
    } else {
        requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
    }
}

