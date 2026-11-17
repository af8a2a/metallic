#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Drawable cut selection, active-page emission and fallback traversal.
uint buildGroupClusterSelectionMask(
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> pageWords,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetPrimitive primitive,
    GPUDrivenStreamAssetGroup group)
{
    if (!groupNeedsTraversal(params, instance, primitive, group)) {
        return 0u;
    }

    uint clusterCount = min(group.clusterCount, 32u);
    if (params.enableGpuLodSelection == 0u && params.selectedLodLevel != kNoDebugLODOverride) {
        return clusterCount == 32u ? 0xffffffffu : ((1u << clusterCount) - 1u);
    }

    GPUDrivenStreamAssetPageTableEntry currentPage = pageTable[group.pageIndex];
    uint pageWord = 0u;
    uint payloadClusterCount = 0u;
    uint clusterOffsetBytes = 0u;
    uint payloadBytes = 0u;
    bool clusterDirectoryAvailable = false;
    uint currentPageOffset = streamPageDeviceOffset(currentPage);
    if (currentPageOffset != kInvalidStreamDeviceOffset &&
        params.pageBufferBytes >= kPayloadHeaderBytes &&
        currentPageOffset <= params.pageBufferBytes - kPayloadHeaderBytes) {
        pageWord = currentPageOffset / 4u;
        payloadClusterCount = pageWords[pageWord + 2u];
        clusterOffsetBytes = pageWords[pageWord + 9u];
        payloadBytes = pageWords[pageWord + 12u];
        clusterDirectoryAvailable = payloadBytes >= kPayloadHeaderBytes &&
            payloadBytes <= params.pageBufferBytes &&
            currentPageOffset <= params.pageBufferBytes - payloadBytes &&
            (clusterOffsetBytes & 3u) == 0u &&
            streamPayloadArrayRangeValid(
                clusterOffsetBytes,
                payloadClusterCount,
                kPayloadClusterBytes,
                payloadBytes);
    }

    uint mask = 0u;
    for (uint clusterChild = 0u; clusterChild < 32u; ++clusterChild) {
        if (clusterChild >= clusterCount) {
            break;
        }
        uint refinedGroupIndex = kInvalidClusterIndex;
        if (clusterDirectoryAvailable) {
            uint refinedGroupByteOffset =
                clusterOffsetBytes + clusterChild * kPayloadClusterBytes + 32u;
            if (clusterChild < payloadClusterCount &&
                refinedGroupByteOffset >= clusterOffsetBytes &&
                refinedGroupByteOffset + 4u >= refinedGroupByteOffset &&
                refinedGroupByteOffset + 4u <= payloadBytes) {
                refinedGroupIndex = pageWords[pageWord + refinedGroupByteOffset / 4u];
            }
        }
        bool renderCluster = refinedGroupIndex == kInvalidClusterIndex;
        if (!renderCluster) {
            if (refinedGroupIndex >= params.sceneGroupCount) {
                renderCluster = true;
            } else {
                GPUDrivenStreamAssetGroup refinedGroup = groups[refinedGroupIndex];
                renderCluster = refinedGroup.primitiveIndex != group.primitiveIndex ||
                    !groupPageDrawable(pageTable, params, refinedGroup) ||
                    !groupNeedsTraversal(params, instance, primitive, refinedGroup);
            }
        }
        if (renderCluster) {
            mask |= 1u << clusterChild;
        }
    }
    return mask;
}

void appendActivePage(
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    uint instanceIndex,
    GPUDrivenStreamAssetGroup sourceGroup,
    uint clusterSelectionMask)
{
    uint pageIndex = sourceGroup.pageIndex;
    if (pageIndex >= params.scenePageCount) {
        return;
    }

    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[pageIndex];
    uint pageDeviceOffset = streamPageDeviceOffset(pageEntry);
    if (!streamPageDrawable(streamPageState(pageEntry)) ||
        pageDeviceOffset == kInvalidStreamDeviceOffset) {
        return;
    }

    uint activeIndex = 0u;
    InterlockedAdd(activeHeader[0].activeGroupCount, 1u, activeIndex);
    if (activeIndex >= activeHeader[0].activeGroupCapacity) {
        InterlockedAdd(activeHeader[0].overflowCount, 1u);
        return;
    }

    GPUDrivenStreamAssetActiveGroup group;
    group.pageDeviceOffsetBytes = pageDeviceOffset;
    group.pageIndex = pageIndex;
    group.clusterCount = sourceGroup.clusterCount;
    group.primitiveIndex = sourceGroup.primitiveIndex;
    group.lodLevel = sourceGroup.lodLevel;
    group.materialIndex = instance.materialIndex;
    group.clusterSelectionMask = clusterSelectionMask;
    group.flags = kActiveGroupResident;
    group.instanceIndex = instanceIndex;
    group.gpuSceneInstanceIndex = instance.gpuSceneInstanceIndex;
    group.padding1 = 0u;
    group.padding2 = 0u;
    group.world0 = instance.world0;
    group.world1 = instance.world1;
    group.world2 = instance.world2;
    group.world3 = instance.world3;
    activeGroups[activeIndex] = group;
}

bool pageRangeDrawable(
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    uint pageOffset,
    uint pageCount)
{
    for (uint page = 0u; page < pageCount; ++page) {
        uint pageIndex = pageOffset + page;
        if (pageIndex >= params.scenePageCount ||
            !streamPageDrawable(streamPageState(pageTable[pageIndex]))) {
            return false;
        }
    }
    return true;
}

void traverseInstancePageDemand(
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance)
{
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        reportInvalidTraversal(requests, params);
        return;
    }

    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    if (primitive.nodeCount == 0u ||
        primitive.nodeOffset >= params.sceneNodeCount ||
        primitive.nodeCount > params.sceneNodeCount - primitive.nodeOffset) {
        reportInvalidTraversal(requests, params);
        return;
    }

    uint stack[kTraversalStackCapacity];
    uint stackCount = 1u;
    stack[0] = primitive.nodeOffset;
    while (stackCount != 0u) {
        uint nodeIndex = stack[--stackCount];
        if (nodeIndex < primitive.nodeOffset ||
            nodeIndex >= primitive.nodeOffset + primitive.nodeCount ||
            nodeIndex >= params.sceneNodeCount) {
            reportInvalidTraversal(requests, params);
            return;
        }

        GPUDrivenStreamAssetNode node = nodes[nodeIndex];
        if (node.primitiveIndex != instance.primitiveIndex ||
            !nodeNeedsTraversal(params, instance, primitive, node)) {
            continue;
        }
        if (node.childCount != 0u) {
            if (node.childOffset < primitive.nodeOffset ||
                node.childOffset >= primitive.nodeOffset + primitive.nodeCount ||
                node.childCount > primitive.nodeOffset + primitive.nodeCount - node.childOffset ||
                node.childCount > kTraversalStackCapacity - stackCount) {
                reportInvalidTraversal(requests, params);
                return;
            }
            for (uint child = 0u; child < node.childCount; ++child) {
                stack[stackCount++] = node.childOffset + child;
            }
            continue;
        }

        if (node.groupIndex >= params.sceneGroupCount) {
            reportInvalidTraversal(requests, params);
            return;
        }
        GPUDrivenStreamAssetGroup group = groups[node.groupIndex];
        if (group.primitiveIndex != instance.primitiveIndex ||
            group.pageIndex >= params.scenePageCount ||
            !groupNeedsTraversal(params, instance, primitive, group)) {
            if (group.primitiveIndex != instance.primitiveIndex || group.pageIndex >= params.scenePageCount) {
                reportInvalidTraversal(requests, params);
                return;
            }
            continue;
        }

        uint state = streamPageState(pageTable[group.pageIndex]);
        if (streamPageAvailable(state)) {
            markStreamPageDemand(pageTable, params, group.pageIndex);
        } else {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
    }
}

void buildInstanceActiveGroups(
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> pageWords,
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    uint instanceIndex)
{
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        InterlockedAdd(activeHeader[0].overflowCount, 1u);
        return;
    }

    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    if (primitive.nodeCount == 0u ||
        primitive.nodeOffset >= params.sceneNodeCount ||
        primitive.nodeCount > params.sceneNodeCount - primitive.nodeOffset) {
        InterlockedAdd(activeHeader[0].overflowCount, 1u);
        return;
    }

    uint stack[kTraversalStackCapacity];
    uint stackCount = 1u;
    stack[0] = primitive.nodeOffset;
    while (stackCount != 0u) {
        uint nodeIndex = stack[--stackCount];
        if (nodeIndex < primitive.nodeOffset ||
            nodeIndex >= primitive.nodeOffset + primitive.nodeCount ||
            nodeIndex >= params.sceneNodeCount) {
            InterlockedAdd(activeHeader[0].overflowCount, 1u);
            return;
        }

        GPUDrivenStreamAssetNode node = nodes[nodeIndex];
        if (node.primitiveIndex != instance.primitiveIndex ||
            !nodeNeedsTraversal(params, instance, primitive, node)) {
            continue;
        }
        if (node.childCount != 0u) {
            if (node.childOffset < primitive.nodeOffset ||
                node.childOffset >= primitive.nodeOffset + primitive.nodeCount ||
                node.childCount > primitive.nodeOffset + primitive.nodeCount - node.childOffset ||
                node.childCount > kTraversalStackCapacity - stackCount) {
                InterlockedAdd(activeHeader[0].overflowCount, 1u);
                return;
            }
            for (uint child = 0u; child < node.childCount; ++child) {
                stack[stackCount++] = node.childOffset + child;
            }
            continue;
        }

        if (node.groupIndex >= params.sceneGroupCount) {
            InterlockedAdd(activeHeader[0].overflowCount, 1u);
            return;
        }
        GPUDrivenStreamAssetGroup group = groups[node.groupIndex];
        if (group.primitiveIndex != instance.primitiveIndex ||
            group.clusterCount == 0u ||
            group.clusterCount > 32u ||
            !groupPageDrawable(pageTable, params, group)) {
            continue;
        }
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
                instanceIndex,
                group,
                clusterSelectionMask);
        }
    }
}

bool enqueueActiveTraversalWork(
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader> traversalHeader,
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem> traversalWork,
    GPUDrivenStreamAssetParams params,
    uint instanceIndex,
    uint nodeIndex)
{
    uint writeIndex = 0u;
    InterlockedAdd(traversalHeader[0].writeCounter, 1u, writeIndex);
    if (writeIndex >= params.traversalWorkCapacity) {
        InterlockedAdd(traversalHeader[0].overflowCount, 1u);
        return false;
    }

    InterlockedAdd(traversalHeader[0].taskCounter, 1u);
    traversalWork[writeIndex].instanceIndex = instanceIndex;
    traversalWork[writeIndex].nodeIndex = nodeIndex;
    traversalWork[writeIndex].padding0 = 0u;
    DeviceMemoryBarrier();
    uint oldReadyFrame = 0u;
    InterlockedExchange(traversalWork[writeIndex].readyFrame, params.frameIndex + 1u, oldReadyFrame);
    return true;
}

void appendInstanceFallbackGroups(
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> pageWords,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    uint instanceIndex,
    GPUDrivenStreamAssetPrimitive primitive)
{
    if (primitive.fallbackGroupOffset > params.sceneGroupCount ||
        primitive.fallbackGroupCount > params.sceneGroupCount - primitive.fallbackGroupOffset) {
        InterlockedAdd(activeHeader[0].overflowCount, 1u);
        return;
    }
    for (uint fallbackChild = 0u; fallbackChild < primitive.fallbackGroupCount; ++fallbackChild) {
        uint groupIndex = primitive.fallbackGroupOffset + fallbackChild;
        GPUDrivenStreamAssetGroup group = groups[groupIndex];
        if (group.primitiveIndex != instance.primitiveIndex ||
            group.clusterCount == 0u ||
            group.clusterCount > 32u ||
            group.pageIndex >= params.scenePageCount) {
            InterlockedAdd(activeHeader[0].overflowCount, 1u);
            reportInvalidTraversal(requests, params);
            continue;
        }
        uint state = streamPageState(pageTable[group.pageIndex]);
        if (streamPageAvailable(state)) {
            markStreamPageDemand(pageTable, params, group.pageIndex);
        } else {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
        if (!groupPageDrawable(pageTable, params, group)) {
            continue;
        }
        uint clusterSelectionMask = 0u;
        if (!groupNeedsTraversal(params, instance, primitive, group)) {
            uint clusterCount = min(group.clusterCount, 32u);
            clusterSelectionMask = clusterCount == 32u
                ? 0xffffffffu
                : ((1u << clusterCount) - 1u);
        } else {
            clusterSelectionMask = buildGroupClusterSelectionMask(
                groups,
                pageWords,
                pageTable,
                params,
                instance,
                primitive,
                group);
        }
        if (clusterSelectionMask != 0u) {
            appendActivePage(
                activeGroups,
                activeHeader,
                pageTable,
                params,
                instance,
                instanceIndex,
                group,
                clusterSelectionMask);
        }
    }
}

