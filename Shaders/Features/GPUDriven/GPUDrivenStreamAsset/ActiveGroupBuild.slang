#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Active-group lifecycle and phase dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetBuildActiveMain(uint3 dispatchThreadID : SV_DispatchThreadID, uniform GPUDrivenStreamAssetPush push)
{
    uint rowBase = dispatchThreadID.y * (65535u * 64u);
    // The final padded row for a near-32-bit state allocation can wrap the
    // flattened index. Those lanes must not clear or build the first entries.
    if (dispatchThreadID.x > 0xffffffffu - rowBase) { return; }
    uint threadIndex = dispatchThreadID.x + rowBase;
    StructuredBuffer<uint> pageHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.pageBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroupHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(push.activeGroupBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeaderHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(push.activeHeaderBuffer));
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
    RWStructuredBuffer<GPUDrivenStreamAssetDrawIndirect> drawIndirectHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetDrawIndirect>>(push.drawIndirectBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader> traversalHeaderHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader>>(push.traversalHeaderBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem> traversalWorkHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>>(push.traversalWorkBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTableHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(push.pageTableBuffer));
    RWStructuredBuffer<uint> requestHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.requestBuffer));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));

    StructuredBuffer<uint> pageWords = pageHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader = activeHeaderHandle;
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives = primitiveHandle;
    StructuredBuffer<GPUDrivenStreamAssetLODLevel> lodLevels = lodLevelHandle;
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups = groupHandle;
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes = nodeHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetDrawIndirect> drawIndirect = drawIndirectHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader> traversalHeader = traversalHeaderHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem> traversalWork = traversalWorkHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    RWStructuredBuffer<uint> requests = requestHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    if (push.activeBuildPhase >= kActiveBuildPhaseFrontier && push.activeBuildPhase <= kActiveBuildPhaseInitializeLODState) {
        StructuredBuffer<uint> topologyHandle =
            resolveUniform(ResourceHandle<StructuredBuffer<uint>>(params.lodTopologyBuffer));
        RWStructuredBuffer<uint> stateHandle =
            resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.lodStateBuffer));
        StructuredBuffer<uint> topology = topologyHandle;
        RWStructuredBuffer<uint> state = stateHandle;
        if (push.activeBuildPhase == kActiveBuildPhaseInitializeLODState) {
            uint wordCount, stride;
            state.GetDimensions(wordCount, stride);
            if (threadIndex < wordCount) { state[threadIndex] = 0u; }
        } else if (push.activeBuildPhase == kActiveBuildPhasePrefix) {
            prefixStreamLodFrontier(state, activeHeader, params, threadIndex);
        } else if (threadIndex < params.sceneInstanceCount) {
            if (push.activeBuildPhase == kActiveBuildPhaseFrontier) {
                buildStreamLodFrontier(threadIndex, instances, primitives, groups,
                    topology, state, pageTable, requests, params, push);
            } else {
                emitStreamLodFrontier(threadIndex, instances, primitives, groups,
                    topology, state, pageTable, activeGroups, activeHeader, params);
            }
        }
        return;
    }
    if (push.activeBuildPhase == kActiveBuildPhaseReset) {
        if (params.demandStatsBuffer != 0xffffffffu && threadIndex < kStreamDemandStatsWords) {
            RWStructuredBuffer<uint> stats = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.demandStatsBuffer));
            stats[threadIndex] = 0u;
        }
        if (threadIndex == 0u) {
            GPUDrivenStreamAssetActiveHeader header;
            header.activeGroupCount = 0u;
            header.activeGroupCapacity = params.activeGroupCount;
            header.maxActiveGroupClusters = params.maxActiveGroupClusters;
            header.overflowCount = 0u;
            header.frameIndex = params.frameIndex;
            header.padding0 = 0u;
            header.padding1 = 0u;
            header.padding2 = 0u;
            activeHeader[0] = header;
            if (params.sceneNodeCount != 0u) {
                GPUDrivenStreamAssetDrawIndirect draw;
                draw.groupCountX = 0u;
                draw.groupCountY = 1u;
                draw.groupCountZ = 1u;
                drawIndirect[0] = draw;
                // Separate tessellation arguments: eight 16-triangle tasks per
                // cluster instead of two ordinary 64-triangle mesh workgroups.
                uint tessGroups = (draw.groupCountX / kMeshletTriangleChunkCount) * 8u;
                draw.groupCountX = min(tessGroups, 65535u);
                draw.groupCountY = max(1u, (tessGroups + 65534u) / 65535u);
                drawIndirect[1] = draw;

                GPUDrivenStreamAssetTraversalHeader traversal;
                traversal.readCounter = 0u;
                traversal.writeCounter = 0u;
                traversal.taskCounter = 0u;
                traversal.overflowCount = 0u;
                traversal.frameIndex = params.frameIndex;
                traversal.padding0 = 0u;
                traversal.padding1 = 0u;
                traversal.padding2 = 0u;
                traversalHeader[0] = traversal;
            }
        }
        return;
    }

    if (push.activeBuildPhase == kActiveBuildPhaseFinalize) {
        if (threadIndex == 0u) {
            GPUDrivenStreamAssetActiveHeader header = activeHeader[0];
            if (params.sceneNodeCount != 0u) {
                header.overflowCount += traversalHeader[0].overflowCount;
                activeHeader[0] = header;
            }
            uint activeCount = min(header.activeGroupCount, header.activeGroupCapacity);
            GPUDrivenStreamAssetDrawIndirect draw;
            draw.groupCountX = min(
                activeCount * header.maxActiveGroupClusters * kMeshletTriangleChunkCount,
                params.drawTaskCount);
            draw.groupCountY = 1u;
            draw.groupCountZ = 1u;
            drawIndirect[0] = draw;
            // Separate tessellation arguments: eight 16-triangle tasks per
            // cluster instead of two ordinary 64-triangle mesh workgroups.
            uint tessGroups = (draw.groupCountX / kMeshletTriangleChunkCount) * 8u;
            draw.groupCountX = min(tessGroups, 65535u);
            draw.groupCountY = max(1u, (tessGroups + 65534u) / 65535u);
            drawIndirect[1] = draw;
        }
        return;
    }

    if (push.activeBuildPhase == kActiveBuildPhaseSeed) {
        uint workerCount = max(params.traversalWorkerCount, 1u);
        if (threadIndex >= workerCount) {
            return;
        }
        for (uint instanceIndex = threadIndex;
             instanceIndex < params.sceneInstanceCount;
             instanceIndex += workerCount) {
            GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
            if (instance.visible == 0u) {
                continue;
            }
            if (instance.primitiveIndex >= params.scenePrimitiveCount) {
                InterlockedAdd(traversalHeader[0].overflowCount, 1u);
                continue;
            }
            GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
            appendInstanceFallbackGroups(
                activeGroups,
                activeHeader,
                groups,
                pageWords,
                pageTable,
                requests,
                params,
                instance,
                instanceIndex,
                primitive);
            if (primitive.nodeCount == 0u ||
                primitive.nodeOffset >= params.sceneNodeCount ||
                primitive.nodeCount > params.sceneNodeCount - primitive.nodeOffset) {
                InterlockedAdd(traversalHeader[0].overflowCount, 1u);
                continue;
            }
            enqueueActiveTraversalWork(
                traversalHeader,
                traversalWork,
                params,
                instanceIndex,
                primitive.nodeOffset);
        }
        return;
    }

    if (push.activeBuildPhase == kActiveBuildPhaseRun) {
        uint workerCount = max(params.traversalWorkerCount, 1u);
        if (threadIndex < workerCount) {
            runActiveTraversalWorker(
                activeGroups,
                activeHeader,
                traversalHeader,
                traversalWork,
                instances,
                primitives,
                groups,
                pageWords,
                nodes,
                pageTable,
                requests,
                params);
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
                buildInstanceActiveGroups(
                    activeGroups,
                    activeHeader,
                    primitives,
                    groups,
                    pageWords,
                    nodes,
                    pageTable,
                    params,
                    instance,
                    instanceIndex);
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
    if (instance.visible == 0u || instance.primitiveIndex >= params.scenePrimitiveCount) {
        return;
    }

    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    if (primitive.groupCount == 0u ||
        localGroupIndex >= primitive.groupCount ||
        primitive.groupOffset >= params.sceneGroupCount ||
        primitive.groupCount > params.sceneGroupCount - primitive.groupOffset) {
        return;
    }

    uint groupIndex = primitive.groupOffset + localGroupIndex;
    GPUDrivenStreamAssetGroup group = groups[groupIndex];
    if (group.primitiveIndex != instance.primitiveIndex ||
        group.clusterCount == 0u ||
        group.clusterCount > 32u ||
        !groupPageDrawable(pageTable, params, group)) {
        return;
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

