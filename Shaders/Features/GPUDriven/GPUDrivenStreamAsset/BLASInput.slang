#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// BLAS cluster references and TLAS instance input generation.
[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetBuildBlasInputMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    // The runtime dispatches BLAS/TLAS work only after cluster RT has created
    // the representation buffers and persistent GPU allocator.
    StructuredBuffer<GPUDrivenStreamAssetParams> initialParams =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));
    GPUDrivenStreamAssetParams rtParams = initialParams[0];
    if (push.activeBuildPhase >= 20u) {
        buildStreamRTRepresentations(dispatchThreadID.x, dispatchThreadID.y, push, rtParams);
        return;
    }
    uint ownerCount = rtParams.rtRepresentationCount;
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeaderHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(rtParams.rtRepresentationHeaderBuffer));
    StructuredBuffer<GPUDrivenStreamAssetCLASPageEntry> clasPageTableHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetCLASPageEntry>>(push.clasPageTableBuffer));
    StructuredBuffer<uint2> clasAddressHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.clasAddressBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetBLASHeader> blasHeaderHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetBLASHeader>>(push.blasHeaderBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetInstanceBLAS> instanceBlasHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetInstanceBLAS>>(rtParams.rtRepresentationBuffer));
    RWStructuredBuffer<uint2> blasClusterReferenceHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.blasClusterReferenceBuffer));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));

    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader = activeHeaderHandle;
    StructuredBuffer<GPUDrivenStreamAssetCLASPageEntry> clasPageTable = clasPageTableHandle;
    StructuredBuffer<uint2> clasAddresses = clasAddressHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetBLASHeader> blasHeader = blasHeaderHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetInstanceBLAS> instanceBlas = instanceBlasHandle;
    RWStructuredBuffer<uint2> blasClusterReferences = blasClusterReferenceHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    uint itemIndex = dispatchThreadID.x;

    uint activeCount = min(activeHeader[0].activeGroupCount, activeHeader[0].activeGroupCapacity);

    if (push.activeBuildPhase == kBLASInputPhaseReset) {
        if (itemIndex == 0u) {
            GPUDrivenStreamAssetBLASHeader previous = (GPUDrivenStreamAssetBLASHeader)0;
            if ((push.traversalPhase & 1u) == 0u) { previous = blasHeader[0]; }
            GPUDrivenStreamAssetBLASHeader header = (GPUDrivenStreamAssetBLASHeader)0;
            header.clusterReferenceCount = 0u;
            header.blasBuildCount = 0u;
            header.clusterReferenceCapacity = params.blasClusterReferenceCapacity;
            header.overflowCount = 0u;
            header.frameIndex = params.frameIndex;
            header.padding0 = push.traversalPhase != 0u ? 1u : 0u;
            header.arenaUsedBytes = previous.arenaUsedBytes;
            header.arenaBase = previous.arenaUsedBytes;
            header.padding1 = activeCount;
            GPUDrivenStreamAssetBLASHeader representationStats = blasHeader[0];
            header.representationCount = representationStats.representationCount;
            header.sharingInstanceCount = representationStats.sharingInstanceCount;
            header.mergedInstanceCount = representationStats.mergedInstanceCount;
            header.cachedInstanceCount = representationStats.cachedInstanceCount;
            header.representationGroupCount = representationStats.representationGroupCount;
            header.representationGroupOverflow = representationStats.representationGroupOverflow;
            blasHeader[0] = header;
        }
        return;
    }

    RWStructuredBuffer<uint4> cacheHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.blasHeaderBuffer));
    RWStructuredBuffer<uint4> cache = cacheHandle;
    if (push.activeBuildPhase == 4u || push.activeBuildPhase == 11u) {
        if (push.activeBuildPhase == 11u && itemIndex < ownerCount) {
            GPUDrivenStreamAssetInstanceBLAS info = instanceBlas[itemIndex];
            if (streamBlasDynamicReady(info)) {
                RWStructuredBuffer<uint> representationState = resolveUniform(
                    ResourceHandle<RWStructuredBuffer<uint>>(params.rtRepresentationStateBuffer));
                uint base = streamRTPrimitiveBase(params, itemIndex / 3u);
                if ((info.flags & kBLASInstanceSemanticCutDirty) != 0u) {
                    representationState[base + 8u + itemIndex % 3u] += 1u;
                }
                if (itemIndex % 3u == 2u) {
                    representationState[base + 11u] = (representationState[base + 11u] & ~kRTCachedLevelMask) |
                        (representationState[base + 2u] + 1u);
                }
            }
        }
        if (itemIndex < activeCount) {
            GPUDrivenStreamAssetActiveGroup group = streamBlasGroup(itemIndex, params);
            if (group.instanceIndex >= ownerCount || group.pageIndex >= params.scenePageCount) { return; }
            uint4 key = uint4(group.pageIndex, group.clusterSelectionMask, group.clusterCount,
                clasPageTable[group.pageIndex].publicationGeneration);
            if (push.activeBuildPhase == 11u) {
                cache[kStreamBLASHeaderWords / 4u + itemIndex] = key;
            } else {
                GPUDrivenStreamAssetInstanceBLAS info = instanceBlas[group.instanceIndex];
                // Compare by rank within the instance, independent of the
                // compact offsets of other instances. Read all old keys before
                // phase 11 publishes the next cache (including removed groups).
                uint rank = itemIndex - info.firstGroup;
                bool dirty = info.cachedValid == 0u || rank >= info.previousGroupCount;
                bool semanticDirty = dirty;
                if (!dirty) {
                    uint4 previous = cache[kStreamBLASHeaderWords / 4u + info.previousFirstGroup + rank];
                    dirty = any(previous != key);
                    semanticDirty = any(previous.xyz != key.xyz);
                    if (previous.w != key.w) {
                        InterlockedAdd(blasHeader[0].publicationInvalidated, 1u);
                        if (group.instanceIndex % 3u == 2u) {
                            InterlockedAdd(blasHeader[0].cacheDependencyInvalidations, 1u);
                        }
                    }
                }
                if (semanticDirty) {
                    InterlockedOr(instanceBlas[group.instanceIndex].flags, kBLASInstanceSemanticCutDirty);
                }
                if (dirty) {
                    InterlockedOr(instanceBlas[group.instanceIndex].flags, kBLASInstanceDirty);
                    InterlockedOr(blasHeader[0].padding0, 1u);
                }
            }
        }
        return;
    }
    if (push.activeBuildPhase == 5u) {
        if (itemIndex < ownerCount) {
            GPUDrivenStreamAssetInstanceBLAS info = instanceBlas[itemIndex];
            // Bit 0 initializes a new cache; bit 1 forces physical rebuilds
            // (publication generation wrap) while preserving semantic history.
            if ((push.traversalPhase & 1u) != 0u) { info = (GPUDrivenStreamAssetInstanceBLAS)0; }
            info.selectedClusterCount = 0u;
            info.insertedClusterCount = 0u;
            info.blasBuildIndex = kInvalidClusterIndex;
            info.flags = (info.flags & kBLASInstancePublishedMask) |
                ((push.traversalPhase & 2u) != 0u ? kBLASInstanceDirty : 0u);
            info.firstGroup = 0xffffffffu;
            info.groupCount = 0u;
            instanceBlas[itemIndex] = info;
        }
        return;
    }

    if (push.activeBuildPhase == kBLASInputPhaseCount) {
        if (itemIndex >= activeCount) {
            return;
        }
        GPUDrivenStreamAssetActiveGroup group = streamBlasGroup(itemIndex, params);
        if (group.instanceIndex >= ownerCount ||
            group.pageIndex >= params.scenePageCount ||
            group.clusterCount == 0u ||
            group.clusterCount > 32u) {
            InterlockedAdd(blasHeader[0].overflowCount, 1u);
            InterlockedAdd(blasHeader[0].invalidGroups, 1u);
            if (group.instanceIndex < ownerCount) {
                InterlockedOr(instanceBlas[group.instanceIndex].flags, kBLASInstanceFallback);
            }
            return;
        }
        InterlockedMin(instanceBlas[group.instanceIndex].firstGroup, itemIndex);
        InterlockedAdd(instanceBlas[group.instanceIndex].groupCount, 1u);
        GPUDrivenStreamAssetCLASPageEntry page = clasPageTable[group.pageIndex];
        if (clasPageState(page) != kCLASPageStateActive) {
            InterlockedOr(instanceBlas[group.instanceIndex].flags, kBLASInstanceFallback);
            return;
        }
        uint validMask = group.clusterCount == 32u
            ? 0xffffffffu
            : ((1u << group.clusterCount) - 1u);
        uint selectedCount = countbits(group.clusterSelectionMask & validMask);
        if (selectedCount != 0u) {
            InterlockedAdd(instanceBlas[group.instanceIndex].selectedClusterCount, selectedCount);
        }
        return;
    }

    // Two-level deterministic exclusive scan. Scratch uint4s follow the cut
    // cache in the header buffer; no physical ranges survive a changed cut.
    RWStructuredBuffer<uint4> scanHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.blasHeaderBuffer));
    RWStructuredBuffer<uint4> scan = scanHandle;
    uint scratchBase = kStreamBLASHeaderWords / 4u + params.rtRepresentationGroupCapacity;
    if (push.activeBuildPhase == 6u) {
        uint lane = itemIndex & 63u;
        uint2 demand = 0u;
        if (itemIndex < ownerCount) {
            GPUDrivenStreamAssetInstanceBLAS info = instanceBlas[itemIndex];
            bool missing = (info.flags & kBLASInstanceFallback) != 0u;
            if (missing) { InterlockedAdd(blasHeader[0].missingClasInstances, 1u); }
            if (info.selectedClusterCount != 0u) {
                InterlockedAdd(blasHeader[0].requestedClusterReferences, info.selectedClusterCount);
                InterlockedAdd(blasHeader[0].requestedInstances, 1u);
                if (!missing) {
                    if (info.selectedClusterCount > params.maxBlasClustersPerBuild) {
                        InterlockedOr(instanceBlas[itemIndex].flags, kBLASInstanceFallback | kBLASInstanceOverflow);
                        InterlockedAdd(blasHeader[0].oversizedInstances, 1u);
                        InterlockedAdd(blasHeader[0].overflowCount, 1u);
                    } else {
                        demand = uint2(info.selectedClusterCount, 1u);
                    }
                }
            }
        }
        // Count is bounded by activeGroupCount * 32; saturating scan also
        // prevents corrupt demand from wrapping into an admitted range.
        uint2 before = scanStreamLodCounts(demand, lane, 0xffffffffu);
        if (itemIndex < ownerCount) {
            instanceBlas[itemIndex].clusterReferenceOffset = before.x;
            instanceBlas[itemIndex].clusterReferenceCapacity = demand.x;
            instanceBlas[itemIndex].blasBuildIndex = before.y;
        }
        if (lane == 63u) { scan[scratchBase + itemIndex / 64u] = uint4(g_StreamLodPrefix[63], 0u, 0u); }
        return;
    }
    if (push.activeBuildPhase == 7u) {
        if (itemIndex == 0u) {
            uint2 prefix = 0u;
            for (uint block = 0u; block < (ownerCount + 63u) / 64u; ++block) {
                uint2 count = scan[scratchBase + block].xy;
                scan[scratchBase + block] = uint4(prefix, 0u, 0u);
                prefix += min(count, 0xffffffffu - prefix);
            }
        }
        return;
    }
    if (push.activeBuildPhase == kBLASInputPhaseSetup) {
        if (itemIndex >= ownerCount) { return; }
        GPUDrivenStreamAssetInstanceBLAS info = instanceBlas[itemIndex];
        if (info.clusterReferenceCapacity == 0u) {
            instanceBlas[itemIndex].blasBuildIndex = kInvalidClusterIndex;
            instanceBlas[itemIndex].cachedValid = 0u;
            InterlockedOr(instanceBlas[itemIndex].flags, kBLASInstanceFallback);
            return;
        }
        uint2 blockPrefix = scan[scratchBase + itemIndex / 64u].xy;
        uint referenceOffset = info.clusterReferenceOffset + min(blockPrefix.x, 0xffffffffu - info.clusterReferenceOffset);
        uint buildIndex = info.blasBuildIndex + min(blockPrefix.y, 0xffffffffu - info.blasBuildIndex);
        instanceBlas[itemIndex].clusterReferenceOffset = referenceOffset;
        instanceBlas[itemIndex].blasBuildIndex = buildIndex;
        bool dirty = info.cachedValid == 0u || info.previousGroupCount != info.groupCount ||
            (info.flags & kBLASInstanceDirty) != 0u;
        bool semanticDirty = info.previousGroupCount != info.groupCount;
        InterlockedOr(instanceBlas[itemIndex].flags, kBLASInstanceDynamic |
            (dirty ? kBLASInstanceDirty : 0u) | (semanticDirty ? kBLASInstanceSemanticCutDirty : 0u));
        InterlockedAdd(blasHeader[0].admittedInstances, 1u);
        return;
    }
    if (push.activeBuildPhase != kBLASInputPhaseInsert || itemIndex >= activeCount) {
        return;
    }

    GPUDrivenStreamAssetActiveGroup group = streamBlasGroup(itemIndex, params);
    if (group.instanceIndex >= ownerCount ||
        group.pageIndex >= params.scenePageCount ||
        group.clusterCount == 0u ||
        group.clusterCount > 32u) {
        return;
    }
    GPUDrivenStreamAssetInstanceBLAS info = instanceBlas[group.instanceIndex];
    if ((info.flags & kBLASInstanceDirty) == 0u || (info.flags & kBLASInstanceDynamic) == 0u ||
        (info.flags & kBLASInstanceFallback) != 0u) {
        return;
    }
    GPUDrivenStreamAssetCLASPageEntry page = clasPageTable[group.pageIndex];
    if (clasPageState(page) != kCLASPageStateActive) {
        InterlockedOr(instanceBlas[group.instanceIndex].flags, kBLASInstanceFallback);
        InterlockedAdd(blasHeader[0].overflowCount, 1u);
        return;
    }

    uint validMask = group.clusterCount == 32u
        ? 0xffffffffu
        : ((1u << group.clusterCount) - 1u);
    uint selectedMask = group.clusterSelectionMask & validMask;
    for (uint clusterIndex = 0u; clusterIndex < group.clusterCount; ++clusterIndex) {
        if ((selectedMask & (1u << clusterIndex)) == 0u) {
            continue;
        }
        uint insertIndex;
        InterlockedAdd(instanceBlas[group.instanceIndex].insertedClusterCount, 1u, insertIndex);
        if (insertIndex >= info.selectedClusterCount || insertIndex >= info.clusterReferenceCapacity) {
            InterlockedOr(instanceBlas[group.instanceIndex].flags, kBLASInstanceOverflow);
            InterlockedAdd(blasHeader[0].overflowCount, 1u);
            continue;
        }
        blasClusterReferences[info.clusterReferenceOffset + insertIndex] =
            clasAddresses[clasPageAddressOffset(page) + clusterIndex];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetBuildTlasInputMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<GPUDrivenStreamAssetInstance> instanceHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(push.instanceBuffer));
    StructuredBuffer<uint2> fallbackBlasAddressHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.fallbackBlasAddressBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetTLASInstance> tlasInstanceHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetTLASInstance>>(push.tlasInstanceBuffer));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));

    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    StructuredBuffer<uint2> fallbackBlasAddresses = fallbackBlasAddressHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetTLASInstance> tlasInstances = tlasInstanceHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    uint instanceIndex = dispatchThreadID.x;
    if (instanceIndex >= params.sceneInstanceCount) {
        return;
    }

    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    GPUDrivenStreamAssetInstanceBLAS blas = (GPUDrivenStreamAssetInstanceBLAS)0;
    StructuredBuffer<uint> representationState = resolveUniform(
        ResourceHandle<StructuredBuffer<uint>>(params.rtRepresentationStateBuffer));
    StructuredBuffer<GPUDrivenStreamAssetInstanceBLAS> representations = resolveUniform(
        ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstanceBLAS>>(params.rtRepresentationBuffer));
    uint selected = representationState[streamRTInstanceBase(params, instanceIndex) + 2u];
    if (selected != 0xffffffffu) { blas = representations[selected]; }
    uint2 fallbackAddress = instance.primitiveIndex < params.scenePrimitiveCount
        ? fallbackBlasAddresses[instance.primitiveIndex] : uint2(0u, 0u);
    uint2 accelerationStructureReference = streamPublishedBlasAddress(blas, params, fallbackAddress);

    bool addressValid =
        accelerationStructureReference.x != 0u ||
        accelerationStructureReference.y != 0u;
    uint visibilityMask = instance.visible != 0u && addressValid ? 0xffu : 0u;

    GPUDrivenStreamAssetTLASInstance output;
    output.transform0 = float4(instance.world0.x, instance.world1.x, instance.world2.x, instance.world3.x);
    output.transform1 = float4(instance.world0.y, instance.world1.y, instance.world2.y, instance.world3.y);
    output.transform2 = float4(instance.world0.z, instance.world1.z, instance.world2.z, instance.world3.z);
    output.customIndexMask = (instanceIndex & 0x00ffffffu) | (visibilityMask << 24u);
    // Standalone runtimes without GPUScene metadata retain software coverage.
    uint opacityFlags = 8u; // VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR
    if (params.gpuSceneInstanceBufferIndexPlusOne != 0u && instance.gpuSceneInstanceIndex != kInvalidClusterIndex) {
        StructuredBuffer<GPUDrivenPreviewInstance> gpuInstances =
            resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(params.gpuSceneInstanceBufferIndexPlusOne - 1u));
        uint coverageFlags = gpuInstances[instance.gpuSceneInstanceIndex].identity.w;
        if ((coverageFlags & (kInstanceAlphaMasked | kInstanceAlphaBlend)) == 0u) {
            opacityFlags = 4u; // VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR
        }
    }
    output.shaderBindingTableFlags = opacityFlags << 24u;
    output.accelerationStructureReferenceLow = accelerationStructureReference.x;
    output.accelerationStructureReferenceHigh = accelerationStructureReference.y;
    tlasInstances[instanceIndex] = output;
}

