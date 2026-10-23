#language slang 2026
module StreamBLASAllocator;

import GPUDrivenStreamAsset;

// Serialized topology mutation is representation-granular, never per cluster.
// Raw ABI mirrored in MeshletStreamBLASAllocator.h.
struct Gap
{
    uint offset, size, previous, next;
    uint binPrevious, binNext, free, unusedNext;
};
struct Allocation
{
    uint node, encodedBytes, rejectedEpoch, oldNode;
    uint pendingNode, pendingOffset, pendingBytes, status;
    uint inputDynamic, inputSelectedClusterCount, prepareDecision, reserved;
};
struct Parameters
{
    uint* state;
    Gap* gaps;
    Allocation* allocations;
    GPUDrivenStreamAssetInstanceBLAS* owners;
    GPUDrivenStreamAssetBLASHeader* header;
    GPUDrivenStreamAssetBLASBuildInfo* buildInfos;
    uint* buildOwners;
    uint* sizes;
    uint64_t* sourceAddresses;
    uint64_t* moveSources;
    uint64_t* moveDestinations;
    uint* moveCounts;
    uint64_t clusterReferenceAddress, arena;
    uint ownerCount, alignment, storageBytes, relocateBudget;
    uint nodeCapacity, relocateScanCount, reuseEnable, reserved;
    uint64_t relocationArena;
    uint* priorityOrder;
    uint* importance;
    uint referenceCapacity, buildCapacity;
    uint* priorityCacheControl;
};
struct Push { Parameters* parameters; uint phase, reserved; };
[[vk::push_constant]] ConstantBuffer<Push> gPush;

static const uint kInvalid = 0xffffffffu;
static const uint kBinCount = 96u, kStateWords = 144u;
static const uint kNextNode = 96u, kFreeNode = 97u, kTail = 98u, kEpoch = 99u, kCursor = 100u;
static const uint kReleaseCount = 101u;
static const uint kEncodedBytes = 102u, kAllocatedBytes = 103u, kLargestFree = 104u, kFreeBytes = 105u;
static const uint kSuppressed = 106u, kRelocated = 107u, kRelocatedBytes = 108u, kFailures = 109u;
static const uint kCompactionTarget = 110u, kPriorityWaitingCount = 111u;
static const uint kPendingBuild = 1u, kPendingRelocate = 2u, kRejected = 3u;
static const uint kBaselineRelease = 4u, kBaselineBuilding = 5u;
static const uint kPriorityBuilding = 6u;
static const uint kPriorityWaiting = 7u;
static const uint kRequestedVisible = 112u, kRequestedOffscreen = 113u;
static const uint kRejectedVisible = 114u, kRejectedOffscreen = 115u;
static const uint kRejectedMaxError = 116u, kRejectedVisibleMaxError = 117u;
static const uint kAdmittedOffscreen = 118u, kPriorityEvicted = 119u;
static const uint kPostBuildBudgetRejected = 120u, kPhysicalGapFailures = 121u;
static const uint kWaitingClaimBytes = 122u, kMaxWaitingBytes = 123u, kUnknownBuilds = 124u;
static const uint kProjectedRetryDeferred = 125u, kProjectedDirtyClaimBytes = 126u;
static const uint kSortReused = 128u, kAdmissionReused = 129u, kInputsChanged = 130u, kCacheValid = 131u;
static const uint kCachedFree = 132u, kCachedLargest = 133u;
static const uint kCachedReferenceCapacity = 134u, kCachedBuildCapacity = 135u;
static const uint kCachedStorageBytes = 136u, kCachedEvictionCapacity = 137u;
static const uint kCachedStorageRejected = 138u, kCachedWaitingCount = 139u;
static const uint kCachedWaitingBytes = 140u, kCachedMaxWaitingBytes = 141u;
static const uint kCachedReferenceRejected = 142u, kCachedBuildRejected = 143u;
static const uint kPrepareNone = 0u, kPrepareStorageRejected = 1u, kPrepareWaiting = 2u;
static const uint kPrepareReferenceRejected = 3u, kPrepareBuildRejected = 4u, kPrepareProjected = 5u;

// The priority topology remains serialized, but every tile's scattered owner
// and allocation reads are issued cooperatively rather than one dependent load
// chain per owner. Only the first workgroup executes these paths.
groupshared uint g_PriorityOwners[64];
groupshared GPUDrivenStreamAssetInstanceBLAS g_PriorityInfo[64];
groupshared Allocation g_PriorityAllocations[64];
groupshared uint g_PriorityEvicted[64];
groupshared uint g_PrioritySizes[64];
groupshared uint g_PriorityWrites[64];
groupshared uint g_PriorityVictimIndices[64];
groupshared uint g_PriorityActiveCount, g_PriorityNegativeCount, g_PriorityResidentCount;
groupshared uint g_PriorityMinNegativeBytes, g_PriorityResidentBytes, g_PriorityResidentOverflow;
groupshared uint g_PriorityFastPath, g_PriorityDeferredCount;
groupshared uint g_PriorityStorageRemaining;

void clearPriorityTileSummary(uint lane)
{
    if (lane == 0u) {
        g_PriorityActiveCount = 0u;
        g_PriorityNegativeCount = 0u;
        g_PriorityResidentCount = 0u;
        g_PriorityMinNegativeBytes = kInvalid;
        g_PriorityResidentBytes = 0u;
        g_PriorityResidentOverflow = 0u;
        g_PriorityFastPath = 0u;
        g_PriorityDeferredCount = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
}

void addPriorityResidentBytes(uint bytes)
{
    uint previous;
    InterlockedAdd(g_PriorityResidentBytes, bytes, previous);
    if (bytes > 0xffffffffu - previous) InterlockedOr(g_PriorityResidentOverflow, 1u);
    InterlockedAdd(g_PriorityResidentCount, 1u);
}

uint sizeBin(Parameters p, uint size)
{
    // Allocation and gap sizes are nonzero multiples of the arena alignment.
    // Exact small bins avoid scanning differently sized common BLAS blocks.
    uint units = size / p.alignment;
    return units <= 64u ? units - 1u : 64u + uint(firstbithigh(units)) - 6u;
}

uint largestFreeGap(Parameters p)
{
    // Every gap in a lower bin is smaller than every gap in this bin.
    for (uint bin = kBinCount; bin != 0u; --bin) {
        uint node = p.state[bin - 1u];
        if (node == kInvalid) continue;
        uint largest = 0u;
        while (node != kInvalid) {
            largest = max(largest, p.gaps[node].size);
            node = p.gaps[node].binNext;
        }
        return largest;
    }
    return 0u;
}

void removeFree(Parameters p, uint id)
{
    Gap gap = p.gaps[id];
    if (gap.binPrevious != kInvalid) p.gaps[gap.binPrevious].binNext = gap.binNext;
    else p.state[sizeBin(p, gap.size)] = gap.binNext;
    if (gap.binNext != kInvalid) p.gaps[gap.binNext].binPrevious = gap.binPrevious;
    p.gaps[id].free = 0u;
}

void insertFree(Parameters p, uint id)
{
    uint bin = sizeBin(p, p.gaps[id].size);
    uint head = p.state[bin];
    p.gaps[id].binPrevious = kInvalid;
    p.gaps[id].binNext = head;
    p.gaps[id].free = 1u;
    if (head != kInvalid) p.gaps[head].binPrevious = id;
    p.state[bin] = id;
}

uint newNode(Parameters p)
{
    uint id = p.state[kFreeNode];
    if (id != kInvalid) p.state[kFreeNode] = p.gaps[id].unusedNext;
    else { id = p.state[kNextNode]; p.state[kNextNode] = id + 1u; }
    return id;
}

void releaseNode(Parameters p, uint id)
{
    p.gaps[id].unusedNext = p.state[kFreeNode];
    p.state[kFreeNode] = id;
}

void releaseAllocation(Parameters p, uint id)
{
    // Neighbor gaps are already counted; only this live allocation is freed.
    p.state[kFreeBytes] += p.gaps[id].size;
    uint previous = p.gaps[id].previous;
    if (previous != kInvalid && p.gaps[previous].free != 0u) {
        removeFree(p, previous);
        p.gaps[previous].size += p.gaps[id].size;
        p.gaps[previous].next = p.gaps[id].next;
        if (p.gaps[id].next != kInvalid) p.gaps[p.gaps[id].next].previous = previous;
        else p.state[kTail] = previous;
        releaseNode(p, id);
        id = previous;
    }
    uint next = p.gaps[id].next;
    if (next != kInvalid && p.gaps[next].free != 0u) {
        removeFree(p, next);
        p.gaps[id].size += p.gaps[next].size;
        p.gaps[id].next = p.gaps[next].next;
        if (p.gaps[next].next != kInvalid) p.gaps[p.gaps[next].next].previous = id;
        else p.state[kTail] = id;
        releaseNode(p, next);
    }
    insertFree(p, id);
    p.state[kEpoch] += 1u;
}

uint takeGap(Parameters p, uint id, uint bytes)
{
    p.state[kFreeBytes] -= bytes;
    Gap gap = p.gaps[id];
    removeFree(p, id);
    if (gap.size > bytes) {
        uint remainder = newNode(p);
        Gap rest = { gap.offset + bytes, gap.size - bytes, id, gap.next,
            kInvalid, kInvalid, 0u, kInvalid };
        p.gaps[remainder] = rest;
        if (gap.next != kInvalid) p.gaps[gap.next].previous = remainder;
        else p.state[kTail] = remainder;
        p.gaps[id].next = remainder;
        p.gaps[id].size = bytes;
        insertFree(p, remainder);
    }
    return id;
}

uint allocate(Parameters p, uint bytes)
{
    for (uint bin = sizeBin(p, bytes); bin < kBinCount; ++bin) {
        uint id = p.state[bin];
        while (id != kInvalid && p.gaps[id].size < bytes) id = p.gaps[id].binNext;
        if (id != kInvalid) return takeGap(p, id, bytes);
    }
    return kInvalid;
}

void rejectOwner(Parameters p, uint owner)
{
    GPUDrivenStreamAssetInstanceBLAS info = p.owners[owner];
    info.flags = (info.flags & kBLASInstancePublishedMask) | kBLASInstanceFallback | kBLASInstanceOverflow;
    info.storageOffset = 0u;
    info.storageCapacity = 0u;
    // A negative cache records the examined cut without claiming a live BLAS.
    info.cachedValid = 2u;
    p.owners[owner] = info;
    InterlockedAdd(p.header[0].storageRejected, 1u);
    InterlockedAdd(p.header[0].overflowCount, 1u);
}

uint alignedBytes(Parameters p, uint encodedBytes)
{
    return encodedBytes <= 0xffffffffu - (p.alignment - 1u)
        ? (encodedBytes + p.alignment - 1u) & ~(p.alignment - 1u) : 0u;
}

GPUDrivenStreamAssetInstanceBLAS rejectedInfo(GPUDrivenStreamAssetInstanceBLAS info)
{
    info.flags = (info.flags & kBLASInstancePublishedMask) | kBLASInstanceFallback | kBLASInstanceOverflow;
    info.storageOffset = 0u;
    info.storageCapacity = 0u;
    return info;
}

void evictPriorityOwner(Parameters p, uint owner)
{
    Allocation allocation = p.allocations[owner];
    releaseAllocation(p, allocation.node);
    allocation.node = kInvalid;
    allocation.status = 0u;
    allocation.prepareDecision = kPrepareStorageRejected;
    allocation.rejectedEpoch = p.state[kEpoch];
    // Retain encodedBytes and the semantic cut for negative caching.
    p.allocations[owner] = allocation;
    rejectOwner(p, owner);
    p.state[kPriorityEvicted] += 1u;
}

void preparePriorityBuilds(Parameters p, uint lane)
{
    uint storageRemaining = p.storageBytes;
    // Historical sizes predict retry usefulness only. Native Build still measures
    // every admitted dirty cut, and phase 2 alone decides exact storage ownership.
    uint projectedRemaining = p.storageBytes;
    uint projectedDeferred = 0u, projectedDirtyBytes = 0u;
    uint referencesRemaining = p.referenceCapacity;
    uint buildsRemaining = p.buildCapacity;
    uint unallocatedClaim = 0u;
    uint victimCursor = p.ownerCount;
    uint largestEpoch = p.state[kEpoch];
    uint buildCount = p.header[0].blasBuildCount;
    uint referenceCount = p.header[0].clusterReferenceCount;
    uint dirtyCount = 0u, storageRejected = 0u, referenceRejected = 0u, buildRejected = 0u;
    uint suppressed = 0u, waitingCount = 0u, waitingBytes = 0u, maxWaitingBytes = 0u, unknownBuilds = 0u;
    uint compactionTarget = p.state[kCompactionTarget];
    // One ranked pass separates persistent storage claims from this frame's
    // actual build work. A fragmented negative entry retains its storage claim
    // without preventing a lower-ranked build that fits an existing gap.
    for (uint tile = 0u; tile < p.ownerCount; tile += 64u) {
        clearPriorityTileSummary(lane);
        uint rank = tile + lane;
        if (rank < p.ownerCount) {
            uint owner = p.priorityOrder[rank];
            g_PriorityOwners[lane] = owner;
            g_PriorityInfo[lane] = p.owners[owner];
            g_PriorityAllocations[lane] = p.allocations[owner];
            g_PriorityEvicted[lane] = 0u;
            g_PriorityWrites[lane] = 0u;
            GPUDrivenStreamAssetInstanceBLAS info = g_PriorityInfo[lane];
            Allocation allocation = g_PriorityAllocations[lane];
            if ((info.flags & kBLASInstanceDynamic) != 0u) {
                InterlockedAdd(g_PriorityActiveCount, 1u);
                bool dirty = (info.flags & kBLASInstanceDirty) != 0u || p.reuseEnable == 0u;
                uint bytes = alignedBytes(p, allocation.encodedBytes);
                if (!dirty && allocation.node != kInvalid && bytes != 0u) addPriorityResidentBytes(bytes);
                else if (!dirty && allocation.node == kInvalid && allocation.encodedBytes != 0u) {
                    InterlockedAdd(g_PriorityNegativeCount, 1u);
                    if (bytes != 0u) InterlockedMin(g_PriorityMinNegativeBytes, bytes);
                }
            }
        }
        AllMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            if (g_PriorityActiveCount == 0u) {
                g_PriorityFastPath = 1u;
            } else if (g_PriorityResidentCount == g_PriorityActiveCount &&
                g_PriorityResidentOverflow == 0u && g_PriorityResidentBytes <= storageRemaining) {
                storageRemaining -= g_PriorityResidentBytes;
                projectedRemaining -= min(projectedRemaining, g_PriorityResidentBytes);
                g_PriorityFastPath = 1u;
            } else if (g_PriorityNegativeCount == g_PriorityActiveCount &&
                g_PriorityMinNegativeBytes > min(storageRemaining, projectedRemaining)) {
                // Every active owner is unconditionally rejected or deferred;
                // no capacity, work budget, eviction, or backfill decision changes.
                g_PriorityStorageRemaining = storageRemaining;
                storageRejected += g_PriorityActiveCount;
                suppressed += g_PriorityActiveCount;
                g_PriorityFastPath = 2u;
            }
            if (g_PriorityFastPath == 0u) {
                for (uint local = 0u; local < min(64u, p.ownerCount - tile); ++local) {
                    if (g_PriorityEvicted[local] != 0u) continue;
                    uint rank = tile + local;
                    uint owner = g_PriorityOwners[local];
                    GPUDrivenStreamAssetInstanceBLAS info = g_PriorityInfo[local];
                    Allocation allocation = g_PriorityAllocations[local];
                    if ((info.flags & kBLASInstanceDynamic) == 0u) continue;
                    bool dirty = (info.flags & kBLASInstanceDirty) != 0u || p.reuseEnable == 0u;
                    bool resident = !dirty && allocation.node != kInvalid;
                    // A changed cut has no trustworthy size until native Build.
                    uint bytes = dirty ? 0u : alignedBytes(p, allocation.encodedBytes);
                    bool known = !dirty && allocation.encodedBytes != 0u;
                    bool fits = !known || (bytes != 0u && bytes <= storageRemaining);
                    if (!fits) {
                        if (resident) {
                            if (p.state[kPriorityEvicted] < p.relocateScanCount) evictPriorityOwner(p, owner);
                        } else {
                            allocation.rejectedEpoch = p.state[kEpoch];
                            allocation.prepareDecision = kPrepareStorageRejected;
                            info = rejectedInfo(info);
                            info.cachedValid = 2u;
                            g_PriorityAllocations[local] = allocation;
                            g_PriorityInfo[local] = info;
                            g_PriorityWrites[local] = 3u;
                            ++storageRejected;
                            ++suppressed;
                        }
                        continue;
                    }
                    if (resident) {
                        storageRemaining -= bytes;
                        projectedRemaining -= min(projectedRemaining, bytes);
                        continue;
                    }
                    if (known && bytes > projectedRemaining) {
                        // A historical estimate may delay this valid negative cut,
                        // but does not reserve real storage or evict an existing cache.
                        info = rejectedInfo(info);
                        info.cachedValid = 2u;
                        allocation.prepareDecision = kPrepareProjected;
                        g_PriorityAllocations[local] = allocation;
                        g_PriorityInfo[local] = info;
                        g_PriorityWrites[local] = 3u;
                        ++storageRejected;
                        ++suppressed;
                        ++projectedDeferred;
                        continue;
                    }
                    bool referencesFit = info.selectedClusterCount <= referencesRemaining;
                    if (!referencesFit || buildsRemaining == 0u) {
                        // Budget-denied work neither reserves storage nor evicts a cache.
                        info = rejectedInfo(info);
                        // Keep measured negative entries; an unbuilt cut remains invalid.
                        if (allocation.encodedBytes == 0u) info.cachedValid = 0u;
                        allocation.prepareDecision = referencesFit ? kPrepareBuildRejected : kPrepareReferenceRejected;
                        g_PriorityAllocations[local] = allocation;
                        g_PriorityInfo[local] = info;
                        g_PriorityWrites[local] = 3u;
                        if (referencesFit) ++buildRejected;
                        else ++referenceRejected;
                        continue;
                    }
                    storageRemaining -= bytes;
                    projectedRemaining -= min(projectedRemaining, bytes);
                    unallocatedClaim += bytes;
                    uint historicalBytes = allocation.pendingBytes;
                    allocation.pendingBytes = bytes;
                    // Make only the aggregate space claimed by higher-priority known cuts.
                    // A monotonically decreasing cursor visits each potential victim once.
                    // Fragmentation uses bounded compaction, not further over-eviction.
                    while (p.state[kFreeBytes] < unallocatedClaim && victimCursor > rank + 1u &&
                        p.state[kPriorityEvicted] < p.relocateScanCount) {
                        uint victimRank = --victimCursor;
                        uint victim = p.priorityOrder[victimRank];
                        if (p.allocations[victim].node != kInvalid) {
                            evictPriorityOwner(p, victim);
                            if (victimRank < tile + 64u) g_PriorityEvicted[victimRank - tile] = 1u;
                        }
                    }
                    // Phase 1 supplied a fresh maximum; only releases mutate this topology
                    // before native Build. Reuse it across unchanged waiting requests.
                    if (p.state[kEpoch] != largestEpoch) {
                        p.state[kLargestFree] = largestFreeGap(p);
                        largestEpoch = p.state[kEpoch];
                    }
                    if (bytes > p.state[kLargestFree] || p.state[kFreeBytes] < unallocatedClaim) {
                        allocation.status = kPriorityWaiting;
                        allocation.prepareDecision = kPrepareWaiting;
                        ++waitingCount;
                        waitingBytes += bytes;
                        maxWaitingBytes = max(maxWaitingBytes, bytes);
                        allocation.rejectedEpoch = p.state[kEpoch];
                        compactionTarget = max(compactionTarget, bytes);
                        info = rejectedInfo(info);
                        info.cachedValid = 2u;
                        ++storageRejected;
                        ++suppressed;
                        g_PriorityWrites[local] = 3u;
                    } else {
                        uint build = buildCount++;
                        referencesRemaining -= info.selectedClusterCount;
                        buildsRemaining -= 1u;
                        info.blasBuildIndex = build;
                        info.clusterReferenceOffset = referenceCount;
                        info.flags |= kBLASInstanceDirty;
                        info.cachedValid = 0u;
                        allocation.status = kPriorityBuilding;
                        allocation.prepareDecision = kPrepareNone;
                        ++dirtyCount;
                        referenceCount += info.selectedClusterCount;
                        if (!known) {
                            ++unknownBuilds;
                            projectedRemaining -= min(projectedRemaining, historicalBytes);
                            projectedDirtyBytes += min(historicalBytes, p.storageBytes - projectedDirtyBytes);
                        }
                        g_PriorityWrites[local] = 7u;
                    }
                    g_PriorityInfo[local] = info;
                    g_PriorityAllocations[local] = allocation;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (g_PriorityFastPath == 2u && rank < p.ownerCount &&
            (g_PriorityInfo[lane].flags & kBLASInstanceDynamic) != 0u) {
            GPUDrivenStreamAssetInstanceBLAS info = rejectedInfo(g_PriorityInfo[lane]);
            info.cachedValid = 2u;
            g_PriorityInfo[lane] = info;
            uint bytes = alignedBytes(p, g_PriorityAllocations[lane].encodedBytes);
            if (bytes != 0u && bytes <= g_PriorityStorageRemaining) {
                // Projection-only deferrals retain their entire measured history.
                InterlockedAdd(g_PriorityDeferredCount, 1u);
                g_PriorityAllocations[lane].prepareDecision = kPrepareProjected;
                g_PriorityWrites[lane] = 3u;
            } else {
                g_PriorityAllocations[lane].rejectedEpoch = p.state[kEpoch];
                g_PriorityAllocations[lane].prepareDecision = kPrepareStorageRejected;
                g_PriorityWrites[lane] = 3u;
            }
        }
        if (rank < p.ownerCount && g_PriorityEvicted[lane] == 0u) {
            uint writes = g_PriorityWrites[lane];
            uint owner = g_PriorityOwners[lane];
            GPUDrivenStreamAssetInstanceBLAS info = g_PriorityInfo[lane];
            if ((writes & 1u) != 0u) p.owners[owner] = info;
            if ((writes & 2u) != 0u) p.allocations[owner] = g_PriorityAllocations[lane];
            if ((writes & 4u) != 0u) {
                uint64_t references = p.clusterReferenceAddress + uint64_t(info.clusterReferenceOffset) * 8ul;
                GPUDrivenStreamAssetBLASBuildInfo buildInfo;
                buildInfo.clusterReferencesCount = info.selectedClusterCount;
                buildInfo.clusterReferencesStride = 8u;
                buildInfo.clusterReferencesAddressLow = uint(references);
                buildInfo.clusterReferencesAddressHigh = uint(references >> 32u);
                p.buildInfos[info.blasBuildIndex] = buildInfo;
                p.buildOwners[info.blasBuildIndex] = owner;
            }
        }
        AllMemoryBarrierWithGroupSync();
        if (lane == 0u) projectedDeferred += g_PriorityDeferredCount;
    }
    if (lane == 0u) {
        p.header[0].blasBuildCount = buildCount;
        p.header[0].clusterReferenceCount = referenceCount;
        p.header[0].dirtyInstances += dirtyCount;
        // Only real storage pressure may trigger next-frame idle-cache eviction.
        // Projection-only deferrals retain their separate retry counter.
        p.header[0].storageRejected += storageRejected - projectedDeferred;
        p.header[0].referenceBudgetRejected += referenceRejected;
        p.header[0].buildBudgetRejected += buildRejected;
        // Immediate evictions and earlier stages have already charged these fields.
        p.header[0].overflowCount += storageRejected + referenceRejected + buildRejected;
        if (dirtyCount != 0u) p.header[0].padding0 |= 1u;
        p.state[kSuppressed] += suppressed;
        p.state[kPriorityWaitingCount] = waitingCount;
        p.state[kCompactionTarget] = compactionTarget;
        p.state[kWaitingClaimBytes] = waitingBytes;
        p.state[kMaxWaitingBytes] = maxWaitingBytes;
        p.state[kUnknownBuilds] = unknownBuilds;
        p.state[kProjectedRetryDeferred] = projectedDeferred;
        p.state[kProjectedDirtyClaimBytes] = projectedDirtyBytes;
        if (p.priorityCacheControl != nullptr) {
            // No native size or residency decision remains outstanding. Later
            // clean Moves may change addresses, but admission depends only on
            // total free bytes and the largest gap, not the individual gaps.
            p.state[kCacheValid] = buildCount == 0u && p.state[kPriorityEvicted] == 0u ? 1u : 0u;
            p.state[kCachedFree] = p.state[kFreeBytes];
            p.state[kCachedLargest] = p.state[kLargestFree];
            p.state[kCachedReferenceCapacity] = p.referenceCapacity;
            p.state[kCachedBuildCapacity] = p.buildCapacity;
            p.state[kCachedStorageBytes] = p.storageBytes;
            p.state[kCachedEvictionCapacity] = p.relocateScanCount;
            p.state[kCachedStorageRejected] = storageRejected;
            p.state[kCachedWaitingCount] = waitingCount;
            p.state[kCachedWaitingBytes] = waitingBytes;
            p.state[kCachedMaxWaitingBytes] = maxWaitingBytes;
            p.state[kCachedReferenceRejected] = referenceRejected;
            p.state[kCachedBuildRejected] = buildRejected;
        }
    }
}

void restorePriorityAdmission(Parameters p, uint owner)
{
    if (owner < p.ownerCount) {
        Allocation allocation = p.allocations[owner];
        GPUDrivenStreamAssetInstanceBLAS info = p.owners[owner];
        if (allocation.inputDynamic != 0u && allocation.node == kInvalid) {
            info = rejectedInfo(info);
            // Unknown/dirty cuts invalidate the cache in phase 5. These are
            // measured negative entries with the same semantic cut and demand.
            info.cachedValid = 2u;
            if (allocation.prepareDecision == kPrepareStorageRejected ||
                allocation.prepareDecision == kPrepareWaiting) allocation.rejectedEpoch = p.state[kEpoch];
            if (allocation.prepareDecision == kPrepareWaiting) allocation.status = kPriorityWaiting;
            p.allocations[owner] = allocation;
            p.owners[owner] = info;
        }
    }
    if (owner == 0u) {
        uint storageRejected = p.state[kCachedStorageRejected];
        uint referenceRejected = p.state[kCachedReferenceRejected];
        uint buildRejected = p.state[kCachedBuildRejected];
        p.header[0].storageRejected += storageRejected;
        p.header[0].referenceBudgetRejected += referenceRejected;
        p.header[0].buildBudgetRejected += buildRejected;
        p.header[0].overflowCount += storageRejected + referenceRejected + buildRejected;
        p.state[kSuppressed] = storageRejected;
        p.state[kPriorityWaitingCount] = p.state[kCachedWaitingCount];
        p.state[kWaitingClaimBytes] = p.state[kCachedWaitingBytes];
        p.state[kMaxWaitingBytes] = p.state[kCachedMaxWaitingBytes];
    }
}

void allocatePriorityBuilds(Parameters p, uint lane)
{
    // Without new native sizes, phase 6 already performed this exact selection.
    // Waiting claims still reach the common bounded compaction planner below.
    if (p.header[0].blasBuildCount == 0u &&
        (p.priorityCacheControl != nullptr || p.state[kPriorityWaitingCount] == 0u)) return;
    uint storageRemaining = p.storageBytes;
    uint victimCount = 0u, budgetRejected = 0u;
    // Native sizes make the same ranked selection exact. Waiting higher-ranked
    // negative entries retain their claim while bounded compaction progresses.
    for (uint tile = 0u; tile < p.ownerCount; tile += 64u) {
        clearPriorityTileSummary(lane);
        uint rank = tile + lane;
        if (rank < p.ownerCount) {
            uint owner = p.priorityOrder[rank];
            g_PriorityOwners[lane] = owner;
            g_PriorityAllocations[lane] = p.allocations[owner];
            g_PrioritySizes[lane] = g_PriorityAllocations[lane].status == kPriorityBuilding
                ? p.sizes[p.owners[owner].blasBuildIndex] : 0u;
            g_PriorityWrites[lane] = 0u;
            g_PriorityVictimIndices[lane] = kInvalid;
            Allocation allocation = g_PriorityAllocations[lane];
            bool pending = allocation.status == kPriorityBuilding || allocation.status == kPriorityWaiting;
            if (pending || allocation.node != kInvalid) {
                InterlockedAdd(g_PriorityActiveCount, 1u);
                uint bytes = alignedBytes(p, allocation.encodedBytes);
                if (!pending && allocation.node != kInvalid && bytes != 0u) addPriorityResidentBytes(bytes);
            }
        }
        AllMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            if (g_PriorityActiveCount == 0u) {
                g_PriorityFastPath = 1u;
            } else if (g_PriorityResidentCount == g_PriorityActiveCount &&
                g_PriorityResidentOverflow == 0u && g_PriorityResidentBytes <= storageRemaining) {
                storageRemaining -= g_PriorityResidentBytes;
                g_PriorityFastPath = 1u;
            }
            if (g_PriorityFastPath == 0u) {
                for (uint local = 0u; local < min(64u, p.ownerCount - tile); ++local) {
                    Allocation allocation = g_PriorityAllocations[local];
                    bool building = allocation.status == kPriorityBuilding;
                    bool waiting = allocation.status == kPriorityWaiting;
                    if (!building && !waiting && allocation.node == kInvalid) continue;
                    uint encodedBytes = building ? g_PrioritySizes[local] : allocation.encodedBytes;
                    uint bytes = alignedBytes(p, encodedBytes);
                    allocation.encodedBytes = encodedBytes;
                    allocation.pendingBytes = bytes;
                    if (bytes != 0u && bytes <= storageRemaining) {
                        storageRemaining -= bytes;
                        if (!building && !waiting) continue;
                    } else if (allocation.node != kInvalid) {
                        // Compact ranks in ascending order; consume worst first below.
                        // Unselected old residents remain valid if this frame's bounded
                        // eviction allowance runs out, and remain eligible for compaction.
                        g_PriorityVictimIndices[local] = victimCount++;
                    } else if (building) {
                        allocation.status = kRejected;
                        allocation.rejectedEpoch = p.state[kEpoch];
                        ++budgetRejected;
                    } else {
                        allocation.status = 0u;
                    }
                    g_PriorityAllocations[local] = allocation;
                    g_PriorityWrites[local] = 2u;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (rank < p.ownerCount) {
            uint owner = g_PriorityOwners[lane];
            if (g_PriorityWrites[lane] != 0u) p.allocations[owner] = g_PriorityAllocations[lane];
            uint victim = g_PriorityVictimIndices[lane];
            // The relocation source list is unused until after this function;
            // native commands consume only the later published move counts.
            if (victim != kInvalid) p.moveSources[p.ownerCount + victim] = uint64_t(owner);
        }
        AllMemoryBarrierWithGroupSync();
    }
    if (lane != 0u) return;

    while (victimCount != 0u && p.state[kPriorityEvicted] < p.relocateScanCount) {
        uint owner = uint(p.moveSources[p.ownerCount + --victimCount]);
        evictPriorityOwner(p, owner);
    }
    uint moveCount = 0u;
    uint gapFailures = 0u;
    uint compactionTarget = p.state[kCompactionTarget];
    for (uint build = 0u; build < p.header[0].blasBuildCount; ++build) {
        uint owner = p.buildOwners[build];
        Allocation allocation = p.allocations[owner];
        if (allocation.status != kPriorityBuilding) continue;
        uint bytes = allocation.pendingBytes;
        uint node = allocate(p, bytes);
        if (node == kInvalid) {
            allocation.rejectedEpoch = p.state[kEpoch];
            allocation.status = kRejected;
            ++gapFailures;
            compactionTarget = max(compactionTarget, bytes);
        } else {
            allocation.pendingNode = node;
            allocation.pendingOffset = p.gaps[node].offset;
            allocation.status = kPendingBuild;
            p.moveSources[moveCount] = p.sourceAddresses[build];
            p.moveDestinations[moveCount] = p.arena + uint64_t(allocation.pendingOffset);
            ++moveCount;
        }
        p.allocations[owner] = allocation;
    }
    p.moveCounts[0] = moveCount;
    p.state[kFailures] += budgetRejected + gapFailures;
    p.state[kPostBuildBudgetRejected] = budgetRejected;
    p.state[kPhysicalGapFailures] = gapFailures;
    p.state[kCompactionTarget] = compactionTarget;
}

void recordImportance(Parameters p, uint owner, bool live)
{
    if (p.importance == nullptr) return;
    uint base = owner * 6u;
    uint visible = p.importance[base + 4u];
    uint offscreen = p.importance[base + 5u];
    InterlockedAdd(p.state[kRequestedVisible], visible);
    InterlockedAdd(p.state[kRequestedOffscreen], offscreen);
    if (live) InterlockedAdd(p.state[kAdmittedOffscreen], offscreen);
    else {
        InterlockedAdd(p.state[kRejectedVisible], visible);
        InterlockedAdd(p.state[kRejectedOffscreen], offscreen);
        InterlockedMax(p.state[kRejectedMaxError], p.importance[base]);
        if (visible != 0u) InterlockedMax(p.state[kRejectedVisibleMaxError], p.importance[base]);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void allocatorMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Parameters p = *gPush.parameters;
    uint id = dispatchThreadID.x + dispatchThreadID.y * 65535u * 64u;
    if (gPush.phase == 0u) {
        if (id < p.ownerCount) {
            Allocation allocation = { kInvalid, 0u, 0u, kInvalid, kInvalid, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
            p.allocations[id] = allocation;
        }
        if (id == 0u) {
            for (uint word = 0u; word < kStateWords; ++word) p.state[word] = 0u;
            for (uint bin = 0u; bin < kBinCount; ++bin) p.state[bin] = kInvalid;
            p.state[kNextNode] = 1u;
            p.state[kFreeNode] = kInvalid;
            p.state[kEpoch] = 1u;
            p.state[kFreeBytes] = p.storageBytes;
            Gap initial = { 0u, p.storageBytes, kInvalid, kInvalid, kInvalid, kInvalid, 0u, kInvalid };
            p.gaps[0] = initial;
            insertFree(p, 0u);
            p.moveCounts[0] = 0u; p.moveCounts[1] = 0u;
        }
        return;
    }
    if (gPush.phase == 5u) {
        if (id >= p.ownerCount) return;
        Allocation allocation = p.allocations[id];
        GPUDrivenStreamAssetInstanceBLAS info = p.owners[id];
        allocation.status = 0u;
        allocation.oldNode = kInvalid;
        allocation.pendingNode = kInvalid;
        bool dynamic = (info.flags & kBLASInstanceDynamic) != 0u;
        bool dirty = (info.flags & kBLASInstanceDirty) != 0u || p.reuseEnable == 0u;
        if (p.priorityCacheControl != nullptr) {
            if (allocation.inputDynamic != uint(dynamic) || allocation.inputSelectedClusterCount != info.selectedClusterCount ||
                (dynamic && (dirty || (allocation.node == kInvalid && allocation.encodedBytes == 0u)))) {
                InterlockedOr(p.state[kInputsChanged], 1u);
            }
            allocation.inputDynamic = uint(dynamic);
            allocation.inputSelectedClusterCount = info.selectedClusterCount;
        }
        if (p.priorityOrder != nullptr && dirty && allocation.node == kInvalid) allocation.encodedBytes = 0u;
        if (allocation.node != kInvalid && (!dynamic || dirty)) {
            if (p.priorityOrder == nullptr) allocation.status = kBaselineRelease;
            uint index;
            InterlockedAdd(p.state[kReleaseCount], 1u, index);
            p.buildOwners[index] = id;
        }
        p.allocations[id] = allocation;
        return;
    }
    if (gPush.phase == 6u) {
        if (p.priorityOrder != nullptr) {
            if (p.state[kAdmissionReused] != 0u) {
                restorePriorityAdmission(p, id);
                return;
            }
            if (id < 64u) preparePriorityBuilds(p, id);
            return;
        }
        if (id >= p.ownerCount) return;
        GPUDrivenStreamAssetInstanceBLAS info = p.owners[id];
        Allocation allocation = p.allocations[id];
        if ((info.flags & kBLASInstanceDynamic) == 0u) return;
        bool dirty = (info.flags & kBLASInstanceDirty) != 0u || p.reuseEnable == 0u;
        // Relocating unrelated objects changes the epoch without necessarily
        // creating a fitting hole. Retry an unchanged rejected cut only when
        // current free topology can hold its previously measured size.
        if (info.cachedValid == 2u && !dirty &&
            (allocation.rejectedEpoch == p.state[kEpoch] || allocation.encodedBytes > p.state[kLargestFree])) {
            rejectOwner(p, id);
            InterlockedAdd(p.state[kSuppressed], 1u);
            return;
        }
        if (!dirty && allocation.node != kInvalid) {
            info.insertedClusterCount = info.selectedClusterCount;
            InterlockedAdd(p.header[0].reusedInstances, 1u);
            InterlockedAdd(p.header[0].liveClusterReferences, info.selectedClusterCount);
        } else {
            uint build;
            InterlockedAdd(p.header[0].blasBuildCount, 1u, build);
            info.blasBuildIndex = build;
            info.flags |= kBLASInstanceDirty;
            info.cachedValid = 0u;
            uint64_t references = p.clusterReferenceAddress + uint64_t(info.clusterReferenceOffset) * 8ul;
            GPUDrivenStreamAssetBLASBuildInfo buildInfo;
            buildInfo.clusterReferencesCount = info.selectedClusterCount;
            buildInfo.clusterReferencesStride = 8u;
            buildInfo.clusterReferencesAddressLow = uint(references);
            buildInfo.clusterReferencesAddressHigh = uint(references >> 32u);
            p.buildInfos[build] = buildInfo;
            p.buildOwners[build] = id;
            // Native Build order may follow thread arrival. Storage admission
            // follows owner ID instead and reads this owner's resulting index.
            p.allocations[id].status = kBaselineBuilding;
            InterlockedAdd(p.header[0].dirtyInstances, 1u);
            InterlockedAdd(p.header[0].clusterReferenceCount, info.selectedClusterCount);
            InterlockedOr(p.header[0].padding0, 1u);
        }
        p.owners[id] = info;
        return;
    }
    if (gPush.phase == 3u) {
        if (id >= p.ownerCount) return;
        Allocation allocation = p.allocations[id];
        GPUDrivenStreamAssetInstanceBLAS info = p.owners[id];
        if (allocation.status == kRejected) {
            rejectOwner(p, id);
            info = p.owners[id];
        } else if (allocation.status == kPendingBuild || allocation.status == kPendingRelocate) {
            info.storageOffset = allocation.pendingOffset;
            info.storageCapacity = allocation.pendingBytes;
            info.cachedValid = 1u;
            allocation.node = allocation.pendingNode;
            if (allocation.status == kPendingBuild)
                InterlockedAdd(p.header[0].liveClusterReferences, info.selectedClusterCount);
            allocation.oldNode = kInvalid;
            allocation.pendingNode = kInvalid;
        }
        // Key publication follows this dispatch, including negative cut entries.
        info.previousFirstGroup = info.firstGroup;
        info.previousGroupCount = info.groupCount;
        if (allocation.node != kInvalid) {
            Gap gap = p.gaps[allocation.node];
            InterlockedAdd(p.state[kEncodedBytes], allocation.encodedBytes);
            InterlockedAdd(p.state[kAllocatedBytes], gap.size);
            InterlockedMax(p.header[0].arenaUsedBytes, gap.offset + gap.size);
        }
        bool resident = allocation.node != kInvalid && (info.flags & kBLASInstanceDynamic) != 0u &&
            (info.flags & (kBLASInstanceFallback | kBLASInstanceOverflow)) == 0u &&
            info.cachedValid == 1u && info.storageCapacity != 0u;
        if (p.priorityOrder != nullptr && resident && allocation.status != kPendingBuild) {
            info.insertedClusterCount = info.selectedClusterCount;
            InterlockedAdd(p.header[0].reusedInstances, 1u);
            InterlockedAdd(p.header[0].liveClusterReferences, info.selectedClusterCount);
        }
        recordImportance(p, id, resident && info.insertedClusterCount == info.selectedClusterCount);
        p.allocations[id] = allocation;
        p.owners[id] = info;
        return;
    }
    if (gPush.phase == 4u) {
        if (id >= p.ownerCount) return;
        // Baseline instrumentation preserves the legacy allocation policy.
        GPUDrivenStreamAssetInstanceBLAS info = p.owners[id];
        bool live = (info.flags & kBLASInstanceDynamic) != 0u &&
            (info.flags & (kBLASInstanceFallback | kBLASInstanceOverflow)) == 0u &&
            info.cachedValid == 1u && info.storageCapacity != 0u;
        if (!live) p.allocations[id].encodedBytes = 0u;
        else {
            if ((info.flags & kBLASInstanceDirty) != 0u)
                p.allocations[id].encodedBytes = p.sizes[info.blasBuildIndex];
            InterlockedAdd(p.state[kEncodedBytes], p.allocations[id].encodedBytes);
            InterlockedAdd(p.state[kAllocatedBytes], info.storageCapacity);
            InterlockedAdd(p.state[kFreeBytes], 0u - info.storageCapacity);
        }
        return;
    }
    if (gPush.phase == 2u && p.priorityOrder != nullptr) {
        if (id >= 64u) return;
        allocatePriorityBuilds(p, id);
    }
    if (id != 0u) return;
    if (gPush.phase == 9u) {
        p.state[kEncodedBytes] = 0u;
        p.state[kAllocatedBytes] = 0u;
        p.state[kFreeBytes] = p.storageBytes;
        // The legacy allocator can only use its tail without a full repack.
        p.state[kLargestFree] = p.storageBytes - min(p.header[0].arenaUsedBytes, p.storageBytes);
        return;
    }
    if (gPush.phase == 1u) {
        uint releaseCount = p.state[kReleaseCount];
        // Readers of the previous frame have completed before arena mutation.
        // Baseline releases must have a stable order too: inserting disjoint
        // holes in atomic-gather order changes free-bin first-fit selection.
        // An empty release list skips the owner scan entirely.
        uint releaseLimit = releaseCount == 0u ? 0u :
            (p.priorityOrder == nullptr ? p.ownerCount : releaseCount);
        for (uint index = 0u; index < releaseLimit; ++index) {
            uint owner = p.priorityOrder == nullptr ? index : p.buildOwners[index];
            Allocation allocation = p.allocations[owner];
            if (p.priorityOrder == nullptr && allocation.status != kBaselineRelease) continue;
            releaseAllocation(p, allocation.node);
            allocation.node = kInvalid;
            allocation.encodedBytes = 0u;
            if (p.priorityOrder == nullptr) allocation.status = 0u;
            p.allocations[owner] = allocation;
            p.owners[owner].storageOffset = 0u;
            p.owners[owner].storageCapacity = 0u;
        }
        // A fragmented request survives frames where the bounded scan finds no
        // movable owner. Negative-cache suppression must not stop its progress.
        for (uint word = kReleaseCount; word < kCompactionTarget; ++word) {
            if (word != kFreeBytes) p.state[word] = 0u;
        }
        // Gap sizes are aligned, so encodedBytes <= largestFree is equivalent
        // to the aligned allocation fitting. Phase 6 consumes this snapshot.
        p.state[kLargestFree] = largestFreeGap(p);
        p.moveCounts[0] = 0u; p.moveCounts[1] = 0u;
        p.header[0].blasBuildCount = 0u;
        p.header[0].arenaRepack = 0u;
        p.state[kPriorityWaitingCount] = 0u;
        for (uint word = kRequestedVisible; word < kSortReused; ++word) p.state[word] = 0u;
        bool cacheEnabled = p.priorityCacheControl != nullptr;
        p.state[kSortReused] = cacheEnabled ? p.priorityCacheControl[1] : 0u;
        p.state[kAdmissionReused] = cacheEnabled && p.priorityOrder != nullptr && p.reuseEnable != 0u &&
            p.state[kCacheValid] != 0u && p.priorityCacheControl[0] == 0u && p.state[kInputsChanged] == 0u &&
            p.state[kCachedFree] == p.state[kFreeBytes] && p.state[kCachedLargest] == p.state[kLargestFree] &&
            p.state[kCachedReferenceCapacity] == p.referenceCapacity && p.state[kCachedBuildCapacity] == p.buildCapacity &&
            p.state[kCachedStorageBytes] == p.storageBytes && p.state[kCachedEvictionCapacity] == p.relocateScanCount ? 1u : 0u;
        if (!cacheEnabled) p.state[kCacheValid] = 0u;
        p.state[kInputsChanged] = 0u;
        return;
    }
    if (gPush.phase == 2u) {
        if (p.priorityOrder == nullptr) {
            uint moveCount = 0u;
            // Preserve parallel native Build packing while making budget
            // winners independent of workgroup/atomic arrival order. Only
            // actual Builds scan owners; stable frames retain zero work.
            uint ownerLimit = p.header[0].blasBuildCount == 0u ? 0u : p.ownerCount;
            for (uint owner = 0u; owner < ownerLimit; ++owner) {
                Allocation allocation = p.allocations[owner];
                if (allocation.status != kBaselineBuilding) continue;
                uint build = p.owners[owner].blasBuildIndex;
                uint encodedBytes = p.sizes[build];
                uint bytes = encodedBytes <= 0xffffffffu - (p.alignment - 1u)
                    ? (encodedBytes + p.alignment - 1u) & ~(p.alignment - 1u) : 0u;
                uint node = bytes != 0u && bytes <= p.storageBytes ? allocate(p, bytes) : kInvalid;
                if (node == kInvalid) {
                    allocation.rejectedEpoch = p.state[kEpoch];
                    allocation.status = kRejected;
                    allocation.encodedBytes = encodedBytes;
                    p.state[kFailures] += 1u;
                    if (bytes <= p.storageBytes) p.state[kCompactionTarget] = max(p.state[kCompactionTarget], bytes);
                } else {
                    allocation.pendingNode = node;
                    allocation.pendingOffset = p.gaps[node].offset;
                    allocation.pendingBytes = bytes;
                    allocation.encodedBytes = encodedBytes;
                    allocation.status = kPendingBuild;
                    p.moveSources[moveCount] = p.sourceAddresses[build];
                    p.moveDestinations[moveCount] = p.arena + uint64_t(allocation.pendingOffset);
                    ++moveCount;
                }
                p.allocations[owner] = allocation;
            }
            p.moveCounts[0] = moveCount;
        }
        uint target = p.state[kCompactionTarget];
        if (target != 0u && p.relocateBudget != 0u) {
            // Releases and dirty allocations changed the bins this frame; the
            // previous telemetry snapshot cannot decide current pressure.
            uint largest = largestFreeGap(p);
            uint available = p.state[kFreeBytes];
            // A larger rejected request must not prevent smaller requests from
            // benefiting from coalescing the capacity that is actually free.
            target = min(target, available);
            if (largest >= target) target = 0u;
            p.state[kCompactionTarget] = target;
        }
        uint relocatedBytes = 0u;
        uint relocatedCount = 0u;
        uint cursor = p.state[kCursor];
        uint scans = target != 0u ? min(p.ownerCount, p.relocateScanCount) : 0u;
        for (uint scan = 0u; scan < scans && relocatedBytes < p.relocateBudget; ++scan) {
            uint owner = cursor++;
            if (cursor == p.ownerCount) cursor = 0u;
            Allocation allocation = p.allocations[owner];
            if (allocation.node == kInvalid || allocation.status != 0u) continue;
            Gap source = p.gaps[allocation.node];
            if (source.size > p.relocateBudget - relocatedBytes || source.previous == kInvalid ||
                p.gaps[source.previous].free == 0u) continue;
            allocation.oldNode = allocation.node;
            allocation.pendingOffset = relocatedBytes;
            allocation.pendingBytes = source.size;
            allocation.status = kPendingRelocate;
            // Stage first: even a hole smaller than every live BLAS can then
            // merge with its following source, without overlapping native Move.
            p.moveSources[p.ownerCount + relocatedCount] = p.arena + uint64_t(source.offset);
            p.moveDestinations[p.ownerCount + relocatedCount] = p.relocationArena + uint64_t(relocatedBytes);
            p.buildOwners[relocatedCount] = owner;
            ++relocatedCount;
            relocatedBytes += source.size;
            p.allocations[owner] = allocation;
        }
        p.state[kCursor] = cursor;
        p.state[kRelocated] = relocatedCount;
        p.state[kRelocatedBytes] = relocatedBytes;
        p.moveCounts[1] = relocatedCount;
        return;
    }
    if (gPush.phase == 8u) {
        for (uint move = 0u; move < p.moveCounts[1]; ++move) {
            uint owner = p.buildOwners[move];
            Allocation allocation = p.allocations[owner];
            uint previous = p.gaps[allocation.oldNode].previous;
            uint destination = previous != kInvalid && p.gaps[previous].free != 0u ? previous : allocation.oldNode;
            // All clean sources have reached separate staging. Reserve from
            // this source's newly coalesced gap so concurrent relocation plans
            // cannot consume each other's only fitting destination.
            releaseAllocation(p, allocation.oldNode);
            takeGap(p, destination, allocation.pendingBytes);
            p.moveSources[p.ownerCount + move] = p.relocationArena + uint64_t(allocation.pendingOffset);
            allocation.pendingNode = destination;
            allocation.pendingOffset = p.gaps[destination].offset;
            p.moveDestinations[p.ownerCount + move] = p.arena + uint64_t(allocation.pendingOffset);
            p.allocations[owner] = allocation;
        }
        uint largestFree = largestFreeGap(p);
        uint freeBytes = p.state[kFreeBytes];
        p.state[kEncodedBytes] = 0u;
        p.state[kAllocatedBytes] = 0u;
        p.state[kLargestFree] = largestFree;
        uint target = min(p.state[kCompactionTarget], freeBytes);
        p.state[kCompactionTarget] = largestFree >= target ? 0u : target;
        p.header[0].arenaUsedBytes = 0u;
    }
}
