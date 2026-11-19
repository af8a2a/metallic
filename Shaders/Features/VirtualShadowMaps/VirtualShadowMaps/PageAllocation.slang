#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_RECEIVER_MASK
#define VSM_RECEIVER_MASK 0
#endif
#ifndef VSM_CLUSTER_PAGE_BITSET
#define VSM_CLUSTER_PAGE_BITSET 0
#endif
#ifndef VSM_SHARED_PHYSICAL_POOL
#define VSM_SHARED_PHYSICAL_POOL 0
#endif
#ifndef VSM_COMPACT_SHADOW_VIEWS
#define VSM_COMPACT_SHADOW_VIEWS 0
#endif

#if VSM_COMPACT_SHADOW_VIEWS
void vsmPublishFineViewBits(ShadowPush push, uint viewBits)
{
    uint mask = WaveActiveBitOr(viewBits);
    if (mask != 0u && WaveIsFirstLane()) {
        RWStructuredBuffer<uint> data = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.sampleReserved));
        InterlockedOr(data[kVSMActiveFineViewMask], mask);
    }
}
#endif

// Allocation scratch follows the physical owner table in the same buffer:
// local page ranks, block counts/prefixes, then free/evictable physical slots.
// Every scan invocation participates, including rounded dispatch tails.
groupshared uint3 gVsmWarpTotals[8];
groupshared uint3 gVsmAllocationTotal;

uint3 vsmAllocationPrefix(uint3 value, uint lane)
{
    uint3 warpPrefix = WavePrefixSum(value);
    uint3 warpTotal = WaveActiveSum(value);
    uint laneInWarp = WaveGetLaneIndex();
    uint warpIndex = lane / WaveGetLaneCount();
    if (laneInWarp == 0u) {
        gVsmWarpTotals[warpIndex] = warpTotal;
    }
    GroupMemoryBarrierWithGroupSync();

    uint warpCount = (256u + WaveGetLaneCount() - 1u) / WaveGetLaneCount();
    if (lane < warpCount) {
        uint3 total = gVsmWarpTotals[lane];
        uint3 prefix = WavePrefixSum(total);
        uint3 totalSum = WaveActiveSum(total);
        gVsmWarpTotals[lane] = prefix;
        if (lane == 0u) {
            gVsmAllocationTotal = totalSum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    return warpPrefix + gVsmWarpTotals[warpIndex];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmCountMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    uniform ShadowPush push)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> scratch = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint3 counts = uint3(0u);
    if (id.x < push.totalPages) {
        uint4 entry = pages[id.x];
        counts.x = id.x + 1u < push.totalPages && (entry.y & 1u) != 0u ? 1u : 0u;
        if ((push.cacheMode & 1u) != 0u && entry.x != 0xffffffffu) {
            uint mappingFlags = scratch[push.physicalPageCount + id.x];
            counts.y = mappingFlags & 1u;
            counts.z = (mappingFlags >> 1u) & 1u;
        }
    }
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    if (id.x < push.totalPages) { scratch[push.physicalPageCount + id.x] = prefix.x; }
    if (lane == 255u) {
        uint3 total = prefix + counts;
        scratch[push.physicalPageCount + push.totalPages + group.x] = total.x;
        InterlockedAdd(statistics[18], total.y);
        InterlockedAdd(statistics[19], total.z);
    }
}

// The established allocation entry now prefixes per-block demand counts;
// its one group replaces the old serial traversal of the full virtual table.
[shader("compute")]
[numthreads(256, 1, 1)]
void vsmAllocateMain(uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    RWStructuredBuffer<uint> scratch = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint blockCount = (push.totalPages + 255u) / 256u;
    uint blocksPerLane = (blockCount + 255u) / 256u;
    uint first = lane * blocksPerLane, end = min(first + blocksPerLane, blockCount);
    uint offset = push.physicalPageCount + push.totalPages;
    uint count = 0u;
    for (uint block = first; block < end; ++block) { count += scratch[offset + block]; }
    uint prefix = vsmAllocationPrefix(uint3(count, 0u, 0u), lane).x;
    for (uint block = first; block < end; ++block) {
        uint value = scratch[offset + block];
        scratch[offset + block] = prefix;
        prefix += value;
    }
    if (lane == 0u) {
        uint requested = gVsmAllocationTotal.x;
        statistics[0] = requested;
#if !VSM_SHARED_PHYSICAL_POOL
        uint allocated = min(requested, push.physicalPageCount - 1u);
        statistics[1] = allocated;
        statistics[2] = requested - allocated;
#endif
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmAdmitMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
    uniform ShadowPush push)
{
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint3 counts = uint3(0u);
#if VSM_COMPACT_SHADOW_VIEWS
    uint fineViewBit = 0u;
#endif
    uint rankOffset = push.physicalPageCount;
    uint blockOffset = rankOffset + push.totalPages;
#if VSM_SHARED_PHYSICAL_POOL
    RWStructuredBuffer<uint4> physicalPool = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.physicalPool));
    uint admittedFine = physicalPool[1u + push.poolFace].x;
    uint physicalOwnerBase = 1u + physicalPool[0].x;
    if (id.x == 0u) {
        statistics[1] = admittedFine;
        statistics[2] = statistics[0] - admittedFine;
        statistics[12] = admittedFine + 1u;
    }
#else
    uint admittedFine = push.physicalPageCount - 1u;
#endif
    if (id.x < push.totalPages) {
        uint4 entry = pages[id.x];
        bool fine = id.x + 1u < push.totalPages;
        if (fine && (entry.y & 1u) != 0u) {
            uint rank = owners[blockOffset + group.x] + owners[rankOffset + id.x];
            // Keep requested parent coverage before spending the remaining
            // budget on refinement. A missing fine page can then use a nearby
            // coarser mip instead of the complete low-resolution scene map.
            rank = statistics[0] - 1u - rank;
            if (rank < admittedFine) {
                entry.y |= 2u;
#if VSM_RECEIVER_MASK
                // Reset carried coverage through the same oldIndex remap as
                // residency. Mark is complete before Admit: retain a clean page
                // only when its existing depth covers every requested subtile.
                if (entry.x != 0xffffffffu && (entry.w & 3u) == 1u) {
                    uint offset = vsmReceiverMaskOffset(push) + id.x * 2u;
                    uint2 requested = uint2(owners[offset], owners[offset + 1u]);
                    ByteAddressBuffer validCoverage = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.pages));
                    uint2 valid = validCoverage.Load2(vsmValidReceiverMaskByteOffset(push, id.x));
                    if (any((requested & ~valid) != uint2(0u))) {
                        entry.w = (entry.w & ~1u) | 2u;
                        InterlockedAdd(statistics[18], 1u);
                        vsmCacheAdd(push, kVSMCacheCompositeInvalidatedResidentPages, 1u);
                    }
                }
#endif
            }
            else { entry.x = 0xffffffffu; entry.w = 2u | ((push.cacheMode & 16u) != 0u ? 8u : 0u); }
        }
        pages[id.x] = entry;
        if (entry.x != 0xffffffffu) {
            owners[entry.x] = id.x + 1u;
#if VSM_SHARED_PHYSICAL_POOL
            // Reset retained only this pool generation's mappings. Every slot
            // therefore has one writer across all face Admit dispatches.
            physicalPool[physicalOwnerBase + entry.x] = uint4(push.poolFace + 1u, id.x + 1u,
                uint(!fine || (entry.y & 2u) != 0u), 0u);
#endif
        }
        if (fine && (entry.y & 2u) != 0u) {
            counts.x = entry.x == 0xffffffffu ? 1u : 0u;
            counts.y = entry.x != 0xffffffffu && (entry.w & 2u) != 0u ? 1u : 0u;
            counts.z = entry.x != 0xffffffffu && (entry.w & 3u) == 1u ? 1u : 0u;
#if VSM_CLUSTER_PAGE_BITSET
            // Both raster layers consume the same Build list, including static-only merges.
            if (entry.x != 0xffffffffu && (entry.w & 10u) != 0u) {
                InterlockedOr(owners[vsmClusterPageBitsetOffset(push) + (id.x >> 5u)], 1u << (id.x & 31u));
#if VSM_COMPACT_SHADOW_VIEWS
                fineViewBit = 1u << entry.z;
#endif
            }
#endif
        }
    }
#if VSM_COMPACT_SHADOW_VIEWS
    vsmPublishFineViewBits(push, fineViewBit);
#endif
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    if (id.x < push.totalPages) { owners[rankOffset + id.x] = counts.x != 0u ? prefix.x : 0xffffffffu; }
    if (lane == 255u) {
        uint3 total = prefix + counts;
        owners[blockOffset + group.x] = total.x;
        InterlockedAdd(statistics[16], total.y);
        InterlockedAdd(statistics[17], total.z);
    }
}

uint2 vsmPhysicalHistory(ShadowPush push, uint previousBase, uint index)
{
    if ((push.cacheMode & 3u) == 3u) {
        StructuredBuffer<uint2> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.previousPages));
        return previous[previousBase + index];
    }
    return uint2(index, 0u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmPhysicalMain(uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    uint blockCount = (push.totalPages + 255u) / 256u;
    uint blocksPerLane = (blockCount + 255u) / 256u;
    uint firstBlock = lane * blocksPerLane, endBlock = min(firstBlock + blocksPerLane, blockCount);
    uint blockOffset = push.physicalPageCount + push.totalPages;
    uint candidatesOffset = blockOffset + blockCount;
    uint slotsPerLane = (push.physicalPageCount - 1u + 255u) / 256u;
    uint firstSlot = 1u + lane * slotsPerLane, endSlot = min(firstSlot + slotsPerLane, push.physicalPageCount);
    // Use the actual descriptor range: the pool may return an oversized
    // buffer, and the previous snapshot can have a different mapping layout.
    uint historySize = (push.physicalPageCount + 1u) & ~1u;
    uint previousBase = 0u, nextBase = 0u, frame = 1u;
    if ((push.cacheMode & 3u) == 3u) {
        StructuredBuffer<uint2> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.previousPages));
        uint size, stride;
        previous.GetDimensions(size, stride);
        previousBase = size - historySize;
        frame = previous[previousBase].y + 1u;
    }
    RWStructuredBuffer<uint2> history = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.pages));
    if ((push.cacheMode & 2u) != 0u) {
        uint size, stride;
        history.GetDimensions(size, stride);
        nextBase = size - historySize;
        if (lane == 0u) { history[nextBase] = uint2(0u, frame); }
    }
    uint3 counts = uint3(0u);
    for (uint block = firstBlock; block < endBlock; ++block) { counts.x += owners[blockOffset + block]; }
    for (uint index = firstSlot; index < endSlot; ++index) {
        uint owner = owners[vsmPhysicalHistory(push, previousBase, index).x];
        if (owner == 0u) { ++counts.y; }
        else if ((pages[owner - 1u].y & 2u) == 0u) { ++counts.z; }
    }
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    uint3 totals = gVsmAllocationTotal;
    uint candidateCount = totals.y + totals.z;
    for (uint block = firstBlock; block < endBlock; ++block) {
        uint count = owners[blockOffset + block];
        owners[blockOffset + block] = prefix.x;
        prefix.x += count;
    }
    // Stable partition in the previous frame's LRU order: free, unrequested,
    // pinned. Assign consumes the first missing candidates. Rotate that prefix
    // to the newest end so every surviving unrequested owner keeps its order.
    // Same-frame hits tie deterministically; ordering never compares frame IDs.
    for (uint index = firstSlot; index < endSlot; ++index) {
        uint2 prior = vsmPhysicalHistory(push, previousBase, index);
        uint physical = prior.x;
        uint owner = owners[physical];
        uint rank;
        if (owner == 0u) { rank = prefix.y++; }
        else if ((pages[owner - 1u].y & 2u) == 0u) { rank = totals.y + prefix.z++; }
        else { rank = candidateCount + index - 1u - prefix.y - prefix.z; }
        if (rank < candidateCount) { owners[candidatesOffset + rank] = physical; }
        if ((push.cacheMode & 2u) != 0u) {
            bool assigned = rank < totals.x;
            uint nextRank = assigned ? rank + push.physicalPageCount - 1u - totals.x : rank - totals.x;
            uint lastRequestedFrame = assigned || rank >= candidateCount ? frame : (owner != 0u ? prior.y : 0u);
            history[nextBase + 1u + nextRank] = uint2(physical, lastRequestedFrame);
#if VSM_WORKLOAD_DIAGNOSTICS
            if (assigned && owner != 0u) {
                vsmWorkAdd(push, kVSMWorkAllocatorLRUEvictions, 1u);
                vsmWorkAdd(push, kVSMWorkAllocatorLRUEvictionAgeFrames, frame - prior.y);
            }
#endif
        }
    }
    if (lane == 0u) {
        uint missing = totals.x;
        statistics[20] = missing;
        statistics[16] += missing;
        uint coarseFlags = pages[push.totalPages - 1u].w;
        statistics[21] = push.enabled != 0u && (coarseFlags & 2u) != 0u ? 1u : 0u;
        if ((push.cacheMode & 32u) != 0u) {
            // Slot zero is pinned: the final word of the physical candidate array
            // is unused. Publish a snapshot before Finalize updates coarse validity.
            owners[candidatesOffset + push.physicalPageCount - 1u] = coarseFlags;
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmAssignMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uniform ShadowPush push)
{
    if (id.x >= push.totalPages) { return; }
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    uint blockCount = (push.totalPages + 255u) / 256u;
    uint blockOffset = push.physicalPageCount + push.totalPages;
#if VSM_SHARED_PHYSICAL_POOL
    RWStructuredBuffer<uint4> physicalPool = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.physicalPool));
    uint4 header = physicalPool[0];
    uint4 face = physicalPool[1u + push.poolFace];
    bool coarse = id.x + 1u == push.totalPages;
    uint rank;
    if (coarse) {
        if (face.w == 0u) { return; }
        rank = face.z + face.y;
    } else {
        uint localRank = owners[push.physicalPageCount + id.x];
        if (localRank == 0xffffffffu) { return; }
        rank = face.z + owners[blockOffset + group.x] + localRank;
    }
    uint ownerBase = 1u + header.x;
    uint physical = physicalPool[ownerBase + push.physicalPageCount + rank].x;
    uint4 evicted = physicalPool[ownerBase + physical];
    if (evicted.x != 0u) {
        // Global Physical selected only unrequested owners. A victim cannot
        // also be an Assign invocation, even when it belongs to another face.
        StructuredBuffer<ShadowPush> jobs = resolveUniform(ResourceHandle<StructuredBuffer<ShadowPush>>(header.z));
        ShadowPush victim = jobs[evicted.x - 1u];
        RWStructuredBuffer<uint4> victimPages = resolveNonUniform(ResourceHandle<RWStructuredBuffer<uint4>>(victim.pages));
        RWStructuredBuffer<uint> victimOwners = resolveNonUniform(ResourceHandle<RWStructuredBuffer<uint>>(victim.owners));
        uint virtualPage = evicted.y - 1u;
        uint4 old = victimPages[virtualPage];
        old.x = 0xffffffffu;
        old.w = 2u | ((victim.cacheMode & 16u) != 0u ? 8u : 0u);
        victimPages[virtualPage] = old;
        victimOwners[physical] = 0u;
        // Victim permutations may differ from this allocator's permutation.
        // The producer enables this tail exactly for layered directional SMRT.
        if (victim.lightType == 0u && (victim.cacheMode & 16u) != 0u && victim.filterMode == 2u) {
            RWByteAddressBuffer validCoverage = resolveNonUniform(ResourceHandle<RWByteAddressBuffer>(victim.pages));
            validCoverage.Store2(victim.totalPages * 32u + virtualPage * 8u, uint2(0u));
        }
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmWorkAdd(push, kVSMWorkAllocatorLRUEvictions, 1u);
        vsmWorkAdd(push, kVSMWorkAllocatorLRUEvictionAgeFrames, header.w - evicted.w);
#endif
    }
    physicalPool[ownerBase + physical] = uint4(push.poolFace + 1u, id.x + 1u, 1u, header.w);
#else
    uint localRank = owners[push.physicalPageCount + id.x];
    if (localRank == 0xffffffffu) { return; }
    uint rank = owners[blockOffset + group.x] + localRank;
    uint physical = owners[blockOffset + blockCount + rank];
    uint evicted = owners[physical];
    if (evicted != 0u) {
        // Only missing admitted pages run this phase. An evicted unrequested
        // page has no competing invocation reading/writing its page entry.
        uint4 old = pages[evicted - 1u];
        old.x = 0xffffffffu; old.w = 2u;
        pages[evicted - 1u] = old;
#if VSM_RECEIVER_MASK
        RWByteAddressBuffer validCoverage = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.pages));
        validCoverage.Store2(vsmValidReceiverMaskByteOffset(push, evicted - 1u), uint2(0u));
#endif
    }
#endif
    uint4 entry = pages[id.x];
    entry.x = physical; entry.w = 2u | ((push.cacheMode & 16u) != 0u ? 8u : 0u);
    pages[id.x] = entry;
    owners[physical] = id.x + 1u;
#if VSM_CLUSTER_PAGE_BITSET
    // This path handles only admitted missing pages. The writes above make
    // each one resident and dirty; no invocation reads another evicted page.
    if (id.x + 1u < push.totalPages) {
        InterlockedOr(owners[vsmClusterPageBitsetOffset(push) + (id.x >> 5u)], 1u << (id.x & 31u));
#if VSM_COMPACT_SHADOW_VIEWS
        // A wave may contain only the allocating lanes. Each subset merges its views.
        vsmPublishFineViewBits(push, 1u << entry.z);
#endif
    }
#endif
}
