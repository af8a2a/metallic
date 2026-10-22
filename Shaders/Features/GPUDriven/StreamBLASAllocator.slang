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
};
struct Push { Parameters* parameters; uint phase, reserved; };
[[vk::push_constant]] ConstantBuffer<Push> gPush;

static const uint kInvalid = 0xffffffffu;
static const uint kBinCount = 96u, kStateWords = 112u;
static const uint kNextNode = 96u, kFreeNode = 97u, kTail = 98u, kEpoch = 99u, kCursor = 100u;
static const uint kReleaseCount = 101u;
static const uint kEncodedBytes = 102u, kAllocatedBytes = 103u, kLargestFree = 104u, kFreeBytes = 105u;
static const uint kSuppressed = 106u, kRelocated = 107u, kRelocatedBytes = 108u, kFailures = 109u;
static const uint kCompactionTarget = 110u;
static const uint kPendingBuild = 1u, kPendingRelocate = 2u, kRejected = 3u;

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

[shader("compute")]
[numthreads(64, 1, 1)]
void allocatorMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Parameters p = *gPush.parameters;
    uint id = dispatchThreadID.x + dispatchThreadID.y * 65535u * 64u;
    if (gPush.phase == 0u) {
        if (id < p.ownerCount) {
            Allocation allocation = { kInvalid, 0u, 0u, kInvalid, kInvalid, 0u, 0u, 0u };
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
        if (allocation.node != kInvalid && (!dynamic || dirty)) {
            uint index;
            InterlockedAdd(p.state[kReleaseCount], 1u, index);
            p.buildOwners[index] = id;
        }
        p.allocations[id] = allocation;
        return;
    }
    if (gPush.phase == 6u) {
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
        // Only dirty/dead owners, gathered in parallel, require topology work.
        for (uint index = 0u; index < releaseCount; ++index) {
            uint owner = p.buildOwners[index];
            Allocation allocation = p.allocations[owner];
            releaseAllocation(p, allocation.node);
            allocation.node = kInvalid;
            allocation.encodedBytes = 0u;
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
        return;
    }
    if (gPush.phase == 2u) {
        uint moveCount = 0u;
        for (uint build = 0u; build < p.header[0].blasBuildCount; ++build) {
            uint owner = p.buildOwners[build];
            Allocation allocation = p.allocations[owner];
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
