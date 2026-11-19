#language slang 2026
implementing VirtualShadowMaps;

groupshared uint gVsmSharedMissingCount;

uint vsmSharedRequestedFine(ShadowPush job)
{
    StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(job.statistics));
    return statistics[0];
}

// Host admission guarantees faceCount <= capacity. One coarse page per face
// is reserved before max-min fair sharing of the remaining fine capacity.
[shader("compute")]
[numthreads(256, 1, 1)]
void vsmSharedPoolPrepareMain(uint lane : SV_GroupIndex, uniform VirtualShadowMapPhysicalPoolPush push)
{
    StructuredBuffer<ShadowPush> jobs = resolveUniform(ResourceHandle<StructuredBuffer<ShadowPush>>(push.jobs));
    RWStructuredBuffer<uint4> metadata = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.metadata));
    for (uint physical = lane; physical < push.capacity; physical += 256u) {
        metadata[1u + push.faceCount + physical] = uint4(0u);
    }
    uint clearHeaderQuad = vsmSharedPoolClearHeaderQuad(push.faceCount, push.capacity);
    uint clearInitCount = 1u + 2u * vsmSharedPoolClearBatches(push.capacity);
    for (uint i = lane; i < clearInitCount; i += 256u) {
        metadata[clearHeaderQuad + i] = (i == 0u) ? uint4(0u) : uint4(0u, 1u, 1u, 0u);
    }
    if (lane == 0u) {
        metadata[0] = uint4(push.faceCount, push.capacity, push.jobs, push.frameIndex);
        if (push.faceCount == 0u) { return; }
        uint available = push.capacity - push.faceCount;
        uint waterLevel = available / push.faceCount;
        uint remainder = available % push.faceCount;
        if (push.demandDriven != 0u) {
            uint lower = 0u, upper = available;
            while (lower < upper) {
                uint level = lower + (upper - lower + 1u) / 2u;
                uint needed = 0u;
                for (uint face = 0u; face < push.faceCount; ++face) {
                    needed += min(vsmSharedRequestedFine(jobs[face]), level);
                }
                if (needed <= available) { lower = level; }
                else { upper = level - 1u; }
            }
            waterLevel = lower;
            remainder = available;
            for (uint face = 0u; face < push.faceCount; ++face) {
                remainder -= min(vsmSharedRequestedFine(jobs[face]), waterLevel);
            }
        }
        uint requestedFine = 0u;
        for (uint face = 0u; face < push.faceCount; ++face) {
            uint target = vsmSharedRequestedFine(jobs[face]);
            requestedFine += target;
            uint grant = min(target, waterLevel);
            if (push.demandDriven != 0u) {
                if (target > waterLevel && remainder != 0u) { ++grant; --remainder; }
            } else if (face < remainder && target > waterLevel) {
                ++grant;
            }
            metadata[1u + face] = uint4(grant, 0u, 0u, 0u);
        }
        metadata[1u + push.faceCount + push.capacity * 2u] = uint4(requestedFine, 0u, 0u, 0u);
    }
}

uint2 vsmSharedPreviousPhysical(VirtualShadowMapPhysicalPoolPush push, uint index)
{
    if (push.previousHistory != 0xffffffffu) {
        StructuredBuffer<uint2> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.previousHistory));
        return previous[index];
    }
    return uint2(index, 0u);
}

uint2 vsmSharedPreviousAssignment(VirtualShadowMapPhysicalPoolPush push, uint physicalPage)
{
    if (push.previousHistory != 0xffffffffu) {
        StructuredBuffer<uint2> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint2>>(push.previousHistory));
        return previous[push.capacity + physicalPage];
    }
    return uint2(0u);
}

// Admit for every face has published its local owner table and the disjoint
// global owner records. Requested owners cannot be taken by any other face.
[shader("compute")]
[numthreads(256, 1, 1)]
void vsmSharedPoolPhysicalMain(uint lane : SV_GroupIndex, uniform VirtualShadowMapPhysicalPoolPush push)
{
    StructuredBuffer<ShadowPush> jobs = resolveUniform(ResourceHandle<StructuredBuffer<ShadowPush>>(push.jobs));
    RWStructuredBuffer<uint4> metadata = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.metadata));
    RWStructuredBuffer<uint2> history = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.history));
    if (lane == 0u) {
        uint missingBase = 0u;
        for (uint face = 0u; face < push.faceCount; ++face) {
            ShadowPush job = jobs[face];
            RWStructuredBuffer<uint> scratch = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(job.owners));
            RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(job.statistics));
            StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(job.pages));
            uint blockCount = (job.totalPages + 255u) / 256u;
            uint blockOffset = job.physicalPageCount + job.totalPages;
            uint missingFine = 0u;
            for (uint block = 0u; block < blockCount; ++block) {
                uint count = scratch[blockOffset + block];
                scratch[blockOffset + block] = missingFine;
                missingFine += count;
            }
            uint4 coarse = pages[job.totalPages - 1u];
            uint missingCoarse = uint(coarse.x == 0xffffffffu);
            uint grant = metadata[1u + face].x;
            metadata[1u + face] = uint4(grant, missingFine, missingBase, missingCoarse);
            missingBase += missingFine + missingCoarse;
            statistics[20] = missingFine;
            statistics[16] += missingFine;
            statistics[21] = job.enabled != 0u && (coarse.w & 2u) != 0u ? 1u : 0u;
            if ((job.cacheMode & 32u) != 0u) {
                // Shared allocation uses the global candidate list. This word
                // retains the existing static-merge snapshot ABI for Finalize.
                scratch[blockOffset + blockCount + job.physicalPageCount - 1u] = coarse.w;
            }
        }
        gVsmSharedMissingCount = missingBase;
    }
    GroupMemoryBarrierWithGroupSync();

    uint slotsPerLane = (push.capacity + 255u) / 256u;
    uint first = lane * slotsPerLane, end = min(first + slotsPerLane, push.capacity);
    uint ownerBase = 1u + push.faceCount;
    uint candidateBase = ownerBase + push.capacity;
    uint3 counts = uint3(0u);
    for (uint index = first; index < end; ++index) {
        uint physicalPage = vsmSharedPreviousPhysical(push, index).x;
        uint4 owner = metadata[ownerBase + physicalPage];
        bool free = owner.x == 0u &&
            (push.preserveDormant == 0u || all(vsmSharedPreviousAssignment(push, physicalPage) == 0u));
        if (free) { ++counts.x; }
        else if (owner.z == 0u) { ++counts.y; }
        else { ++counts.z; }
    }
    uint3 prefix = vsmAllocationPrefix(counts, lane);
    uint3 totals = gVsmAllocationTotal;
    uint candidateCount = totals.x + totals.y;
    uint missing = gVsmSharedMissingCount;
    for (uint index = first; index < end; ++index) {
        uint2 prior = vsmSharedPreviousPhysical(push, index);
        uint4 owner = metadata[ownerBase + prior.x];
        uint2 priorAssignment = vsmSharedPreviousAssignment(push, prior.x);
        bool free = owner.x == 0u && (push.preserveDormant == 0u || all(priorAssignment == 0u));
        uint rank = free ? prefix.x++ :
            owner.z == 0u ? totals.x + prefix.y++ : candidateCount + prefix.z++;
        if (rank < candidateCount) {
            metadata[candidateBase + rank] = uint4(prior.x, prior.y, 0u, 0u);
        }
        bool assigned = rank < missing;
        uint nextRank = assigned ? rank + push.capacity - missing : rank - missing;
        uint lastRequested = assigned || owner.z != 0u ? push.frameIndex : (free ? 0u : prior.y);
        history[nextRank] = uint2(prior.x, lastRequested);
        history[push.capacity + prior.x] = assigned ? uint2(push.allocationSerialLow, push.allocationSerialHigh) : priorAssignment;
        metadata[ownerBase + prior.x].w = prior.y;
    }
}

// Publication follows every face's Assign/Build and the public table uploads.
// One writer aggregates six point faces without racing a per-light budget.
[shader("compute")]
[numthreads(256, 1, 1)]
void vsmSharedPoolPublishMain(uint lane : SV_GroupIndex, uniform VirtualShadowMapPhysicalPoolPublishPush push)
{
    if (lane != 0u) { return; }
    StructuredBuffer<ShadowPush> jobs = resolveUniform(ResourceHandle<StructuredBuffer<ShadowPush>>(push.jobs));
    StructuredBuffer<uint4> metadata = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.metadata));
    StructuredBuffer<uint> publicationFaces = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.publicationFaces));
    RWByteAddressBuffer lights = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.lights));
    RWStructuredBuffer<uint> budget = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.budgetDiagnostics));
    for (uint face = 0u; face < push.faceCount; ++face) {
        uint entry = publicationFaces[face] / 6u;
        lights.Store(64u + entry * 64u + 24u, 0u);
    }
    uint capacity = metadata[0].y;
    uint available = capacity - push.faceCount;
    uint equalFine = push.faceCount != 0u ? available / push.faceCount : 0u;
    uint equalRemainder = push.faceCount != 0u ? available % push.faceCount : 0u;
    uint allocated = 0u, newFine = 0u, reusedFine = 0u;
    for (uint face = 0u; face < push.faceCount; ++face) {
        ShadowPush job = jobs[face];
        StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(job.statistics));
        uint publication = publicationFaces[face];
        uint grant = metadata[1u + face].x + 1u;
        uint requested = statistics[0];
        uint target = requested;
        uint offset = 8u + publication * 8u;
        budget[offset] = 1u;
        budget[offset + 1u] = requested;
        budget[offset + 2u] = target;
        budget[offset + 3u] = grant;
        budget[offset + 4u] = 1u + min(target, equalFine + uint(face < equalRemainder));
        budget[offset + 5u] = 0u;
        budget[offset + 6u] = statistics[2];
        budget[offset + 7u] = push.frameIndex;
        uint entryBudgetOffset = 64u + (publication / 6u) * 64u + 24u;
        lights.Store(entryBudgetOffset, lights.Load(entryBudgetOffset) + grant);
        allocated += grant;
        newFine += statistics[20];
        reusedFine += statistics[17];
    }
    budget[0] = 2u;
    budget[3] = push.faceCount;
    budget[5] = allocated;
    budget[6] = newFine;
    budget[7] = reusedFine;
}
