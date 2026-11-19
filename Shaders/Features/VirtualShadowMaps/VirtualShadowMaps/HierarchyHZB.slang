#language slang 2026
implementing VirtualShadowMaps;

// Preorder forest from the caster publication. Escape skips this complete
// subtree; an internal node's first child is the immediately following node.
struct VSMCasterHierarchyNode
{
    float3 minimum;
    uint escape;
    float3 maximum;
    uint instance;
};

uint vsmHierarchyRetestArgument(uint batch, ShadowPush push)
{
    return vsmRasterHZBPageArgument(0u, 0u, push) + 2u * vsmRasterHZBPageBatches(push) + batch;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmPrepareHierarchyHZBMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush batchPush)
{
    ShadowPush push;
    if (!vsmBatchComputePush(group, batchPush, push)) { return; }
    uint page = group.x * 64u + lane;
    if (page == 0u) {
        RWStructuredBuffer<uint> args = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
        uint instanceCount = vsmCacheMetadata(push, kVSMCacheInstanceCount);
        for (uint batch = 0u; batch < (instanceCount + 65534u) / 65535u; ++batch) {
            uint word = vsmHierarchyRetestArgument(batch, push) * 3u;
            args[word] = 0u;
            args[word + 1u] = args[word + 2u] = 1u;
        }
    }
    if (page >= push.totalPages - 1u) { return; }
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    uint4 entry = pages[page];
    if (entry.x != 0xffffffffu && vsmRasterClearsEntry(entry, 0u)) {
        vsmQueueRasterHZBPage(page, 0u, push);
    }
}

static const uint kVSMHierarchyNoDirtyPage = 0u;
static const uint kVSMHierarchyOccluded = 1u;
static const uint kVSMHierarchyVisible = 2u;

// Test only fine pages on which Build can emit triangles. Sorted outcomes
// let the group combine levels: a single visible/uncertain page keeps the node.
uint vsmHierarchyLevelVisibility(VSMCasterHierarchyNode node, uint level, bool current, ShadowPush push)
{
    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint base = vsmPageHierarchyOffset(push) + level * vsmPageHierarchyLevelWords(push);
    uint2 dirtyFirst = uint2(owners[base], owners[base + 1u]);
    uint2 dirtyLast = uint2(owners[base + 2u], owners[base + 3u]);
    if (any(dirtyFirst > dirtyLast)) { return kVSMHierarchyNoDirtyPage; }
    if (any(node.minimum > node.maximum)) { return kVSMHierarchyVisible; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    ShadowProjection projection = projections[level];
    float2 x = vsmHZBProjectedRange(node.minimum, node.maximum, projection.rowX);
    float2 y = vsmHZBProjectedRange(node.minimum, node.maximum, projection.rowY);
    float2 z = vsmHZBProjectedRange(node.minimum, node.maximum, projection.rowZ);
    if (!all(isfinite(x)) || !all(isfinite(y)) || !all(isfinite(z))) { return kVSMHierarchyVisible; }
    float2 lo = float2(x.x, y.x), hi = float2(x.y, y.y);
    if (any(hi < -1.0) || any(lo > 1.0)) { return kVSMHierarchyNoDirtyPage; }
    uint axis = projection.page.y;
    uint2 first = max(dirtyFirst, min(uint2(floor(saturate(lo * 0.5 + 0.5) * float(axis))), uint2(axis - 1u)));
    uint2 last = min(dirtyLast, min(uint2(floor(saturate(hi * 0.5 + 0.5) * float(axis))), uint2(axis - 1u)));
    uint outcome = kVSMHierarchyNoDirtyPage;
    uint bitset = vsmClusterPageBitsetOffset(push);
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    for (uint row = first.y; row <= last.y; ++row) {
        for (uint col = first.x; col <= last.x; ++col) {
            uint page = projection.page.x + row * axis + col;
            if ((owners[bitset + (page >> 5u)] & (1u << (page & 31u))) == 0u) { continue; }
            uint4 entry = pages[page];
            // Before Build, the history builder zeros slots without a certified
            // predecessor. Current-depth rejection also needs a valid main draw.
            if (z.x < 0.0 || z.y >= 1.0 || entry.x == 0xffffffffu || (current && (entry.w & 1u) == 0u) ||
                !vsmRasterBoundsOccluded(node.minimum, node.maximum, page, entry.x, projection, push)) {
                return kVSMHierarchyVisible;
            }
            outcome = kVSMHierarchyOccluded;
        }
    }
    return outcome;
}

void vsmDeferHierarchyNode(uint index, ShadowPush push)
{
    RWStructuredBuffer<uint> metadata = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.sampleReserved));
    uint slot;
    InterlockedAdd(metadata[kVSMHierarchyHZBDeferredCount], 1u, slot);
    RWStructuredBuffer<uint> queue = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(metadata[kVSMHierarchyHZBQueue]));
    queue[slot] = index;
    RWStructuredBuffer<uint> args = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    InterlockedAdd(args[vsmHierarchyRetestArgument(slot / 65535u, push) * 3u], 1u);
}

groupshared uint gVsmHierarchyVisibility;

void vsmVisitHZBSubtree(uint root, uint lane, bool current, ShadowPush push)
{
    StructuredBuffer<VSMCasterHierarchyNode> nodes = resolveUniform(ResourceHandle<StructuredBuffer<VSMCasterHierarchyNode>>(
        vsmCacheMetadata(push, kVSMHierarchyHZBNodes)));
    StructuredBuffer<VSMInstanceRange> instances = resolveUniform(ResourceHandle<StructuredBuffer<VSMInstanceRange>>(
        vsmCacheMetadata(push, kVSMCacheInstanceTable)));
    uint end = nodes[root].escape;
    for (uint index = root; index < end;) {
        VSMCasterHierarchyNode node = nodes[index];
        if (lane == 0u) { gVsmHierarchyVisibility = kVSMHierarchyNoDirtyPage; }
        GroupMemoryBarrierWithGroupSync();
        if (lane < push.clipmapCount) {
            InterlockedMax(gVsmHierarchyVisibility, vsmHierarchyLevelVisibility(node, lane, current, push));
        }
        GroupMemoryBarrierWithGroupSync();
        uint outcome = gVsmHierarchyVisibility;
        // A fast lane must not reset the result for the next node before all
        // waves have consumed this one.
        GroupMemoryBarrierWithGroupSync();
        bool leaf = node.instance != 0xffffffffu;
        if (lane == 0u) {
            if (!current && outcome == kVSMHierarchyOccluded) {
                vsmDeferHierarchyNode(index, push);
#if VSM_WORKLOAD_DIAGNOSTICS
                vsmCacheAdd(push, leaf ? kVSMHierarchyHZBDeferredInstances : kVSMHierarchyHZBDeferredNodes, 1u);
#endif
            }
#if VSM_WORKLOAD_DIAGNOSTICS
            if (current && index == root) {
                uint counter = outcome == kVSMHierarchyOccluded ?
                    (leaf ? kVSMHierarchyHZBRejectedInstances : kVSMHierarchyHZBRejectedNodes) :
                    (leaf ? kVSMHierarchyHZBRecoveredInstances : kVSMHierarchyHZBRecoveredNodes);
                vsmCacheAdd(push, counter, 1u);
            }
            if (leaf) {
                vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticInstanceTests, 1u);
                vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticInstanceRejects, uint(outcome != kVSMHierarchyVisible));
            }
#endif
        }
        if (outcome == kVSMHierarchyVisible && leaf) {
            VSMInstanceRange instance = instances[node.instance];
            if (instance.count != 0u) { vsmEmitInstanceRange(instance.first, instance.first + instance.count, lane, push); }
        }
        index = outcome == kVSMHierarchyVisible && !leaf ? index + 1u : node.escape;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmResetHierarchyPostMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush batchPush)
{
    ShadowPush push;
    if (!vsmBatchComputePush(group, batchPush, push)) { return; }
    uint batch = group.x * 64u + lane;
    if (batch == 0u) {
        StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
        RWStructuredBuffer<uint> metadata = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.sampleReserved));
        metadata[kVSMHierarchyHZBMainTickets] = work[0].x;
        RWStructuredBuffer<uint> candidates = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
        candidates[vsmCompactCandidateOffset(push)] = 0u;
    }
    if (batch < vsmCompactBatchCount(push)) {
        RWStructuredBuffer<uint> args = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
        args[(push.batchCount + push.coarseBatchCount + batch) * 3u] = 0u;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmRetestHierarchyHZBMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush batchPush)
{
    ShadowPush push;
    if (!vsmBatchComputePush(group, batchPush, push)) { return; }
    StructuredBuffer<uint> queue = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(vsmCacheMetadata(push, kVSMHierarchyHZBQueue)));
    vsmVisitHZBSubtree(queue[push.workBase + group.x], lane, true, push);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmFinalizeHierarchyPostMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush batchPush)
{
    ShadowPush push;
    if (!vsmBatchComputePush(group, batchPush, push)) { return; }
    uint index = group.x * 64u + lane;
    RWStructuredBuffer<uint> stats = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    if (index == 0u) { stats[4] = min(stats[3], push.workCapacity); }
    if (index < push.batchCount) {
        StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
        uint delta = work[0].x - vsmCacheMetadata(push, kVSMHierarchyHZBMainTickets);
        uint first = index * 65535u;
        RWStructuredBuffer<uint> args = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
        args[index * 3u] = first < delta ? min(delta - first, 65535u) : 0u;
    }
    if (index < push.totalPages - 1u && stats[5] != 0u) {
        RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
        uint4 entry = pages[index];
        if (vsmRasterClearsEntry(entry, 0u) && (entry.w & 1u) != 0u) {
            pages[index].w = entry.w & ~1u;
            InterlockedAdd(stats[24], 0xffffffffu);
        }
    }
}
