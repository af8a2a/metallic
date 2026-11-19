#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_DIRECTIONAL_BUILD
#define VSM_DIRECTIONAL_BUILD 0
#endif
#ifndef VSM_COMPACT_BUILD
#define VSM_COMPACT_BUILD 0
#endif
#ifndef VSM_INSTANCE_BUILD
#define VSM_INSTANCE_BUILD 0
#endif
#ifndef VSM_DIRTY_PAGE_HIERARCHY
#define VSM_DIRTY_PAGE_HIERARCHY 0
#endif
#ifndef VSM_BATCHED_BUILD_RASTER
#define VSM_BATCHED_BUILD_RASTER 0
#endif


groupshared float3 g_vsmClusterMinimum, g_vsmClusterMaximum;
groupshared uint g_vsmClusterValid, g_vsmClusterLevelMask;

public void vsmBuildPageHierarchy(uint index, ShadowPush push)
{
#if VSM_DIRTY_PAGE_HIERARCHY
    uint bitsetWords = (push.totalPages + 31u) / 32u;
    if (index >= bitsetWords) { return; }
    StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.statistics));
    if (statistics[16] == 0u && (push.cacheMode & 32u) == 0u) { return; }
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    uint bits = owners[vsmClusterPageBitsetOffset(push) + index];
    uint pageStride = push.virtualPagesPerAxis * push.virtualPagesPerAxis;
    uint paddedAxis = vsmPageHierarchyAxis(push);
    while (bits != 0u) {
        uint page = index * 32u + uint(firstbitlow(bits));
        bits &= bits - 1u;
        // Admit/Assign publish only resident, admitted, dirty fine-page bits.
        uint level = page / pageStride;
        uint axis = push.lightType == 0u ? push.virtualPagesPerAxis : max(1u, push.virtualPagesPerAxis >> level);
        uint localPage = page % pageStride;
        uint2 xy = uint2(localPage % axis, localPage / axis);
        uint base = vsmPageHierarchyOffset(push) + level * vsmPageHierarchyLevelWords(push);
        InterlockedMin(owners[base], xy.x);
        InterlockedMin(owners[base + 1u], xy.y);
        InterlockedMax(owners[base + 2u], xy.x);
        InterlockedMax(owners[base + 3u], xy.y);
        uint mipOffset = base + 4u;
        for (uint mipAxis = paddedAxis >> 1u; mipAxis != 0u; mipAxis >>= 1u) {
            xy >>= 1u;
            uint previous;
            InterlockedOr(owners[mipOffset + xy.y * mipAxis + xy.x], 1u, previous);
            // The first writer propagates to the root. The dispatch publication
            // barrier completes that writer before any cluster consumes nodes.
            if (previous != 0u) { break; }
            mipOffset += mipAxis * mipAxis;
        }
    }
#endif
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmBuildPageHierarchyMain(uint3 id : SV_DispatchThreadID,
#if VSM_BATCHED_BUILD_RASTER
    uniform ShadowPush batchPush)
#else
    uniform ShadowPush push)
#endif
{
#if VSM_BATCHED_BUILD_RASTER
    uint3 group = uint3(id.x / 64u, id.y, 0u);
    ShadowPush push;
    if (!vsmBatchComputePush(group, batchPush, push)) { return; }
    id = uint3(group.x * 64u + id.x % 64u, 0u, 0u);
#endif
    vsmBuildPageHierarchy(id.x, push);
}

#if VSM_DIRTY_PAGE_HIERARCHY
public bool vsmHierarchyOverlaps(int2 first, int2 last, ShadowProjection projection, ShadowPush push)
{
    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint base = vsmPageHierarchyOffset(push) + projection.page.z * vsmPageHierarchyLevelWords(push);
    uint2 low = uint2(owners[base], owners[base + 1u]);
    uint2 high = uint2(owners[base + 2u], owners[base + 3u]);
    bool empty = any(low > high) || any(uint2(last) < low) || any(uint2(first) > high);
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterLevelTests, 1u);
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterHierarchyLevelTests, 1u);
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterHierarchyBoundsRejects, empty ? 1u : 0u);
#endif
    if (empty) {
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterEmptyLevels, 1u);
#endif
        return false;
    }
    uint2 clippedFirst = max(uint2(first), low), clippedLast = min(uint2(last), high);
    uint2 span = clippedLast - clippedFirst;
    uint maxSpan = max(span.x, span.y);
    // Inclusive extent <= 2^mip intersects at most two aligned cells per axis.
    uint mip = maxSpan == 0u ? 0u : uint(firstbithigh(maxSpan)) + 1u;
    uint nodeTests = 0u;
    uint flags = 0u;
    if (mip == 0u) {
        uint page = projection.page.x + clippedFirst.y * projection.page.y + clippedFirst.x;
        flags = owners[vsmClusterPageBitsetOffset(push) + (page >> 5u)] & (1u << (page & 31u));
        nodeTests = 1u;
    } else {
        uint paddedAxis = vsmPageHierarchyAxis(push);
        uint axis = paddedAxis >> mip;
        uint offset = base + 4u + vsmPageHierarchyMipOffset(paddedAxis, mip);
        uint2 start = clippedFirst >> mip, end = clippedLast >> mip;
        for (uint y = start.y; y <= end.y; ++y) {
            for (uint x = start.x; x <= end.x; ++x) {
                flags |= owners[offset + y * axis + x];
                ++nodeTests;
            }
        }
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterHierarchyNodeTests, nodeTests);
    vsmBuildDiagnosticAdd(push, flags != 0u ? kVSMBuildDiagnosticClusterPageHits : kVSMBuildDiagnosticClusterEmptyLevels, 1u);
#endif
    return flags != 0u;
}
#endif

#if VSM_STATIC_PAGE_HZB
bool vsmClusterVisibleInStaticHZB(int2 first, int2 last, float3 minimum, float3 maximum,
    ShadowProjection projection, StructuredBuffer<uint4> pages, ShadowPush push, bool afterStaticRaster)
{
    // Query only on a broad-phase hit. Bound the extra work on large clusters;
    // keeping a level falls back to the original per-triangle page enumeration.
    if (any(last - first >= 4)) { return true; }
    bool tested = false;
    for (int y = first.y; y <= last.y; ++y) {
        for (int x = first.x; x <= last.x; ++x) {
            uint2 xy = uint2(x, y);
            uint4 entry = pages[projection.page.x + xy.y * projection.page.y + xy.x];
            if ((entry.y & 2u) == 0u || (entry.w & (afterStaticRaster ? 2u : 10u)) == 0u) { continue; }
            // Build can use only retained static depth. After static Raster and
            // its HZB update, newly rendered complete pages also certify occlusion.
            if ((entry.w & (afterStaticRaster ? 4u : 12u)) != 4u) { return true; }
#if VSM_WORKLOAD_DIAGNOSTICS
            if (afterStaticRaster) {
                if (!tested) { vsmWorkAdd(push, kVSMWorkRasterHZBClusterLevelTests, 1u); }
            } else { vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticHZBPageTests, 1u); }
#endif
            tested = true;
            if (!vsmStaticPageHidesBounds(push, projection, xy, entry.x, minimum, maximum)) { return true; }
        }
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    if (afterStaticRaster) { vsmWorkAdd(push, kVSMWorkRasterHZBRejectedLevels, uint(tested)); }
    else { vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticHZBLevelRejects, uint(tested)); }
#endif
    // A positive HMip node may contain no exact dirty pages in this rectangle.
    // Keep its original decision unless we actually proved occlusion.
    return !tested;
}

#if VSM_CLUSTER_HZB
[shader("compute")]
[numthreads(64, 1, 1)]
void vsmCullDynamicClustersHZBMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
#if VSM_BATCHED_BUILD_RASTER
    uniform ShadowPush batchPush)
#else
    uniform ShadowPush push)
#endif
{
#if VSM_BATCHED_BUILD_RASTER
    ShadowPush push;
    if (!vsmBatchComputePush(group, batchPush, push)) { return; }
#endif
    uint cluster = push.workBase + group.x;
    uint firstTriangle = cluster * 64u;
    if (lane == 0u) {
        uint dynamicBits = loadVirtualShadowMapDynamicMaskWord(push.geometry, cluster * 2u);
        if (push.triangleCount - firstTriangle > 32u) {
            dynamicBits |= loadVirtualShadowMapDynamicMaskWord(push.geometry, cluster * 2u + 1u);
        }
        g_vsmClusterLevelMask = 0xffffffffu;
        g_vsmClusterValid = 0u;
        if (dynamicBits != 0u) {
            float3 minimum, maximum;
            g_vsmClusterValid = uint(loadVirtualShadowMapClusterBounds(push.geometry, firstTriangle, minimum, maximum));
            g_vsmClusterMinimum = minimum;
            g_vsmClusterMaximum = maximum;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_vsmClusterValid != 0u && lane <= push.clipmapCount && vsmLODSelectedRange(push, firstTriangle, lane)) {
        StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
        ShadowProjection projection = projections[lane];
        float3 minimum = g_vsmClusterMinimum, maximum = g_vsmClusterMaximum;
        // Enumerate a superset of the pages Build can emit for any enclosed
        // triangle. The coarse level always covers one physical page.
        float2 x = vsmHZBProjectedRange(minimum, maximum, projection.rowX);
        float2 y = vsmHZBProjectedRange(minimum, maximum, projection.rowY);
        float2 lo = float2(x.x, y.x), hi = float2(x.y, y.y);
        if (all(isfinite(lo)) && all(isfinite(hi)) && all(lo <= 1.0) && all(hi >= -1.0)) {
            int axis = int(projection.page.y);
            int2 first = min(int2(floor(saturate(lo * 0.5 + 0.5) * float(axis))), int2(axis - 1));
            int2 last = min(int2(floor(saturate(hi * 0.5 + 0.5) * float(axis))), int2(axis - 1));
            if (!vsmClusterVisibleInStaticHZB(first, last, minimum, maximum, projection, pages, push, true)) {
                InterlockedAnd(g_vsmClusterLevelMask, ~(1u << lane));
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        RWStructuredBuffer<uint> visibility = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, 40u)));
        visibility[cluster] = g_vsmClusterLevelMask;
    }
}
#endif
#endif

bool vsmClusterNeedsLevel(float3 minimum, float3 maximum, ShadowProjection projection,
    StructuredBuffer<uint4> pages, ShadowPush push)
{
    float2 lo = float2(1e30), hi = -lo;
    for (uint corner = 0u; corner < 8u; ++corner) {
        float3 position = float3((corner & 1u) != 0u ? maximum.x : minimum.x,
            (corner & 2u) != 0u ? maximum.y : minimum.y,
            (corner & 4u) != 0u ? maximum.z : minimum.z);
        float4 clip = vsmProjectClip(position, projection);
#if !VSM_DIRECTIONAL_BUILD
        // A box crossing the perspective near plane cannot be bounded by
        // dividing its corners. Retain this level for the exact triangle clip.
        if (projection.metric.w > 0.5 && clip.w <= projection.originNear.w) {
#if VSM_WORKLOAD_DIAGNOSTICS
            vsmRecordClusterLevel(push, 0u, kVSMBuildDiagnosticClusterNearPlaneKeeps);
#endif
            return true;
        }
        float2 projected = projection.metric.w > 0.5 ? clip.xy / clip.w : clip.xy;
#else
        float2 projected = clip.xy;
#endif
        lo = min(lo, projected); hi = max(hi, projected);
    }
    int axis = int(projection.page.y);
    // Expand independently of the world-space bound to cover projected float
    // rounding and floor-to-page boundaries. Do not cull on projected depth.
    float guard = 2.0 / float(axis);
    if (any(hi < -1.0 - guard) || any(lo > 1.0 + guard)) {
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmRecordClusterLevel(push, 0u, kVSMBuildDiagnosticClusterOutsideLevels);
#endif
        return false;
    }
    int2 first = clamp(int2(floor(saturate(lo * 0.5 + 0.5) * float(axis))) - 1, int2(0), int2(axis - 1));
    int2 last = clamp(int2(floor(saturate(hi * 0.5 + 0.5) * float(axis))) + 1, int2(0), int2(axis - 1));
#if VSM_DIRTY_PAGE_HIERARCHY
#if VSM_STATIC_PAGE_HZB
    if (!vsmHierarchyOverlaps(first, last, projection, push)) { return false; }
    return vsmClusterVisibleInStaticHZB(first, last, minimum, maximum, projection, pages, push, false);
#else
    return vsmHierarchyOverlaps(first, last, projection, push);
#endif
#else
    StructuredBuffer<uint> bitset = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint offset = vsmClusterPageBitsetOffset(push);
    uint cachedWord = 0xffffffffu, rawWord = 0u;
#if VSM_WORKLOAD_DIAGNOSTICS
    uint wordTests = 0u;
#endif
    for (int y = first.y; y <= last.y; ++y) {
        uint firstPage = projection.page.x + uint(y * axis + first.x);
        uint lastPage = projection.page.x + uint(y * axis + last.x);
        uint firstWord = firstPage >> 5u, lastWord = lastPage >> 5u;
        for (uint word = firstWord; word <= lastWord; ++word) {
            // Row ranges advance monotonically. Keep the unmasked word when
            // an unaligned next row shares it (for example a 127-page axis).
            if (word != cachedWord) {
                rawWord = bitset[offset + word];
                cachedWord = word;
#if VSM_WORKLOAD_DIAGNOSTICS
                ++wordTests;
#endif
            }
            uint bits = rawWord;
            if (word == firstWord) { bits &= 0xffffffffu << (firstPage & 31u); }
            if (word == lastWord) { bits &= 0xffffffffu >> (31u - (lastPage & 31u)); }
            if (bits != 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                vsmRecordClusterLevel(push, wordTests, kVSMBuildDiagnosticClusterPageHits);
#endif
#if VSM_STATIC_PAGE_HZB
                return vsmClusterVisibleInStaticHZB(first, last, minimum, maximum, projection, pages, push, false);
#else
                return true;
#endif
            }
        }
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmRecordClusterLevel(push, wordTests, kVSMBuildDiagnosticClusterEmptyLevels);
#endif
    return false;
#endif
}

#if VSM_COMPACT_BUILD
uint vsmCullCluster(uint firstTriangle, uint lane, ShadowPush push,
    StructuredBuffer<ShadowProjection> projections, StructuredBuffer<uint4> pages)
{
    if (lane == 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterGroups, 1u);
#endif
        g_vsmClusterLevelMask = 0u;
        float3 minimum, maximum;
        g_vsmClusterValid = loadVirtualShadowMapClusterBounds(push.geometry, firstTriangle,
            minimum, maximum) ? 1u : 0u;
        g_vsmClusterMinimum = minimum;
        g_vsmClusterMaximum = maximum;
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterBoundsValidGroups, g_vsmClusterValid);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    uint clusterLevel = vsmFineViewForLane(push, lane);
#if VSM_WORKLOAD_DIAGNOSTICS
    // Keep the predicate's diagnostic side effects inside the lane/valid guard.
    if (clusterLevel < push.clipmapCount && g_vsmClusterValid != 0u && vsmLODSelectedRange(push, firstTriangle, clusterLevel)) {
        if (vsmClusterNeedsLevel(g_vsmClusterMinimum, g_vsmClusterMaximum, projections[clusterLevel], pages, push)) {
            InterlockedOr(g_vsmClusterLevelMask, 1u << clusterLevel);
        }
    }
#else
    if (clusterLevel < push.clipmapCount && g_vsmClusterValid != 0u && vsmLODSelectedRange(push, firstTriangle, clusterLevel) &&
        vsmClusterNeedsLevel(g_vsmClusterMinimum, g_vsmClusterMaximum, projections[clusterLevel], pages, push)) {
        InterlockedOr(g_vsmClusterLevelMask, 1u << clusterLevel);
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    uint levelMask = g_vsmClusterLevelMask;
#if VSM_WORKLOAD_DIAGNOSTICS
    if (lane == 0u && levelMask != 0u) { vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterSurvivingGroups, 1u); }
#endif
    return levelMask;
}
#endif

#if VSM_COMPACT_BUILD
#if VSM_INSTANCE_BUILD
struct VSMInstanceRange
{
    float3 minimum;
    uint first;
    float3 maximum;
    uint count;
};

bool vsmInstanceNeedsLevel(VSMInstanceRange instance, ShadowProjection projection, ShadowPush push)
{
    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint base = vsmPageHierarchyOffset(push) + projection.page.z * vsmPageHierarchyLevelWords(push);
    uint2 low = uint2(owners[base], owners[base + 1u]);
    uint2 high = uint2(owners[base + 2u], owners[base + 3u]);
    if (any(low > high)) { return false; }
    if (any(instance.minimum > instance.maximum)) { return true; }
    // The instance path is compiled only for orthographic directional lights.
    // Use the same projection order and one-page rounding guard as cluster culling.
    float2 lo = float2(1e30), hi = -lo;
    for (uint corner = 0u; corner < 8u; ++corner) {
        float3 position = float3((corner & 1u) ? instance.maximum.x : instance.minimum.x,
            (corner & 2u) ? instance.maximum.y : instance.minimum.y,
            (corner & 4u) ? instance.maximum.z : instance.minimum.z);
        float2 xy = vsmProjectClip(position, projection).xy;
        lo = min(lo, xy); hi = max(hi, xy);
    }
    float axis = float(projection.page.y);
    return all((hi * 0.5 + 0.5) * axis >= float2(low) - 1.0) &&
        all((lo * 0.5 + 0.5) * axis <= float2(high) + 2.0);
}

groupshared uint g_vsmInstanceTicketBase;

void vsmEmitInstanceRange(uint first, uint end, uint lane, ShadowPush push)
{
    uint clusterFirst = first >> 6u;
    uint count = ((end - 1u) >> 6u) - clusterFirst + 1u;
    RWStructuredBuffer<uint> candidates = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    uint offset = vsmCompactCandidateOffset(push);
    if (lane == 0u) {
        uint slot;
        InterlockedAdd(candidates[offset], count, slot);
        g_vsmInstanceTicketBase = slot;
        RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
        // Contiguous reservations have no holes; publish the high-water mark in each indirect bin.
        uint stop = slot + count;
        for (uint i = slot; i < stop;) {
            uint bin = i / 65535u;
            uint remaining = min(stop - i, 65535u - i % 65535u);
            InterlockedMax(arguments[(push.batchCount + push.coarseBatchCount + bin) * 3u], i % 65535u + remaining);
            i += remaining;
        }
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticInstanceClusterTickets, count);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint j = lane; j < count; j += 64u) {
        uint aligned = (clusterFirst + j) * 64u;
        uint ticket = offset + 1u + (g_vsmInstanceTicketBase + j) * 2u;
        candidates[ticket] = max(aligned, first);
        candidates[ticket + 1u] = aligned + min(end - aligned, 64u);
    }
    // A hierarchy group may emit several consecutive instances using this reservation.
#if VSM_HIERARCHY_HZB
    GroupMemoryBarrierWithGroupSync();
#endif
}
#endif
[shader("compute")]
[numthreads(64, 1, 1)]
void vsmCompactMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
#if VSM_BATCHED_BUILD_RASTER
    uniform ShadowPush batchPush)
#else
    uniform ShadowPush push)
#endif
{
#if VSM_BATCHED_BUILD_RASTER
    ShadowPush push;
    if (!vsmBatchComputePush(group, batchPush, push)) { return; }
#endif
#if VSM_INSTANCE_BUILD
    StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.statistics));
    uint input = push.workBase + group.x;
#if VSM_HIERARCHY_HZB
    if ((push.cacheMode & 1u) != 0u) {
        if (statistics[16] != 0u) { vsmVisitHZBSubtree(input * kVSMHierarchyHZBTreeStride, lane, false, push); }
        return;
    }
#endif
    uint first, end;
    if (statistics[16] == 0u) {
        if ((push.cacheMode & 32u) == 0u || input >= vsmCacheMetadata(push, kVSMCacheMergeClusterCount)) { return; }
        first = vsmMergeCluster(push, input).x;
        end = first + min(push.triangleCount - first, 64u);
    } else {
        if (input >= vsmCacheMetadata(push, kVSMCacheInstanceCount)) { return; }
        StructuredBuffer<VSMInstanceRange> instances = resolveUniform(ResourceHandle<StructuredBuffer<VSMInstanceRange>>(
            vsmCacheMetadata(push, kVSMCacheInstanceTable)));
        VSMInstanceRange instance = instances[input];
        if (lane == 0u) { g_vsmClusterLevelMask = 0u; }
        GroupMemoryBarrierWithGroupSync();
        StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        uint level = vsmFineViewForLane(push, lane);
        if (instance.count != 0u && level < push.clipmapCount && vsmInstanceNeedsLevel(instance, projections[level], push)) {
            InterlockedOr(g_vsmClusterLevelMask, 1u << level);
        }
        GroupMemoryBarrierWithGroupSync();
#if VSM_WORKLOAD_DIAGNOSTICS
        if (lane == 0u) {
            vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticInstanceTests, 1u);
            vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticInstanceRejects, uint(g_vsmClusterLevelMask == 0u));
        }
#endif
        if (g_vsmClusterLevelMask == 0u) { return; }
        first = instance.first;
        end = first + instance.count;
    }
    vsmEmitInstanceRange(first, end, lane, push);
#else
    // The source ranges are aligned to 64 triangles. Rounded tail lanes still
    // participate in the identical group-cooperative cluster predicate.
    uint firstTriangle = push.workBase + group.x * 64u;
    if (firstTriangle >= push.triangleCount || push.enabled == 0u) { return; }
    StructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.statistics));
    if (statistics[16] == 0u && (push.cacheMode & 32u) == 0u) { return; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    if (statistics[16] == 0u) {
        uint cluster = firstTriangle >> 6u;
        if (cluster >= vsmCacheMetadata(push, kVSMCacheMergeClusterCount)) { return; }
        firstTriangle = vsmMergeCluster(push, cluster).x;
    }
    uint levelMask = vsmCullCluster(firstTriangle, lane, push, projections, pages);
    if (lane == 0u && levelMask != 0u) {
        RWStructuredBuffer<uint> candidates = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
        uint offset = vsmCompactCandidateOffset(push);
        uint slot;
        InterlockedAdd(candidates[offset], 1u, slot);
        // At most one candidate per source group: the full source-cluster
        // capacity bounds this append and every indirect bin to 65535 groups.
        candidates[offset + 1u + slot * 2u] = firstTriangle;
        candidates[offset + 2u + slot * 2u] = levelMask;
        RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
        uint argument = (push.batchCount + push.coarseBatchCount + slot / 65535u) * 3u;
        InterlockedAdd(arguments[argument], 1u);
    }
#endif
}
#endif

