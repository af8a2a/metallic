#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_RECEIVER_MASK
#define VSM_RECEIVER_MASK 0
#endif
#ifndef VSM_DIRECTIONAL_BUILD
#define VSM_DIRECTIONAL_BUILD 0
#endif
#ifndef VSM_COMPACT_BUILD
#define VSM_COMPACT_BUILD 0
#endif
#ifndef VSM_PAGE_WINDOW_RASTER
#define VSM_PAGE_WINDOW_RASTER 0
#endif
#ifndef VSM_CLUSTER_PAGE_RASTER
#define VSM_CLUSTER_PAGE_RASTER 0
#endif
#ifndef VSM_PACKED_RASTER_WORK
#define VSM_PACKED_RASTER_WORK 0
#endif
#ifndef VSM_INSTANCE_BUILD
#define VSM_INSTANCE_BUILD 0
#endif
#ifndef VSM_BATCHED_BUILD_RASTER
#define VSM_BATCHED_BUILD_RASTER 0
#endif
#ifndef VSM_PACKED_COARSE_RASTER
#define VSM_PACKED_COARSE_RASTER 0
#endif

#if VSM_PACKED_COARSE_RASTER
[shader("compute")]
[numthreads(64, 1, 1)]
void vsmBuildCoarseWorkMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush batchPush)
{
    ShadowPush push;
    if (!vsmBatchComputePush(group, batchPush, push)) { return; }
    const uint firstTriangle = (group.x * 64u + lane) * kVSMCoarseRasterSlotsPerTicket;
    const uint coarseView = push.clipmapCount;
    float3 minimum, maximum;
    const bool drawable = firstTriangle < push.triangleCount &&
        vsmLODSelectedRange(push, firstTriangle, coarseView, kVSMCoarseRasterSlotsPerTicket) &&
        loadVirtualShadowMapClusterBounds(push.geometry, firstTriangle, minimum, maximum);
    const uint prefix = WavePrefixCountBits(drawable);
    const uint count = WaveActiveCountBits(drawable);
    if (count == 0u) { return; }

    RWStructuredBuffer<uint> queue = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(
        vsmCacheMetadata(push, kVSMCoarseRasterQueue)));
    uint first = 0u;
    // The queue has one slot for each possible 16-triangle range. Each lane
    // owns one range and can emit at most one ticket.
    if (WaveIsFirstLane()) { InterlockedAdd(queue[0], count, first); }
    first = WaveReadLaneFirst(first);
    if (drawable) { queue[1u + first + prefix] = firstTriangle; }
}
#endif

#if VSM_CLUSTER_PAGE_RASTER
static const uint kVSMClusterPageTableSize = 512u;
static const uint kVSMClusterPageTableMask = kVSMClusterPageTableSize - 1u;
static const uint kVSMClusterPageTablePasses = kVSMClusterPageTableSize / 64u;
groupshared Atomic<uint> g_vsmClusterPageKey[kVSMClusterPageTableSize];
groupshared uint g_vsmClusterPageMaskLo[kVSMClusterPageTableSize];
groupshared uint g_vsmClusterPageMaskHi[kVSMClusterPageTableSize];
groupshared uint g_vsmClusterUsedSlots[kVSMClusterPageTableSize];
groupshared uint g_vsmClusterUsedCount[2];
groupshared uint g_vsmClusterTotalTris[2];
groupshared uint g_vsmClusterTotalTickets[2];
#if !VSM_PACKED_RASTER_WORK
groupshared uint g_vsmClusterBaseTri;
#endif
groupshared uint g_vsmClusterBaseTicket;

bool vsmTryInsertClusterPage(uint page, uint lane, uint parity)
{
    uint slot = (page * 2654435761u) & kVSMClusterPageTableMask;
    uint bit = 1u << (lane & 31u);
    bool high = lane >= 32u;
    for (uint probe = 0u; probe < 16u; ++probe) {
        uint idx = (slot + probe) & kVSMClusterPageTableMask;
        uint cur = g_vsmClusterPageKey[idx].load(MemoryOrder.Relaxed);
        if (cur != page) {
            if (cur == 0xffffffffu) {
                uint prev = g_vsmClusterPageKey[idx].compareExchange(0xffffffffu, page,
                    MemoryOrder.Relaxed, MemoryOrder.Relaxed);
                if (prev == 0xffffffffu) {
                    uint pos;
                    InterlockedAdd(g_vsmClusterUsedCount[parity], 1u, pos);
                    g_vsmClusterUsedSlots[pos] = idx;
                    cur = page;
                } else {
                    cur = prev;
                }
            }
            if (cur != page) { continue; }
        }
        if (high) { InterlockedOr(g_vsmClusterPageMaskHi[idx], bit); }
        else { InterlockedOr(g_vsmClusterPageMaskLo[idx], bit); }
        return true;
    }
    return false;
}

void vsmEmitSingleClusterPage(uint page, uint lane, uint firstTriangle, ShadowPush push)
{
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint4> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.workItems));
    uint ticket;
#if VSM_PACKED_RASTER_WORK
    InterlockedAdd(statistics[3], 1u);
    InterlockedAdd(work[0].x, 1u, ticket);
    if (ticket < push.workCapacity) {
#else
    uint first;
    InterlockedAdd(statistics[3], 1u, first);
    if (first < push.workCapacity) {
        InterlockedAdd(work[0].x, 1u, ticket);
#endif
        uint bit = 1u << (lane & 31u);
        bool high = lane >= 32u;
        work[1u + ticket] = uint4(firstTriangle, page, high ? 0u : bit, high ? bit : 0u);
    } else {
        InterlockedAdd(statistics[5], 1u);
    }
}

bool vsmFlushClusterPages(uint firstTriangle, uint lane, uint parity, bool moreLevels, ShadowPush push)
{
    uint usedCount = g_vsmClusterUsedCount[parity];
    if (usedCount == 0u) { return false; }
    uint myTris = 0u;
    uint myTickets = 0u;
    for (uint pos = lane; pos < usedCount; pos += 64u) {
        uint s = g_vsmClusterUsedSlots[pos];
        uint lo = g_vsmClusterPageMaskLo[s];
        uint hi = g_vsmClusterPageMaskHi[s];
        myTris += countbits(lo) + countbits(hi);
        myTickets += uint((lo & 0xffffu) != 0u) + uint((lo >> 16u) != 0u) +
            uint((hi & 0xffffu) != 0u) + uint((hi >> 16u) != 0u);
    }
#if !VSM_PACKED_RASTER_WORK
    uint myTriOffset = 0u;
#endif
    uint myTicketOffset = 0u;
    if (myTris != 0u) {
#if VSM_PACKED_RASTER_WORK
        InterlockedAdd(g_vsmClusterTotalTris[parity], myTris);
#else
        InterlockedAdd(g_vsmClusterTotalTris[parity], myTris, myTriOffset);
#endif
        InterlockedAdd(g_vsmClusterTotalTickets[parity], myTickets, myTicketOffset);
    }
    GroupMemoryBarrierWithGroupSync();
    uint totalTris = g_vsmClusterTotalTris[parity];
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint4> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.workItems));
    uint capacity = push.workCapacity;
    if (lane == 0u) {
        uint nextParity = 1u - parity;
        g_vsmClusterUsedCount[nextParity] = 0u;
        g_vsmClusterTotalTris[nextParity] = 0u;
        g_vsmClusterTotalTickets[nextParity] = 0u;
        if (totalTris != 0u) {
#if VSM_PACKED_RASTER_WORK
            InterlockedAdd(statistics[3], totalTris);
            uint baseTicket;
            InterlockedAdd(work[0].x, g_vsmClusterTotalTickets[parity], baseTicket);
            g_vsmClusterBaseTicket = baseTicket;
#else
            uint baseTri;
            InterlockedAdd(statistics[3], totalTris, baseTri);
            g_vsmClusterBaseTri = baseTri;
            if (baseTri < capacity && totalTris <= capacity - baseTri) {
                uint baseTicket;
                InterlockedAdd(work[0].x, g_vsmClusterTotalTickets[parity], baseTicket);
                g_vsmClusterBaseTicket = baseTicket;
            }
#endif
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < usedCount) {
#if VSM_PACKED_RASTER_WORK
        // Reservations count complete masked records. Keep the public counters
        // in logical triangle/page units, including masks rejected by the queue.
        uint ticket = g_vsmClusterBaseTicket + myTicketOffset;
        uint droppedTris = 0u;
        for (uint pos = lane; pos < usedCount; pos += 64u) {
            uint s = g_vsmClusterUsedSlots[pos];
            uint page = g_vsmClusterPageKey[s].load(MemoryOrder.Relaxed);
            uint lo = g_vsmClusterPageMaskLo[s];
            uint hi = g_vsmClusterPageMaskHi[s];
            g_vsmClusterPageKey[s].store(0xffffffffu, MemoryOrder.Relaxed);
            g_vsmClusterPageMaskLo[s] = 0u;
            g_vsmClusterPageMaskHi[s] = 0u;
            [unroll]
            for (uint q = 0u; q < 4u; ++q) {
                uint mask = q < 2u ? lo : hi;
                mask &= (q & 1u) != 0u ? 0xffff0000u : 0xffffu;
                if (mask == 0u) { continue; }
                if (ticket < capacity) {
                    work[1u + ticket] = uint4(firstTriangle, page, q < 2u ? mask : 0u, q >= 2u ? mask : 0u);
                } else {
                    droppedTris += countbits(mask);
                }
                ++ticket;
            }
        }
        if (droppedTris != 0u) { InterlockedAdd(statistics[5], droppedTris); }
#else
        uint baseTri = g_vsmClusterBaseTri;
        if (baseTri < capacity && totalTris <= capacity - baseTri) {
            uint cursor = 1u + g_vsmClusterBaseTicket + myTicketOffset;
            for (uint pos = lane; pos < usedCount; pos += 64u) {
                uint s = g_vsmClusterUsedSlots[pos];
                uint p = g_vsmClusterPageKey[s].load(MemoryOrder.Relaxed);
                uint lo = g_vsmClusterPageMaskLo[s];
                uint hi = g_vsmClusterPageMaskHi[s];
                g_vsmClusterPageKey[s].store(0xffffffffu, MemoryOrder.Relaxed);
                g_vsmClusterPageMaskLo[s] = 0u;
                g_vsmClusterPageMaskHi[s] = 0u;
                uint q0 = lo & 0xffffu;
                uint q1 = lo & 0xffff0000u;
                uint q2 = hi & 0xffffu;
                uint q3 = hi & 0xffff0000u;
                if (q0 != 0u) { work[cursor++] = uint4(firstTriangle, p, q0, 0u); }
                if (q1 != 0u) { work[cursor++] = uint4(firstTriangle, p, q1, 0u); }
                if (q2 != 0u) { work[cursor++] = uint4(firstTriangle, p, 0u, q2); }
                if (q3 != 0u) { work[cursor++] = uint4(firstTriangle, p, 0u, q3); }
            }
        } else {
            uint triCursor = baseTri + myTriOffset;
            for (uint pos = lane; pos < usedCount; pos += 64u) {
                uint s = g_vsmClusterUsedSlots[pos];
                uint p = g_vsmClusterPageKey[s].load(MemoryOrder.Relaxed);
                uint lo = g_vsmClusterPageMaskLo[s];
                uint hi = g_vsmClusterPageMaskHi[s];
                g_vsmClusterPageKey[s].store(0xffffffffu, MemoryOrder.Relaxed);
                g_vsmClusterPageMaskLo[s] = 0u;
                g_vsmClusterPageMaskHi[s] = 0u;
                for (uint q = 0u; q < 4u; ++q) {
                    uint mask = q == 0u ? (lo & 0xffffu) : q == 1u ? (lo & 0xffff0000u) :
                        q == 2u ? (hi & 0xffffu) : (hi & 0xffff0000u);
                    if (mask == 0u) { continue; }
                    uint c = countbits(mask);
                    uint admitted = min(capacity - min(capacity, triCursor), c);
                    triCursor += c;
                    if (admitted < c) {
                        InterlockedAdd(statistics[5], c - admitted);
                        for (uint k = admitted; k < c; ++k) { mask &= mask - 1u; }
                    }
                    if (mask != 0u) {
                        uint ticket;
                        InterlockedAdd(work[0].x, 1u, ticket);
                        work[1u + ticket] = uint4(firstTriangle, p, q < 2u ? mask : 0u, q >= 2u ? mask : 0u);
                    }
                }
            }
        }
#endif
    }
    if (moreLevels) {
        GroupMemoryBarrierWithGroupSync();
    }
    return true;
}

bool vsmLoadClusterTrianglePositions(uint geometry, uint triangleIndex, out float3 v0, out float3 v1, out float3 v2)
{
    VirtualShadowMapTriangle tri;
    if (!loadVirtualShadowMapTriangle(geometry, triangleIndex, tri)) {
        v0 = v1 = v2 = float3(0.0);
        return false;
    }
    v0 = tri.vertices[0].position.xyz;
    v1 = tri.vertices[1].position.xyz;
    v2 = tri.vertices[2].position.xyz;
    return true;
}
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmBuildMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID,
    uint lane : SV_GroupIndex,
#if VSM_BATCHED_BUILD_RASTER
    uniform ShadowPush batchPush)
#else
    uniform ShadowPush push)
#endif
{
#if VSM_BATCHED_BUILD_RASTER
    ShadowPush push;
    if (!vsmBatchComputePush(group, batchPush, push)) { return; }
    id = uint3(group.x * 64u + lane, 0u, 0u);
#endif
#if VSM_COMPACT_BUILD
    StructuredBuffer<uint> candidates = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint candidate = vsmCompactCandidateOffset(push) + 1u + (push.workBase + group.x) * 2u;
#if VSM_INSTANCE_BUILD
    uint firstSlot = candidates[candidate];
    uint firstTriangle = firstSlot & ~63u;
    uint triangleIndex = firstTriangle + lane;
#else
    uint firstTriangle = candidates[candidate];
    uint triangleIndex = firstTriangle + lane;
    uint levelMask = candidates[candidate + 1u];
#if !VSM_CLUSTER_PAGE_RASTER
    if (triangleIndex >= push.triangleCount) { return; }
#endif
#endif
#else
    uint triangleIndex = push.workBase + id.x;
    // Each CPU dispatch starts at a multiple of 64, including batched draws.
    // Only uniform exits precede the two group barriers; tail lanes participate.
    uint firstTriangle = push.workBase + group.x * 64u;
    if (firstTriangle >= push.triangleCount || push.enabled == 0u) { return; }
#endif
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    bool mergeFrame = (push.cacheMode & 32u) != 0u;
    bool mergeOnly = mergeFrame && statistics[16] == 0u;
#if !VSM_COMPACT_BUILD
    if (!mergeFrame && statistics[16] == 0u) { return; }
    bool selected = true;
    if (mergeOnly) {
        // Any independent static invalidation also dirties the composite. When
        // no fine composite needs work, the bitset contains only static merges.
        uint cluster = firstTriangle >> 6u;
        if (cluster >= vsmCacheMetadata(push, kVSMCacheMergeClusterCount)) { return; }
        uint4 record = vsmMergeCluster(push, cluster);
        firstTriangle = record.x;
        triangleIndex = firstTriangle + lane;
        selected = vsmMergeClusterHasLane(record, lane);
    }
#endif
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
#if VSM_INSTANCE_BUILD
    uint levelMask = vsmCullCluster(firstTriangle, lane, push, projections, pages);
    if (levelMask == 0u) { return; }
#if !VSM_CLUSTER_PAGE_RASTER
    if (triangleIndex < firstSlot || triangleIndex >= candidates[candidate + 1u]) { return; }
#endif
#endif
#if !VSM_COMPACT_BUILD
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
    if (levelMask == 0u) { return; }
#if !VSM_CLUSTER_PAGE_RASTER
    if (!selected || triangleIndex >= push.triangleCount) { return; }
#endif
#endif
#if VSM_CLUSTER_PAGE_RASTER
    // All 64 lanes of this surviving cluster participate in per-level LDS init and flush.
#if VSM_INSTANCE_BUILD
    bool laneActive = triangleIndex >= firstSlot && triangleIndex < candidates[candidate + 1u];
#elif VSM_COMPACT_BUILD
    bool laneActive = triangleIndex < push.triangleCount;
#else
    bool laneActive = selected && triangleIndex < push.triangleCount;
#endif
    bool mergeTriangle = true;
    if (laneActive && mergeFrame && (VSM_COMPACT_BUILD != 0 || !mergeOnly)) {
        uint word = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, kVSMCacheMergeGeometry), triangleIndex >> 5u);
        mergeTriangle = ((word >> (triangleIndex & 31u)) & 1u) != 0u;
        if (mergeOnly && !mergeTriangle) { laneActive = false; }
    }
    float3 triPos0 = float3(0.0), triPos1 = float3(0.0), triPos2 = float3(0.0);
    if (laneActive) {
        vsmWorkAdd(push, kVSMWorkBuildActiveTriangleSlots, 1u);
        if (!vsmLoadClusterTrianglePositions(push.geometry, triangleIndex, triPos0, triPos1, triPos2)) {
            laneActive = false;
        }
    }
    if (laneActive) {
        vsmWorkAdd(push, kVSMWorkBuildDecodedTriangles, 1u);
        vsmWorkAdd(push, kVSMWorkBuildLevelTests, countbits(levelMask));
    }
    StructuredBuffer<uint> bitset = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint bitsetOffset = vsmClusterPageBitsetOffset(push);
#if VSM_WORKLOAD_DIAGNOSTICS
    uint wordTests = 0u, pageHits = 0u;
    bool layeredBuild = (push.cacheMode & 16u) != 0u;
    bool dynamicTriangle = false;
    bool staticFineRequired = false, compositeFineHit = false;
    if (laneActive && layeredBuild) {
        // The dynamic layer descriptor has the same slot indices as full Build.
        uint mask = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, 34u), triangleIndex >> 5u);
        dynamicTriangle = ((mask >> (triangleIndex & 31u)) & 1u) != 0u;
        vsmBuildDiagnosticAdd(push, dynamicTriangle ? kVSMBuildDiagnosticLayeredDynamicDecodedTriangles :
            kVSMBuildDiagnosticLayeredStaticDecodedTriangles, 1u);
    }
#endif
    if (lane == 0u) {
        g_vsmClusterUsedCount[0] = 0u;
        g_vsmClusterUsedCount[1] = 0u;
        g_vsmClusterTotalTris[0] = 0u;
        g_vsmClusterTotalTris[1] = 0u;
        g_vsmClusterTotalTickets[0] = 0u;
        g_vsmClusterTotalTickets[1] = 0u;
    }
    for (uint pass = 0u; pass < kVSMClusterPageTablePasses; ++pass) {
        uint s = lane + pass * 64u;
        g_vsmClusterPageKey[s].store(0xffffffffu, MemoryOrder.Relaxed);
        g_vsmClusterPageMaskLo[s] = 0u;
        g_vsmClusterPageMaskHi[s] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint parity = 0u;
    for (uint level = 0u; level < push.clipmapCount; ++level) {
        if ((levelMask & (1u << level)) == 0u) { continue; }
        ShadowProjection projection = projections[level];
        if (laneActive && vsmLODSelected(push, triangleIndex, level)) {
            bool levelValid = true;
            float3 lo, hi;
#if !VSM_DIRECTIONAL_BUILD
            if (projection.metric.w > 0.5) {
                ShadowClipVertex polygon[9];
                polygon[0].position = vsmProjectClip(triPos0, projection); polygon[0].uv = float2(0.0);
                polygon[1].position = vsmProjectClip(triPos1, projection); polygon[1].uv = float2(0.0);
                polygon[2].position = vsmProjectClip(triPos2, projection); polygon[2].uv = float2(0.0);
                uint count = vsmClipProjection(polygon, float4(-1.0, -1.0, 1.0, 1.0), true);
                if (count < 3u) { levelValid = false; }
                else {
                    lo = hi = polygon[0].position.xyz / polygon[0].position.w;
                    for (uint vertex = 1u; vertex < count; ++vertex) {
                        float3 projected = polygon[vertex].position.xyz / polygon[vertex].position.w;
                        lo = min(lo, projected); hi = max(hi, projected);
                    }
                }
            } else
#endif
            {
                float3 a = vsmProjectPosition(triPos0, projection);
                float3 b = vsmProjectPosition(triPos1, projection);
                float3 c = vsmProjectPosition(triPos2, projection);
                lo = min(a, min(b, c)); hi = max(a, max(b, c));
            }
            if (levelValid && !(hi.z < 0.0 || lo.z > 1.0 || any(hi.xy < -1.0) || any(lo.xy > 1.0))) {
                int axis = int(projection.page.y);
                int2 first = min(int2(floor(saturate(lo.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
                int2 last = min(int2(floor(saturate(hi.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
#if VSM_RECEIVER_MASK
                float resolution = float(uint(axis) * push.pageSize);
                float3 positionMagnitude = max(abs(triPos0), max(abs(triPos1), abs(triPos2)));
                float2 clipMagnitude = float2(dot(abs(projection.rowX.xyz), positionMagnitude),
                    dot(abs(projection.rowY.xyz), positionMagnitude)) + abs(float2(projection.rowX.w, projection.rowY.w)) + 1.0;
                float2 pageTranslationMagnitude = (abs(projection.originNear.xy) + float(axis) + 1.0) * (2.0 / float(axis));
                float2 texelPadding = 1.0 + (clipMagnitude + pageTranslationMagnitude) *
                    (256.0 * 5.960464477539063e-8) * (0.5 * resolution);
                uint2 firstTexel = uint2(clamp(saturate(lo.xy * 0.5 + 0.5) * resolution - texelPadding, 0.0, resolution - 1.0));
                uint2 lastTexel = uint2(clamp(saturate(hi.xy * 0.5 + 0.5) * resolution + texelPadding, 0.0, resolution - 1.0));
#endif
                uint cachedWord = 0xffffffffu, rawWord = 0u;
                for (int y = first.y; y <= last.y; ++y) {
                    uint firstPage = projection.page.x + uint(y * axis + first.x);
                    uint lastPage = projection.page.x + uint(y * axis + last.x);
                    uint firstWord = firstPage >> 5u, lastWord = lastPage >> 5u;
                    for (uint word = firstWord; word <= lastWord; ++word) {
                        if (word != cachedWord) {
                            rawWord = bitset[bitsetOffset + word];
                            cachedWord = word;
#if VSM_WORKLOAD_DIAGNOSTICS
                            ++wordTests;
#endif
                        }
                        uint bits = rawWord;
                        if (word == firstWord) { bits &= 0xffffffffu << (firstPage & 31u); }
                        if (word == lastWord) { bits &= 0xffffffffu >> (31u - (lastPage & 31u)); }
                        while (bits != 0u) {
                            uint page = (word << 5u) + uint(firstbitlow(bits));
                            bits &= bits - 1u;
                            if (!mergeTriangle && (pages[page].w & (2u | kVSMPageMergeStatic)) == kVSMPageMergeStatic) {
                                continue;
                            }
#if VSM_WORKLOAD_DIAGNOSTICS
                            ++pageHits;
#endif
#if VSM_RECEIVER_MASK
                            if ((pages[page].w & 8u) == 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                                vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticReceiverMaskPageTests, 1u);
#endif
                                uint localPage = page - projection.page.x;
                                uint2 mask = vsmReceiverTexelMask(firstTexel, lastTexel,
                                    uint2(localPage % uint(axis), localPage / uint(axis)), push.pageSize);
                                uint offset = vsmReceiverMaskOffset(push) + page * 2u;
                                if (((mask.x & bitset[offset]) | (mask.y & bitset[offset + 1u])) == 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                                    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticReceiverMaskPageRejects, 1u);
#endif
                                    continue;
                                }
                            }
#endif
#if VSM_WORKLOAD_DIAGNOSTICS
                            if (layeredBuild && !dynamicTriangle) {
                                uint flags = pages[page].w;
                                staticFineRequired = staticFineRequired || ((flags & 8u) != 0u &&
                                    ((flags & kVSMPageMergeStatic) == 0u || mergeTriangle));
                                compositeFineHit = compositeFineHit || (flags & 2u) != 0u;
                            }
#endif
                            if (!vsmTryInsertClusterPage(page, lane, parity)) {
                                vsmEmitSingleClusterPage(page, lane, firstTriangle, push);
                            }
                        }
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        bool moreLevels = (level + 1u) < push.clipmapCount && (levelMask >> (level + 1u)) != 0u;
        if (vsmFlushClusterPages(firstTriangle, lane, parity, moreLevels, push)) {
            parity ^= 1u;
        }
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    if (laneActive) {
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticTriangleBitsetWordTests, wordTests);
        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticTriangleBitsetPageHits, pageHits);
        if (layeredBuild && !dynamicTriangle) {
            vsmBuildDiagnosticAdd(push, staticFineRequired ? kVSMBuildDiagnosticStaticRequiredFineTriangles :
                (compositeFineHit ? kVSMBuildDiagnosticStaticCompositeOnlyTriangles :
                    kVSMBuildDiagnosticStaticNoFinePageTriangles), 1u);
        }
    }
#endif
#else
    bool mergeTriangle = true;
    if (mergeFrame && (VSM_COMPACT_BUILD != 0 || !mergeOnly)) {
        uint word = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, kVSMCacheMergeGeometry), triangleIndex >> 5u);
        mergeTriangle = ((word >> (triangleIndex & 31u)) & 1u) != 0u;
        if (mergeOnly && !mergeTriangle) { return; }
    }
    vsmWorkAdd(push, kVSMWorkBuildActiveTriangleSlots, 1u);
    VirtualShadowMapTriangle triangle;
    if (!loadVirtualShadowMapTriangle(push.geometry, triangleIndex, triangle)) { return; }
    float3 triPos0 = triangle.vertices[0].position.xyz;
    float3 triPos1 = triangle.vertices[1].position.xyz;
    float3 triPos2 = triangle.vertices[2].position.xyz;
#if VSM_PAGE_WINDOW_RASTER
    RWStructuredBuffer<uint4> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.workItems));
#else
    RWStructuredBuffer<uint2> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.workItems));
#endif
    vsmWorkAdd(push, kVSMWorkBuildDecodedTriangles, 1u);
    vsmWorkAdd(push, kVSMWorkBuildLevelTests, countbits(levelMask));
    StructuredBuffer<uint> bitset = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
    uint bitsetOffset = vsmClusterPageBitsetOffset(push);
#if VSM_WORKLOAD_DIAGNOSTICS
    uint wordTests = 0u, pageHits = 0u;
    bool layeredBuild = (push.cacheMode & 16u) != 0u;
    bool dynamicTriangle = false;
    bool staticFineRequired = false, compositeFineHit = false;
    if (layeredBuild) {
        // The dynamic layer descriptor has the same slot indices as full Build.
        uint mask = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, 34u), triangleIndex >> 5u);
        dynamicTriangle = ((mask >> (triangleIndex & 31u)) & 1u) != 0u;
        vsmBuildDiagnosticAdd(push, dynamicTriangle ? kVSMBuildDiagnosticLayeredDynamicDecodedTriangles :
            kVSMBuildDiagnosticLayeredStaticDecodedTriangles, 1u);
    }
#endif
#if VSM_PAGE_WINDOW_RASTER
    uint geometrySlot = 0xffffffffu;
#endif
    for (uint level = 0u; level < push.clipmapCount; ++level) {
        if ((levelMask & (1u << level)) == 0u || !vsmLODSelected(push, triangleIndex, level)) { continue; }
        ShadowProjection projection = projections[level];
        float3 lo, hi;
#if !VSM_DIRECTIONAL_BUILD
        if (projection.metric.w > 0.5) {
            ShadowClipVertex polygon[9];
            polygon[0].position = vsmProjectClip(triPos0, projection); polygon[0].uv = float2(0.0);
            polygon[1].position = vsmProjectClip(triPos1, projection); polygon[1].uv = float2(0.0);
            polygon[2].position = vsmProjectClip(triPos2, projection); polygon[2].uv = float2(0.0);
            // Projecting only the original vertices misses triangles crossing
            // near/w=0. Clip the complete face before any perspective division.
            uint count = vsmClipProjection(polygon, float4(-1.0, -1.0, 1.0, 1.0), true);
            if (count < 3u) { continue; }
            lo = hi = polygon[0].position.xyz / polygon[0].position.w;
            for (uint vertex = 1u; vertex < count; ++vertex) {
                float3 projected = polygon[vertex].position.xyz / polygon[vertex].position.w;
                lo = min(lo, projected); hi = max(hi, projected);
            }
        } else
#endif
        {
            float3 a = vsmProjectPosition(triPos0, projection);
            float3 b = vsmProjectPosition(triPos1, projection);
            float3 c = vsmProjectPosition(triPos2, projection);
            lo = min(a, min(b, c)); hi = max(a, max(b, c));
        }
        if (hi.z < 0.0 || lo.z > 1.0 || any(hi.xy < -1.0) || any(lo.xy > 1.0)) { continue; }
        int axis = int(projection.page.y);
        // Clip the projected bounds before integer conversion; off-clipmap
        // world coordinates need not fit the page-index integer range.
        int2 first = min(int2(floor(saturate(lo.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
        int2 last = min(int2(floor(saturate(hi.xy * 0.5 + 0.5) * float(axis))), int2(axis - 1));
#if VSM_RECEIVER_MASK
        // Build uses clipmap translation; Raster forms canonical page translation.
        // Bound the operand magnitudes of both paths before converting their
        // rounding allowance to texels. The 256-epsilon allowance follows Mark's
        // conservative footprint policy; one extra texel covers integer boundaries.
        float resolution = float(uint(axis) * push.pageSize);
        float3 positionMagnitude = max(abs(triPos0), max(abs(triPos1), abs(triPos2)));
        float2 clipMagnitude = float2(dot(abs(projection.rowX.xyz), positionMagnitude),
            dot(abs(projection.rowY.xyz), positionMagnitude)) + abs(float2(projection.rowX.w, projection.rowY.w)) + 1.0;
        float2 pageTranslationMagnitude = (abs(projection.originNear.xy) + float(axis) + 1.0) * (2.0 / float(axis));
        float2 texelPadding = 1.0 + (clipMagnitude + pageTranslationMagnitude) *
            (256.0 * 5.960464477539063e-8) * (0.5 * resolution);
        uint2 firstTexel = uint2(clamp(saturate(lo.xy * 0.5 + 0.5) * resolution - texelPadding, 0.0, resolution - 1.0));
        uint2 lastTexel = uint2(clamp(saturate(hi.xy * 0.5 + 0.5) * resolution + texelPadding, 0.0, resolution - 1.0));
#endif
        uint cachedWord = 0xffffffffu, rawWord = 0u;
        for (int y = first.y; y <= last.y; ++y) {
            uint firstPage = projection.page.x + uint(y * axis + first.x);
            uint lastPage = projection.page.x + uint(y * axis + last.x);
            uint firstWord = firstPage >> 5u, lastWord = lastPage >> 5u;
            for (uint word = firstWord; word <= lastWord; ++word) {
                // Keep the raw word: adjacent non-aligned rows can share it,
                // but their masks must not consume one another's set bits.
                if (word != cachedWord) {
                    rawWord = bitset[bitsetOffset + word];
                    cachedWord = word;
#if VSM_WORKLOAD_DIAGNOSTICS
                    ++wordTests;
#endif
                }
                uint bits = rawWord;
                if (word == firstWord) { bits &= 0xffffffffu << (firstPage & 31u); }
                if (word == lastWord) { bits &= 0xffffffffu >> (31u - (lastPage & 31u)); }
#if VSM_PAGE_WINDOW_RASTER
                // A nibble stays within this row and dirty word, including when the row starts mid-word.
                uint pending = bits;
                while (pending != 0u) {
                    uint shift = uint(firstbitlow(pending)) & ~3u;
                    bits = pending & (15u << shift);
                    pending ^= bits;
                    uint basePage = (word << 5u) + shift;
                    uint accepted = 0u;
#endif
                while (bits != 0u) {
                    uint page = (word << 5u) + uint(firstbitlow(bits));
                    bits &= bits - 1u;
                    // Admit/Assign set exactly resident, admitted, dirty fine
                    // pages. Physical pins admitted owners through this build.
                    // Increasing set bits preserve this triangle's row order.
                    if (!mergeTriangle && (pages[page].w & (2u | kVSMPageMergeStatic)) == kVSMPageMergeStatic) {
                        continue;
                    }
#if VSM_WORKLOAD_DIAGNOSTICS
                    ++pageHits;
#endif
#if VSM_RECEIVER_MASK
                    // Both layers share this work list. Static-dirty pages must
                    // remain complete for reuse by future receiver footprints.
                    if ((pages[page].w & 8u) == 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                        vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticReceiverMaskPageTests, 1u);
#endif
                        uint localPage = page - projection.page.x;
                        uint2 mask = vsmReceiverTexelMask(firstTexel, lastTexel,
                            uint2(localPage % uint(axis), localPage / uint(axis)), push.pageSize);
                        uint offset = vsmReceiverMaskOffset(push) + page * 2u;
                        if (((mask.x & bitset[offset]) | (mask.y & bitset[offset + 1u])) == 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                            vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticReceiverMaskPageRejects, 1u);
#endif
                            continue;
                        }
                    }
#endif
#if VSM_WORKLOAD_DIAGNOSTICS
                    if (layeredBuild && !dynamicTriangle) {
                        uint flags = pages[page].w;
                        // Incremental static merge consumes only the transition mask.
                        staticFineRequired = staticFineRequired || ((flags & 8u) != 0u &&
                            ((flags & kVSMPageMergeStatic) == 0u || mergeTriangle));
                        compositeFineHit = compositeFineHit || (flags & 2u) != 0u;
                    }
#endif
#if VSM_PAGE_WINDOW_RASTER
                    accepted |= 1u << (page - basePage);
#else
                    uint index;
                    InterlockedAdd(statistics[3], 1u, index);
                    if (index < push.workCapacity) { work[index] = uint2(triangleIndex, page); }
                    else { InterlockedAdd(statistics[5], 1u); }
#endif
                }
#if VSM_PAGE_WINDOW_RASTER
                    if (accepted == 0u) { continue; }
                    uint count = countbits(accepted), firstWork;
                    InterlockedAdd(statistics[3], count, firstWork);
                    uint admitted = min(count, push.workCapacity - min(firstWork, push.workCapacity));
                    if (admitted < count) {
                        InterlockedAdd(statistics[5], count - admitted);
                        for (uint i = admitted; i < count; ++i) { accepted &= accepted - 1u; }
                        if (accepted == 0u) { continue; }
                    }
                    uint ticket;
                    InterlockedAdd(work[0].x, 1u, ticket);
                    if (geometrySlot == 0xffffffffu) {
                        geometrySlot = push.triangleCount <= push.workCapacity ? triangleIndex : ticket;
                    }
                    work[1u + ticket] = uint4(triangleIndex, basePage, accepted, geometrySlot);
                }
#endif
            }
        }
    }
#if VSM_PAGE_WINDOW_RASTER
    // Build owns each selected triangle once, including split instance ranges and demotion.
    // The Build -> mesh barrier publishes this payload together with every referring ticket.
    if (geometrySlot != 0xffffffffu) {
        uint base = 1u + push.workCapacity + 4u * geometrySlot;
        for (uint i = 0u; i < 3u; ++i) {
            work[base + i] = asuint(float4(triangle.vertices[i].position.xyz, triangle.vertices[i].uv.x));
        }
        work[base + 3u] = uint4(asuint(float3(triangle.vertices[0].uv.y, triangle.vertices[1].uv.y,
            triangle.vertices[2].uv.y)), triangle.materialIndex);
    }
#endif
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticTriangleBitsetWordTests, wordTests);
    vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticTriangleBitsetPageHits, pageHits);
    if (layeredBuild && !dynamicTriangle) {
        vsmBuildDiagnosticAdd(push, staticFineRequired ? kVSMBuildDiagnosticStaticRequiredFineTriangles :
            (compositeFineHit ? kVSMBuildDiagnosticStaticCompositeOnlyTriangles :
                kVSMBuildDiagnosticStaticNoFinePageTriangles), 1u);
    }
#endif
#endif
}

