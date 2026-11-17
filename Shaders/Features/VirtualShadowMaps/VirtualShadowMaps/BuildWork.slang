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
#ifndef VSM_INSTANCE_BUILD
#define VSM_INSTANCE_BUILD 0
#endif
#ifndef VSM_BATCHED_BUILD_RASTER
#define VSM_BATCHED_BUILD_RASTER 0
#endif

#if VSM_CLUSTER_PAGE_RASTER
void vsmAppendClusterPage(uint page, uint triangleIndex, RWStructuredBuffer<uint4> work,
    RWStructuredBuffer<uint> statistics, uint capacity)
{
    // Every invocation here belongs to the same original aligned cluster,
    // including indirect Build and sparse demotion. Each 16-slot quarter
    // bounds worst-case clipped mesh output to 144 vertices and 112 primitives.
    uint key = (page << 2u) | ((triangleIndex >> 4u) & 3u);
    for (;;) {
        uint firstKey = WaveReadLaneFirst(key);
        if (key == firstKey) {
            uint mask = WaveActiveBitOr(1u << (triangleIndex & 31u));
            if (WaveIsFirstLane()) {
                uint count = countbits(mask);
                uint first;
                InterlockedAdd(statistics[3], count, first);
                uint admitted = min(capacity - min(capacity, first), count);
                if (admitted < count) {
                    InterlockedAdd(statistics[5], count - admitted);
                    for (uint i = admitted; i < count; ++i) { mask &= mask - 1u; }
                }
                if (mask != 0u) {
                    uint ticket;
                    InterlockedAdd(work[0].x, 1u, ticket);
                    bool high = (triangleIndex & 32u) != 0u;
                    work[1u + ticket] = uint4(triangleIndex & ~63u, page, high ? 0u : mask, high ? mask : 0u);
                }
            }
            break;
        }
    }
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
    uint triangleIndex = candidates[candidate] + lane;
    uint levelMask = candidates[candidate + 1u];
    if (triangleIndex >= push.triangleCount) { return; }
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
    if (levelMask == 0u || triangleIndex < firstSlot || triangleIndex >= candidates[candidate + 1u]) { return; }
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
#if VSM_WORKLOAD_DIAGNOSTICS
    // Keep the predicate's diagnostic side effects inside the lane/valid guard.
    if (lane < push.clipmapCount && g_vsmClusterValid != 0u && vsmLODSelectedRange(push, firstTriangle, lane)) {
        if (vsmClusterNeedsLevel(g_vsmClusterMinimum, g_vsmClusterMaximum, projections[lane], pages, push)) {
            InterlockedOr(g_vsmClusterLevelMask, 1u << lane);
        }
    }
#else
    if (lane < push.clipmapCount && g_vsmClusterValid != 0u && vsmLODSelectedRange(push, firstTriangle, lane) &&
        vsmClusterNeedsLevel(g_vsmClusterMinimum, g_vsmClusterMaximum, projections[lane], pages, push)) {
        InterlockedOr(g_vsmClusterLevelMask, 1u << lane);
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    uint levelMask = g_vsmClusterLevelMask;
#if VSM_WORKLOAD_DIAGNOSTICS
    if (lane == 0u && levelMask != 0u) { vsmBuildDiagnosticAdd(push, kVSMBuildDiagnosticClusterSurvivingGroups, 1u); }
#endif
    if (!selected || triangleIndex >= push.triangleCount || levelMask == 0u) { return; }
#endif
    bool mergeTriangle = true;
    if (mergeFrame && (VSM_COMPACT_BUILD != 0 || !mergeOnly)) {
        uint word = loadVirtualShadowMapDynamicMaskWord(vsmCacheMetadata(push, kVSMCacheMergeGeometry), triangleIndex >> 5u);
        mergeTriangle = ((word >> (triangleIndex & 31u)) & 1u) != 0u;
        if (mergeOnly && !mergeTriangle) { return; }
    }
    vsmWorkAdd(push, kVSMWorkBuildActiveTriangleSlots, 1u);
    VirtualShadowMapTriangle triangle;
    if (!loadVirtualShadowMapTriangle(push.geometry, triangleIndex, triangle)) { return; }
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
#if VSM_CLUSTER_PAGE_RASTER || VSM_PAGE_WINDOW_RASTER
    RWStructuredBuffer<uint4> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.workItems));
#else
    RWStructuredBuffer<uint2> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint2>>(push.workItems));
#endif
    for (uint level = 0u; level < push.clipmapCount; ++level) {
        if ((levelMask & (1u << level)) == 0u || !vsmLODSelected(push, triangleIndex, level)) { continue; }
        ShadowProjection projection = projections[level];
        float3 lo, hi;
#if !VSM_DIRECTIONAL_BUILD
        if (projection.metric.w > 0.5) {
            ShadowClipVertex polygon[9];
            for (uint vertex = 0u; vertex < 3u; ++vertex) {
                polygon[vertex].position = vsmProjectClip(triangle.vertices[vertex].position.xyz, projection);
                polygon[vertex].uv = float2(0.0);
            }
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
            float3 a = vsmProjectPosition(triangle.vertices[0].position.xyz, projection);
            float3 b = vsmProjectPosition(triangle.vertices[1].position.xyz, projection);
            float3 c = vsmProjectPosition(triangle.vertices[2].position.xyz, projection);
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
        float3 positionMagnitude = max(abs(triangle.vertices[0].position.xyz),
            max(abs(triangle.vertices[1].position.xyz), abs(triangle.vertices[2].position.xyz)));
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
#elif VSM_CLUSTER_PAGE_RASTER
                    vsmAppendClusterPage(page, triangleIndex, work, statistics, push.workCapacity);
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
}

