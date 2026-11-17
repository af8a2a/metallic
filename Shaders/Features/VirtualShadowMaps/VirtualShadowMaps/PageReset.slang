#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_RECEIVER_MASK
#define VSM_RECEIVER_MASK 0
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
#ifndef VSM_DIRTY_PAGE_HIERARCHY
#define VSM_DIRTY_PAGE_HIERARCHY 0
#endif
#ifndef VSM_CLUSTER_PAGE_BITSET
#define VSM_CLUSTER_PAGE_BITSET 0
#endif
#ifndef VSM_SHARED_PHYSICAL_POOL
#define VSM_SHARED_PHYSICAL_POOL 0
#endif

#if VSM_STATIC_PAGE_HZB
// Support points give the extrema of an orthographic row over the entire world AABB.
// Bound dot/clip rounding by operand magnitude even when translation nearly cancels.
float2 vsmHZBProjectedRange(float3 minimum, float3 maximum, float4 row)
{
    float low = dot(row, float4(select(row.xyz >= 0.0, minimum, maximum), 1.0));
    float high = dot(row, float4(select(row.xyz >= 0.0, maximum, minimum), 1.0));
    float error = (dot(abs(row.xyz), max(abs(minimum), abs(maximum))) + abs(row.w) + 1.0) *
        (256.0 * 5.960464477539063e-8);
    return float2(low - error, high + error);
}

bool vsmStaticPageHidesBounds(ShadowPush push, ShadowProjection projection, uint2 page,
    uint physicalPage, float3 minimum, float3 maximum)
{
    // Match fine Raster's canonical page translation, including after clipmap scroll.
    float axis = float(projection.page.y);
    float halfPage = 1.0 / axis;
    if (projection.page.w == 0u) {
        float2 center = projection.originNear.xy + float2(page) + 0.5 - 0.5 * axis;
        projection.rowX.w = -center.x * (2.0 * halfPage);
        projection.rowY.w = -center.y * (2.0 * halfPage);
    }
    float2 x = vsmHZBProjectedRange(minimum, maximum, projection.rowX);
    float2 y = vsmHZBProjectedRange(minimum, maximum, projection.rowY);
    float closestDepth = vsmHZBProjectedRange(minimum, maximum, projection.rowZ).y;
    float scale = float(push.pageSize) / (2.0 * halfPage);
    float2 lo = (float2(x.x, y.x) + halfPage) * scale - 1.0;
    float2 hi = (float2(x.y, y.y) + halfPage) * scale + 1.0;
    if (!all(isfinite(lo)) || !all(isfinite(hi)) || !isfinite(closestDepth)) { return false; }
    float last = float(push.pageSize - 1u);
    return vsmStaticHZBOccludes(push, physicalPage, uint2(clamp(lo, 0.0, last)),
        uint2(clamp(hi, 0.0, last)), closestDepth);
}
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmResetMain(uint3 id : SV_DispatchThreadID, uniform ShadowPush push)
{
    if (id.x == 0u) { vsmLODPublishChangedCasters(push); }
#if VSM_CLUSTER_PAGE_RASTER || VSM_PAGE_WINDOW_RASTER
    if (id.x == 0u) {
        RWStructuredBuffer<uint4> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.workItems));
        work[0] = uint4(0u);
    }
#endif
#if VSM_STATIC_PAGE_HZB
    if (id.x == 0u) {
        RWStructuredBuffer<uint> jobs = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, 38u)));
        jobs[0] = 0u; jobs[1] = 1u; jobs[2] = 1u;
    }
#endif
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    RWStructuredBuffer<uint> owners = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.owners));
    if (id.x < push.totalPages) {
        bool coarse = id.x == push.totalPages - 1u;
#if VSM_RECEIVER_MASK
        uint2 validReceiverMask = uint2(0u);
#endif
        uint initialPhysical = 0xffffffffu;
#if !VSM_SHARED_PHYSICAL_POOL
        initialPhysical = coarse ? 0u : 0xffffffffu;
#endif
        uint4 entry = uint4(initialPhysical, coarse && push.enabled != 0u ? 3u : 0u,
            coarse ? push.clipmapCount : id.x / (push.virtualPagesPerAxis * push.virtualPagesPerAxis),
            2u | ((push.cacheMode & 16u) != 0u ? 8u : 0u));
        uint mappingFlags = 0u;
        if ((push.cacheMode & 1u) != 0u) {
            // Fine levels occupy fixed VPA^2 strides; local mips only use the
            // packed axis^2 prefix. The remaining slots must stay unmapped.
            StructuredBuffer<uint4> changes = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.reuse));
            uint4 change = uint4(0u);
            uint2 page = uint2(0u);
            uint oldIndex = coarse ? id.x : 0xffffffffu;
            if (!coarse) {
                uint stride = push.virtualPagesPerAxis * push.virtualPagesPerAxis;
                uint level = id.x / stride, local = id.x % stride;
                uint axis = push.lightType == 0u ? push.virtualPagesPerAxis : max(1u, push.virtualPagesPerAxis >> level);
                change = changes[level];
                if (local < axis * axis) {
                    page = uint2(local % axis, local / axis);
                    int2 oldPage = int2(page) + asint(change.xy);
                    if (all(uint2(oldPage) < uint2(axis))) {
                        oldIndex = level * stride + uint(oldPage.y) * axis + uint(oldPage.x);
                    }
                }
            }
            if (oldIndex != 0xffffffffu) {
                StructuredBuffer<uint4> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.previousPages));
                uint4 old = previous[oldIndex];
                bool mappingDirty = (push.cacheMode & 4u) != 0u || (coarse && push.dirtyBoundsCount != 0u);
                // Only resident pages need rectangle tests. Count consumes these
                // flags before replacing the same scratch words with scan ranks.
                if (old.x != 0xffffffffu) {
                    bool lodDirty = vsmLODPageChanged(push, entry.z, page, coarse);
                    mappingDirty = mappingDirty || lodDirty;
                    uint staticFlags = 0u;
                    if ((push.cacheMode & 16u) != 0u) {
                        bool staticDirty = lodDirty || (push.cacheMode & 4u) != 0u || (coarse && vsmCacheMetadata(push, 35u) != 0u);
                        if (!coarse) {
                            uint4 staticChange = changes[vsmCacheMetadata(push, 36u) + entry.z];
                            for (uint rectangle = 0u; !staticDirty && rectangle < staticChange.w; ++rectangle) {
                                uint4 bounds = changes[staticChange.z + rectangle];
                                staticDirty = all(page >= bounds.xy) && all(page <= bounds.zw);
                            }
                        }
                        vsmCacheAdd(push, kVSMCacheStaticInvalidatedResidentPages, staticDirty ? 1u : 0u);
                        staticFlags = staticDirty || (old.w & 4u) == 0u ? 8u : 4u;
                    }
#if VSM_STATIC_PAGE_HZB && VSM_WORKLOAD_DIAGNOSTICS
                    bool hadOverlap = false;
#endif
#if VSM_STATIC_PAGE_HZB
                    StructuredBuffer<ShadowProjection> projections = resolveUniform(
                        ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
                    if (coarse && mappingDirty && staticFlags == 4u && (old.w & 1u) != 0u && push.dirtyBoundsCount <= 32u) {
                        // The coarse page has no per-level rect list. Its stable
                        // projection covers the whole scene. Only a proof for
                        // every old/new dirty AABB permits keeping its depth.
                        uint base = vsmCacheMetadata(push, 39u);
                        mappingDirty = false;
                        for (uint b = 0u; b < push.dirtyBoundsCount && !mappingDirty; ++b) {
                            uint4 lo = changes[base + b * 2u];
                            mappingDirty = lo.w == 0u || !vsmStaticPageHidesBounds(push, projections[entry.z], page, old.x,
                                asfloat(lo.xyz), asfloat(changes[base + b * 2u + 1u].xyz));
                        }
#if VSM_WORKLOAD_DIAGNOSTICS
                        hadOverlap = true;
#endif
                    }
#endif
                    for (uint rectangle = 0u; !mappingDirty && rectangle < change.w; ++rectangle) {
                        uint4 bounds = changes[change.z + rectangle];
#if VSM_STATIC_PAGE_HZB
                        mappingDirty = all(page >= bounds.xy) && all(page <= uint2(bounds.z & 0xffffu, bounds.z >> 16u));
                        if (mappingDirty && staticFlags == 4u && (old.w & 1u) != 0u) {
#if VSM_WORKLOAD_DIAGNOSTICS
                            hadOverlap = true;
#endif
                            uint index = vsmCacheMetadata(push, 39u) + 2u * bounds.w;
                            mappingDirty = !vsmStaticPageHidesBounds(push, projections[entry.z], page, old.x,
                                asfloat(changes[index].xyz), asfloat(changes[index + 1u].xyz));
                        }
#else
                        mappingDirty = all(page >= bounds.xy) && all(page <= bounds.zw);
#endif
                    }
                    if ((push.cacheMode & 32u) != 0u) {
                        bool transitioned = coarse;
                        if (!coarse) {
                            uint4 merge = changes[vsmCacheMetadata(push, kVSMCacheMergeRemap) + entry.z];
                            for (uint rectangle = 0u; !transitioned && rectangle < merge.w; ++rectangle) {
                                uint4 bounds = changes[merge.z + rectangle];
                                transitioned = all(page >= bounds.xy) && all(page <= bounds.zw);
                            }
                        }
                        if (transitioned) {
                            // The rasterized caster has not changed. Its depth was already in
                            // the composite atlas. Independent invalidations still require restore.
                            if (staticFlags == 4u) { staticFlags = 8u | kVSMPageMergeStatic; }
                            else { mappingDirty = true; }
                        }
                    }
                    entry.x = old.x;
                    // A deferred or overflowing static merge must recover from all static
                    // geometry before this slot can be restored into the composite again.
                    bool dirty = mappingDirty || (old.w & 1u) == 0u ||
                        ((push.cacheMode & 16u) != 0u && (old.w & 4u) == 0u);
#if VSM_RECEIVER_MASK
                    if (!dirty && !coarse) {
                        ByteAddressBuffer previousCoverage = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.previousPages));
                        validReceiverMask = previousCoverage.Load2(vsmValidReceiverMaskByteOffset(push, oldIndex));
                        // Full redraw on a merge would publish complete page depth. Do not
                        // upgrade a partially covered composite to full without rasterizing it.
                        if ((staticFlags & kVSMPageMergeStatic) != 0u && any(validReceiverMask != uint2(0xffffffffu))) {
                            dirty = true;
                            mappingDirty = true;
                            validReceiverMask = uint2(0u);
                        }
                    }
#endif
#if VSM_STATIC_PAGE_HZB && VSM_WORKLOAD_DIAGNOSTICS
                    vsmCacheAdd(push, kVSMCacheHZBSavedInvalidations, hadOverlap && !mappingDirty ? 1u : 0u);
#endif
                    mappingFlags = (mappingDirty ? 1u : 0u) | (oldIndex != id.x ? 2u : 0u);
                    entry.w = (dirty ? 2u : 1u) | staticFlags;
                    vsmCacheAdd(push, kVSMCacheCompositeInvalidatedResidentPages, mappingDirty ? 1u : 0u);
                }
            }
        }
        pages[id.x] = entry;
#if VSM_RECEIVER_MASK
        RWByteAddressBuffer validCoverage = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.pages));
        validCoverage.Store2(vsmValidReceiverMaskByteOffset(push, id.x), validReceiverMask);
#endif
        owners[push.physicalPageCount + id.x] = mappingFlags;
    }
    // Each counter has exactly one writer, including the diagnostic constants.
    if (id.x < 32u) {
        uint value = id.x == 10u ? push.triangleCount : id.x == 11u ? push.clipmapCount :
            id.x == 12u ? push.physicalPageCount : id.x == 14u ? 1u - push.enabled :
            id.x == 22u ? ((push.cacheMode & 8u) != 0u ? 1u : 0u) :
            id.x == 23u ? push.cacheEpoch : id.x == 25u ? push.dirtyBoundsCount :
            id.x == 26u ? ((push.cacheMode & 4u) != 0u ? 1u : 0u) :
            id.x == 27u ? ((push.cacheMode & 2u) != 0u ? 1u : 0u) :
            id.x == 28u ? push.cancelledCount : 0u;
        statistics[id.x] = value;
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    if (id.x < (kVirtualShadowMapWorkloadSize + kVirtualShadowMapBuildDiagnosticSize) / 4u) { statistics[32u + id.x] = 0u; }
#endif
    if (id.x < push.physicalPageCount) { owners[id.x] = 0u; }
#if VSM_CLUSTER_PAGE_BITSET
    if (id.x < (push.totalPages + 31u) / 32u) { owners[vsmClusterPageBitsetOffset(push) + id.x] = 0u; }
#endif
#if VSM_RECEIVER_MASK
    if (id.x < push.totalPages * 2u) { owners[vsmReceiverMaskOffset(push) + id.x] = 0u; }
#endif
#if VSM_DIRTY_PAGE_HIERARCHY
    vsmResetPageHierarchy(id.x, push, owners);
#endif
#if VSM_COMPACT_BUILD
    uint argumentBatches = push.batchCount + push.coarseBatchCount;
    if (id.x == 0u) { owners[vsmCompactCandidateOffset(push)] = 0u; }
    argumentBatches += vsmCompactBatchCount(push);
    if (id.x < argumentBatches * 3u) {
#else
    if (id.x < (push.batchCount + push.coarseBatchCount) * 3u) {
#endif
        arguments[id.x] = id.x % 3u == 0u ? 0u : 1u;
    }
}

