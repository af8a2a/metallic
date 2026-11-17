#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_RECEIVER_MASK
#define VSM_RECEIVER_MASK 0
#endif
#ifndef VSM_PAGE_WINDOW_RASTER
#define VSM_PAGE_WINDOW_RASTER 0
#endif
#ifndef VSM_CLUSTER_PAGE_RASTER
#define VSM_CLUSTER_PAGE_RASTER 0
#endif
#ifndef VSM_BATCHED_BUILD_RASTER
#define VSM_BATCHED_BUILD_RASTER 0
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmFinalizeMain(uint3 id : SV_DispatchThreadID,
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
    RWStructuredBuffer<uint> statistics = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.statistics));
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    uint count = min(statistics[3], push.workCapacity);
    if (id.x == 0u) { statistics[4] = count; }
    uint coarsePage = push.totalPages - 1u;
    bool redrawCoarse = statistics[21] != 0u;
    bool mergeFrame = (push.cacheMode & 32u) != 0u;
#if VSM_CLUSTER_PAGE_RASTER
    if (id.x == 0u) {
        uint clearOffset = vsmCacheMetadata(push, kVSMCacheMergeDrawOffset) +
            (mergeFrame ? push.batchCount + push.coarseBatchCount : 0u);
        arguments[clearOffset * 3u] = statistics[16] != 0u || redrawCoarse ? push.physicalPageCount : 0u;
        arguments[clearOffset * 3u + 1u] = 1u;
        arguments[clearOffset * 3u + 2u] = 1u;
    }
#endif
#if VSM_CLUSTER_HZB
    uint clusterCount = push.triangleCount / 64u + uint((push.triangleCount & 63u) != 0u);
    if (id.x < (clusterCount + 65534u) / 65535u && vsmCacheMetadata(push, 40u) != 0xffffffffu) {
        RWStructuredBuffer<uint> jobs = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, 38u)));
        uint offset = 6u + push.physicalPageCount + id.x * 3u;
        jobs[offset] = ((count != 0u && statistics[16] != 0u) || redrawCoarse) ? min(clusterCount - id.x * 65535u, 65535u) : 0u;
        jobs[offset + 1u] = 1u;
        jobs[offset + 2u] = 1u;
    }
#endif
    if (id.x < push.totalPages) {
        uint4 entry = pages[id.x];
        if (push.enabled != 0u && (entry.y & 2u) != 0u && (entry.w & 10u) != 0u) {
            if (id.x == coarsePage || statistics[5] == 0u) { entry.w |= 1u; }
            else { entry.w &= ~1u; }
            if ((entry.w & 2u) != 0u) {
                vsmCacheAdd(push, id.x == coarsePage ? kVSMCacheCompositeDirtyCoarsePages : kVSMCacheCompositeDirtyFinePages, 1u);
            }
            if ((push.cacheMode & 16u) != 0u && (entry.w & 8u) != 0u) {
                if (id.x == coarsePage || statistics[5] == 0u) { entry.w |= 4u; }
                else { entry.w &= ~4u; }
#if VSM_WORKLOAD_DIAGNOSTICS
                if ((entry.w & (4u | kVSMPageMergeStatic)) == (4u | kVSMPageMergeStatic)) {
                    vsmWorkAdd(push, kVSMWorkRasterStaticMergedPages, 1u);
                }
#endif
#if VSM_STATIC_PAGE_HZB
                if ((entry.w & 4u) != 0u) {
                    RWStructuredBuffer<uint> jobs = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, 38u)));
                    uint job;
                    InterlockedAdd(jobs[0], 1u, job);
                    jobs[3u + job] = entry.x;
                }
#endif
                vsmCacheAdd(push, id.x == coarsePage ? kVSMCacheStaticDirtyCoarsePages : kVSMCacheStaticDirtyFinePages, 1u);
            }
            pages[id.x] = entry;
#if VSM_RECEIVER_MASK
            // The pending cache publishes this tail with the page table only
            // after submission; the same validity bit gates projection while
            // raster completes the declared draw list. Overflow cannot certify
            // partial work. Redraw replaces coverage, never unions stale depth.
            RWByteAddressBuffer validCoverage = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.pages));
            uint2 valid = uint2(0u);
            if (id.x != coarsePage && (entry.w & 1u) != 0u) {
                if ((entry.w & 8u) != 0u) { valid = uint2(0xffffffffu); }
                else {
                    StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
                    uint offset = vsmReceiverMaskOffset(push) + id.x * 2u;
                    valid = uint2(owners[offset], owners[offset + 1u]);
                }
            }
            validCoverage.Store2(vsmValidReceiverMaskByteOffset(push, id.x), valid);
#endif
        }
        if ((push.cacheMode & 16u) != 0u && id.x != coarsePage && entry.x != 0xffffffffu &&
            (entry.y & 2u) != 0u && (entry.w & 12u) == 4u) { vsmCacheAdd(push, kVSMCacheStaticReusedFinePages, 1u); }
        if (id.x != coarsePage && entry.x != 0xffffffffu && (entry.w & 1u) != 0u) {
            InterlockedAdd(statistics[24], 1u);
        }
    }
    if (id.x < push.batchCount + push.coarseBatchCount) {
        bool coarse = id.x >= push.batchCount;
        uint first = (coarse ? id.x - push.batchCount : id.x) * 65535u;
        // Every demotion touches the complete coarse static page, even when its
        // previously composed depth is still exact.
        uint fineDrawCount = count;
#if VSM_CLUSTER_PAGE_RASTER || VSM_PAGE_WINDOW_RASTER
        StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
        fineDrawCount = work[0].x;
#endif
        uint drawCount = coarse ? ((redrawCoarse || (mergeFrame && push.enabled != 0u)) ? push.triangleCount : 0u) : fineDrawCount;
        uint dynamicCount = (coarse ? redrawCoarse : statistics[16] != 0u) ? drawCount : 0u;
        if (coarse && mergeFrame && drawCount != 0u) {
            StructuredBuffer<uint> owners = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.owners));
            uint coarseFlags = owners[vsmClusterPageBitsetOffset(push) - 1u];
            if ((coarseFlags & kVSMPageMergeStatic) != 0u) {
                uint clusters = vsmCacheMetadata(push, kVSMCacheMergeClusterCount);
                drawCount = clusters <= push.triangleCount / 64u ? clusters * 64u : push.triangleCount;
            }
        }
        arguments[id.x * 3u] = first < drawCount ? min(drawCount - first, 65535u) : 0u;
        arguments[id.x * 3u + 1u] = 1u;
        arguments[id.x * 3u + 2u] = 1u;
        if (mergeFrame) {
            uint offset = (vsmCacheMetadata(push, kVSMCacheMergeDrawOffset) + id.x) * 3u;
            arguments[offset] = first < dynamicCount ? min(dynamicCount - first, 65535u) : 0u;
            arguments[offset + 1u] = 1u;
            arguments[offset + 2u] = 1u;
        }
    }
}

// Projection-only metadata follows the unchanged residency/ownership prefix.
// Finalize must publish all validity bits before this independent dispatch.
void vsmBuildPropagatedMapping(uint3 id, ShadowPush push)
{
    if (id.x >= push.totalPages) { return; }
    RWStructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.pages));
    pages[push.totalPages + id.x] = uint4(0xffffffffu, push.clipmapCount, 0u, 0u);
    if (id.x == push.totalPages - 1u) { return; }
    uint axis = push.virtualPagesPerAxis, stride = axis * axis;
    uint level = id.x / stride, local = id.x % stride;
    uint4 entry = pages[id.x];
    if (entry.x != 0xffffffffu && (entry.y & 2u) != 0u && (entry.w & 1u) != 0u) {
        uint address = push.lightType == 0u ?
            vsmPackDirectionalMappingAddress(level, entry.x, push.physicalPagesX) : level;
        pages[push.totalPages + id.x] = uint4(entry.x, address, 0u, 1u);
        return;
    }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    ShadowProjection source = projections[level];
    if (push.lightType != 0u) {
        uint sourceAxis = source.page.y;
        if ((sourceAxis & (sourceAxis - 1u)) != 0u || local >= sourceAxis * sourceAxis) { return; }
        uint2 sourcePage = uint2(local % sourceAxis, local / sourceAxis);
        uint target = push.clipmapCount, physical = 0xffffffffu;
        for (uint next = level + 1u; next < push.clipmapCount; ++next) {
            uint delta = next - level;
            ShadowProjection parent = projections[next];
            if (parent.page.y != max(1u, sourceAxis >> delta) ||
                any(parent.rowX != source.rowX) || any(parent.rowY != source.rowY) ||
                any(parent.rowZ != source.rowZ) || any(parent.rowW != source.rowW)) { return; }
            // Local projections share identical normalized coordinates and
            // inclusive containment. Scaling their nonnegative address by a
            // power of two commutes with floor, including upper-edge clamping.
            uint2 parentPage = sourcePage >> min(delta, uint(firstbithigh(sourceAxis)));
            uint4 parentEntry = pages[parent.page.x + parentPage.y * parent.page.y + parentPage.x];
            if (parentEntry.x != 0xffffffffu && (parentEntry.y & 2u) != 0u && (parentEntry.w & 1u) != 0u) {
                target = next; physical = parentEntry.x;
                break;
            }
        }
        pages[push.totalPages + id.x] = uint4(physical, target, 0u, 3u);
        return;
    }
    if (axis < 2u || (axis & (axis - 1u)) != 0u) { return; }
    if (any(abs(source.originNear.xy) >= 67108864.0)) { return; }
    int2 basePage = int2(local % axis, local / axis) + int2(source.originNear.xy) - int(axis / 2u);
    uint packedParentPage = 0u;
    uint address = push.clipmapCount, physicalPage = 0xffffffffu;
    for (uint next = level + 1u; next < push.clipmapCount; ++next) {
        uint delta = next - level;
        float scale = asfloat((127u - delta) << 23u);
        ShadowProjection parent = projections[next];
        if (any(abs(parent.originNear.xy) >= 67108864.0) ||
            any(parent.rowX.xyz != source.rowX.xyz * scale) ||
            any(parent.rowY.xyz != source.rowY.xyz * scale) ||
            any(parent.rowZ != source.rowZ)) { return; }
        int2 parentPage = (basePage >> delta) - int2(parent.originNear.xy) + int(axis / 2u);
        if (any(parentPage < 0) || any(parentPage >= int(axis))) { continue; }
        // The half-texel containment inset cuts through outer page interiors.
        // Preserve that domain with the original search on these pages.
        if (any(parentPage == 0) || any(parentPage == int(axis) - 1)) { return; }
        uint4 parentEntry = pages[parent.page.x + uint(parentPage.y) * axis + uint(parentPage.x)];
        if (parentEntry.x != 0xffffffffu && (parentEntry.y & 2u) != 0u && (parentEntry.w & 1u) != 0u) {
            physicalPage = parentEntry.x;
            address = vsmPackDirectionalMappingAddress(next, physicalPage, push.physicalPagesX);
            packedParentPage = uint(parentPage.x) | (uint(parentPage.y) << 16u);
            break;
        }
    }
    pages[push.totalPages + id.x] = uint4(physicalPage, address, packedParentPage, 2u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmPropagateMappingMain(uint3 id : SV_DispatchThreadID,
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
#if VSM_STATIC_PAGE_HZB
    if (id.x == 0u) {
        // Finalize has completed all appends. Keep its count intact for the
        // rounded final row, and publish legal two-dimensional dispatch args.
        RWStructuredBuffer<uint> jobs = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, 38u)));
        uint count = jobs[0];
        uint offset = 3u + push.physicalPageCount;
        jobs[offset] = min(count, 65535u);
        jobs[offset + 1u] = max(1u, (count + 65534u) / 65535u);
        jobs[offset + 2u] = 1u;
    }
#endif
    vsmBuildPropagatedMapping(id, push);
}

