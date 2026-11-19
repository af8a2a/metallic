#language slang 2026
implementing VirtualShadowMaps;

// Each slot stores a reversed-Z minimum hierarchy: 1, 2x2, 4x4 and 8x8.
// Static and composite raster reuse it sequentially, as do history and retest.
uint vsmRasterHZBPageCapacity(ShadowPush push)
{
    return min(push.totalPages - 1u, push.physicalPageCount);
}

uint vsmRasterHZBPageBatches(ShadowPush push)
{
    return (vsmRasterHZBPageCapacity(push) + 65534u) / 65535u;
}

uint vsmRasterHZBPageArgument(uint layer, uint batch, ShadowPush push)
{
    return vsmCacheMetadata(push, kVSMHybridDrawOffset) + 6u * push.batchCount + layer * vsmRasterHZBPageBatches(push) + batch;
}

uint vsmRasterHZBPageByteOffset(uint layer, uint index, ShadowPush push)
{
    return vsmHardwareTicketByteOffset(2u * push.workCapacity + layer * vsmRasterHZBPageCapacity(push) + index, push);
}

void vsmQueueRasterHZBPage(uint virtualPage, uint layer, ShadowPush push)
{
    RWStructuredBuffer<uint> counters = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.sampleReserved));
    uint index;
    InterlockedAdd(counters[kVSMHZBPageCount + layer], 1u, index);
    RWByteAddressBuffer work = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.workItems));
    work.Store(vsmRasterHZBPageByteOffset(layer, index, push), virtualPage);
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    InterlockedAdd(arguments[vsmRasterHZBPageArgument(layer, index / 65535u, push) * 3u], 1u);
}

bool vsmHasRasterHZBHistory(uint virtualPage, uint4 page, ShadowProjection projection, ShadowPush push)
{
    uint axis = projection.page.y;
    uint local = virtualPage - projection.page.x;
    StructuredBuffer<uint4> changes = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.reuse));
    int2 previousXY = int2(local % axis, local / axis) + asint(changes[page.z].xy);
    if (any(uint2(previousXY) >= axis)) { return false; }
    StructuredBuffer<uint4> previous = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.previousPages));
    uint4 old = previous[projection.page.x + uint(previousXY.y) * axis + uint(previousXY.x)];
    uint valid = (push.rasterCoarse & 2u) != 0u ? 4u : 1u;
    return old.x == page.x && (old.w & valid) != 0u && vsmRetainsSharedPage(old.x, push);
}

groupshared float gVsmRasterHZBMin[64];

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmBuildRasterHZBMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
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
    uint layer = (push.rasterCoarse >> 1u) & 1u;
    bool current = (push.rasterCoarse & 8u) != 0u;
    if (current && vsmCacheMetadata(push, kVSMHZBDeferredCount + layer) == 0u
#if VSM_HIERARCHY_HZB
        && vsmCacheMetadata(push, kVSMHierarchyHZBDeferredCount) == 0u
#endif
    ) { return; }
    ByteAddressBuffer work = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.workItems));
    uint virtualPage = work.Load(vsmRasterHZBPageByteOffset(layer, push.workBase + group.x, push));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    uint4 page = pages[virtualPage];

    uint base = page.x * kVSMRasterHZBWordsPerSlot;
    RWStructuredBuffer<float> hzb = resolveUniform(ResourceHandle<RWStructuredBuffer<float>>(vsmCacheMetadata(push, kVSMRasterHZB)));
    if (!current) {
        StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
        if (!vsmHasRasterHZBHistory(virtualPage, page, projections[page.z], push)) {
            // An acquired/recycled slot has no depth from this face and world page.
            hzb[base + lane] = 0.0;
            if (lane < 21u) { hzb[base + 64u + lane] = 0.0; }
            return;
        }
    }

    VSMAtlas atlas = resolveUniform(ResourceHandle<VSMAtlas>(layer != 0u ? vsmCacheMetadata(push, 32u) : push.atlas));
    uint2 origin = vsmPhysicalPageCoord(page.x, push.physicalPagesX) * push.pageSize;
    uint2 xy = uint2(lane & 7u, lane >> 3u);
    uint2 first = xy * push.pageSize / 8u;
    // For small pages several cells refer to the same texel.
    uint extent = max(1u, push.pageSize / 8u);
    float furthest = 1.0;
    for (uint y = 0u; y < extent; ++y) {
        for (uint x = 0u; x < extent; ++x) {
            furthest = min(furthest, vsmAtlasLoad(atlas, origin + first + uint2(x, y), push));
        }
    }
    gVsmRasterHZBMin[lane] = furthest;
    hzb[base + 21u + lane] = furthest;
    for (uint axis = 4u, stride = 1u; axis != 0u; axis >>= 1u, stride <<= 1u) {
        GroupMemoryBarrierWithGroupSync();
        if (lane < axis * axis) {
            uint child = 2u * stride * ((lane / axis) * 8u + lane % axis);
            uint row = 8u * stride;
            float value = min(min(gVsmRasterHZBMin[child], gVsmRasterHZBMin[child + stride]),
                min(gVsmRasterHZBMin[child + row], gVsmRasterHZBMin[child + row + stride]));
            gVsmRasterHZBMin[child] = value;
            hzb[base + (axis * axis - 1u) / 3u + lane] = value;
        }
    }
}

float2 vsmRasterHZBDivideRange(float2 numerator, float2 w)
{
    float2 a = numerator / w.x, b = numerator / w.y;
    return float2(min(a.x, b.x), max(a.y, b.y));
}

bool vsmRasterBoundsOccluded(float3 minimum, float3 maximum, uint virtualPage, uint slot,
    ShadowProjection projection, ShadowPush push)
{
    float4 bounds = vsmPageBounds(virtualPage, projection);
    if (projection.metric.w < 0.5) { bounds = vsmDirectionalPageBounds(virtualPage, projection); }
    float2 x = vsmHZBProjectedRange(minimum, maximum, projection.rowX);
    float2 y = vsmHZBProjectedRange(minimum, maximum, projection.rowY);
    float2 z = vsmHZBProjectedRange(minimum, maximum, projection.rowZ);
    if (projection.metric.w > 0.5) {
        // Independent clip-space intervals conservatively bound z/w even when
        // the nearest corner differs per component. Never reject near-plane crossings.
        float2 w = vsmHZBProjectedRange(minimum, maximum, projection.rowW);
        if (w.x <= 0.0 || z.x < 0.0 || z.y >= w.x) { return false; }
        x = vsmRasterHZBDivideRange(x, w);
        y = vsmRasterHZBDivideRange(y, w);
        z.y /= w.x;
    }
    if (z.y < 0.0 || z.y >= 1.0) { return false; }
    float2 scale = float(push.pageSize) / (bounds.zw - bounds.xy);
    float2 lo = floor((float2(x.x, y.x) - bounds.xy) * scale - 1.0);
    float2 hi = ceil((float2(x.y, y.y) - bounds.xy) * scale + 1.0);
    if (!all(isfinite(lo)) || !all(isfinite(hi)) || !isfinite(z.y)) { return false; }
    float last = float(push.pageSize - 1u);
    if (any(lo > last) || any(hi < 0.0)) { return false; }
    uint2 a = uint2(clamp(lo, 0.0, last)) * 8u / push.pageSize;
    uint2 b = uint2(clamp(hi, 0.0, last)) * 8u / push.pageSize;
    uint span = max(b.x - a.x, b.y - a.y);
    uint shift = span == 0u ? 0u : uint(firstbithigh(span)) + 1u;
    uint axis = 8u >> shift;
    a >>= shift;
    b >>= shift;
    uint start = slot * kVSMRasterHZBWordsPerSlot + (axis * axis - 1u) / 3u;
    StructuredBuffer<float> hzb = resolveUniform(ResourceHandle<StructuredBuffer<float>>(vsmCacheMetadata(push, kVSMRasterHZB)));
    float furthest = min(min(hzb[start + a.y * axis + a.x], hzb[start + a.y * axis + b.x]),
        min(hzb[start + b.y * axis + a.x], hzb[start + b.y * axis + b.x]));
    return furthest > z.y;
}

bool vsmRasterTicketOccluded(uint4 ticket, uint4 page, ShadowPush push)
{
    uint valid = (push.rasterCoarse & 2u) != 0u ? 4u : 1u;
    if ((push.cacheMode & 1u) == 0u || (page.w & valid) == 0u || !vsmRasterClearsEntry(page, push.rasterCoarse)) { return false; }
    float3 minimum, maximum;
    if (!loadVirtualShadowMapClusterBounds(push.geometry, ticket.x, minimum, maximum)) { return false; }
    StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
    return vsmRasterBoundsOccluded(minimum, maximum, ticket.y, page.x, projections[page.z], push);
}
