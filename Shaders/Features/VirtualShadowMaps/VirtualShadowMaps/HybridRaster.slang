#language slang 2026
implementing VirtualShadowMaps;

#ifndef VSM_BATCHED_BUILD_RASTER
#define VSM_BATCHED_BUILD_RASTER 0
#endif
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif
#ifndef VSM_CLUSTER_HYBRID_RASTER
#define VSM_CLUSTER_HYBRID_RASTER 0
#endif

// Packed tickets stay immutable. Hybrid HW entries carry a second word for
// the quarter's hardware lanes. Deferred entries still refer to whole tickets.
uint vsmHardwareTicketByteOffset(uint index, ShadowPush push)
{
    return (1u + push.workCapacity) * 16u + index * (VSM_CLUSTER_HYBRID_RASTER ? 8u : 4u);
}

uint vsmDeferredRasterTicketByteOffset(uint index, ShadowPush push)
{
    return vsmHardwareTicketByteOffset(push.workCapacity, push) + index * 4u;
}

static const uint kVSMMainDraw = 0u, kVSMDeferredRetest = 1u, kVSMRecoveredDraw = 2u;

void vsmQueueRasterTicket(ShadowPush push, uint ticketIndex, uint stage, uint hardwareMask = 0xffffu)
{
    uint layer = (push.rasterCoarse >> 1u) & 1u;
    uint index;
#if VSM_TWO_PASS_HZB
    if (stage != kVSMMainDraw) {
        RWStructuredBuffer<uint> counters = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.sampleReserved));
        uint counter = stage == kVSMDeferredRetest ? kVSMHZBDeferredCount : kVSMHZBPostCount;
        InterlockedAdd(counters[counter + layer], 1u, index);
    } else
#endif
    {
        RWStructuredBuffer<uint4> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.workItems));
        if (layer != 0u) { InterlockedAdd(work[0].z, 1u, index); }
        else { InterlockedAdd(work[0].y, 1u, index); }
    }
    RWByteAddressBuffer queue = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.workItems));
#if VSM_TWO_PASS_HZB
    if (stage == kVSMDeferredRetest) {
        queue.Store(vsmDeferredRasterTicketByteOffset(index, push), ticketIndex);
    } else
#endif
    {
#if VSM_CLUSTER_HYBRID_RASTER
        queue.Store2(vsmHardwareTicketByteOffset(index, push), uint2(ticketIndex, hardwareMask));
#else
        queue.Store(vsmHardwareTicketByteOffset(index, push), ticketIndex);
#endif
    }
    uint batch = vsmCacheMetadata(push, kVSMHybridDrawOffset) + (2u * stage + layer) * push.batchCount + index / 65535u;
    RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
    InterlockedAdd(arguments[batch * 3u], 1u);
}

#if VSM_CLUSTER_HYBRID_RASTER
uint vsmHybridDepthAddress(uint slot, uint2 texel, ShadowPush push)
{
    return (slot * push.pageSize + texel.y) * push.pageSize + texel.x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmHybridClearMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush clearPush)
{
#if VSM_BATCHED_BUILD_RASTER
    ByteAddressBuffer poolBytes = resolveUniform(ResourceHandle<ByteAddressBuffer>(clearPush.physicalPool));
    uint2 poolHeader = poolBytes.Load2(0u);
    uint staticLayer = (clearPush.rasterCoarse & 2u) != 0u ? 1u : 0u;
    uint listByte = vsmSharedPoolClearListQuad(poolHeader.x, poolHeader.y) * 16u +
        (staticLayer * poolHeader.y + clearPush.workBase + group.x) * 4u;
    uint slot = poolBytes.Load(listByte);
#else
    uint slot = clearPush.workBase + group.x;
#endif
    if (slot >= vsmCacheMetadata(clearPush, kVSMHybridPageCount)) { return; }
    ShadowPush push;
    if (!vsmRasterClearsPage(slot, clearPush, push)) { return; }
    // The owning face may use only hardware raster. Clear the shared scratch.
    RWStructuredBuffer<uint> depth = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(clearPush, kVSMHybridDepth)));
    uint pageArea = clearPush.pageSize * clearPush.pageSize;
    for (uint i = lane; i < pageArea; i += 64u) { depth[slot * pageArea + i] = 0u; }
}

struct VSMSoftwareVertex
{
    int2 subpixel;
    float depth;
};

VSMSoftwareVertex vsmSoftwareVertex(ShadowClipVertex vertex, float4 bounds, uint slot, ShadowPush push, int scale)
{
    // Match the mesh output and the positive-height Vulkan viewport before
    // snapping. Subtract the physical origin afterwards to bound edge products.
    float4 clip = vsmRasterVertex(vertex, bounds, slot, 0u, push).position;
    uint2 grid = uint2(push.physicalPagesX, push.physicalPagesY);
    float2 halfViewport = 0.5 * float2(grid * push.pageSize);
    float2 screen = mad(clip.xy / clip.w, halfViewport, halfViewport);
    uint2 page = vsmPhysicalPageCoord(slot, push.physicalPagesX);
    page.y %= push.physicalPagesY;
    VSMSoftwareVertex result;
    result.subpixel = int2(round(screen * scale)) - int2(page * push.pageSize) * scale;
    result.depth = clip.z / clip.w;
    return result;
}

int vsmSoftwareEdge(int2 a, int2 b, int2 p)
{
    int2 delta = b - a, offset = p - a;
    return delta.x * offset.y - delta.y * offset.x;
}

int vsmSoftwareTopLeftBias(int2 a, int2 b)
{
    return b.y < a.y || (b.y == a.y && b.x > a.x) ? 0 : -1;
}

uint vsmSoftwareTriangle(VSMSoftwareVertex a, VSMSoftwareVertex b, VSMSoftwareVertex c,
    bool doubleSided, uint slot, ShadowPush push, int scale, RWStructuredBuffer<uint> software)
{
    int area = vsmSoftwareEdge(a.subpixel, b.subpixel, c.subpixel);
    // CCW front faces have negative signed area with our Vulkan viewport.
    if (area == 0 || (!doubleSided && area > 0)) { return 0u; }
    if (area < 0) {
        VSMSoftwareVertex tmp = b; b = c; c = tmp;
        area = -area;
    }
    int2 lo = max(int2(0), int2(ceil(float2(min(a.subpixel, min(b.subpixel, c.subpixel))) / scale - 0.5)));
    int2 hi = min(int2(push.pageSize) - 1, int2(floor(float2(max(a.subpixel, max(b.subpixel, c.subpixel))) / scale - 0.5)));
    int3 bias = int3(vsmSoftwareTopLeftBias(b.subpixel, c.subpixel), vsmSoftwareTopLeftBias(c.subpixel, a.subpixel),
        vsmSoftwareTopLeftBias(a.subpixel, b.subpixel));
    uint covered = 0u;
    for (int y = lo.y; y <= hi.y; ++y) {
        for (int x = lo.x; x <= hi.x; ++x) {
            int2 center = int2(x, y) * scale + scale / 2;
            int3 edge = int3(vsmSoftwareEdge(b.subpixel, c.subpixel, center), vsmSoftwareEdge(c.subpixel, a.subpixel, center),
                vsmSoftwareEdge(a.subpixel, b.subpixel, center));
            if (any(edge + bias < 0)) { continue; }
            float z = a.depth + (float(edge.y) * (b.depth - a.depth) + float(edge.z) * (c.depth - a.depth)) / float(area);
            InterlockedMax(software[vsmHybridDepthAddress(slot, uint2(x, y), push)], asuint(saturate(z)) & 0x7fffffffu);
            ++covered;
        }
    }
    return covered;
}

groupshared uint gVsmHybridCandidate;
groupshared uint gVsmHybridHardwareMask;
#if VSM_WORKLOAD_DIAGNOSTICS
groupshared uint gVsmHybridSoftwareMask;

void vsmRecordHybridRoutes(ShadowPush push, uint softwareMask, uint hardwareMask)
{
    if (softwareMask != 0u) {
        vsmCacheAdd(push, kVSMHybridSoftwareTickets, 1u);
        vsmCacheAdd(push, kVSMHybridSoftwareTriangles, countbits(softwareMask));
    }
    if (hardwareMask != 0u) {
        vsmCacheAdd(push, kVSMHybridHardwareTickets, 1u);
        vsmCacheAdd(push, kVSMHybridHardwareTriangles, countbits(hardwareMask));
        if (softwareMask != 0u) { vsmCacheAdd(push, kVSMHybridMixedTickets, 1u); }
    }
}
#endif
#endif

[shader("compute")]
[numthreads(32, 1, 1)]
void vsmHybridRasterMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex,
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
    StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
    uint ticketIndex = 1u + push.workBase + group.x;
    uint4 ticket = work[ticketIndex];
    if (lane == 0u) {
        StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
        uint4 page = pages[ticket.y];
#if VSM_TWO_PASS_HZB
        if (vsmRasterTicketOccluded(ticket, page, push)) {
            vsmQueueRasterTicket(push, ticketIndex, kVSMDeferredRetest);
#if VSM_WORKLOAD_DIAGNOSTICS
            vsmCacheAdd(push, kVSMHZBDeferredTickets, 1u);
#endif
#if VSM_CLUSTER_HYBRID_RASTER
            gVsmHybridCandidate = 2u;
#endif
        } else
#endif
        {
#if VSM_CLUSTER_HYBRID_RASTER
            gVsmHybridCandidate = uint(vsmRasterClearsEntry(page, push.rasterCoarse) &&
                page.x < vsmCacheMetadata(push, kVSMHybridPageCount));
            gVsmHybridHardwareMask = 0u;
#if VSM_WORKLOAD_DIAGNOSTICS
            gVsmHybridSoftwareMask = 0u;
#endif
#else
            vsmQueueRasterTicket(push, ticketIndex, kVSMMainDraw);
#endif
        }
    }
#if VSM_CLUSTER_HYBRID_RASTER
    GroupMemoryBarrierWithGroupSync();
    if (gVsmHybridCandidate == 2u) { return; }
    uint bits = ticket.z != 0u ? ticket.z : ticket.w;
    uint quarter = (ticket.z != 0u ? 0u : 32u) + ((bits & 0xffffu) != 0u ? 0u : 16u);
    uint slotMask = (bits >> (quarter & 31u)) & 0xffffu;
    if (gVsmHybridCandidate == 0u) {
        if (lane == 0u) {
            vsmQueueRasterTicket(push, ticketIndex, kVSMMainDraw, slotMask);
#if VSM_WORKLOAD_DIAGNOSTICS
            vsmRecordHybridRoutes(push, 0u, slotMask);
#endif
        }
        return;
    }
    uint laneBit = 1u << lane;
    ShadowClipVertex polygon[9];
    uint count = 0u, material = 0u, slot = 0u;
    float4 bounds;
    if ((slotMask & laneBit) != 0u) {
        count = vsmRasterPolygon(ticket.x + quarter + lane, ticket.y, push, true, polygon, material, slot, bounds, true);
    }
    bool softwareRoute = count >= 3u && count != kVSMHardwareOnlyPolygon;
    uint hardware = WaveActiveBitOr(count == kVSMHardwareOnlyPolygon ? laneBit : 0u);
#if VSM_WORKLOAD_DIAGNOSTICS
    uint softwareLanes = WaveActiveBitOr(softwareRoute ? laneBit : 0u);
#endif
    if (WaveIsFirstLane()) {
        InterlockedOr(gVsmHybridHardwareMask, hardware);
#if VSM_WORKLOAD_DIAGNOSTICS
        InterlockedOr(gVsmHybridSoftwareMask, softwareLanes);
#endif
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) {
        if (gVsmHybridHardwareMask != 0u) {
            vsmQueueRasterTicket(push, ticketIndex, kVSMMainDraw, gVsmHybridHardwareMask);
        }
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmRecordHybridRoutes(push, gVsmHybridSoftwareMask, gVsmHybridHardwareMask);
#endif
    }
    if (!softwareRoute) { return; }
    int scale = 1 << vsmCacheMetadata(push, kVSMHybridSubpixelBits);
    RWStructuredBuffer<uint> software = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, kVSMHybridDepth)));
    StructuredBuffer<ShadowMaterial> materials = resolveUniform(ResourceHandle<StructuredBuffer<ShadowMaterial>>(push.materials));
    bool doubleSided = materials[material].resources.y != 0u;
    VSMSoftwareVertex a = vsmSoftwareVertex(polygon[0], bounds, slot, push, scale);
    VSMSoftwareVertex b = vsmSoftwareVertex(polygon[1], bounds, slot, push, scale);
    uint covered = 0u;
    for (uint i = 2u; i < count; ++i) {
        VSMSoftwareVertex c = vsmSoftwareVertex(polygon[i], bounds, slot, push, scale);
        covered += vsmSoftwareTriangle(a, b, c, doubleSided, slot, push, scale, software);
        b = c;
    }
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmCacheAdd(push, kVSMHybridCoveredSamples, covered);
#endif
#endif
}

#if VSM_TWO_PASS_HZB
[shader("compute")]
[numthreads(1, 1, 1)]
void vsmRetestRasterHZBMain(uint3 group : SV_GroupID,
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
    ByteAddressBuffer queue = resolveUniform(ResourceHandle<ByteAddressBuffer>(push.workItems));
    uint ticketIndex = queue.Load(vsmDeferredRasterTicketByteOffset(push.workBase + group.x, push));
    StructuredBuffer<uint4> work = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.workItems));
    StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
    uint4 ticket = work[ticketIndex];
    bool occluded = vsmRasterTicketOccluded(ticket, pages[ticket.y], push);
    if (!occluded) { vsmQueueRasterTicket(push, ticketIndex, kVSMRecoveredDraw); }
#if VSM_WORKLOAD_DIAGNOSTICS
    vsmCacheAdd(push, occluded ? kVSMHZBRejectedTickets : kVSMHZBRecoveredTickets, 1u);
#if VSM_CLUSTER_HYBRID_RASTER
    if (!occluded) { vsmRecordHybridRoutes(push, 0u, ticket.z | ticket.w); }
#endif
#endif
}
#endif
