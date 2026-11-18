#language slang 2026
implementing VirtualShadowMaps;

#ifndef VSM_BATCHED_BUILD_RASTER
#define VSM_BATCHED_BUILD_RASTER 0
#endif
#ifndef VSM_WORKLOAD_DIAGNOSTICS
#define VSM_WORKLOAD_DIAGNOSTICS 0
#endif

uint vsmHybridDepthAddress(uint slot, uint2 texel, ShadowPush push)
{
    return (slot * push.pageSize + texel.y) * push.pageSize + texel.x;
}

// Packed tickets stay immutable until both raster layers finish. Each layer
// reuses the uint queue directly following the last allocated uint4 ticket.
uint vsmHardwareTicketByteOffset(uint index, ShadowPush push)
{
    return (1u + push.workCapacity) * 16u + index * 4u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vsmHybridClearMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush clearPush)
{
    uint slot = clearPush.workBase + group.x;
    if (slot >= vsmCacheMetadata(clearPush, kVSMHybridPageCount)) { return; }
    ShadowPush push;
    if (!vsmRasterClearsPage(slot, clearPush, push)) { return; }
    RWStructuredBuffer<uint> depth = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, kVSMHybridDepth)));
    uint pageArea = push.pageSize * push.pageSize;
    for (uint i = lane; i < pageArea; i += 64u) { depth[slot * pageArea + i] = 0u; }
}

// A Build ticket covers part of a 64-slot caster cluster. Use the original
// cluster's extent in this shadow view so splitting a ticket at a page seam
// cannot turn a large cluster into a software microtriangle candidate.
bool vsmHybridClusterFits(uint firstTriangle, ShadowProjection projection, ShadowPush push)
{
    float3 minimum, maximum;
    if (!loadVirtualShadowMapClusterBounds(push.geometry, firstTriangle, minimum, maximum)) { return false; }
    float2 lo = float2(3.402823466e38), hi = -lo;
    for (uint i = 0u; i < 8u; ++i) {
        float3 corner = float3((i & 1u) != 0u ? maximum.x : minimum.x, (i & 2u) != 0u ? maximum.y : minimum.y,
            (i & 4u) != 0u ? maximum.z : minimum.z);
        float4 clip = vsmProjectClip(corner, projection);
        if (clip.w <= 0.0 || clip.z < 0.0 || clip.z > clip.w) { return false; }
        float2 ndc = clip.xy / clip.w;
        lo = min(lo, ndc); hi = max(hi, ndc);
    }
    float2 texelExtent = (hi - lo) * (0.5 * float(projection.page.y * push.pageSize));
    return dot(texelExtent, texelExtent) <= 32.0 * 32.0;
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
groupshared uint gVsmHybridMasked;

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
    RWStructuredBuffer<uint4> work = resolveUniform(ResourceHandle<RWStructuredBuffer<uint4>>(push.workItems));
    uint ticketIndex = 1u + push.workBase + group.x;
    uint4 ticket = work[ticketIndex];
    uint firstTriangle = ticket.x;
    if (lane == 0u) {
        StructuredBuffer<uint4> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint4>>(push.pages));
        uint4 page = pages[ticket.y];
        bool eligible = false;
        if (vsmRasterClearsEntry(page, push.rasterCoarse) && page.x < vsmCacheMetadata(push, kVSMHybridPageCount)) {
            StructuredBuffer<ShadowProjection> projections = resolveUniform(ResourceHandle<StructuredBuffer<ShadowProjection>>(push.projections));
            eligible = vsmHybridClusterFits(firstTriangle, projections[page.z], push);
        }
        gVsmHybridCandidate = uint(eligible);
        gVsmHybridMasked = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint bits = ticket.z != 0u ? ticket.z : ticket.w;
    uint quarter = (ticket.z != 0u ? 0u : 32u) + ((bits & 0xffffu) != 0u ? 0u : 16u);
    bool active = lane < 16u && (((bits >> (quarter & 31u)) >> lane) & 1u) != 0u;
    uint triangleIndex = firstTriangle + quarter + lane;
    StructuredBuffer<ShadowMaterial> materials = resolveUniform(ResourceHandle<StructuredBuffer<ShadowMaterial>>(push.materials));
    ShadowClipVertex polygon[9];
    uint count = 0u, material = 0u, slot = 0u;
    float4 bounds;
    if (gVsmHybridCandidate != 0u && active) {
        count = vsmRasterPolygon(triangleIndex, ticket.y, push, true, polygon, material, slot, bounds, false);
        if (count >= 3u && materials[material].alpha.z >= 0.5) {
            InterlockedOr(gVsmHybridMasked, 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    bool softwareRoute = gVsmHybridCandidate != 0u && gVsmHybridMasked == 0u;
    if (lane == 0u) {
        if (!softwareRoute) {
            uint index;
            bool staticPass = (push.rasterCoarse & 2u) != 0u;
            if (staticPass) { InterlockedAdd(work[0].z, 1u, index); }
            else { InterlockedAdd(work[0].y, 1u, index); }
            RWByteAddressBuffer queue = resolveUniform(ResourceHandle<RWByteAddressBuffer>(push.workItems));
            queue.Store(vsmHardwareTicketByteOffset(index, push), ticketIndex);
            uint batch = vsmCacheMetadata(push, kVSMHybridDrawOffset) + (staticPass ? push.batchCount : 0u) + index / 65535u;
            RWStructuredBuffer<uint> arguments = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.arguments));
            InterlockedAdd(arguments[batch * 3u], 1u);
        }
#if VSM_WORKLOAD_DIAGNOSTICS
        vsmCacheAdd(push, softwareRoute ? kVSMHybridSoftwareTickets : kVSMHybridHardwareTickets, 1u);
#endif
    }
    if (!softwareRoute || count < 3u) { return; }
    int scale = 1 << vsmCacheMetadata(push, kVSMHybridSubpixelBits);
    RWStructuredBuffer<uint> software = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(vsmCacheMetadata(push, kVSMHybridDepth)));
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
}
