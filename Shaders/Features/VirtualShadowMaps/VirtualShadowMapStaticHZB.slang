#language slang 2026
implementing VirtualShadowMaps;

#ifndef VSM_SHARED_PHYSICAL_POOL
#define VSM_SHARED_PHYSICAL_POOL 0
#endif

// Each physical slot stores the minimum (furthest reversed-Z) of every 2x2
// static depth footprint, up to the page root. Store coarse mips first so the
// start of a square level depends only on its side length.
uint vsmStaticHZBLevelOffset(uint axis)
{
    return (axis * axis - 1u) / 3u;
}

uint vsmStaticHZBPageOffset(uint physical, uint pageSize)
{
#if VSM_SHARED_PHYSICAL_POOL
    physical &= 4095u;
#endif
    return physical * max(1u, vsmStaticHZBLevelOffset(pageSize));
}

uint vsmStaticHZBHandle(ShadowPush push, uint physical)
{
    uint handle = vsmCacheMetadata(push, 37u);
#if VSM_SHARED_PHYSICAL_POOL
    StructuredBuffer<uint> banks = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(handle));
    return banks[physical >> 12u];
#else
    return handle;
#endif
}

bool vsmStaticHZBOccludes(ShadowPush push, uint physical, uint2 firstTexel, uint2 lastTexel, float closestDepth)
{
    uint2 delta = lastTexel - firstTexel;
    uint shift = 1u + uint(firstbithigh(max(1u, max(delta.x, delta.y))));
    uint axis = max(1u, push.pageSize >> shift);
    uint2 a = firstTexel >> shift, b = lastTexel >> shift;
    uint start = vsmStaticHZBPageOffset(physical, push.pageSize) + vsmStaticHZBLevelOffset(axis);
#if VSM_SHARED_PHYSICAL_POOL
    StructuredBuffer<float> hzb = resolveNonUniform(ResourceHandle<StructuredBuffer<float>>(vsmStaticHZBHandle(push, physical)));
#else
    StructuredBuffer<float> hzb = resolveUniform(ResourceHandle<StructuredBuffer<float>>(vsmStaticHZBHandle(push, physical)));
#endif
    // A 2^shift cell covers at least the query width and height. The query
    // intersects at most two cells per axis. Clear texels lower the min to 0.
    float furthest = min(min(hzb[start + a.y * axis + a.x], hzb[start + a.y * axis + b.x]),
        min(hzb[start + b.y * axis + a.x], hzb[start + b.y * axis + b.x]));
    return furthest > closestDepth;
}

float vsmStaticHZBMin2x2(VSMAtlas atlas, uint2 p, ShadowPush push)
{
    return min(min(vsmAtlasLoad(atlas, p, push), vsmAtlasLoad(atlas, p + uint2(1, 0), push)),
        min(vsmAtlasLoad(atlas, p + uint2(0, 1), push), vsmAtlasLoad(atlas, p + 1u, push)));
}

// A 128x128 page contains at most 32x32 four-texel-wide tiles.
groupshared float gVsmStaticHZBTileMin[1024];

[shader("compute")]
[numthreads(256, 1, 1)]
void vsmBuildStaticPageHZBMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ShadowPush push)
{
    StructuredBuffer<uint> jobs = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(vsmCacheMetadata(push, 38u)));
    uint job = group.x + group.y * 65535u;
    if (job >= jobs[0]) { return; }
    uint physical = jobs[3u + job];
    uint2 origin = uint2(physical % push.physicalPagesX, physical / push.physicalPagesX) * push.pageSize;
    uint start = vsmStaticHZBPageOffset(physical, push.pageSize);
    VSMAtlas atlas = resolveUniform(ResourceHandle<VSMAtlas>(vsmCacheMetadata(push, 32u)));
#if VSM_SHARED_PHYSICAL_POOL
    RWStructuredBuffer<float> hzb = resolveNonUniform(ResourceHandle<RWStructuredBuffer<float>>(vsmStaticHZBHandle(push, physical)));
#else
    RWStructuredBuffer<float> hzb = resolveUniform(ResourceHandle<RWStructuredBuffer<float>>(vsmStaticHZBHandle(push, physical)));
#endif
#if VSM_WORKLOAD_DIAGNOSTICS
    if (lane == 0u) { vsmCacheAdd(push, kVSMCacheStaticHZBBuiltPages, 1u); }
#endif
    if (push.pageSize <= 2u) {
        if (lane == 0u) {
            hzb[start] = push.pageSize == 1u ? vsmAtlasLoad(atlas, origin, push) : vsmStaticHZBMin2x2(atlas, origin, push);
        }
        return;
    }
    uint tiles = push.pageSize >> 2u;
    uint baseAxis = tiles * 2u;
    uint base = start + vsmStaticHZBLevelOffset(baseAxis);
    for (uint tile = lane; tile < tiles * tiles; tile += 256u) {
        uint2 xy = uint2(tile % tiles, tile / tiles);
        uint2 p = origin + 4u * xy;
        float a = vsmStaticHZBMin2x2(atlas, p, push);
        float b = vsmStaticHZBMin2x2(atlas, p + uint2(2, 0), push);
        float c = vsmStaticHZBMin2x2(atlas, p + uint2(0, 2), push);
        float d = vsmStaticHZBMin2x2(atlas, p + 2u, push);
        uint dst = base + 2u * (xy.y * baseAxis + xy.x);
        hzb[dst] = a; hzb[dst + 1u] = b;
        hzb[dst + baseAxis] = c; hzb[dst + baseAxis + 1u] = d;
        float furthest = min(min(a, b), min(c, d));
        gVsmStaticHZBTileMin[tile] = furthest;
        hzb[start + vsmStaticHZBLevelOffset(tiles) + tile] = furthest;
    }
    for (uint axis = tiles >> 1u, stride = 1u; axis > 0u; axis >>= 1u, stride <<= 1u) {
        GroupMemoryBarrierWithGroupSync();
        // Retain each parent's minimum in the first child's original slot.
        // Neighbouring reduction blocks never overwrite another block's input.
        for (uint cell = lane; cell < axis * axis; cell += 256u) {
            uint src = 2u * stride * ((cell / axis) * tiles + cell % axis);
            uint row = stride * tiles;
            float furthest = min(min(gVsmStaticHZBTileMin[src], gVsmStaticHZBTileMin[src + stride]),
                min(gVsmStaticHZBTileMin[src + row], gVsmStaticHZBTileMin[src + row + stride]));
            gVsmStaticHZBTileMin[src] = furthest;
            hzb[start + vsmStaticHZBLevelOffset(axis) + cell] = furthest;
        }
    }
}
