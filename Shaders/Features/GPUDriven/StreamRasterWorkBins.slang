// Local to a 128-thread cluster. The payload retains the original triangle
// ordinal: bucket order must never become visibility/material identity.
groupshared uint g_StreamRasterWorkCounts[16];
groupshared uint g_StreamRasterWorkOrder[128];

uint streamRasterWorkBucket(int2 p0, int2 p1, int2 p2, bool doubleSided,
    uint width, uint height, uint bits)
{
    int scale = 1 << bits;
    int2 d = p1 - p0, q = p2 - p0;
    int area = d.x * q.y - d.y * q.x;
    if (area == 0 || (!doubleSided && area > 0)) { return 3u; }
    int2 lower = max(int2(0), int2(ceil(float2(min(p0, min(p1, p2))) / scale - 0.5)));
    int2 upper = min(int2(width, height) - 1,
        int2(floor(float2(max(p0, max(p1, p2))) / scale - 0.5)));
    if (any(lower > upper)) { return 3u; }
    uint2 extent = uint2(upper - lower + 1);
    // Use division rather than a potentially overflowing area product. Rows
    // distinguish tall/narrow scans from short/wide boxes of equal area.
    if (extent.y <= 2u && extent.x <= 4u / extent.y) { return 0u; }
    if (extent.y <= 4u && extent.x <= 16u / extent.y) { return 1u; }
    return 2u;
}

uint streamRasterWorkBucket(float4 a, float4 b, float4 c, bool doubleSided,
    uint width, uint height, uint bits)
{
    float2 viewport = float2(width, height);
    return streamRasterWorkBucket(hybridPrepareVertex(a, viewport, bits).position,
        hybridPrepareVertex(b, viewport, bits).position, hybridPrepareVertex(c, viewport, bits).position,
        doubleSided, width, height, bits);
}

// All 128 lanes participate, including unused triangles (bucket 3). Stable
// rank inside each bucket uses wave prefix + preceding-wave population.
uint streamRasterReorderWork(uint bucket, uint lane)
{
    uint waveSize = WaveGetLaneCount();
    uint wave = lane / waveSize;
    uint waves = 128u / waveSize;
    uint rank = 0u;
    [unroll] for (uint bin = 0u; bin < 4u; ++bin) {
        bool match = bucket == bin;
        uint prefix = WavePrefixCountBits(match);
        uint count = WaveActiveCountBits(match);
        if (match) { rank = prefix; }
        if (WaveIsFirstLane()) { g_StreamRasterWorkCounts[bin * waves + wave] = count; }
    }
    GroupMemoryBarrierWithGroupSync();
    uint offset = rank;
    for (uint i = 0u; i < bucket * waves + wave; ++i) { offset += g_StreamRasterWorkCounts[i]; }
    g_StreamRasterWorkOrder[offset] = (bucket << 7u) | lane;
    GroupMemoryBarrierWithGroupSync();
    return g_StreamRasterWorkOrder[lane];
}
