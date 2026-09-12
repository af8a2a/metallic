#ifndef HYBRID_RASTER_COMMON_SLANG
#define HYBRID_RASTER_COMMON_SLANG

// Header: count/capacity/width/height, then immutable width/height/maxPixels/capacity.
// Each following 64-byte entry contains three clip positions and a visibility ID.
// The queue is bounded. A failed reservation must keep the triangle in HW raster.
bool hybridTriangleFits(float4 a, float4 b, float4 c, float2 viewport, float maxPixels)
{
    if (!all(isfinite(a)) || !all(isfinite(b)) || !all(isfinite(c)) ||
        min(a.w, min(b.w, c.w)) <= 0.000001 ||
        min(a.z, min(b.z, c.z)) < 0.0 ||
        min(a.w - a.z, min(b.w - b.z, c.w - c.z)) < 0.0) { return false; }
    float2 p0 = (a.xy / a.w * 0.5 + 0.5) * viewport;
    float2 p1 = (b.xy / b.w * 0.5 + 0.5) * viewport;
    float2 p2 = (c.xy / c.w * 0.5 + 0.5) * viewport;
    float2 extent = max(p0, max(p1, p2)) - min(p0, min(p1, p2));
    if (max(extent.x, extent.y) > maxPixels) { return false; }
    // Offscreen/large coordinates stay with the hardware clipper. This also
    // bounds integer edge products in the software rasterizer.
    if (any(min(p0, min(p1, p2)) < -32.0) ||
        any(max(p0, max(p1, p2)) > viewport + 32.0)) { return false; }
    return true;
}

bool hybridEnqueueTriangle(float4 a, float4 b, float4 c, uint visibilityId,
    bool doubleSided, uint queueIndex, uint bufferBase)
{
    if (queueIndex == 0xffffffffu) { return false; }
    DescriptorHandle<RWStructuredBuffer<uint4>> handle =
        DescriptorHandle<RWStructuredBuffer<uint4>>(uint2(bufferBase + queueIndex, 0));
    RWStructuredBuffer<uint4> queue = handle;
    uint4 config = queue[1];
    if (!hybridTriangleFits(a, b, c, float2(config.xy), asfloat(config.z))) { return false; }
    uint index;
    InterlockedAdd(queue[0].x, 1u, index);
    if (index >= config.w) { return false; }
    uint offset = 2u + index * 4u;
    queue[offset] = asuint(a);
    queue[offset + 1u] = asuint(b);
    queue[offset + 2u] = asuint(c);
    queue[offset + 3u] = uint4(visibilityId, doubleSided ? 1u : 0u, 0u, 0u);
    return true;
}
#endif
