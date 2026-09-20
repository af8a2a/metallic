#language slang 2026
implementing GPUDriven;

namespace Metallic.GPUDriven
{
internal int hybridEdge(int2 a, int2 b, int2 p)
{
    int2 d = b - a;
    int2 q = p - a;
    return d.x * q.y - d.y * q.x;
}

// For positive edge equations in a top-left origin framebuffer.
internal int hybridEdgeBias(int2 a, int2 b)
{
    int2 d = b - a;
    return d.y < 0 || (d.y == 0 && d.x > 0) ? 0 : -1;
}

public void hybridRasterTriangle(float4 c0, float4 c1, float4 c2, uint visibilityId,
    bool doubleSided, uint width, uint height, uint reversedZ, uint subpixelBits,
    uint pixelBuffer, uint bufferBase)
{
    float2 viewport = float2(width, height);
    float2 s0 = (c0.xy / c0.w * 0.5 + 0.5) * viewport;
    float2 s1 = (c1.xy / c1.w * 0.5 + 0.5) * viewport;
    float2 s2 = (c2.xy / c2.w * 0.5 + 0.5) * viewport;
    int scale = 1 << subpixelBits;
    int2 p0 = int2(floor(s0 * scale + 0.5));
    int2 p1 = int2(floor(s1 * scale + 0.5));
    int2 p2 = int2(floor(s2 * scale + 0.5));
    int area = hybridEdge(p0, p1, p2);
    // Vulkan CCW fronts have negative area after the positive-height viewport.
    if (area == 0 || (!doubleSided && area > 0)) { return; }
    if (area < 0) {
        int2 temp = p1; p1 = p2; p2 = temp;
        float4 clipTemp = c1; c1 = c2; c2 = clipTemp;
        float2 screenTemp = s1; s1 = s2; s2 = screenTemp;
        area = -area;
    }
    int2 lower = max(int2(0), int2(ceil(float2(min(p0, min(p1, p2))) / scale - 0.5)));
    int2 upper = min(int2(width, height) - 1,
        int2(floor(float2(max(p0, max(p1, p2))) / scale - 0.5)));
    int3 bias = int3(hybridEdgeBias(p1, p2), hybridEdgeBias(p2, p0), hybridEdgeBias(p0, p1));
    float3 z = float3(c0.z / c0.w, c1.z / c1.w, c2.z / c2.w);
    // Derive the depth plane from the same snapped vertices as coverage.
    s0 = float2(p0) / scale;
    s1 = float2(p1) / scale;
    s2 = float2(p2) / scale;
    float2 e1 = s1 - s0;
    float2 e2 = s2 - s0;
    float determinant = e1.x * e2.y - e1.y * e2.x;
    if (abs(determinant) < 1e-12) { return; }
    DescriptorHandle<RWStructuredBuffer<uint64_t>> pixelHandle =
        DescriptorHandle<RWStructuredBuffer<uint64_t>>(uint2(bufferBase + pixelBuffer, 0));
    RWStructuredBuffer<uint64_t> pixels = pixelHandle;
    for (int y = lower.y; y <= upper.y; ++y) {
        for (int x = lower.x; x <= upper.x; ++x) {
            int2 point = int2(x, y) * scale + scale / 2;
            int3 edges = int3(hybridEdge(p1, p2, point), hybridEdge(p2, p0, point), hybridEdge(p0, p1, point));
            if (any(edges + bias < 0)) { continue; }
            float2 q = float2(x, y) + 0.5 - s0;
            float b1 = (q.x * e2.y - q.y * e2.x) / determinant;
            float b2 = (e1.x * q.y - e1.y * q.x) / determinant;
            float depth = saturate(z.x + b1 * (z.y - z.x) + b2 * (z.z - z.x));
            uint key = asuint(depth);
            if (reversedZ == 0u) { key = ~key; }
            uint64_t packed = (uint64_t(key) << 32u) | uint64_t(visibilityId);
            InterlockedMax(pixels[uint(y) * width + uint(x)], packed);
        }
    }
}

// One setup per unique vertex, shared by all triangles in a cluster. Keep
// division/snap expressions identical to the reference rasterizer above.
public struct HybridScreenVertex
{
    public int2 position;
    public float depth;
};

public HybridScreenVertex hybridPrepareVertex(float4 clip, float2 viewport, uint subpixelBits)
{
    float2 screen = (clip.xy / clip.w * 0.5 + 0.5) * viewport;
    HybridScreenVertex result;
    result.position = int2(floor(screen * (1 << subpixelBits) + 0.5));
    result.depth = clip.z / clip.w;
    return result;
}

public void hybridRasterPreparedTriangle(HybridScreenVertex a, HybridScreenVertex b, HybridScreenVertex c,
    uint visibilityId, bool doubleSided, uint width, uint height, uint reversedZ, uint subpixelBits,
    uint pixelBuffer, uint bufferBase, bool incrementalDepth)
{
    int2 p0 = a.position, p1 = b.position, p2 = c.position;
    int area = hybridEdge(p0, p1, p2);
    if (area == 0 || (!doubleSided && area > 0)) { return; }
    if (area < 0) {
        int2 temp = p1; p1 = p2; p2 = temp;
        float z = b.depth; b.depth = c.depth; c.depth = z;
    }
    int scale = 1 << subpixelBits;
    int2 lower = max(int2(0), int2(ceil(float2(min(p0, min(p1, p2))) / scale - 0.5)));
    int2 upper = min(int2(width, height) - 1, int2(floor(float2(max(p0, max(p1, p2))) / scale - 0.5)));
    if (any(lower > upper)) { return; }
    int3 bias = int3(hybridEdgeBias(p1,p2), hybridEdgeBias(p2,p0), hybridEdgeBias(p0,p1));
    int2 point = lower * scale + scale / 2;
    int3 rowEdges = int3(hybridEdge(p1,p2,point), hybridEdge(p2,p0,point), hybridEdge(p0,p1,point)) + bias;
    int3 stepX = -int3(p2.y-p1.y, p0.y-p2.y, p1.y-p0.y) * scale;
    int3 stepY = int3(p2.x-p1.x, p0.x-p2.x, p1.x-p0.x) * scale;
    float2 s0 = float2(p0) / scale;
    float2 e1 = float2(p1) / scale - s0, e2 = float2(p2) / scale - s0;
    float determinant = e1.x * e2.y - e1.y * e2.x;
    if (abs(determinant) < 1e-12) { return; }
    float3 z = float3(a.depth,b.depth,c.depth);
    float2 gradient = float2(e2.y*(z.y-z.x)-e1.y*(z.z-z.x), e1.x*(z.z-z.x)-e2.x*(z.y-z.x)) / determinant;
    DescriptorHandle<RWStructuredBuffer<uint64_t>> handle = DescriptorHandle<RWStructuredBuffer<uint64_t>>(uint2(bufferBase + pixelBuffer, 0));
    RWStructuredBuffer<uint64_t> pixels = handle;
    for (int y = lower.y; y <= upper.y; ++y) {
        int3 edges = rowEdges;
        // Rebase the optional float plane at each row to limit drift. The
        // default path preserves reference depth arithmetic at covered pixels.
        float rowDepth = z.x + (float(lower.x) + 0.5 - s0.x) * gradient.x + (float(y) + 0.5 - s0.y) * gradient.y;
        float scannedDepth = rowDepth;
        for (int x = lower.x; x <= upper.x; ++x) {
            if (all(edges >= 0)) {
                float depth;
                if (incrementalDepth) { depth = saturate(scannedDepth); }
                else {
                    float2 q = float2(x,y) + 0.5 - s0;
                    float b1 = (q.x * e2.y - q.y * e2.x) / determinant;
                    float b2 = (e1.x * q.y - e1.y * q.x) / determinant;
                    depth = saturate(z.x + b1 * (z.y-z.x) + b2 * (z.z-z.x));
                }
                uint key = asuint(depth);
                if (reversedZ == 0u) { key = ~key; }
                InterlockedMax(pixels[uint(y)*width+uint(x)], (uint64_t(key)<<32u) | uint64_t(visibilityId));
            }
            edges += stepX;
            if (incrementalDepth) { scannedDepth += gradient.x; }
        }
        rowEdges += stepY;
    }
}
} // namespace Metallic.GPUDriven
