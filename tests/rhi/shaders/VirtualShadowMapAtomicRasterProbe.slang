module VirtualShadowMapAtomicRasterProbe;

import Core;
using Metallic;

// The pinned Slang format enum is supplied by the host. Untyped descriptor resolution
// emits Unknown, which Vulkan disallows for OpImageTexelPointer.
typealias DepthAtlas = RWTexture2D<uint, 0, VSM_R32UI_FORMAT>;
struct Push { uint atlas; uint subpixelBits; };
static const uint pageSize = 16u;
static const uint trianglesPerPage = 9u;

// Projected page-local texels and reverse depth. Hardware and compute receive
// the same vertices, including pixels on a shared edge and a physical-page seam.
float3 probeVertex(uint triangle, uint corner)
{
    uint t = triangle % trianglesPerPage;
    float3 a = float3(1.5, 1.5, .25), b = float3(5.5, 1.5, .25), c = float3(1.5, 5.5, .25);
    switch (t) {
    case 0: break;
    case 1: a.xy = float2(5.5); b.xy = float2(1.5, 5.5); c.xy = float2(5.5, 1.5);
            a.z = b.z = c.z = .75; break;
    case 2: a.z = b.z = c.z = .5; break; // Overlap in a different workgroup.
    case 3: a.xy += float2(13, 0); b.xy += float2(13, 0); c.xy += float2(13, 0); break;
    case 4: a = float3(-1, 7, .25); b = float3(3, 7, .5); c = float3(-1, 11, .75); break;
    case 5: a = float3(7.25, 3.25, 1); b = float3(7.25, 3.875, 1); c = float3(7.875, 3.25, 1); break;
    case 6: a.xy = b.xy; break;
    case 7: a.xy += float2(4, 9); b.xy += float2(4, 9); c.xy += float2(4, 9);
            a.z = b.z = c.z = .0625; break; // Behind retained depth.
    case 8: a.xy = float2(14, 8); b.xy = float2(15, 8); c.xy = float2(15, 9);
            a.z = b.z = c.z = .875; break;
    }
    // Distinguish pages if a cross-page scatter or lookup uses the wrong origin.
    if (triangle / trianglesPerPage != 0u) { a.y += 2; b.y += 2; c.y += 2; }
    return corner == 0u ? a : (corner == 1u ? b : c);
}

[shader("vertex")]
float4 depthVertex(uint vertex : SV_VulkanVertexID) : SV_Position
{
    uint triangle = vertex / 3u;
    float3 p = probeVertex(triangle, vertex % 3u);
    float2 atlas = float2((triangle / trianglesPerPage) * pageSize, 0) + p.xy;
    return float4(atlas * (2.0 / float2(pageSize * 2, pageSize)) - 1.0, p.z, 1.0);
}

[shader("fragment")]
void depthFragment() {}

[shader("compute")]
[numthreads(8, 8, 1)]
void initializeDepth(uint3 id : SV_DispatchThreadID, uniform Push push)
{
    resolveUniform(ResourceHandle<DepthAtlas>(push.atlas))[id.xy] = asuint(.125f);
}

int edge(int2 a, int2 b, int2 p)
{
    int2 delta = b - a, offset = p - a;
    return delta.x * offset.y - delta.y * offset.x;
}

int edgeBias(int2 a, int2 b)
{
    return b.y < a.y || (b.y == a.y && b.x > a.x) ? 0 : -1;
}

// One small opaque triangle per invocation. Binning limits the integer setup
// to short edges. The test inputs are already inside the homogeneous Z range.
[shader("compute")]
[numthreads(2, 1, 1)]
void rasterDepth(uint3 id : SV_DispatchThreadID, uniform Push push)
{
    if (id.x >= 2u * trianglesPerPage) { return; }
    float3 a = probeVertex(id.x, 0), b = probeVertex(id.x, 1), c = probeVertex(id.x, 2);
    int scale = 1 << push.subpixelBits;
    int2 p0 = int2(floor(a.xy * scale + .5)), p1 = int2(floor(b.xy * scale + .5)), p2 = int2(floor(c.xy * scale + .5));
    int area = edge(p0, p1, p2);
    if (area == 0) { return; }
    if (area < 0) {
        int2 tmp = p1; p1 = p2; p2 = tmp;
        float z = b.z; b.z = c.z; c.z = z;
        area = -area;
    }
    int2 lo = max(int2(0), int2(ceil(float2(min(p0, min(p1, p2))) / scale - .5)));
    int2 hi = min(int2(pageSize - 1u), int2(floor(float2(max(p0, max(p1, p2))) / scale - .5)));
    int3 bias = int3(edgeBias(p1, p2), edgeBias(p2, p0), edgeBias(p0, p1));
    DepthAtlas atlas = resolveUniform(ResourceHandle<DepthAtlas>(push.atlas));
    for (int y = lo.y; y <= hi.y; ++y) {
        for (int x = lo.x; x <= hi.x; ++x) {
            int2 center = int2(x, y) * scale + scale / 2;
            int3 e = int3(edge(p1, p2, center), edge(p2, p0, center), edge(p0, p1, center));
            if (any(e + bias < 0)) { continue; }
            float z = a.z + (float(e.y) * (b.z - a.z) + float(e.z) * (c.z - a.z)) / float(area);
            InterlockedMax(atlas[uint2(id.x / trianglesPerPage * pageSize + x, y)], asuint(z));
        }
    }
}
