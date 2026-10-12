#language slang 2026
module VisibilityTessellation;

import ShaderCore;
using Metallic;

// Domain evaluation for streamed patches. The table and
// material descriptors are immutable for the lifetime of the graph bindings.
public struct TessMaterial
{
    public uint texture;
    public float magnitude;
    public float centre;
    public uint enabled;
    public float4 uv0;
    public float4 uv1;
    public uint4 reserved;
};

public TessMaterial tessMaterial(StructuredBuffer<uint> data, uint material)
{
    TessMaterial m = (TessMaterial)0;
    m.texture = 0xffffffffu;
    if (material >= data[1]) { return m; }
    uint base = 16u + material * 16u;
    m.texture = data[base];
    m.magnitude = asfloat(data[base + 1u]);
    m.centre = asfloat(data[base + 2u]);
    m.enabled = data[base + 3u];
    m.uv0 = asfloat(uint4(data[base + 4u], data[base + 5u], data[base + 6u], data[base + 7u]));
    m.uv1 = asfloat(uint4(data[base + 8u], data[base + 9u], data[base + 10u], data[base + 11u]));
    return m;
}

public float tessHeight(TessMaterial material, float2 uv)
{
    if (material.enabled == 0u || material.texture == 0xffffffffu) { return 0.0; }
    uv = float2(dot(material.uv0.xy, uv) + material.uv0.z, dot(material.uv1.xy, uv) + material.uv1.z);
    Texture2D<float4> handle = resolveNonUniform(ResourceHandle<Texture2D<float4>>(material.texture));
    Texture2D<float4> image = handle;
    uint width, height;
    image.GetDimensions(width, height);
    // Keep the integer conversion nonnegative at repeat seams, including UV=0.
    float2 texel = frac(uv) * float2(width, height) + float2(width, height) - 0.5;
    uint2 p = uint2(floor(texel));
    float2 f = frac(texel);
    uint2 size = uint2(width, height);
    uint2 a = p % size;
    uint2 b = (a + 1) % size;
    float h = lerp(lerp(image.Load(int3(a, 0)).r, image.Load(int3(b.x, a.y, 0)).r, f.x),
        lerp(image.Load(int3(a.x, b.y, 0)).r, image.Load(int3(b, 0)).r, f.x), f.y);
    return (saturate(h) - material.centre) * material.magnitude;
}

public uint tessEdgeRate(float3 a, float3 b, float3 eye, float3 forward, float pixelScale,
    float nearPlane, bool orthographic, float targetPixels, uint maximum)
{
    // Unjittered, endpoint-symmetric rate. Projection is conservatively based
    // on view-space length so foreshortening cannot hide displacement detail.
    float distance = orthographic ? 1.0 : max(min(dot(a - eye, forward), dot(b - eye, forward)), nearPlane);
    float rate = ceil(length(b - a) * pixelScale / max(distance * targetPixels, 1e-6));
    return uint(clamp(rate, 1.0, float(maximum)));
}

public uint4 tessPattern(StructuredBuffer<uint> data, uint3 rates)
{
    uint index = data[0] + ((rates.x - 1u) + 8u * (rates.y - 1u) + 64u * (rates.z - 1u)) * 4u;
    return uint4(data[index], data[index + 1u], data[index + 2u], data[index + 3u]);
}

public float3 tessBary(StructuredBuffer<uint> data, uint4 pattern, uint vertex)
{
    uint word = data[data[0] + pattern.x + vertex];
    float2 yz = float2(word & 65535u, word >> 16u) / 65535.0;
    return float3(1.0 - yz.x - yz.y, yz);
}

public uint3 tessTriangle(StructuredBuffer<uint> data, uint4 pattern, uint triangle)
{
    uint word = data[data[0] + pattern.z + triangle];
    return uint3(word & 255u, (word >> 8u) & 255u, (word >> 16u) & 255u);
}

public float2 tessEncodeNormal(float3 n)
{
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20);
    return n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
}

public float4 tessSurfaceDomain(float2 bary, float3 world, float2 uv, bool front)
{
    float3 dx = ddx(world), dy = ddy(world);
    float3 area = cross(dx, dy);
    float3 n = dot(area, area) > 1e-30 ? normalize(area) * (front ? -1.0 : 1.0) : float3(0, 1, 0);
    uint2 oct = uint2(round(saturate(tessEncodeNormal(n) * 0.5 + 0.5) * 4095.0));
    // A 24-bit integer is exactly representable as float. Unlike bit-casting,
    // this encoding never depends on NaN payload preservation by an attachment.
    float packedNormal = float(oct.x | (oct.y << 12u));
    float2 uvDx = ddx(uv), uvDy = ddy(uv);
    float uvArea = abs(uvDx.x * uvDy.y - uvDx.y * uvDy.x);
    float lod = uvArea > 1e-20 ? 0.5 * log2(uvArea / max(length(area), 1e-20)) : 0.0;
    return float4(bary, lod, packedNormal);
}

public struct TessPrimitive
{
    public uint visibilityId : SV_PrimitiveID;
};
public struct TessFragment
{
    public uint visibility : SV_Target0;
    public float4 domain : SV_Target1;
    public float4 debugColor : SV_Target2;
};

public uint tessDebugHash(uint value)
{
    value ^= value >> 16u; value *= 0x7feb352du;
    value ^= value >> 15u; value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

// Geometry identity and source-domain corners survive temporary draw-record
// relocation. SV_PrimitiveID supplies the diced triangle within this patch.
public uint tessDebugPatchSeed(uint geometrySeed, uint sourceTriangle, uint3 patch)
{
    return geometrySeed ^ tessDebugHash(sourceTriangle + 1u) ^
        tessDebugHash(patch.x) ^ tessDebugHash(patch.y + 0x9e3779b9u) ^
        tessDebugHash(patch.z + 0x85ebca6bu);
}


public float3 tessNormal(float3 n, float3 fallback)
{
    return dot(n, n) > 1e-20 ? normalize(n) : fallback;
}

__include "VisibilityTessellation/VisibilityTessellationSplit.slang";
