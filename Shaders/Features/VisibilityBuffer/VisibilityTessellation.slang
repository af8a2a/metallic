struct TessellationData
{
    DataSpan<uint> words;
    uint load(uint index) { return words.contains(index) ? words.data[index] : 0u; }
};

// Shared domain evaluation for resident and streamed patches. The table and
// material descriptors are immutable for the lifetime of the graph bindings.
struct TessMaterial
{
    DescriptorHandle<Texture2D<float4>> texture;
    float magnitude;
    float centre;
    uint enabled;
    float4 uv0;
    float4 uv1;
    uint4 reserved;
};

TessMaterial tessMaterial(TessellationData data, uint material)
{
    TessMaterial m;
    m.texture = DescriptorHandle<Texture2D<float4>>(uint2(0xffffffffu));
    m.magnitude = m.centre = 0.0;
    m.enabled = 0u;
    m.uv0 = m.uv1 = float4(0);
    m.reserved = uint4(0);
    if (material >= data.load(1)) { return m; }
    uint base = 16u + material * 16u;
    m.texture = DescriptorHandle<Texture2D<float4>>(uint2(data.load(base), data.load(base + 12u)));
    m.magnitude = asfloat(data.load(base + 1u));
    m.centre = asfloat(data.load(base + 2u));
    m.enabled = data.load(base + 3u);
    m.uv0 = asfloat(uint4(data.load(base + 4u), data.load(base + 5u), data.load(base + 6u), data.load(base + 7u)));
    m.uv1 = asfloat(uint4(data.load(base + 8u), data.load(base + 9u), data.load(base + 10u), data.load(base + 11u)));
    return m;
}

float tessHeight(TessMaterial material, float2 uv)
{
    if (material.enabled == 0u) { return 0.0; }
    uv = float2(dot(material.uv0.xy, uv) + material.uv0.z, dot(material.uv1.xy, uv) + material.uv1.z);
    Texture2D<float4> image = material.texture;
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

uint tessEdgeRate(float3 a, float3 b, float3 eye, float3 forward, float pixelScale,
    float nearPlane, bool orthographic, float targetPixels, uint maximum)
{
    // Unjittered, endpoint-symmetric rate. Projection is conservatively based
    // on view-space length so foreshortening cannot hide displacement detail.
    float distance = orthographic ? 1.0 : max(min(dot(a - eye, forward), dot(b - eye, forward)), nearPlane);
    float rate = ceil(length(b - a) * pixelScale / max(distance * targetPixels, 1e-6));
    return uint(clamp(rate, 1.0, float(maximum)));
}

uint4 tessPattern(TessellationData data, uint3 rates)
{
    uint index = data.load(0) + ((rates.x - 1u) + 8u * (rates.y - 1u) + 64u * (rates.z - 1u)) * 4u;
    return uint4(data.load(index), data.load(index + 1u), data.load(index + 2u), data.load(index + 3u));
}

float3 tessBary(TessellationData data, uint4 pattern, uint vertex)
{
    uint word = data.load(data.load(0) + pattern.x + vertex);
    float2 yz = float2(word & 65535u, word >> 16u) / 65535.0;
    return float3(1.0 - yz.x - yz.y, yz);
}

uint3 tessTriangle(TessellationData data, uint4 pattern, uint triangle)
{
    uint word = data.load(data.load(0) + pattern.z + triangle);
    return uint3(word & 255u, (word >> 8u) & 255u, (word >> 16u) & 255u);
}

float2 tessEncodeNormal(float3 n)
{
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20);
    return n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
}

float4 tessSurfaceDomain(float2 bary, float3 world, float2 uv, bool front)
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

struct TessVertex
{
    float4 position : SV_Position;
    [vk::location(0)] float2 uv : TEXCOORD0;
    [vk::location(1)] float2 bary : TEXCOORD1;
#if VISIBILITY_BUFFER_ALPHA_MASKED
    [vk::location(2)] nointerpolation uint material : TEXCOORD2;
#endif
    [vk::location(3)] float3 world : TEXCOORD3;
    [vk::location(5)] nointerpolation uint originalVisibilityId : TEXCOORD5;
    [vk::location(6)] nointerpolation uint debugPatchSeed : TEXCOORD6;
#if VISIBILITY_TESSELLATION_STREAM
    [vk::location(4)] nointerpolation uint doubleSided : TEXCOORD4;
    [vk::location(7)] nointerpolation uint streamInstance : TEXCOORD7;
#endif
};
struct TessPrimitive
{
    uint visibilityId : SV_PrimitiveID;
};
struct TessFragment
{
    uint visibility : SV_Target0;
    float4 domain : SV_Target1;
    float4 debugColor : SV_Target2;
};

uint tessDebugHash(uint value)
{
    value ^= value >> 16u; value *= 0x7feb352du;
    value ^= value >> 15u; value *= 0x846ca68bu;
    return value ^ (value >> 16u);
}

// Geometry identity and source-domain corners survive temporary draw-record
// relocation. SV_PrimitiveID supplies the diced triangle within this patch.
uint tessDebugPatchSeed(uint geometrySeed, uint sourceTriangle, uint3 patch)
{
    return geometrySeed ^ tessDebugHash(sourceTriangle + 1u) ^
        tessDebugHash(patch.x) ^ tessDebugHash(patch.y + 0x9e3779b9u) ^
        tessDebugHash(patch.z + 0x85ebca6bu);
}

float4 tessDebugColor(TessVertex input, uint primitiveId)
{
    uint triangle = input.originalVisibilityId != 0u ? primitiveId : (primitiveId & 127u);
    uint color = tessDebugHash(input.debugPatchSeed ^ tessDebugHash(triangle + 1u));
    return float4(0.3 + 0.5 * float3(color & 255u, (color >> 8u) & 255u, (color >> 16u) & 255u) / 255.0, 1.0);
}

float3 tessNormal(float3 n, float3 fallback)
{
    return dot(n, n) > 1e-20 ? normalize(n) : fallback;
}
