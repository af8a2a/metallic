// Shared domain evaluation for resident and streamed patches. The table and
// material descriptors are immutable for the lifetime of the graph bindings.
struct TessMaterial
{
    uint texture;
    float magnitude;
    float centre;
    uint enabled;
    float4 uv0;
    float4 uv1;
    uint4 reserved;
};

TessMaterial tessMaterial(StructuredBuffer<uint> data, uint material)
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

float tessHeight(TessMaterial material, float2 uv, uint imageBase)
{
    if (material.enabled == 0u || material.texture == 0xffffffffu) { return 0.0; }
    uv = float2(dot(material.uv0.xy, uv) + material.uv0.z, dot(material.uv1.xy, uv) + material.uv1.z);
    DescriptorHandle<Texture2D<float4>> handle = DescriptorHandle<Texture2D<float4>>(uint2(imageBase + material.texture, 0));
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

uint tessEdgeRate(float3 a, float3 b, float3 eye, float3 forward, float pixelScale,
    float nearPlane, bool orthographic, float targetPixels, uint maximum)
{
    // Unjittered, endpoint-symmetric rate. Projection is conservatively based
    // on view-space length so foreshortening cannot hide displacement detail.
    float distance = orthographic ? 1.0 : max(min(dot(a - eye, forward), dot(b - eye, forward)), nearPlane);
    float rate = ceil(length(b - a) * pixelScale / max(distance * targetPixels, 1e-6));
    return uint(clamp(rate, 1.0, float(maximum)));
}

uint4 tessPattern(StructuredBuffer<uint> data, uint3 rates)
{
    uint index = data[0] + ((rates.x - 1u) + 8u * (rates.y - 1u) + 64u * (rates.z - 1u)) * 4u;
    return uint4(data[index], data[index + 1u], data[index + 2u], data[index + 3u]);
}

float3 tessBary(StructuredBuffer<uint> data, uint4 pattern, uint vertex)
{
    uint word = data[data[0] + pattern.x + vertex];
    float2 yz = float2(word & 65535u, word >> 16u) / 65535.0;
    return float3(1.0 - yz.x - yz.y, yz);
}

uint3 tessTriangle(StructuredBuffer<uint> data, uint4 pattern, uint triangle)
{
    uint word = data[data[0] + pattern.z + triangle];
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
#if VISIBILITY_TESSELLATION_STREAM
    [vk::location(4)] nointerpolation uint doubleSided : TEXCOORD4;
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
};

float3 tessNormal(float3 n, float3 fallback)
{
    return dot(n, n) > 1e-20 ? normalize(n) : fallback;
}

// Canonical evaluation of boundary vertices gives shared edges identical
// floating point positions in either winding, including odd edge rates.
void tessDomain(inout float3 bary, uint3 rates, float3 p[3], float3 n[3], float2 uv[3],
    out float3 position, out float3 normal, out float2 texcoord)
{
    uint zero = bary.z < 1e-7 ? 2u : (bary.x < 1e-7 ? 0u : (bary.y < 1e-7 ? 1u : 3u));
    if (zero < 3u) {
        uint a = (zero + 1u) % 3u, b = (zero + 2u) % 3u;
        uint rate = rates[a];
        uint step = uint(round(bary[b] * rate));
        bool swap = p[a].x > p[b].x || (p[a].x == p[b].x &&
            (p[a].y > p[b].y || (p[a].y == p[b].y && p[a].z > p[b].z)));
        if (swap) { uint t = a; a = b; b = t; step = rate - step; }
        float t = float(step) / float(rate);
        bary = float3(0); bary[a] = 1.0 - t; bary[b] = t;
        position = lerp(p[a], p[b], t);
        normal = lerp(n[a], n[b], t);
        texcoord = lerp(uv[a], uv[b], t);
    } else {
        position = p[0] * bary.x + p[1] * bary.y + p[2] * bary.z;
        normal = n[0] * bary.x + n[1] * bary.y + n[2] * bary.z;
        texcoord = uv[0] * bary.x + uv[1] * bary.y + uv[2] * bary.z;
    }
    normal = tessNormal(normal, tessNormal(cross(p[1] - p[0], p[2] - p[0]), float3(0, 1, 0)));
}
