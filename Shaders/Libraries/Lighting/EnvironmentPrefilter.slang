#ifndef METALLIC_ENVIRONMENT_PREFILTER
#define METALLIC_ENVIRONMENT_PREFILTER

static const uint kEnvironmentFilterWidth = 256u;
static const uint kEnvironmentFilterHeight = 128u;
static const uint kEnvironmentFilterLayers = 8u;

float2 environmentFilterUv(float3 direction)
{
    return float2(atan2(direction.z, direction.x) / 6.28318530718 + 0.5,
        acos(clamp(direction.y, -1.0, 1.0)) / 3.14159265359);
}

float3 environmentFilterDirection(float2 uv)
{
    float theta = uv.y * 3.14159265359;
    float phi = (uv.x - 0.5) * 6.28318530718;
    return float3(cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta));
}

int2 environmentFilterTexel(int2 pixel, int2 size)
{
    return int2((pixel.x % size.x + size.x) % size.x, clamp(pixel.y, 0, size.y - 1));
}

float3 environmentFilterSource(Texture2D<float4> source, float3 direction)
{
    uint width, height;
    source.GetDimensions(width, height);
    float2 texel = environmentFilterUv(direction) * float2(width, height) - 0.5;
    int2 p = int2(floor(texel)), size = int2(width, height);
    float2 f = frac(texel);
    return lerp(lerp(source.Load(int3(environmentFilterTexel(p, size), 0)).rgb,
                     source.Load(int3(environmentFilterTexel(p + int2(1, 0), size), 0)).rgb, f.x),
                lerp(source.Load(int3(environmentFilterTexel(p + int2(0, 1), size), 0)).rgb,
                     source.Load(int3(environmentFilterTexel(p + int2(1, 1), size), 0)).rgb, f.x), f.y);
}

float3 environmentFilterLayer(StructuredBuffer<float4> source, float3 direction, uint layer)
{
    int2 size = int2(kEnvironmentFilterWidth, kEnvironmentFilterHeight);
    float2 texel = environmentFilterUv(direction) * float2(size) - 0.5;
    int2 p = int2(floor(texel));
    float2 f = frac(texel);
    uint base = layer * kEnvironmentFilterWidth * kEnvironmentFilterHeight;
    int2 a = environmentFilterTexel(p, size), b = environmentFilterTexel(p + int2(1, 0), size);
    int2 c = environmentFilterTexel(p + int2(0, 1), size), d = environmentFilterTexel(p + 1, size);
    return lerp(lerp(source[base + a.y * size.x + a.x].rgb, source[base + b.y * size.x + b.x].rgb, f.x),
                lerp(source[base + c.y * size.x + c.x].rgb, source[base + d.y * size.x + d.x].rgb, f.x), f.y);
}
#endif
