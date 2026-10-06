#language slang 2026
module OpenPBRTextureLUT;

namespace Metallic.OpenPBR
{
// Adobe's precomputed energy tables are linear R16_UNORM; LTC is RGBA32_FLOAT.
// One mip, linear min/mag, clamp-to-edge on all axes. Coordinates already include
// the half-texel transform in the OpenPBR module. Never apply sRGB conversion.
public uint openPBRLut2DSlot(int id)
{
    return id == 1 ? 0u : id == 2 ? 1u : id == 4 ? 2u : id == 5 ? 3u : id == 6 ? 4u : 5u;
}

public uint openPBRLut3DSlot(int id)
{
    return id == 3 ? 1u : 0u;
}

public float4 sampleOpenPBRTexture2D(Texture2D<float4> texture, SamplerState linearClamp, float2 uv)
{
    return texture.SampleLevel(linearClamp, uv, 0.0);
}

public float4 sampleOpenPBRTexture3D(Texture3D<float4> texture, SamplerState linearClamp, float3 uvw)
{
    // Keep the inter-slice interpolation in float. Native 3D filtering can add
    // intermediate weight rounding beyond the 2D sub-texel precision bound.
    // Both energy volumes are 32 slices; each lookup still uses hardware XY
    // filtering (two texture samples instead of eight loads and three-axis ALU).
    float z = clamp(uvw.z * 32.0 - 0.5, 0.0, 31.0);
    float slice = floor(z);
    float4 lo = texture.SampleLevel(linearClamp, float3(uvw.xy, (slice + 0.5) / 32.0), 0.0);
    float4 hi = texture.SampleLevel(linearClamp, float3(uvw.xy, (slice + 1.5) / 32.0), 0.0);
    return lerp(lo, hi, z - slice);
}

} // namespace Metallic.OpenPBR
