module LumenShortRangeAOPacking;

namespace Metallic.Lumen
{

// UE 5.7.4 PackUnpack.ush, UNORM 11/11/10 encoding.
uint packShortRangeAORGB111110(float3 rgb)
{
    uint r = uint(floor(saturate(rgb.r) * 2047.0f + 0.5f));
    uint g = uint(floor(saturate(rgb.g) * 2047.0f + 0.5f));
    uint b = uint(floor(saturate(rgb.b) * 1023.0f + 0.5f));
    return (r << 21) | (g << 10) | b;
}

float3 unpackShortRangeAORGB111110(uint rgb)
{
    float r = float(rgb >> 21) * (1.0f / 2047.0f);
    float g = float((rgb >> 10) & 2047u) * (1.0f / 2047.0f);
    float b = float(rgb & 1023u) * (1.0f / 1023.0f);
    return float3(r, g, b);
}

// UE 5.7.4 LumenScreenSpaceBentNormal.ush.
public uint packLumenScreenBentNormal(float3 bentNormal)
{
    return packShortRangeAORGB111110(bentNormal * 0.5f + 0.5f);
}

public float3 unpackLumenScreenBentNormal(uint packedBentNormal)
{
    return unpackShortRangeAORGB111110(packedBentNormal) * 2.0f - 1.0f;
}

// UE 5.7.4 GetDownsampleJitter2x2, four-rooks pattern in a 2x2 cell.
public uint2 lumenShortRangeAOJitter(uint2 downsampledScreenCoord, uint frameIndex)
{
    uint2 cellIndex = downsampledScreenCoord % 2u;
    uint linearIndex = cellIndex.x + cellIndex.y * 2u;
    linearIndex = (linearIndex + frameIndex) % 4u;

    uint2 jitter;
    jitter.x = (linearIndex & 0x02u) != 0u ? 1u : 0u;
    jitter.y = (linearIndex & 0x01u) != 0u ? 0u : 1u;
    return jitter;
}

// EncodeNormal/DecodeNormal with the PF_A2B10G10R10 guide texture's RGB quantization.
public float3 lumenShortRangeAODecodeNormal(float3 normal)
{
    float3 encoded = saturate(normal * 0.5f + 0.5f);
    float3 quantized = floor(encoded * 1023.0f + 0.5f) / 1023.0f;
    return quantized * 2.0f - 1.0f;
}

} // namespace Metallic.Lumen
