#language slang 2026
implementing ShaderCore;

namespace Metallic
{

// Shared octahedral encodings for streamed meshlet attributes.
internal float3 decodeSceneOctahedron(float2 oct)
{
    float3 direction = float3(oct, 1.0 - abs(oct.x) - abs(oct.y));
    float fold = max(-direction.z, 0.0);
    direction.xy += float2(direction.x >= 0.0 ? -fold : fold, direction.y >= 0.0 ? -fold : fold);
    return normalize(direction);
}

public float3 decodeSceneNormal(uint packed)
{
    if (packed == 0x80008000u) { return float3(0.0); }
    int2 quantized = int2(int(packed << 16) >> 16, int(packed) >> 16);
    return decodeSceneOctahedron(float2(quantized) / 32767.0);
}

public float4 decodeSceneTangent(uint packed)
{
    float sign = (packed & 0x40000000u) != 0u ? -1.0 : 1.0;
    if ((packed & 0x80000000u) == 0u) { return float4(0.0, 0.0, 0.0, sign); }
    int2 quantized = int2(int(packed << 17) >> 17, int(packed << 2) >> 17);
    return float4(decodeSceneOctahedron(float2(quantized) / 16383.0), sign);
}
} // namespace Metallic
