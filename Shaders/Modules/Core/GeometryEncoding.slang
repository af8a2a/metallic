#language slang 2026
implementing ShaderCore;

namespace Metallic
{

// Shared octahedral encodings for streamed meshlet attributes.
internal float3 decodeSceneOctahedron(float2 oct)
{
    float3 direction = float3(oct, 1.0 - abs(oct.x) - abs(oct.y));
    float fold = max(-direction.z, 0.0);
    direction.xy += select(direction.xy >= float2(0.0), float2(-fold), float2(fold));
    return normalize(direction);
}

public float3 decodeSceneNormal(uint packed)
{
    int2 quantized = int2(int(packed << 16) >> 16, int(packed) >> 16);
    float3 decoded = decodeSceneOctahedron(float2(quantized) / 32767.0);
    return select(bool3(packed == 0x80008000u), float3(0.0), decoded);
}

public float4 decodeSceneTangent(uint packed)
{
    float sign = select((packed & 0x40000000u) != 0u, -1.0, 1.0);
    int2 quantized = int2(int(packed << 17) >> 17, int(packed << 2) >> 17);
    float3 decoded = decodeSceneOctahedron(float2(quantized) / 16383.0);
    return float4(select(bool3((packed & 0x80000000u) == 0u), float3(0.0), decoded), sign);
}
} // namespace Metallic
