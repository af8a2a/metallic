#ifndef METALLIC_SCENE_SHADING_VERTEX_SLANG
#define METALLIC_SCENE_SHADING_VERTEX_SLANG

#ifndef SCENE_RAYQUERY_ENABLE_POSITION_FETCH
#define SCENE_RAYQUERY_ENABLE_POSITION_FETCH 0
#endif

// Matches SceneShadingVertex.h: 16 bytes; no position or padded UV components.
struct SceneShadingVertex
{
    uint normal;
    uint tangent;
    float2 texcoord;
};

[[vk::binding(2, 0)]] StructuredBuffer<SceneShadingVertex> gVertices;
#if !SCENE_RAYQUERY_ENABLE_POSITION_FETCH
// Present only on devices that cannot fetch triangle positions from the BLAS.
// Three scalars keep a 12-byte stride under std430 as well as scalar layout;
// an array of float3 would have a 16-byte stride under std430.
struct SceneVertexPosition { float x; float y; float z; };
[[vk::binding(54, 0)]] StructuredBuffer<SceneVertexPosition> gVertexPositions;

float3 loadSceneVertexPosition(uint index)
{
    SceneVertexPosition position = gVertexPositions[index];
    return float3(position.x, position.y, position.z);
}
#endif

float3 decodeSceneOctahedron(float2 oct)
{
    float3 direction = float3(oct, 1.0 - abs(oct.x) - abs(oct.y));
    float fold = max(-direction.z, 0.0);
    direction.xy += float2(direction.x >= 0.0 ? -fold : fold, direction.y >= 0.0 ? -fold : fold);
    return normalize(direction);
}

float3 decodeSceneNormal(uint packed)
{
    if (packed == 0x80008000u) { return float3(0.0); }
    int2 quantized = int2(int(packed << 16) >> 16, int(packed) >> 16);
    return decodeSceneOctahedron(float2(quantized) / 32767.0);
}

float4 decodeSceneTangent(uint packed)
{
    float sign = (packed & 0x40000000u) != 0u ? -1.0 : 1.0;
    if ((packed & 0x80000000u) == 0u) { return float4(0.0, 0.0, 0.0, sign); }
    int2 quantized = int2(int(packed << 17) >> 17, int(packed << 2) >> 17);
    return float4(decodeSceneOctahedron(float2(quantized) / 16383.0), sign);
}

#endif
