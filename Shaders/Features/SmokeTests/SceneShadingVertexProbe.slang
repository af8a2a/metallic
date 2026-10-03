import Core;
using Metallic;

#define gDecoded resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<SceneResourceParameters>().probeOutput)

[shader("compute")]
[numthreads(1, 1, 1)]
void sceneShadingVertexProbeMain(uint3 id : SV_DispatchThreadID)
{
    SceneShadingVertex vertex = resolveBuffer<StructuredBuffer<SceneShadingVertex>>(getResourceParameters<SceneResourceParameters>().vertices)[id.x];
    gDecoded[id.x * 3u] = float4(decodeSceneNormal(vertex.normal), 0.0);
    gDecoded[id.x * 3u + 1u] = decodeSceneTangent(vertex.tangent);
    gDecoded[id.x * 3u + 2u] = float4(vertex.texcoord, 0.0, 0.0);
}
