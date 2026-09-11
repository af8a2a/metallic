#include "../../Libraries/Resources/ComputeResources.slang"

#include "../../Libraries/Scene/SceneShadingVertex.slang"

#define gDecoded METALLIC_RESOURCE(RWStructuredBuffer<float4>, 63)

[shader("compute")]
[numthreads(1, 1, 1)]
void sceneShadingVertexProbeMain(uint3 id : SV_DispatchThreadID)
{
    SceneShadingVertex vertex = gVertices[id.x];
    gDecoded[id.x * 3u] = float4(decodeSceneNormal(vertex.normal), 0.0);
    gDecoded[id.x * 3u + 1u] = decodeSceneTangent(vertex.tangent);
    gDecoded[id.x * 3u + 2u] = float4(vertex.texcoord, 0.0, 0.0);
}
