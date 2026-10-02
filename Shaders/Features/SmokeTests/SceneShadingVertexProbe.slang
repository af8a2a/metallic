#include "../../../Source/Runtime/Render/Core/SceneProbeParameters.h"
using Metallic;

[[vk::push_constant]] ConstantBuffer<SceneVertexProbeParameters> gProbe;
#define gDecoded gProbe.output.data

[shader("compute")]
[numthreads(1, 1, 1)]
void sceneShadingVertexProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (!gProbe.vertices.contains(id.x) || id.x >= gProbe.output.length() / 3u) { return; }
    SceneShadingVertex vertex = gProbe.vertices.data[id.x];
    gDecoded[id.x * 3u] = float4(decodeSceneNormal(vertex.normal), 0.0);
    gDecoded[id.x * 3u + 1u] = decodeSceneTangent(vertex.tangent);
    gDecoded[id.x * 3u + 2u] = float4(vertex.texcoord, 0.0, 0.0);
}
