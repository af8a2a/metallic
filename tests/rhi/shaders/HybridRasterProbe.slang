import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

#include "../HybridProbeParameters.h"
struct Vertex { float4 position : SV_Position; nointerpolation uint id : TEXCOORD0; };
[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void probeMeshMain(uint3 group : SV_GroupID, out vertices Vertex outputVertices[3],
    out indices uint3 triangles[1], uniform HybridMeshProbeParameters push)
{
    if (group.x >= push.vertices.length() / 3u) { SetMeshOutputCounts(0, 0); return; }
    float4* input = push.vertices.data;
    float4 a = input[group.x * 3u];
    float4 b = input[group.x * 3u + 1u];
    float4 c = input[group.x * 3u + 2u];
    bool queued = hybridEnqueueTriangle(a, b, c, group.x + 1u, push.doubleSided != 0,
        push.queue, push.enabled != 0u);
    SetMeshOutputCounts(3, 1);
    outputVertices[0].position = a;
    outputVertices[1].position = b;
    outputVertices[2].position = c;
    for (uint i = 0; i < 3; ++i) { outputVertices[i].id = group.x + 1u; }
    triangles[0] = queued ? uint3(0u) : uint3(0, 1, 2);
}
[shader("fragment")]
uint probeFragmentMain(Vertex input) : SV_Target0 { return input.id; }
