#include "../../../Source/Runtime/Render/Core/SceneProbeParameters.h"
using Metallic;
[[vk::push_constant]] ConstantBuffer<ScenePositionProbeParameters> gProbePush;
#define gPathTraceParameters (*gProbePush.resources)
#include "../PathTracing/ScenePathTrace.slang"
#define gProbeOutput gProbePush.output.data

[shader("compute")]
[numthreads(4, 1, 1)]
void scenePositionFetchProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= 4u || !gProbePush.output.contains(id.x * 6u + 5u)) { return; }
    // Two indexed triangles, a back-face hit, and a miss. The fixture has a
    // nonuniform object transform, so treating fetched positions as world space
    // or changing the authored normal orientation cannot pass unnoticed.
    Ray ray;
    ray.origin = float3(id.x == 1u ? 6.0 : 4.0, id.x == 1u ? 0.25 : -1.25, 10.0);
    ray.direction = float3(0.0, 0.0, -1.0);
    ray.origin.x += gProbePush.translationX;
    if (id.x == 2u) {
        ray.origin.z = -10.0;
        ray.direction.z = 1.0;
    } else if (id.x == 3u) {
        ray.origin.y = 20.0;
    }
    ScenePathTracePush push = {};
    push.clipOrtho = float4(0.001, 100.0, 0.0, 0.0);
    push.bitangentFlip = 1.0;
    HitInfo hit;
    bool found = traceClosest(ray, push, hit);
    uint base = id.x * 6u;
    gProbeOutput[base + 0u] = float4(hit.position, hit.t);
    gProbeOutput[base + 1u] = float4(hit.normal, hit.frontFace ? 1.0 : 0.0);
    gProbeOutput[base + 2u] = float4(hit.geometryNormal, found ? 1.0 : 0.0);
    gProbeOutput[base + 3u] = hit.tangent;
    gProbeOutput[base + 4u] = float4(hit.bitangent, float(hit.triangleIndex));
    gProbeOutput[base + 5u] = float4(hit.texcoord, float(hit.materialIndex), float(hit.instanceIndex));
    if (id.x == 3u) {
        PathTraceInstance instance;
        PathTracePrimitive primitive;
        uint3 indices;
        gProbeOutput[base + 5u] = float4(
            !loadPathTraceTriangle(0xffffffffu, 0u, instance, primitive, indices),
            !loadPathTraceTriangle(gInstances.length(), 0u, instance, primitive, indices),
            !loadPathTraceTriangle(0u, 0xffffffffu, instance, primitive, indices),
            !loadPathTraceTriangle(0u, 0x55555556u, instance, primitive, indices));
    }
}
