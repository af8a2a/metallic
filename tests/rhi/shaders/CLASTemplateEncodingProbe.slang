module CLASTemplateEncodingProbe;

import Core;
using Metallic;
#include "../TestResourceParameters.h"

struct EncodingProbeConstants
{
    uint64_t rays, vertices, indices;
    uint count, verticesPerPatch;
    uint trianglesPerPatch, patchCount;
};

struct EncodingRay
{
    float3 origin; float minimum;
    float3 direction; float maximum;
};

struct EncodingObservation
{
    uint hit, cluster, geometry, primitive;
    float distance, u, v, front;
    float3 position; float sourceU;
    float3 normal; float sourceV;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void clasTemplateEncodingProbeMain(uint3 id : SV_DispatchThreadID)
{
    let p = getConstants<EncodingProbeConstants>();
    if (id.x >= p.count) { return; }
    let input = ((EncodingRay*)p.rays)[id.x];
    RayDesc ray = {input.origin, input.minimum, input.direction, input.maximum};
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(resolveAccelerationStructure(getResourceParameters<UnifiedTopLevelProbeResources>().scene), 0, 1, ray);
    while (query.Proceed()) {}
    EncodingObservation result = (EncodingObservation)0;
    result.cluster = result.geometry = result.primitive = 0xffffffffu;
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        result.hit = 1u;
        result.cluster = uint(query.CommittedClusterID());
        result.geometry = query.CommittedGeometryIndex();
        result.primitive = query.CommittedPrimitiveIndex();
        result.distance = query.CommittedRayT();
        float2 bary = query.CommittedTriangleBarycentrics();
        result.u = bary.x; result.v = bary.y;
        result.front = query.CommittedTriangleFrontFace() ? 1.0 : 0.0;
        // A broken native ID remains visible to the CPU assertion without
        // turning a diagnostic failure into an out-of-range shader read.
        if (result.cluster >= p.patchCount || result.primitive >= p.trianglesPerPatch) {
            resolveBuffer<RWStructuredBuffer<EncodingObservation>>(
                getResourceParameters<UnifiedTopLevelProbeResources>().output)[id.x] = result;
            return;
        }
        uint3 indices = ((uint3*)p.indices)[result.cluster * p.trianglesPerPatch + result.primitive];
        float4 a = ((float4*)p.vertices)[(result.cluster * p.verticesPerPatch + indices.x) * 2u];
        float4 b = ((float4*)p.vertices)[(result.cluster * p.verticesPerPatch + indices.y) * 2u];
        float4 c = ((float4*)p.vertices)[(result.cluster * p.verticesPerPatch + indices.z) * 2u];
        float3 sourceV = float3(
            ((float4*)p.vertices)[(result.cluster * p.verticesPerPatch + indices.x) * 2u + 1u].x,
            ((float4*)p.vertices)[(result.cluster * p.verticesPerPatch + indices.y) * 2u + 1u].x,
            ((float4*)p.vertices)[(result.cluster * p.verticesPerPatch + indices.z) * 2u + 1u].x);
        float3 weights = float3(1.0 - bary.x - bary.y, bary);
        result.position = input.origin + input.direction * result.distance;
        result.sourceU = dot(float3(a.w, b.w, c.w), weights);
        result.sourceV = dot(sourceV, weights);
        float3 normal = cross(b.xyz - a.xyz, c.xyz - a.xyz);
        result.normal = dot(normal, normal) > 1e-30 ? normalize(normal) : float3(0, 1, 0);
    }
    resolveBuffer<RWStructuredBuffer<EncodingObservation>>(
        getResourceParameters<UnifiedTopLevelProbeResources>().output)[id.x] = result;
}
