module TemplateCLASProbe;

import Core;
using Metallic;
#include "../TestResourceParameters.h"

struct TemplateProbeConstants
{
    uint64_t rays;
    uint count;
    uint reserved;
};

struct TemplateRayObservation
{
    uint hit, cluster, geometry, primitive;
    float distance, u, v, front;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void templateCLASProbeMain(uint3 id : SV_DispatchThreadID)
{
    let constants = getConstants<TemplateProbeConstants>();
    if (id.x >= constants.count) { return; }
    float4 input = ((float4*)constants.rays)[id.x];
    RayDesc ray = {input.xyz, 0.001, float3(0, 0, -1), 10.0};
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(resolveAccelerationStructure(getResourceParameters<UnifiedTopLevelProbeResources>().scene), 0, 1, ray);
    while (query.Proceed()) {}
    TemplateRayObservation result = (TemplateRayObservation)0;
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
    }
    resolveBuffer<RWStructuredBuffer<TemplateRayObservation>>(
        getResourceParameters<UnifiedTopLevelProbeResources>().output)[id.x] = result;
}
