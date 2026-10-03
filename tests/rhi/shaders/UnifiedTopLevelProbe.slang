import Core;
using Metallic;
#include "../TestResourceParameters.h"

struct RayObservation
{
    uint hit, instance, primitive, front;
    float distance, u, v, padding;
};

[shader("compute")]
[numthreads(6, 1, 1)]
void unifiedTopLevelMain(uint3 id : SV_DispatchThreadID)
{
    float x = id.x == 1 ? 0.65 : id.x == 2 ? 3.0 : 0.2;
    float y = id.x == 1 ? 0.15 : id.x == 2 ? 0.2 : 0.3;
    RayDesc ray = {float3(x, y, id.x == 5 ? 5.0 : 0), 0.001,
        float3(0, 0, id.x == 5 ? -1.0 : 1.0), id.x == 4 ? 1.0 : 10.0};
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(resolveAccelerationStructure(getResourceParameters<UnifiedTopLevelProbeResources>().scene), 0, id.x == 3 ? 2 : 1, ray);
    while (query.Proceed()) {}
    bool hit = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    RayObservation result = {hit ? 1u : 0u, hit ? query.CommittedInstanceID() : 0xffffffffu,
        hit ? query.CommittedPrimitiveIndex() : 0xffffffffu,
        hit && query.CommittedTriangleFrontFace() ? 1u : 0u,
        hit ? query.CommittedRayT() : 0.0, 0.0, 0.0, 0.0};
    if (hit) { float2 bary = query.CommittedTriangleBarycentrics(); result.u = bary.x; result.v = bary.y; }
    resolveBuffer<RWStructuredBuffer<RayObservation>>(getResourceParameters<UnifiedTopLevelProbeResources>().output)[id.x] = result;
}
