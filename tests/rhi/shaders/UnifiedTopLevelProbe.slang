import Core;
using Metallic;

[shader("compute")]
[numthreads(3, 1, 1)]
void unifiedTopLevelMain(uint3 id : SV_DispatchThreadID)
{
    RayDesc ray = {float3(id.x == 1 ? 3.0 : 0.25, 0.25, 0), 0.001, float3(0, 0, 1), 10.0};
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(getResource<RaytracingAccelerationStructure>(0), 0, id.x == 2 ? 2 : 1, ray);
    while (query.Proceed()) {}
    bool hit = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    getResource<RWStructuredBuffer<uint2>>(1)[id.x] = uint2(hit ? 1 : 0, hit ? query.CommittedInstanceID() : 0xffffffffu);
}
