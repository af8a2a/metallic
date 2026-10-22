module StreamRTRepresentationProbe;

import Core;
using Metallic;
#include "../TestResourceParameters.h"

struct Observation
{
    uint directInstance;
    uint reflectedInstance;
    float directDistance;
    float reflectedDistance;
};

// Eight translated copies of the same bunny. The mirror is an analytic plane
// at z=.4; secondary rays use the reflected incoming direction and begin there.
[shader("compute")]
[numthreads(64, 1, 1)]
void streamRTRepresentationProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint instance = id.x / 256u;
    if (instance >= 8u) { return; }
    uint pixel = id.x % 256u;
    float2 xy = float2(-.016, .11) + (float2(pixel % 16u, pixel / 16u) + .5) / 16.0 * .18 - .09;
    xy.x += float(instance) * .3;
    RaytracingAccelerationStructure scene = resolveAccelerationStructure(getResourceParameters<UnifiedTopLevelProbeResources>().scene);
    RayDesc direct = {float3(xy, .5), .0001, float3(0, 0, -1), 1.0};
    RayQuery<RAY_FLAG_FORCE_OPAQUE> first;
    first.TraceRayInline(scene, 0, 255, direct);
    while (first.Proceed()) {}
    float3 incoming = float3(0, 0, 1);
    float3 reflected = reflect(incoming, float3(0, 0, -1));
    RayDesc secondary = {float3(xy, .4), .0001, reflected, 1.0};
    RayQuery<RAY_FLAG_FORCE_OPAQUE> second;
    second.TraceRayInline(scene, 0, 255, secondary);
    while (second.Proceed()) {}
    bool hit = first.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    bool reflectedHit = second.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    Observation result = {hit ? first.CommittedInstanceID() : 0xffffffffu,
        reflectedHit ? second.CommittedInstanceID() : 0xffffffffu,
        hit ? first.CommittedRayT() : 0.0,
        reflectedHit ? second.CommittedRayT() : 0.0};
    resolveBuffer<RWStructuredBuffer<Observation>>(getResourceParameters<UnifiedTopLevelProbeResources>().output)[id.x] = result;
}
