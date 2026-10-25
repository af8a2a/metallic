module CLASTemplateTraversalProbe;
import Core;
using Metallic;
#include "../TestResourceParameters.h"

struct TraversalConstants
{
    uint64_t rays;
    uint width, height;
    uint selection, padding;
};
struct CapturedShadowRay
{
    float4 originTMin, directionTMax;
    uint4 control;
    float4 distances;
    uint4 ids;
};
struct TraversalObservation
{
    uint hit, proceedCalls, candidates, cluster;
    float distance;
    uint primitive, instance, accepted;
};
[shader("compute")]
[numthreads(8, 8, 1)]
void clasTemplateTraversalProbeMain(uint3 id : SV_DispatchThreadID)
{
    let p = getConstants<TraversalConstants>();
    if (id.x >= p.width || id.y >= p.height) { return; }
    uint index = id.y * p.width + id.x;
    let source = ((CapturedShadowRay*)p.rays)[index];
    TraversalObservation result = (TraversalObservation)0;
    result.distance = -1;
    result.cluster = result.primitive = result.instance = 0xffffffffu;
    if (source.control.x != 0 && (p.selection == 0 ||
        (p.selection == 1 && source.control.y != 0) || (p.selection == 2 && source.control.y == 0))) {
        RayDesc ray = {source.originTMin.xyz, source.originTMin.w, source.directionTMax.xyz, source.directionTMax.w};
        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(resolveAccelerationStructure(getResourceParameters<UnifiedTopLevelProbeResources>().scene),
            RAY_FLAG_NONE, 0xff, ray);
        for (;;) {
            ++result.proceedCalls;
            if (!query.Proceed()) { break; }
            if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
                ++result.candidates;
                // The reader requires the captured fixed scene to be FORCE_OPAQUE.
                // Any candidate is therefore a visible diagnostic contract failure.
            }
        }
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            result.hit = 1;
            result.distance = query.CommittedRayT();
            result.cluster = uint(query.CommittedClusterID());
            result.primitive = query.CommittedPrimitiveIndex();
            result.instance = query.CommittedInstanceID();
        }
    }
    resolveBuffer<RWStructuredBuffer<TraversalObservation>>(
        getResourceParameters<UnifiedTopLevelProbeResources>().output)[index] = result;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void clasTemplateTraversalTimingMain(uint3 id : SV_DispatchThreadID)
{
    let p = getConstants<TraversalConstants>();
    if (id.x >= p.width || id.y >= p.height) { return; }
    uint index = id.y * p.width + id.x;
    let source = ((CapturedShadowRay*)p.rays)[index];
    float distance = -1;
    if (source.control.x != 0 && (p.selection == 0 ||
        (p.selection == 1 && source.control.y != 0) || (p.selection == 2 && source.control.y == 0))) {
        RayDesc ray = {source.originTMin.xyz, source.originTMin.w, source.directionTMax.xyz, source.directionTMax.w};
        RayQuery<RAY_FLAG_NONE> query;
        query.TraceRayInline(resolveAccelerationStructure(getResourceParameters<UnifiedTopLevelProbeResources>().scene),
            RAY_FLAG_NONE, 0xff, ray);
        // The host and the separate correctness dispatch establish FORCE_OPAQUE.
        // Keep the closest-hit query; timing does not inspect IDs or candidates.
        while (query.Proceed()) { }
        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) { distance = query.CommittedRayT(); }
    }
    resolveBuffer<RWStructuredBuffer<float>>(
        getResourceParameters<UnifiedTopLevelProbeResources>().output)[index] = distance;
}
