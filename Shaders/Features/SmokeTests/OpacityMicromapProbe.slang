#include "../../Libraries/Resources/ComputeResources.slang"
#include "../PathTracing/ScenePathTrace.slang"

#define gProbeOutput METALLIC_RESOURCE(RWStructuredBuffer<uint2>, 63)
struct ProbePush { uint materialTextureCount; };
#define gProbePush METALLIC_CONSTANTS(ProbePush)

[shader("compute")]
[numthreads(8, 8, 1)]
void opacityMicromapProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= 64u)) { return; }
    float2 uv = (float2(id.xy) + float2(0.31, 0.67)) / 64.0;
    RayDesc ray;
    ray.Origin = float3(uv, 0.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.001;
    ray.TMax = 4.0;
    ScenePathTracePush push = {};
    push.materialTextureCount = gProbePush.materialTextureCount;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(gScene, RAY_FLAG_NONE, 0xff, ray);
    uint candidates = 0;
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            ++candidates;
            if (alphaMaskAcceptsHit(query.CandidateInstanceIndex(), query.CandidatePrimitiveIndex(),
                    query.CandidateTriangleBarycentrics(), push)) {
                query.CommitNonOpaqueTriangleHit();
            }
        }
    }
    gProbeOutput[id.y * 64u + id.x] = uint2(query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 1 : 0, candidates);
}
