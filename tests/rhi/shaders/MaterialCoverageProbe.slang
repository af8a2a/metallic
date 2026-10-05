// Compile and call the actual production shadow and RT candidate evaluators.
#include "../../../Shaders/Features/Lighting/ScreenSpaceShadows.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void materialCoverageProbeMain(uint3 id : SV_DispatchThreadID)
{
    float2 uv = 1.0 - (float2(id.xy) + 0.5) / 64.0;
    RayDesc ray;
    ray.Origin = float3(uv, 0); ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001; ray.TMax = 4;
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.materialTextureCount = getConstants<ShadowGeometryPush>().materialTextureCount;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(gScene, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            alphaMaskAcceptsHit(query.CandidateInstanceIndex(), query.CandidatePrimitiveIndex(),
                query.CandidateTriangleBarycentrics(), push)) { query.CommitNonOpaqueTriangleHit(); }
    }
    float t = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? query.CommittedRayT() : 0.0;
    float shadow = traceGeometryShadow(ray.Origin, ray.Direction, 4.0);
    float opacity = evaluateSceneCoverage(0,loadSceneMaterial(0),uv,push).opacity;
    resolveUniform(getResourceParameters<SceneResourceParameters>().output)[id.xy] = float4(t, shadow, opacity, 1);
}
