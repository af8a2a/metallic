// Compile and call the actual production shadow and RT candidate evaluators.
module MaterialCoverageProbe;
import Core;
import Material;
import ScenePathTypes;
import SceneSurface;
import ScenePathTracing;
import SceneShadowQueries;
using Metallic;
using Metallic.Material;
using Metallic.SceneSurface;
using Metallic.ScenePathTracing;
using Metallic.SceneShadowQueries;

// Compare the production specialized shadow hit against the previous full-hit
// classification contract. Opaque callers consume only t and materialIndex;
// transparent callers also consume position, geometry normal, UV and front face.
uint compareShadowHit(Ray ray, ScenePathTracePush push, float expectedDistance)
{
    HitInfo reference;
    bool referenceFound = traceClosest(ray, push, reference);
    bool referenceOpaque = false;
    if (referenceFound) {
        SceneMaterialData material = loadSceneMaterial(reference.materialIndex);
        referenceOpaque = material.textureParams.w <= 1.5 && material.glassParams.x == 0.0;
    }
    HitInfo actual;
    bool actualOpaque;
    bool actualFound = traceClosestShadow(ray, push, actualOpaque, actual);
    uint failures = referenceFound != (expectedDistance > 0.0) ? 1u : 0u;
    failures |= actualFound != referenceFound ? 2u : 0u;
    failures |= actualOpaque != referenceOpaque ? 4u : 0u;
    if (!referenceFound || !actualFound) { return failures; }
    failures |= !(abs(reference.t - expectedDistance) <= 1e-5) ? 8u : 0u;
    failures |= asuint(actual.t) != asuint(reference.t) ? 16u : 0u;
    failures |= actual.materialIndex != reference.materialIndex ? 32u : 0u;
    if (!referenceOpaque) {
        failures |= !all(asuint(actual.position) == asuint(reference.position)) ? 64u : 0u;
        failures |= !all(asuint(actual.geometryNormal) == asuint(reference.geometryNormal)) ? 128u : 0u;
        failures |= !all(asuint(actual.texcoord) == asuint(reference.texcoord)) ? 256u : 0u;
        failures |= actual.frontFace != reference.frontFace ? 512u : 0u;
    }
    return failures;
}

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
    Ray surfaceRay; surfaceRay.origin = ray.Origin; surfaceRay.direction = ray.Direction;
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(sceneAccelerationStructure(), RAY_FLAG_FORCE_NON_OPAQUE, 0xff, ray);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            streamRayAlphaAccepts(query.CandidateInstanceID(), uint(query.CandidateClusterID()),
                query.CandidatePrimitiveIndex(), query.CandidateTriangleBarycentrics(), surfaceRay,
                query.CandidateTriangleRayT(), push)) { query.CommitNonOpaqueTriangleHit(); }
    }
    float t = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? query.CommittedRayT() : 0.0;
    float shadow = traceGeometryShadow(ray.Origin, ray.Direction, 4.0);
    float opacity = evaluateSceneCoverage(0,sceneMaterialCoverageParameters(loadSceneMaterial(0)),uv,push).opacity;
    push.clipOrtho = float4(0.001, 4.0, 0.0, 0.0);
    uint failures = compareShadowHit(surfaceRay, push, t);
    Ray backRay; backRay.origin = float3(uv, 4); backRay.direction = float3(0, 0, -1);
    failures |= compareShadowHit(backRay, push, 1.0);
    Ray missRay = surfaceRay; missRay.origin.x += 2.0;
    failures |= compareShadowHit(missRay, push, 0.0);
    push.clipOrtho.y = 1.0;
    failures |= compareShadowHit(surfaceRay, push, 0.0);
    push.clipOrtho.y = 4.0;
    surfaceRay.direction *= 2.0;
    failures |= compareShadowHit(surfaceRay, push, t * 0.5);
    resolveUniform(getResourceParameters<SceneResourceParameters>().output)[id.xy] = float4(t, shadow, opacity, float(failures));
}
