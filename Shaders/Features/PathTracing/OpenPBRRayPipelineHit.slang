#language slang 2026
module OpenPBRRayPipelineHit;
import Core;
import ScenePathTypes;
import ScenePathTracing;
import StreamRayHit;
using Metallic;
using Metallic.ScenePathTracing;
using Metallic.StreamGeometry;

#if !METALLIC_PT_RAY_PIPELINE
#error OpenPBR ray-pipeline entry requires the global METALLIC_PT_RAY_PIPELINE request policy.
#endif

[shader("miss")]
void openPbrRayPipelineMiss(inout StreamRayHit payload)
{
    payload.t = -1.0;
}

[shader("closesthit")]
void openPbrRayPipelineClosestHit(inout StreamRayHit payload, BuiltInTriangleIntersectionAttributes attributes)
{
    payload.t = RayTCurrent();
    payload.instanceId = InstanceID();
    payload.clusterId = uint(GetClusterID());
    payload.primitiveId = PrimitiveIndex();
    payload.barycentrics = attributes.barycentrics;
}

[shader("anyhit")]
void openPbrRayPipelineAnyHit(inout StreamRayHit payload, BuiltInTriangleIntersectionAttributes attributes)
{
    Ray ray;
    ray.origin = WorldRayOrigin();
    ray.direction = WorldRayDirection();
    // Match the inline-query coverage test. BLEND randomness belongs to the
    // integrator after the closest accepted hit; no RNG is consumed here.
    // Shadow queries only override clip distances, which this test does not use.
    if (!streamRayAlphaAccepts(InstanceID(), uint(GetClusterID()), PrimitiveIndex(),
            attributes.barycentrics, ray, RayTCurrent(), withTextureFeedbackSampling(getConstants<ScenePathTracePush>(), DispatchRaysIndex().xy))) {
        IgnoreHit();
    }
}
