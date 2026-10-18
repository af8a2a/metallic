#language slang 2026
module VkGltfRayPipelineHit;
import Core;
import ScenePathTypes;
import ExperimentalRayTraversal;
import StreamRayHit;
using Metallic;
using Metallic.StreamGeometry;

[shader("miss")]
void vkGltfRayPipelineMiss(inout StreamRayHit payload)
{
    payload.t = -1.0;
}

[shader("closesthit")]
void vkGltfRayPipelineClosestHit(inout StreamRayHit payload, BuiltInTriangleIntersectionAttributes attributes)
{
    payload.t = RayTCurrent();
    payload.instanceId = InstanceID();
    payload.clusterId = uint(GetClusterID());
    payload.primitiveId = PrimitiveIndex();
    payload.barycentrics = attributes.barycentrics;
}

[shader("anyhit")]
void vkGltfRayPipelineAnyHit(inout StreamRayHit payload, BuiltInTriangleIntersectionAttributes attributes)
{
    Ray ray;
    ray.origin = WorldRayOrigin();
    ray.direction = WorldRayDirection();
    if (!Metallic.ExperimentalPath::alphaAccepts(InstanceID(), uint(GetClusterID()), PrimitiveIndex(),
        attributes.barycentrics, ray, RayTCurrent(), withTextureFeedbackSampling(getConstants<ScenePathTracePush>(), DispatchRaysIndex().xy))) { IgnoreHit(); }
}
