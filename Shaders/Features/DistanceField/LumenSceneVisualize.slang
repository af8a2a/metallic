module LumenSceneVisualize;

import ShaderCore;
import ParameterRoot;
import GlobalDistanceField;
import LumenGlobalRay;

using Metallic;
using Metallic.DistanceField;
using Metallic.Lumen;

typealias LumenVisualizeTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;

struct LumenSceneVisualizeParams
{
    LumenGlobalTraceBindings scene;
    ResourceHandle<LumenVisualizeTexture> sceneRadiance;
    ResourceHandle<LumenVisualizeTexture> geometryNormals;
    ResourceHandle<LumenVisualizeTexture> surfaceCache;
    uint clipmapCount;
    uint padding;
    ViewCameraConstants camera;
};

LumenGlobalTraceRay makeVisualizeCameraRay(uint2 pixel, uint2 extent, ViewCameraConstants camera)
{
    float2 viewportUV = (float2(pixel) + 0.5f) / float2(extent);
    // UE 5.7.4 LumenVisualize.usf, VisualizeQuadsCS. RenderView supplies an unjittered camera.
    float2 screenPosition = float2(2.0f, -2.0f) * viewportUV + float2(-1.0f, 1.0f);
    float3 forward = normalize(camera.center.xyz - camera.eye.xyz);
    float3 right = normalize(cross(forward, camera.upProjection.xyz));
    float3 up = cross(right, forward);
    float3 lateral = right * (screenPosition.x * camera.viewport.x) + up * screenPosition.y;

    LumenGlobalTraceRay ray;
    if (camera.upProjection.w > 0.5f) {
        float3 origin = camera.eye.xyz + forward * camera.clipOrtho.x + lateral * (0.5f * camera.clipOrtho.z);
        ray.originAndMinDistance = float4(origin, 0.0f);
        ray.directionAndMaxDistance = float4(forward, camera.clipOrtho.y - camera.clipOrtho.x);
    } else {
        float3 direction = normalize(forward + lateral * tan(0.5f * camera.viewport.w));
        ray.originAndMinDistance = float4(camera.eye.xyz, camera.clipOrtho.x);
        ray.directionAndMaxDistance = float4(direction, camera.clipOrtho.y);
    }
    return ray;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void lumenSceneVisualizeMain(uint3 thread : SV_DispatchThreadID)
{
    LumenSceneVisualizeParams params = getParameters<LumenSceneVisualizeParams>();
    LumenVisualizeTexture sceneRadiance = resolveUniform(params.sceneRadiance);
    uint2 extent;
    sceneRadiance.GetDimensions(extent.x, extent.y);
    if (any(thread.xy >= extent)) {
        return;
    }

    LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
    LumenGlobalTraceRay ray = makeVisualizeCameraRay(thread.xy, extent, params.camera);
    LumenGlobalTraceResult trace = traceLumenGlobalRay(scene, ray);
    bool geometryHit = trace.state.x == kGlobalDistanceFieldHit;
    float3 geometryNormal = float3(0.0f);
    float3 coverage = trace.radiance.xyz;
    if (geometryHit) {
        // UE 5.7.4 EvaluateRayHitFromCardSampleAccumulator. The fixed card atlas
        // exposes coverage and weight; it has no independent mesh-card cull flag.
        geometryNormal = trace.normal.xyz * 0.5f + 0.5f;
        if (trace.metrics.w > 0.0f && trace.metrics.z <= 0.0f) {
            coverage = float3(1.0f, 0.0f, 1.0f);
        }
    }

    sceneRadiance[thread.xy] = float4(trace.radiance.xyz, 1.0f);
    resolveUniform(params.geometryNormals)[thread.xy] = float4(geometryNormal, 1.0f);
    resolveUniform(params.surfaceCache)[thread.xy] = float4(coverage, 1.0f);
}
