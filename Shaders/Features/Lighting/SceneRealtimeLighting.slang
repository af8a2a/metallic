#include "../../Libraries/Resources/ComputeResources.slang"

#include "OpenPBRDirectLighting.slang"

// Ray-query primary visibility, shared OpenPBR punctual lighting and SH GI.
[shader("compute")]
[numthreads(8, 8, 1)]
void sceneRealtimeLightingMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    ScenePathTracePush push = METALLIC_CONSTANTS(ScenePathTracePush);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) { return; }
    float2 samplePixel = float2(pixel) + 0.5;
    Ray ray;
    ray.origin = rayOriginFromPixel(samplePixel, push);
    ray.direction = rayDirectionFromPixel(samplePixel, push);
    HitInfo hit;
    uint seed = hash(pixel.x + pixel.y * push.width);
    float3 radiance = traceClosest(ray, push, hit)
        ? shadeOpenPBRDirectSurface(ray, hit, samplePixel, push, seed, false)
        : environmentRadiance(ray.direction, push, true);
    if (push.outputLinear != 0u) {
        gOutput[pixel] = float4(max(radiance, float3(0.0)), 1.0);
        return;
    }
    float3 exposed = max(radiance * lightingExposure(), float3(0.0));
    float3 color = pow(exposed / (exposed + 1.0), 1.0 / 2.2);
    gOutput[pixel] = float4(saturate(color), 1.0);
}
