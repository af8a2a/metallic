#language slang 2026
module SceneRealtimeLighting;
import Core;
import Material;
import MaterialProgram;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import Lighting;
import ScenePathTypes;
import ScenePathMath;
import SceneSurface;
import Atmosphere;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.SceneSurface;
using Metallic.ScenePath;
import OpenPBR;
import OpenPBRClosure;
import OpenPBRSurface;
import SceneSurfaceMaterial;
import SurfaceLighting;
import ScenePathTracing;
import SHARC;
#if METALLIC_CUSTOM_MATERIALS
import SlabClosure;
#endif
using Metallic.OpenPBR;
using Metallic.SceneOpenPBR;
using Metallic.ScenePathTracing;
using Metallic.SHARC;

import OpenPBRDirectLighting;
using Metallic.SceneDirectLighting;

// Ray-query primary visibility, shared OpenPBR punctual lighting and SH GI.
[shader("compute")]
[numthreads(8, 8, 1)]
void sceneRealtimeLightingMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    ScenePathTracePush push = withTextureFeedbackSampling(getConstants<ScenePathTracePush>(), dispatchThreadId.xy);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) { return; }
    float2 samplePixel = float2(pixel) + 0.5;
    Ray ray;
    ray.origin = rayOriginFromPixel(samplePixel, push);
    ray.direction = rayDirectionFromPixel(samplePixel, push);
    HitInfo hit;
    uint seed = hash(pixel.x + pixel.y * push.width);
    bool hasSurface = traceClosest(ray, push, hit);
    float3 radiance = hasSurface
        ? shadeOpenPBRDirectSurface(ray, hit, samplePixel, push, seed, false)
        : (push.debugView != kOpenPBRDebugViewFinal ? float3(0.0) : environmentRadiance(ray.direction, push, true, ray.origin));
    if (push.outputLinear != 0u) {
        sceneOutput()[pixel] = float4(max(radiance, float3(0.0)), 1.0);
        return;
    }
    float3 exposed = max(Metallic.WorkingColor::toLinearRec709(radiance) * lightingExposure(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights)), float3(0.0));
    float3 color = pow(exposed / (exposed + 1.0), 1.0 / 2.2);
    sceneOutput()[pixel] = float4(saturate(color), 1.0);
}
