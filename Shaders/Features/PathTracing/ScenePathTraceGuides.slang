#language slang 2026
module ScenePathTraceGuides;
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

import ScenePathTracing;
using Metallic.ScenePathTracing;

#define gDlssRrAlbedo resolveUniform(getSceneResourceParameters().guideAlbedo)

#define gDlssRrSpecularAlbedo resolveUniform(getSceneResourceParameters().guideSpecularAlbedo)

#define gDlssRrNormalRoughness resolveUniform(getSceneResourceParameters().guideNormalRoughness)

#define gDlssRrMotionVectors resolveUniform(getSceneResourceParameters().guideMotion)

#define gDlssRrLinearDepth resolveUniform(getSceneResourceParameters().guideLinearDepth)

#define gDlssRrSpecularHitDistance resolveUniform(getSceneResourceParameters().guideSpecularHitDistance)

#define gDlssDepth resolveUniform(getSceneResourceParameters().guideDepth)

import SceneStandardGuides;
using Metallic.SceneStandardGuides;

void writeDlssRrGuides(uint2 pixel, DLSSRRGuideData data, ScenePathTracePush push)
{
    gDlssRrAlbedo[pixel] = float4(saturate(data.albedo), 1.0);
    gDlssRrSpecularAlbedo[pixel] = float4(saturate(data.specularAlbedo), 1.0);
    gDlssRrNormalRoughness[pixel] = float4(Metallic.ScenePath::safeNormalize(data.normal, float3(0.0, 1.0, 0.0)), saturate(data.roughness));
    gDlssRrMotionVectors[pixel] = data.motionVector;
    gDlssRrLinearDepth[pixel] = max(data.linearDepth, 0.0);
    gDlssRrSpecularHitDistance[pixel] = max(data.specularHitDistance, 0.0);
    gDlssDepth[pixel] = cameraHardwareDepth(data.linearDepth, push);
}

DLSSRRGuideData evaluateStandardDlssRrGuides(
    Ray ray,
    RayCone rayCone,
    float2 samplePixel,
    ScenePathTracePush push,
    inout uint seed)
{
    DLSSRRGuideData data = makeDefaultDlssRrGuideData(push);

    HitInfo hit;
    if (!traceClosest(ray, push, hit)) {
        data.motionVector = skyMotionVectorToPrevious(
            ray.direction,
            samplePixel,
            push);
        return data;
    }

    return evaluateStandardDlssRrGuidesAtHit(ray, rayCone, samplePixel, push, seed, hit);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceGuidesMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    ScenePathTracePush push = withTextureFeedbackSampling(getConstants<ScenePathTracePush>(), dispatchThreadId.xy);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    uint samples = clamp(push.samples, 1u, 16u);
    uint seed = hash(
        pixel.x +
        pixel.y * push.width +
        (push.sampleFrame + 1u) * 0x9e3779b9u);
    float3 color = float3(0.0, 0.0, 0.0);
    DLSSRRGuideData guides = makeDefaultDlssRrGuideData(push);
    float firstSampleSpecularHitDistance = 0.0;
    PathTraceCacheContext cache = makeCacheContext();
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        float2 jitter = float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY);
        float2 samplePixel = float2(pixel) + jitter;

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
        float sampleSpecularHitDistance;
        Metallic.Atmosphere::AtmosphereSegment primaryAerial;
        color += pathTrace(ray, rayCone, push, cache, seed, sampleSpecularHitDistance, primaryAerial);

        if (sampleIndex == 0u) {
            firstSampleSpecularHitDistance = sampleSpecularHitDistance;
            guides = evaluateStandardDlssRrGuides(
                ray,
                rayCone,
                float2(pixel) + float2(0.5, 0.5),
                push,
                seed);
            guides.specularHitDistance = firstSampleSpecularHitDistance;
        }
    }
    color /= float(samples);
    color = max(color, float3(0.0, 0.0, 0.0));

    if (push.enableAccumulation != 0u) {
        color = accumulateStreamPathTrace(pixel, color, push);
    }

    sceneOutput()[pixel] = float4(color * (push.outputLinear != 0u ? 1.0 : lightingExposure(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getSceneResourceParameters().lights))), 1.0);
    writeDlssRrGuides(pixel, guides, push);
}
