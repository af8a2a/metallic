implementing OpenPBRPathTracing;

import Core;
import MaterialTextureDecode;
import RTXCRMaterialAdapter;
import RayMaterialExecution;
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

namespace Metallic.OpenPBRPathTracing
{


public struct DLSSRRGuideData
{
    public float3 albedo;
    public float3 specularAlbedo;
    public float3 normal;
    public float roughness;
    public float2 motionVector;
    public float linearDepth;
    public float specularHitDistance;
};

public DLSSRRGuideData makeDefaultDlssRrGuideData(ScenePathTracePush push)
{
    DLSSRRGuideData data;
    data.albedo = float3(0.0, 0.0, 0.0);
    data.specularAlbedo = float3(0.0, 0.0, 0.0);
    data.normal = float3(0.0, 1.0, 0.0);
    data.roughness = 1.0;
    data.motionVector = float2(0.0, 0.0);
    data.linearDepth = max(push.clipOrtho.y, 0.0);
    data.specularHitDistance = 0.0;
    return data;
}

public void writeDlssRrGuides(uint2 pixel, DLSSRRGuideData data, ScenePathTracePush push)
{
    let resources = getResourceParameterField<SceneResourceParameters, SceneGuideResourceParameters>(kSceneGuidesOffset);
    resolveUniform(resources.guideAlbedo)[pixel] = float4(saturate(data.albedo), 1.0);
    resolveUniform(resources.guideSpecularAlbedo)[pixel] = float4(saturate(data.specularAlbedo), 1.0);
    resolveUniform(resources.guideNormalRoughness)[pixel] = float4(Metallic.ScenePath::safeNormalize(data.normal, float3(0.0, 1.0, 0.0)), saturate(data.roughness));
    resolveUniform(resources.guideMotion)[pixel] = data.motionVector;
    resolveUniform(resources.guideLinearDepth)[pixel] = max(data.linearDepth, 0.0);
    resolveUniform(resources.guideSpecularHitDistance)[pixel] = max(data.specularHitDistance, 0.0);
    resolveUniform(resources.guideDepth)[pixel] = cameraHardwareDepth(data.linearDepth, push);
}

public DLSSRRGuideData evaluateOpenPbrDlssRrGuides(
    Ray ray,
    RayCone rayCone,
    float2 samplePixel,
    ScenePathTracePush push,
    inout uint seed,
    bool primaryFound,
    HitInfo hit)
{
    DLSSRRGuideData data = makeDefaultDlssRrGuideData(push);

    if (!primaryFound) {
        data.motionVector = skyMotionVectorToPrevious(
            ray.direction,
            samplePixel,
            push);
        return data;
    }

    rayCone = propagateRayCone(rayCone, hit.t);
    SceneSurfaceSummary summary = evaluateSceneSurfaceGuides(hit, push, rayCone, ray.direction, seed);
    float roughness = clamp(summary.roughness, 0.001, 1.0);
    data.albedo = summary.albedo;
    data.normal = Metallic.ScenePath::safeNormalize(hit.normal, hit.geometryNormal);
    float nDotV = saturate(dot(data.normal, -ray.direction));
    data.specularAlbedo = saturate(approxSpecularIntegralGGX(summary.guideSpecularReflectance, roughness * roughness, nDotV));
    data.roughness = roughness;
    data.motionVector = motionVectorToPrevious(hit.position, samplePixel, push);
    data.linearDepth = cameraLinearDepth(hit.position, push);
    return data;
}

#if !SHARC_UPDATE
// Static guides are consumed before any deep path. Only the first specular
// endpoint depends on continuation and is stored after it completes.
public void writeDlssRrPrimaryGuides(uint2 pixel, DLSSRRGuideData data, ScenePathTracePush push)
{
    let resources = getResourceParameterField<SceneResourceParameters, SceneGuideResourceParameters>(kSceneGuidesOffset);
    resolveUniform(resources.guideAlbedo)[pixel] = float4(saturate(data.albedo), 1.0);
    resolveUniform(resources.guideSpecularAlbedo)[pixel] = float4(saturate(data.specularAlbedo), 1.0);
    resolveUniform(resources.guideNormalRoughness)[pixel] = float4(Metallic.ScenePath::safeNormalize(data.normal, float3(0.0, 1.0, 0.0)), saturate(data.roughness));
    resolveUniform(resources.guideMotion)[pixel] = data.motionVector;
    resolveUniform(resources.guideLinearDepth)[pixel] = max(data.linearDepth, 0.0);
    resolveUniform(resources.guideDepth)[pixel] = cameraHardwareDepth(data.linearDepth, push);
}

#if !SHARC_UPDATE
public void evaluateAndWriteOpenPbrPrimaryGuides(uint2 pixel, Ray ray, RayCone rayCone,
    ScenePathTracePush push, bool found, HitInfo primaryHit)
{
    // This stream and material program belong to guides. Do not consume the
    // path seed or reuse its closure: texture filtering and normal policies differ.
    uint guideSeed = hash(pixel.x + pixel.y * max(push.width, 1u) +
        (push.sampleFrame + 1u) * 0x85ebca6bu);
    DLSSRRGuideData guides = evaluateOpenPbrDlssRrGuides(ray, rayCone,
        float2(pixel) + float2(0.5), push, guideSeed, found, primaryHit);
    writeDlssRrPrimaryGuides(pixel, guides, push);
}
#endif

// Keep the established one-ray stage unchanged; only authored hit data escapes.
public bool queryOpenPbrPrimaryStage(uint2 pixel, Ray ray, RayCone rayCone,
    ScenePathTracePush push, out HitInfo primaryHit)
{
    bool found = traceClosest(ray, push, primaryHit);
    uint guideSeed = hash(pixel.x + pixel.y * max(push.width, 1u) +
        (push.sampleFrame + 1u) * 0x85ebca6bu);
    DLSSRRGuideData guides = evaluateOpenPbrDlssRrGuides(ray, rayCone,
        float2(pixel) + float2(0.5), push, guideSeed, found, primaryHit);
    writeDlssRrPrimaryGuides(pixel, guides, push);
    return found;
}


public void writeOpenPbrPathColor(uint2 pixel, float3 color, ScenePathTracePush push)
{
    color = push.debugView != kOpenPBRDebugViewFinal
        ? openPbrDisplayDebugColor(push, color) : max(color, float3(0.0));

    if (push.enableAccumulation != 0u) {
        color = accumulateStreamPathTrace(pixel, color, push);
    }

    sceneOutput()[pixel] = float4(color * (push.outputLinear != 0u ? 1.0 : lightingExposure(sceneLights())), 1.0);
}
#endif

public void dispatchOpenPbrRayQueryPathTraceGuides(uint3 dispatchThreadId)
{
#if METALLIC_PT_GUIDE_SPECIALIZATION
    dispatchOpenPbrRayQueryPathTracePolicy(dispatchThreadId, getConstants<ScenePathTracePush>(), kOpenPBRGuideExportEnabled);
#else
#if SHARC_UPDATE
    // Sparse cache training never touches the full-resolution RR inputs.
    dispatchOpenPbrRayQueryPathTrace(dispatchThreadId);
#else
    ScenePathTracePush push = getConstants<ScenePathTracePush>();
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    uint samples = clamp(push.samples, 1u, 16u);
    uint seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (push.sampleFrame + 1u) * 0x9e3779b9u);
    // RR samples share the jittered camera ray, but retain independent BSDF,
    // coverage and texture RNG consumption. Share only authored hit data;
    // guides still evaluate their own material and face-forward policy.
    float2 jitter = float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY);
    float2 samplePixel = float2(pixel) + jitter;
    Ray ray;
    ray.origin = rayOriginFromPixel(samplePixel, push);
    ray.direction = rayDirectionFromPixel(samplePixel, push);
    RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
#if METALLIC_PT_SINGLE_SAMPLE
    {
        float specularHitDistance;
        float3 color = pathTraceOpenPbrSingleSample(pixel, ray, rayCone, push, seed, specularHitDistance);
        writeOpenPbrPathColor(pixel, color, push);
        let resources = getResourceParameterField<SceneResourceParameters, SceneGuideResourceParameters>(kSceneGuidesOffset);
        resolveUniform(resources.guideSpecularHitDistance)[pixel] = max(specularHitDistance, 0.0);
    }
#else
    {
        // Multi-sample pixels retain the general authored-primary reuse path.
        float3 color = float3(0.0);
        float firstSampleSpecularHitDistance = 0.0;
        OpenPBRPrimaryQuery primaryQuery = {};
        for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
            float sampleSpecularHitDistance;
            color += pathTraceOpenPBRWithPrimaryQuery(ray, rayCone, push, seed,
                sampleSpecularHitDistance, primaryQuery, pixel);

            if (sampleIndex == 0u) {
                firstSampleSpecularHitDistance = sampleSpecularHitDistance;
            }
        }
        color /= float(samples);
        writeOpenPbrPathColor(pixel, color, push);
        let resources = getResourceParameterField<SceneResourceParameters, SceneGuideResourceParameters>(kSceneGuidesOffset);
        resolveUniform(resources.guideSpecularHitDistance)[pixel] = max(firstSampleSpecularHitDistance, 0.0);
    }
#endif
#endif
#endif
}
} // namespace Metallic.OpenPBRPathTracing
