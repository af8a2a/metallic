#include "OpenPBRRayQueryPathTrace.slang"

[[vk::binding(14, 0)]]
RWTexture2D<float4> gDlssRrAlbedo;

[[vk::binding(15, 0)]]
RWTexture2D<float4> gDlssRrSpecularAlbedo;

[[vk::binding(16, 0)]]
RWTexture2D<float4> gDlssRrNormalRoughness;

[[vk::binding(17, 0)]]
RWTexture2D<float2> gDlssRrMotionVectors;

[[vk::binding(18, 0)]]
RWTexture2D<float> gDlssRrLinearDepth;

[[vk::binding(19, 0)]]
RWTexture2D<float> gDlssRrSpecularHitDistance;

[[vk::binding(20, 0)]]
RWTexture2D<float> gDlssDepth;

struct DlssRrGuideData
{
    float3 albedo;
    float3 specularAlbedo;
    float3 normal;
    float roughness;
    float2 motionVector;
    float linearDepth;
    float specularHitDistance;
};

DlssRrGuideData makeDefaultDlssRrGuideData(ScenePathTracePush push)
{
    DlssRrGuideData data;
    data.albedo = float3(0.0, 0.0, 0.0);
    data.specularAlbedo = float3(0.0, 0.0, 0.0);
    data.normal = float3(0.0, 1.0, 0.0);
    data.roughness = 1.0;
    data.motionVector = float2(0.0, 0.0);
    data.linearDepth = max(push.clipOrtho.y, 0.0);
    data.specularHitDistance = 0.0;
    return data;
}

void writeDlssRrGuides(uint2 pixel, DlssRrGuideData data, ScenePathTracePush push)
{
    gDlssRrAlbedo[pixel] = float4(saturate(data.albedo), 1.0);
    gDlssRrSpecularAlbedo[pixel] = float4(saturate(data.specularAlbedo), 1.0);
    gDlssRrNormalRoughness[pixel] = float4(safeNormalize(data.normal, float3(0.0, 1.0, 0.0)), saturate(data.roughness));
    gDlssRrMotionVectors[pixel] = data.motionVector;
    gDlssRrLinearDepth[pixel] = max(data.linearDepth, 0.0);
    gDlssRrSpecularHitDistance[pixel] = max(data.specularHitDistance, 0.0);
    gDlssDepth[pixel] = cameraHardwareDepth(data.linearDepth, push);
}

DlssRrGuideData evaluateOpenPbrDlssRrGuides(
    Ray ray,
    RayCone rayCone,
    float2 samplePixel,
    ScenePathTracePush push,
    inout uint seed)
{
    DlssRrGuideData data = makeDefaultDlssRrGuideData(push);

    HitInfo hit;
    if (!traceClosest(ray, push, hit)) {
        data.motionVector = motionVectorToPrevious(
            ray.origin + ray.direction * max(push.clipOrtho.y, 1.0),
            samplePixel,
            push);
        return data;
    }

    rayCone = propagateRayCone(rayCone, hit.t);
    PathTraceMaterial material = gMaterials[hit.materialIndex];
    float4 normalSample = loadMaterialTextureForMaterial(
        material.normalTexture,
        hit.texcoord,
        float4(0.5, 0.5, 1.0, 1.0),
        push,
        rayCone,
        hit,
        ray.direction,
        materialNeedsDeterministicTextureMip0(material),
        seed);
    hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent, normalSample, material.textureParams.x);
    hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, ray.direction);

    OpenPBRMaterialEval materialEval = evaluateOpenPBRMaterial(
        hit,
        material,
        push,
        rayCone,
        ray.direction,
        seed);
    float metallic = saturate(materialEval.inputs.base_metalness);
    float roughness = clamp(materialEval.inputs.specular_roughness, 0.001, 1.0);
    float ior = clamp(materialEval.inputs.specular_ior, 1.0, 3.0);
    float dielectricF0 = (ior - 1.0) / (ior + 1.0);
    dielectricF0 *= dielectricF0;

    data.albedo = saturate(materialEval.inputs.base_color) * (1.0 - metallic);
    data.normal = safeNormalize(hit.normal, hit.geometryNormal);
    float nDotV = saturate(dot(data.normal, -ray.direction));
    float3 specularReflectance = lerp(
        saturate(materialEval.inputs.specular_color) * dielectricF0,
        saturate(materialEval.inputs.base_color),
        metallic);
    data.specularAlbedo = saturate(approxSpecularIntegralGGX(specularReflectance, roughness * roughness, nDotV));
    data.roughness = roughness;
    data.motionVector = motionVectorToPrevious(hit.position, samplePixel, push);
    data.linearDepth = cameraLinearDepth(hit.position, push);
    return data;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void openPbrRayQueryPathTraceGuidesMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                                        uniform ScenePathTracePush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    uint samples = clamp(push.samples, 1u, 16u);
    uint seed = hash(
        pixel.x +
        pixel.y * max(push.width, 1u) +
        (push.sampleFrame + 1u) * 0x9e3779b9u);
    float3 color = float3(0.0, 0.0, 0.0);
    DlssRrGuideData guides = makeDefaultDlssRrGuideData(push);
    float firstSampleSpecularHitDistance = 0.0;
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        float2 jitter = float2(0.5, 0.5) + float2(push.jitterOffsetX, push.jitterOffsetY);
        float2 samplePixel = float2(pixel) + jitter;

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
        float sampleSpecularHitDistance;
        color += pathTraceOpenPBR(ray, rayCone, push, seed, sampleSpecularHitDistance);

        if (sampleIndex == 0u) {
            firstSampleSpecularHitDistance = sampleSpecularHitDistance;
            guides = evaluateOpenPbrDlssRrGuides(
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
        if (push.hasHistory != 0u) {
            float accumulationWeight = 1.0 / float(push.accumulationFrame + 1u);
            float3 previousColor = gHistoryPrevious[pixel].rgb;
            color = lerp(previousColor, color, accumulationWeight);
        }
        gHistoryCurrent[pixel] = float4(color, 1.0);
    }

    gOutput[pixel] = float4(color, 1.0);
    writeDlssRrGuides(pixel, guides, push);
}
