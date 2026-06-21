#include "scene_path_trace.slang"

[[vk::binding(14, 0)]]
RWTexture2D<float4> gDlssRrAlbedo;

[[vk::binding(15, 0)]]
RWTexture2D<float4> gDlssRrSpecularAlbedo;

[[vk::binding(16, 0)]]
RWTexture2D<float4> gDlssRrNormalRoughness;

[[vk::binding(17, 0)]]
RWTexture2D<float4> gDlssRrMotionVectors;

[[vk::binding(18, 0)]]
RWTexture2D<float> gDlssRrLinearDepth;

[[vk::binding(19, 0)]]
RWTexture2D<float> gDlssRrSpecularHitDistance;

struct DlssRrGuideData
{
    float3 albedo;
    float3 specularAlbedo;
    float3 normal;
    float roughness;
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
    data.linearDepth = max(push.clipOrtho.y, 0.0);
    data.specularHitDistance = 0.0;
    return data;
}

void writeDlssRrGuides(uint2 pixel, DlssRrGuideData data)
{
    float3 packedNormal = saturate(data.normal * 0.5 + 0.5);
    gDlssRrAlbedo[pixel] = float4(saturate(data.albedo), 1.0);
    gDlssRrSpecularAlbedo[pixel] = float4(saturate(data.specularAlbedo), 1.0);
    gDlssRrNormalRoughness[pixel] = float4(packedNormal, saturate(data.roughness));
    gDlssRrMotionVectors[pixel] = float4(0.0, 0.0, 0.0, 0.0);
    gDlssRrLinearDepth[pixel] = max(data.linearDepth, 0.0);
    gDlssRrSpecularHitDistance[pixel] = max(data.specularHitDistance, 0.0);
}

DlssRrGuideData evaluateStandardDlssRrGuides(
    Ray ray,
    RayCone rayCone,
    ScenePathTracePush push,
    inout uint seed)
{
    DlssRrGuideData data = makeDefaultDlssRrGuideData(push);

    HitInfo hit;
    if (!traceClosest(ray, push, hit)) {
        return data;
    }

    rayCone = propagateRayCone(rayCone, hit.t);
    PathTraceMaterial material = gMaterials[hit.materialIndex];
    bool deterministicGlassTextures = materialNeedsDeterministicTextureMip0(material);
    float4 baseColorSample = loadMaterialTextureForMaterial(
        material.baseColorTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        ray.direction,
        deterministicGlassTextures,
        seed);
    float4 metallicRoughnessSample = loadMaterialTextureForMaterial(
        material.metallicRoughnessTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        ray.direction,
        deterministicGlassTextures,
        seed);
    float4 normalSample = loadMaterialTextureForMaterial(
        material.normalTexture,
        hit.texcoord,
        float4(0.5, 0.5, 1.0, 1.0),
        push,
        rayCone,
        hit,
        ray.direction,
        deterministicGlassTextures,
        seed);

    hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent, normalSample, material.textureParams.x);
    hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, ray.direction);

    float3 baseColor = saturate(material.baseColor.rgb * srgbToLinear(baseColorSample.rgb));
    float metallic = saturate(material.params.x * metallicRoughnessSample.b);
    float roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.02, 1.0);
    float ior = clamp(material.glassParams.y, 1.0, 3.0);
    float dielectricF0 = (ior - 1.0) / (ior + 1.0);
    dielectricF0 *= dielectricF0;

    data.albedo = baseColor * (1.0 - metallic);
    data.specularAlbedo = lerp(float3(dielectricF0, dielectricF0, dielectricF0), baseColor, metallic);
    data.normal = safeNormalize(hit.normal, hit.geometryNormal);
    data.roughness = roughness;
    data.linearDepth = hit.t;
    data.specularHitDistance = hit.t;
    return data;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceGuidesMain(uint3 dispatchThreadId : SV_DispatchThreadID,
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
        (push.accumulationFrame + 1u) * 0x9e3779b9u);
    float3 color = float3(0.0, 0.0, 0.0);
    DlssRrGuideData guides = makeDefaultDlssRrGuideData(push);
    for (uint sampleIndex = 0u; sampleIndex < samples; ++sampleIndex) {
        float2 jitter = samples == 1u ? float2(0.5, 0.5) : float2(rand(seed), rand(seed));
        float2 samplePixel = float2(pixel) + jitter;

        Ray ray;
        ray.origin = rayOriginFromPixel(samplePixel, push);
        ray.direction = rayDirectionFromPixel(samplePixel, push);
        RayCone rayCone = primaryRayConeFromPixel(samplePixel, push);
        color += pathTrace(ray, rayCone, push, seed);

        if (sampleIndex == 0u) {
            guides = evaluateStandardDlssRrGuides(ray, rayCone, push, seed);
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
    writeDlssRrGuides(pixel, guides);
}
