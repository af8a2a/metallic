import Core;
using Metallic;
import Lighting;
using Metallic.Lighting;

#include "../PathTracing/OpenPBRSurface.slang"

#ifndef METALLIC_DEFERRED_FP16
#define METALLIC_DEFERRED_FP16 1
#endif
#if METALLIC_DEFERRED_FP16
typedef half DeferredWeight;
typedef half3 DeferredColor;
#else
// FP32 reference for precision regressions; the same shading algorithm.
typedef float DeferredWeight;
typedef float3 DeferredColor;
#endif

#define gEnvironmentIrradianceSH resolveBuffer<StructuredBuffer<float4>>(getResourceParameters<SceneResourceParameters>().environmentIrradianceSH)


#define gDeferredGridParams resolveBuffer<StructuredBuffer<ClusterLightGridParams>>(getResourceParameters<SceneResourceParameters>().lightGridParams)
#define gDeferredGridLights resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lightGridLights)
#define gDeferredGridCandidates resolveBuffer<StructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().lightGridCandidates)
#define gDeferredGridCells resolveBuffer<StructuredBuffer<ClusterLightGridCell>>(getResourceParameters<SceneResourceParameters>().lightGridCells)
#define gDeferredGridIndices resolveBuffer<StructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().lightGridIndices)


#define gRealtimeShadow resolveUniform(getResourceParameters<SceneResourceParameters>().shadowVisibility)
#define gRealtimeShadowParameters resolveBuffer<StructuredBuffer<ShadowParameters>>(getResourceParameters<SceneResourceParameters>().shadowParameters)
#define gEnvironmentSpecular resolveBuffer<StructuredBuffer<float4>>(getResourceParameters<SceneResourceParameters>().previousEnvironmentIrradianceSH)

float3 realtimeEnvironmentSpecular(float3 direction, float roughness, ScenePathTracePush push)
{
    float3 local = environmentLocalDirectionToWorld(direction, -push.environmentRotationRadians);
    float level = saturate(roughness) * float(kEnvironmentFilterLayers - 1u);
    uint lo = uint(floor(level)), hi = min(lo + 1u, kEnvironmentFilterLayers - 1u);
    // Retain the original HDRI resolution for mirror reflections.
    float3 a = lo == 0u ? environmentRadiance(direction, push, false)
        : environmentFilterLayer(gEnvironmentSpecular, local, lo) * push.environmentIntensity;
    float3 b = environmentFilterLayer(gEnvironmentSpecular, local, hi) * push.environmentIntensity;
    return lerp(a, b, frac(level));
}

float3 environmentDiffuseIrradiance(float3 normal, ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) { return float3(0.0); }
    float3 localNormal = environmentLocalDirectionToWorld(normal, -push.environmentRotationRadians);
    SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(gEnvironmentIrradianceSH);
    float3 irradiance = environment.evaluate(localNormal);
    return max(irradiance, float3(0.0)) * push.environmentIntensity;
}

float3 shadeVisibilitySurface(Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, int fixedMetalness = -1, bool opaquePrimary = false)
{
    RayCone cone = propagateRayCone(primaryRayConeFromPixel(samplePixel, push), hit.t);
    PathTraceMaterial material = gMaterials[hit.materialIndex];
    float3 authoredNormal = hit.normal;
    float4 normalSample = loadMaterialTextureForMaterial(material.normalTexture, hit.texcoord,
        float4(0.5, 0.5, 1.0, 1.0), push, cone, hit, ray.direction, materialNeedsDeterministicTextureMip0(material), seed);
    hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent,
        hit.bitangent, normalSample, material.textureParams.x);
    // TBN and geometric normal remain authored; face-forward only the evaluated normal.
    hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, ray.direction);
    OpenPBRMaterialEval surface = evaluateOpenPBRMaterial(hit, material, push, cone, ray.direction, seed);
    // The tile classifier only specializes values guaranteed by source materials.
    if (fixedMetalness >= 0) { surface.inputs.base_metalness = float(fixedMetalness); }
    if (opaquePrimary) { surface.inputs.transmission_weight = 0.0; }
    if (push.debugView != kOpenPBRDebugViewFinal) {
        return openPbrPrimaryDebugColor(push, hit, authoredNormal, normalSample, surface, ray.direction);
    }
    if (material.emissive.w != 0.0) { return surface.inputs.base_color; }
    float3 viewDirection = -ray.direction;
    OpenPBR_PreparedBsdf prepared = prepareOpenPBRMaterial(surface.inputs, float3(1.0),
        OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, viewDirection);
    float3 radiance = max(prepared.emission, float3(0.0));
    if (!openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting)) {
        ClusterLightGridParams grid = gDeferredGridParams[0];
        ClusterLightGridLookup lookup = clusterLightGridLookup(grid, gDeferredGridCells,
            uint2(samplePixel), cameraLinearDepth(hit.position, push));
        uint count = lookup.count + clusterLightGridGlobalLightCount(grid);
        for (uint i = 0u; i < count; ++i) {
            uint source = i < lookup.count
                ? clusterLightGridLocalLightIndex(lookup, gDeferredGridIndices, gDeferredGridCandidates, i)
                : clusterLightGridGlobalLightIndex(grid, gDeferredGridCandidates, i - lookup.count);
            PunctualLightSample light = samplePunctualLightData(gDeferredGridLights[source], hit.position);
            float3 bsdf = max(openpbr_get_sum_of_diffuse_specular(
                evalOpenPBRMaterialProjected(prepared, light.direction)), float3(0.0));
            if (luminanceMax(bsdf * light.illuminance) <= 0.0) { continue; }
            float visibility = 1.0;
            ShadowParameters shadow = gRealtimeShadowParameters[0];
            if (!openPbrDebugFlag(push, kOpenPBRDebugDisableShadows) &&
                shadow.control.x != 0u && source == shadow.control.z) {
                float encoded = gRealtimeShadow.Load(int3(uint2(samplePixel), 0));
                visibility = encoded * encoded;
            }
            radiance += bsdf * light.illuminance * visibility;
        }
    }
    if (push.environmentMode != kEnvironmentModeDisabled && push.environmentIntensity > 0.0) {
        float3 normal = hit.normal;
        float roughness = saturate(surface.inputs.specular_roughness);
        // Only bounded energy weights use FP16; radiance and geometric math stay FP32.
        DeferredWeight metalness = DeferredWeight(saturate(surface.inputs.base_metalness));
        float ior = max(surface.inputs.specular_ior, 1.0);
        float dielectricF0 = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
        float3 f0 = lerp(dielectricF0 * surface.inputs.specular_color * surface.inputs.specular_weight,
            surface.inputs.base_color, metalness);
        DeferredColor specularWeight = DeferredColor(saturate(approxSpecularIntegralGGX(f0, max(roughness * roughness, 0.0001),
            saturate(dot(normal, viewDirection)))));
        DeferredWeight transmissionWeight = DeferredWeight(saturate(surface.inputs.transmission_weight));
        DeferredWeight diffuseWeight = DeferredWeight(saturate(surface.inputs.base_weight)) * (DeferredWeight(1.0) - metalness) *
            (DeferredWeight(1.0) - transmissionWeight);
        DeferredColor diffuseColor = DeferredColor(surface.inputs.base_color) * diffuseWeight * (DeferredColor(1.0) - specularWeight);
        DeferredColor transmissionColor = DeferredColor(surface.inputs.transmission_color) * transmissionWeight *
            (DeferredWeight(1.0) - metalness) * (DeferredColor(1.0) - specularWeight);
        float3 diffuse = environmentDiffuseIrradiance(normal, push) / kPi *
            float3(diffuseColor);
        float3 specular = realtimeEnvironmentSpecular(reflect(-viewDirection, normal), roughness, push) * float3(specularWeight);
        float3 transmitted = float3(0.0);
        float3 refraction = refract(-viewDirection, normal, hit.frontFace ? 1.0 / ior : ior);
        if (dot(refraction, refraction) > 0.0 && surface.inputs.transmission_weight > 0.0) {
            transmitted = realtimeEnvironmentSpecular(refraction, roughness, push) *
                float3(transmissionColor);
        }
        radiance += (diffuse + specular + transmitted) * surface.occlusion;
    }
    return radiance;
}
