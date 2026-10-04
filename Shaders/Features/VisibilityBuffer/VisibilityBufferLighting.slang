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
typedef Metallic.WorkingColor::BoundedEnergyColor DeferredColor;
typedef Metallic.WorkingColor::BoundedEnergyInput DeferredInput;
#else
// FP32 reference for precision regressions; the same shading algorithm.
typedef float DeferredWeight;
typedef float3 DeferredColor;
typedef float DeferredInput;
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

struct VisibilitySurfaceLights : ISurfaceLightProvider
{
    ScenePathTracePush push;
    float3 position;
    float2 pixel;
    ClusterLightGridParams grid;
    ClusterLightGridLookup lookup;
    uint count()
    {
        return openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting)
            ? 0u : lookup.count + clusterLightGridGlobalLightCount(grid);
    }
    SurfaceLightSample sample(uint index)
    {
        SurfaceLightSample result;
        result.source = index < lookup.count
            ? clusterLightGridLocalLightIndex(lookup, gDeferredGridIndices, gDeferredGridCandidates, index)
            : clusterLightGridGlobalLightIndex(grid, gDeferredGridCandidates, index - lookup.count);
        PunctualLightSample light = samplePunctualLightData(gDeferredGridLights[result.source], position);
        result.direction = light.direction;
        result.illuminance = light.illuminance;
        result.distance = light.distance;
        result.inverseSourcePdf = 1.0;
        return result;
    }
    float3 visibility(SurfaceLightSample light)
    {
        ShadowParameters shadow = gRealtimeShadowParameters[0];
        if (!openPbrDebugFlag(push, kOpenPBRDebugDisableShadows) &&
            shadow.control.x != 0u && light.source == shadow.control.z) {
            float encoded = gRealtimeShadow.Load(int3(uint2(pixel), 0));
            return float3(encoded * encoded);
        }
        return float3(1);
    }
};

float3 shadeVisibilitySurface(Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, int fixedMetalness = -1, bool opaquePrimary = false)
{
    RayCone cone = propagateRayCone(primaryRayConeFromPixel(samplePixel, push), hit.t);
    float3 authoredNormal = hit.normal;
    SceneOpenPBRMaterialProgram program = makeOpenPBRMaterialProgram(hit, push, cone, ray.direction, seed, true, false);
    program.fixedMetalness = fixedMetalness;
    program.opaquePrimary = opaquePrimary;
    let evaluated = program.evaluate(sceneSurfaceMaterialContextForRay(hit, cone, ray.direction),
        legacyMaterialInstanceRef(hit.materialIndex));
    let surface = evaluated.closure;
    hit = program.hit;
    seed = program.seed;
    float4 normalSample = program.normalSample;
    PathTraceMaterial material = program.material;
    if (push.debugView != kOpenPBRDebugViewFinal) {
        return openPbrPrimaryDebugColor(push, hit, authoredNormal, normalSample, surface, ray.direction);
    }
    if (material.emissive.w != 0.0) { return surface.inputs.base_color; }
    float3 viewDirection = -ray.direction;
    VisibilitySurfaceLights lights;
    lights.push = push;
    lights.position = hit.position;
    lights.pixel = samplePixel;
    lights.grid = gDeferredGridParams[0];
    lights.lookup = clusterLightGridLookup(lights.grid, gDeferredGridCells,
        uint2(samplePixel), cameraLinearDepth(hit.position, push));
    SurfaceSamplingContext sampling;
    sampling.throughput = float3(1);
    sampling.wavelengths = OpenPBR_BaseRgbWavelengths_nm;
    sampling.exteriorIor = OpenPBR_VacuumIor;
    let shaded = shadeEvaluatedSurface(evaluated, viewDirection, TransportMode.Radiance, sampling, lights);
    float3 radiance = shaded.radiance;
    if (push.environmentMode != kEnvironmentModeDisabled && push.environmentIntensity > 0.0) {
        float3 normal = hit.normal;
        float roughness = saturate(surface.inputs.specular_roughness);
        // Bounded scalar energy uses FP16. AP1 color-vector weights retain FP32
        // through WorkingColor to bound display-conversion error.
        DeferredInput metalness = DeferredInput(saturate(surface.inputs.base_metalness));
        float ior = max(surface.inputs.specular_ior, 1.0);
        float dielectricF0 = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
        float3 f0 = lerp(dielectricF0 * surface.inputs.specular_color * surface.inputs.specular_weight,
            surface.inputs.base_color, metalness);
        DeferredColor specularWeight = DeferredColor(saturate(approxSpecularIntegralGGX(f0, max(roughness * roughness, 0.0001),
            saturate(dot(normal, viewDirection)))));
        DeferredInput transmissionWeight = DeferredInput(saturate(surface.inputs.transmission_weight));
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
