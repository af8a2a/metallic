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
            ? 0u : kCelestialLightCount + lookup.count + clusterLightGridGlobalLightCount(grid);
    }
    SurfaceLightSample sample(uint index)
    {
        SurfaceLightSample result;
        PunctualLightSample light;
        if (index < kCelestialLightCount) {
            result.source = celestialSourceIdentity(index);
            light = sampleCelestialLightAtPosition(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getResourceParameters<SceneResourceParameters>().environment.celestialLights), index, position);
        } else {
            index -= kCelestialLightCount;
            result.source = index < lookup.count
                ? clusterLightGridLocalLightIndex(lookup, gDeferredGridIndices, gDeferredGridCandidates, index)
                : clusterLightGridGlobalLightIndex(grid, gDeferredGridCandidates, index - lookup.count);
            light = samplePunctualLightData(gDeferredGridLights[result.source], position);
        }
        result.direction = light.direction;
        result.illuminance = light.illuminance;
        result.distance = light.distance;
        result.inverseSourcePdf = any(light.illuminance > 0.0) ? 1.0 : 0.0;
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

float3 shadeSceneVisibilitySurface<TProgram : ISceneSurfaceMaterialProgram>(inout Material<TProgram> program, Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed)
{
    let evaluated = program.evaluate(sceneSurfaceMaterialContextForRay(hit,
        propagateRayCone(primaryRayConeFromPixel(samplePixel, push), hit.t), ray.direction),
        sceneMaterialInstanceRef(hit.materialIndex));
    float3 authoredNormal = hit.normal;
    hit = program.program.evaluatedHit();
    seed = program.program.evaluatedSeed();
    float4 normalSample = program.program.sampledNormal();
    SceneSurfaceSummary summary = program.program.surfaceSummary();
    if (push.debugView != kOpenPBRDebugViewFinal) {
        return openPbrPrimaryDebugColor(push, hit, authoredNormal, normalSample, summary, ray.direction);
    }
    if (summary.unlit) { return applyPrimaryAerialPerspective(summary.baseColor, hit.position, ray.origin); }
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
    sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
    sampling.exteriorIor = OpenPBRVacuumIor;
    let shaded = shadeEvaluatedSurface(evaluated, viewDirection, TransportMode.Radiance, sampling, lights);
    float3 radiance = shaded.radiance;
#if METALLIC_CUSTOM_MATERIALS
    if (summary.numericalEnvironment) {
        return applyPrimaryAerialPerspective(radiance + sceneSurfaceEnvironment(shaded.prepared, push) * summary.occlusion, hit.position, ray.origin);
    }
#endif
    if (push.environmentMode != kEnvironmentModeDisabled && push.environmentIntensity > 0.0) {
        float3 normal = hit.normal;
        float roughness = saturate(summary.roughness);
        float ior = summary.ior;
        // Keep the established FP16 approximation policy: round bounded
        // coefficients before deriving color weights, while HDR stays FP32.
        DeferredInput conductorFraction = DeferredInput(summary.conductorFraction);
        DeferredInput surfaceTransmission = DeferredInput(saturate(summary.surfaceTransmission));
        float3 f0 = lerp(summary.dielectricReflectance, summary.baseColor, conductorFraction);
        DeferredColor specularWeight = DeferredColor(saturate(approxSpecularIntegralGGX(f0,
            max(roughness * roughness, 0.0001), saturate(dot(normal, viewDirection)))));
        DeferredWeight diffuseWeight = DeferredWeight(saturate(summary.diffuseWeight)) *
            (DeferredWeight(1.0) - conductorFraction) * (DeferredWeight(1.0) - surfaceTransmission);
        DeferredColor diffuseColor = DeferredColor(summary.baseColor) * diffuseWeight *
            (DeferredColor(1.0) - specularWeight);
        DeferredColor transmissionColor = DeferredColor(summary.transmissionColor) * surfaceTransmission *
            (DeferredWeight(1.0) - conductorFraction) * (DeferredColor(1.0) - specularWeight);
        float3 diffuse = environmentDiffuseIrradiance(normal, push) / kPi *
            float3(diffuseColor);
        float3 specular = realtimeEnvironmentSpecular(reflect(-viewDirection, normal), roughness, push) * float3(specularWeight);
        float3 transmitted = float3(0.0);
        float3 refraction = refract(-viewDirection, normal, hit.frontFace ? 1.0 / ior : ior);
        if (dot(refraction, refraction) > 0.0 && summary.transmissionWeight > 0.0) {
            transmitted = realtimeEnvironmentSpecular(refraction, roughness, push) *
                float3(transmissionColor);
        }
        radiance += (diffuse + specular + transmitted) * summary.occlusion;
    }
    return applyPrimaryAerialPerspective(radiance, hit.position, ray.origin);
}


float3 shadeVisibilitySurface(Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, int fixedMetalness = -1, bool opaquePrimary = false)
{
    RayCone cone = propagateRayCone(primaryRayConeFromPixel(samplePixel, push), hit.t);
#if METALLIC_CUSTOM_MATERIALS
#if defined(MATERIAL_PROGRAM_ID)
    uint specializedProgram = MATERIAL_PROGRAM_ID;
#else
    uint specializedProgram = 0u;
#endif
    uint family = sceneSurfaceMaterialFamily(hit.materialIndex, specializedProgram);
    if (family == 1u) {
        var program = makeSceneSlabMaterialProgram<SceneSingleSlabModel>(hit, push, cone, ray.direction, seed, true, false, specializedProgram);
        return shadeSceneVisibilitySurface(program, ray, hit, samplePixel, push, seed);
    }
    if (family == 2u) {
        var program = makeSceneSlabMaterialProgram<SceneDualSlabModel>(hit, push, cone, ray.direction, seed, true, false, specializedProgram);
        return shadeSceneVisibilitySurface(program, ray, hit, samplePixel, push, seed);
    }
#endif
    var program = makeOpenPBRMaterialProgram(hit, push, cone, ray.direction, seed, true, false);
    program.program.fixedMetalness = fixedMetalness;
    program.program.opaquePrimary = opaquePrimary;
    return shadeSceneVisibilitySurface(program, ray, hit, samplePixel, push, seed);
}
