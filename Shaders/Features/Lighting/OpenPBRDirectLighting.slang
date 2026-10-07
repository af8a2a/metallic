import Core;
using Metallic;
import Lighting;
using Metallic.Lighting;

#include "../PathTracing/OpenPBRRayQueryPathTrace.slang"

#define gEnvironmentIrradianceSH resolveBuffer<StructuredBuffer<float4>>(getResourceParameters<SceneResourceParameters>().environmentIrradianceSH)

#if METALLIC_DEFERRED_LIGHT_GRID

#define gDeferredGridParams resolveBuffer<StructuredBuffer<ClusterLightGridParams>>(getResourceParameters<SceneResourceParameters>().lightGridParams)
#define gDeferredGridLights resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lightGridLights)
#define gDeferredGridCandidates resolveBuffer<StructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().lightGridCandidates)
#define gDeferredGridCells resolveBuffer<StructuredBuffer<ClusterLightGridCell>>(getResourceParameters<SceneResourceParameters>().lightGridCells)
#define gDeferredGridIndices resolveBuffer<StructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().lightGridIndices)
#endif

#if METALLIC_REALTIME_DEFERRED

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
#endif

float3 environmentDiffuseIrradiance(float3 normal, ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) { return float3(0.0); }
    float3 localNormal = environmentLocalDirectionToWorld(normal, -push.environmentRotationRadians);
    SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(gEnvironmentIrradianceSH);
    float3 irradiance = environment.evaluate(localNormal);
    return max(irradiance, float3(0.0)) * push.environmentIntensity;
}

float deferredRadicalInverse(uint index, uint base)
{
    float value = 0.0;
    float scale = 1.0 / float(base);
    while (index != 0u) {
        value += float(index % base) * scale;
        index /= base;
        scale /= float(base);
    }
    return value;
}

float3 openPbrDirectVisibility(float3 origin, float3 direction, float distance, ScenePathTracePush push)
{
    if (openPbrDebugFlag(push, kOpenPBRDebugDisableShadows)) { return float3(1.0); }
    if (openPbrDebugFlag(push, kOpenPBRDebugUseOpaqueShadows)) {
        return traceShadow(origin, direction, distance, push) ? float3(0.0) : float3(1.0);
    }
    return openPbrShadowTransmittance(origin, direction, distance, makeOpenPBRInteriorStack(), push);
}

struct RealtimeSurfaceLights : ISurfaceLightProvider
{
    ScenePathTracePush push;
    HitInfo hit;
    float2 pixel;
#if METALLIC_DEFERRED_LIGHT_GRID
    ClusterLightGridParams grid;
    ClusterLightGridLookup lookup;
#endif
    uint count()
    {
        if (openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting)) { return 0u; }
#if METALLIC_DEFERRED_LIGHT_GRID
        return kCelestialLightCount + lookup.count + clusterLightGridGlobalLightCount(grid);
#else
        return kCelestialLightCount + punctualLightCount(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights));
#endif
    }
    SurfaceLightSample sample(uint index)
    {
        SurfaceLightSample result;
        PunctualLightSample light;
        if (index < kCelestialLightCount) {
            result.source = celestialSourceIdentity(index);
            light = sampleCelestialLightAtPosition(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getResourceParameters<SceneResourceParameters>().environment.celestialLights), index, hit.position);
        } else {
            index -= kCelestialLightCount;
#if METALLIC_DEFERRED_LIGHT_GRID
            result.source = index < lookup.count
                ? clusterLightGridLocalLightIndex(lookup, gDeferredGridIndices, gDeferredGridCandidates, index)
                : clusterLightGridGlobalLightIndex(grid, gDeferredGridCandidates, index - lookup.count);
            light = samplePunctualLightData(gDeferredGridLights[result.source], hit.position);
#else
            result.source = index;
            light = samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), index, hit.position);
#endif
        }
        result.direction = light.direction;
        result.illuminance = light.illuminance;
        result.distance = light.distance;
        result.inverseSourcePdf = any(light.illuminance > 0.0) ? 1.0 : 0.0;
        return result;
    }
    float3 visibility(SurfaceLightSample light)
    {
        if ((light.source & kCelestialSourceBit) != 0u &&
            !celestialLightCastsShadow(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getResourceParameters<SceneResourceParameters>().environment.celestialLights),
                light.source & ~kCelestialSourceBit)) { return float3(1.0); }
        float3 origin = offsetRay(hit.position, openPbrRayOffsetNormal(light.direction, hit.geometryNormal));
        float distance = max(light.distance - length(origin - hit.position), 0.0);
#if METALLIC_REALTIME_DEFERRED
        ShadowParameters shadow = gRealtimeShadowParameters[0];
        // SIGMA attenuation is applied only to its selected source.
        float encoded = gRealtimeShadow.Load(int3(uint2(pixel), 0));
        if (!openPbrDebugFlag(push, kOpenPBRDebugDisableShadows) &&
            shadow.control.x != 0u && light.source == shadow.control.z) {
            return float3(encoded * encoded);
        }
#endif
        return openPbrDirectVisibility(origin, light.direction, distance, push);
    }
};

float3 shadeOpenPBRDirectSurface(Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, bool sampledEnvironment,
    int fixedMetalness = -1, bool opaquePrimary = false)
{
    RayCone cone = propagateRayCone(primaryRayConeFromPixel(samplePixel, push), hit.t);
    float3 authoredNormal = hit.normal;
    SceneOpenPBRMaterialProgram program = makeOpenPBRMaterialProgram(hit, push, cone, ray.direction, seed, true, false);
#if defined(MATERIAL_PROGRAM_ID)
    program.specializedProgramId = MATERIAL_PROGRAM_ID;
#endif
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
    if (material.emissive.w != 0.0) { return applyPrimaryAerialPerspective(surface.inputs.base_color, hit.position, ray.origin); }
    float3 viewDirection = -ray.direction;
    RealtimeSurfaceLights lights;
    lights.push = push;
    lights.hit = hit;
    lights.pixel = samplePixel;
#if METALLIC_DEFERRED_LIGHT_GRID
    lights.grid = gDeferredGridParams[0];
    lights.lookup = clusterLightGridLookup(lights.grid, gDeferredGridCells,
        uint2(samplePixel), cameraLinearDepth(hit.position, push));
#endif
    SurfaceSamplingContext sampling;
    sampling.throughput = float3(1);
    sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
    sampling.exteriorIor = OpenPBRVacuumIor;
    let shaded = shadeEvaluatedSurface(evaluated, viewDirection, TransportMode.Radiance, sampling, lights);
    let prepared = shaded.prepared;
    float3 radiance = shaded.radiance;
#if METALLIC_REALTIME_DEFERRED
    if (push.environmentMode != kEnvironmentModeDisabled && push.environmentIntensity > 0.0) {
#if METALLIC_CUSTOM_MATERIALS
        if (surface.family != 0u || surface.inputs.coat_weight > 0.0 || surface.inputs.fuzz_weight > 0.0 ||
            surface.inputs.specular_roughness_anisotropy > 0.0) {
            return applyPrimaryAerialPerspective(radiance + sceneSurfaceEnvironment(prepared, push) * surface.occlusion, hit.position, ray.origin);
        }
#endif
        float3 normal = hit.normal;
        float roughness = saturate(surface.inputs.specular_roughness);
        float metalness = saturate(surface.inputs.base_metalness);
        float ior = max(surface.inputs.specular_ior, 1.0);
        float dielectricF0 = ((ior - 1.0) / (ior + 1.0)) * ((ior - 1.0) / (ior + 1.0));
        float3 f0 = lerp(dielectricF0 * surface.inputs.specular_color * surface.inputs.specular_weight,
            surface.inputs.base_color, metalness);
        float3 specularWeight = saturate(approxSpecularIntegralGGX(f0, max(roughness * roughness, 0.0001),
            saturate(dot(normal, viewDirection))));
        float diffuseWeight = surface.inputs.base_weight * (1.0 - metalness) *
            (1.0 - surface.inputs.transmission_weight);
        float3 diffuse = environmentDiffuseIrradiance(normal, push) / kPi *
            surface.inputs.base_color * diffuseWeight * (1.0 - specularWeight);
        float3 specular = realtimeEnvironmentSpecular(reflect(-viewDirection, normal), roughness, push) * specularWeight;
        float3 transmitted = float3(0.0);
        float3 refraction = refract(-viewDirection, normal, hit.frontFace ? 1.0 / ior : ior);
        if (dot(refraction, refraction) > 0.0 && surface.inputs.transmission_weight > 0.0) {
            transmitted = realtimeEnvironmentSpecular(refraction, roughness, push) *
                surface.inputs.transmission_color * surface.inputs.transmission_weight *
                (1.0 - metalness) * (1.0 - specularWeight);
        }
        radiance += (diffuse + specular + transmitted) * surface.occlusion;
    }
#else
    if (sampledEnvironment && push.environmentMode != kEnvironmentModeDisabled && push.environmentIntensity > 0.0) {
        float3 environment = float3(0.0);
        float3 rotation = float3(rand(seed), rand(seed), rand(seed));
        uint count = max(push.samples, 1u);
        for (uint i = 0u; i < count; ++i) {
            float3 random = frac(float3((float(i) + 0.5) / float(count),
                deferredRadicalInverse(i, 2u), deferredRadicalInverse(i, 3u)) + rotation);
            BSDFWeightSample scatter = sampleSurfaceWeight(prepared, random);
            if (scatter.pdf <= 0.0 || scatter.flags == 0u) { continue; }
            float3 direction = scatter.wiWS;
            float3 origin = offsetRay(hit.position, openPbrRayOffsetNormal(direction, hit.geometryNormal));
            float3 visibility = openPbrDirectVisibility(origin, direction, max(push.clipOrtho.y, 1.0), push);
            environment += max(scatter.weight, float3(0.0)) *
                environmentRadiance(direction, push, false) * visibility;
        }
        radiance += environment * surface.occlusion / float(count);
    } else if (!sampledEnvironment) {
#if METALLIC_CUSTOM_MATERIALS
        if (surface.family != 0u) { return applyPrimaryAerialPerspective(radiance + sceneSurfaceEnvironment(prepared, push), hit.position, ray.origin); }
#endif
        float3 normal = dot(hit.normal, viewDirection) >= 0.0 ? hit.normal : -hit.normal;
        float diffuseWeight = surface.inputs.base_weight * (1.0 - surface.inputs.base_metalness) *
            (1.0 - surface.inputs.transmission_weight);
        radiance += surface.inputs.base_color * diffuseWeight * surface.occlusion *
            environmentDiffuseIrradiance(normal, push) / kPi;
    }
#endif
    return applyPrimaryAerialPerspective(radiance, hit.position, ray.origin);
}
