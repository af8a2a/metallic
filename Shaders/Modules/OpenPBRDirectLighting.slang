#language slang 2026
module OpenPBRDirectLighting;
import Core;
import Lighting;
import Material;
import MaterialProgram;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import ScenePathTypes;
import ScenePathMath;
import SceneSurface;
import Atmosphere;
import OpenPBR;
import OpenPBRClosure;
import OpenPBRSurface;
import SceneSurfaceMaterial;
import SurfaceLighting;
import ScenePathTracing;
import SHARC;
import SlabClosure;
import OpenPBRPathTracing;
namespace Metallic.SceneDirectLighting
{
using Metallic;
using Metallic.Lighting;

using Metallic.Material;
using Metallic.SceneSurface;
using Metallic.ScenePath;
#if METALLIC_CUSTOM_MATERIALS
#endif
using Metallic.OpenPBR;
using Metallic.SceneOpenPBR;
using Metallic.ScenePathTracing;
using Metallic.SHARC;
using Metallic.OpenPBRPathTracing;

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

public float3 realtimeEnvironmentSpecular(float3 direction, float roughness, ScenePathTracePush push)
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

public float3 environmentDiffuseIrradiance(float3 normal, ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) { return float3(0.0); }
    float3 localNormal = environmentLocalDirectionToWorld(normal, -push.environmentRotationRadians);
    SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(gEnvironmentIrradianceSH);
    float3 irradiance = environment.evaluate(localNormal);
    return max(irradiance, float3(0.0)) * push.environmentIntensity;
}

public float deferredRadicalInverse(uint index, uint base)
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

public float3 openPbrDirectVisibility(float3 origin, float3 direction, float distance, ScenePathTracePush push)
{
    if (openPbrDebugFlag(push, kOpenPBRDebugDisableShadows)) { return float3(1.0); }
    if (openPbrDebugFlag(push, kOpenPBRDebugUseOpaqueShadows)) {
        return traceShadow(origin, direction, distance, push) ? float3(0.0) : float3(1.0);
    }
    return openPbrShadowTransmittance(origin, direction, distance, makeOpenPBRInteriorStack(), push);
}

public struct RealtimeSurfaceLights : ISurfaceLightProvider
{
    public ScenePathTracePush push;
    public HitInfo hit;
    public float2 pixel;
#if METALLIC_DEFERRED_LIGHT_GRID
    public ClusterLightGridParams grid;
    public ClusterLightGridLookup lookup;
#endif
    public uint count()
    {
        if (openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting)) { return 0u; }
#if METALLIC_DEFERRED_LIGHT_GRID
        return kCelestialLightCount + lookup.count + clusterLightGridGlobalLightCount(grid);
#else
        return kCelestialLightCount + punctualLightCount(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights));
#endif
    }
    public SurfaceLightSample sample(uint index)
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
    public float3 visibility(SurfaceLightSample light)
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

public float3 shadeSceneDirectSurface<TProgram : ISceneSurfaceMaterialProgram>(inout Material<TProgram> program, Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, bool sampledEnvironment)
{
    let evaluated = program.evaluate(sceneSurfaceMaterialContextForRay(hit,
        propagateRayCone(primaryRayConeFromPixel(samplePixel, push), hit.t), ray.direction, false),
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
        if (summary.numericalEnvironment) {
            return applyPrimaryAerialPerspective(radiance + sceneSurfaceEnvironment(prepared, push) * summary.occlusion, hit.position, ray.origin);
        }
#endif
        float3 normal = hit.normal;
        float roughness = saturate(summary.roughness);
        float ior = summary.ior;
        float3 specularWeight = saturate(approxSpecularIntegralGGX(summary.specularReflectance,
            max(roughness * roughness, 0.0001), saturate(dot(normal, viewDirection))));
        float3 diffuse = environmentDiffuseIrradiance(normal, push) / kPi *
            summary.diffuseColor * (1.0 - specularWeight);
        float3 specular = realtimeEnvironmentSpecular(reflect(-viewDirection, normal), roughness, push) * specularWeight;
        float3 transmitted = float3(0.0);
        float3 refraction = refract(-viewDirection, normal, hit.frontFace ? 1.0 / ior : ior);
        if (dot(refraction, refraction) > 0.0 && summary.transmissionWeight > 0.0) {
            transmitted = realtimeEnvironmentSpecular(refraction, roughness, push) *
                summary.transmissionColor * summary.transmissionWeight * (1.0 - specularWeight);
        }
        radiance += (diffuse + specular + transmitted) * summary.occlusion;
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
        radiance += environment * summary.occlusion / float(count);
    } else if (!sampledEnvironment) {
#if METALLIC_CUSTOM_MATERIALS
        if (summary.numericalEnvironment) { return applyPrimaryAerialPerspective(radiance + sceneSurfaceEnvironment(prepared, push), hit.position, ray.origin); }
#endif
        float3 normal = dot(hit.normal, viewDirection) >= 0.0 ? hit.normal : -hit.normal;
        radiance += summary.diffuseColor * summary.occlusion *
            environmentDiffuseIrradiance(normal, push) / kPi;
    }
#endif
    return applyPrimaryAerialPerspective(radiance, hit.position, ray.origin);
}


public float3 shadeOpenPBRDirectSurface(Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, bool sampledEnvironment,
    int fixedMetalness = -1, bool opaquePrimary = false)
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
        return shadeSceneDirectSurface(program, ray, hit, samplePixel, push, seed, sampledEnvironment);
    }
    if (family == 2u) {
        var program = makeSceneSlabMaterialProgram<SceneDualSlabModel>(hit, push, cone, ray.direction, seed, true, false, specializedProgram);
        return shadeSceneDirectSurface(program, ray, hit, samplePixel, push, seed, sampledEnvironment);
    }
#endif
    var program = makeOpenPBRMaterialProgram(hit, push, cone, ray.direction, seed, true, false);
    program.program.fixedMetalness = fixedMetalness;
    program.program.opaquePrimary = opaquePrimary;
    return shadeSceneDirectSurface(program, ray, hit, samplePixel, push, seed, sampledEnvironment);
}
}
