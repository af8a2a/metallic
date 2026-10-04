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

float3 shadeOpenPBRDirectSurface(Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, bool sampledEnvironment,
    int fixedMetalness = -1, bool opaquePrimary = false)
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
    let prepared = surface.prepare(viewDirection, TransportMode.Radiance);
    float3 radiance = max(prepared.emission(), float3(0.0));
    if (!openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting)) {
#if METALLIC_DEFERRED_LIGHT_GRID
        ClusterLightGridParams grid = gDeferredGridParams[0];
        ClusterLightGridLookup lookup = clusterLightGridLookup(grid, gDeferredGridCells,
            uint2(samplePixel), cameraLinearDepth(hit.position, push));
        uint count = lookup.count + clusterLightGridGlobalLightCount(grid);
#else
        uint count = punctualLightCount(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights));
#endif
        for (uint i = 0u; i < count; ++i) {
#if METALLIC_DEFERRED_LIGHT_GRID
            uint source = i < lookup.count
                ? clusterLightGridLocalLightIndex(lookup, gDeferredGridIndices, gDeferredGridCandidates, i)
                : clusterLightGridGlobalLightIndex(grid, gDeferredGridCandidates, i - lookup.count);
            PunctualLightSample light = samplePunctualLightData(gDeferredGridLights[source], hit.position);
#else
            PunctualLightSample light = samplePunctualLight(resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameters<SceneResourceParameters>().lights), i, hit.position);
#endif
            float3 bsdf = max(evaluateSurfaceProjected(prepared, light.direction), float3(0.0));
            if (luminanceMax(bsdf * light.illuminance) <= 0.0) { continue; }
            float3 origin = offsetRay(hit.position, openPbrRayOffsetNormal(light.direction, hit.geometryNormal));
            float distance = max(light.distance - length(origin - hit.position), 0.0);
            float3 visibility;
#if METALLIC_REALTIME_DEFERRED
            ShadowParameters shadow = gRealtimeShadowParameters[0];
            // SIGMA encodes sqrt(visibility); attenuate only the selected light.
            float encoded = gRealtimeShadow.Load(int3(uint2(samplePixel), 0));
            if (!openPbrDebugFlag(push, kOpenPBRDebugDisableShadows) &&
                shadow.control.x != 0u && source == shadow.control.z) {
                visibility = float3(encoded * encoded);
                // The selected light already uses the sampled TLAS ray before SIGMA.
                // A center-ray hard-shadow clamp here would destroy its soft edge.
            } else {
                visibility = openPbrDirectVisibility(origin, light.direction, distance, push);
            }
#else
            visibility = openPbrDirectVisibility(origin, light.direction, distance, push);
#endif
            radiance += bsdf * light.illuminance * visibility;
        }
    }
#if METALLIC_REALTIME_DEFERRED
    if (push.environmentMode != kEnvironmentModeDisabled && push.environmentIntensity > 0.0) {
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
        float3 normal = dot(hit.normal, viewDirection) >= 0.0 ? hit.normal : -hit.normal;
        float diffuseWeight = surface.inputs.base_weight * (1.0 - surface.inputs.base_metalness) *
            (1.0 - surface.inputs.transmission_weight);
        radiance += surface.inputs.base_color * diffuseWeight * surface.occlusion *
            environmentDiffuseIrradiance(normal, push) / kPi;
    }
#endif
    return radiance;
}
