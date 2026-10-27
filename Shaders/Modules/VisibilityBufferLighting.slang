#language slang 2026
module VisibilityBufferLighting;

import WorkingColor;
import Core;
using Metallic;
import Lighting;
using Metallic.Lighting;
import OpenPBRSurface;
import SceneSurface;
import ScenePathTypes;
import ScenePathMath;
import SceneSurfaceMaterial;
import Material;
import MaterialProgram;
import OpenPBR;
import OpenPBRClosure;
import OpenPBRMaterialParameters;
import SurfaceLighting;
import VirtualShadowMapCommon;
#include "../../Source/Runtime/Render/Shadow/VirtualShadowMapLights.h"
using Metallic.SceneOpenPBR;
using Metallic.SceneSurface;
using Metallic.ScenePath;
using Metallic.Material;
using Metallic.OpenPBR;
using Metallic.Lighting;

namespace Metallic.VisibilityLighting
{


#ifndef METALLIC_DEFERRED_RUNTIME_PRECISION
#define METALLIC_DEFERRED_RUNTIME_PRECISION 0
#endif
#if METALLIC_DEFERRED_RUNTIME_PRECISION && METALLIC_WORKING_SPACE_ACESCG
public typedef float DeferredWeight;
public typedef float3 DeferredColor;
public typedef float DeferredInput;
#else
#ifndef METALLIC_DEFERRED_FP16
#define METALLIC_DEFERRED_FP16 1
#endif
#if METALLIC_DEFERRED_FP16
public typedef half DeferredWeight;
public typedef Metallic.WorkingColor::BoundedEnergyColor DeferredColor;
public typedef Metallic.WorkingColor::BoundedEnergyInput DeferredInput;
#else
// FP32 reference for precision regressions; the same shading algorithm.
public typedef float DeferredWeight;
public typedef float3 DeferredColor;
public typedef float DeferredInput;
#endif
#endif

#if METALLIC_DEFERRED_RUNTIME_PRECISION && METALLIC_WORKING_SPACE_ACESCG
[SpecializationConstant]
public const bool kDeferredUseFP16Precision = false;

float deferredPrecisionRoundTrip(float x, bool useFP16Precision)
{
    if (useFP16Precision) {
        return unpackHalf2x16ToFloat(packHalf2x16(float2(x, 0.0f))).x;
    }
    return x;
}
#endif

#define gEnvironmentIrradianceSH resolveBuffer<StructuredBuffer<float4>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneEnvironmentIrradianceSHOffset))


#define gDeferredGridParams resolveBuffer<StructuredBuffer<ClusterLightGridParams>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneLightGridParamsOffset))
#define gDeferredGridLights resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneLightGridLightsOffset))
#define gDeferredGridCandidates resolveBuffer<StructuredBuffer<uint>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneLightGridCandidatesOffset))
#define gDeferredGridCells resolveBuffer<StructuredBuffer<ClusterLightGridCell>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneLightGridCellsOffset))
#define gDeferredGridIndices resolveBuffer<StructuredBuffer<uint>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneLightGridIndicesOffset))


#define gRealtimeShadow resolveUniform(getResourceParameterField<SceneResourceParameters, ResourceHandle<Texture2D<float>>>(kSceneShadowVisibilityOffset))
#define gRealtimeShadowParameters resolveBuffer<StructuredBuffer<ShadowParameters>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneShadowParametersOffset))
#define gRealtimeMoonShadow resolveUniform(getResourceParameterField<SceneResourceParameters, ResourceHandle<Texture2D<float>>>(kSceneMoonShadowVisibilityOffset))
#define gRealtimeMoonShadowParameters resolveBuffer<StructuredBuffer<ShadowParameters>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneMoonShadowParametersOffset))
#define gEnvironmentSpecular resolveBuffer<StructuredBuffer<float4>>(getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kScenePreviousEnvironmentIrradianceSHOffset))

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

public float3 environmentDiffuseIrradiance(float3 normal, ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) { return float3(0.0); }
    float3 localNormal = environmentLocalDirectionToWorld(normal, -push.environmentRotationRadians);
    SH::IrradianceL2RGB environment = SH::loadIrradianceL2RGB(gEnvironmentIrradianceSH);
    float3 irradiance = environment.evaluate(localNormal);
    return max(irradiance, float3(0.0)) * push.environmentIntensity;
}

public struct VisibilitySurfaceLights : ISurfaceLightProvider
{
    public ScenePathTracePush push;
    public float3 position;
    public float2 pixel;
    public ClusterLightGridParams grid;
    public ClusterLightGridLookup lookup;
    public uint count()
    {
        return openPbrDebugFlag(push, kOpenPBRDebugDisableDirectLighting)
            ? 0u : kCelestialLightCount + lookup.count + clusterLightGridGlobalLightCount(grid);
    }
    public SurfaceLightSample sample(uint index)
    {
        SurfaceLightSample result;
        PunctualLightSample light;
        if (index < kCelestialLightCount) {
            result.source = celestialSourceIdentity(index);
            light = sampleCelestialLightAtPosition(resolveBuffer<StructuredBuffer<GPUCelestialLight>>(
                getResourceParameterField<SceneResourceParameters, EnvironmentResourceParameters>(kSceneEnvironmentOffset).celestialLights), index, position);
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
    public float3 visibility(SurfaceLightSample light)
    {
        if (openPbrDebugFlag(push, kOpenPBRDebugDisableShadows)) { return float3(1); }
        let tableHandle = getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneVirtualShadowLightsOffset);
        if (tableHandle.index != 0xffffffffu) {
            let table = resolveUniform(tableHandle);
            VirtualShadowMapLightsHeader header = table.Load<VirtualShadowMapLightsHeader>(0);
            for (uint index = 0u; index < header.count; ++index) {
                VirtualShadowMapLightEntry entry = table.Load<VirtualShadowMapLightEntry>(64u + index * 64u);
                if (entry.source != light.source) { continue; }
                if (entry.status != 1u) { return float3(1); }
                uint face = entry.kind == 2u && (header.reserved[0] & 1u) == 0u ? Metallic.Shadow::virtualShadowMapPointFace(
                    position - float3(entry.positionRange[0], entry.positionRange[1], entry.positionRange[2])) : 0u;
                let shadows = resolveUniform(getResourceParameterField<SceneResourceParameters,
                    ResourceHandle<Texture2DArray<float>>>(kSceneVirtualShadowsOffset));
                return float3(shadows.Load(int4(uint2(pixel), entry.baseLayer + face, 0)));
            }
            // A connected multi-light publication owns this domain, including
            // explicit disabled/budget-rejected entries; legacy pins are ignored.
            return float3(1);
        }
        ShadowParameters shadow = gRealtimeShadowParameters[0];
        if (shadow.control.x != 0u && light.source == shadow.control.z) {
            float encoded = gRealtimeShadow.Load(int3(uint2(pixel), 0));
            return float3(encoded * encoded);
        }
        if (light.source == celestialSourceIdentity(1u)) {
            ShadowParameters moon = gRealtimeMoonShadowParameters[0];
            if (moon.control.x != 0u && moon.control.z == celestialSourceIdentity(1u)) {
                float encoded = gRealtimeMoonShadow.Load(int3(uint2(pixel), 0));
                return float3(encoded * encoded);
            }
        }
        return float3(1);
    }
};

public float3 shadeSceneVisibilitySurface<TProgram : ISceneSurfaceMaterialProgram>(inout Material<TProgram> program, Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed)
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
#if METALLIC_DEFERRED_RUNTIME_PRECISION && METALLIC_WORKING_SPACE_ACESCG
        bool useFP16Precision = kDeferredUseFP16Precision;
        DeferredWeight diffuseWeight = deferredPrecisionRoundTrip(DeferredWeight(saturate(summary.diffuseWeight)), useFP16Precision) *
            (DeferredWeight(1.0) - conductorFraction) * (DeferredWeight(1.0) - surfaceTransmission);
        diffuseWeight = deferredPrecisionRoundTrip(diffuseWeight, useFP16Precision);
#else
        DeferredWeight diffuseWeight = DeferredWeight(saturate(summary.diffuseWeight)) *
            (DeferredWeight(1.0) - conductorFraction) * (DeferredWeight(1.0) - surfaceTransmission);
#endif
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


public float3 shadeVisibilitySurface(Ray ray, HitInfo hit, float2 samplePixel,
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
} // namespace Metallic.VisibilityLighting
