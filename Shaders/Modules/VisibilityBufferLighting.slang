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
import ClusteredSurfaceLighting;
#include "../../Source/Runtime/Render/Shadow/VirtualShadowMapLights.h"
using Metallic.SceneOpenPBR;
using Metallic.SceneSurface;
using Metallic.ScenePath;
using Metallic.Material;
using Metallic.OpenPBR;
using Metallic.Lighting;

#ifndef METALLIC_DEFERRED_LUMEN_RECEIVERS
#define METALLIC_DEFERRED_LUMEN_RECEIVERS 0
#endif
#if METALLIC_DEFERRED_LUMEN_RECEIVERS
import LumenProbeMath;
#endif

namespace Metallic.VisibilityLighting
{

#ifndef MATERIAL_CLASS
#define MATERIAL_CLASS -1
#endif

public static const bool kExportLumenReceivers = METALLIC_DEFERRED_LUMEN_RECEIVERS != 0;

public struct DeferredLumenReceiver
{
    public float4 positionAndDepth;
    public float3 normal;
    public float3 diffuse;
#if METALLIC_DEFERRED_LUMEN_AO_ALBEDO
    public float3 diffuseAlbedo;
#endif
#if METALLIC_DEFERRED_LUMEN_SPECULAR
    public float4 specularAndRoughness;
#endif
};

[ForceInline]
float3 lumenReceiverDiffuse(float3 diffuseColor, float materialAO)
{
#if METALLIC_DEFERRED_LUMEN_RECEIVERS
    return diffuseColor * Metallic.Lumen::lumenDistantIlluminationRescale(diffuseColor) * materialAO;
#else
    return diffuseColor * materialAO;
#endif
}

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

#define gEnvironmentIrradianceSH resolveBuffer<StructuredBuffer<float4>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneEnvironmentIrradianceSHOffset))


#define gRealtimeShadow resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float>>>(kSceneShadowVisibilityOffset))
#define gRealtimeShadowParameters resolveBuffer<StructuredBuffer<ShadowParameters>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneShadowParametersOffset))
#define gRealtimeMoonShadow resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float>>>(kSceneMoonShadowVisibilityOffset))
#define gRealtimeMoonShadowParameters resolveBuffer<StructuredBuffer<ShadowParameters>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneMoonShadowParametersOffset))
#define gEnvironmentSpecular resolveBuffer<StructuredBuffer<float4>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kScenePreviousEnvironmentIrradianceSHOffset))

public float3 realtimeEnvironmentSpecular(float3 direction, float roughness, ScenePathTracePush push)
{
    float3 local = environmentLocalDirectionToWorld(direction, -push.environmentRotationRadians);
    float level = saturate(roughness) * float(kEnvironmentFilterLayers - 1u);
    uint lo = uint(floor(level)), hi = min(lo + 1u, kEnvironmentFilterLayers - 1u);
    let textureHandle = getSceneResourceParameterField<ResourceHandle<Texture2DArray<float4>>>(kSceneEnvironmentSpecularTextureOffset);
    if (textureHandle.index != 0xffffffffu) {
        let source = resolveUniform(textureHandle);
        let samplerState = resolveUniform(getSceneResourceParameterField<SamplerHandle>(kSceneEnvironmentSpecularSamplerOffset));
        float2 uv = environmentFilterUv(local);
        // Mirror reflections retain the original HDRI, including its full HDR range.
        float3 a = lo == 0u ? environmentIndirectRadiance(direction, push)
            : environmentFilterLayer(source, samplerState, uv, lo) * push.environmentIntensity;
        float3 b = environmentFilterLayer(source, samplerState, uv, hi) * push.environmentIntensity;
        return lerp(a, b, frac(level));
    }
    // RGBA32F linear filtering is optional on Vulkan devices.
    // Retain the original HDRI resolution for mirror reflections.
    float3 a = lo == 0u ? environmentIndirectRadiance(direction, push)
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

public struct RasterLightVisibility : IClusteredLightVisibility
{
    public uint debugFlags;
    public float3 position;
    public float2 pixel;

    public float3 evaluate(SurfaceLightSample light)
    {
        if ((debugFlags & kOpenPBRDebugDisableShadows) != 0u) { return float3(1); }
        let tableHandle = getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneVirtualShadowLightsOffset);
        if (tableHandle.index != 0xffffffffu) {
            let table = resolveUniform(tableHandle);
            // A connected multi-light publication owns this domain, including
            // explicit disabled/budget-rejected entries; legacy pins are ignored.
            // VirtualShadowMapLightsHeader::count is at byte offset 4.
            uint headerCount = table.Load(4u);
            uint index = (light.source & kCelestialSourceBit) != 0u
                ? light.source & ~kCelestialSourceBit : light.source + kCelestialLightCount;
            if (index >= headerCount) { return float3(1); }
            uint entryOffset = 64u + index * 64u;
            // VirtualShadowMapLightEntry::status is at byte offset 28.
            if (table.Load(entryOffset + 28u) != 1u) { return float3(1); }
            // VirtualShadowMapLightEntry::kind (offset 4) and baseLayer (offset 8).
            uint2 kindAndBaseLayer = table.Load2(entryOffset + 4u);
            uint face = 0u;
            // VirtualShadowMapLightsHeader::reserved[0] is at byte offset 56; positionRange[0..2] at entryOffset + 32.
            if (kindAndBaseLayer.x == 2u && (table.Load(56u) & 1u) == 0u) {
                float3 lightPosition = asfloat(table.Load3(entryOffset + 32u));
                face = Metallic.Shadow::virtualShadowMapPointFace(position - lightPosition);
            }
            let shadows = resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2DArray<float>>>(kSceneVirtualShadowsOffset));
            return float3(shadows.Load(int4(uint2(pixel), kindAndBaseLayer.y + face, 0)));
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

[ForceInline]
internal float3 shadeSceneVisibilitySurfaceRadiance<TProgram : ISceneSurfaceMaterialProgram>(inout Material<TProgram> program, Ray ray, inout HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, int fixedMetalness, bool opaquePrimary, out bool applyAerial,
    inout DeferredLumenReceiver receiver, bool useRectLights)
{
    applyAerial = true;
    let evaluated = program.evaluate(sceneSurfaceMaterialContextForRay(hit,
        propagateRayCone(primaryRayConeFromPixel(samplePixel, push), hit.t), ray.direction, false),
        sceneMaterialInstanceRef(hit.materialIndex));
    float3 authoredNormal = hit.normal;
    hit = program.program.evaluatedHit();
    seed = program.program.evaluatedSeed();
    float4 normalSample = program.program.sampledNormal();
    SceneSurfaceSummary summary = program.program.surfaceSummary();
    if (push.debugView != kOpenPBRDebugViewFinal) {
        applyAerial = false;
        return openPbrPrimaryDebugColor(push, hit, authoredNormal, normalSample, summary, ray.direction);
    }
    if (summary.unlit) { return summary.baseColor; }
    float3 viewDirection = -ray.direction;
    RasterLightVisibility visibility;
    visibility.debugFlags = push.debugFlags;
    visibility.position = hit.position;
    visibility.pixel = samplePixel;
    var lights = makeClusteredSurfaceLights(visibility, hit, viewDirection, samplePixel, push, summary, useRectLights);
    SurfaceSamplingContext sampling;
    sampling.throughput = float3(1);
    sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
    sampling.exteriorIor = OpenPBRVacuumIor;
#if METALLIC_CUSTOM_MATERIALS
    let shaded = shadeEvaluatedSurface(evaluated, viewDirection, TransportMode.Radiance, sampling, lights);
    float3 radiance = shaded.radiance;
    if (summary.numericalEnvironment) {
        return radiance + sceneSurfaceEnvironment(shaded.prepared, push) * summary.occlusion;
    }
#else
    let prepared = evaluated.closure.prepare(viewDirection, TransportMode.Radiance, sampling);
    float3 emission = max(prepared.surfaceEmission(evaluated.emission), float3(0));
    float3 radiance = integrateDirectLighting(prepared, lights, emission);
#endif
    const bool hasEnvironment = push.environmentMode != kEnvironmentModeDisabled && push.environmentIntensity > 0.0;
    const bool isOpaque = (MATERIAL_CLASS >= 1 && MATERIAL_CLASS <= 3) || opaquePrimary || (summary.transmissionWeight <= 0.0);
    const bool replaceDiffuse = kExportLumenReceivers && isOpaque;
    if (hasEnvironment || replaceDiffuse) {
        float3 normal = hit.normal;
        float roughness = saturate(summary.roughness);
        float3 viewDirectionWS = viewDirection;
        float NoV = saturate(dot(normal, viewDirectionWS));

        const bool isDielectric = (MATERIAL_CLASS == 1) || (fixedMetalness == 0);
        const bool isConductor = (MATERIAL_CLASS == 2) || (fixedMetalness == 1);

#if MATERIAL_CLASS == 1
        // Dielectric class compile-time specialization: conductorFraction is 0, F0 is purely dielectric reflectance.
        float3 f0 = summary.dielectricReflectance;
#elif MATERIAL_CLASS == 2
        // Conductor class compile-time specialization: conductorFraction is 1, F0 is purely baseColor.
        float3 f0 = summary.baseColor;
#else
        DeferredInput conductorFraction = DeferredInput(summary.conductorFraction);
        float3 f0 = isDielectric ? summary.dielectricReflectance :
            (isConductor ? summary.baseColor : lerp(summary.dielectricReflectance, summary.baseColor, conductorFraction));
#endif

        DeferredColor specularWeight = DeferredColor(saturate(approxSpecularIntegralGGX(f0,
            max(roughness * roughness, 0.0001), NoV)));

#if MATERIAL_CLASS == 1
        // Dielectric: conductorFraction is 0, surfaceTransmission is 0 (opaque).
#if METALLIC_DEFERRED_RUNTIME_PRECISION && METALLIC_WORKING_SPACE_ACESCG
        bool useFP16Precision = kDeferredUseFP16Precision;
        DeferredWeight diffuseWeight = deferredPrecisionRoundTrip(DeferredWeight(saturate(summary.diffuseWeight)), useFP16Precision);
#else
        DeferredWeight diffuseWeight = DeferredWeight(saturate(summary.diffuseWeight));
#endif
        DeferredColor diffuseColor = DeferredColor(summary.baseColor) * diffuseWeight *
            (DeferredColor(1.0) - specularWeight);
        float3 diffuse = float3(0.0);
        if (replaceDiffuse) {
            receiver.diffuse = lumenReceiverDiffuse(float3(diffuseColor), summary.occlusion);
#if METALLIC_DEFERRED_LUMEN_AO_ALBEDO
            receiver.diffuseAlbedo = float3(diffuseColor);
#endif
        } else {
            diffuse = environmentDiffuseIrradiance(normal, push) / kPi * float3(diffuseColor);
        }
#elif MATERIAL_CLASS == 2
        // Conductor: zero diffuse contribution.
        float3 diffuse = float3(0.0);
#else
        float3 diffuse = float3(0.0);
        if (!isConductor) {
            DeferredInput surfaceTransmission = DeferredInput(saturate(summary.surfaceTransmission));
#if METALLIC_DEFERRED_RUNTIME_PRECISION && METALLIC_WORKING_SPACE_ACESCG
            bool useFP16Precision = kDeferredUseFP16Precision;
            DeferredWeight diffuseWeight = deferredPrecisionRoundTrip(DeferredWeight(saturate(summary.diffuseWeight)), useFP16Precision) *
                (DeferredWeight(1.0) - conductorFraction) * (isOpaque ? DeferredWeight(1.0) : (DeferredWeight(1.0) - surfaceTransmission));
            diffuseWeight = deferredPrecisionRoundTrip(diffuseWeight, useFP16Precision);
#else
            DeferredWeight diffuseWeight = DeferredWeight(saturate(summary.diffuseWeight)) *
                (DeferredWeight(1.0) - conductorFraction) * (isOpaque ? DeferredWeight(1.0) : (DeferredWeight(1.0) - surfaceTransmission));
#endif
            DeferredColor diffuseColor = DeferredColor(summary.baseColor) * diffuseWeight *
                (DeferredColor(1.0) - specularWeight);
            if (replaceDiffuse) {
                receiver.diffuse = lumenReceiverDiffuse(float3(diffuseColor), summary.occlusion);
#if METALLIC_DEFERRED_LUMEN_AO_ALBEDO
                receiver.diffuseAlbedo = float3(diffuseColor);
#endif
            } else {
                diffuse = environmentDiffuseIrradiance(normal, push) / kPi * float3(diffuseColor);
            }
        }
#endif

        if (replaceDiffuse) {
            receiver.positionAndDepth = float4(hit.position, cameraLinearDepth(hit.position, push));
            receiver.normal = normal;
#if METALLIC_DEFERRED_LUMEN_SPECULAR
            if (!summary.numericalEnvironment) {
                receiver.specularAndRoughness = float4(float3(specularWeight) * summary.occlusion, roughness);
            }
#endif
        }
        if (!hasEnvironment) { return radiance; }

        float3 specular = realtimeEnvironmentSpecular(reflect(-viewDirectionWS, normal), roughness, push) * float3(specularWeight);

#if MATERIAL_CLASS >= 1 && MATERIAL_CLASS <= 3
        // All opaque classes (Dielectric, Conductor, General Opaque) are compile-time guaranteed zero transmission.
        radiance += (diffuse + specular) * summary.occlusion;
#else
        float3 transmitted = float3(0.0);
        if (!isOpaque && summary.transmissionWeight > 0.0) {
            DeferredInput surfaceTransmission = DeferredInput(saturate(summary.surfaceTransmission));
            DeferredColor transmissionColor = DeferredColor(summary.transmissionColor) * surfaceTransmission *
                (DeferredWeight(1.0) - conductorFraction) * (DeferredColor(1.0) - specularWeight);
            float ior = summary.ior;
            float3 refraction = refract(-viewDirectionWS, normal, hit.frontFace ? 1.0 / ior : ior);
            if (dot(refraction, refraction) > 0.0) {
                transmitted = realtimeEnvironmentSpecular(refraction, roughness, push) *
                    float3(transmissionColor);
            }
        }
        radiance += (diffuse + specular + transmitted) * summary.occlusion;
#endif
    }
    return radiance;
}

[ForceInline]
public float3 shadeSceneVisibilitySurface<TProgram : ISceneSurfaceMaterialProgram>(inout Material<TProgram> program, Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, inout DeferredLumenReceiver receiver, int fixedMetalness,
    bool opaquePrimary, bool useRectLights)
{
    bool applyAerial;
    float3 radiance = shadeSceneVisibilitySurfaceRadiance(program, ray, hit, samplePixel, push, seed,
        fixedMetalness, opaquePrimary, applyAerial, receiver, useRectLights);
    if (kExportLumenReceivers && applyAerial) {
        let resources = getSceneResourceParameterField<EnvironmentResourceParameters>(kSceneEnvironmentOffset);
        let segment = physicalAerialSegment(resources, hit.position, ray.origin);
        receiver.diffuse *= segment.transmittance;
#if METALLIC_DEFERRED_LUMEN_SPECULAR
        receiver.specularAndRoughness.rgb *= segment.transmittance;
#endif
        return segment.transmittance * radiance + segment.radiance;
    }
    return applyAerial ? applyPrimaryAerialPerspective(radiance, hit.position, ray.origin) : radiance;
}


public struct DeferredOpenPBRContext : IOpenPBRContext
{
#if METALLIC_CUSTOM_MATERIALS
    public static const bool kEnableSheenAndCoat = true;
#else
    public static const bool kEnableSheenAndCoat = false;
#endif
    public static const bool kEnableDispersion = false;
    public static const bool kEnableThinFilm = false;
    public static const bool kEnableSampling = false;
    public static bool enableSheenAndCoat()
    {
#if METALLIC_CUSTOM_MATERIALS
        return true;
#else
        return false;
#endif
    }
    public static bool enableDispersion() { return false; }
    public static bool enableTranslucency() { return true; }
    public static bool enableMetallic() { return true; }
    public float4 sample2D(int id, float2 uv) { return sampleOpenPBRLut2D(id, uv); }
    public float4 sample3D(int id, float3 uvw) { return sampleOpenPBRLut3D(id, uvw); }
};

public struct DeferredOpenPBRMaterialSource : IOpenPBRMaterialSource
{
    public bool hasTextures(ScenePathTracePush push) { return sceneHasMaterialTextures(push); }
    public typealias Context = DeferredOpenPBRContext;
    public Context context() { return DeferredOpenPBRContext(); }
    public OpenPBRMaterialParameters loadParameters(MaterialInstanceRef instance)
    {
        return loadSceneOpenPBRMaterial(instance.parameterOffset / kSceneMaterialRecordBytes);
    }
    public float4 loadTexture(MaterialTextureInfo textureInfo, float2 uv, float4 fallback,
        ScenePathTracePush push, RayCone cone, HitInfo hit, float3 rayDirection,
        bool deterministicMip0, inout uint seed, TextureFootprint footprint, out bool sampled)
    {
        SceneMaterialTextureSource source;
        return source.loadTexture(textureInfo, uv, fallback, push, cone, hit, rayDirection,
            deterministicMip0, seed, footprint, sampled);
    }
};

[ForceInline]
public float3 shadeVisibilitySurface(Ray ray, HitInfo hit, float2 samplePixel,
    ScenePathTracePush push, inout uint seed, inout DeferredLumenReceiver receiver, int fixedMetalness = -1, bool opaquePrimary = false)
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
        return shadeSceneVisibilitySurface(program, ray, hit, samplePixel, push, seed, receiver, fixedMetalness, opaquePrimary, false);
    }
    if (family == 2u) {
        var program = makeSceneSlabMaterialProgram<SceneDualSlabModel>(hit, push, cone, ray.direction, seed, true, false, specializedProgram);
        return shadeSceneVisibilitySurface(program, ray, hit, samplePixel, push, seed, receiver, fixedMetalness, opaquePrimary, false);
    }
#endif
    Material<OpenPBRMaterialProgram<DeferredOpenPBRMaterialSource>> program = {};
    program.program.hit = hit;
    program.program.push = push;
    program.program.rayCone = cone;
    program.program.rayDirection = ray.direction;
    program.program.seed = seed;
    program.program.faceForwardNormal = true;
    program.program.applyDebugNormalOverrides = false;
    program.program.fixedMetalness = fixedMetalness;
    program.program.opaquePrimary = opaquePrimary;
    return shadeSceneVisibilitySurface(program, ray, hit, samplePixel, push, seed, receiver, fixedMetalness, opaquePrimary, true);
}
} // namespace Metallic.VisibilityLighting
