#ifndef METALLIC_OPENPBR_SURFACE
#define METALLIC_OPENPBR_SURFACE

import Core;
using Metallic;

#define OPENPBR_LANGUAGE_TARGET_SLANG 1
#define OPENPBR_USE_TEXTURE_LUTS 1
#define OPENPBR_ENERGY_TABLES_USE_UINT16 0
#if METALLIC_CUSTOM_MATERIALS
#define OPENPBR_FEATURE_EnableSheenAndCoat true
#else
#define OPENPBR_FEATURE_EnableSheenAndCoat false
#endif
#define OPENPBR_FEATURE_EnableDispersion false
#define OPENPBR_FEATURE_EnableTranslucency true
#define OPENPBR_FEATURE_EnableMetallic true
#define OPENPBR_GET_SPECIALIZATION_CONSTANT(name) OPENPBR_FEATURE_##name

#define gOpenPBRLut2D(index) resolveNonUniform(ResourceHandle<Texture2D<float4>>(getResourceParameters<SceneResourceParameters>().openPBRLut2D.load(index)))

#define gOpenPBRLut3D(index) resolveNonUniform(ResourceHandle<Texture3D<float4>>(getResourceParameters<SceneResourceParameters>().openPBRLut3D.load(index)))

uint openPBRLut2DSlot(int lutId)
{
    if (lutId == 1) {
        return 0u;
    }
    if (lutId == 2) {
        return 1u;
    }
    if (lutId == 4) {
        return 2u;
    }
    if (lutId == 5) {
        return 3u;
    }
    if (lutId == 6) {
        return 4u;
    }
    return 5u;
}

uint openPBRLut3DSlot(int lutId)
{
    return lutId == 3 ? 1u : 0u;
}

float4 sampleOpenPBRLut2D(int lutId, float2 uv)
{
    Texture2D<float4> texture = gOpenPBRLut2D(openPBRLut2DSlot(lutId));
    uint width;
    uint height;
    texture.GetDimensions(width, height);
    float2 size = float2(max(width, 1u), max(height, 1u));
    float2 texel = clamp(uv * size - 0.5, float2(0.0, 0.0), size - 1.0);
    int2 baseTexel = int2(floor(texel));
    int2 nextTexel = min(baseTexel + int2(1, 1), int2(width - 1u, height - 1u));
    float2 frac = texel - float2(baseTexel);
    float4 v00 = texture.Load(int3(baseTexel.x, baseTexel.y, 0));
    float4 v10 = texture.Load(int3(nextTexel.x, baseTexel.y, 0));
    float4 v01 = texture.Load(int3(baseTexel.x, nextTexel.y, 0));
    float4 v11 = texture.Load(int3(nextTexel.x, nextTexel.y, 0));
    return lerp(lerp(v00, v10, frac.x), lerp(v01, v11, frac.x), frac.y);
}

float4 sampleOpenPBRLut3D(int lutId, float3 uvw)
{
    Texture3D<float4> texture = gOpenPBRLut3D(openPBRLut3DSlot(lutId));
    uint width;
    uint height;
    uint depth;
    texture.GetDimensions(width, height, depth);
    float3 size = float3(max(width, 1u), max(height, 1u), max(depth, 1u));
    float3 texel = clamp(uvw * size - 0.5, float3(0.0, 0.0, 0.0), size - 1.0);
    int3 baseTexel = int3(floor(texel));
    int3 nextTexel = min(baseTexel + int3(1, 1, 1), int3(width - 1u, height - 1u, depth - 1u));
    float3 frac = texel - float3(baseTexel);
    float4 v000 = texture.Load(int4(baseTexel.x, baseTexel.y, baseTexel.z, 0));
    float4 v100 = texture.Load(int4(nextTexel.x, baseTexel.y, baseTexel.z, 0));
    float4 v010 = texture.Load(int4(baseTexel.x, nextTexel.y, baseTexel.z, 0));
    float4 v110 = texture.Load(int4(nextTexel.x, nextTexel.y, baseTexel.z, 0));
    float4 v001 = texture.Load(int4(baseTexel.x, baseTexel.y, nextTexel.z, 0));
    float4 v101 = texture.Load(int4(nextTexel.x, baseTexel.y, nextTexel.z, 0));
    float4 v011 = texture.Load(int4(baseTexel.x, nextTexel.y, nextTexel.z, 0));
    float4 v111 = texture.Load(int4(nextTexel.x, nextTexel.y, nextTexel.z, 0));
    float4 z0 = lerp(lerp(v000, v100, frac.x), lerp(v010, v110, frac.x), frac.y);
    float4 z1 = lerp(lerp(v001, v101, frac.x), lerp(v011, v111, frac.x), frac.y);
    return lerp(z0, z1, frac.z);
}

#ifndef OPENPBR_SAMPLE_2D_TEXTURE
#define OPENPBR_SAMPLE_2D_TEXTURE(lut_id, uv) sampleOpenPBRLut2D(lut_id, uv)
#endif
#ifndef OPENPBR_SAMPLE_3D_TEXTURE
#define OPENPBR_SAMPLE_3D_TEXTURE(lut_id, uvw) sampleOpenPBRLut3D(lut_id, uvw)
#endif

#include "../../../External/openpbr-bsdf/openpbr.h"
#include "../../Interop/OpenPBRClosure.hlsli"
import SurfaceLighting;
using Metallic.Lighting;
#include "SceneSurface.slang"
#include "../../Interop/SceneSurfaceClosure.hlsli"

static const uint kOpenPBRDebugViewFinal = 0u;
static const uint kOpenPBRDebugViewGeometryNormal = 1u;
static const uint kOpenPBRDebugViewShadingNormal = 2u;
static const uint kOpenPBRDebugViewMappedNormal = 3u;
static const uint kOpenPBRDebugViewTangent = 4u;
static const uint kOpenPBRDebugViewBitangent = 5u;
static const uint kOpenPBRDebugViewTangentHandedness = 6u;
static const uint kOpenPBRDebugViewTexcoord = 7u;
static const uint kOpenPBRDebugViewFrontFace = 8u;
static const uint kOpenPBRDebugViewMaterial = 9u;
static const uint kOpenPBRDebugViewInstance = 10u;
static const uint kOpenPBRDebugViewTriangle = 11u;
static const uint kOpenPBRDebugViewBaseColor = 12u;
static const uint kOpenPBRDebugViewNormalTexture = 13u;
static const uint kOpenPBRDebugViewShadowTransmittance = 14u;
static const uint kOpenPBRDebugViewShadingSide = 15u;
static const uint kOpenPBRDebugDisableNormalMap = 1u << 0u;
static const uint kOpenPBRDebugForceGeometryNormal = 1u << 1u;
static const uint kOpenPBRDebugDisableMaterialTextures = 1u << 2u;
static const uint kOpenPBRDebugDisableDirectLighting = 1u << 3u;
static const uint kOpenPBRDebugUseOpaqueShadows = 1u << 4u;
static const uint kOpenPBRDebugDisableTransmission = 1u << 5u;
static const uint kOpenPBRDebugDisableShadows = 1u << 6u;
static const uint kOpenPBRDebugDisableVolumeAttenuation = 1u << 7u;

bool openPbrDebugFlag(ScenePathTracePush push, uint flag)
{
    return (push.debugFlags & flag) != 0u;
}

float3 openPbrDebugIdColor(uint id)
{
    uint value = hash(id + 1u);
    return float3(
        float(value & 0xffu),
        float((value >> 8u) & 0xffu),
        float((value >> 16u) & 0xffu)) / 255.0;
}


// Statically supplied resource access. The test provider uses actual GPU data
// and counts logical texture reads without adding counters to production.
interface IOpenPBRMaterialSource
{
    PathTraceMaterial loadParameters(MaterialInstanceRef instance);
    float4 loadTexture(PathTraceTextureInfo textureInfo, float2 uv, float4 fallback,
        ScenePathTracePush push, RayCone cone, HitInfo hit, float3 rayDirection,
        bool deterministicMip0, inout uint seed, TextureFootprint footprint);
};

struct SceneOpenPBRMaterialSource : IOpenPBRMaterialSource
{
    PathTraceMaterial loadParameters(MaterialInstanceRef instance)
    {
        LegacyMaterialProgram loader;
        return loader.loadParameters(gMaterials, instance);
    }
    float4 loadTexture(PathTraceTextureInfo textureInfo, float2 uv, float4 fallback,
        ScenePathTracePush push, RayCone cone, HitInfo hit, float3 rayDirection,
        bool deterministicMip0, inout uint seed, TextureFootprint footprint)
    {
        return loadMaterialTextureForMaterial(textureInfo, uv, fallback, push, cone,
            hit, rayDirection, deterministicMip0, seed, footprint);
    }
};

OpenPBRClosure resolveOpenPBRInputs<TSource : IOpenPBRMaterialSource>(
    HitInfo hit,
    inout PathTraceMaterial material,
    ScenePathTracePush push,
    RayCone rayCone,
    float3 rayDirection,
    inout uint seed, TSource source, TextureFootprint footprint)
{
    bool deterministicGlassTextures = materialNeedsDeterministicTextureMip0(material);
#if METALLIC_CUSTOM_MATERIALS
    MaterialValueTextureContext valueContext;
    valueContext.normalizedRayLod = (footprint.flags & kTextureFootprintIsotropic0) != 0u
        ? footprint.isotropicLod0 : sceneSurfaceMaterialContextForRay(hit, rayCone, rayDirection).footprint.isotropicLod0;
    valueContext.textureCount = push.materialTextureCount;
    valueContext.ntcCount = push.ntcTextureSetCount;
    MaterialValueOpenPBRInputs surface = defaultMaterialValueOpenPBRInputs();
    material = evaluateMaterialValue(hit.materialIndex, hit.position, hit.geometryNormal, hit.texcoord, material, valueContext, surface);
#endif
    float4 baseColorSample = source.loadTexture(
        material.baseColorTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed, footprint);
    float4 metallicRoughnessSample = source.loadTexture(
        material.metallicRoughnessTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed, footprint);
    float4 emissiveSample = source.loadTexture(
        material.emissiveTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed, footprint);
    float4 occlusionSample = source.loadTexture(
        material.occlusionTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed, footprint);
    float4 transmissionSample = source.loadTexture(
        material.transmissionTexture,
        hit.texcoord,
        float4(1.0, 1.0, 1.0, 1.0),
        push,
        rayCone,
        hit,
        rayDirection,
        deterministicGlassTextures,
        seed, footprint);
    float4 specularSample = source.loadTexture(material.specularTexture, hit.texcoord,
        float4(1.0), push, rayCone, hit, rayDirection, deterministicGlassTextures, seed, footprint);
    float4 specularColorSample = source.loadTexture(material.specularColorTexture, hit.texcoord,
        float4(1.0), push, rayCone, hit, rayDirection, deterministicGlassTextures, seed, footprint);
    if (openPbrDebugFlag(push, kOpenPBRDebugDisableMaterialTextures)) {
        baseColorSample = float4(1.0, 1.0, 1.0, 1.0);
        metallicRoughnessSample = float4(1.0, 1.0, 1.0, 1.0);
        emissiveSample = float4(1.0, 1.0, 1.0, 1.0);
        occlusionSample = float4(1.0, 1.0, 1.0, 1.0);
        transmissionSample = float4(1.0, 1.0, 1.0, 1.0);
        specularSample = specularColorSample = float4(1.0);
    }
    float3 baseColor = saturate(materialTextureModulate(material.baseColor.rgb, baseColorSample.rgb, material.baseColorTexture.transform0.w));
    float metallic = saturate(material.params.x * metallicRoughnessSample.b);
    float roughness = clamp(material.params.y * metallicRoughnessSample.g, 0.001, 1.0);
    float transmission = saturate(material.glassParams.x * transmissionSample.r);
    if (openPbrDebugFlag(push, kOpenPBRDebugDisableTransmission)) {
        transmission = 0.0;
    }
    float ior = clamp(material.glassParams.y, 1.0, 3.0);
    float3 emissive = max(materialTextureModulate(material.emissive.rgb, emissiveSample.rgb, material.emissiveTexture.transform0.w), float3(0.0, 0.0, 0.0));

    OpenPBRClosure result;
    result.inputs = openpbr_make_default_resolved_inputs();
    result.occlusion = lerp(1.0, saturate(occlusionSample.r), saturate(material.textureParams.y));

    result.inputs.base_weight = 1.0;
    result.inputs.base_color = baseColor;
    // glTF specifies a Lambertian diffuse lobe, independently of GGX roughness.
    result.inputs.base_diffuse_roughness = 0.0;
    result.inputs.base_metalness = metallic;

    result.inputs.specular_weight = lerp(saturate(material.specular.w * specularSample.a), 1.0, metallic);
    result.inputs.specular_color = lerp(saturate(materialTextureModulate(material.specular.rgb, specularColorSample.rgb, material.specularColorTexture.transform0.w)), float3(1.0), metallic);
    result.inputs.specular_roughness = roughness;
    result.inputs.specular_roughness_anisotropy = 0.0;
    result.inputs.specular_ior = ior;
    result.inputs.specular_anisotropy_rotation_cos_sin = float2(1.0, 0.0);

    result.inputs.transmission_weight = transmission * (1.0 - metallic);
    result.inputs.transmission_color = powColor(baseColor, 0.5);
    result.inputs.transmission_depth = 0.0;
    result.inputs.transmission_scatter = float3(0.0, 0.0, 0.0);
    result.inputs.transmission_scatter_anisotropy = 0.0;

    result.inputs.subsurface_weight = 0.0;

    if (material.emissive.w != 0.0) {
        // KHR_materials_unlit: base color is the final radiance, independent of lights.
        result.inputs.base_weight = 0.0;
        result.inputs.specular_weight = 0.0;
        result.inputs.transmission_weight = 0.0;
        emissive = baseColor;
    }
    float emissionLuminance = luminanceMax(emissive);
    result.inputs.emission_luminance = emissionLuminance;
    result.inputs.emission_color = emissionLuminance > 0.0 ? emissive / emissionLuminance : float3(1.0, 1.0, 1.0);

    result.inputs.geometry_opacity = saturate(material.baseColor.a * baseColorSample.a);
    result.inputs.geometry_thin_walled = false;
    result.inputs.geometry_basis = openpbr_make_basis(
        safeNormalize(hit.normal, hit.geometryNormal),
        safeNormalize(hit.tangent.xyz, float3(1.0, 0.0, 0.0)),
        safeNormalize(hit.bitangent, float3(0.0, 1.0, 0.0)));
    result.inputs.geometry_coat_basis = result.inputs.geometry_basis;
#if METALLIC_CUSTOM_MATERIALS
    result.inputs.coat_weight = surface.coatWeight;
    result.inputs.coat_roughness = surface.coatRoughness;
    result.inputs.coat_ior = surface.coatIOR;
    result.inputs.fuzz_weight = surface.fuzzWeight;
    result.inputs.fuzz_color = surface.fuzzColor;
    result.inputs.fuzz_roughness = surface.fuzzRoughness;
    result.inputs.specular_roughness_anisotropy = surface.specularAnisotropy;
    result.inputs.specular_anisotropy_rotation_cos_sin = dot(surface.anisotropyTangent, surface.anisotropyTangent) > 1e-12
        ? normalize(surface.anisotropyTangent) : float2(1, 0);
    // Explicit OpenPBR volume color is applied by the interior stack, without
    // the legacy glTF base-color surface tint being applied a second time.
    if (surface.volumeColor) { result.inputs.transmission_color = float3(1); }
#endif
    return result;
}


// Program-local source state is discarded after evaluate. Closures contain only
// resolved numeric inputs; stochastic filtering never advances inside lighting.
struct OpenPBRMaterialProgram<TSource : IOpenPBRMaterialSource> : ISurfaceMaterialProgram
{
    typealias Closure = SceneSurfaceClosure;
    TSource source;
    ScenePathTracePush push;
    RayCone rayCone;
    float3 rayDirection;
    HitInfo hit;
    uint seed;
    bool faceForwardNormal;
    bool applyDebugNormalOverrides;
    int fixedMetalness;
    bool opaquePrimary;
    PathTraceMaterial material;
    float4 normalSample;
    uint specializedProgramId;

    [mutating]
    SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        material = source.loadParameters(instance);
        hit.position = context.positionWS;
        hit.geometryNormal = context.geometryNormalWS;
        hit.normal = context.shadingNormalWS;
        hit.tangent.xyz = context.tangentWS;
        hit.bitangent = context.bitangentWS;
        hit.texcoord = context.texCoord0;
        hit.frontFace = context.frontFace != 0u;
        hit.materialIndex = instance.parameterOffset / kLegacyMaterialRecordBytes;
        normalSample = source.loadTexture(material.normalTexture, hit.texcoord,
            float4(0.5, 0.5, 1.0, 1.0), push, rayCone, hit, rayDirection,
            materialNeedsDeterministicTextureMip0(material), seed, context.footprint);
        if (applyDebugNormalOverrides && (openPbrDebugFlag(push, kOpenPBRDebugDisableNormalMap) ||
            openPbrDebugFlag(push, kOpenPBRDebugDisableMaterialTextures))) {
            normalSample = float4(0.5, 0.5, 1.0, 1.0);
        }
        hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent,
            normalSample, material.textureParams.x);
        if (applyDebugNormalOverrides && openPbrDebugFlag(push, kOpenPBRDebugForceGeometryNormal)) {
            hit.normal = hit.geometryNormal;
            float3 referenceAxis = abs(hit.normal.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0);
            float3 fallbackTangent = safeNormalize(cross(referenceAxis, hit.normal), float3(1, 0, 0));
            float3 tangent = hit.tangent.xyz - hit.normal * dot(hit.normal, hit.tangent.xyz);
            tangent = safeNormalize(tangent, fallbackTangent);
            float tangentSign = hit.tangent.w >= 0.0 ? 1.0 : -1.0;
            hit.tangent = float4(tangent, tangentSign);
            hit.bitangent = safeNormalize(cross(hit.normal, tangent) * tangentSign,
                cross(hit.normal, fallbackTangent));
        }
        // Compatibility policy: deferred/guides face-forward after mapping;
        // the path tracer preserves both sides for nested dielectric transport.
        if (faceForwardNormal) {
            hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, rayDirection);
        }
        SurfaceMaterialResult<Closure> result = (SurfaceMaterialResult<Closure>)0;
#if METALLIC_CUSTOM_MATERIALS
        uint family = materialClosureFamily(hit.materialIndex, specializedProgramId);
        if (family != 0u) {
            MaterialValueTextureContext valueContext;
            valueContext.normalizedRayLod = (context.footprint.flags & kTextureFootprintIsotropic0) != 0u
                ? context.footprint.isotropicLod0 : sceneSurfaceMaterialContextForRay(hit, rayCone, rayDirection).footprint.isotropicLod0;
            valueContext.textureCount = push.materialTextureCount;
            valueContext.ntcCount = push.ntcTextureSetCount;
            let values = evaluateMaterialSlab(hit.materialIndex, hit.position, hit.geometryNormal, hit.texcoord,
                material, valueContext, specializedProgramId);
            result.closure.family = family;
            result.closure.slab.first = values.inputs.first;
            result.closure.slab.second = values.inputs.second;
            result.closure.slab.control = values.inputs.control;
            result.closure.slab.normal = float4(hit.normal, 0);
            result.closure.resolvedEmission = values.emission;
            result.closure.occlusion = 1.0;
            result.closure.inputs = openpbr_make_default_resolved_inputs();
            result.closure.inputs.base_color = values.inputs.first.reflectance.rgb;
            if (values.inputs.control.x == 1.0) {
                result.closure.inputs.base_color = lerp(values.inputs.first.reflectance.rgb,
                    values.inputs.second.reflectance.rgb, values.inputs.control.y);
            } else if (values.inputs.control.x == 2.0) {
                // Normal-incidence reflectance bound for diagnostic/guide consumers.
                float3 transmission = 1.0 - values.inputs.first.reflectance.rgb;
                result.closure.inputs.base_color += transmission * transmission * values.inputs.second.reflectance.rgb *
                    exp(-2.0 * values.inputs.first.opticalDepth.rgb);
            }
            result.closure.inputs.base_metalness = 0.0;
            result.closure.inputs.specular_weight = 0.0;
            result.closure.inputs.specular_ior = 1.0;
            result.closure.inputs.specular_roughness = 1.0;
            result.closure.inputs.geometry_basis = openpbr_make_basis(hit.normal, hit.tangent.xyz, hit.bitangent);
            result.emission = values.emission;
            return result;
        }
        let openPBR = resolveOpenPBRInputs(hit, material, push, rayCone, rayDirection, seed, source, context.footprint);
        result.closure.inputs = openPBR.inputs;
        result.closure.occlusion = openPBR.occlusion;
#else
        result.closure = resolveOpenPBRInputs(hit, material, push, rayCone, rayDirection, seed, source, context.footprint);
#endif
        if (fixedMetalness >= 0) { result.closure.inputs.base_metalness = float(fixedMetalness); }
        if (opaquePrimary) { result.closure.inputs.transmission_weight = 0.0; }
        result.emission = result.closure.inputs.emission_luminance * result.closure.inputs.emission_color;
        result.flags = 0u;
        return result;
    }
};

typealias SceneOpenPBRMaterialProgram = OpenPBRMaterialProgram<SceneOpenPBRMaterialSource>;

#if METALLIC_CUSTOM_MATERIALS
// Realtime IBL integrates the actual Slab BSDF; OpenPBR's analytic split-sum
// approximation cannot represent the angle-dependent two-slab layer.
float3 sceneSurfaceEnvironment(ScenePreparedSurfaceClosure prepared, ScenePathTracePush push, inout uint seed)
{
    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) { return float3(0); }
    float3 radiance = float3(0);
    uint count = max(push.samples, 1u);
    for (uint i = 0u; i < count; ++i) {
        BSDFWeightSample scatter = sampleSurfaceWeight(prepared, float3(rand(seed), rand(seed), rand(seed)));
        if (scatter.pdf > 0.0 && scatter.flags != 0u) {
            radiance += max(scatter.weight, float3(0)) * environmentRadiance(scatter.wiWS, push, false);
        }
    }
    return radiance / float(count);
}
#endif

SceneOpenPBRMaterialProgram makeOpenPBRMaterialProgram(HitInfo hit, ScenePathTracePush push,
    RayCone rayCone, float3 rayDirection, uint seed, bool faceForwardNormal, bool debugNormalOverrides)
{
    SceneOpenPBRMaterialProgram program = (SceneOpenPBRMaterialProgram)0;
    program.hit = hit;
    program.push = push;
    program.rayCone = rayCone;
    program.rayDirection = rayDirection;
    program.seed = seed;
    program.faceForwardNormal = faceForwardNormal;
    program.applyDebugNormalOverrides = debugNormalOverrides;
    program.fixedMetalness = -1;
    return program;
}

float3 openPbrPrimaryDebugColor(
    ScenePathTracePush push,
    HitInfo hit,
    float3 shadingNormal,
    float4 normalSample,
    SceneSurfaceClosure materialEval,
    float3 rayDirection)
{
    if (push.debugView == kOpenPBRDebugViewGeometryNormal) {
        return hit.geometryNormal * 0.5 + 0.5;
    }
    if (push.debugView == kOpenPBRDebugViewShadingNormal) {
        return shadingNormal * 0.5 + 0.5;
    }
    if (push.debugView == kOpenPBRDebugViewMappedNormal) {
        return hit.normal * 0.5 + 0.5;
    }
    if (push.debugView == kOpenPBRDebugViewTangent) {
        return safeNormalize(hit.tangent.xyz, float3(1.0, 0.0, 0.0)) * 0.5 + 0.5;
    }
    if (push.debugView == kOpenPBRDebugViewBitangent) {
        return safeNormalize(hit.bitangent, float3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    }
    if (push.debugView == kOpenPBRDebugViewTangentHandedness) {
        return hit.tangent.w >= 0.0
            ? float3(0.1, 0.9, 0.2)
            : float3(0.9, 0.1, 0.2);
    }
    if (push.debugView == kOpenPBRDebugViewTexcoord) {
        return float3(frac(hit.texcoord), 0.0);
    }
    if (push.debugView == kOpenPBRDebugViewFrontFace) {
        return hit.frontFace
            ? float3(0.1, 0.9, 0.2)
            : float3(0.9, 0.1, 0.2);
    }
    if (push.debugView == kOpenPBRDebugViewMaterial) {
        return openPbrDebugIdColor(hit.materialIndex);
    }
    if (push.debugView == kOpenPBRDebugViewInstance) {
        return openPbrDebugIdColor(hit.instanceIndex);
    }
    if (push.debugView == kOpenPBRDebugViewTriangle) {
        return openPbrDebugIdColor(hit.triangleIndex);
    }
    if (push.debugView == kOpenPBRDebugViewBaseColor) {
        return Metallic.WorkingColor::toLinearRec709(materialEval.inputs.base_color);
    }
    if (push.debugView == kOpenPBRDebugViewNormalTexture) {
        return normalSample.rgb;
    }
    if (push.debugView == kOpenPBRDebugViewShadingSide) {
        return dot(hit.normal, -rayDirection) >= 0.0
            ? float3(0.1, 0.9, 0.2)
            : float3(0.9, 0.1, 0.2);
    }
    return float3(1.0, 0.0, 1.0);
}

float3 openPbrDisplayDebugColor(ScenePathTracePush push, float3 color)
{
    // BaseColor is already display-linear Rec.709; normal/ID values are display code values.
    if (push.debugView == kOpenPBRDebugViewShadowTransmittance) {
        color = Metallic.WorkingColor::toLinearRec709(color);
    }
    return saturate(color);
}


#endif
