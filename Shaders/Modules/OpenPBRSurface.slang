#language slang 2026
module OpenPBRSurface;

import Core;
import OpenPBR;
import OpenPBRClosure;
import OpenPBRTextureLUT;
import MaterialProgram;
import SurfaceLighting;
import SceneSurface;
import SceneSurfaceMaterial;
import Material;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import ScenePathTypes;
import ScenePathMath;
import MaterialTextureDecode;
import Lighting;
import WorkingColor;
#if METALLIC_CUSTOM_MATERIALS
import SlabClosure;
#endif

namespace Metallic.SceneOpenPBR
{
using Metallic;
using Metallic.Material;
using Metallic.OpenPBR;
using Metallic.Lighting;
using Metallic.SceneSurface;



public float4 sampleOpenPBRLut2D(int lutId, float2 uv)
{
#if SHARC_UPDATE
    let resources = getSceneResourceParameters();
#else
    let resources = getSceneResourceParameterField<SceneOpenPBRLUTResourceParameters>(kSceneOpenPBRLUTsOffset);
#endif
    let texture = ResourceHandle<Texture2D<float4>>(resources.openPBRLut2D.load(openPBRLut2DSlot(lutId)));
    return sampleOpenPBRTexture2D(resolveNonUniform(texture), resolveUniform(resources.openPBRLutSampler), uv);
}

public float4 sampleOpenPBRLut3D(int lutId, float3 uvw)
{
#if SHARC_UPDATE
    let resources = getSceneResourceParameters();
#else
    let resources = getSceneResourceParameterField<SceneOpenPBRLUTResourceParameters>(kSceneOpenPBRLUTsOffset);
#endif
    let texture = ResourceHandle<Texture3D<float4>>(resources.openPBRLut3D.load(openPBRLut3DSlot(lutId)));
    return sampleOpenPBRTexture3D(resolveNonUniform(texture), resolveUniform(resources.openPBRLutSampler), uvw);
}

// The closure module receives a statically specialized BSDF LUT context.
// Tests can supply a concrete context without importing this program's resources.
public struct SceneOpenPBRContext : IOpenPBRContext
{
#if METALLIC_CUSTOM_MATERIALS
    public static const bool kEnableSheenAndCoat = true;
#else
    public static const bool kEnableSheenAndCoat = false;
#endif
    public static const bool kEnableDispersion = false;
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
public typealias SceneOpenPBRClosure = OpenPBRClosure<SceneOpenPBRContext>;
public typealias SceneOpenPBRPreparedClosure = OpenPBRPreparedClosure<SceneOpenPBRContext>;


public static const uint kOpenPBRDebugViewFinal = 0u;
public static const uint kOpenPBRDebugViewGeometryNormal = 1u;
public static const uint kOpenPBRDebugViewShadingNormal = 2u;
public static const uint kOpenPBRDebugViewMappedNormal = 3u;
public static const uint kOpenPBRDebugViewTangent = 4u;
public static const uint kOpenPBRDebugViewBitangent = 5u;
public static const uint kOpenPBRDebugViewTangentHandedness = 6u;
public static const uint kOpenPBRDebugViewTexcoord = 7u;
public static const uint kOpenPBRDebugViewFrontFace = 8u;
public static const uint kOpenPBRDebugViewMaterial = 9u;
public static const uint kOpenPBRDebugViewInstance = 10u;
public static const uint kOpenPBRDebugViewTriangle = 11u;
public static const uint kOpenPBRDebugViewBaseColor = 12u;
public static const uint kOpenPBRDebugViewNormalTexture = 13u;
public static const uint kOpenPBRDebugViewShadowTransmittance = 14u;
public static const uint kOpenPBRDebugViewShadingSide = 15u;
public static const uint kOpenPBRDebugDisableNormalMap = 1u << 0u;
public static const uint kOpenPBRDebugForceGeometryNormal = 1u << 1u;
public static const uint kOpenPBRDebugDisableMaterialTextures = 1u << 2u;
public static const uint kOpenPBRDebugDisableDirectLighting = 1u << 3u;
public static const uint kOpenPBRDebugUseOpaqueShadows = 1u << 4u;
public static const uint kOpenPBRDebugDisableTransmission = 1u << 5u;
public static const uint kOpenPBRDebugDisableShadows = 1u << 6u;
public static const uint kOpenPBRDebugDisableVolumeAttenuation = 1u << 7u;

public bool openPbrDebugFlag(ScenePathTracePush push, uint flag)
{
    return (push.debugFlags & flag) != 0u;
}

public float3 openPbrDebugIdColor(uint id)
{
    uint value = Metallic.ScenePath.hash(id + 1u);
    return float3(
        float(value & 0xffu),
        float((value >> 8u) & 0xffu),
        float((value >> 16u) & 0xffu)) / 255.0;
}

// Statically supplied resource access. The test provider uses actual GPU data
// and counts logical texture reads without adding counters to production.
public interface ISceneMaterialTextureSource
{
    public float4 loadTexture(MaterialTextureInfo textureInfo, float2 uv, float4 fallback,
        ScenePathTracePush push, RayCone cone, HitInfo hit, float3 rayDirection,
        bool deterministicMip0, inout uint seed, TextureFootprint footprint);
};

public interface IOpenPBRMaterialSource : ISceneMaterialTextureSource
{
    public associatedtype Context : IOpenPBRContext;
    public Context context();
    public OpenPBRMaterialParameters loadParameters(MaterialInstanceRef instance);
};

public struct SceneMaterialTextureSource : ISceneMaterialTextureSource
{
    public float4 loadTexture(MaterialTextureInfo textureInfo, float2 uv, float4 fallback,
        ScenePathTracePush push, RayCone cone, HitInfo hit, float3 rayDirection,
        bool deterministicMip0, inout uint seed, TextureFootprint footprint)
    {
        return loadMaterialTextureForMaterial(textureInfo, uv, fallback, push, cone,
            hit, rayDirection, deterministicMip0, seed, footprint);
    }
};

public struct SceneOpenPBRMaterialSource : IOpenPBRMaterialSource
{
    public typealias Context = SceneOpenPBRContext;
    public Context context() { return SceneOpenPBRContext(); }
    public OpenPBRMaterialParameters loadParameters(MaterialInstanceRef instance)
    {
        return loadSceneOpenPBRMaterial(instance.parameterOffset / kSceneMaterialRecordBytes);
    }
    public float4 loadTexture(MaterialTextureInfo textureInfo, float2 uv, float4 fallback,
        ScenePathTracePush push, RayCone cone, HitInfo hit, float3 rayDirection,
        bool deterministicMip0, inout uint seed, TextureFootprint footprint)
    {
        SceneMaterialTextureSource source;
        return source.loadTexture(textureInfo, uv, fallback, push, cone, hit, rayDirection,
            deterministicMip0, seed, footprint);
    }
};

public void updateSceneSurfaceHit(inout HitInfo hit, SurfaceMaterialContext context, uint materialIndex)
{
    hit.position = context.positionWS;
    hit.geometryNormal = context.geometryNormalWS;
    hit.normal = context.shadingNormalWS;
    hit.tangent.xyz = context.tangentWS;
    hit.bitangent = context.bitangentWS;
    hit.texcoord = context.texCoord0;
    hit.frontFace = context.frontFace != 0u;
    hit.materialIndex = materialIndex;
}

public void resolveSceneSurfaceNormal<TSource : ISceneMaterialTextureSource>(
    inout HitInfo hit, MaterialTextureInfo normalTexture, float normalScale,
    ScenePathTracePush push, RayCone rayCone, float3 rayDirection, bool deterministicMip0,
    bool faceForwardNormal, bool applyDebugNormalOverrides, TextureFootprint footprint,
    TSource source, inout uint seed, out float4 normalSample)
{
    normalSample = source.loadTexture(normalTexture, hit.texcoord,
        float4(0.5, 0.5, 1.0, 1.0), push, rayCone, hit, rayDirection,
        deterministicMip0, seed, footprint);
    if (applyDebugNormalOverrides && (openPbrDebugFlag(push, kOpenPBRDebugDisableNormalMap) ||
        openPbrDebugFlag(push, kOpenPBRDebugDisableMaterialTextures))) {
        normalSample = float4(0.5, 0.5, 1.0, 1.0);
    }
    hit.normal = applyNormalMap(hit.normal, hit.geometryNormal, hit.tangent, hit.bitangent,
        normalSample, normalScale);
    if (applyDebugNormalOverrides && openPbrDebugFlag(push, kOpenPBRDebugForceGeometryNormal)) {
        hit.normal = hit.geometryNormal;
        float3 referenceAxis = abs(hit.normal.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0);
        float3 fallbackTangent = Metallic.SceneSurface.safeNormalize(cross(referenceAxis, hit.normal), float3(1, 0, 0));
        float3 tangent = hit.tangent.xyz - hit.normal * dot(hit.normal, hit.tangent.xyz);
        tangent = Metallic.SceneSurface.safeNormalize(tangent, fallbackTangent);
        float tangentSign = hit.tangent.w >= 0.0 ? 1.0 : -1.0;
        hit.tangent = float4(tangent, tangentSign);
        hit.bitangent = Metallic.SceneSurface.safeNormalize(cross(hit.normal, tangent) * tangentSign,
            cross(hit.normal, fallbackTangent));
    }
    // Apply the per-pass face-forward policy only after authored TBN mapping.
    if (faceForwardNormal) {
        hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, rayDirection);
    }
}

public OpenPBRClosure<TSource.Context> resolveOpenPBRInputs<TSource : IOpenPBRMaterialSource>(
    inout HitInfo hit,
    inout OpenPBRMaterialParameters material,
    ScenePathTracePush push,
    RayCone rayCone,
    float3 rayDirection,
    inout uint seed, TSource source, TextureFootprint footprint, float3 authoredNormal, bool faceForwardNormal)
{
    bool deterministicGlassTextures = materialNeedsDeterministicTextureMip0(material);
#if METALLIC_CUSTOM_MATERIALS
    MaterialValueTextureContext valueContext;
    valueContext.normalizedRayLod = (footprint.flags & kTextureFootprintIsotropic0) != 0u
        ? footprint.isotropicLod0 : sceneSurfaceMaterialContextForRay(hit, rayCone, rayDirection).footprint.isotropicLod0;
    valueContext.textureCount = push.materialTextureCount;
    MaterialValueOpenPBRInputs surface = defaultMaterialValueOpenPBRInputs();
    material = evaluateMaterialValue(hit.materialIndex, hit.position, hit.geometryNormal, hit.texcoord, material, valueContext, surface);
    if (surface.hasNormal && !openPbrDebugFlag(push, kOpenPBRDebugDisableNormalMap) &&
        !openPbrDebugFlag(push, kOpenPBRDebugDisableMaterialTextures) &&
        !openPbrDebugFlag(push, kOpenPBRDebugForceGeometryNormal)) {
        // Graph normals replace the legacy map, using stable authored TBN before face-forward.
        hit.normal = applyNormalMap(authoredNormal, hit.geometryNormal, hit.tangent, hit.bitangent,
            float4(Metallic.SceneSurface.safeNormalize(surface.normalTS, float3(0,0,1)) * 0.5 + 0.5, 1), 1.0);
        if (faceForwardNormal) { hit.normal = faceForwardShadingNormal(hit.normal, hit.geometryNormal, rayDirection); }
    }
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
#if METALLIC_CUSTOM_MATERIALS
    // Frontends author final surface inputs; texture expressions must not be multiplied twice.
    if ((surface.overrideMask & 1u) != 0u) { baseColor = material.baseColor.rgb; }
    if ((surface.overrideMask & 2u) != 0u) { metallic = saturate(material.params.x); }
    if ((surface.overrideMask & 4u) != 0u) { roughness = clamp(material.params.y, 0.001, 1.0); }
    if ((surface.overrideMask & 8u) != 0u) { emissive = material.emissive.rgb; }
#endif

    OpenPBRClosure<TSource.Context> result;
    result.context = source.context();
    result.inputs = openPBRMakeDefaultResolvedInputs();
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
    result.inputs.geometry_basis = openPBRMakeBasis(
        Metallic.SceneSurface.safeNormalize(hit.normal, hit.geometryNormal),
        Metallic.SceneSurface.safeNormalize(hit.tangent.xyz, float3(1.0, 0.0, 0.0)),
        Metallic.SceneSurface.safeNormalize(hit.bitangent, float3(0.0, 1.0, 0.0)));
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
public struct OpenPBRMaterialProgramImpl<TSource : IOpenPBRMaterialSource> : ISceneSurfaceMaterialProgram
{
    public typealias Closure = OpenPBRClosure<TSource.Context>;
    public TSource source;
    public ScenePathTracePush push;
    public RayCone rayCone;
    public float3 rayDirection;
    public HitInfo hit;
    public uint seed;
    public bool faceForwardNormal;
    public bool applyDebugNormalOverrides;
    public int fixedMetalness;
    public bool opaquePrimary;
    public OpenPBRMaterialParameters material;
    public float4 normalSample;
    public SceneSurfaceSummary summary;
    public HitInfo evaluatedHit() { return hit; }
    public uint evaluatedSeed() { return seed; }
    public float4 sampledNormal() { return normalSample; }
    public SceneSurfaceSummary surfaceSummary() { return summary; }

    [mutating]
    public SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        material = source.loadParameters(instance);
        updateSceneSurfaceHit(hit, context, instance.parameterOffset / kSceneMaterialRecordBytes);
        resolveSceneSurfaceNormal(hit, material.normalTexture, material.textureParams.x, push,
            rayCone, rayDirection, materialNeedsDeterministicTextureMip0(material), faceForwardNormal,
            applyDebugNormalOverrides, context.footprint, source, seed, normalSample);
        SurfaceMaterialResult<Closure> result = (SurfaceMaterialResult<Closure>)0;
        result.closure = resolveOpenPBRInputs(hit, material, push, rayCone, rayDirection, seed, source,
            context.footprint, context.shadingNormalWS, faceForwardNormal);
        if (fixedMetalness >= 0) { result.closure.inputs.base_metalness = float(fixedMetalness); }
        if (opaquePrimary) { result.closure.inputs.transmission_weight = 0.0; }
        result.emission = result.closure.inputs.emission_luminance * result.closure.inputs.emission_color;
        result.flags = 0u;
        summary = summarizeOpenPBRSurface(result.closure, material.emissive.w != 0.0);
        return result;
    }
};

public typealias OpenPBRMaterialProgram<TSource : IOpenPBRMaterialSource> = OpenPBRMaterialProgramImpl<TSource>;

public typealias SceneOpenPBRMaterialProgram = OpenPBRMaterialProgram<SceneOpenPBRMaterialSource>;

public uint sceneSurfaceMaterialFamily(uint materialIndex, uint specializedProgram = 0u)
{
#if METALLIC_CUSTOM_MATERIALS
    return materialClosureFamily(materialIndex, specializedProgram);
#else
    return 0u;
#endif
}

#if METALLIC_CUSTOM_MATERIALS
// Each static model produces only its own closure. Family selection belongs to
// the renderer's hit/pass boundary, outside material evaluation and lighting.
public interface ISceneSlabModel
{
    public associatedtype Closure : ISurfaceClosure;
    public Closure resolve(SlabMaterialInputs inputs, float3 normalWS);
};

public struct SceneSingleSlabModel : ISceneSlabModel
{
    public typealias Closure = SingleSlabClosure;
    public Closure resolve(SlabMaterialInputs inputs, float3 normalWS)
    {
        Closure closure;
        closure.slab = inputs.first;
        closure.normal = float4(normalWS, 0);
        return closure;
    }
};

public struct SceneDualSlabModel : ISceneSlabModel
{
    public typealias Closure = DualSlabClosure;
    public Closure resolve(SlabMaterialInputs inputs, float3 normalWS)
    {
        Closure closure;
        closure.first = inputs.first;
        closure.second = inputs.second;
        closure.control = inputs.control;
        closure.normal = float4(normalWS, 0);
        return closure;
    }
};

public struct SceneSlabMaterialProgram<TModel : ISceneSlabModel> : ISceneSurfaceMaterialProgram
{
    public typealias Closure = TModel.Closure;
    public TModel model;
    public uint specializedProgramId;
    public ScenePathTracePush push;
    public RayCone rayCone;
    public float3 rayDirection;
    public HitInfo hit;
    public uint seed;
    public bool faceForwardNormal;
    public bool applyDebugNormalOverrides;
    public float4 normalSample;
    public SceneSurfaceSummary summary;

    public HitInfo evaluatedHit() { return hit; }
    public uint evaluatedSeed() { return seed; }
    public float4 sampledNormal() { return normalSample; }
    public SceneSurfaceSummary surfaceSummary() { return summary; }

    [mutating]
    public SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        uint materialIndex = instance.parameterOffset / kSceneMaterialRecordBytes;
        updateSceneSurfaceHit(hit, context, materialIndex);
        let transport = loadSceneMaterial(materialIndex);
        SceneMaterialTextureSource textureSource;
        resolveSceneSurfaceNormal(hit, transport.normalTexture, transport.textureParams.x, push,
            rayCone, rayDirection, false, faceForwardNormal, applyDebugNormalOverrides,
            context.footprint, textureSource, seed, normalSample);
        MaterialValueTextureContext valueContext;
        valueContext.normalizedRayLod = (context.footprint.flags & kTextureFootprintIsotropic0) != 0u
            ? context.footprint.isotropicLod0
            : sceneSurfaceMaterialContextForRay(hit, rayCone, rayDirection).footprint.isotropicLod0;
        valueContext.textureCount = push.materialTextureCount;
        let values = evaluateMaterialSlab(materialIndex, hit.position, hit.geometryNormal, hit.texcoord,
            sceneMaterialValueParameters(transport), valueContext, specializedProgramId);
        SurfaceMaterialResult<Closure> result = (SurfaceMaterialResult<Closure>)0;
        result.closure = model.resolve(values.inputs, hit.normal);
        result.emission = values.emission;
        summary = summarizeSlabSurface(values.inputs);
        return result;
    }
};

public typealias SceneSingleSlabMaterialProgram = SceneSlabMaterialProgram<SceneSingleSlabModel>;
public typealias SceneDualSlabMaterialProgram = SceneSlabMaterialProgram<SceneDualSlabModel>;

public Material<SceneSlabMaterialProgram<TModel>> makeSceneSlabMaterialProgram<TModel : ISceneSlabModel>(
    HitInfo hit, ScenePathTracePush push, RayCone rayCone, float3 rayDirection, uint seed,
    bool faceForwardNormal, bool debugNormalOverrides, uint specializedProgram = 0u)
{
    Material<SceneSlabMaterialProgram<TModel>> material = (Material<SceneSlabMaterialProgram<TModel>>)0;
    material.program.specializedProgramId = specializedProgram;
    material.program.hit = hit;
    material.program.push = push;
    material.program.rayCone = rayCone;
    material.program.rayDirection = rayDirection;
    material.program.seed = seed;
    material.program.faceForwardNormal = faceForwardNormal;
    material.program.applyDebugNormalOverrides = debugNormalOverrides;
    return material;
}
#endif

#if METALLIC_CUSTOM_MATERIALS
public float sceneEnvironmentRadicalInverse(uint index, uint base)
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

// Integrate the actual layered/anisotropic BSDF where split-sum IBL is insufficient.
// These passes have no temporal accumulation. Share deterministic quadrature across
// pixels instead of decorrelating a few samples into high-variance spatial noise.
public float3 sceneSurfaceEnvironment<TPrepared : IPreparedSurfaceClosure>(TPrepared prepared, ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) { return float3(0); }
    float3 radiance = float3(0);
    uint count = max(push.samples, 1u);
    // Reuse the environment importance distribution, with the same directions
    // at every pixel. MIS covers small HDR emitters as well as narrow BSDF lobes.
    // The procedural sky sampler represents only its key light, so it cannot
    // be combined with an integral of the full sky using this estimator.
    bool useEnvironmentMis = push.environmentMode == kEnvironmentModeMap;
    uint environmentSeed = 0x51ed270bu;
    for (uint i = 0u; i < count; ++i) {
        if (useEnvironmentMis) {
            EnvironmentLightSample light;
            if (sampleEnvironmentLight(push, environmentSeed, light)) {
                float bsdfPdf = prepared.pdf(light.direction);
                radiance += evaluateSurfaceProjected(prepared, light.direction) * light.radiance *
                    (misPowerHeuristic(light.pdfSolidAngle, bsdfPdf) / light.pdfSolidAngle);
            }
        }
        float3 random = float3((float(i) + 0.5) / float(count),
            sceneEnvironmentRadicalInverse(i + 1u, 2u), sceneEnvironmentRadicalInverse(i + 1u, 3u));
        BSDFWeightSample scatter = sampleSurfaceWeight(prepared, random);
        if (scatter.pdf > 0.0 && scatter.flags != 0u) {
            float misWeight = useEnvironmentMis
                ? environmentBsdfMisWeight(push, scatter.pdf, scatter.wiWS, (scatter.flags & kBSDFDelta) != 0u)
                : 1.0;
            radiance += max(scatter.weight, float3(0)) * environmentRadiance(scatter.wiWS, push, false) * misWeight;
        }
    }
    return radiance / float(count);
}
#endif

public Material<SceneOpenPBRMaterialProgram> makeOpenPBRMaterialProgram(HitInfo hit, ScenePathTracePush push,
    RayCone rayCone, float3 rayDirection, uint seed, bool faceForwardNormal, bool debugNormalOverrides)
{
    Material<SceneOpenPBRMaterialProgram> material = {};
    material.program.hit = hit;
    material.program.push = push;
    material.program.rayCone = rayCone;
    material.program.rayDirection = rayDirection;
    material.program.seed = seed;
    material.program.faceForwardNormal = faceForwardNormal;
    material.program.applyDebugNormalOverrides = debugNormalOverrides;
    material.program.fixedMetalness = -1;
    return material;
}

public SceneSurfaceSummary evaluateSceneSurfaceSummary<TProgram : ISceneSurfaceMaterialProgram>(
    inout Material<TProgram> material, SurfaceMaterialContext context, MaterialInstanceRef instance,
    inout HitInfo hit, inout uint seed)
{
    material.evaluate(context, instance);
    hit = material.program.evaluatedHit();
    seed = material.program.evaluatedSeed();
    return material.program.surfaceSummary();
}

public SceneSurfaceSummary evaluateSceneSurfaceGuides(inout HitInfo hit, ScenePathTracePush push,
    RayCone rayCone, float3 rayDirection, inout uint seed)
{
    let transport = loadSceneMaterial(hit.materialIndex);
    if (materialDomain(transport) == kMaterialDomainFiber) {
        hit.normal = faceForwardShadingNormal(Metallic.SceneSurface.safeNormalize(hit.normal, hit.geometryNormal),
            hit.geometryNormal, rayDirection);
        return summarizeFiberGuides(sceneFiberParameters(transport));
    }
    let context = sceneSurfaceMaterialContextForRay(hit, rayCone, rayDirection);
    let instance = sceneMaterialInstanceRef(hit.materialIndex);
#if METALLIC_CUSTOM_MATERIALS
    uint family = sceneSurfaceMaterialFamily(hit.materialIndex);
    if (family == 1u) {
        var material = makeSceneSlabMaterialProgram<SceneSingleSlabModel>(hit, push, rayCone, rayDirection, seed, true, false);
        return evaluateSceneSurfaceSummary(material, context, instance, hit, seed);
    }
    if (family == 2u) {
        var material = makeSceneSlabMaterialProgram<SceneDualSlabModel>(hit, push, rayCone, rayDirection, seed, true, false);
        return evaluateSceneSurfaceSummary(material, context, instance, hit, seed);
    }
#endif
    var material = makeOpenPBRMaterialProgram(hit, push, rayCone, rayDirection, seed, true, false);
    return evaluateSceneSurfaceSummary(material, context, instance, hit, seed);
}

public float3 openPbrPrimaryDebugColor(
    ScenePathTracePush push,
    HitInfo hit,
    float3 shadingNormal,
    float4 normalSample,
    SceneSurfaceSummary summary,
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
        return Metallic.SceneSurface.safeNormalize(hit.tangent.xyz, float3(1.0, 0.0, 0.0)) * 0.5 + 0.5;
    }
    if (push.debugView == kOpenPBRDebugViewBitangent) {
        return Metallic.SceneSurface.safeNormalize(hit.bitangent, float3(0.0, 1.0, 0.0)) * 0.5 + 0.5;
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
        return Metallic.WorkingColor::toLinearRec709(summary.baseColor);
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

public float3 openPbrDisplayDebugColor(ScenePathTracePush push, float3 color)
{
    // BaseColor is already display-linear Rec.709; normal/ID values are display code values.
    if (push.debugView == kOpenPBRDebugViewShadowTransmittance) {
        color = Metallic.WorkingColor::toLinearRec709(color);
    }
    return saturate(color);
}

} // namespace Metallic.SceneOpenPBR
