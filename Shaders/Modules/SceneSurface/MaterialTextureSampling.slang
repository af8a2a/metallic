implementing SceneSurface;

import MaterialTextureDecode;
import AlphaCoverage;
import Core;
import Material;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import MaterialProgram;
import Lighting;
import RTXCRHair;
import ScenePathTypes;
import ScenePathMath;
import RTXCRMaterialAdapter;
import MaterialValueTextureSampling;
import RayMaterialExecution;
import StreamTextureSampling;
import WorkingColor;

#ifndef METALLIC_DEFERRED_STOCHASTIC_FILTERING
#define METALLIC_DEFERRED_STOCHASTIC_FILTERING 0
#endif

#if !METALLIC_REALTIME_DEFERRED || METALLIC_DEFERRED_STOCHASTIC_FILTERING
#ifndef STF_SHADER_STAGE
#define STF_SHADER_STAGE 6
#endif
#ifndef STF_SHADER_MODEL_MAJOR
#define STF_SHADER_MODEL_MAJOR 6
#endif
#ifndef STF_SHADER_MODEL_MINOR
#define STF_SHADER_MODEL_MINOR 6
#endif
#ifndef STF_ALLOW_WAVE_READ
#define STF_ALLOW_WAVE_READ 0
#endif
#include "../../../External/Rtxtf/STFSamplerState.hlsli"
#endif

namespace Metallic.SceneSurface
{
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.ScenePath;

public bool hasMaterialTexture(MaterialTextureInfo textureInfo, ScenePathTracePush push)
{
    return textureInfo.textureIndex < push.materialTextureCount;
}

// ScenePathTraceResources reserves descriptor 0 for its white fallback and
// maps authored textures to indices >= 1. This is a scene-provider contract,
// not a rule for raw sampling APIs or custom texture sources.
public bool sceneHasMaterialTextures(ScenePathTracePush push)
{
    return push.materialTextureCount > 1u;
}

public TextureFootprint resolveRayConeTextureFootprint(
    HitInfo hit,
    RayCone rayCone,
    float3 rayDirection,
    TextureFootprint footprint)
{
    if ((footprint.flags & kTextureFootprintIsotropic0) == 0u) {
        SurfaceMaterialContext context = sceneSurfaceMaterialContext(hit, (TextureFootprint)0);
        float normalTerm = sqrt(max(abs(dot(rayDirection, context.geometryNormalWS)), 0.000001));
        footprint.isotropicLod0 = hit.rayConeLodConstant + safeLog2(max(abs(rayCone.width), 0.000001) / normalTerm);
        footprint.flags |= kTextureFootprintIsotropic0;
    }
    return footprint;
}

public float textureDimensionLod(float term)
{
    // NPOT CPU/GPU log2 rounding can straddle mip boundaries. Preserve the
    // original GPU arithmetic for those dimensions without querying the image.
    return term >= 0.0 ? term : 0.5 * safeLog2(-term);
}

public float materialSourceTextureLod(MaterialTextureInfo textureInfo, float normalizedLod, float uvBias)
{
    return clamp((normalizedLod + textureDimensionLod(textureInfo.sourceLodTerm)) + uvBias,
        0.0, float(max(textureInfo.sourceMipCount, 1u) - 1u));
}

public float materialResidentTextureLod(TextureSamplingMetadata metadata, float normalizedLod, float uvBias)
{
    // Actual tail dimensions preserve NPOT rounding and axes saturated at one.
    // Subtracting firstMip from source LOD would change those footprints.
    return clamp((normalizedLod + textureDimensionLod(metadata.residentLodTerm)) + uvBias,
        0.0, float(max(metadata.residentMipCount, 1u) - 1u));
}

public SurfaceMaterialContext sceneSurfaceMaterialContextForRay(HitInfo hit, RayCone rayCone, float3 rayDirection,
    bool resolveFootprint = true)
{
    SurfaceMaterialContext context = sceneSurfaceMaterialContext(hit, (TextureFootprint)0);
    if (resolveFootprint) {
        context.footprint = resolveRayConeTextureFootprint(hit, rayCone, rayDirection, context.footprint);
    }
    return context;
}

public RayMaterialWork sceneRayMaterialWork(HitInfo hit, RayCone cone, float3 direction, uint seed, uint bounce,
    bool resolveFootprint = true)
{
    RayMaterialWork work = (RayMaterialWork)0;
    // Scene surface programs may defer footprint preparation until material evaluation.
    work.surface = sceneSurfaceMaterialContextForRay(hit, cone, direction, resolveFootprint);
    work.woWS = safeNormalize(-direction, hit.normal);
    work.instance = sceneMaterialInstanceRef(hit.materialIndex);
    work.seed = seed;
    work.bounce = bounce;
    return work;
}

#if !METALLIC_REALTIME_DEFERRED || METALLIC_DEFERRED_STOCHASTIC_FILTERING
// Public API composition for the fixed material STF policy. No vendor changes.
// Load addressing follows Rtxtf/STFSamplerStateImpl.hlsli:1825-1832;
// NVIDIA's original 2025 copyright/license notice remains in that pinned owner.
// The same public sample-position API executes the original stochastic math.
internal float4 loadMaterialSTFTextureFromSamplePosition(
    inout STF_SamplerState samplerState,
    Texture2D<float4> texture,
    float2 uv,
    float mipLevel,
    uint width,
    uint height,
    uint numberOfLevels)
{
    float3 samplePos = samplerState.Texture2DGetSamplePosLevel(width, height, numberOfLevels, uv, mipLevel);
    uint lod = uint(samplePos.z);
    width = width >> lod;
    height = height >> lod;

    bool isBorder = false;
    const float2 pixelIndex = uint2(width, height) * STF_ApplyAddressingMode2D(
        samplePos.xy, uint2(width, height), samplerState.GetAddressingModes().xy, isBorder);
    return texture.Load(uint3((uint)pixelIndex.x, (uint)pixelIndex.y, lod));
}

internal STF_SamplerState makeMaterialStfSampler(ScenePathTracePush push, inout uint seed)
{
    STF_SamplerState samplerState = STF_SamplerState::Create(rand4(seed));
    samplerState.SetFrameIndex(push.accumulationFrame);
    samplerState.SetFilterType(STF_FILTER_TYPE_LINEAR);
    samplerState.SetMagMethod(STF_MAGNIFICATION_METHOD_NONE);
    samplerState.SetAnisoMethod(STF_ANISO_LOD_METHOD_NONE);
    samplerState.SetAddressingModes(uint3(STF_ADDRESS_MODE_WRAP, STF_ADDRESS_MODE_WRAP, STF_ADDRESS_MODE_WRAP));
    samplerState.SetReseedOnSample(false);
    return samplerState;
}
#endif


public float materialTextureSamplingFloor(uint textureIndex)
{
    return streamTextureSamplingFloor(sceneTextureSamplingMetadata(), textureIndex);
}

public float4 loadMaterialTextureMip0(
    MaterialTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push)
{
    if (!hasMaterialTexture(textureInfo, push)) {
        return fallback;
    }

    uint textureIndex = textureInfo.textureIndex;
    TextureSamplingMetadata metadata = sceneTextureSamplingMetadata()[textureIndex];

    float2 wrappedUv = frac(transformMaterialUv(textureInfo, uv));
    if (push.textureFeedbackEnabled != 0u && streamTextureDemandEnabled(metadata)) {
        recordTextureDemand(sceneTextureDemandMip(), sceneTextureDemandHits(), textureIndex, 0u);
    }
#if METALLIC_REALTIME_DEFERRED
    // Glass explicitly requests deterministic finest-available sampling in
    // both comparison modes, including the streaming publication floor.
    return decodeMaterialTextureSample(sampleResidentMaterialTexture(sceneMaterialTexture(textureIndex),
        sceneMaterialSampler(), wrappedUv, 0.0, max(metadata.samplingFloor, 0.0)),
        textureInfo.transform0.w);
#else
    uint mip = uint(ceil(max(metadata.samplingFloor, 0.0)));
    uint width = max(metadata.residentWidth >> mip, 1u);
    uint height = max(metadata.residentHeight >> mip, 1u);
    uint2 texel = min(uint2(wrappedUv * float2(width, height)), uint2(width - 1u, height - 1u));
    return decodeMaterialTextureSample(sceneMaterialTexture(textureIndex).Load(int3(texel, mip)), textureInfo.transform0.w);
#endif
}

public float loadMaterialAlphaCoverage(MaterialTextureInfo textureInfo, float2 uv, ScenePathTracePush push)
{
    if (!hasMaterialTexture(textureInfo, push)) { return 1.0; }
    if (push.textureFeedbackEnabled != 0u) {
        requestStreamTextureMip0(sceneTextureSamplingMetadata(), sceneTextureDemandMip(), sceneTextureDemandHits(), textureInfo.textureIndex);
    }
    return sampleAlphaCoverage(sceneMaterialTexture(textureInfo.textureIndex), transformMaterialUv(textureInfo, uv), uint(ceil(materialTextureSamplingFloor(textureInfo.textureIndex))));
}

internal float4 loadResidentMaterialTexture(
    MaterialTextureInfo textureInfo,
    float2 uv,
    ScenePathTracePush push,
    TextureSamplingMetadata metadata,
    float mipLevel,
    inout uint seed,
    uint textureIndex)
{
    float mipFloor = max(metadata.samplingFloor, 0.0);
#if METALLIC_REALTIME_DEFERRED
#if !METALLIC_DEFERRED_STOCHASTIC_FILTERING
    rand4(seed);
    return decodeMaterialTextureSample(sampleResidentMaterialTexture(sceneMaterialTexture(textureIndex),
        sceneMaterialSampler(), transformMaterialUv(textureInfo, uv), mipLevel, mipFloor),
        textureInfo.transform0.w);
#else
    if ((push.debugFlags & 256u) == 0u) {
        // Preserve the subsequent lighting/feedback random dimensions when
        // comparing filtering modes; only the texture estimator changes.
        rand4(seed);
        return decodeMaterialTextureSample(sampleResidentMaterialTexture(sceneMaterialTexture(textureIndex),
            sceneMaterialSampler(), transformMaterialUv(textureInfo, uv), mipLevel, mipFloor),
            textureInfo.transform0.w);
    }
    STF_SamplerState samplerState = makeMaterialStfSampler(push, seed);
    return decodeMaterialTextureSample(loadMaterialSTFTextureFromSamplePosition(samplerState,
        sceneMaterialTexture(textureIndex),
        transformMaterialUv(textureInfo, uv),
        max(mipLevel, mipFloor), metadata.residentWidth, metadata.residentHeight,
        metadata.residentMipCount), textureInfo.transform0.w);
#endif
#else
    STF_SamplerState samplerState = makeMaterialStfSampler(push, seed);
    return decodeMaterialTextureSample(loadMaterialSTFTextureFromSamplePosition(samplerState,
        sceneMaterialTexture(textureIndex),
        transformMaterialUv(textureInfo, uv),
        max(mipLevel, mipFloor), metadata.residentWidth, metadata.residentHeight,
        metadata.residentMipCount), textureInfo.transform0.w);
#endif
}

public float4 loadMaterialTexture(
    MaterialTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push,
    RayCone rayCone,
    HitInfo hit,
    float3 rayDirection,
    inout uint seed,
    TextureFootprint footprint = (TextureFootprint)0)
{
    if (!hasMaterialTexture(textureInfo, push)) { return fallback; }
    TextureSamplingMetadata metadata = sceneTextureSamplingMetadata()[textureInfo.textureIndex];
    footprint = resolveRayConeTextureFootprint(hit, rayCone, rayDirection, footprint);
    float mipLevel = materialResidentTextureLod(metadata, footprint.isotropicLod0, textureTransformLodBias(textureInfo));
    return loadResidentMaterialTexture(textureInfo, uv, push, metadata, mipLevel, seed, textureInfo.textureIndex);
}

// Batch callers resolve the ray footprint once before sampling their slots.
// deterministicMip0 ignores the footprint; otherwise isotropicLod0 must be resolved.
public float4 loadMaterialTextureForMaterialResolved(
    MaterialTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push,
    bool deterministicMip0,
    inout uint seed,
    TextureFootprint footprint,
    out bool sampled)
{
    sampled = hasMaterialTexture(textureInfo, push);
    if (!sampled) { return fallback; }
    if (deterministicMip0) {
        return loadMaterialTextureMip0(textureInfo, uv, fallback, push);
    }
    TextureSamplingMetadata metadata = sceneTextureSamplingMetadata()[textureInfo.textureIndex];
    float uvBias = textureTransformLodBias(textureInfo);
    // The host publishes metadata for every material slot with its image generation.
    // Demand and sampling reuse the per-hit footprint and per-slot UV transform.
    if (push.textureFeedbackEnabled != 0u && (metadata.residentFirstMipAndFlags & 0x80000000u) != 0u) {
        float wanted = materialSourceTextureLod(textureInfo, footprint.isotropicLod0, uvBias);
        recordTextureDemand(sceneTextureDemandMip(), sceneTextureDemandHits(),
            textureInfo.textureIndex, uint(floor(wanted)));
    }
    float mipLevel = materialResidentTextureLod(metadata, footprint.isotropicLod0, uvBias);
    return loadResidentMaterialTexture(textureInfo, uv, push, metadata, mipLevel, seed, textureInfo.textureIndex);
}

public float4 loadMaterialTextureForMaterialResolved(
    MaterialTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push,
    bool deterministicMip0,
    inout uint seed,
    TextureFootprint footprint)
{
    bool sampled;
    return loadMaterialTextureForMaterialResolved(textureInfo, uv, fallback, push,
        deterministicMip0, seed, footprint, sampled);
}

// Compatibility entry for standalone callers with an unresolved footprint.
public float4 loadMaterialTextureForMaterial(
    MaterialTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push,
    RayCone rayCone,
    HitInfo hit,
    float3 rayDirection,
    bool deterministicMip0,
    inout uint seed,
    TextureFootprint footprint,
    out bool sampled)
{
    if (!deterministicMip0 && hasMaterialTexture(textureInfo, push)) {
        footprint = resolveRayConeTextureFootprint(hit, rayCone, rayDirection, footprint);
    }
    return loadMaterialTextureForMaterialResolved(textureInfo, uv, fallback, push,
        deterministicMip0, seed, footprint, sampled);
}

public float4 loadMaterialTextureForMaterial(
    MaterialTextureInfo textureInfo,
    float2 uv,
    float4 fallback,
    ScenePathTracePush push,
    RayCone rayCone,
    HitInfo hit,
    float3 rayDirection,
    bool deterministicMip0,
    inout uint seed,
    TextureFootprint footprint = (TextureFootprint)0)
{
    bool sampled;
    return loadMaterialTextureForMaterial(textureInfo, uv, fallback, push, rayCone, hit,
        rayDirection, deterministicMip0, seed, footprint, sampled);
}

// Generated Value IR retains its callback signature; resource-independent
// filtering lives in the module. Reject missing textures before resolving a handle.
#if METALLIC_CUSTOM_MATERIALS
internal float4 sampleMaterialValueTexture(MaterialValueParameters material, uint slot, float2 uv,
    uint policy, float4 dxLod, float4 dy, MaterialValueTextureContext context, bool linearColor)
{
    MaterialTextureInfo info = material.baseColorTexture;
    if (slot == 1u) { info = material.metallicRoughnessTexture; }
    else if (slot == 2u) { info = material.normalTexture; }
    else if (slot == 3u) { info = material.occlusionTexture; }
    else if (slot == 4u) { info = material.emissiveTexture; }
    else if (slot == 5u) { info = material.specularTexture; }
    float4 fallback = linearColor && slot == 2u ? float4(0.5, 0.5, 1.0, 1.0) : float4(1.0);
    if (info.textureIndex >= context.textureCount) { return fallback; }
    return sampleMaterialValueTextureResource(sceneMaterialTexture(info.textureIndex), sceneTextureSamplingMetadata(),
        info, uv, policy, dxLod, dy, context.normalizedRayLod, linearColor);
}
#endif

public bool materialNeedsDeterministicTextureMip0(OpenPBRMaterialParameters material)
{
    // Specular transmission magnifies stochastic/material-unaware mips into hard surface bands.
    return material.glassParams.x > 0.01;
}


public CoverageResult evaluateSceneCoverage(uint materialIndex, MaterialCoverageParameters material, float2 uv, ScenePathTracePush push)
{
    CoverageContext context;
    context.uv = uv; context.baseAlpha = 1.0;
    context.cutoff = material.cutoff; context.alphaMode = material.alphaMode;
    if (context.alphaMode < 0.5) { return coverageFromOpacity(context, 1.0); }
    if (material.hasProgram) {
        CoverageProgram program;
        program.inputs = sceneMaterialValues();
        MaterialInstanceRef instance = coverageInstanceRef(materialIndex);
        if (program.usesBaseAlpha(instance)) {
            context.baseAlpha = material.baseAlpha * loadMaterialAlphaCoverage(material.baseColorTexture, uv, push);
        }
        return program.evaluateCoverage(context, instance);
    }
    context.baseAlpha = material.baseAlpha * loadMaterialAlphaCoverage(material.baseColorTexture, uv, push);
    return coverageFromOpacity(context, context.baseAlpha);
}

} // namespace Metallic.SceneSurface
