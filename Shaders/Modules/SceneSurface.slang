#language slang 2026
module SceneSurface;

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
#include "../../External/Rtxtf/STFSamplerState.hlsli"

#if METALLIC_CUSTOM_MATERIALS
__include "MaterialValueDispatch.slang";
#endif

namespace Metallic.SceneSurface
{
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.ScenePath;


#if SHARC_UPDATE
public RWTexture2D<float4> sceneOutput()
{
    return resolveUniform(getSceneResourceParameters().output);
}
#else
public RWTexture2D<float4> sceneOutput()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<RWTexture2D<float4>>>(kSceneOutputOffset));
}
#endif

public struct PathTracePrimitive
{
    public uint firstVertex;
    public uint vertexCount;
    public uint firstIndex;
    public uint indexCount;
    public uint flags;
    public uint padding0;
    public uint padding1;
    public uint padding2;
};

public struct PathTraceInstance
{
    public uint primitiveIndex;
    public uint materialIndex;
    public uint flags;
    public float rayConeLodConstant;
};


#if SHARC_UPDATE
public StructuredBuffer<SceneMaterialData> sceneMaterials()
{
    return resolveBuffer<StructuredBuffer<SceneMaterialData>>(getSceneResourceParameters().materials);
}
#else
public StructuredBuffer<SceneMaterialData> sceneMaterials()
{
    return resolveBuffer<StructuredBuffer<SceneMaterialData>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneMaterialsOffset));
}
#endif

#if SHARC_UPDATE
public RWTexture2D<float4> sceneHistoryCurrent()
{
    return resolveUniform(getSceneResourceParameters().accumulationCurrent);
}
#else
public RWTexture2D<float4> sceneHistoryCurrent()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<RWTexture2D<float4>>>(kSceneAccumulationCurrentOffset));
}
#endif

#if SHARC_UPDATE
public RWTexture2D<float4> sceneHistoryPrevious()
{
    return resolveUniform(getSceneResourceParameters().accumulationPrevious);
}
#else
public RWTexture2D<float4> sceneHistoryPrevious()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<RWTexture2D<float4>>>(kSceneAccumulationPreviousOffset));
}
#endif

#if SHARC_UPDATE
public Texture2D<float4> sceneMaterialTexture(uint index)
{
    return resolveNonUniform(ResourceHandle<Texture2D<float4>>(getSceneResourceParameterField<BufferSpan<uint>>(kSceneMaterialTexturesOffset).load(index)));
}
public SamplerState sceneMaterialSampler()
{
    return resolveUniform(getSceneResourceParameters().materialSampler);
}
public RWStructuredBuffer<uint> sceneTextureFeedback()
{
    return resolveBuffer<RWStructuredBuffer<uint>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneTextureFeedbackOffset));
}
public ByteAddressBuffer sceneMaterialValues()
{
    return resolveUniform(getSceneResourceParameters().materialValues);
}
public Texture2D<float4> sceneEnvironmentMap()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float4>>>(kSceneEnvironmentMapOffset));
}
public Texture2D<float> sceneEnvironmentImportancePdf()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float>>>(kSceneEnvironmentPdfOffset));
}
#else
public Texture2D<float4> sceneMaterialTexture(uint index)
{
    return resolveNonUniform(ResourceHandle<Texture2D<float4>>(getSceneResourceParameterField<BufferSpan<uint>>(kSceneMaterialTexturesOffset).load(index)));
}
public SamplerState sceneMaterialSampler()
{
    return resolveUniform(getSceneResourceParameterField<SamplerHandle>(kSceneMaterialSamplerOffset));
}
public RWStructuredBuffer<uint> sceneTextureFeedback()
{
    return resolveBuffer<RWStructuredBuffer<uint>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneTextureFeedbackOffset));
}
public ByteAddressBuffer sceneMaterialValues()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneMaterialValuesOffset));
}
public Texture2D<float4> sceneEnvironmentMap()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float4>>>(kSceneEnvironmentMapOffset));
}
public Texture2D<float> sceneEnvironmentImportancePdf()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float>>>(kSceneEnvironmentPdfOffset));
}
#endif

#if SHARC_UPDATE
public StructuredBuffer<GPUPunctualLight> sceneLights()
{
    return resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getSceneResourceParameters().lights);
}
public Texture2D<float> sceneLightsPdf()
{
    return resolveUniform(getSceneResourceParameters().lightsPdf);
}
public StructuredBuffer<uint4> sceneReGIRGrid()
{
    return resolveBuffer<StructuredBuffer<uint4>>(getSceneResourceParameters().regirGrid);
}
#else
public StructuredBuffer<GPUPunctualLight> sceneLights()
{
    return resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneLightsOffset));
}
public Texture2D<float> sceneLightsPdf()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float>>>(kSceneLightsPdfOffset));
}
public StructuredBuffer<uint4> sceneReGIRGrid()
{
    return resolveBuffer<StructuredBuffer<uint4>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneReGIRGridOffset));
}
#endif

public EnvironmentResourceParameters getSceneEnvironmentResources()
{
#if SHARC_UPDATE
    return getSceneResourceParameters().environment;
#else
    return getSceneResourceParameterField<EnvironmentResourceParameters>(kSceneEnvironmentOffset);
#endif
}


public static const uint kInvalidMaterialTextureIndex = 0xffffffffu;
public static const uint kPrimitiveHasAuthoredTangents = 1u;
public static const uint kEnvironmentModeProcedural = 0u;
public static const uint kEnvironmentModeMap = 1u;
public static const uint kEnvironmentModeDisabled = 2u;


public SceneMaterialData loadSceneMaterial(uint materialIndex)
{
    return sceneMaterials()[materialIndex];
}

public OpenPBRMaterialParameters loadSceneOpenPBRMaterial(uint materialIndex)
{
    return sceneOpenPBRParameters(loadSceneMaterial(materialIndex));
}

public RTXCRHairMaterialParameters loadSceneFiberMaterial(uint materialIndex)
{
    return sceneFiberParameters(loadSceneMaterial(materialIndex));
}

public static const float kPi = 3.14159265358979323846;
public static const float3 kEnvironmentKeyLightDirection = float3(0.35, 0.88, 0.22);
public static const float3 kEnvironmentKeyLightRadiance = float3(5.2, 4.8, 4.2);


// Keep path math unambiguous when programs also import GPUDriven helpers.
public float3 safeNormalize(float3 value, float3 fallback)
{
    return Metallic.ScenePath.safeNormalize(value, fallback);
}


public float3 approxSpecularIntegralGGX(float3 specularReflectance, float alpha, float cosTheta)
{
    cosTheta = abs(cosTheta);

    float4 x = float4(1.0, cosTheta, cosTheta * cosTheta, cosTheta * cosTheta * cosTheta);
    float4 y = float4(1.0, alpha, alpha * alpha, alpha * alpha * alpha);
    float2x2 m1 = float2x2(
        0.995367, -1.38839,
        -0.24751, 1.97442);
    float3x3 m2 = float3x3(
        1.0, 2.68132, 52.366,
        16.0932, -3.98452, 59.3013,
        -5.18731, 255.259, 2544.07);
    float2x2 m3 = float2x2(
        -0.0564526, 3.82901,
        16.91, -11.0303);
    float3x3 m4 = float3x3(
        1.0, 4.11118, -1.37886,
        19.3254, -28.9947, 16.9514,
        0.545386, 96.0994, -79.4492);

    float bias = dot(mul(m1, x.xy), y.xy) / max(dot(mul(m2, x.xyw), y.xyw), 0.000001);
    float scale = dot(mul(m3, x.xy), y.xy) / max(dot(mul(m4, x.xzw), y.xyw), 0.000001);
    bias *= saturate(dot(specularReflectance, float3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)) * 50.0);
    return specularReflectance * max(scale, 0.0) + max(bias, 0.0);
}

public float3 environmentLocalDirectionToWorld(float3 direction, float rotationRadians)
{
    float sine = sin(rotationRadians);
    float cosine = cos(rotationRadians);
    return float3(
        direction.x * cosine + direction.z * sine,
        direction.y,
        -direction.x * sine + direction.z * cosine);
}

public float3 environmentKeyLightDirection(ScenePathTracePush push)
{
    return safeNormalize(
        environmentLocalDirectionToWorld(kEnvironmentKeyLightDirection, push.environmentRotationRadians),
        float3(0.0, 1.0, 0.0));
}

public float3 environmentKeyLightRadiance(ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled) {
        return float3(0.0, 0.0, 0.0);
    }
    return Metallic.WorkingColor::fromLinearRec709(kEnvironmentKeyLightRadiance) * max(push.environmentIntensity, 0.0);
}

public float3 environmentLocalDirectionFromUv(float2 uv)
{
    float phi = (uv.x - 0.5) * (2.0 * kPi);
    float theta = uv.y * kPi;
    float sinTheta = sin(theta);
    return float3(cos(phi) * sinTheta, cos(theta), sin(phi) * sinTheta);
}

public float environmentTexelSolidAngle(uint y, uint width, uint height)
{
    float centerTheta = (float(y) + 0.5) * (kPi / float(max(height, 1u)));
    return max((2.0 * kPi / float(max(width, 1u))) * (kPi / float(max(height, 1u))) * sin(centerTheta), 0.0000001);
}

public float misPowerHeuristic(float pdfA, float pdfB)
{
    float a = max(pdfA, 0.0);
    float b = max(pdfB, 0.0);
    // Smooth transmission can produce finite PDFs whose squares overflow.
    // A common power-of-two multiplier cancels from the heuristic. Keep it
    // normal: dividing by a near-FLT_MAX PDF can flush its reciprocal to zero
    // on the GPU. The largest scaled normal PDF is in [1, 4).
    float maxPdf = max(a, b);
    if (maxPdf == 0.0) { return 1.0; }
    uint exponent = (asuint(maxPdf) >> 23u) & 0xffu;
    float scale = asfloat((254u - min(exponent, 253u)) << 23u);
    a *= scale;
    b *= scale;
    a *= a;
    b *= b;
    return a / (a + b);
}

public struct EnvironmentLightSample
{
    public float3 direction;
    public float3 radiance;
    public float pdfSolidAngle;
};

public float loadEnvironmentImportanceWeight(uint2 position, uint mipLevel)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    sceneEnvironmentImportancePdf().GetDimensions(mipLevel, width, height, mipCount);
    return all(position < uint2(width, height))
        ? max(sceneEnvironmentImportancePdf().Load(int3(position, mipLevel)), 0.0)
        : 0.0;
}

public bool sampleEnvironmentImportanceMipmap(
    inout uint seed,
    out uint2 position,
    out float probability)
{
    position = uint2(0u);
    probability = 1.0;
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    sceneEnvironmentImportancePdf().GetDimensions(0u, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u) {
        probability = 0.0;
        return false;
    }
    for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
        position *= 2u;
        float4 weights = float4(
            loadEnvironmentImportanceWeight(position + uint2(0u, 0u), uint(mipLevel)),
            loadEnvironmentImportanceWeight(position + uint2(0u, 1u), uint(mipLevel)),
            loadEnvironmentImportanceWeight(position + uint2(1u, 0u), uint(mipLevel)),
            loadEnvironmentImportanceWeight(position + uint2(1u, 1u), uint(mipLevel)));
        float weightSum = weights.x + weights.y + weights.z + weights.w;
        if (weightSum <= 0.0) {
            probability = 0.0;
            return false;
        }
        weights /= weightSum;
        float randomValue = rand(seed);
        if (randomValue < weights.x) {
            probability *= weights.x;
        } else if (randomValue < weights.x + weights.y) {
            position += uint2(0u, 1u);
            probability *= weights.y;
        } else if (randomValue < weights.x + weights.y + weights.z) {
            position += uint2(1u, 0u);
            probability *= weights.z;
        } else {
            position += uint2(1u, 1u);
            probability *= weights.w;
        }
    }
    return probability > 0.0;
}

public float environmentImportanceTexelProbability(uint2 position)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    sceneEnvironmentImportancePdf().GetDimensions(0u, width, height, mipCount);
    if (mipCount == 0u || any(position >= uint2(width, height))) {
        return 0.0;
    }
    float averageWeight = loadEnvironmentImportanceWeight(uint2(0u), mipCount - 1u);
    float totalWeight = averageWeight * float(width * height);
    return totalWeight > 0.0
        ? loadEnvironmentImportanceWeight(position, 0u) / totalWeight
        : 0.0;
}

public bool sampleEnvironmentLight(ScenePathTracePush push, inout uint seed, out EnvironmentLightSample sample)
{
    sample.direction = float3(0.0, 1.0, 0.0);
    sample.radiance = float3(0.0, 0.0, 0.0);
    sample.pdfSolidAngle = 1.0;

    if (push.environmentMode == kEnvironmentModeDisabled || push.environmentIntensity <= 0.0) {
        return false;
    }

    if (push.environmentMode != kEnvironmentModeMap) {
        sample.direction = environmentKeyLightDirection(push);
        sample.radiance = environmentKeyLightRadiance(push);
        return max(sample.radiance.x, max(sample.radiance.y, sample.radiance.z)) > 0.0;
    }

    uint width = 1u;
    uint height = 1u;
    sceneEnvironmentMap().GetDimensions(width, height);
    uint textureTexelCount = width * height;
    if (textureTexelCount == 0u) {
        return false;
    }

    uint2 texel = uint2(0u);
    float texelProbability = 0.0;
    if (!sampleEnvironmentImportanceMipmap(seed, texel, texelProbability) ||
        texel.x >= width || texel.y >= height) {
        uint texelIndex = min(uint(rand(seed) * float(textureTexelCount)), textureTexelCount - 1u);
        texel = uint2(texelIndex % width, texelIndex / width);
        texelProbability = 1.0 / float(textureTexelCount);
    }

    float2 uv = (float2(texel) + float2(rand(seed), rand(seed))) / float2(width, height);
    float texelSolidAngle = environmentTexelSolidAngle(texel.y, width, height);

    float3 localDirection = environmentLocalDirectionFromUv(uv);
    sample.direction = safeNormalize(
        environmentLocalDirectionToWorld(localDirection, push.environmentRotationRadians),
        float3(0.0, 1.0, 0.0));
    sample.radiance = max(sceneEnvironmentMap().Load(int3(int2(texel), 0)).rgb, float3(0.0, 0.0, 0.0)) *
        max(push.environmentIntensity, 0.0);
    sample.pdfSolidAngle = texelProbability / texelSolidAngle;
    return max(sample.radiance.x, max(sample.radiance.y, sample.radiance.z)) > 0.0 && sample.pdfSolidAngle > 0.0;
}

public float3 proceduralSkyRadiance(float3 direction, float rotationRadians)
{
    return proceduralEnvironmentLocalRadiance(environmentLocalDirectionToWorld(direction, -rotationRadians));
}

public float2 environmentUvFromDirection(float3 direction, float rotationRadians)
{
    float phi = atan2(direction.z, direction.x) + rotationRadians;
    float u = frac(phi / (2.0 * kPi) + 0.5);
    float v = acos(clamp(direction.y, -1.0, 1.0)) / kPi;
    return float2(u, v);
}

public float environmentPdfSolidAngle(float3 direction, ScenePathTracePush push)
{
    if (push.environmentMode != kEnvironmentModeMap || push.environmentIntensity <= 0.0) {
        return 0.0;
    }

    uint width = 1u;
    uint height = 1u;
    sceneEnvironmentMap().GetDimensions(width, height);
    uint textureTexelCount = width * height;
    if (textureTexelCount == 0u) {
        return 0.0;
    }

    float2 uv = environmentUvFromDirection(direction, push.environmentRotationRadians);
    uint2 texel = min(uint2(uv * float2(width, height)), uint2(width - 1u, height - 1u));
    float texelProbability = environmentImportanceTexelProbability(texel);
    if (texelProbability <= 0.0) {
        texelProbability = 1.0 / float(textureTexelCount);
    }
    return texelProbability / environmentTexelSolidAngle(texel.y, width, height);
}

public float environmentLightMisWeight(ScenePathTracePush push, float environmentPdf, float bsdfPdf)
{
    return push.environmentMode == kEnvironmentModeMap
        ? misPowerHeuristic(environmentPdf, bsdfPdf)
        : 1.0;
}

public float environmentBsdfMisWeight(ScenePathTracePush push, float bsdfPdf, float3 direction, bool sampledDelta)
{
    if (push.environmentMode != kEnvironmentModeMap || sampledDelta) {
        return 1.0;
    }
    return misPowerHeuristic(bsdfPdf, environmentPdfSolidAngle(direction, push));
}

public float3 environmentMapRadiance(float3 direction, ScenePathTracePush push)
{
    uint width = 1u;
    uint height = 1u;
    sceneEnvironmentMap().GetDimensions(width, height);
    if (width == 0u || height == 0u) {
        return float3(0.0, 0.0, 0.0);
    }

    float2 uv = environmentUvFromDirection(direction, push.environmentRotationRadians);
    uint2 texel = min(uint2(uv * float2(width, height)), uint2(width - 1u, height - 1u));
    return max(sceneEnvironmentMap().Load(int3(texel, 0)).rgb, float3(0.0, 0.0, 0.0));
}

public float3 environmentRadiance(float3 direction, ScenePathTracePush push, bool primaryMiss, float3 primaryWorldOrigin)
{
    if (push.environmentMode == kEnvironmentModeDisabled) {
        return float3(0.0, 0.0, 0.0);
    }
    if (primaryMiss && push.environmentVisible == 0u) {
        return float3(0.0, 0.0, 0.0);
    }
    // The proposal/capture excludes celestial disks. Only the primary visible
    // background queries the physical sky with disks, avoiding direct-light MIS duplication.
    let physicalResources = getSceneEnvironmentResources();
    if (primaryMiss && physicalEnvironmentEnabled(physicalResources)) {
        return physicalVisibleEnvironment(physicalResources, direction, primaryWorldOrigin) * max(push.environmentIntensity, 0.0);
    }

    float3 radiance = push.environmentMode == kEnvironmentModeMap
        ? environmentMapRadiance(direction, push)
        : proceduralSkyRadiance(direction, push.environmentRotationRadians);
    return radiance * max(push.environmentIntensity, 0.0);
}

public float3 environmentRadiance(float3 direction, ScenePathTracePush push, bool primaryMiss)
{
    return environmentRadiance(direction, push, primaryMiss, push.eye.xyz);
}

public float3 sampleCosineHemisphere(float3 normal, inout uint seed)
{
    float r0 = rand(seed);
    float r1 = rand(seed);
    float phi = 2.0 * kPi * r0;
    float cosTheta = sqrt(1.0 - r1);
    float sinTheta = sqrt(r1);

    float3 tangent = abs(normal.z) < 0.999
        ? safeNormalize(cross(float3(0.0, 0.0, 1.0), normal), float3(1.0, 0.0, 0.0))
        : safeNormalize(cross(float3(0.0, 1.0, 0.0), normal), float3(1.0, 0.0, 0.0));
    float3 bitangent = cross(normal, tangent);
    return safeNormalize(
        tangent * (cos(phi) * sinTheta) + bitangent * (sin(phi) * sinTheta) + normal * cosTheta,
        normal);
}

public bool hasMaterialTexture(MaterialTextureInfo textureInfo, ScenePathTracePush push)
{
    return textureInfo.textureIndex < push.materialTextureCount;
}


internal TextureFootprint resolveRayConeTextureFootprint(
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

public float rayConeTextureLod(
    MaterialTextureInfo textureInfo,
    RayCone rayCone,
    HitInfo hit,
    float3 rayDirection,
    uint width,
    uint height,
    uint mipCount,
    TextureFootprint footprint = (TextureFootprint)0)
{
    float textureDimensionLod = 0.5 * safeLog2(float(max(width, 1u)) * float(max(height, 1u)));
    footprint = resolveRayConeTextureFootprint(hit, rayCone, rayDirection, footprint);
    float normalizedLod = footprint.isotropicLod0;
    float mipLevel = normalizedLod + textureDimensionLod + textureTransformLodBias(textureInfo);
    return clamp(mipLevel, 0.0, float(max(mipCount, 1u) - 1u));
}

public SurfaceMaterialContext sceneSurfaceMaterialContextForRay(HitInfo hit, RayCone rayCone, float3 rayDirection)
{
    SurfaceMaterialContext context = sceneSurfaceMaterialContext(hit, (TextureFootprint)0);
    float normalTerm = sqrt(max(abs(dot(rayDirection, context.geometryNormalWS)), 0.000001));
    context.footprint.isotropicLod0 = hit.rayConeLodConstant + safeLog2(max(abs(rayCone.width), 0.000001) / normalTerm);
    context.footprint.flags = kTextureFootprintIsotropic0;
    return context;
}

public RayMaterialWork sceneRayMaterialWork(HitInfo hit, RayCone cone, float3 direction, uint seed, uint bounce)
{
    RayMaterialWork work = (RayMaterialWork)0;
    work.surface = sceneSurfaceMaterialContextForRay(hit, cone, direction);
    work.woWS = safeNormalize(-direction, hit.normal);
    work.instance = sceneMaterialInstanceRef(hit.materialIndex);
    work.seed = seed;
    work.bounce = bounce;
    return work;
}

// Public API composition for the fixed material STF policy. No vendor changes.
// Load addressing follows Rtxtf/STFSamplerStateImpl.hlsli:1825-1832;
// NVIDIA's original 2025 copyright/license notice remains in that pinned owner.
// The same public sample-position API executes the original stochastic math.
internal float4 loadMaterialSTFTextureFromSamplePosition(
    inout STF_SamplerState samplerState,
    Texture2D<float4> texture,
    float2 uv,
    float mipLevel)
{
    uint width;
    uint height;
    uint numberOfLevels;
    texture.GetDimensions(0, width, height, numberOfLevels);

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


public float materialTextureSamplingFloor(uint textureIndex)
{
    return streamTextureSamplingFloor(sceneTextureFeedback(), textureIndex);
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

    uint width = 1u;
    uint height = 1u;
    uint textureIndex = textureInfo.textureIndex;
    sceneMaterialTexture(textureIndex).GetDimensions(width, height);
    if (width == 0u || height == 0u) {
        return fallback;
    }

    float2 wrappedUv = frac(transformMaterialUv(textureInfo, uv));
    if (push.textureFeedbackEnabled != 0u) {
        requestStreamTextureMip0(sceneTextureFeedback(), textureIndex);
    }
#if METALLIC_REALTIME_DEFERRED
    // Glass explicitly requests deterministic finest-available sampling in
    // both comparison modes, including the streaming publication floor.
    return decodeMaterialTextureSample(sampleResidentMaterialTexture(sceneMaterialTexture(textureIndex),
        sceneMaterialSampler(), wrappedUv, 0.0, materialTextureSamplingFloor(textureIndex)),
        textureInfo.transform0.w);
#else
    uint mip = uint(ceil(materialTextureSamplingFloor(textureIndex)));
    uint mipCount;
    sceneMaterialTexture(textureIndex).GetDimensions(mip, width, height, mipCount);
    uint2 texel = min(uint2(wrappedUv * float2(width, height)), uint2(width - 1u, height - 1u));
    return decodeMaterialTextureSample(sceneMaterialTexture(textureIndex).Load(int3(texel, mip)), textureInfo.transform0.w);
#endif
}

public float loadMaterialAlphaCoverage(MaterialTextureInfo textureInfo, float2 uv, ScenePathTracePush push)
{
    if (!hasMaterialTexture(textureInfo, push)) { return 1.0; }
    if (push.textureFeedbackEnabled != 0u) {
        requestStreamTextureMip0(sceneTextureFeedback(), textureInfo.textureIndex);
    }
    return sampleAlphaCoverage(sceneMaterialTexture(textureInfo.textureIndex), transformMaterialUv(textureInfo, uv), uint(ceil(materialTextureSamplingFloor(textureInfo.textureIndex))));
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
    if (!hasMaterialTexture(textureInfo, push)) {
        return fallback;
    }

    uint width = 1u;
    uint height = 1u;
    uint mipCount = 1u;
    uint textureIndex = textureInfo.textureIndex;
    sceneMaterialTexture(textureIndex).GetDimensions(0, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u) {
        return fallback;
    }

    float mipLevel = rayConeTextureLod(textureInfo, rayCone, hit, rayDirection, width, height, mipCount, footprint);
    float mipFloor = materialTextureSamplingFloor(textureIndex);
#if METALLIC_REALTIME_DEFERRED
    if ((push.debugFlags & 256u) == 0u) {
        // Preserve the subsequent lighting/feedback random dimensions when
        // comparing filtering modes; only the texture estimator changes.
        rand4(seed);
        return decodeMaterialTextureSample(sampleResidentMaterialTexture(sceneMaterialTexture(textureIndex),
            sceneMaterialSampler(), transformMaterialUv(textureInfo, uv), mipLevel, mipFloor),
            textureInfo.transform0.w);
    }
#endif
    STF_SamplerState samplerState = makeMaterialStfSampler(push, seed);
    return decodeMaterialTextureSample(loadMaterialSTFTextureFromSamplePosition(samplerState,
        sceneMaterialTexture(textureIndex),
        transformMaterialUv(textureInfo, uv),
        max(mipLevel, mipFloor)), textureInfo.transform0.w);
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
    // Demand uses the source dimensions, never the currently coarse resident tail.
    // The same UV transform and ray-cone footprint drive sampling and residency.
    // Deterministic sampling records mip 0 once in loadMaterialTextureMip0.
    // The host/tile gate precedes every demand-buffer access; sampling floors
    // remain active even when demand capture is disabled.
    if (push.textureFeedbackEnabled != 0u && !deterministicMip0 && hasMaterialTexture(textureInfo, push)) {
        RWStructuredBuffer<uint> feedback = sceneTextureFeedback();
        uint count, stride;
        feedback.GetDimensions(count, stride);
        uint offset = textureInfo.textureIndex * 8u;
        if (offset + 7u < count && feedback[offset] != 0u) {
            // Reuse the normalized footprint when sampling the resident texture below.
            footprint = resolveRayConeTextureFootprint(hit, rayCone, rayDirection, footprint);
            float wanted = rayConeTextureLod(textureInfo, rayCone, hit, rayDirection,
                feedback[offset], feedback[offset + 1u], feedback[offset + 2u], footprint);
            recordTextureDemand(feedback, textureInfo.textureIndex, uint(max(floor(wanted), 0.0)));
        }
    }
    if (deterministicMip0) {
        return loadMaterialTextureMip0(textureInfo, uv, fallback, push);
    }
    return loadMaterialTexture(textureInfo, uv, fallback, push, rayCone, hit, rayDirection, seed, footprint);
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
    return sampleMaterialValueTextureResource(sceneMaterialTexture(info.textureIndex), sceneTextureFeedback(),
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


public float3 transformObjectNormalToWorld(float3 normal, float3x4 worldToObject)
{
    return mul(normal, worldToObject).xyz;
}

public float3 faceForwardShadingNormal(float3 normal, float3 geometryNormal, float3 rayDirection)
{
    float3 facingGeometryNormal = dot(geometryNormal, -rayDirection) < 0.0
        ? -geometryNormal
        : geometryNormal;
    if (dot(normal, facingGeometryNormal) < 0.0) {
        normal = -normal;
    }
    return safeNormalize(normal, facingGeometryNormal);
}

public float luminanceMax(float3 value)
{
    return max(value.x, max(value.y, value.z));
}

public float fresnelDielectric(float ior, float cosTheta)
{
    float f0 = (ior - 1.0) / (ior + 1.0);
    f0 *= f0;
    float x = 1.0 - saturate(cosTheta);
    return saturate(f0 + (1.0 - f0) * x * x * x * x * x);
}

public bool refractDirection(float3 incident, float3 normal, float eta, out float3 transmitted)
{
    float cosTheta = saturate(dot(-incident, normal));
    float sin2Theta = max(0.0, 1.0 - cosTheta * cosTheta);
    float sin2Transmitted = eta * eta * sin2Theta;
    if (sin2Transmitted >= 1.0) {
        transmitted = float3(0.0, 0.0, 0.0);
        return false;
    }

    float cosTransmitted = sqrt(max(0.0, 1.0 - sin2Transmitted));
    transmitted = safeNormalize(eta * incident + (eta * cosTheta - cosTransmitted) * normal, incident);
    return true;
}

public float3 powColor(float3 value, float exponent)
{
    float3 clamped = max(value, float3(0.001, 0.001, 0.001));
    return float3(pow(clamped.x, exponent), pow(clamped.y, exponent), pow(clamped.z, exponent));
}

public float3 volumeTransmittance(float3 baseColor, float transmission, float thickness, float attenuationDistance, float3 attenuationColor, float distance)
{
    float exponent = 0.0;
    if (attenuationDistance > 0.0001) {
        exponent = max(distance, 0.0) / attenuationDistance;
    }

    float3 attenuation = powColor(max(attenuationColor, float3(0.001, 0.001, 0.001)), exponent);
    float3 baseTint = lerp(float3(1.0, 1.0, 1.0), powColor(baseColor, 0.5), transmission);
    return saturate(attenuation * baseTint);
}

public float3 srgbToLinear(float3 color)
{
    return float3(
        color.r <= 0.04045 ? color.r / 12.92 : pow((color.r + 0.055) / 1.055, 2.4),
        color.g <= 0.04045 ? color.g / 12.92 : pow((color.g + 0.055) / 1.055, 2.4),
        color.b <= 0.04045 ? color.b / 12.92 : pow((color.b + 0.055) / 1.055, 2.4));
}

public float fallbackBsdfPdf(
    float3 incidentDirection,
    float3 normal,
    float3 sampledDirection,
    float metallic,
    float roughness,
    float diffuseTransmission)
{
    float specularChance = saturate(0.08 + metallic * 0.72 + (1.0 - roughness) * 0.18);
    float transmissionChance = saturate(diffuseTransmission * (1.0 - metallic));
    float frontCos = saturate(dot(normal, sampledDirection));
    float backCos = saturate(dot(-normal, sampledDirection));
    float diffusePdf = frontCos / kPi;
    float transmissionPdf = backCos / kPi;

    float3 reflectedDirection = safeNormalize(reflect(incidentDirection, normal), normal);
    float specularPower = lerp(96.0, 8.0, roughness);
    float specularPdf = ((specularPower + 1.0) / (2.0 * kPi)) *
        pow(saturate(dot(reflectedDirection, sampledDirection)), specularPower);
    return specularChance * specularPdf +
        (1.0 - specularChance) *
            ((1.0 - transmissionChance) * diffusePdf + transmissionChance * transmissionPdf);
}

public float3 applyNormalMap(
    float3 normal,
    float3 geometryNormal,
    float4 tangent,
    float3 bitangent,
    float4 normalSample,
    float normalScale)
{
    normal = safeNormalize(normal, geometryNormal);
    float3 referenceAxis = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    float3 fallbackTangent = safeNormalize(cross(referenceAxis, normal), float3(1.0, 0.0, 0.0));
    float3 tangentDirection = tangent.xyz - normal * dot(normal, tangent.xyz);
    tangentDirection = safeNormalize(tangentDirection, fallbackTangent);
    float tangentSign = tangent.w >= 0.0 ? 1.0 : -1.0;
    float3 bitangentFallback = cross(normal, tangentDirection) * tangentSign;
    float3 bitangentDirection = safeNormalize(bitangent - normal * dot(normal, bitangent), bitangentFallback);

    float3 tangentNormal = float3(normalSample.xy * 2.0 - 1.0, normalSample.z * 2.0 - 1.0);
    tangentNormal.xy *= max(normalScale, 0.0);
    tangentNormal = safeNormalize(tangentNormal, float3(0.0, 0.0, 1.0));
    return safeNormalize(
        tangentDirection * tangentNormal.x +
        bitangentDirection * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}

} // namespace Metallic.SceneSurface
