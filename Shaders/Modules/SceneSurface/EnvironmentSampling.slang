implementing SceneSurface;

import MaterialTextureDecode;
import AlphaCoverage;
import Core;
import Material;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import MaterialProgram;
import Lighting;
import Atmosphere;
import RTXCRHair;
import ScenePathTypes;
import ScenePathMath;
import RTXCRMaterialAdapter;
import MaterialValueTextureSampling;
import RayMaterialExecution;
import StreamTextureSampling;
import WorkingColor;

namespace Metallic.SceneSurface
{
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.ScenePath;

public static const float kPi = 3.14159265358979323846;
public static const float3 kEnvironmentKeyLightDirection = float3(0.35, 0.88, 0.22);
public static const float3 kEnvironmentKeyLightRadiance = float3(5.2, 4.8, 4.2);
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

public float environmentTexelPdfSolidAngle(float probability, uint y, uint width, uint height)
{
    float texelCount = float(max(width, 1u)) * float(max(height, 1u));
    float centerTheta = (float(y) + 0.5) * (kPi / float(max(height, 1u)));
    // Move the texel-area division into the numerator, retaining the original
    // solid-angle floor at high resolutions and near the poles. Sampling and
    // MIS evaluation must use the same conversion.
    return (probability * texelCount) /
        max((2.0 * kPi * kPi) * sin(centerTheta), 0.0000001 * texelCount);
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

public float loadEnvironmentImportanceWeight(
    Texture2D<float> pdfTex, uint2 position, uint mipLevel, uint2 mipDimensions)
{
    return all(position < mipDimensions)
        ? max(pdfTex.Load(int3(position, mipLevel)), 0.0)
        : 0.0;
}

public float loadEnvironmentImportanceWeight(uint2 position, uint mipLevel, uint2 mipDimensions)
{
    return loadEnvironmentImportanceWeight(sceneEnvironmentImportancePdf(), position, mipLevel, mipDimensions);
}

public float loadEnvironmentImportanceWeight(uint2 position, uint mipLevel)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    sceneEnvironmentImportancePdf().GetDimensions(mipLevel, width, height, mipCount);
    return loadEnvironmentImportanceWeight(position, mipLevel, uint2(width, height));
}

// Requires a nonempty mip chain. Share the exact root/leaf normalization
// between sampling and MIS evaluation, including rectangular mip chains.
public float environmentImportanceProbabilityFromWeight(
    Texture2D<float> pdfTex, float weight, uint mipCount)
{
    float topWeight = max(pdfTex.Load(int3(0, 0, mipCount - 1u)), 0.0);
    // PrepareLightsPdf multiplies by 1/4 even for 1xN/Nx1 levels.
    float totalWeight = ldexp(topWeight, int(2u * (mipCount - 1u)));
    return totalWeight > 0.0 ? weight / totalWeight : 0.0;
}

// Child order is (0,0), (0,1), (1,0), (1,1). Preserve strict-CDF tie
// handling, including repeated thresholds for empty children.
public uint selectEnvironmentImportanceChild(float4 weights, float randomValue, out float chosenWeight)
{
    float c0 = weights.x;
    float c1 = c0 + weights.y;
    float c2 = c1 + weights.z;
    uint index = uint(randomValue >= c0) + uint(randomValue >= c1) + uint(randomValue >= c2);
    bool odd = (index & 1u) != 0u;
    chosenWeight = select(index >= 2u,
        select(odd, weights.w, weights.z), select(odd, weights.y, weights.x));
    return index;
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
    // Reuse the resolved resource for every level and child of this traversal.
    Texture2D<float> pdfTex = sceneEnvironmentImportancePdf();
    pdfTex.GetDimensions(0u, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u) {
        probability = 0.0;
        return false;
    }
    float chosenWeight = 1.0;
    for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
        position *= 2u;
        // All four children share this mip's extent. Derive it from mip 0
        // instead of querying the texture inside each dependent weight load.
        // Keep zero weight for missing children (not edge clamping): thin and
        // non-power-of-two mips must preserve the original sampling distribution.
        uint2 mipDimensions = max(uint2(width, height) >> uint(mipLevel), uint2(1u));
        float4 weights = float4(
            loadEnvironmentImportanceWeight(pdfTex, position + uint2(0u, 0u), uint(mipLevel), mipDimensions),
            loadEnvironmentImportanceWeight(pdfTex, position + uint2(0u, 1u), uint(mipLevel), mipDimensions),
            loadEnvironmentImportanceWeight(pdfTex, position + uint2(1u, 0u), uint(mipLevel), mipDimensions),
            loadEnvironmentImportanceWeight(pdfTex, position + uint2(1u, 1u), uint(mipLevel), mipDimensions));
        float weightSum = weights.x + weights.y + weights.z + weights.w;
        if (weightSum <= 0.0) {
            probability = 0.0;
            return false;
        }
        float randomValue = rand(seed) * weightSum;
        uint index = selectEnvironmentImportanceChild(weights, randomValue, chosenWeight);
        position += uint2(index >> 1u, index & 1u);
    }
    // Preserve the single-mip path's probability=1 and zero RNG consumption.
    if (mipCount > 1u) {
        probability = environmentImportanceProbabilityFromWeight(pdfTex, chosenWeight, mipCount);
    }
    return probability > 0.0;
}

public float environmentImportanceTexelProbability(uint2 position)
{
    uint width = 0u;
    uint height = 0u;
    uint mipCount = 0u;
    Texture2D<float> pdfTex = sceneEnvironmentImportancePdf();
    pdfTex.GetDimensions(0u, width, height, mipCount);
    if (mipCount == 0u || any(position >= uint2(width, height))) {
        return 0.0;
    }
    // Traversal starts at top-mip texel (0, 0). Partial/NPOT chains can
    // represent a smaller subtree than the complete base-level extent.
    if (any((position >> (mipCount - 1u)) != uint2(0u))) {
        return 0.0;
    }
    float weight = loadEnvironmentImportanceWeight(pdfTex, position, 0u, uint2(width, height));
    return environmentImportanceProbabilityFromWeight(pdfTex, weight, mipCount);
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
    float3 localDirection = environmentLocalDirectionFromUv(uv);
    // The spherical parameterization and Y rotation preserve unit length.
    sample.direction = environmentLocalDirectionToWorld(localDirection, push.environmentRotationRadians);
    sample.radiance = max(sceneEnvironmentMap().Load(int3(int2(texel), 0)).rgb, float3(0.0, 0.0, 0.0)) *
        max(push.environmentIntensity, 0.0);
    sample.pdfSolidAngle = environmentTexelPdfSolidAngle(texelProbability, texel.y, width, height);
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
    return environmentTexelPdfSolidAngle(texelProbability, texel.y, width, height);
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

public float3 environmentIndirectRadiance(float3 direction, ScenePathTracePush push)
{
    if (push.environmentMode == kEnvironmentModeDisabled) {
        return float3(0.0, 0.0, 0.0);
    }
    float3 radiance = push.environmentMode == kEnvironmentModeMap
        ? environmentMapRadiance(direction, push)
        : proceduralSkyRadiance(direction, push.environmentRotationRadians);
    return radiance * max(push.environmentIntensity, 0.0);
}

public float3 environmentRadiance<TShadow : Metallic.Atmosphere::IAtmosphereLightShadow>(
    float3 direction, ScenePathTracePush push, bool primaryMiss, float3 primaryWorldOrigin, TShadow shadow)
{
    if (push.environmentMode == kEnvironmentModeDisabled) {
        return float3(0.0, 0.0, 0.0);
    }
    if (primaryMiss) {
        if (push.environmentVisible == 0u) {
            return float3(0.0, 0.0, 0.0);
        }
        // The proposal/capture excludes celestial disks. Only the primary visible
        // background queries the physical sky with disks, avoiding direct-light MIS duplication.
        let physicalResources = getSceneEnvironmentResources();
        if (physicalEnvironmentEnabled(physicalResources)) {
            return physicalVisibleEnvironment(physicalResources, direction, primaryWorldOrigin, shadow) * max(push.environmentIntensity, 0.0);
        }
    }

    return environmentIndirectRadiance(direction, push);
}

public float3 environmentRadiance(float3 direction, ScenePathTracePush push, bool primaryMiss, float3 primaryWorldOrigin)
{
    return environmentRadiance(direction, push, primaryMiss, primaryWorldOrigin, Metallic.Atmosphere::UnshadowedAtmosphere());
}

public float3 environmentRadiance(float3 direction, ScenePathTracePush push, bool primaryMiss)
{
    return environmentRadiance(direction, push, primaryMiss, push.eye.xyz);
}

} // namespace Metallic.SceneSurface
