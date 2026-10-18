module EnvironmentImportanceProbe;

import Core;
import SceneSurface;
import ScenePathMath;
using Metallic;
using Metallic.SceneSurface;
using Metallic.ScenePath;

// Frozen pre-optimization implementation: queries dimensions for every load.
float referenceWeight(uint2 position, uint mipLevel)
{
    uint width = 0u, height = 0u, mipCount = 0u;
    sceneEnvironmentImportancePdf().GetDimensions(mipLevel, width, height, mipCount);
    return all(position < uint2(width, height))
        ? max(sceneEnvironmentImportancePdf().Load(int3(position, mipLevel)), 0.0)
        : 0.0;
}

bool referenceSample(inout uint seed, out uint2 position, out float probability)
{
    position = uint2(0u);
    probability = 1.0;
    uint width = 0u, height = 0u, mipCount = 0u;
    sceneEnvironmentImportancePdf().GetDimensions(0u, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u) {
        probability = 0.0;
        return false;
    }
    for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
        position *= 2u;
        float4 weights = float4(
            referenceWeight(position + uint2(0u, 0u), uint(mipLevel)),
            referenceWeight(position + uint2(0u, 1u), uint(mipLevel)),
            referenceWeight(position + uint2(1u, 0u), uint(mipLevel)),
            referenceWeight(position + uint2(1u, 1u), uint(mipLevel)));
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

float referenceProbability(uint2 position)
{
    uint width = 0u, height = 0u, mipCount = 0u;
    sceneEnvironmentImportancePdf().GetDimensions(0u, width, height, mipCount);
    if (mipCount == 0u || any(position >= uint2(width, height))) { return 0.0; }
    float averageWeight = referenceWeight(uint2(0u), mipCount - 1u);
    float totalWeight = averageWeight * float(width * height);
    return totalWeight > 0.0 ? referenceWeight(position, 0u) / totalWeight : 0.0;
}

// Keep both candidates covered regardless of which implementation production selects.
bool cachedSample(bool queryPerLevel, inout uint seed, out uint2 position, out float probability)
{
    position = uint2(0u);
    probability = 1.0;
    Texture2D<float> pdfTex = sceneEnvironmentImportancePdf();
    uint width = 0u, height = 0u, mipCount = 0u;
    pdfTex.GetDimensions(0u, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u) {
        probability = 0.0;
        return false;
    }
    for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
        position *= 2u;
        uint2 mipDimensions;
        if (queryPerLevel) {
            uint mipWidth = 0u, mipHeight = 0u, queriedMipCount = 0u;
            pdfTex.GetDimensions(uint(mipLevel), mipWidth, mipHeight, queriedMipCount);
            mipDimensions = uint2(mipWidth, mipHeight);
        } else {
            mipDimensions = max(uint2(width, height) >> uint(mipLevel), uint2(1u));
        }
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

[shader("compute")]
[numthreads(1, 1, 1)]
void environmentImportanceProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint seed = id.x == 0u ? 0u : id.x * 747796405u + 2891336453u;
    uint referenceSeed = seed;
    uint derivedSeed = seed, perLevelSeed = seed;
    uint2 position, referencePosition;
    uint2 derivedPosition, perLevelPosition;
    float probability, referencePdf;
    float derivedProbability, perLevelProbability;
    bool valid = sampleEnvironmentImportanceMipmap(seed, position, probability);
    bool referenceValid = referenceSample(referenceSeed, referencePosition, referencePdf);
    bool derivedValid = cachedSample(false, derivedSeed, derivedPosition, derivedProbability);
    bool perLevelValid = cachedSample(true, perLevelSeed, perLevelPosition, perLevelProbability);
    uint width = 0u, height = 0u, mipCount = 0u;
    sceneEnvironmentImportancePdf().GetDimensions(0u, width, height, mipCount);
    uint mipLevel = id.x % mipCount;
    uint2 mipDimensions = max(uint2(width, height) >> mipLevel, uint2(1u));
    uint2 probePosition = uint2((id.x / mipCount) % (mipDimensions.x + 1u),
        (id.x / (mipCount * (mipDimensions.x + 1u))) % (mipDimensions.y + 1u));
    uint2 queryPosition = uint2(id.x % (width + 1u), (id.x / (width + 1u)) % (height + 1u));
    if (id.x == 255u) { probePosition = queryPosition = uint2(0xffffffffu); }
    let output = resolveBuffer<RWStructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().probeOutput);
    uint base = id.x * 8u;
    output[base] = uint4(uint(valid), position, seed);
    output[base + 1u] = uint4(uint(referenceValid), referencePosition, referenceSeed);
    output[base + 2u] = asuint(float4(probability, referencePdf,
        environmentImportanceTexelProbability(queryPosition), referenceProbability(queryPosition)));
    output[base + 3u] = uint4(asuint(loadEnvironmentImportanceWeight(probePosition, mipLevel, mipDimensions)),
        asuint(loadEnvironmentImportanceWeight(probePosition, mipLevel)),
        asuint(referenceWeight(probePosition, mipLevel)), mipLevel);
    output[base + 4u] = uint4(probePosition, queryPosition);
    output[base + 5u] = uint4(uint(derivedValid), derivedPosition, derivedSeed);
    output[base + 6u] = uint4(uint(perLevelValid), perLevelPosition, perLevelSeed);
    output[base + 7u] = uint4(asuint(derivedProbability), asuint(perLevelProbability),
        asuint(loadEnvironmentImportanceWeight(sceneEnvironmentImportancePdf(), probePosition, mipLevel, mipDimensions)), 0u);
}
