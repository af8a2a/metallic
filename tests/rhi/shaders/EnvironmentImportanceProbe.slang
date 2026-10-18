module EnvironmentImportanceProbe;

import Core;
import SceneSurface;
import ScenePathMath;
import ScenePathTypes;
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

// Frozen immediate pre-branchless baseline: raw CDF, serial RNG, actual root PDF.
// Do not share child selection or normalization helpers with production here.
uint branchedChild(float4 weights, float randomValue, out float chosenWeight)
{
    if (randomValue < weights.x) { chosenWeight = weights.x; return 0u; }
    else if (randomValue < weights.x + weights.y) { chosenWeight = weights.y; return 1u; }
    else if (randomValue < weights.x + weights.y + weights.z) { chosenWeight = weights.z; return 2u; }
    chosenWeight = weights.w;
    return 3u;
}

bool rawBranchedSample(inout uint seed, out uint2 position, out float probability)
{
    position = uint2(0u);
    probability = 1.0;
    uint width = 0u, height = 0u, mipCount = 0u;
    Texture2D<float> pdfTex = sceneEnvironmentImportancePdf();
    pdfTex.GetDimensions(0u, width, height, mipCount);
    if (width == 0u || height == 0u || mipCount == 0u) {
        probability = 0.0;
        return false;
    }
    float chosenWeight = 1.0;
    for (int mipLevel = int(mipCount) - 2; mipLevel >= 0; --mipLevel) {
        position *= 2u;
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
        uint index = branchedChild(weights, randomValue, chosenWeight);
        position += uint2(index >> 1u, index & 1u);
    }
    if (mipCount > 1u) {
        float topWeight = max(pdfTex.Load(int3(0, 0, mipCount - 1u)), 0.0);
        float totalWeight = ldexp(topWeight, int(2u * (mipCount - 1u)));
        probability = totalWeight > 0.0 ? chosenWeight / totalWeight : 0.0;
    }
    return probability > 0.0;
}

// 12 finite nonnegative quartets x 11 targets: zero, each CDF boundary
// and its adjacent representable values, and nextDown(sum). Zero CDFs
// clamp nextDown to zero because negative random targets are unreachable.
void childBoundaryCase(uint lane, out float4 weights, out float target)
{
    switch ((lane / 11u) % 12u) {
    case 0u: weights = float4(1.0, 2.0, 3.0, 4.0); break;
    case 1u: weights = float4(0.0, 1.0, 0.0, 2.0); break;
    case 2u: weights = float4(0.0, 0.0, 1.0, 0.0); break;
    case 3u: weights = float4(1.0, 0.0, 0.0, 0.0); break;
    case 4u: weights = float4(0.0, 0.0, 0.0, 1.0); break;
    case 5u: weights = float4(1.0, 0.0, 2.0, 0.0); break;
    case 6u: weights = float4(0.0); break;
    case 7u: weights = float4(16777216.0, 1.0, 2.0, 0.0); break;
    case 8u: weights = float4(909.0, 221.0, 255.0, 185.0); break;
    case 9u: weights = float4(0.000244140625, 4096.0, 0.125, 8.0); break;
    case 10u: weights = float4(1e-10, 2e-10, 3e-10, 4e-10); break;
    default: weights = float4(1e20, 2e20, 3e20, 4e20); break;
    }
    float c0 = weights.x, c1 = c0 + weights.y, c2 = c1 + weights.z;
    uint targetCase = lane % 11u;
    if (targetCase == 0u) { target = 0.0; return; }
    float boundary = targetCase <= 3u ? c0 : (targetCase <= 6u ? c1 : c2);
    if (targetCase == 10u) { boundary = c2 + weights.w; }
    uint boundaryBits = asuint(boundary);
    uint neighbor = (targetCase - 1u) % 3u;
    if (targetCase == 10u || neighbor == 0u) {
        target = asfloat(boundaryBits > 0u ? boundaryBits - 1u : 0u);
    } else if (neighbor == 1u) { target = boundary; }
    else { target = asfloat(boundaryBits + 1u); }
}

// Frozen pre-rewrite solid angle, deliberately independent of the new helper.
float referenceSolidAngle(uint y, uint width, uint height)
{
    float centerTheta = (float(y) + 0.5) * (kPi / float(max(height, 1u)));
    return max((2.0 * kPi / float(max(width, 1u))) *
        (kPi / float(max(height, 1u))) * sin(centerTheta), 0.0000001);
}

void solidAngleCase(uint lane, out uint2 dimensions, out uint row, out float probability)
{
    switch ((lane / 25u) % 8u) {
    case 0u: dimensions = uint2(1u, 1u); break;
    case 1u: dimensions = uint2(2u, 1u); break;
    case 2u: dimensions = uint2(7u, 5u); break;
    case 3u: dimensions = uint2(2048u, 1024u); break;
    case 4u: dimensions = uint2(8192u, 4096u); break;
    case 5u: dimensions = uint2(65536u, 65536u); break;
    case 6u: dimensions = uint2(1u, 65536u); break;
    default: dimensions = uint2(65536u, 1u); break;
    }
    uint rowCase = (lane / 5u) % 5u;
    row = rowCase == 0u ? 0u : (rowCase == 1u ? min(1u, dimensions.y - 1u) :
        (rowCase == 2u ? dimensions.y / 2u :
        (rowCase == 3u ? dimensions.y - min(2u, dimensions.y) : dimensions.y - 1u)));
    switch (lane % 5u) {
    case 0u: probability = 0.0; break;
    case 1u: probability = 1.0; break;
    case 2u: probability = 1.0 / (float(dimensions.x) * float(dimensions.y)); break;
    case 3u: probability = 1e-20; break;
    default: probability = 0.1234567; break;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void environmentImportanceProbeMain(uint3 id : SV_DispatchThreadID)
{
    // Keep synchronized with importanceSeed in the CPU harness. These exercise
    // uint wrap and zero hash inputs, including the warp refresh depths 4/8.
    uint seed = id.x * 747796405u + 2891336453u;
    switch (id.x) {
    case 0u: seed = 0u; break;
    case 1u: seed = 0x7f92df0au; break;
    case 2u: seed = 0u - 0xd1b54a35u; break;
    case 3u: seed = 0xffffffffu; break;
    case 4u: seed = 0u - 0x9e3779b9u; break;
    case 5u: seed = 0u - 0x9e3779b9u - 1u; break;
    case 6u: seed = 0u - 0x9e3779b9u - 4u; break;
    case 7u: seed = 0u - 0x9e3779b9u - 8u; break;
    }
    uint initialSeed = seed;
    uint referenceSeed = seed, rawBranchedSeed = seed;
    uint derivedSeed = seed, perLevelSeed = seed;
    uint2 position, referencePosition, rawBranchedPosition;
    uint2 derivedPosition, perLevelPosition;
    float probability, referencePdf, rawBranchedProbability;
    float derivedProbability, perLevelProbability;
    bool valid = sampleEnvironmentImportanceMipmap(seed, position, probability);
    bool rawBranchedValid = rawBranchedSample(rawBranchedSeed, rawBranchedPosition, rawBranchedProbability);
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
    if (id.x == 65535u) { probePosition = queryPosition = uint2(0xffffffffu); }
    let output = resolveBuffer<RWStructuredBuffer<uint4>>(getResourceParameters<SceneResourceParameters>().probeOutput);
    uint base = id.x * 18u;
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
    // Evaluate the production-selected texel independently. The CPU requires exact
    // PDF bits against the sampler for successful multi-mip samples; single-mip
    // zero keeps the sampler's legacy PDF=1 versus evaluator PDF=0 exception.
    output[base + 7u] = uint4(asuint(derivedProbability), asuint(perLevelProbability),
        asuint(loadEnvironmentImportanceWeight(sceneEnvironmentImportancePdf(), probePosition, mipLevel, mipDimensions)), asuint(environmentImportanceTexelProbability(position)));
    float4 boundaryWeights;
    float boundaryTarget, selectedWeight, referenceSelectedWeight;
    childBoundaryCase(id.x, boundaryWeights, boundaryTarget);
    uint selectedIndex = selectEnvironmentImportanceChild(boundaryWeights, boundaryTarget, selectedWeight);
    uint referenceIndex = branchedChild(boundaryWeights, boundaryTarget, referenceSelectedWeight);
    output[base + 8u] = uint4(uint(rawBranchedValid), rawBranchedPosition, rawBranchedSeed);
    output[base + 9u] = uint4(asuint(rawBranchedProbability), selectedIndex, referenceIndex, id.x % 132u);
    output[base + 10u] = uint4(asuint(selectedWeight), asuint(referenceSelectedWeight), asuint(boundaryTarget), 0u);
    output[base + 11u] = asuint(boundaryWeights);

    uint2 solidDimensions;
    uint solidRow;
    float solidProbability;
    solidAngleCase(id.x, solidDimensions, solidRow, solidProbability);
    output[base + 12u] = asuint(float4(
        environmentTexelPdfSolidAngle(solidProbability, solidRow, solidDimensions.x, solidDimensions.y),
        solidProbability / referenceSolidAngle(solidRow, solidDimensions.x, solidDimensions.y),
        solidProbability, referenceSolidAngle(solidRow, solidDimensions.x, solidDimensions.y)));
    output[base + 13u] = uint4(solidDimensions, solidRow, id.x % 200u);

    // Exercise the actual production map sampler, including uniform fallback and
    // two jitter draws. A frozen map conversion uses the same independently
    // checked raw-CDF traversal, retaining its old normalization and PDF formula.
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.environmentMode = kEnvironmentModeMap;
    push.environmentIntensity = 1.0;
    push.environmentRotationRadians = (float(id.x % 257u) - 128.0) * (kPi / 32.0);
    uint lightSeed = initialSeed;
    EnvironmentLightSample lightSample;
    bool lightValid = sampleEnvironmentLight(push, lightSeed, lightSample);
    uint frozenSeed = rawBranchedSeed;
    uint2 frozenTexel = rawBranchedPosition;
    float frozenProbability = rawBranchedProbability;
    if (!rawBranchedValid || any(frozenTexel >= uint2(width, height))) {
        uint count = width * height;
        uint index = min(uint(rand(frozenSeed) * float(count)), count - 1u);
        frozenTexel = uint2(index % width, index / width);
        frozenProbability = 1.0 / float(count);
    }
    float2 uv = (float2(frozenTexel) + float2(rand(frozenSeed), rand(frozenSeed))) / float2(width, height);
    float3 frozenDirection = Metallic.SceneSurface.safeNormalize(
        environmentLocalDirectionToWorld(environmentLocalDirectionFromUv(uv), push.environmentRotationRadians),
        float3(0.0, 1.0, 0.0));
    float frozenPdf = frozenProbability / referenceSolidAngle(frozenTexel.y, width, height);
    float3 frozenRadiance = max(sceneEnvironmentMap().Load(int3(int2(frozenTexel), 0)).rgb, float3(0.0));
    bool frozenValid = max(frozenRadiance.x, max(frozenRadiance.y, frozenRadiance.z)) > 0.0 && frozenPdf > 0.0;
    output[base + 14u] = asuint(float4(lightSample.direction, lightSample.pdfSolidAngle));
    output[base + 15u] = asuint(float4(frozenDirection, frozenPdf));
    output[base + 16u] = uint4(lightSeed, frozenSeed, uint(lightValid), uint(frozenValid));
    output[base + 17u] = asuint(float4(dot(lightSample.direction, lightSample.direction),
        length(lightSample.direction - frozenDirection), lightSample.radiance.x, frozenRadiance.x));
}
