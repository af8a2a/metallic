module LumenSurfaceCache;

#include "../Interop/LumenCardUpdate.h"

namespace Metallic.Lumen
{
public struct SurfaceCacheCard
{
    public float4 origin;
    public float4 axisX;
    public float4 axisY;
    public float4 axisZ;
    public uint4 metadata;
};

public float3 decodeSurfaceCacheNormal(SurfaceCacheCard card, float2 encodedNormal)
{
    float3 cardSpaceNormal;
    cardSpaceNormal.xy = encodedNormal * 2.0f - 1.0f;
    cardSpaceNormal.z = sqrt(max(1.0f - dot(cardSpaceNormal.xy, cardSpaceNormal.xy), 0.0001f));
    return normalize(card.axisX.xyz * cardSpaceNormal.x + card.axisY.xyz * cardSpaceNormal.y +
        card.axisZ.xyz * cardSpaceNormal.z);
}

// UE 5.7.4 LumenCardCommon.ush GetCardLocalPosition/GetCardWorldPosition,
// using the tile texel center to address the resident card.
public float3 surfaceCacheWorldPosition(SurfaceCacheCard card, float2 cardUV, float depth)
{
    cardUV.x = 1.0f - cardUV.x;
    float3 localPosition;
    localPosition.xy = float2(card.axisX.w, card.axisY.w) * (1.0f - 2.0f * cardUV);
    localPosition.z = -(2.0f * depth - 1.0f) * card.axisZ.w;
    return card.axisX.xyz * localPosition.x + card.axisY.xyz * localPosition.y +
        card.axisZ.xyz * localPosition.z + card.origin.xyz;
}

public struct LumenSurfaceCache
{
    public StructuredBuffer<SurfaceCacheCard> cards;
    public Texture2D<float> depth;
    public Texture2D<float4> lighting;
    public SamplerState sampler;
    public StructuredBuffer<uint> scene;
    public uint cardCount;
    public uint tileResolution;
};

public struct LumenCardSampleAccumulator
{
    public float sampleWeightSum;
    public float3 lightingSum;
    public uint cardIndex;
    public float maxSampleWeight;
    public float2 physicalAtlasUV;
};

public LumenCardSampleAccumulator initLumenCardSampleAccumulator()
{
    LumenCardSampleAccumulator accumulator = (LumenCardSampleAccumulator)0;
    accumulator.cardIndex = ~0u;
    return accumulator;
}

struct LumenCardSample
{
    float2 physicalAtlasUV;
    float4 texelBilinearWeights;
};

// UE 5.7.4 LumenSurfaceCacheSampling.ush:108, with one resident tile per card.
LumenCardSample computeSurfaceCacheSample(LumenSurfaceCache cache, uint cardIndex, float2 localSamplePosition,
    float2 localExtent)
{
    float2 cardUV = min(saturate(float2(+0.5f, -0.5f) * (localSamplePosition / localExtent) + 0.5f), 0.999999f);
    uint width, height;
    cache.depth.GetDimensions(width, height);
    uint tilesPerRow = width / cache.tileResolution;
    uint2 atlasBias = uint2(cardIndex % tilesPerRow, cardIndex / tilesPerRow) * cache.tileResolution;
    float subTexelPrecisionBias = 1.0f / 512.0f;
    float2 coordInTile = clamp(cardUV * float(cache.tileResolution), 0.5f,
        float(cache.tileResolution) - 0.5f - subTexelPrecisionBias);
    float2 physicalAtlasSize = float2(width, height);
    float2 physicalAtlasUV = (coordInTile + float2(atlasBias)) * (1.0f / physicalAtlasSize);
    float2 fracUV = frac(physicalAtlasUV * physicalAtlasSize + 0.5f + subTexelPrecisionBias);

    LumenCardSample sample;
    sample.physicalAtlasUV = physicalAtlasUV;
    sample.texelBilinearWeights.x = (1.0f - fracUV.x) * fracUV.y;
    sample.texelBilinearWeights.y = fracUV.x * fracUV.y;
    sample.texelBilinearWeights.z = fracUV.x * (1.0f - fracUV.y);
    sample.texelBilinearWeights.w = (1.0f - fracUV.x) * (1.0f - fracUV.y);
    return sample;
}

// UE 5.7.4 LumenSurfaceCacheSampling.ush:246.
float3 sampleSurfaceCacheAtlas(Texture2D<float4> atlas, SamplerState sampler, float2 atlasUV, float4 texelWeights)
{
    float4 sampleX4 = atlas.GatherRed(sampler, atlasUV);
    float4 sampleY4 = atlas.GatherGreen(sampler, atlasUV);
    float4 sampleZ4 = atlas.GatherBlue(sampler, atlasUV);
    float3 sample;
    sample.x = dot(sampleX4, texelWeights);
    sample.y = dot(sampleY4, texelWeights);
    sample.z = dot(sampleZ4, texelWeights);
    return sample;
}

// UE 5.7.4 LumenSurfaceCacheSampling.ush:258, opaque mesh cards and final lighting.
void sampleLumenCard(LumenSurfaceCache cache, SurfaceCacheCard card, uint cardIndex, float3 worldPosition,
    float normalWeight, float surfaceCacheBias, inout LumenCardSampleAccumulator accumulator)
{
    float3 offset = worldPosition - card.origin.xyz;
    float3 cardSpacePosition = float3(dot(offset, card.axisX.xyz), dot(offset, card.axisY.xyz), dot(offset, card.axisZ.xyz));
    float3 localExtent = float3(card.axisX.w, card.axisY.w, card.axisZ.w);
    if (!all(abs(cardSpacePosition) <= localExtent + 0.5f * surfaceCacheBias)) {
        return;
    }

    cardSpacePosition.xy = clamp(cardSpacePosition.xy, -localExtent.xy, localExtent.xy);
    LumenCardSample cardSample = computeSurfaceCacheSample(cache, cardIndex, cardSpacePosition.xy, localExtent.xy);
    float4 texelDepths = cache.depth.GatherRed(cache.sampler, cardSample.physicalAtlasUV);
    float normalizedHitDistance = -(cardSpacePosition.z / localExtent.z) * 0.5f + 0.5f;
    float biasThreshold = surfaceCacheBias / localExtent.z;
    float biasFalloff = 0.25f * biasThreshold;
    float4 texelVisibility = 0.0f;
    for (uint texelIndex = 0u; texelIndex < 4u; ++texelIndex) {
        if (texelDepths[texelIndex] < 1.0f) {
            texelVisibility[texelIndex] = 1.0f - saturate(
                (abs(normalizedHitDistance - texelDepths[texelIndex]) - biasThreshold) / biasFalloff);
        }
    }

    float4 texelWeights = cardSample.texelBilinearWeights * texelVisibility;
    float cardSampleWeight = normalWeight * dot(texelWeights, 1.0f);
    if (cardSampleWeight > 0.0f) {
        float texelWeightSum = dot(texelWeights, 1.0f);
        texelWeights /= texelWeightSum;
        float3 finalLighting = sampleSurfaceCacheAtlas(cache.lighting, cache.sampler, cardSample.physicalAtlasUV, texelWeights);
        accumulator.lightingSum += finalLighting * cardSampleWeight;
        accumulator.sampleWeightSum += cardSampleWeight;
        if (cardSampleWeight > accumulator.maxSampleWeight) {
            accumulator.cardIndex = cardIndex;
            accumulator.physicalAtlasUV = cardSample.physicalAtlasUV;
            accumulator.maxSampleWeight = cardSampleWeight;
        }
    }
}

void sampleLumenCardRange(LumenSurfaceCache cache, uint first, uint end, float3 worldPosition, float3 worldNormal,
    float surfaceCacheBias, inout LumenCardSampleAccumulator accumulator)
{
    for (uint cardIndex = first; cardIndex < end; ++cardIndex) {
        SurfaceCacheCard card = cache.cards[cardIndex];
        float normalProjection = dot(worldNormal, card.axisZ.xyz);
        if (normalProjection > 0.0f) {
            float normalWeight = normalProjection * normalProjection;
            sampleLumenCard(cache, card, cardIndex, worldPosition, normalWeight, surfaceCacheBias, accumulator);
        }
    }
}

// UE 5.7.4 LumenSurfaceCacheSampling.ush:476. Baked cards are cardinal in mesh space, so the signed
// mesh-axis projection uses the card's world-space Z axis, including mirrored instances.
public void sampleLumenMeshCards(LumenSurfaceCache cache, uint objectId, float3 worldPosition, float3 worldNormal,
    float surfaceCacheBias, inout LumenCardSampleAccumulator accumulator)
{
    uint words, stride;
    cache.scene.GetDimensions(words, stride);
    uint rangeCount = 0u;
    if (words >= kLumenCardSceneRangeBase) {
        if (cache.scene[kLumenCardSceneHeaderWords] == kLumenCardSceneRangeTag) {
            rangeCount = cache.scene[kLumenCardSceneRangeCount];
            if (rangeCount > (words - kLumenCardSceneRangeBase) / kLumenCardSceneRangeStride) { rangeCount = 0u; }
        }
    }
    if (rangeCount == 0u) {
        // The original four-word metadata is also accepted from custom producers.
        for (uint slot = 0u; slot < cache.cardCount; ++slot) {
            if (cache.cards[slot].metadata.x == objectId) {
                sampleLumenCardRange(cache, slot, slot + 1u, worldPosition, worldNormal, surfaceCacheBias, accumulator);
            }
        }
        return;
    }

    uint first = 0u;
    uint end = rangeCount;
    while (first < end) {
        const uint middle = first + (end - first) / 2u;
        const uint word = kLumenCardSceneRangeBase + middle * kLumenCardSceneRangeStride;
        if (cache.scene[word] < objectId) { first = middle + 1u; }
        else { end = middle; }
    }
    for (; first < rangeCount; ++first) {
        const uint word = kLumenCardSceneRangeBase + first * kLumenCardSceneRangeStride;
        if (cache.scene[word] != objectId) { break; }
        sampleLumenCardRange(cache, cache.scene[word + 1u], cache.scene[word + 2u], worldPosition, worldNormal,
            surfaceCacheBias, accumulator);
    }
}

// UE 5.7.4 EvaluateRayHitFromCardSampleAccumulator, final outgoing radiance only.
public float3 evaluateLumenCardRadiance(LumenCardSampleAccumulator accumulator)
{
    return accumulator.sampleWeightSum > 0.0f ? accumulator.lightingSum / accumulator.sampleWeightSum : float3(0.0f);
}
} // namespace Metallic.Lumen
