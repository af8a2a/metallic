module LumenSurfaceCache;

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

public struct LumenSurfaceCache
{
    public StructuredBuffer<SurfaceCacheCard> cards;
    public Texture2D<float> depth;
    public Texture2D<float4> lighting;
    public SamplerState sampler;
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

// UE 5.7.4 LumenSurfaceCacheSampling.ush:476. Baked cards are cardinal in mesh space, so the signed
// mesh-axis projection can use the card's world-space Z axis directly, including mirrored instances.
// The fixed atlas has one tile per card, so virtual atlas mip selection is specialized away.
public void sampleLumenMeshCards(LumenSurfaceCache cache, uint objectId, float3 worldPosition, float3 worldNormal,
    float surfaceCacheBias, inout LumenCardSampleAccumulator accumulator)
{
    for (uint cardIndex = 0u; cardIndex < cache.cardCount; ++cardIndex) {
        SurfaceCacheCard card = cache.cards[cardIndex];
        if (card.metadata.x != objectId) {
            continue;
        }

        float normalProjection = dot(worldNormal, card.axisZ.xyz);
        if (normalProjection > 0.0f) {
            float normalWeight = normalProjection * normalProjection;
            sampleLumenCard(cache, card, cardIndex, worldPosition, normalWeight, surfaceCacheBias, accumulator);
        }
    }
}

// UE 5.7.4 EvaluateRayHitFromCardSampleAccumulator, final outgoing radiance only.
public float3 evaluateLumenCardRadiance(LumenCardSampleAccumulator accumulator)
{
    return accumulator.sampleWeightSum > 0.0f ? accumulator.lightingSum / accumulator.sampleWeightSum : float3(0.0f);
}
} // namespace Metallic.Lumen
