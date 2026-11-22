module LumenAdaptiveProbes;

import LumenProbeMath;

namespace Metallic.Lumen
{
public static const uint kLumenProbeDownsampleFactor = 16u;
public static const uint kLumenProbeGatherResolution = 8u;
#if defined(METALLIC_LUMEN_TRACE_RESOLUTION_16)
public static const uint kLumenProbeTraceResolution = 16u;
#else
public static const uint kLumenProbeTraceResolution = 8u;
#endif
public static const uint kLumenProbeSlotsPerTile = 9u;
public static const uint kLumenIrradianceProbeResolution = 6u;
public static const uint kLumenIrradianceProbeWithBorderResolution = kLumenIrradianceProbeResolution + 2u;
public static const uint kLumenProbeSlotsPerTileAxis = 3u;
public static const uint kLumenIrradianceProbeTileStride = kLumenProbeSlotsPerTileAxis * kLumenIrradianceProbeWithBorderResolution;
public static const uint kLumenInvalidProbe = ~0u;

static const uint kAdaptiveProbesPerTile = kLumenProbeSlotsPerTile - 1u;
static const float kProbeInterpolationDepthWeight = -200.0f;
static const float kMinProbeInterpolationWeight = 0.01f;
static const float kMinProbeInterpolationFallbackWeight = 0.0001f;

public struct LumenProbeSample
{
    public uint4 indices;
    public float4 weights;
    public bool lightingValid;
};

public uint lumenProbeIndex(uint2 tile, uint slot, uint probeWidth)
{
    return (tile.y * probeWidth + tile.x) * kLumenProbeSlotsPerTile + slot;
}

// A screen tile owns one uniform and eight adaptive slots, packed into 3x3 bordered probes.
public uint2 lumenProbeAtlasCoord(uint probeIndex, uint probeWidth)
{
    uint tile = probeIndex / kLumenProbeSlotsPerTile;
    uint slot = probeIndex % kLumenProbeSlotsPerTile;
    return uint2(tile % probeWidth, tile / probeWidth) * kLumenProbeSlotsPerTileAxis
        + uint2(slot % kLumenProbeSlotsPerTileAxis, slot / kLumenProbeSlotsPerTileAxis);
}

public uint lumenProbeGatherAddress(uint probeIndex, uint2 ray)
{
    return (probeIndex * kLumenProbeGatherResolution + ray.y) * kLumenProbeGatherResolution + ray.x;
}

public uint lumenProbeTraceAddress(uint probeIndex, uint2 ray)
{
    return (probeIndex * kLumenProbeTraceResolution + ray.y) * kLumenProbeTraceResolution + ray.x;
}

uint2 probeTileJitter(uint frameIndex)
{
    return uint2(lumenHammersley16(frameIndex % 8u, 8u, uint2(0u)) * kLumenProbeDownsampleFactor);
}

// UE 5.7.4 GetScreenTileCoord, including probes clamped to the final partial tile.
public uint2 lumenScreenTileCoord(uint2 probePixel, uint frameIndex)
{
    return (probePixel - probeTileJitter(frameIndex)) / kLumenProbeDownsampleFactor;
}

public uint2 lumenUniformProbePosition(uint2 tile, uint2 viewSize, uint frameIndex)
{
    return min(tile * kLumenProbeDownsampleFactor + probeTileJitter(frameIndex), viewSize - 1u);
}

uint2 adaptiveProbePosition(uint2 tile, uint sampleIndex, uint frameIndex)
{
    uint2 uniformPosition = tile * kLumenProbeDownsampleFactor + probeTileJitter(frameIndex);
    uint2 sampleSeed = lumenRand3DPCG16(int3(tile, frameIndex % 8u)).xy;
    return uniformPosition + uint2(clamp(lumenHammersley16(sampleIndex, kAdaptiveProbesPerTile, sampleSeed) *
        kLumenProbeDownsampleFactor, 0.0f, float(kLumenProbeDownsampleFactor - 1u)));
}

uint2 firstUniformCorner(uint2 pixel, uint2 viewSize, uint2 probeSize, uint frameIndex, out uint2 screenCoord)
{
    screenCoord = uint2(clamp(float2(pixel) - float2(probeTileJitter(frameIndex)), 0.0f, float2(viewSize) - 1.0f));
    return min(screenCoord / kLumenProbeDownsampleFactor, max(probeSize, 2u) - 2u);
}

uint2 uniformCornerTile(uint2 firstCorner, uint corner, uint2 probeSize)
{
    return min(firstCorner + uint2(corner % 2u, corner / 2u), probeSize - 1u);
}

// UE 5.7.4 CalculateUniformUpsampleInterpolationWeights for opaque material mip 0.
LumenProbeSample calculateUniformProbeWeights(uint2 pixel, uint2 viewSize, uint2 probeSize, float4 receiver,
    float3 normal, Texture2D<float4> position, uint frameIndex, out uint2 firstCorner)
{
    uint2 screenCoord;
    firstCorner = firstUniformCorner(pixel, viewSize, probeSize, frameIndex, screenCoord);
    const uint bilinearExpand = 1u;
    float2 bilinear = float2(screenCoord - firstCorner * kLumenProbeDownsampleFactor + bilinearExpand) /
        float(kLumenProbeDownsampleFactor + 2u * bilinearExpand);

    LumenProbeSample sample;
    sample.lightingValid = false;
    sample.weights = float4((1.0f - bilinear.y) * (1.0f - bilinear.x), (1.0f - bilinear.y) * bilinear.x,
        bilinear.y * (1.0f - bilinear.x), bilinear.y * bilinear.x);
    float4 scenePlane = float4(normal, dot(receiver.xyz, normal));
    [unroll]
    for (uint corner = 0u; corner < 4u; ++corner) {
        uint2 tile = uniformCornerTile(firstCorner, corner, probeSize);
        sample.indices[corner] = lumenProbeIndex(tile, 0u, probeSize.x);
        float4 probe = position.Load(int3(lumenUniformProbePosition(tile, viewSize, frameIndex), 0));
        float planeDistance = abs(dot(float4(probe.xyz, -1.0f), scenePlane));
        float relativeDepthDifference = abs(planeDistance / receiver.w);
        sample.weights[corner] *= probe.w > 0.0f ? exp2(kProbeInterpolationDepthWeight * relativeDepthDifference) : 0.0f;
    }
    return sample;
}

// UE 5.7.4 GetAdaptiveProbeInterpolationWeight, PLANE_WEIGHTING opaque branch.
float getAdaptiveProbeWeight(uint2 pixel, float4 scenePlane, float sceneDepth, uint2 probePixel, float3 probePosition)
{
    float planeDistance = abs(dot(float4(probePosition, -1.0f), scenePlane));
    float relativeDepthDifference = abs(planeDistance / sceneDepth);
    float depthWeight = exp2(kProbeInterpolationDepthWeight * relativeDepthDifference);
    float2 distanceToScreenProbe = abs(float2(probePixel) - float2(pixel));
    float cornerWeight = 1.0f - saturate(min(distanceToScreenProbe.x, distanceToScreenProbe.y) /
        float(kLumenProbeDownsampleFactor));
    return depthWeight * cornerWeight;
}

// UE 5.7.4 ScreenProbeAdaptivePlacementMarkCS. Positions and linear depth come from Deferred.
bool markAdaptiveProbe(uint2 pixel, uint2 viewSize, uint2 probeSize, Texture2D<float4> position,
    Texture2D<float4> normal, uint frameIndex, out float4 receiver, out float3 receiverNormal, out uint2 firstCorner)
{
    receiver = 0.0f;
    receiverNormal = 0.0f;
    firstCorner = uint2(0u);
    if (any(pixel >= viewSize)) {
        return false;
    }
    receiver = position.Load(int3(pixel, 0));
    if (receiver.w <= 0.0f) {
        return false;
    }
    receiverNormal = normalize(normal.Load(int3(pixel, 0)).xyz);
    LumenProbeSample sample = calculateUniformProbeWeights(pixel, viewSize, probeSize, receiver, receiverNormal,
        position, frameIndex, firstCorner);
    return dot(sample.weights, 1.0f) < kMinProbeInterpolationWeight;
}

// UE 5.7.4 ScreenProbeAdaptivePlacementSpawnCS. Evaluate the lower-index Mark predicates
// from read-only guides, and write deterministic tile-local slots instead of a global compacted atlas.
public uint2 placeLumenAdaptiveProbe(uint2 tile, uint slot, uint2 viewSize, uint2 probeSize,
    Texture2D<float4> position, Texture2D<float4> normal, uint frameIndex)
{
    const uint2 invalid = uint2(kLumenInvalidProbe);
    if (slot == 0u) {
        uint2 uniformPixel = lumenUniformProbePosition(tile, viewSize, frameIndex);
        return position.Load(int3(uniformPixel, 0)).w > 0.0f ? uniformPixel : invalid;
    }

    uint sampleIndex = slot - 1u;
    uint2 pixel = adaptiveProbePosition(tile, sampleIndex, frameIndex);
    float4 receiver;
    float3 receiverNormal;
    uint2 firstCorner;
    if (!markAdaptiveProbe(pixel, viewSize, probeSize, position, normal, frameIndex, receiver, receiverNormal, firstCorner)) {
        return invalid;
    }

    float4 scenePlane = float4(receiverNormal, dot(receiver.xyz, receiverNormal));
    float4 cornerWeights = 0.0f;
    for (uint corner = 0u; corner < 4u; ++corner) {
        uint2 cornerTile = uniformCornerTile(firstCorner, corner, probeSize);
        for (uint candidate = 0u; candidate < sampleIndex; ++candidate) {
            uint2 cornerPixel = adaptiveProbePosition(cornerTile, candidate, frameIndex);
            float4 cornerReceiver;
            float3 cornerNormal;
            uint2 unusedCorner;
            if (markAdaptiveProbe(cornerPixel, viewSize, probeSize, position, normal, frameIndex, cornerReceiver,
                    cornerNormal, unusedCorner)) {
                float weight = getAdaptiveProbeWeight(pixel, scenePlane, receiver.w, cornerPixel, cornerReceiver.xyz);
                cornerWeights[corner] = max(cornerWeights[corner], weight);
                if (dot(cornerWeights, 1.0f) >= kMinProbeInterpolationWeight) {
                    return invalid;
                }
            }
        }
    }
    return pixel;
}

// UE 5.7.4 CalculateUpsampleInterpolationWeights. Substitute the best adaptive probe for a weak uniform corner.
public LumenProbeSample calculateLumenAdaptiveInterpolation(uint2 pixel, uint2 viewSize, uint2 probeSize, float4 receiver,
    float3 normal, Texture2D<float4> position, StructuredBuffer<uint2> probes, uint frameIndex)
{
    uint2 firstCorner;
    LumenProbeSample sample = calculateUniformProbeWeights(pixel, viewSize, probeSize, receiver, normal, position,
        frameIndex, firstCorner);
    float4 scenePlane = float4(normal, dot(receiver.xyz, normal));
    [unroll]
    for (uint corner = 0u; corner < 4u; ++corner) {
        if (sample.weights[corner] >= kMinProbeInterpolationWeight) {
            continue;
        }
        uint2 tile = uniformCornerTile(firstCorner, corner, probeSize);
        for (uint slot = 1u; slot < kLumenProbeSlotsPerTile; ++slot) {
            uint index = lumenProbeIndex(tile, slot, probeSize.x);
            uint2 probePixel = probes[index];
            if (probePixel.x == kLumenInvalidProbe) {
                continue;
            }
            float3 probePosition = position.Load(int3(probePixel, 0)).xyz;
            float weight = getAdaptiveProbeWeight(pixel, scenePlane, receiver.w, probePixel, probePosition);
            if (weight > sample.weights[corner]) {
                sample.weights[corner] = weight;
                sample.indices[corner] = index;
            }
        }
    }

    float weightSum = dot(sample.weights, 1.0f);
    sample.lightingValid = weightSum >= kMinProbeInterpolationWeight;
    if (weightSum >= kMinProbeInterpolationFallbackWeight) {
        sample.weights /= max(weightSum, kMinProbeInterpolationFallbackWeight);
    } else {
        sample.weights = 0.0f;
    }
    return sample;
}
} // namespace Metallic.Lumen
