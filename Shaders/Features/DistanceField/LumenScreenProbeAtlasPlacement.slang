module LumenScreenProbeAtlasPlacement;

import ShaderCore;
import ParameterRoot;
import LumenAdaptiveProbes;

using Metallic;
using Metallic.Lumen;

// UE 5.7.4's 8x8 group with 4x2 samples per uniform probe covers 2x4 tiles.
static const uint kPlacementGroupSize = 8u;
static const uint kSamplesPerTileX = 4u;
static const uint kSamplesPerTileY = 2u;
static const uint kTilesPerGroupX = kPlacementGroupSize / kSamplesPerTileX;
static const uint kTilesPerGroupY = kPlacementGroupSize / kSamplesPerTileY;
static const uint2 kSamplesPerTile = uint2(kSamplesPerTileX, kSamplesPerTileY);
static const uint2 kTilesPerGroup = uint2(kTilesPerGroupX, kTilesPerGroupY);

struct LumenScreenProbeAtlasPlacementParams
{
    ResourceHandle<Texture2D<float4>> position;
    ResourceHandle<Texture2D<float4>> normal;
    ResourceHandle<RWStructuredBuffer<uint2>> probePositions;
    ResourceHandle<RWStructuredBuffer<uint>> probeOrigins;
    ResourceHandle<RWStructuredBuffer<uint>> placementMasks;
    ResourceHandle<RWStructuredBuffer<uint>> adaptiveCount;
    ResourceHandle<RWStructuredBuffer<uint>> tileCounts;
    ResourceHandle<RWStructuredBuffer<uint>> tileIndices;
    uint width;
    uint height;
    uint probeWidth;
    uint probeHeight;
    uint maxAdaptiveProbes;
    uint frameIndex;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void lumenResetCompactProbeAtlasMain()
{
    let params = getParameters<LumenScreenProbeAtlasPlacementParams>();
    resolveUniform(params.adaptiveCount)[0] = 0u;
}

groupshared uint sharedPlacementMasks[kTilesPerGroupX][kTilesPerGroupY];

// UE ScreenProbeAdaptivePlacementMarkCS with Deferred receiver guides and a buffer per tile.
[shader("compute")]
[numthreads(kPlacementGroupSize, kPlacementGroupSize, 1)]
void lumenMarkCompactProbeAtlasMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    let params = getParameters<LumenScreenProbeAtlasPlacementParams>();
    if (all(lane.xy < kTilesPerGroup)) {
        sharedPlacementMasks[lane.x][lane.y] = 0u;
    }

    GroupMemoryBarrierWithGroupSync();

    const uint2 localTile = lane.xy / kSamplesPerTile;
    const uint2 sampleCoord = lane.xy % kSamplesPerTile;
    const uint2 tile = group.xy * kTilesPerGroup + localTile;
    const uint sampleIndex = sampleCoord.x + kSamplesPerTile.x * sampleCoord.y;
    const uint2 viewSize = uint2(params.width, params.height);
    const uint2 probeSize = uint2(params.probeWidth, params.probeHeight);
    const bool validTile = all(tile < probeSize);
    let position = resolveUniform(params.position);

    if (validTile && params.maxAdaptiveProbes != 0u) {
        const uint2 samplePixel = adaptiveProbePosition(tile, sampleIndex, params.frameIndex);
        float4 receiver;
        float3 receiverNormal;
        uint2 firstCorner;
        if (markAdaptiveProbe(samplePixel, viewSize, probeSize, position, resolveUniform(params.normal),
                params.frameIndex, receiver, receiverNormal, firstCorner)) {
            InterlockedOr(sharedPlacementMasks[localTile.x][localTile.y], 1u << sampleIndex);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if (sampleIndex == 0u && validTile) {
        const uint tileIndex = lumenPackedUniformProbeIndex(tile, params.probeWidth);
        resolveUniform(params.placementMasks)[tileIndex] = sharedPlacementMasks[localTile.x][localTile.y];
        resolveUniform(params.tileCounts)[tileIndex] = 0u;
        const uint2 pixel = lumenUniformProbePosition(tile, viewSize, params.frameIndex);
        resolveUniform(params.probePositions)[tileIndex] = position.Load(int3(pixel, 0)).w > 0.0f ?
            pixel : uint2(kLumenInvalidProbe);
        resolveUniform(params.probeOrigins)[tileIndex] = lumenProbeIndex(tile, 0u, params.probeWidth);
    }
}

groupshared uint sharedNumProbesToAllocate;
groupshared uint sharedAdaptiveProbeBaseIndex;

// UE ScreenProbeAdaptivePlacementSpawnCS, retaining source slot metadata for diagnostics and ray coordinates.
[shader("compute")]
[numthreads(kPlacementGroupSize, kPlacementGroupSize, 1)]
void lumenSpawnCompactProbeAtlasMain(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    let params = getParameters<LumenScreenProbeAtlasPlacementParams>();
    if (params.maxAdaptiveProbes == 0u) {
        return;
    }
    if (all(lane == 0u)) {
        sharedNumProbesToAllocate = 0u;
    }

    GroupMemoryBarrierWithGroupSync();

    const uint2 localTile = lane.xy / kSamplesPerTile;
    const uint2 sampleCoord = lane.xy % kSamplesPerTile;
    const uint2 tile = group.xy * kTilesPerGroup + localTile;
    const uint sampleIndex = sampleCoord.x + kSamplesPerTile.x * sampleCoord.y;
    const uint2 viewSize = uint2(params.width, params.height);
    const uint2 probeSize = uint2(params.probeWidth, params.probeHeight);
    const uint tileIndex = lumenPackedUniformProbeIndex(tile, params.probeWidth);
    bool placeProbe = false;
    uint2 pixel = 0u;
    if (all(tile < probeSize)) {
        let masks = resolveUniform(params.placementMasks);
        const uint sampleMask = 1u << sampleIndex;
        if ((masks[tileIndex] & sampleMask) != 0u) {
            pixel = adaptiveProbePosition(tile, sampleIndex, params.frameIndex);
            placeProbe = shouldSpawnPackedAdaptiveProbe(pixel, sampleIndex, viewSize, probeSize,
                resolveUniform(params.position), resolveUniform(params.normal), masks, params.frameIndex);
        }
    }

    uint sharedListIndex = 0u;
    if (placeProbe) {
        InterlockedAdd(sharedNumProbesToAllocate, 1u, sharedListIndex);
    }

    GroupMemoryBarrierWithGroupSync();

    if (all(lane == 0u)) {
        let adaptiveCount = resolveUniform(params.adaptiveCount);
        InterlockedAdd(adaptiveCount[0], sharedNumProbesToAllocate, sharedAdaptiveProbeBaseIndex);
    }

    GroupMemoryBarrierWithGroupSync();

    const uint adaptiveIndex = sharedAdaptiveProbeBaseIndex + sharedListIndex;
    if (placeProbe && adaptiveIndex < params.maxAdaptiveProbes) {
        const uint probeIndex = params.probeWidth * params.probeHeight + adaptiveIndex;
        resolveUniform(params.probePositions)[probeIndex] = pixel;
        resolveUniform(params.probeOrigins)[probeIndex] = lumenProbeIndex(tile, sampleIndex + 1u, params.probeWidth);

        // GetScreenTileCoord(pixel) resolves to tile for the generated jittered sample.
        let tileCounts = resolveUniform(params.tileCounts);
        uint listIndex;
        InterlockedAdd(tileCounts[tileIndex], 1u, listIndex);
        resolveUniform(params.tileIndices)[tileIndex * kLumenAdaptiveProbesPerTile + listIndex] = adaptiveIndex;
    }
}
