module LumenReflectionSchedule;

import ShaderCore;
import ParameterRoot;

using Metallic;

struct LumenReflectionScheduleParams
{
    ResourceHandle<Texture2D<float4>> position, specular, rayData;
    ResourceHandle<Texture2D<float>> rayHitDistance;
    ResourceHandle<Texture2D<float4>> trace;
    ResourceHandle<RWStructuredBuffer<uint>> tileMask, activeTiles, clearTiles, workCounts, rayList, indirectArgs;
    uint width, height, maxDispatchX;
    float maxRoughness, meshSDFTraceDistance, cardTraceEndDistance, maxTraceDistance;
};

static const uint kReflectionTileSize = 8u;
static const uint kReflectionTilePixels = kReflectionTileSize * kReflectionTileSize;
static const uint kReflectionTileBit = 0x08u;
static const uint kActiveTileCount = 0u;
static const uint kClearTileCount = 1u;
static const uint kDetailRayCount = 2u;
static const uint kGDFRayCount = 3u;
static const uint kActiveTileArgs = 0u;
static const uint kClearTileArgs = 1u;
static const uint kCompactRayArgs = 2u;
static const uint kDetailRayArgs = 3u;
static const uint kGDFRayArgs = 4u;

// UE r.Lumen.Reflections.TraceCompaction.GroupSizeInTraceTiles, with the fallback forced to one tile.
#if defined(METALLIC_LUMEN_REFLECTION_WAVE_OPS)
static const uint kTilesPerCompactionGroup = 16u;
#else
static const uint kTilesPerCompactionGroup = 1u;
#endif
static const uint kCompactionGroupThreads = kTilesPerCompactionGroup * kReflectionTilePixels;

// UE Common.ush ZOrder2D(Index, log2(8)), for tile batches and pixels within each tile.
uint2 reflectionScheduleZOrder2D(uint index)
{
    uint2 coord = 0u;
    [unroll]
    for (uint i = 0u; i < 3u; ++i) {
        coord.x |= ((index >> (2u * i + 0u)) & 0x1u) << i;
        coord.y |= ((index >> (2u * i + 1u)) & 0x1u) << i;
    }
    return coord;
}

uint packReflectionTile(uint2 tile)
{
    return (tile.x & 0xfffu) | ((tile.y & 0xfffu) << 12u);
}

uint2 unpackReflectionTile(uint tile)
{
    return uint2(tile & 0xfffu, (tile >> 12u) & 0xfffu);
}

uint packReflectionTraceTexel(uint2 pixel)
{
    return (pixel.x & 0x3fffu) | ((pixel.y & 0x3fffu) << 14u);
}

void writeReflectionScheduleArgs(RWStructuredBuffer<uint> args, uint index, uint groups, uint maxDispatchX)
{
    const uint offset = index * 3u;
    args[offset] = min(groups, maxDispatchX);
    args[offset + 1u] = max(1u, (groups + maxDispatchX - 1u) / maxDispatchX);
    args[offset + 2u] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lumenReflectionInitWorkMain()
{
    let params = getParameters<LumenReflectionScheduleParams>();
    let workCounts = resolveUniform(params.workCounts);
    workCounts[kActiveTileCount] = 0u;
    workCounts[kClearTileCount] = 0u;
    workCounts[kDetailRayCount] = 0u;
    workCounts[kGDFRayCount] = 0u;
}

#if defined(METALLIC_LUMEN_REFLECTION_WAVE_OPS)
groupshared uint sharedReflectionTileMask;
#else
groupshared uint sharedReflectionTileMask[kReflectionTilePixels];
#endif

// Opaque, single-closure reflection bit from UE StochasticLightingTileClassificationMarkCS.
[shader("compute")]
[numthreads(kReflectionTileSize, kReflectionTileSize, 1)]
void lumenReflectionMarkTilesMain(uint3 group : SV_GroupID, uint3 thread : SV_DispatchThreadID, uint index : SV_GroupIndex)
{
    let params = getParameters<LumenReflectionScheduleParams>();
    uint reflectionBit = 0u;
    if (all(thread.xy < uint2(params.width, params.height))) {
        const float4 position = resolveUniform(params.position).Load(int3(thread.xy, 0));
        const float4 specular = resolveUniform(params.specular).Load(int3(thread.xy, 0));
        // Match the existing Metallic full-resolution ray generation material adapter.
        const float fadeAlpha = saturate((params.maxRoughness - specular.w) * (1.0f / 0.1f));
        reflectionBit = position.w > 0.0f && any(specular.xyz > 0.0f) && fadeAlpha > 0.0f ? kReflectionTileBit : 0u;
    }
#if defined(METALLIC_LUMEN_REFLECTION_WAVE_OPS)
    reflectionBit = WaveActiveBitOr(reflectionBit);
    if (WaveGetLaneCount() < kReflectionTilePixels) {
        if (index == 0u) { sharedReflectionTileMask = 0u; }
        GroupMemoryBarrierWithGroupSync();
        if (WaveIsFirstLane()) { InterlockedOr(sharedReflectionTileMask, reflectionBit); }
        GroupMemoryBarrierWithGroupSync();
        reflectionBit = sharedReflectionTileMask;
    }
#else
    sharedReflectionTileMask[index] = reflectionBit;
    [unroll]
    for (uint offset = kReflectionTilePixels / 2u; offset > 0u; offset >>= 1u) {
        GroupMemoryBarrierWithGroupSync();
        if (index < offset) {
            sharedReflectionTileMask[index] |= sharedReflectionTileMask[index + offset];
        }
    }
    if (index == 0u) { reflectionBit = sharedReflectionTileMask[0]; }
#endif
    if (index == 0u) {
        const uint tilesX = (params.width + kReflectionTileSize - 1u) / kReflectionTileSize;
        resolveUniform(params.tileMask)[group.y * tilesX + group.x] = reflectionBit;
    }
}

#if !defined(METALLIC_LUMEN_REFLECTION_WAVE_OPS)
groupshared uint sharedNumTiles;
groupshared uint sharedNumClearTiles;
groupshared uint sharedTileData[kReflectionTilePixels];
groupshared uint sharedGlobalTileOffset;
groupshared uint sharedGlobalClearTileOffset;
#endif

// UE ReflectionTileClassificationBuildListsCS, full resolution and no overflow closures.
[shader("compute")]
[numthreads(kReflectionTileSize, kReflectionTileSize, 1)]
void lumenReflectionBuildTileListsMain(uint3 group : SV_GroupID, uint index : SV_GroupIndex)
{
    let params = getParameters<LumenReflectionScheduleParams>();
    let tileMask = resolveUniform(params.tileMask);
    let workCounts = resolveUniform(params.workCounts);
    let activeTiles = resolveUniform(params.activeTiles);
    let clearTiles = resolveUniform(params.clearTiles);
    const uint2 tileDimensions = (uint2(params.width, params.height) + kReflectionTileSize - 1u) / kReflectionTileSize;
#if defined(METALLIC_LUMEN_REFLECTION_WAVE_OPS)
    const uint2 tile = group.xy * kReflectionTileSize + reflectionScheduleZOrder2D(index);
    if (all(tile < tileDimensions)) {
        const bool tileUsed = (tileMask[tile.y * tileDimensions.x + tile.x] & kReflectionTileBit) != 0u;
        const uint activeInWave = WaveActiveCountBits(tileUsed);
        uint globalTileOffset = 0u;
        if (WaveIsFirstLane() && activeInWave > 0u) {
            InterlockedAdd(workCounts[kActiveTileCount], activeInWave, globalTileOffset);
        }
        globalTileOffset = WaveReadLaneFirst(globalTileOffset);
        if (tileUsed) {
            activeTiles[globalTileOffset + WavePrefixCountBits(true)] = packReflectionTile(tile);
        }

        const uint clearInWave = WaveActiveCountBits(!tileUsed);
        uint globalClearTileOffset = 0u;
        if (WaveIsFirstLane() && clearInWave > 0u) {
            InterlockedAdd(workCounts[kClearTileCount], clearInWave, globalClearTileOffset);
        }
        globalClearTileOffset = WaveReadLaneFirst(globalClearTileOffset);
        if (!tileUsed) {
            clearTiles[globalClearTileOffset + WavePrefixCountBits(true)] = packReflectionTile(tile);
        }
    }
#else
    if (index == 0u) {
        sharedNumTiles = 0u;
        sharedNumClearTiles = 0u;
        for (uint x = 0u; x < kReflectionTilePixels; ++x) {
            const uint2 tile = group.xy * kReflectionTileSize + reflectionScheduleZOrder2D(x);
            if (all(tile < tileDimensions)) {
                const bool tileUsed = (tileMask[tile.y * tileDimensions.x + tile.x] & kReflectionTileBit) != 0u;
                if (tileUsed) {
                    sharedTileData[sharedNumTiles] = packReflectionTile(tile);
                    sharedNumTiles += 1u;
                } else {
                    sharedTileData[kReflectionTilePixels - 1u - sharedNumClearTiles] = packReflectionTile(tile);
                    sharedNumClearTiles += 1u;
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (index == 0u && sharedNumTiles > 0u) {
        InterlockedAdd(workCounts[kActiveTileCount], sharedNumTiles, sharedGlobalTileOffset);
    }
    if (index == 0u && sharedNumClearTiles > 0u) {
        InterlockedAdd(workCounts[kClearTileCount], sharedNumClearTiles, sharedGlobalClearTileOffset);
    }
    GroupMemoryBarrierWithGroupSync();
    if (index < sharedNumTiles) {
        activeTiles[sharedGlobalTileOffset + index] = sharedTileData[index];
    } else {
        const uint localIndex = index - sharedNumTiles;
        if (localIndex < sharedNumClearTiles) {
            clearTiles[sharedGlobalClearTileOffset + localIndex] = sharedTileData[kReflectionTilePixels - 1u - localIndex];
        }
    }
#endif
}

// UE keeps the real list length in the allocators. The Vulkan records may span two dimensions.
[shader("compute")]
[numthreads(1, 1, 1)]
void lumenReflectionSetupTileArgsMain()
{
    let params = getParameters<LumenReflectionScheduleParams>();
    let counts = resolveUniform(params.workCounts);
    let args = resolveUniform(params.indirectArgs);
    const uint active = counts[kActiveTileCount];
    writeReflectionScheduleArgs(args, kActiveTileArgs, active, params.maxDispatchX);
    writeReflectionScheduleArgs(args, kClearTileArgs, counts[kClearTileCount], params.maxDispatchX);
    writeReflectionScheduleArgs(args, kCompactRayArgs,
        (active + kTilesPerCompactionGroup - 1u) / kTilesPerCompactionGroup, params.maxDispatchX);
    writeReflectionScheduleArgs(args, kDetailRayArgs, 0u, params.maxDispatchX);
    writeReflectionScheduleArgs(args, kGDFRayArgs, 0u, params.maxDispatchX);
}

groupshared uint sharedGlobalTraceTexelStartOffset;
#if defined(METALLIC_LUMEN_REFLECTION_WAVE_OPS)
groupshared uint sharedGroupSum;
#else
groupshared uint sharedTraceTexelAllocator;
groupshared uint sharedTraceTexels[kCompactionGroupThreads];
#endif

// UE ReflectionCompactTracesCS default software mode. Ray validity and hit state use Metallic's trace resources.
[shader("compute")]
[numthreads(kCompactionGroupThreads, 1, 1)]
void lumenReflectionCompactRaysMain(uint3 group : SV_GroupID, uint index : SV_GroupIndex)
{
    let params = getParameters<LumenReflectionScheduleParams>();
    let counts = resolveUniform(params.workCounts);
#if defined(METALLIC_LUMEN_REFLECTION_WAVE_OPS)
    if (index == 0u) { sharedGroupSum = 0u; }
#else
    if (index == 0u) { sharedTraceTexelAllocator = 0u; }
    GroupMemoryBarrierWithGroupSync();
#endif

    const uint active = counts[kActiveTileCount];
    const uint compactGroups = (active + kTilesPerCompactionGroup - 1u) / kTilesPerCompactionGroup;
    const uint linearGroup = group.y * min(compactGroups, params.maxDispatchX) + group.x;
    const uint tileIndex = linearGroup * kTilesPerCompactionGroup + index / kReflectionTilePixels;
    uint traceTexel = 0u;
    bool traceValid = false;
    if (tileIndex < active) {
        const uint2 tile = unpackReflectionTile(resolveUniform(params.activeTiles)[tileIndex]);
        const uint2 pixel = tile * kReflectionTileSize + reflectionScheduleZOrder2D(index % kReflectionTilePixels);
        if (all(pixel < uint2(params.width, params.height))) {
            const float sceneDepth = resolveUniform(params.position).Load(int3(pixel, 0)).w;
            const float rayCone = resolveUniform(params.rayData).Load(int3(pixel, 0)).w;
            const float traceClass = resolveUniform(params.trace).Load(int3(pixel, 0)).y;
            const float traceHitDistanceCm = resolveUniform(params.rayHitDistance).Load(int3(pixel, 0));
            const bool acceptTrace = rayCone > 0.0f && traceClass != 5.0f && traceClass != 9.0f;
#if defined(METALLIC_LUMEN_REFLECTION_COMPACT_DETAIL)
            traceValid = sceneDepth > 0.0f && acceptTrace && sceneDepth < params.cardTraceEndDistance &&
                traceHitDistanceCm * 0.01f <= params.meshSDFTraceDistance;
#else
            // Ray generation also caps and quantizes the tracing extent in centimeters.
            const float maxTraceDistanceCm = f16tof32(f32tof16(min(params.maxTraceDistance * 100.0f, 65504.0f)) & 0x7fffu);
            traceValid = sceneDepth > 0.0f && acceptTrace && traceHitDistanceCm <= maxTraceDistanceCm;
#endif
            if (traceValid) {
                traceTexel = packReflectionTraceTexel(pixel);
#if !defined(METALLIC_LUMEN_REFLECTION_WAVE_OPS)
                uint sharedOffset;
                InterlockedAdd(sharedTraceTexelAllocator, 1u, sharedOffset);
                sharedTraceTexels[sharedOffset] = traceTexel;
#endif
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

#if defined(METALLIC_LUMEN_REFLECTION_COMPACT_DETAIL)
    const uint countIndex = kDetailRayCount;
#else
    const uint countIndex = kGDFRayCount;
#endif
    let rayList = resolveUniform(params.rayList);
#if defined(METALLIC_LUMEN_REFLECTION_WAVE_OPS)
    const uint lastLane = WaveGetLaneCount() - 1u;
    const uint lane = WaveGetLaneIndex();
    const uint offsetInWave = WavePrefixCountBits(traceValid);
    uint offsetInGroup = 0u;
    if (lane == lastLane) {
        const uint waveSum = offsetInWave + (traceValid ? 1u : 0u);
        InterlockedAdd(sharedGroupSum, waveSum, offsetInGroup);
    }
    offsetInGroup = WaveReadLaneAt(offsetInGroup, lastLane) + offsetInWave;
    GroupMemoryBarrierWithGroupSync();
    if (index == 0u) {
        InterlockedAdd(counts[countIndex], sharedGroupSum, sharedGlobalTraceTexelStartOffset);
    }
    GroupMemoryBarrierWithGroupSync();
    if (traceValid) {
        rayList[sharedGlobalTraceTexelStartOffset + offsetInGroup] = traceTexel;
    }
#else
    if (index == 0u) {
        InterlockedAdd(counts[countIndex], sharedTraceTexelAllocator, sharedGlobalTraceTexelStartOffset);
    }
    GroupMemoryBarrierWithGroupSync();
    if (index < sharedTraceTexelAllocator) {
        rayList[sharedGlobalTraceTexelStartOffset + index] = sharedTraceTexels[index];
    }
#endif
}

// UE SetupCompactedTracesIndirectArgsCS, the 64-thread software ray worker variant.
[shader("compute")]
[numthreads(1, 1, 1)]
void lumenReflectionSetupDetailArgsMain()
{
    let params = getParameters<LumenReflectionScheduleParams>();
    const uint rays = resolveUniform(params.workCounts)[kDetailRayCount];
    writeReflectionScheduleArgs(resolveUniform(params.indirectArgs), kDetailRayArgs, (rays + 63u) / 64u, params.maxDispatchX);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lumenReflectionSetupGDFArgsMain()
{
    let params = getParameters<LumenReflectionScheduleParams>();
    const uint rays = resolveUniform(params.workCounts)[kGDFRayCount];
    writeReflectionScheduleArgs(resolveUniform(params.indirectArgs), kGDFRayArgs, (rays + 63u) / 64u, params.maxDispatchX);
}
