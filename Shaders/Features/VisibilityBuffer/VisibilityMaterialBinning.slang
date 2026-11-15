module VisibilityMaterialBinning;

import Core;
import ParameterRoot;
using Metallic;
import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;


#ifndef METALLIC_MATERIAL_BINNING_MODE
#define METALLIC_MATERIAL_BINNING_MODE 0
#endif

// Wire layout matches MaterialBinningParams.h, verified by GPU producer/consumer tests.
struct MaterialBinningParams
{
    ResourceHandle<Texture2D<uint>> visibility;
    RWBufferSpan<GPUDrivenPreviewInstance> instances;
    RWBufferSpan<GPUDrivenPreviewMaterial> materials;
    RWBufferSpan<SceneMaterialData> shadingMaterials;
    RWBufferSpan<MaterialBin> bins;
    RWBufferSpan<MaterialTile> tiles;
    RWBufferSpan<uint> arguments;
    RWBufferSpan<CompactStreamVisibleRecord> streamRecords;
    RWBufferSpan<StreamActiveGroup> streamGroups;
    uint width;
    uint height;
    uint tileCount;
    BufferSpan<uint> materialProgramBins;
    uint programBinCount;
    RWBufferSpan<uint> tileSummaries;
    RWBufferSpan<uint> tilePixelBins;
};

uint materialClassFromRecord(uint recordIndex, MaterialBinningParams push)
{
    uint count = push.streamRecords.length();
    if (recordIndex >= count) { return kMaterialClassBackground; }
    VisibleClusterRecord record = unpackStreamVisibleRecord(push.streamRecords[recordIndex]);
    if (record.dataIndex >= push.streamGroups.length()) { return kMaterialClassBackground; }
    record.instanceIndex = push.streamGroups[record.dataIndex].gpuSceneInstanceIndex;
    count = push.instances.length();
    if (record.instanceIndex >= count) { return kMaterialClassBackground; }
    uint material = push.instances[record.instanceIndex].identity.y;
    count = push.materials.length();
    if (material >= count) { return kMaterialClassBackground; }
    count = push.shadingMaterials.length();
    if (count == 0u) { return kMaterialClassBackground; }
#if METALLIC_MATERIAL_BINNING_MODE == 2
    // Single active foreground program bin: CPU validation guarantees every
    // valid shading material maps to bin 1.
    return 1u;
#elif METALLIC_MATERIAL_BINNING_MODE == 1
    uint sourceMaterial = push.materials[material].identity.z;
    // Match the deferred reconstruction's fallback material exactly.
    sourceMaterial = sourceMaterial < count ? sourceMaterial : 0u;
    // MaterialBinning::record validates every entry before uploading the table.
    return push.materialProgramBins.load(sourceMaterial);
#else
    uint sourceMaterial = push.materials[material].identity.z;
    // Match the deferred reconstruction's fallback material exactly.
    sourceMaterial = sourceMaterial < count ? sourceMaterial : 0u;
    return classifyDeferredMaterial(push.shadingMaterials[sourceMaterial]);
#endif
}

uint materialClassForPixel(uint2 pixel)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    uint recordIndex, triangleIndex;
    if (!unpackVisibilityId(resolveUniform(push.visibility).Load(int3(pixel, 0)), recordIndex, triangleIndex)) { return kMaterialClassBackground; }
    return materialClassFromRecord(recordIndex, push);
}

static const uint kMaterialBinSmemLimit = 64u;
groupshared uint gsMaterialBinCounts[kMaterialBinSmemLimit];

uint addMaterialBinCount(RWBufferSpan<MaterialBin> bins, uint bin)
{
    uint slot;
    resolveUniform(bins.resource).InterlockedAdd(bins.byteOffset + bin * sizeof(MaterialBin) + 4u, 1u, slot);
    return slot;
}

void emitAllocatedMaterialBinTile(RWBufferSpan<MaterialBin> bins, RWBufferSpan<uint> binOffsets, RWBufferSpan<MaterialTile> tiles, uint bin, uint mask, MaterialTile tile)
{
    if (WaveIsFirstLane() && mask != 0u) {
        uint slot = addMaterialBinCount(bins, bin);
        tile.laneMask = mask;
        tiles.store(binOffsets.load(bin) + slot, tile);
    }
}

struct MaterialBinningWaveTile
{
    uint tileIndex;
    uint2 pixel;
    bool valid;
    uint validMask;
};

MaterialBinningWaveTile loadMaterialBinningWaveTile(uint3 group, uint tid, uint lane, MaterialBinningParams push)
{
    uint waveIdx = tid >> 5u;
    uint columns = (push.width + kMaterialTileWidth - 1u) / kMaterialTileWidth;
    uint rows = (push.height + kMaterialTileHeight - 1u) / kMaterialTileHeight;
    uint tileCol = group.x * 4u + (waveIdx & 3u);
    uint tileRow = group.y * 2u + (waveIdx >> 2u);
    bool inBounds = (tileCol < columns) && (tileRow < rows);
    uint2 pixel = uint2(tileCol * kMaterialTileWidth + (lane & 7u), tileRow * kMaterialTileHeight + (lane >> 3u));
    bool valid = inBounds && all(pixel < uint2(push.width, push.height));
    MaterialBinningWaveTile tile;
    tile.tileIndex = tileRow * columns + tileCol;
    tile.pixel = pixel;
    tile.valid = valid;
    tile.validMask = WaveActiveBallot(valid).x;
    return tile;
}

void storePackedTilePixelBins(RWBufferSpan<uint> tilePixelBins, uint programBinCount, uint tileIndex, uint lane, uint materialClass, bool valid)
{
    uint laneBin = valid ? materialClass : 0u;
    if (programBinCount <= 16u) {
        uint packed = (laneBin & 0xfu) << ((lane & 7u) * 4u);
        packed |= WaveReadLaneAt(packed, lane ^ 1u);
        packed |= WaveReadLaneAt(packed, lane ^ 2u);
        packed |= WaveReadLaneAt(packed, lane ^ 4u);
        if ((lane & 7u) == 0u) {
            tilePixelBins.store(tileIndex * 4u + (lane >> 3u), packed);
        }
    } else if (programBinCount <= 256u) {
        uint packed = (laneBin & 0xffu) << ((lane & 3u) * 8u);
        packed |= WaveReadLaneAt(packed, lane ^ 1u);
        packed |= WaveReadLaneAt(packed, lane ^ 2u);
        if ((lane & 3u) == 0u) {
            tilePixelBins.store(tileIndex * 8u + (lane >> 2u), packed);
        }
    } else {
        uint oddBin = WaveReadLaneAt(laneBin, lane | 1u);
        if ((lane & 1u) == 0u) {
            tilePixelBins.store(tileIndex * 16u + (lane >> 1u), (laneBin & 0xffffu) | (oddBin << 16u));
        }
    }
}

uint loadPackedTilePixelBin(RWBufferSpan<uint> tilePixelBins, uint programBinCount, uint tileIndex, uint lane)
{
    if (programBinCount <= 16u) {
        uint packed = tilePixelBins.load(tileIndex * 4u + (lane >> 3u));
        return (packed >> ((lane & 7u) * 4u)) & 0xfu;
    }
    if (programBinCount <= 256u) {
        uint packed = tilePixelBins.load(tileIndex * 8u + (lane >> 2u));
        return (packed >> ((lane & 3u) * 8u)) & 0xffu;
    }
    uint packed = tilePixelBins.load(tileIndex * 16u + (lane >> 1u));
    return (lane & 1u) == 0u ? (packed & 0xffffu) : (packed >> 16u);
}

uint resolveWaveMaterialClass(uint2 pixel, bool valid, uint lane, MaterialBinningParams push)
{
    uint recordIndex = 0u;
    uint triangleIndex = 0u;
    bool hasSurface = valid && unpackVisibilityId(resolveUniform(push.visibility).Load(int3(pixel, 0)), recordIndex, triangleIndex);
    uint remaining = WaveActiveBallot(hasSurface).x;

    uint materialClass = kMaterialClassBackground;
    while (remaining != 0u) {
        uint sampleLane = firstbitlow(remaining);
        uint rec = WaveReadLaneAt(recordIndex, sampleLane);
        uint classForRec = materialClassFromRecord(rec, push);
        uint matchMask = WaveActiveBallot(hasSurface && recordIndex == rec).x;
        if ((matchMask & (1u << lane)) != 0u) {
            materialClass = classForRec;
        }
        remaining &= ~matchMask;
    }
    return materialClass;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningResetMain(uint3 id : SV_DispatchThreadID)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    RWBufferSpan<MaterialBin> bins = push.bins;
    uint lane = id.x;
    if (bins.contains(lane)) {
        bool directOffsets = push.programBinCount == 0u || push.programBinCount == 2u;
        MaterialBin bin = {directOffsets ? lane * push.tileCount : 0u, 0u};
        bins.store(lane, bin);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void materialBinningCountMain(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    RWBufferSpan<MaterialBin> bins = push.bins;
    RWBufferSpan<uint> tileSummaries = push.tileSummaries;
    RWBufferSpan<uint> tilePixelBins = push.tilePixelBins;
    if (tid < kMaterialBinSmemLimit) {
        gsMaterialBinCounts[tid] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint lane = WaveGetLaneIndex();
    MaterialBinningWaveTile waveTile = loadMaterialBinningWaveTile(group, tid, lane, push);
    if (waveTile.validMask != 0u) {
        uint materialClass = resolveWaveMaterialClass(waveTile.pixel, waveTile.valid, lane, push);
        uint firstType = WaveReadLaneAt(materialClass, firstbitlow(waveTile.validMask));
        uint firstMask = WaveActiveBallot(waveTile.valid && materialClass == firstType).x;
        if (firstMask == waveTile.validMask) {
            if (WaveIsFirstLane()) {
                if (firstType < kMaterialBinSmemLimit) {
                    InterlockedAdd(gsMaterialBinCounts[firstType], 1u);
                } else {
                    addMaterialBinCount(bins, firstType);
                }
                tileSummaries.store(waveTile.tileIndex, (firstType << 1u) | 1u);
            }
        } else {
            if (WaveIsFirstLane()) {
                if (firstType < kMaterialBinSmemLimit) {
                    InterlockedAdd(gsMaterialBinCounts[firstType], 1u);
                } else {
                    addMaterialBinCount(bins, firstType);
                }
                tileSummaries.store(waveTile.tileIndex, 0u);
            }
            uint remaining = waveTile.validMask & ~firstMask;
            while (remaining != 0u) {
                uint type = WaveReadLaneAt(materialClass, firstbitlow(remaining));
                uint mask = WaveActiveBallot(waveTile.valid && materialClass == type).x;
                if (WaveIsFirstLane()) {
                    if (type < kMaterialBinSmemLimit) {
                        InterlockedAdd(gsMaterialBinCounts[type], 1u);
                    } else {
                        addMaterialBinCount(bins, type);
                    }
                }
                remaining &= ~mask;
            }
            storePackedTilePixelBins(tilePixelBins, push.programBinCount, waveTile.tileIndex, lane, materialClass, waveTile.valid);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid < kMaterialBinSmemLimit) {
        uint count = gsMaterialBinCounts[tid];
        if (count != 0u) {
            uint ignoredSlot;
            resolveUniform(bins.resource).InterlockedAdd(bins.byteOffset + tid * sizeof(MaterialBin) + 4u, count, ignoredSlot);
        }
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void materialBinningAllocateMain()
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    RWBufferSpan<MaterialBin> bins = push.bins;
    RWBufferSpan<uint> arguments = push.arguments;
    uint offset = 0u;
    for (uint bin = 0u; bin < push.programBinCount; ++bin) {
        uint count = bins[bin].count;
        MaterialBin allocated = {offset, 0u};
        bins.store(bin, allocated);
        // Reuse the unused indirect-argument buffer during Classify so bin offset reads
        // do not contend with InterlockedAdd updates on bins[bin].count.
        arguments.store(bin, offset);
        offset += count;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void materialBinningClassifyMain(uint3 group : SV_GroupID, uint tid : SV_GroupIndex)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    RWBufferSpan<MaterialBin> bins = push.bins;
    RWBufferSpan<MaterialTile> tiles = push.tiles;
#if METALLIC_MATERIAL_BINNING_MODE == 1
    static const uint kFlushBinCount = kMaterialBinSmemLimit;
#elif METALLIC_MATERIAL_BINNING_MODE == 0
    static const uint kFlushBinCount = kMaterialClassCount;
#else
    static const uint kFlushBinCount = 2u;
#endif
    if (tid < kFlushBinCount) {
        gsMaterialBinCounts[tid] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint lane = WaveGetLaneIndex();
    MaterialBinningWaveTile waveTile = loadMaterialBinningWaveTile(group, tid, lane, push);
    bool myEmit = false;
    uint myBin = 0u;
    uint myMask = 0u;
    uint myLocalSlot = 0u;
#if METALLIC_MATERIAL_BINNING_MODE == 1
    RWBufferSpan<uint> binOffsets = push.arguments;
    if (waveTile.validMask != 0u) {
        uint summary = WaveReadLaneFirst(WaveIsFirstLane() ? push.tileSummaries.load(waveTile.tileIndex) : 0u);
        if ((summary & 1u) != 0u) {
            uint uniformBin = summary >> 1u;
            if (uniformBin < kMaterialBinSmemLimit) {
                if (WaveIsFirstLane()) {
                    InterlockedAdd(gsMaterialBinCounts[uniformBin], 1u, myLocalSlot);
                    myEmit = true;
                    myBin = uniformBin;
                    myMask = waveTile.validMask;
                }
            } else {
                MaterialTile tile = {waveTile.tileIndex, 0u};
                emitAllocatedMaterialBinTile(bins, binOffsets, tiles, uniformBin, waveTile.validMask, tile);
            }
        } else {
            uint materialClass = loadPackedTilePixelBin(push.tilePixelBins, push.programBinCount, waveTile.tileIndex, lane);
            uint remaining = waveTile.validMask;
            while (remaining != 0u) {
                uint leaderLane = firstbitlow(remaining);
                uint type = WaveReadLaneAt(materialClass, leaderLane);
                uint mask = WaveActiveBallot(waveTile.valid && materialClass == type).x;
                if (type < kMaterialBinSmemLimit) {
                    if (lane == leaderLane) {
                        InterlockedAdd(gsMaterialBinCounts[type], 1u, myLocalSlot);
                        myEmit = true;
                        myBin = type;
                        myMask = mask;
                    }
                } else {
                    MaterialTile tile = {waveTile.tileIndex, 0u};
                    emitAllocatedMaterialBinTile(bins, binOffsets, tiles, type, mask, tile);
                }
                remaining &= ~mask;
            }
        }
    }
#elif METALLIC_MATERIAL_BINNING_MODE == 0
    if (waveTile.validMask != 0u) {
        uint materialClass = resolveWaveMaterialClass(waveTile.pixel, waveTile.valid, lane, push);
        // Keep edge lanes participating in all ballots. Each class gets at most
        // one task per spatial tile, with a disjoint mask, regardless of material IDs.
        [unroll]
        for (uint type = 0u; type < kMaterialClassCount; ++type) {
            uint mask = WaveActiveBallot(waveTile.valid && materialClass == type).x;
            if (mask != 0u && lane == type) {
                InterlockedAdd(gsMaterialBinCounts[type], 1u, myLocalSlot);
                myEmit = true;
                myBin = type;
                myMask = mask;
            }
        }
    }
#else
    if (waveTile.validMask != 0u) {
#if METALLIC_MATERIAL_BINNING_MODE == 3
        uint recordIndex = 0u;
        uint triangleIndex = 0u;
        bool hasSurface = waveTile.valid &&
            unpackVisibilityId(resolveUniform(push.visibility).Load(int3(waveTile.pixel, 0)), recordIndex, triangleIndex);
        uint fgMask = WaveActiveBallot(hasSurface).x;
#else
        uint materialClass = resolveWaveMaterialClass(waveTile.pixel, waveTile.valid, lane, push);
        uint fgMask = WaveActiveBallot(waveTile.valid && materialClass != 0u).x;
#endif
        uint bgMask = waveTile.validMask & ~fgMask;
        uint mask = lane == 0u ? bgMask : fgMask;
        if (lane < 2u && mask != 0u) {
            InterlockedAdd(gsMaterialBinCounts[lane], 1u, myLocalSlot);
            myEmit = true;
            myBin = lane;
            myMask = mask;
        }
    }
#endif
    GroupMemoryBarrierWithGroupSync();
    if (tid < kFlushBinCount) {
        uint count = gsMaterialBinCounts[tid];
        if (count != 0u) {
            uint globalSlot;
            resolveUniform(bins.resource).InterlockedAdd(bins.byteOffset + tid * sizeof(MaterialBin) + 4u, count, globalSlot);
#if METALLIC_MATERIAL_BINNING_MODE == 1
            gsMaterialBinCounts[tid] = binOffsets.load(tid) + globalSlot;
#else
            gsMaterialBinCounts[tid] = tid * push.tileCount + globalSlot;
#endif
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (myEmit) {
        MaterialTile outTile = {waveTile.tileIndex, myMask};
        tiles.store(gsMaterialBinCounts[myBin] + myLocalSlot, outTile);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningArgumentsMain(uint3 id : SV_DispatchThreadID)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    RWBufferSpan<MaterialBin> bins = push.bins;
    RWBufferSpan<uint> arguments = push.arguments;
    uint lane = id.x;
    if (bins.contains(lane)) {
        // The producer allocates three argument words per bin and enough tile
        // slots for every emitted task, including sparse program bins.
        uint count = bins[lane].count;
        arguments.store(lane * 3u, min(count, kMaterialBinMaxGroupsX));
        arguments.store(lane * 3u + 1u, count / kMaterialBinMaxGroupsX + uint(count % kMaterialBinMaxGroupsX != 0u));
        arguments.store(lane * 3u + 2u, 1u);
    }
}
