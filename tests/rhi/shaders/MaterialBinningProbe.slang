#if defined(PROBE_TYPED)
#include "../MaterialBinningProbeParameters.h"
#else
import Core;
#endif
using Metallic;
import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

#if defined(PROBE_TYPED)
[[vk::push_constant]] ConstantBuffer<MaterialProbeParams> gMaterialProbe;
#define gBins gMaterialProbe.bins.data
#define gTiles gMaterialProbe.tiles.data
#define gArguments gMaterialProbe.arguments.data
#define gResult gMaterialProbe.output.data
#else
#define gBins getResource<StructuredBuffer<MaterialBin>>(5)
#define gTiles getResource<StructuredBuffer<MaterialTile>>(6)
#define gArguments getResource<StructuredBuffer<uint>>(7)
#define gResult getResource<RWStructuredBuffer<uint>>(8)
#endif

struct ProbePush { uint width; uint height; uint binCount; uint bin; };
ProbePush probePush()
{
#if defined(PROBE_TYPED)
    MaterialProbeParams params = gMaterialProbe;
    return {params.width, params.height, params.binCount, params.bin};
#else
    return getConstants<ProbePush>();
#endif
}
static const uint kProbeHeader = 27u;



[shader("compute")]
[numthreads(64, 1, 1)]
void materialReadbackResetMain(uint3 id : SV_DispatchThreadID)
{
    ProbePush push = probePush();
    uint i = id.y * kMaterialBinMaxGroupsX * 64u + id.x;
    if (i == 0u) { gResult[kProbeHeader - 2u] = push.bin; gResult[kProbeHeader - 1u] = 0u; }
    if (i < push.binCount) {
        gResult[i * 5u] = gBins[i].offset;
        gResult[i * 5u + 1u] = gBins[i].count;
        for (uint c = 0u; c < 3u; ++c) { gResult[i * 5u + 2u + c] = gArguments[i * 3u + c]; }
    }
    if (i < push.width * push.height) {
        gResult[kProbeHeader + i * 3u] = 0xffffffffu;
        gResult[kProbeHeader + i * 3u + 1u] = 0u;
        gResult[kProbeHeader + i * 3u + 2u] = 0u;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialIndirectProbeMain(uint3 group : SV_GroupID)
{
    ProbePush push = probePush();
    MaterialBin bin = gBins[push.bin];
    uint index = group.y * kMaterialBinMaxGroupsX + group.x;
    if (index >= bin.count) { return; }
    MaterialTile tile = gTiles[bin.offset + index];
    uint lane = WaveGetLaneIndex();
    uint columns = (push.width + 7u) / 8u, rows = (push.height + 3u) / 4u;
    if (tile.tileIndex >= columns * rows || tile.laneMask == 0u) {
        if (WaveIsFirstLane()) { InterlockedAdd(gResult[kProbeHeader - 1u], 1u); }
        return;
    }
    if ((tile.laneMask & (1u << lane)) == 0u) { return; }
    uint2 pixel = materialTilePixel(tile, lane, push.width);
    if (any(pixel >= uint2(push.width, push.height))) { InterlockedAdd(gResult[kProbeHeader - 1u], 1u); return; }
    uint result = kProbeHeader + (pixel.y * push.width + pixel.x) * 3u;
    gResult[result] = push.bin;
    InterlockedAdd(gResult[result + 1u], 1u);
    gResult[result + 2u] = WaveGetLaneCount();
#if PROBE_ALTERNATE
    // Detect ignored permutation switches as well as the actual subgroup width.
    gResult[result + 2u] += 100u;
#endif
}
