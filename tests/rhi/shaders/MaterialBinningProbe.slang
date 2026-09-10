#include "../../../Shaders/Libraries/GPUDriven/GPUDrivenSceneCommon.slang"
#include "../../../Shaders/Libraries/GPUDriven/MaterialBinningCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<uint> gVisibility;
[[vk::binding(1, 0)]] RWStructuredBuffer<VisibleClusterRecord> gRecords;
[[vk::binding(2, 0)]] RWStructuredBuffer<GPUDrivenPreviewInstance> gInstances;
[[vk::binding(3, 0)]] RWStructuredBuffer<GPUDrivenPreviewMaterial> gMaterials;
[[vk::binding(4, 0)]] RWStructuredBuffer<PathTraceMaterial> gShadingMaterials;
[[vk::binding(5, 0)]] StructuredBuffer<MaterialBin> gBins;
[[vk::binding(6, 0)]] StructuredBuffer<MaterialTile> gTiles;
[[vk::binding(7, 0)]] StructuredBuffer<uint> gArguments;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> gResult;

struct ProbePush { uint width; uint height; uint binCount; uint bin; };
static const uint kProbeHeader = 27u;

[shader("compute")]
[numthreads(64, 1, 1)]
void materialFixtureMain(uint3 id : SV_DispatchThreadID, uniform ProbePush push)
{
    uint i = id.y * kMaterialBinMaxGroupsX * 64u + id.x;
    if (i < 260u) {
        VisibleClusterRecord record = {0u, i, 0u, 0u};
        if (i == 257u) { record.flags = 1u << 28u; }
        if (i == 258u) { record.instanceIndex = 10000u; }
        gRecords[i] = record;
        GPUDrivenPreviewInstance instance = {};
        instance.identity.y = i == 259u ? 10000u : (i * 17u) % 257u;
        gInstances[i] = instance;
    }
    if (i < 257u) {
        GPUDrivenPreviewMaterial material = {};
        material.identity.z = i == 0u ? 10000u : (i * 73u) % 257u;
        gMaterials[i] = material;
        // Change executable features while keeping source material IDs stable.
        uint type = 1u + (i + push.bin) % 4u;
        PathTraceMaterial shading = {};
        shading.params.x = type == 1u ? 0.0 : (type == 2u ? 1.0 : 0.5);
        shading.glassParams.x = type == 4u ? 1.0 : 0.0;
        shading.metallicRoughnessTexture.textureIndex = 0xffffffffu;
        shading.metallicRoughnessTexture.ntcTextureSetIndex = 0xffffffffu;
        if (i == 1u || i == 2u) {
            shading.params.x = 1.0;
            shading.glassParams.x = 0.0;
            if (i == 1u) { shading.metallicRoughnessTexture.textureIndex = 7u; }
            else { shading.metallicRoughnessTexture.ntcTextureSetIndex = 0u; }
        }
        gShadingMaterials[i] = shading;
    }
    if (i < push.width * push.height) {
        uint kind = i % 262u;
        // A dense tile list crosses Vulkan's guaranteed 65535-group X limit.
        if (push.width > 4096u) { kind = 0u; }
        uint record = kind < 257u ? kind : kind - 2u;
        uint visibility = ((record + 1u) << 7u) | (i % 127u);
        if (kind == 257u || push.width == 8u) { visibility = 0u; }
        if (kind == 258u) { visibility = 10001u << 7u; }
        gVisibility[uint2(i % push.width, i / push.width)] = visibility;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void materialReadbackResetMain(uint3 id : SV_DispatchThreadID, uniform ProbePush push)
{
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
void materialIndirectProbeMain(uint3 group : SV_GroupID, uniform ProbePush push)
{
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
