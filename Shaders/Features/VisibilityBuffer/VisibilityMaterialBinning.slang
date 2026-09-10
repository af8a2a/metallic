#include "../../Libraries/GPUDriven/GPUDrivenSceneCommon.slang"
#include "../../Libraries/GPUDriven/MaterialBinningCommon.slang"

[[vk::binding(0, 0)]] Texture2D<uint> gVisibility;
[[vk::binding(1, 0)]] StructuredBuffer<VisibleClusterRecord> gRecords;
[[vk::binding(2, 0)]] StructuredBuffer<GPUDrivenPreviewInstance> gInstances;
[[vk::binding(3, 0)]] StructuredBuffer<GPUDrivenPreviewMaterial> gMaterials;
[[vk::binding(4, 0)]] StructuredBuffer<PathTraceMaterial> gShadingMaterials;
[[vk::binding(5, 0)]] RWStructuredBuffer<MaterialBin> gBins;
[[vk::binding(6, 0)]] RWStructuredBuffer<MaterialTile> gTiles;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> gIndirect;

struct MaterialBinningPush
{
    uint width;
    uint height;
    uint tileCount;
    uint padding;
};

uint materialClassForPixel(uint2 pixel)
{
    uint recordIndex, triangleIndex;
    if (!unpackVisibilityId(gVisibility.Load(int3(pixel, 0)), recordIndex, triangleIndex)) { return kMaterialClassBackground; }
    uint count, stride;
    gRecords.GetDimensions(count, stride);
    if (recordIndex >= count) { return kMaterialClassBackground; }
    VisibleClusterRecord record = gRecords[recordIndex];
    if (visibleClusterSource(record) != kVisibleClusterSourceResident) { return kMaterialClassBackground; }
    gInstances.GetDimensions(count, stride);
    if (record.instanceIndex >= count) { return kMaterialClassBackground; }
    uint material = gInstances[record.instanceIndex].identity.y;
    gMaterials.GetDimensions(count, stride);
    if (material >= count) { return kMaterialClassBackground; }
    uint sourceMaterial = gMaterials[material].identity.z;
    gShadingMaterials.GetDimensions(count, stride);
    // Match the deferred reconstruction's fallback material exactly.
    return classifyDeferredMaterial(gShadingMaterials[sourceMaterial < count ? sourceMaterial : 0u]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningResetMain(uint lane : SV_GroupIndex, uniform MaterialBinningPush push)
{
    if (lane < kMaterialClassCount) {
        MaterialBin bin = {lane * push.tileCount, 0u};
        gBins[lane] = bin;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningClassifyMain(uint3 group : SV_GroupID, uniform MaterialBinningPush push)
{
    uint lane = WaveGetLaneIndex();
    uint columns = (push.width + kMaterialTileWidth - 1u) / kMaterialTileWidth;
    MaterialTile tile = {group.y * columns + group.x, 0u};
    uint2 pixel = materialTilePixel(tile, lane, push.width);
    bool valid = all(pixel < uint2(push.width, push.height));
    uint materialClass = valid ? materialClassForPixel(pixel) : kMaterialClassBackground;
    // Keep edge lanes participating in all ballots. Each class gets at most
    // one task per spatial tile, with a disjoint mask, regardless of material IDs.
    [unroll]
    for (uint type = 0u; type < kMaterialClassCount; ++type) {
        uint mask = WaveActiveBallot(valid && materialClass == type).x;
        if (WaveIsFirstLane() && mask != 0u) {
            uint slot;
            InterlockedAdd(gBins[type].count, 1u, slot);
            tile.laneMask = mask;
            gTiles[gBins[type].offset + slot] = tile;
        }
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningArgumentsMain(uint lane : SV_GroupIndex, uniform MaterialBinningPush push)
{
    if (lane < kMaterialClassCount) {
        uint count = gBins[lane].count;
        gIndirect[lane * 3u] = min(count, kMaterialBinMaxGroupsX);
        gIndirect[lane * 3u + 1u] = count / kMaterialBinMaxGroupsX + uint(count % kMaterialBinMaxGroupsX != 0u);
        gIndirect[lane * 3u + 2u] = 1u;
    }
}
