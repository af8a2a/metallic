#include "../../Libraries/Resources/ComputeResources.slang"

#include "../../Libraries/GPUDriven/GPUDrivenSceneCommon.slang"
#include "../../Libraries/GPUDriven/MaterialBinningCommon.slang"

#define gVisibility METALLIC_RESOURCE(Texture2D<uint>, 0)
#define gRecords METALLIC_RESOURCE(StructuredBuffer<VisibleClusterRecord>, 1)
#define gInstances METALLIC_RESOURCE(StructuredBuffer<GPUDrivenPreviewInstance>, 2)
#define gMaterials METALLIC_RESOURCE(StructuredBuffer<GPUDrivenPreviewMaterial>, 3)
#define gShadingMaterials METALLIC_RESOURCE(StructuredBuffer<PathTraceMaterial>, 4)
#define gBins METALLIC_RESOURCE(RWStructuredBuffer<MaterialBin>, 5)
#define gTiles METALLIC_RESOURCE(RWStructuredBuffer<MaterialTile>, 6)
#define gIndirect METALLIC_RESOURCE(RWStructuredBuffer<uint>, 7)

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
void materialBinningResetMain(uint lane : SV_GroupIndex)
{
    MaterialBinningPush push = METALLIC_CONSTANTS(MaterialBinningPush);
    if (lane < kMaterialClassCount) {
        MaterialBin bin = {lane * push.tileCount, 0u};
        gBins[lane] = bin;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningClassifyMain(uint3 group : SV_GroupID)
{
    MaterialBinningPush push = METALLIC_CONSTANTS(MaterialBinningPush);
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
void materialBinningArgumentsMain(uint lane : SV_GroupIndex)
{
    MaterialBinningPush push = METALLIC_CONSTANTS(MaterialBinningPush);
    if (lane < kMaterialClassCount) {
        uint count = gBins[lane].count;
        gIndirect[lane * 3u] = min(count, kMaterialBinMaxGroupsX);
        gIndirect[lane * 3u + 1u] = count / kMaterialBinMaxGroupsX + uint(count % kMaterialBinMaxGroupsX != 0u);
        gIndirect[lane * 3u + 2u] = 1u;
    }
}
