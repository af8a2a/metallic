import Core;
using Metallic;
import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

#define gVisibility getResource<Texture2D<uint>>(0)
#define gRecords getResource<StructuredBuffer<VisibleClusterRecord>>(1)
#define gInstances getResource<StructuredBuffer<GPUDrivenPreviewInstance>>(2)
#define gMaterials getResource<StructuredBuffer<GPUDrivenPreviewMaterial>>(3)
#define gShadingMaterials getResource<StructuredBuffer<PathTraceMaterial>>(4)
#define gBins getResource<RWStructuredBuffer<MaterialBin>>(5)
#define gTiles getResource<RWStructuredBuffer<MaterialTile>>(6)
#define gIndirect getResource<RWStructuredBuffer<uint>>(7)
#define gStreamRecords getResource<StructuredBuffer<VisibleClusterRecord>>(8)

struct MaterialBinningPush
{
    uint width;
    uint height;
    uint tileCount;
    uint residentRecordCount;
};

uint materialClassForPixel(uint2 pixel)
{
    uint recordIndex, triangleIndex;
    if (!unpackVisibilityId(gVisibility.Load(int3(pixel, 0)), recordIndex, triangleIndex)) { return kMaterialClassBackground; }
    uint count, stride;
    MaterialBinningPush push = getConstants<MaterialBinningPush>();
    VisibleClusterRecord record;
    if (recordIndex < push.residentRecordCount) {
        gRecords.GetDimensions(count, stride);
        if (recordIndex >= count) { return kMaterialClassBackground; }
        record = gRecords[recordIndex];
        if (visibleClusterSource(record) != kVisibleClusterSourceResident) { return kMaterialClassBackground; }
    } else {
        recordIndex -= push.residentRecordCount;
        gStreamRecords.GetDimensions(count, stride);
        if (recordIndex >= count) { return kMaterialClassBackground; }
        record = gStreamRecords[recordIndex];
        if (visibleClusterSource(record) != kVisibleClusterSourceStreamPage) { return kMaterialClassBackground; }
    }
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
    MaterialBinningPush push = getConstants<MaterialBinningPush>();
    if (lane < kMaterialClassCount) {
        MaterialBin bin = {lane * push.tileCount, 0u};
        gBins[lane] = bin;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningClassifyMain(uint3 group : SV_GroupID)
{
    MaterialBinningPush push = getConstants<MaterialBinningPush>();
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
    MaterialBinningPush push = getConstants<MaterialBinningPush>();
    if (lane < kMaterialClassCount) {
        uint count = gBins[lane].count;
        gIndirect[lane * 3u] = min(count, kMaterialBinMaxGroupsX);
        gIndirect[lane * 3u + 1u] = count / kMaterialBinMaxGroupsX + uint(count % kMaterialBinMaxGroupsX != 0u);
        gIndirect[lane * 3u + 2u] = 1u;
    }
}
