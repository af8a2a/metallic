import Core;
import ParameterRoot;
using Metallic;
import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

#include "VisibilityStreamDecode.slang"

// Wire layout matches MaterialBinningParams.h, verified by GPU producer/consumer tests.
struct MaterialBinningParams
{
    ResourceHandle<Texture2D<uint>> visibility;
    RWBufferSpan<VisibleClusterRecord> records;
    RWBufferSpan<GPUDrivenPreviewInstance> instances;
    RWBufferSpan<GPUDrivenPreviewMaterial> materials;
    RWBufferSpan<PathTraceMaterial> shadingMaterials;
    RWBufferSpan<MaterialBin> bins;
    RWBufferSpan<MaterialTile> tiles;
    RWBufferSpan<uint> arguments;
    RWBufferSpan<CompactStreamVisibleRecord> streamRecords;
    RWBufferSpan<StreamActiveGroup> streamGroups;
    uint width;
    uint height;
    uint tileCount;
    uint residentRecordCount;
};

#define gVisibility resolveUniform(getParameters<MaterialBinningParams>().visibility)
#define gRecords getParameters<MaterialBinningParams>().records
#define gInstances getParameters<MaterialBinningParams>().instances
#define gMaterials getParameters<MaterialBinningParams>().materials
#define gShadingMaterials getParameters<MaterialBinningParams>().shadingMaterials
#define gBins getParameters<MaterialBinningParams>().bins
#define gTiles getParameters<MaterialBinningParams>().tiles
#define gIndirect getParameters<MaterialBinningParams>().arguments
#define gStreamRecords getParameters<MaterialBinningParams>().streamRecords

uint materialClassForPixel(uint2 pixel)
{
    uint recordIndex, triangleIndex;
    if (!unpackVisibilityId(gVisibility.Load(int3(pixel, 0)), recordIndex, triangleIndex)) { return kMaterialClassBackground; }
    uint count;
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    VisibleClusterRecord record;
    if (recordIndex < push.residentRecordCount) {
        count = push.records.length();
        if (recordIndex >= count) { return kMaterialClassBackground; }
        record = gRecords[recordIndex];
        if (visibleClusterSource(record) != kVisibleClusterSourceResident) { return kMaterialClassBackground; }
    } else {
        recordIndex -= push.residentRecordCount;
        count = push.streamRecords.length();
        if (recordIndex >= count) { return kMaterialClassBackground; }
        record = unpackStreamVisibleRecord(gStreamRecords[recordIndex]);
        if (record.dataIndex >= push.streamGroups.length()) { return kMaterialClassBackground; }
        record.instanceIndex = push.streamGroups[record.dataIndex].gpuSceneInstanceIndex;
        if (visibleClusterSource(record) != kVisibleClusterSourceStreamPage) { return kMaterialClassBackground; }
    }
    count = push.instances.length();
    if (record.instanceIndex >= count) { return kMaterialClassBackground; }
    uint material = gInstances[record.instanceIndex].identity.y;
    count = push.materials.length();
    if (material >= count) { return kMaterialClassBackground; }
    uint sourceMaterial = gMaterials[material].identity.z;
    count = push.shadingMaterials.length();
    if (count == 0u) { return kMaterialClassBackground; }
    // Match the deferred reconstruction's fallback material exactly.
    return classifyDeferredMaterial(gShadingMaterials[sourceMaterial < count ? sourceMaterial : 0u]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningResetMain(uint lane : SV_GroupIndex)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    if (lane < kMaterialClassCount && push.bins.contains(lane)) {
        MaterialBin bin = {lane * push.tileCount, 0u};
        gBins.store(lane, bin);
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningClassifyMain(uint3 group : SV_GroupID)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
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
        if (WaveIsFirstLane() && mask != 0u && push.bins.contains(type)) {
            uint slot;
            resolveUniform(gBins.resource).InterlockedAdd(gBins.byteOffset + type * sizeof(MaterialBin) + 4u, 1u, slot);
            tile.laneMask = mask;
            uint offset = gBins[type].offset;
            if (offset < push.tiles.length() && slot < push.tiles.length() - offset) {
                gTiles.store(offset + slot, tile);
            }
        }
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningArgumentsMain(uint lane : SV_GroupIndex)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    if (lane < kMaterialClassCount && push.bins.contains(lane)) {
        if (push.arguments.length() < (lane + 1u) * 3u) { return; }
        uint offset = gBins[lane].offset;
        uint count = offset < push.tiles.length() ? min(gBins[lane].count, push.tiles.length() - offset) : 0u;
        gIndirect.store(lane * 3u, min(count, kMaterialBinMaxGroupsX));
        gIndirect.store(lane * 3u + 1u, count / kMaterialBinMaxGroupsX + uint(count % kMaterialBinMaxGroupsX != 0u));
        gIndirect.store(lane * 3u + 2u, 1u);
    }
}
