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
    BufferSpan<uint> materialProgramBins;
    uint programBinCount;
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
    sourceMaterial = sourceMaterial < count ? sourceMaterial : 0u;
    if (push.programBinCount != 0u) {
        uint bin = push.materialProgramBins.load(sourceMaterial);
        return bin < push.programBinCount ? bin : 0u;
    }
    return classifyDeferredMaterial(gShadingMaterials[sourceMaterial]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningResetMain(uint3 id : SV_DispatchThreadID)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    uint lane = id.x;
    if (push.bins.contains(lane)) {
        MaterialBin bin = {push.programBinCount == 0u ? lane * push.tileCount : 0u, 0u};
        gBins.store(lane, bin);
    }
}

void classifyMaterialTile(uint3 group, bool scatter)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    uint lane = WaveGetLaneIndex();
    uint columns = (push.width + kMaterialTileWidth - 1u) / kMaterialTileWidth;
    MaterialTile tile = {group.y * columns + group.x, 0u};
    uint2 pixel = materialTilePixel(tile, lane, push.width);
    bool valid = all(pixel < uint2(push.width, push.height));
    uint materialClass = valid ? materialClassForPixel(pixel) : kMaterialClassBackground;
    if (push.programBinCount != 0u) {
        // Visit only the unique program IDs present in this wave, at most 32.
        // Every lane participates, including partial edge tiles.
        uint remaining = WaveActiveBallot(valid).x;
        while (remaining != 0u) {
            uint type = WaveReadLaneAt(materialClass, firstbitlow(remaining));
            uint mask = WaveActiveBallot(valid && materialClass == type).x;
            if (WaveIsFirstLane()) {
                uint slot;
                resolveUniform(gBins.resource).InterlockedAdd(gBins.byteOffset + type * sizeof(MaterialBin) + 4u, 1u, slot);
                if (scatter) {
                    tile.laneMask = mask;
                    gTiles.store(gBins[type].offset + slot, tile);
                }
            }
            remaining &= ~mask;
        }
        return;
    }
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
void materialBinningCountMain(uint3 group : SV_GroupID)
{
    classifyMaterialTile(group, false);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void materialBinningAllocateMain()
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    uint offset = 0u;
    for (uint bin = 0u; bin < push.programBinCount; ++bin) {
        uint count = gBins[bin].count;
        MaterialBin allocated = {offset, 0u};
        gBins.store(bin, allocated);
        offset += count;
    }
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningClassifyMain(uint3 group : SV_GroupID)
{
    classifyMaterialTile(group, true);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void materialBinningArgumentsMain(uint3 id : SV_DispatchThreadID)
{
    MaterialBinningParams push = getParameters<MaterialBinningParams>();
    uint lane = id.x;
    if (push.bins.contains(lane)) {
        if (push.arguments.length() < (lane + 1u) * 3u) { return; }
        uint offset = gBins[lane].offset;
        uint count = offset < push.tiles.length() ? min(gBins[lane].count, push.tiles.length() - offset) : 0u;
        gIndirect.store(lane * 3u, min(count, kMaterialBinMaxGroupsX));
        gIndirect.store(lane * 3u + 1u, count / kMaterialBinMaxGroupsX + uint(count % kMaterialBinMaxGroupsX != 0u));
        gIndirect.store(lane * 3u + 2u, 1u);
    }
}
