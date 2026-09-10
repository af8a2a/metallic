#include "../../Libraries/GPUDriven/GPUDrivenSceneCommon.slang"
#include "../../Libraries/GPUDriven/MaterialBinningCommon.slang"

[[vk::binding(0, 0)]] Texture2D<uint> gVisibility;
[[vk::binding(1, 0)]] StructuredBuffer<VisibleClusterRecord> gRecords;
[[vk::binding(2, 0)]] StructuredBuffer<GPUDrivenPreviewInstance> gInstances;
[[vk::binding(3, 0)]] StructuredBuffer<GPUDrivenPreviewMaterial> gMaterials;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> gCounters;
[[vk::binding(5, 0)]] RWStructuredBuffer<MaterialBin> gBins;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> gPixels;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> gIndirect;

struct MaterialBinningPush
{
    uint width;
    uint height;
    uint binCount; // background + source material count, including fallback
    uint padding;
};

uint materialBinForPixel(uint2 pixel, uint binCount)
{
    uint recordIndex, triangleIndex;
    if (!unpackVisibilityId(gVisibility.Load(int3(pixel, 0)), recordIndex, triangleIndex)) { return 0u; }
    uint count, stride;
    gRecords.GetDimensions(count, stride);
    if (recordIndex >= count) { return 0u; }
    VisibleClusterRecord record = gRecords[recordIndex];
    if (visibleClusterSource(record) != kVisibleClusterSourceResident) { return 0u; }
    gInstances.GetDimensions(count, stride);
    if (record.instanceIndex >= count) { return 0u; }
    uint material = gInstances[record.instanceIndex].identity.y;
    gMaterials.GetDimensions(count, stride);
    if (material >= count) { return 0u; }
    uint sourceMaterial = gMaterials[material].identity.z;
    return sourceMaterial < binCount - 1u ? sourceMaterial + 1u : 1u;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void materialBinningResetMain(uint3 id : SV_DispatchThreadID, uniform MaterialBinningPush push)
{
    if (id.x < push.binCount) { gCounters[id.x] = 0u; }
}

void binPixel(uint2 pixel, MaterialBinningPush push, bool scatter)
{
    uint bin = materialBinForPixel(pixel, push.binCount);
    // Peel one distinct material from the active wave at a time. The loop's
    // active mask excludes lanes already appended; no wave-size assumption or
    // vendor-specific WaveMatch/partitioned-subgroup extension is required.
    bool remaining = true;
    while (remaining) {
        uint selectedBin = WaveReadLaneFirst(bin);
        bool selected = bin == selectedBin;
        uint count = WaveActiveCountBits(selected);
        uint rank = WavePrefixCountBits(selected);
        uint base = 0u;
        if (WaveIsFirstLane()) { InterlockedAdd(gCounters[selectedBin], count, base); }
        base = WaveReadLaneFirst(base);
        if (selected) {
            if (scatter) {
                gPixels[gBins[bin].offset + base + rank] = pixel.y * push.width + pixel.x;
            }
            remaining = false;
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void materialBinningCountMain(uint3 id : SV_DispatchThreadID, uniform MaterialBinningPush push)
{
    if (any(id.xy >= uint2(push.width, push.height))) { return; }
    binPixel(id.xy, push, false);
}

groupshared uint gAllocationEnd;

[shader("compute")]
[numthreads(128, 1, 1)]
void materialBinningAllocateMain(uint lane : SV_GroupIndex, uniform MaterialBinningPush push)
{
    if (lane == 0u) { gAllocationEnd = 0u; }
    GroupMemoryBarrierWithGroupSync();
    for (uint block = 0u; block < push.binCount; block += 128u) {
        uint bin = block + lane;
        uint count = bin < push.binCount ? gCounters[bin] : 0u;
        uint prefix = WavePrefixSum(count);
        uint total = WaveActiveSum(count);
        uint base = 0u;
        if (WaveIsFirstLane()) { InterlockedAdd(gAllocationEnd, total, base); }
        base = WaveReadLaneFirst(base);
        if (bin < push.binCount) {
            // Offsets need not be in material-ID order. Wave reservations are
            // disjoint and together consume exactly width * height uints.
            MaterialBin metadata = {base + prefix, count};
            gBins[bin] = metadata;
            uint groups = count / kMaterialBinGroupSize + uint(count % kMaterialBinGroupSize != 0u);
            gIndirect[bin * 3u] = min(groups, kMaterialBinMaxGroupsX);
            gIndirect[bin * 3u + 1u] = groups / kMaterialBinMaxGroupsX + uint(groups % kMaterialBinMaxGroupsX != 0u);
            gIndirect[bin * 3u + 2u] = 1u;
            gCounters[bin] = 0u;
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void materialBinningScatterMain(uint3 id : SV_DispatchThreadID, uniform MaterialBinningPush push)
{
    if (any(id.xy >= uint2(push.width, push.height))) { return; }
    binPixel(id.xy, push, true);
}
