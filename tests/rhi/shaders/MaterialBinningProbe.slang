#include "../../../Shaders/Libraries/GPUDriven/GPUDrivenSceneCommon.slang"
#include "../../../Shaders/Libraries/GPUDriven/MaterialBinningCommon.slang"

[[vk::binding(0, 0)]] RWTexture2D<uint> gVisibility;
[[vk::binding(1, 0)]] RWStructuredBuffer<VisibleClusterRecord> gRecords;
[[vk::binding(2, 0)]] RWStructuredBuffer<GPUDrivenPreviewInstance> gInstances;
[[vk::binding(3, 0)]] RWStructuredBuffer<GPUDrivenPreviewMaterial> gMaterials;
[[vk::binding(4, 0)]] StructuredBuffer<MaterialBin> gBins;
[[vk::binding(5, 0)]] StructuredBuffer<uint> gPixels;
[[vk::binding(6, 0)]] StructuredBuffer<uint> gArguments;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> gResult;

struct ProbePush { uint width; uint height; uint binCount; uint bin; };

[shader("compute")]
[numthreads(64, 1, 1)]
void materialFixtureMain(uint3 id : SV_DispatchThreadID, uniform ProbePush push)
{
    uint i = id.x;
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
    }
    if (i < push.width * push.height) {
        uint kind = i % 262u;
        // A dense bin crosses Vulkan's guaranteed 65535-group X limit.
        if (push.width > 4096u) { kind = 0u; }
        uint record = kind < 257u ? kind : kind - 2u;
        uint visibility = ((record + 1u) << 7u) | (i % 127u);
        if (kind == 257u) { visibility = 0u; }
        if (kind == 258u) { visibility = 10001u << 7u; }
        gVisibility[uint2(i % push.width, i / push.width)] = visibility;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void materialReadbackResetMain(uint3 id : SV_DispatchThreadID, uniform ProbePush push)
{
    uint i = id.x;
    if (i < push.binCount) {
        gResult[i * 5u] = gBins[i].offset;
        gResult[i * 5u + 1u] = gBins[i].count;
        for (uint c = 0u; c < 3u; ++c) { gResult[i * 5u + 2u + c] = gArguments[i * 3u + c]; }
    }
    if (i < push.width * push.height) {
        gResult[push.binCount * 5u + i * 2u] = 0xffffffffu;
        gResult[push.binCount * 5u + i * 2u + 1u] = 0u;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void materialIndirectProbeMain(uint3 id : SV_DispatchThreadID, uniform ProbePush push)
{
    MaterialBin bin = gBins[push.bin];
    uint rowBase = id.y * kMaterialBinMaxGroupsX * kMaterialBinGroupSize;
    if (rowBase >= bin.count || id.x >= bin.count - rowBase) { return; }
    uint i = rowBase + id.x;
    uint pixel = gPixels[bin.offset + i];
    if (pixel >= push.width * push.height) { return; }
    uint result = push.binCount * 5u + pixel * 2u;
    gResult[result] = push.bin;
    InterlockedAdd(gResult[result + 1u], 1u);
}
