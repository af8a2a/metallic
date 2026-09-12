#include "../../Libraries/GPUDriven/MeshletLod.slang"

struct LodPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint padding0, padding1;
    float4 eye;
    float4 forward;
    float4 projection;
    uint clusters, records, instances, groups;
    uint output, arguments, offset, count;
    uint capacity, instanceCount, groupCount, manualLevel;
    uint scratch, padding2;
};

groupshared uint g_LodPrefix[128];

[shader("compute")]
[numthreads(1, 1, 1)]
void residentLodResetMain(uniform LodPush push)
{
    DescriptorHandle<RWStructuredBuffer<uint4>> output =
        DescriptorHandle<RWStructuredBuffer<uint4>>(uint2(push.bufferShaderIndexBase + push.output, 0));
    output[0] = uint4(0, push.capacity, push.count, 0);
}

bool residentLodSelected(uint index, LodPush push)
{
    if (index >= push.count) { return false; }
    DescriptorHandle<StructuredBuffer<VisibleClusterRecord>> records =
        DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>(uint2(push.bufferShaderIndexBase + push.records, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>> clusters =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>(uint2(push.bufferShaderIndexBase + push.clusters, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>> instances =
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(uint2(push.bufferShaderIndexBase + push.instances, 0));
    DescriptorHandle<StructuredBuffer<MeshletLodGroupRecord>> groups =
        DescriptorHandle<StructuredBuffer<MeshletLodGroupRecord>>(uint2(push.bufferShaderIndexBase + push.groups, 0));
    VisibleClusterRecord record = records[push.offset + index];
    if (record.instanceIndex >= push.instanceCount) { return false; }
    GPUDrivenPreviewInstance instance = instances[record.instanceIndex];
    if ((instance.identity.w & 1u) == 0u) { return false; }
    GPUDrivenPreviewMeshlet cluster = clusters[record.clusterIndex];
    uint owner = cluster.lod.y, refined = cluster.lod.z;
    if (owner != 0xffffffffu) {
        if (owner >= push.groupCount || !meshletLodNeedsFine(groups[owner], instance,
            push.eye, push.forward, push.projection, push.manualLevel)) { return false; }
        if (refined != 0xffffffffu && (refined >= push.groupCount ||
            meshletLodNeedsFine(groups[refined], instance, push.eye, push.forward, push.projection, push.manualLevel))) { return false; }
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void residentLodSelectMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex, uniform LodPush push)
{
    uint block = groupID.x + groupID.y * 65535u;
    if (block >= (push.count + 63u) / 64u) { return; }
    uint index = block * 64u + lane;
    bool selected = residentLodSelected(index, push);
    g_LodPrefix[lane] = selected ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 64u; stride *= 2u) {
        uint previous = lane >= stride ? g_LodPrefix[max(lane, stride) - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        g_LodPrefix[lane] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    DescriptorHandle<RWStructuredBuffer<uint>> scratch =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.scratch, 0));
    if (index < push.count) { scratch[index] = selected ? g_LodPrefix[lane] - 1u : 0xffffffffu; }
    if (lane == 63u) { scratch[push.capacity + block] = g_LodPrefix[lane]; }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void residentLodArgumentsMain(uint lane : SV_GroupIndex, uniform LodPush push)
{
    DescriptorHandle<RWStructuredBuffer<uint>> scratch =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.scratch, 0));
    uint blocks = (push.count + 63u) / 64u;
    uint blocksPerLane = (blocks + 127u) / 128u;
    uint first = min(lane * blocksPerLane, blocks);
    uint end = min(first + blocksPerLane, blocks);
    uint total = 0u;
    for (uint block = first; block < end; ++block) { total += scratch[push.capacity + block]; }
    g_LodPrefix[lane] = total;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 128u; stride *= 2u) {
        uint previous = lane >= stride ? g_LodPrefix[max(lane, stride) - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        g_LodPrefix[lane] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    uint offset = lane != 0u ? g_LodPrefix[max(lane, 1u) - 1u] : 0u;
    for (uint block = first; block < end; ++block) {
        uint count = scratch[push.capacity + block];
        scratch[push.capacity + block] = offset;
        offset += count;
    }
    if (lane != 127u) { return; }
    DescriptorHandle<RWStructuredBuffer<uint4>> output =
        DescriptorHandle<RWStructuredBuffer<uint4>>(uint2(push.bufferShaderIndexBase + push.output, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> arguments =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.arguments, 0));
    uint count = g_LodPrefix[127];
    output[0] = uint4(count, push.capacity, push.count, 0);
    for (uint i = 0u; i < 2u; ++i) {
        uint groups = i == 0u ? count : (count + 31u) / 32u;
        arguments[i * 3u] = min(groups, 65535u);
        arguments[i * 3u + 1u] = max(1u, (groups + 65534u) / 65535u);
        arguments[i * 3u + 2u] = 1u;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void residentLodScatterMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex, uniform LodPush push)
{
    uint block = groupID.x + groupID.y * 65535u;
    uint index = block * 64u + lane;
    if (index >= push.count) { return; }
    DescriptorHandle<StructuredBuffer<uint>> scratch =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.scratch, 0));
    uint rank = scratch[index];
    if (rank == 0xffffffffu) { return; }
    uint destination = scratch[push.capacity + block] + rank;
    DescriptorHandle<StructuredBuffer<VisibleClusterRecord>> records =
        DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>(uint2(push.bufferShaderIndexBase + push.records, 0));
    DescriptorHandle<RWStructuredBuffer<uint4>> output =
        DescriptorHandle<RWStructuredBuffer<uint4>>(uint2(push.bufferShaderIndexBase + push.output, 0));
    VisibleClusterRecord record = records[push.offset + index];
    output[destination + 1u] = uint4(record.instanceIndex, record.clusterIndex, push.offset + index, record.dataIndex);
}
