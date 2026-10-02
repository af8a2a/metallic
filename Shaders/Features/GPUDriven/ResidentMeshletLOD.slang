import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;

#include "../../../Source/Runtime/Render/Core/ResidentLODParameters.h"
[[vk::push_constant]] ConstantBuffer<ResidentLODParameters> gResidentLOD;

bool residentLodRangesValid(ResidentLODParameters p)
{
    uint blocks = p.count / 64u + uint((p.count & 63u) != 0u);
    return p.count <= p.capacity && p.offset <= p.records.length() &&
        p.count <= p.records.length() - p.offset && p.output.length() > p.capacity &&
        p.arguments.length() >= 9u && p.capacity <= p.scratch.length() &&
        blocks <= p.scratch.length() - p.capacity;
}

groupshared uint g_LodPrefix[128];

[shader("compute")]
[numthreads(1, 1, 1)]
void residentLodResetMain()
{
    ResidentLODParameters push = gResidentLOD;
    if (!residentLodRangesValid(push)) { return; }
    push.output.data[0] = uint4(0, push.capacity, push.count, 0);
}

bool residentLodSelected(uint index, ResidentLODParameters push)
{
    if (index >= push.count) { return false; }
    VisibleClusterRecord record = push.records.data[push.offset + index];
    if (record.instanceIndex >= push.instanceCount || !push.instances.contains(record.instanceIndex) || !push.clusters.contains(record.clusterIndex)) { return false; }
    GPUDrivenPreviewInstance instance = push.instances.data[record.instanceIndex];
    if ((instance.identity.w & 1u) == 0u) { return false; }
    GPUDrivenPreviewMeshlet cluster = push.clusters.data[record.clusterIndex];
    uint owner = cluster.lod.y, refined = cluster.lod.z;
    if (owner != 0xffffffffu) {
        if (owner >= push.groupCount || !push.groups.contains(owner) || !meshletLodNeedsFine(push.groups.data[owner], instance,
            push.eye, push.forward, push.projection, push.manualLevel)) { return false; }
        if (refined != 0xffffffffu && (refined >= push.groupCount || !push.groups.contains(refined) ||
            meshletLodNeedsFine(push.groups.data[refined], instance, push.eye, push.forward, push.projection, push.manualLevel))) { return false; }
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void residentLodSelectMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    ResidentLODParameters push = gResidentLOD;
    if (!residentLodRangesValid(push)) { return; }
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
    if (index < push.count) { push.scratch.data[index] = selected ? g_LodPrefix[lane] - 1u : 0xffffffffu; }
    if (lane == 63u) { push.scratch.data[push.capacity + block] = g_LodPrefix[lane]; }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void residentLodArgumentsMain(uint lane : SV_GroupIndex)
{
    ResidentLODParameters push = gResidentLOD;
    if (!residentLodRangesValid(push)) { return; }
    uint blocks = (push.count + 63u) / 64u;
    uint blocksPerLane = (blocks + 127u) / 128u;
    uint first = min(lane * blocksPerLane, blocks);
    uint end = min(first + blocksPerLane, blocks);
    uint total = 0u;
    for (uint block = first; block < end; ++block) { total += push.scratch.data[push.capacity + block]; }
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
        uint count = push.scratch.data[push.capacity + block];
        push.scratch.data[push.capacity + block] = offset;
        offset += count;
    }
    if (lane != 127u) { return; }
    uint count = g_LodPrefix[127];
    push.output.data[0] = uint4(count, push.capacity, push.count, 0);
    for (uint i = 0u; i < 3u; ++i) {
        uint groups = i == 0u ? count : (i == 1u ? (count + 31u) / 32u : count * 8u);
        push.arguments.data[i * 3u] = min(groups, 65535u);
        push.arguments.data[i * 3u + 1u] = max(1u, (groups + 65534u) / 65535u);
        push.arguments.data[i * 3u + 2u] = 1u;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void residentLodScatterMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    ResidentLODParameters push = gResidentLOD;
    if (!residentLodRangesValid(push)) { return; }
    uint block = groupID.x + groupID.y * 65535u;
    uint index = block * 64u + lane;
    if (index >= push.count) { return; }
    uint rank = push.scratch.data[index];
    if (rank == 0xffffffffu) { return; }
    uint destination = push.scratch.data[push.capacity + block] + rank;
    if (destination >= push.capacity) { return; }
    VisibleClusterRecord record = push.records.data[push.offset + index];
    push.output.data[destination + 1u] = uint4(record.instanceIndex, record.clusterIndex, push.offset + index, record.dataIndex);
}
