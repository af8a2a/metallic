// Diagnostic replay only. No visibility/depth writes and no changes to production
// raster kernels. Counts coverage and attempted pixel atomics, not winning writes.
#include "GPUDrivenStreamAsset.slang"

#include "StreamRasterWorkload.slang"

[shader("compute")]
[numthreads(16, 1, 1)]
void streamWorkloadResetMain(uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<RWStructuredBuffer<uint64_t>> counters =
        DescriptorHandle<RWStructuredBuffer<uint64_t>>(uint2(push.hybridQueueBuffer, 0));
    counters[lane] = 0;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void streamWorkloadMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<uint>> bins =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.hybridClusterBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint64_t>> handle =
        DescriptorHandle<RWStructuredBuffer<uint64_t>>(uint2(push.hybridQueueBuffer, 0));
    RWStructuredBuffer<uint64_t> counters = handle;
    uint record;
    if (!hybridBinRecord(bins, kHybridSoftwareBin, hybridFlatGroup(group), record)) { return; }
    if (lane < WaveGetLaneCount()) {
        bool valid = loadStreamSoftwareWave(record, push);
        if (lane == 0) { g_StreamSoftwareValid = valid ? 1u : 0u; }
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamSoftwareValid == 0) {
        if (lane == 0) { InterlockedAdd(counters[15], uint64_t(1)); }
        return;
    }
    if (lane < g_StreamSoftware.vertexCount) {
        HybridScreenVertex vertex = hybridPrepareVertex(streamSoftwareClip(lane, push), float2(bins[6], bins[7]), bins[10]);
        g_StreamRasterWords[lane] = uint4(uint2(vertex.position), asuint(vertex.depth), 0);
    }
    GroupMemoryBarrierWithGroupSync();
    uint64_t4 work = uint64_t4(0);
    bool active = lane < g_StreamSoftware.triangleCount;
    if (active) {
        uint3 tri = streamSoftwareTriangle(lane, push);
        work = workloadTriangle(asint(g_StreamRasterWords[tri.x].xy), asint(g_StreamRasterWords[tri.y].xy),
            asint(g_StreamRasterWords[tri.z].xy), (g_StreamSoftware.instanceFlags & 2u) != 0,
            bins[6], bins[7], bins[10]);
    }
    uint64_t sums[14];
    sums[0] = lane == 0 ? 1 : 0; // valid clusters
    sums[1] = active ? 1 : 0; // submitted triangles, including empty/backface
    sums[2] = lane == 0 ? g_StreamSoftware.vertexCount : 0;
    sums[3] = work.x != 0 ? 1 : 0;
    sums[4] = work.x; // bbox sample visits
    sums[5] = work.y; // covered samples, before depth arbitration
    sums[6] = work.y; // atomic attempts, including depth losers
    sums[7] = work.z;
    sums[8] = active && work.x == 0 ? 1 : 0;
    uint64_t maxRows = WaveActiveMax(uint(work.z));
    sums[9] = WaveIsFirstLane() ? maxRows : 0;
    sums[10] = active ? 1 : 0;
    sums[11] = 1; // launched triangle lanes
    sums[12] = work.x > 0 && work.x <= 4 ? 1 : 0;
    sums[13] = work.x > 4 && work.x <= 16 ? 1 : 0;
    for (uint i = 0; i < 14; ++i) {
        // 16-bit limbs keep each subgroup sum within uint32 even on 128 lanes.
        // Do not require shaderSubgroupExtendedTypes just for diagnostics.
        uint64_t total = uint64_t(WaveActiveSum(uint(sums[i] & 65535u))) +
            (uint64_t(WaveActiveSum(uint((sums[i] >> 16u) & 65535u))) << 16u) +
            (uint64_t(WaveActiveSum(uint((sums[i] >> 32u) & 65535u))) << 32u) +
            (uint64_t(WaveActiveSum(uint(sums[i] >> 48u))) << 48u);
        if (WaveIsFirstLane()) { InterlockedAdd(counters[i], total); }
    }
    uint64_t maxWidth = WaveActiveMax(uint(work.w));
    if (WaveIsFirstLane()) { InterlockedMax(counters[14], maxWidth); }
}
