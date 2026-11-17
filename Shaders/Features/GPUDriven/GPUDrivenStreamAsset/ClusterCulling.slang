#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Cluster candidate scans, visibility culling and hybrid bin admission.
public groupshared uint g_StreamCandidatePrefix[128];

uint streamPhaseCandidateMask(GPUDrivenStreamAssetActiveGroup group, uint index,
    RWStructuredBuffer<uint> bins, StructuredBuffer<uint> visibility, bool early)
{
    uint state = visibility[group.gpuSceneInstanceIndex];
    if (early) {
        bins[hybridRetryBase(bins[5]) + index] = 0u;
        return state == 1u ? group.clusterSelectionMask : 0u;
    }
    if (state == 3u) { return group.clusterSelectionMask; }
    return state == 1u ? group.clusterSelectionMask & bins[hybridRetryBase(bins[5]) + index] : 0u;
}

uint streamCandidateInclusiveScan(uint value, uint lane)
{
    g_StreamCandidatePrefix[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 128u; stride *= 2u) {
        uint previous = lane >= stride ? g_StreamCandidatePrefix[max(lane, stride) - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        g_StreamCandidatePrefix[lane] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    return g_StreamCandidatePrefix[lane];
}

// Setup/count/prefix/scatter use activeBuildPhase, independently of the
// early/late traversalPhase. Count and scatter scale with the actual active
// group count, not the capacity. Original group/cluster order and IDs survive.
// Before classification, bin 0 stores group masks, rank storage holds each
// group's block-local offset, and block storage holds counts/global offsets.
// All three regions are dead until stable binning, which overwrites them.
[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterPrepareMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> headers =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(push.activeHeaderBuffer));
    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> groups =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(push.activeGroupBuffer));
    RWStructuredBuffer<uint> bins =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.hybridClusterBuffer));
    RWStructuredBuffer<uint> args =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.hybridQueueBuffer));
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));
    StructuredBuffer<uint> visibility =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(bindingHandle[0].instanceVisibilityBuffer));
    uint capacity = bins[5];
    if (push.activeBuildPhase == 0u) {
        if (lane == 0u) {
            GPUDrivenStreamAssetActiveHeader header = headers[0];
            // The stream producer guarantees activeGroupCapacity <= capacity.
            uint count = min(header.activeGroupCount, header.activeGroupCapacity);
            bins[13] = header.maxActiveGroupClusters;
            bins[15] = count;
            uint blocks = (count + 127u) / 128u;
            args[6] = min(blocks, kHybridDispatchWidth);
            args[7] = max(1u, (blocks + kHybridDispatchWidth - 1u) / kHybridDispatchWidth);
            args[8] = 1u;
        }
        return;
    }

    uint count = bins[15];
    uint blocks = (count + 127u) / 128u;
    uint blockBase = hybridBlockBase(capacity);
    if (push.activeBuildPhase == 2u) {
        uint perLane = (blocks + 127u) / 128u;
        uint first = min(lane * perLane, blocks), end = min(first + perLane, blocks);
        uint total = 0u;
        for (uint block = first; block < end; ++block) { total += bins[blockBase + block]; }
        uint offset = streamCandidateInclusiveScan(total, lane) - total;
        for (uint block = first; block < end; ++block) {
            uint blockCount = bins[blockBase + block];
            bins[blockBase + block] = offset;
            offset += blockCount;
        }
        if (lane == 127u) {
            uint totalCandidates = g_StreamCandidatePrefix[127];
            // Never partially render a cut. Keep the phase masks and route the
            // entire stage to HW when the independent candidate budget is exceeded.
            uint candidates = totalCandidates <= capacity ? totalCandidates : 0u;
            bins[12] = candidates;
            bins[14] = totalCandidates > capacity ? totalCandidates - capacity : 0u;
            args[0] = min(candidates, kHybridDispatchWidth);
            args[1] = max(1u, (candidates + kHybridDispatchWidth - 1u) / kHybridDispatchWidth);
            args[2] = 1u;
            uint candidateBlocks = (candidates + 127u) / 128u;
            args[3] = min(candidateBlocks, kHybridDispatchWidth);
            args[4] = max(1u, (candidateBlocks + kHybridDispatchWidth - 1u) / kHybridDispatchWidth);
            args[5] = 1u;
        }
        return;
    }

    uint block = hybridFlatGroup(groupID);
    if (block >= blocks) { return; }
    uint index = block * 128u + lane;
    if (push.activeBuildPhase == 1u) {
        uint mask = 0u;
        if (index < count) {
            mask = streamPhaseCandidateMask(groups[index], index, bins, visibility, push.traversalPhase == 0u);
        }
        uint localCount = countbits(mask);
        uint offset = streamCandidateInclusiveScan(localCount, lane) - localCount;
        if (index < count) {
            bins[kHybridBinHeaderWords + index] = mask;
            bins[hybridRankBase(capacity) + index] = offset;
        }
        if (lane == 127u) { bins[blockBase + block] = g_StreamCandidatePrefix[127]; }
    } else if (push.activeBuildPhase == 3u && index < count && bins[14] == 0u) {
        uint mask = bins[kHybridBinHeaderWords + index];
        uint offset = bins[blockBase + block] + bins[hybridRankBase(capacity) + index];
        while (mask != 0u) {
            uint cluster = firstbitlow(mask);
            if (offset < capacity) {
                uint slot = hybridCandidateBase(capacity) + offset * 2u;
                bins[slot] = index * bins[13] + cluster;
                bins[slot + 1u] = 0xffffffffu;
            }
            ++offset;
            mask &= mask - 1u;
        }
    }
}

// One lane validates/culls one candidate. Survivors carry only the geometry
// decode fields needed by classification, in the still-unused bin list storage.
// A wave reserves one range; its scheduling order is intentionally irrelevant:
// classifiers write back to original candidate slots before stable binning.
void cullStreamClusterCandidates(uint3 group, uint lane, GPUDrivenStreamAssetPush push, bool divertHardware)
{
    RWStructuredBuffer<uint> handle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.hybridClusterBuffer));
    RWStructuredBuffer<uint> bins = handle;
    if (push.activeBuildPhase == 1u) {
        if (lane == 0u) {
            RWStructuredBuffer<uint> args =
                resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.hybridQueueBuffer));
            uint count = bins[0];
            args[0] = min(count, kHybridDispatchWidth);
            args[1] = max(1u, (count + kHybridDispatchWidth - 1u) / kHybridDispatchWidth);
            args[2] = 1u;
        }
        return;
    }
    uint candidate = hybridFlatGroup(group) * 128u + lane;
    StreamRasterCluster cluster = (StreamRasterCluster)0;
    bool visible = false;
    if (candidate < bins[12]) {
        uint record = bins[hybridCandidateBase(bins[5]) + candidate * 2u];
        bool retryLate;
        visible = loadStreamRasterCluster(record, true, true, push, cluster, retryLate) && cluster.triangleCount != 0u;
        if (push.traversalPhase == 0u && retryLate) {
            InterlockedOr(bins[hybridRetryBase(bins[5]) + record / bins[13]], 1u << (record % bins[13]));
        }
    }
    // Zero is an internal full-HW comparison mode, never a relaxed SW limit.
    // Publish exactly the same survivors/records without a geometry-classify dispatch.
    if (asfloat(bins[8]) == 0.0) {
        if (visible) { hybridClassifyCluster(bins, 0u, bins[hybridCandidateBase(bins[5]) + candidate * 2u], candidate); }
        return;
    }
    // Coverage/tessellation is a semantic HW requirement, independent of
    // the optional sphere fast path and its malformed-sphere fallback.
    bool fastHardware = visible && cluster.forceHardware &&
        (divertHardware || (cluster.classificationFlags & 1u) == 0u);
    bool fastSoftware = visible && (cluster.classificationFlags & 1u) == 0u && !cluster.forceHardware &&
        streamSphereFitsSoftware(cluster, float2(bins[6], bins[7]), asfloat(bins[8]));
    if (fastHardware || fastSoftware) {
        hybridClassifyCluster(bins, fastHardware ? 0u : kHybridSoftwareBin,
            bins[hybridCandidateBase(bins[5]) + candidate * 2u], candidate);
    }
    // These counters are diagnostic until stable binning replaces all five.
    uint swCount = WaveActiveCountBits(fastSoftware), hwCount = WaveActiveCountBits(fastHardware);
    if (WaveIsFirstLane()) {
        if (swCount != 0u) { InterlockedAdd(bins[1], swCount); }
        if (hwCount != 0u) { InterlockedAdd(bins[2], hwCount); }
    }
    visible = visible && !fastHardware && !fastSoftware;
    uint count = WaveActiveCountBits(visible);
    uint offset = WavePrefixCountBits(visible);
    uint base = 0u;
    if (WaveIsFirstLane() && count != 0u) { InterlockedAdd(bins[0], count, base); }
    base = WaveReadLaneFirst(base);
    if (visible) {
        uint item = hybridStreamClassifyBase(base + offset);
        bins[item] = candidate;
        bins[item + 1u] = cluster.pageWord + cluster.positionOffsetBytes / 4u + cluster.vertexOffset * cluster.positionStrideWords;
        bins[item + 2u] = cluster.vertexCount | (cluster.triangleCount << 16u) | (cluster.positionStrideWords == 3u ? 0x80000000u : 0u);
        bins[item + 3u] = cluster.triangleBaseByte;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterCullMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    cullStreamClusterCandidates(group, lane, push, true);
}

// P0 producer/consumer pair retained for integration and sample A/B profiling.
[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterCullP0Main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    cullStreamClusterCandidates(group, lane, push, false);
}

