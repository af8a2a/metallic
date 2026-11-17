#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;
import WaveWorkDistribution;

// Wave-distributed demand traversal and demand statistics.
[shader("compute")]
[numthreads(64, 1, 1)]
void streamDistributedDemandMain(uint3 workgroup : SV_GroupID, uint lane : SV_GroupIndex,
    uniform GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));
    GPUDrivenStreamAssetParams params = paramsHandle[0];
    if (params.demandBuffer == 0xffffffffu) { return; }
    RWStructuredBuffer<uint> demand =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.demandBuffer));
    if (push.activeBuildPhase == kActiveBuildPhaseDemandReset) {
        RWStructuredBuffer<uint> words = demand;
        uint size, stride;
        words.GetDimensions(size, stride);
        uint index = (workgroup.x + workgroup.y * 65535u) * 64u + lane;
        if (index < size) { demand[index] = 0u; }
        return;
    }
    RWStructuredBuffer<uint> state =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.lodStateBuffer));
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(push.instanceBuffer));
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>(push.primitiveBuffer));
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>(push.groupBuffer));
    StructuredBuffer<uint> topology =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(params.lodTopologyBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(push.pageTableBuffer));
    RWStructuredBuffer<uint> requests =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.requestBuffer));
    StructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem> tasks =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>>(push.traversalWorkBuffer));
    uint taskCount = demand[16];
    float3 delta = params.center.xyz - params.eye.xyz;
    float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
    float4 eye = float4(params.eye.xyz, params.clipOrtho.x);
    float4 direction = float4(forward, params.upProjection.w);
    float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5), params.clipOrtho.z, params.lodPixelError);
    uint manual = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
    uint processed = 0u, visitedTotal = 0u, testedTotal = 0u, maxVisited = 0u, maxTested = 0u, nonempty = 0u, slots = 0u;
    uint histogram[8];
    [unroll] for (uint bin = 0u; bin < 8u; ++bin) { histogram[bin] = 0u; }
    uint instanceIndex = 0u, tile = 0u, end = 0u, visited = 0u, tested = 0u;
    bool haveTask = false, exhausted = false;
    GPUDrivenStreamAssetInstance instance = {};
    GPUDrivenStreamAssetPrimitive primitive = {};
    for (;;) {
        uint firstGroup = 0u, count = 0u, leaf = 0u;
        // Fill productive sources before flattening their children. Refill
        // empty lanes across task boundaries; cap the search so one lane cannot
        // hold a ready wave while walking the remaining empty task tail.
        for (uint refill = 0u; refill < 8u && WaveActiveAnyTrue(count == 0u && !exhausted); ++refill) {
            bool needTask = count == 0u && !exhausted && tile == end;
            uint claims = WaveActiveCountBits(needTask);
            uint rank = WavePrefixCountBits(needTask);
            uint firstTask = 0u;
            if (WaveIsFirstLane() && claims != 0u) { InterlockedAdd(demand[0], claims, firstTask); }
            firstTask = WaveReadLaneFirst(firstTask);
            if (needTask) {
                if (haveTask) {
                    visitedTotal += visited; testedTotal += tested;
                    maxVisited = max(maxVisited, visited); maxTested = max(maxTested, tested);
                    nonempty += tested != 0u ? 1u : 0u;
                    uint bin = tested == 0u ? 0u : tested <= 8u ? 1u : min(7u, uint(firstbithigh(tested - 1u)) - 1u);
                    ++histogram[bin];
                }
                uint task = firstTask + rank;
                haveTask = task < taskCount;
                exhausted = !haveTask;
                visited = 0u; tested = 0u;
                if (haveTask) {
                    ++processed;
                    GPUDrivenStreamAssetTraversalWorkItem work = tasks[task];
                    instanceIndex = work.instanceIndex;
                    instance = instances[instanceIndex];
                    primitive = primitives[instance.primitiveIndex];
                    tile = work.nodeIndex;
                    end = topology[primitive.lodTileOffset + 1u + tile * kStreamLodBvhNodeWords + 7u];
                }
            }
            if (count == 0u && !exhausted) {
                while (tile < end) {
                    uint offset = primitive.lodTileOffset + 1u + tile * kStreamLodBvhNodeWords;
                    ++visited;
                    if (!streamLodBvhNeedsFine(topology, offset, instance, eye, direction, projection, manual, params)) {
                        tile = topology[offset + 7u];
                        continue;
                    }
                    ++tile;
                    count = topology[offset + 9u];
                    if (count != 0u) {
                        leaf = tile - 1u;
                        firstGroup = primitive.groupOffset + topology[offset + 8u] + count - 1u;
                        break;
                    }
                }
                tested += count;
            }
        }
        if (!WaveActiveAnyTrue(count != 0u) && WaveActiveAllTrue(exhausted)) { break; }
        WaveWorkDistribution work = waveWorkBegin(count);
        for (uint batch = 0u; batch < work.total; batch += WaveGetLaneCount()) {
            uint item = batch + WaveGetLaneIndex();
            uint localItem;
            uint source = waveWorkNext(work, batch, localItem);
            uint groupIndex = WaveReadLaneAt(firstGroup, source) - localItem;
            uint owner = WaveReadLaneAt(instanceIndex, source);
            uint ownerLeaf = WaveReadLaneAt(leaf, source);
            if (item < work.total) {
                GPUDrivenStreamAssetGroup group = groups[groupIndex];
                GPUDrivenStreamAssetInstance ownerInstance = instances[owner];
                GPUDrivenStreamAssetPrimitive ownerPrimitive = primitives[ownerInstance.primitiveIndex];
                uint ownerBase = topology[params.lodInstanceOffsetsOffset + owner];
                bool wanted = streamLodGroupDemand(group, ownerInstance, eye, direction, projection,
                    manual, params, state, ownerBase, ownerPrimitive.groupCount, groupIndex - ownerPrimitive.groupOffset);
                if (wanted) {
                    uint groupBits = topology[params.demandInstanceOffsetsOffset + owner * 4u];
                    uint bit = groupBits + groupIndex - ownerPrimitive.groupOffset;
                    InterlockedOr(demand[kStreamDemandStatsWords + (bit >> 5u)], 1u << (bit & 31u));
                    // Publish the demand path once per leaf. Other lanes may
                    // stop at an already marked ancestor: its winner will
                    // finish propagating before the dispatch barrier.
                    uint tileBits = topology[params.demandInstanceOffsetsOffset + owner * 4u + 1u];
                    uint parentTable = ownerPrimitive.lodTileOffset + 1u + topology[ownerPrimitive.lodTileOffset] * kStreamLodBvhNodeWords;
                    for (uint node = ownerLeaf; node != 0xffffffffu; node = topology[parentTable + node]) {
                        uint nodeBit = tileBits + node;
                        uint mask = 1u << (nodeBit & 31u), previous;
                        InterlockedOr(demand[kStreamDemandStatsWords + (nodeBit >> 5u)], mask, previous);
                        if ((previous & mask) != 0u) { break; }
                    }
                    if (streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) {
                        markStreamPageDemand(pageTable, params, group.pageIndex);
                    } else {
                        uint table = requests[kStreamRequestPriorityTableOffsetWord];
                        if (table != 0u) {
                            float benefit = streamGroupScreenBenefit(group, ownerInstance, params, eye, direction, projection);
                            InterlockedMax(requests[table + group.pageIndex], asuint(benefit));
                        }
                        requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
                    }
                }
            }
            if (WaveIsFirstLane()) { slots += WaveGetLaneCount(); }
        }
    }
    // Reduce once per worker wave, not per task. Histograms count all admitted
    // tasks and make the remaining task tail observable.
    uint p = WaveActiveSum(processed), v = WaveActiveSum(visitedTotal), t = WaveActiveSum(testedTotal);
    uint mv = WaveActiveMax(maxVisited), mt = WaveActiveMax(maxTested), n = WaveActiveSum(nonempty);
    if (WaveIsFirstLane()) {
        InterlockedAdd(demand[1], p); InterlockedAdd(demand[2], v); InterlockedAdd(demand[3], t);
        InterlockedMax(demand[4], mv); InterlockedMax(demand[5], mt); InterlockedAdd(demand[6], n);
        InterlockedAdd(demand[7], slots);
        InterlockedAdd(demand[18], t);
    }
    if (workgroup.x == 0u && workgroup.y == 0u && lane == 0u) { demand[17] = 1u; }
    [unroll] for (uint bin = 0u; bin < 8u; ++bin) {
        uint value = WaveActiveSum(histogram[bin]);
        if (WaveIsFirstLane()) { InterlockedAdd(demand[8u + bin], value); }
    }
}

