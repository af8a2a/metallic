#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Workgroup-cooperative LOD frontier, masking and prefetch.
public groupshared uint g_CooperativeWaveCounts[64];
public groupshared uint g_CooperativeRoots;
public groupshared uint g_CooperativeTelemetryContinuous;
public groupshared uint g_CooperativeMissingRoots;
public static const uint kCooperativeActiveWordCount = 128u;
public groupshared uint g_CooperativeActiveBits[kCooperativeActiveWordCount];

// Stable workgroup scan, with no fixed subgroup size assumption.
uint streamCooperativePrefix(uint value, uint lane, out uint total)
{
    uint wave = lane / WaveGetLaneCount();
    uint waves = (64u + WaveGetLaneCount() - 1u) / WaveGetLaneCount();
    uint prefix = WavePrefixSum(value);
    uint sum = WaveActiveSum(value);
    if (WaveIsFirstLane()) { g_CooperativeWaveCounts[wave] = sum; }
    GroupMemoryBarrierWithGroupSync();
    total = 0u;
    for (uint i = 0u; i < waves; ++i) {
        uint count = g_CooperativeWaveCounts[i];
        if (i < wave) { prefix += count; }
        total += count;
    }
    GroupMemoryBarrierWithGroupSync();
    return prefix;
}

void streamCooperativeMasks(StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> topology, RWStructuredBuffer<uint> state,
    GPUDrivenStreamAssetPrimitive primitive, uint header, uint base, uint sparse,
    uint count, bool ready, uint lane)
{
    uint selected = 0u;
    if (ready) {
        for (uint index = lane; index < count; index += 64u) {
            uint localGroup = state[sparse + 4u + index];
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
            uint mask = 0u;
            for (uint cluster = 0u; cluster < group.clusterCount; ++cluster) {
                uint refined = topology[group.clusterRefinedOffset + cluster];
                if (refined == kInvalidClusterIndex || state[base + (refined - primitive.groupOffset) * 2u] == 0u) { mask |= 1u << cluster; }
            }
            state[base + localGroup * 2u + 1u] = mask;
            selected += mask != 0u ? 1u : 0u;
        }
    }
    uint totalSelected;
    streamCooperativePrefix(selected, lane, totalSelected);
    if (lane == 0u) { state[header] = totalSelected; }
}

[ForceInline]
void streamCooperativeLod(uint3 workgroup, uint lane, GPUDrivenStreamAssetPush push,
    uint cooperativeWidth)
{
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));
    GPUDrivenStreamAssetParams params = paramsHandle[0];
    uint instanceIndex = workgroup.x + workgroup.y * 65535u;
    if (instanceIndex >= params.sceneInstanceCount) { return; }
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(push.instanceBuffer));
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>(push.primitiveBuffer));
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>(push.groupBuffer));
    StructuredBuffer<uint> topology =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(params.lodTopologyBuffer));
    RWStructuredBuffer<uint> state =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.lodStateBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(push.pageTableBuffer));
    RWStructuredBuffer<uint> requests =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.requestBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(push.activeGroupBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(push.activeHeaderBuffer));
    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    uint header = instanceIndex * 4u;
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        if (lane < 4u && push.activeBuildPhase != kActiveBuildPhasePrefetch) { state[header + lane] = 0u; }
        return;
    }
    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    // Request-only lookahead: parent residency still gates the actual frontier.
    // Run after all demand requests so speculation cannot consume their slots.
    if (push.activeBuildPhase == kActiveBuildPhasePrefetch) {
        uint limit = requests[kStreamRequestPrefetchLimitWord];
        if (params.enableGpuLodSelection == 0u || params.prefetchParams.z == 0.0 ||
            instance.visible == 0u || primitive.lodTileOffset == 0xffffffffu || limit == 0u) { return; }
        GPUDrivenStreamAssetParams forecast = params;
        if (params.prefetchParams.w != 0.0) {
            forecast.eye.xyz = params.prefetchEye.xyz;
            forecast.center.xyz = params.prefetchCenter.xyz;
            forecast.upProjection.xyz = params.prefetchUp.xyz;
        }
        forecast.viewport.w = 2.0 * atan(tan(params.viewport.w * 0.5) * params.prefetchParams.x);
        forecast.clipOrtho.z *= params.prefetchParams.x;
        forecast.lodPixelError *= params.prefetchParams.y;
        if (!streamRayLodEnabled(params) && !streamSphereInsideFrustum(instance.boundsCenterRadius.xyz,
                max(instance.boundsCenterRadius.w, 0.000001), currentStreamCamera(forecast))) { return; }
        float3 delta = forecast.center.xyz - forecast.eye.xyz;
        float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
        float4 eye = float4(forecast.eye.xyz, params.clipOrtho.x);
        float4 direction = float4(forward, params.upProjection.w);
        float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5), params.clipOrtho.z, forecast.lodPixelError);
        uint table = requests[kStreamRequestPriorityTableOffsetWord];
        uint tiles = topology[primitive.lodTileOffset];
        for (uint tile = 0u; tile < tiles;) {
            // The CPU publishes only spare admission capacity. Stop searching
            // once it is filled instead of repeatedly reporting deferred work.
            if (requests[kStreamRequestPrefetchCounterWord] >= limit) { return; }
            uint offset = primitive.lodTileOffset + 1u + tile * kStreamLodBvhNodeWords;
            if (!streamLodBvhNeedsFine(topology, offset, instance, eye, direction, projection, 0xffffffffu, forecast)) {
                tile = topology[offset + 7u];
                continue;
            }
            ++tile;
            uint count = topology[offset + 9u];
            if (count == 0u) { continue; }
            if (lane >= count) { continue; }
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + topology[offset + 8u] + count - 1u - lane];
            if ((group.flags & 1u) != 0u || group.pageIndex >= params.scenePageCount ||
                streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) { continue; }
            uint historyBase = topology[params.lodInstanceOffsetsOffset + instanceIndex];
            if (!streamLodGroupDemand(group, instance, eye, direction, projection, 0xffffffffu,
                    forecast, state, historyBase, primitive.groupCount,
                    topology[offset + 8u] + count - 1u - lane, false)) { continue; }
            if (table != 0u) {
                float benefit = streamGroupScreenBenefit(group, instance, forecast, eye, direction, projection);
                InterlockedMax(requests[table + group.pageIndex], asuint(benefit));
            }
            uint previous;
            InterlockedMax(pageTable[group.pageIndex].lastRequestFrame, params.frameIndex, previous);
            if (previous >= params.frameIndex) { continue; }
            uint index;
            InterlockedAdd(requests[kStreamRequestPrefetchCounterWord], 1u, index);
            if (index >= limit) {
                InterlockedAdd(requests[kStreamRequestPrefetchDroppedWord], 1u);
                continue;
            }
            for (;;) {
                uint count = requests[kStreamRequestLoadCounterWord];
                if (count >= requests[kStreamRequestMaxLoadsWord]) {
                    InterlockedAdd(requests[kStreamRequestPrefetchDroppedWord], 1u);
                    break;
                }
                uint observed;
                InterlockedCompareExchange(requests[kStreamRequestLoadCounterWord], count, count + 1u, observed);
                if (observed == count) {
                    requests[kStreamRequestHeaderWords + count] = group.pageIndex | kStreamPrefetchPageTag;
                    break;
                }
            }
        }
        return;
    }
    if (primitive.lodTileOffset == 0xffffffffu) {
        if (push.activeBuildPhase == kActiveBuildPhaseClear || push.activeBuildPhase == kActiveBuildPhaseMask) { return; }
        if (lane == 0u) {
            if (push.activeBuildPhase == kActiveBuildPhaseFrontier) {
                buildStreamLodFrontier(instanceIndex, instances, primitives, groups, topology, state, pageTable, requests, params, push);
            } else {
                emitStreamLodFrontier(instanceIndex, instances, primitives, groups, topology, state, pageTable, activeGroups, activeHeader, params);
            }
        }
        return;
    }
    uint base = topology[params.lodInstanceOffsetsOffset + instanceIndex];
    uint sparse = base + primitive.groupCount * 2u;
    if (push.activeBuildPhase == kActiveBuildPhaseMask) {
        uint count = state[sparse];
        bool ready = state[header + 3u] != 0u;
        bool terminalOnly = ready && count == state[header + 2u];
        bool diagnostic = params.enableTerminalMaskDiagnostics != 0u && params.demandStatsBuffer != 0xffffffffu;
        if (diagnostic && lane == 0u) {
            RWStructuredBuffer<uint> stats = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.demandStatsBuffer));
            if (ready) { InterlockedAdd(stats[29], 1u); }
            if (terminalOnly) { InterlockedAdd(stats[30], 1u); }
        }
        if (terminalOnly) {
            // Split Frontier has published every active group and all ready
            // terminals. Equal counts prove that no refined group is active.
            // This does not imply absent fine demand: residency may block it.
            uint clusters = 0u;
            for (uint index = lane; index < count; index += 64u) {
                uint localGroup = state[sparse + 4u + index];
                uint clusterCount = groups[primitive.groupOffset + localGroup].clusterCount;
                state[base + localGroup * 2u + 1u] = clusterCount == 32u ? 0xffffffffu : (1u << clusterCount) - 1u;
                if (diagnostic) { clusters += clusterCount; }
            }
            if (diagnostic) {
                uint total = WaveActiveSum(clusters);
                if (WaveIsFirstLane() && total != 0u) {
                    RWStructuredBuffer<uint> stats = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.demandStatsBuffer));
                    InterlockedAdd(stats[31], total);
                }
            }
            if (lane == 0u) { state[header] = count; }
            return;
        }
        streamCooperativeMasks(groups, topology, state, primitive, header, base, sparse,
            count, ready, lane);
        return;
    }
    if (push.activeBuildPhase == kActiveBuildPhaseEmit) {
        if (state[header + 3u] == 0u || activeHeader[0].padding1 != 0u) { return; }
        uint count = state[sparse];
        uint output = state[header + 1u];
        bool fallback = (state[header + 3u] & 2u) != 0u;
        for (uint first = 0u; first < count; first += 64u) {
            uint index = first + lane;
            GPUDrivenStreamAssetGroup source;
            uint mask = 0u;
            if (index < count) {
                uint localGroup = state[sparse + 4u + count - 1u - index];
                source = groups[primitive.groupOffset + localGroup];
                mask = state[base + localGroup * 2u + 1u];
                if (fallback) { mask = (source.flags & 1u) == 0u ? 0u :
                    (source.clusterCount == 32u ? 0xffffffffu : (1u << source.clusterCount) - 1u); }
            }
            uint emitted;
            uint offset = streamCooperativePrefix(mask != 0u ? 1u : 0u, lane, emitted);
            if (mask != 0u) {
                uint localGroup = state[sparse + 4u + count - 1u - index];
                selectStreamLodTelemetry(state, base, primitive.groupCount, localGroup, mask,
                    (source.flags & 1u) != 0u, fallback, params);
                GPUDrivenStreamAssetActiveGroup group;
                group.pageDeviceOffsetBytes = streamPageDeviceOffset(pageTable[source.pageIndex]);
                group.pageIndex = source.pageIndex;
                group.clusterCount = source.clusterCount;
                group.primitiveIndex = source.primitiveIndex;
                group.lodLevel = source.lodLevel;
                group.materialIndex = instance.materialIndex;
                group.clusterSelectionMask = mask;
                group.flags = kActiveGroupResident;
                group.instanceIndex = instanceIndex;
                group.gpuSceneInstanceIndex = instance.gpuSceneInstanceIndex;
                group.padding1 = 0u; group.padding2 = 0u;
                group.world0 = instance.world0; group.world1 = instance.world1;
                group.world2 = instance.world2; group.world3 = instance.world3;
                activeGroups[output + offset] = group;
            }
            output += emitted;
        }
        return;
    }

    if (push.activeBuildPhase == kActiveBuildPhaseClear || params.splitFrontier == 0u) {
        uint previousCount = state[sparse];
        GroupMemoryBarrierWithGroupSync();
        for (uint index = lane; index < previousCount; index += 64u) {
            uint localGroup = state[sparse + 4u + index];
            state[base + localGroup * 2u] = 0u;
            state[base + localGroup * 2u + 1u] = 0u;
        }
        if (lane < 4u) { state[header + lane] = 0u; state[sparse + lane] = 0u; }
        if (lane == 0u) { prepareStreamRayLodHistory(state, base, primitive.groupCount, instance.visible != 0u, params); }
    }
    bool mirrorActive = primitive.groupCount <= kCooperativeActiveWordCount * 32u;
    if (push.activeBuildPhase != kActiveBuildPhaseClear) {
        if (lane == 0u) { g_CooperativeRoots = 0u; g_CooperativeMissingRoots = 0u; }
        if (mirrorActive) {
            uint words = (primitive.groupCount + 31u) / 32u;
            for (uint word = lane; word < words; word += cooperativeWidth) {
                g_CooperativeActiveBits[word] = 0u;
            }
        }
        AllMemoryBarrierWithGroupSync();
    }
    if (params.enableGpuLodSelection != 0u && !streamRayLodEnabled(params) && push.rasterBindingsBuffer != 0xffffffffu) {
        StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindings =
            resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));
        if ((bindings[0].cullingFlags & 1u) != 0u && !streamSphereInsideFrustum(instance.boundsCenterRadius.xyz,
                max(instance.boundsCenterRadius.w, 0.000001) + bindings[0].displacementBound, currentStreamCamera(params))) { instance.visible = 0u; }
    }
    if (push.activeBuildPhase != kActiveBuildPhaseClear && params.enableLodTransitionTelemetry != 0u) {
        if (lane == 0u) {
            g_CooperativeTelemetryContinuous = beginStreamLodTelemetry(state, base, primitive.groupCount,
                instance.visible != 0u, params) ? 1u : 0u;
        }
        AllMemoryBarrierWithGroupSync();
    }
    if (instance.visible == 0u) { return; }
    if (push.activeBuildPhase == kActiveBuildPhaseClear) {
        if (params.demandBuffer != 0xffffffffu) {
            RWStructuredBuffer<uint> demand = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.demandBuffer));
            RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem> tasks =
                resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>>(push.traversalWorkBuffer));
            uint meta = params.demandInstanceOffsetsOffset + instanceIndex * 4u;
            uint first = topology[meta + 2u], count = topology[meta + 3u];
            float3 delta = params.center.xyz - params.eye.xyz;
            float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
            float4 eye = float4(params.eye.xyz, params.clipOrtho.x);
            float4 direction = float4(forward, params.upProjection.w);
            float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5), params.clipOrtho.z, params.lodPixelError);
            uint manual = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
            for (uint batch = 0u; batch < count; batch += 64u) {
                uint index = batch + lane, root = 0u;
                bool wanted = false;
                if (index < count) {
                    root = topology[params.demandTaskOffset + first + index];
                    wanted = streamLodBvhNeedsFine(topology, primitive.lodTileOffset + 1u + root * kStreamLodBvhNodeWords,
                        instance, eye, direction, projection, manual, params);
                }
                // Task order has no cut/budget semantics. Reserve independently
                // per wave; a dispatch barrier publishes all payloads to workers.
                uint total = WaveActiveCountBits(wanted);
                uint rank = WavePrefixCountBits(wanted);
                uint firstTask = 0u;
                if (WaveIsFirstLane() && total != 0u) { InterlockedAdd(demand[16], total, firstTask); }
                firstTask = WaveReadLaneFirst(firstTask);
                if (wanted) {
                    GPUDrivenStreamAssetTraversalWorkItem work;
                    work.instanceIndex = instanceIndex; work.nodeIndex = root;
                    work.readyFrame = 0u; work.padding0 = 0u;
                    tasks[firstTask + rank] = work;
                }
            }
            // The lane that cleared this word also publishes the statistic;
            // no wave operation publishes another lane's device-memory stores.
            if (lane == 3u) { state[sparse + 3u] = count; }
        }
        return;
    }
    float3 delta = params.center.xyz - params.eye.xyz;
    float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
    float4 eye = float4(params.eye.xyz, params.clipOrtho.x);
    float4 direction = float4(forward, params.upProjection.w);
    float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5), params.clipOrtho.z, params.lodPixelError);
    uint manual = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
    uint count = 0u, tested = 0u, visited = 0u;
    uint previousLevel = 0xffffffffu;
    uint tiles = topology[primitive.lodTileOffset];
    for (uint tile = 0u; tile < tiles;) {
        uint offset = primitive.lodTileOffset + 1u + tile * kStreamLodBvhNodeWords;
        ++visited;
        bool visit;
        if (params.demandBuffer != 0xffffffffu) {
            StructuredBuffer<uint> demand = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(params.demandBuffer));
            uint bit = topology[params.demandInstanceOffsetsOffset + instanceIndex * 4u + 1u] + tile;
            visit = (demand[kStreamDemandStatsWords + (bit >> 5u)] & (1u << (bit & 31u))) != 0u;
        } else {
            visit = streamLodBvhNeedsFine(topology, offset, instance, eye, direction, projection, manual, params);
        }
        if (!visit) { tile = topology[offset + 7u]; continue; }
        ++tile;
        uint tileCount = topology[offset + 9u];
        if (tileCount == 0u) { continue; }
        // Ordered leaves contain one LOD level. Sibling tiles are independent;
        // only a finer level can read the active flags written by this level.
        // Telemetry reads parent blocked stamps from device memory. Otherwise
        // the mirror only needs LDS publication until the final mask barrier.
        uint level = topology[offset + 5u];
        if (previousLevel != 0xffffffffu && level != previousLevel) {
            if (mirrorActive && params.enableLodTransitionTelemetry == 0u) {
                GroupMemoryBarrierWithGroupSync();
            } else {
                AllMemoryBarrierWithGroupSync();
            }
        }
        previousLevel = level;
        // Keep the original lane order when a 64-group leaf spans two batches.
        // Groups within this LOD are independent; publish before the finer LOD.
        for (uint first = 0u; first < tileCount; first += cooperativeWidth) {
            uint tileLane = first + lane;
            uint localGroup = topology[offset + 8u] + tileCount - 1u - min(tileLane, tileCount - 1u);
            bool active = false;
            if (tileLane < tileCount) {
                GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
                bool terminal = (group.flags & 1u) != 0u;
                bool parentsActive = true;
                for (uint parent = 0u; parent < group.parentCount; ++parent) {
                    uint id = topology[group.parentOffset + parent];
                    uint localParent = id - primitive.groupOffset;
                    uint parentActive;
                    if (mirrorActive) {
                        // Parent and current-level child bits can share a word;
                        // an atomic read must not race with another lane's OR.
                        uint bits;
                        InterlockedOr(g_CooperativeActiveBits[localParent >> 5u], 0u, bits);
                        parentActive = bits & (1u << (localParent & 31u));
                    } else {
                        parentActive = state[base + localParent * 2u];
                    }
                    parentsActive = parentsActive && parentActive != 0u;
                }
                bool demanded;
                if (params.demandBuffer != 0xffffffffu) {
                    StructuredBuffer<uint> demand = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(params.demandBuffer));
                    uint bit = topology[params.demandInstanceOffsetsOffset + instanceIndex * 4u] + localGroup;
                    demanded = (demand[kStreamDemandStatsWords + (bit >> 5u)] & (1u << (bit & 31u))) != 0u;
                } else {
                    demanded = streamLodGroupDemand(group, instance, eye, direction, projection,
                        manual, params, state, base, primitive.groupCount, localGroup);
                    if (demanded) {
                        if (streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) { markStreamPageDemand(pageTable, params, group.pageIndex); }
                        else {
                            uint table = requests[kStreamRequestPriorityTableOffsetWord];
                            if (table != 0u) {
                                float benefit = streamGroupScreenBenefit(group, instance, params, eye, direction, projection);
                                InterlockedMax(requests[table + group.pageIndex], asuint(benefit));
                            }
                            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
                        }
                    }
                }
                bool drawable = groupPageDrawable(pageTable, params, group);
                active = parentsActive && demanded && drawable;
                observeStreamLodGroup(state, topology, primitive, group, instance, base, localGroup,
                    demanded, drawable, active,
                    params.enableLodTransitionTelemetry != 0u && g_CooperativeTelemetryContinuous != 0u, params);
                if (active) {
                    state[base + localGroup * 2u] = 1u;
                    if (mirrorActive) {
                        InterlockedOr(g_CooperativeActiveBits[localGroup >> 5u], 1u << (localGroup & 31u));
                    }
                }
                if (terminal) {
                    InterlockedAdd(g_CooperativeRoots, 1u);
                    if (!drawable) { InterlockedOr(g_CooperativeMissingRoots, 1u); }
                }
            }
            uint added;
            uint rank;
            if (cooperativeWidth == 32u) {
                // This entry is selected only for a full Wave32 workgroup.
                rank = WavePrefixSum(active ? 1u : 0u);
                added = WaveActiveSum(active ? 1u : 0u);
            } else {
                rank = streamCooperativePrefix(active ? 1u : 0u, lane, added);
            }
            if (active) { state[sparse + 4u + count + rank] = localGroup; }
            count += added;
        }
        tested += tileCount;
    }
    AllMemoryBarrierWithGroupSync();
    if (params.splitFrontier == 0u) {
        streamCooperativeMasks(groups, topology, state, primitive, header, base, sparse,
            count, g_CooperativeMissingRoots == 0u, lane);
    }
    if (lane == 0u) {
        state[sparse] = count; state[sparse + 1u] = visited; state[sparse + 2u] = tested;
        state[header + 2u] = g_CooperativeMissingRoots == 0u ? g_CooperativeRoots : 0u;
        state[header + 3u] = g_CooperativeMissingRoots == 0u ? 1u : 0u;
        if (params.demandBuffer == 0xffffffffu && params.demandStatsBuffer != 0xffffffffu) {
            RWStructuredBuffer<uint> stats = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.demandStatsBuffer));
            InterlockedAdd(stats[18], tested);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void streamCooperativeLodMain(uint3 workgroup : SV_GroupID, uint lane : SV_GroupIndex,
    uniform GPUDrivenStreamAssetPush push)
{
    streamCooperativeLod(workgroup, lane, push, 64u);
}

// The runtime selects this entry only for split/distributed Frontier on fixed
// Wave32 devices. Clear, Mask, Emit and Prefetch retain the 64-thread entry.
[shader("compute")]
[numthreads(32, 1, 1)]
void streamCooperativeLod32Main(uint3 workgroup : SV_GroupID, uint lane : SV_GroupIndex,
    uniform GPUDrivenStreamAssetPush push)
{
    streamCooperativeLod(workgroup, lane, push, 32u);
}

