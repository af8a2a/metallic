#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// BVH frontier evaluation, transition telemetry and prefix/emission stages.
// MeshletLODBVHNode: ten metric/topology words, then refinement demand min/max.
// Preorder leaves cover descending group IDs; escape links preserve parent-first
// evaluation without a stack/queue limit.
static const uint kStreamLodBvhNodeWords = 16u;

bool streamLodBvhNeedsFine(StructuredBuffer<uint> topology, uint offset,
    GPUDrivenStreamAssetInstance instance, float4 eye, float4 forward,
    float4 projection, uint manualLevel, GPUDrivenStreamAssetParams params)
{
    if ((topology[offset + 6u] & 1u) != 0u) { return true; }
    if (manualLevel != 0xffffffffu) { return topology[offset + 5u] >= manualLevel; }
    float4 sphere = float4(asfloat(topology[offset]), asfloat(topology[offset + 1u]),
        asfloat(topology[offset + 2u]), asfloat(topology[offset + 3u]));
    // Computing parent and child centers independently in float can break
    // containment after large-coordinate cancellation. Bound both affine/view
    // roundoff errors using all possible centers inside this object sphere.
    float worldPadding = 0.0;
    if (forward.w == 0.0) {
        float3 extent = abs(sphere.xyz) + sphere.www;
        float3 magnitude = abs(instance.world0.xyz) * extent.x + abs(instance.world1.xyz) * extent.y +
            abs(instance.world2.xyz) * extent.z + abs(instance.world3.xyz) + abs(eye.xyz);
        worldPadding = (32.0 * 1.1920928955078125e-7) * (magnitude.x + magnitude.y + magnitude.z);
    }
    // A node may contain retained fine groups or on-screen children. Start at
    // the lowest hysteresis threshold; only proven offscreen bounds may scale it.
    float threshold = projection.w * (streamRayLodEnabled(params)
        ? 1.0 - clamp(params.rtLodHysteresis, 0.0, 0.9) : 1.0);
    float pruneThreshold = threshold - max(abs(threshold) * 1e-4, 1e-5);
    float error = asfloat(topology[offset + 4u]);
    float upperError;
    if (streamRayLodEnabled(params) && (params.rtLodFlags & 2u) != 0u) {
        upperError = streamLodMetric(sphere, error,
            instance, eye, forward, projection, params, false, worldPadding);
        // Every descendant uses at least this threshold, including retained
        // fine groups. Scale the threshold, never the ray error bound.
        float offscreenThreshold = threshold * max(params.rtOffscreenErrorScale, 1.0);
        float offscreenPruneThreshold = offscreenThreshold - max(abs(offscreenThreshold) * 1e-4, 1e-5);
        // Classification can only help below the largest possible threshold.
        // Retain near/inside-sphere and non-finite bounds without metadata reads.
        if (!(upperError < offscreenPruneThreshold)) { return true; }
        float3 low = float3(asfloat(topology[offset + 10u]), asfloat(topology[offset + 11u]), asfloat(topology[offset + 12u]));
        float3 high = float3(asfloat(topology[offset + 13u]), asfloat(topology[offset + 14u]), asfloat(topology[offset + 15u]));
        // These bounds contain every group's refinement descendants, unlike
        // the metric sphere. Only a wholly offscreen subtree may omit raster error.
        if (streamDemandBoundsVisible(low, high, instance, params)) {
            upperError = max(upperError, meshletLodProjectedError(sphere, error,
                instance.world0, instance.world1, instance.world2, instance.world3,
                eye, forward, projection, worldPadding));
        } else {
            pruneThreshold = offscreenPruneThreshold;
        }
    } else {
        upperError = streamLodMetric(sphere, error,
            instance, eye, forward, projection, params, true, worldPadding);
    }
    // Outward-rounded node spheres plus a small conservative comparison guard
    // prevent floating point cancellation at an LOD threshold from dropping a
    // desired leaf. Non-finite upper bounds also descend; leaves use exact error.
    return !(upperError < pruneThreshold);
}

// Optional telemetry follows sparse IDs and any RT demand history: two words
// per group and one full-width auto-visible frame stamp per instance. No scan
// is required per frame; the runtime resets this state when the 26-bit tag wraps.
uint streamLodHistoryBase(uint base, uint groupCount, GPUDrivenStreamAssetParams params)
{
    return base + groupCount * 3u + 4u + (streamRayLodEnabled(params)
        ? 3u + groupCount * 2u : 0u);
}

bool beginStreamLodTelemetry(RWStructuredBuffer<uint> state, uint base, uint groupCount,
    bool visible, GPUDrivenStreamAssetParams params)
{
    if (params.enableLodTransitionTelemetry == 0u) { return false; }
    uint address = streamLodHistoryBase(base, groupCount, params) + groupCount * 2u;
    uint previous = state[address];
    bool eligible = visible && params.enableGpuLodSelection != 0u &&
        (params.frameIndex & kLODTelemetryFrameMask) != 0u && params.frameIndex != 0xffffffffu;
    bool continuous = eligible && previous == params.frameIndex;
    state[address] = eligible ? params.frameIndex + 1u : 0u;
    return continuous;
}

void observeStreamLodGroup(RWStructuredBuffer<uint> state, StructuredBuffer<uint> topology,
    GPUDrivenStreamAssetPrimitive primitive, GPUDrivenStreamAssetGroup group, GPUDrivenStreamAssetInstance instance,
    uint base, uint localGroup, bool demanded, bool drawable, bool active,
    bool continuous, GPUDrivenStreamAssetParams params)
{
    if (params.enableLodTransitionTelemetry == 0u) { return; }
    uint history = streamLodHistoryBase(base, primitive.groupCount, params);
    uint address = history + localGroup * 2u;
    uint previous = state[address];
    uint tag = params.frameIndex & kLODTelemetryFrameMask;
    bool eligible = params.enableGpuLodSelection != 0u && tag != 0u && params.frameIndex != 0xffffffffu;
    bool previousObserved = eligible && (previous >> 6u) == tag - 1u &&
        (previous & kLODObservationEligible) != 0u;
    bool previousDemand = previousObserved && (previous & kLODObservedDemand) != 0u;
    bool previousSelected = previousObserved && (previous & kLODObservedSelected) != 0u;
    bool parentBlocked = false;
    for (uint parent = 0u; parent < group.parentCount; ++parent) {
        uint id = topology[group.parentOffset + parent] - primitive.groupOffset;
        parentBlocked = parentBlocked || state[history + id * 2u + 1u] == params.frameIndex + 1u;
    }
    bool ownBlocked = demanded && !drawable;
    bool dependencyBlocked = demanded && drawable && parentBlocked;
    bool catchup = eligible && continuous && active && previousDemand &&
        state[address + 1u] == params.frameIndex;
    bool threshold = false;
    if (eligible && continuous && active && !previousDemand && (group.flags & 1u) == 0u && !streamRayLodEnabled(params)) {
        // A missing observation is not evidence of a threshold crossing: the
        // demand BVH may have pruned an offscreen group. Re-evaluate the actual
        // preceding camera and threshold and require its refinement to be visible.
        GPUDrivenStreamAssetParams previousParams = params;
        previousParams.eye = params.previousEye;
        previousParams.center = params.previousCenter;
        previousParams.upProjection = params.previousUpProjection;
        previousParams.viewport = params.previousViewport;
        previousParams.clipOrtho = params.previousClipOrtho;
        previousParams.lodPixelError = params.previousLodPixelError;
        float3 previousForward = safeNormalize(previousParams.center.xyz - previousParams.eye.xyz, float3(0, 0, -1));
        bool previouslyFine = meshletLodGroupNeedsFine(
            float4(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ, group.boundsRadius),
            group.maxQuadricError, group.lodLevel, group.flags,
            instance.world0, instance.world1, instance.world2, instance.world3,
            float4(previousParams.eye.xyz, previousParams.clipOrtho.x),
            float4(previousForward, previousParams.upProjection.w),
            float4(previousParams.viewport.z, tan(previousParams.viewport.w * 0.5),
                previousParams.clipOrtho.z, previousParams.lodPixelError), 0xffffffffu);
        float3 low = float3(group.demandMinX, group.demandMinY, group.demandMinZ);
        float3 high = float3(group.demandMaxX, group.demandMaxY, group.demandMaxZ);
        // Visibility's conservative invalid-bounds fallback is suitable for
        // rendering, but cannot establish that the prior refinement was in view.
        threshold = !previouslyFine && all(isfinite(low)) && all(isfinite(high)) && all(low <= high) &&
            streamRefinementDemandVisible(group, instance, previousParams);
    }
    state[address] = (tag << 6u) | (eligible ? kLODObservationEligible : 0u) |
        (demanded ? kLODObservedDemand : 0u) | (previousSelected ? kLODPreviouslySelected : 0u) |
        (threshold ? kLODThresholdCandidate : 0u) | (catchup ? kLODCatchupCandidate : 0u);
    if (ownBlocked || dependencyBlocked) { state[address + 1u] = params.frameIndex + 1u; }
    if ((group.flags & 1u) == 0u && params.demandStatsBuffer != 0xffffffffu && eligible) {
        RWStructuredBuffer<uint> stats = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.demandStatsBuffer));
        if (demanded) { InterlockedAdd(stats[19], 1u); }
        if (ownBlocked) { InterlockedAdd(stats[20], 1u); }
        if (dependencyBlocked) { InterlockedAdd(stats[21], 1u); }
        if (catchup) { InterlockedAdd(stats[22], 1u); }
    }
}

void selectStreamLodTelemetry(RWStructuredBuffer<uint> state, uint base, uint groupCount,
    uint localGroup, uint mask, bool terminal, bool fallback, GPUDrivenStreamAssetParams params)
{
    if (params.enableLodTransitionTelemetry == 0u || terminal || fallback || mask == 0u) { return; }
    uint address = streamLodHistoryBase(base, groupCount, params) + localGroup * 2u;
    uint observation = state[address];
    state[address] = observation | kLODObservedSelected;
    if (params.demandStatsBuffer == 0xffffffffu || (observation & kLODObservationEligible) == 0u) { return; }
    uint counter = (observation & kLODCatchupCandidate) != 0u ? 23u :
        (observation & kLODThresholdCandidate) != 0u ? 25u :
        (observation & kLODPreviouslySelected) == 0u ? 27u : 0u;
    if (counter != 0u) {
        RWStructuredBuffer<uint> stats = resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(params.demandStatsBuffer));
        InterlockedAdd(stats[counter], 1u);
        InterlockedAdd(stats[counter + 1u], countbits(mask));
    }
}

void buildStreamLodBvhFrontier(uint instanceIndex,
    GPUDrivenStreamAssetInstance instance, GPUDrivenStreamAssetPrimitive primitive,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups, StructuredBuffer<uint> topology,
    RWStructuredBuffer<uint> state,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests, GPUDrivenStreamAssetParams params, bool telemetryContinuous)
{
    uint header = instanceIndex * 4u;
    uint base = topology[params.lodInstanceOffsetsOffset + instanceIndex];
    // After the existing active/mask pairs: all-active count, node visits,
    // group tests, reserved; then groupCount sparse local group IDs.
    uint sparse = base + primitive.groupCount * 2u;
    uint previousCount = state[sparse];
    for (uint index = 0u; index < previousCount; ++index) {
        uint localGroup = state[sparse + 4u + index];
        state[base + localGroup * 2u] = 0u;
        state[base + localGroup * 2u + 1u] = 0u;
    }
    state[sparse] = 0u;
    state[sparse + 1u] = 0u;
    state[sparse + 2u] = 0u;
    state[sparse + 3u] = 0u;
    if (instance.visible == 0u) { return; }

    float3 delta = params.center.xyz - params.eye.xyz;
    float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
    float4 lodEye = float4(params.eye.xyz, params.clipOrtho.x);
    float4 lodForward = float4(forward, params.upProjection.w);
    float4 lodProjection = float4(params.viewport.z, tan(params.viewport.w * 0.5),
        params.clipOrtho.z, params.lodPixelError);
    uint manualLevel = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
    uint activeCount = 0u, rootCount = 0u, visitedNodes = 0u, testedGroups = 0u;
    bool rootsReady = true;
    uint nodeIndex = 0u;
    while (nodeIndex < primitive.lodBvhNodeCount) {
        uint offset = primitive.lodBvhOffset + nodeIndex * kStreamLodBvhNodeWords;
        ++visitedNodes;
        if (!streamLodBvhNeedsFine(topology, offset, instance,
                lodEye, lodForward, lodProjection, manualLevel, params)) {
            nodeIndex = topology[offset + 7u];
            continue;
        }
        uint firstGroup = topology[offset + 8u];
        uint groupCount = topology[offset + 9u];
        for (uint remaining = groupCount; remaining != 0u; --remaining) {
            uint localGroup = firstGroup + remaining - 1u;
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
            ++testedGroups;
            bool terminal = (group.flags & 1u) != 0u;
            bool parentsActive = true;
            for (uint parent = 0u; parent < group.parentCount; ++parent) {
                uint parentId = topology[group.parentOffset + parent];
                parentsActive = parentsActive && state[base + (parentId - primitive.groupOffset) * 2u] != 0u;
            }
            // Demand comes from resident metadata, independently of the safe cut.
            bool demanded = streamLodGroupDemand(group, instance, lodEye, lodForward, lodProjection,
                manualLevel, params, state, base, primitive.groupCount, localGroup);
            if (demanded) {
                if (streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) {
                    markStreamPageDemand(pageTable, params, group.pageIndex);
                } else {
                    requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
                }
            }
            bool drawable = groupPageDrawable(pageTable, params, group);
            observeStreamLodGroup(state, topology, primitive, group, instance, base, localGroup,
                demanded, drawable, parentsActive && demanded && drawable, telemetryContinuous, params);
            if (parentsActive && demanded && drawable) {
                state[base + localGroup * 2u] = 1u;
                state[sparse + 4u + activeCount++] = localGroup;
            }
            if (terminal) {
                rootsReady = rootsReady && drawable;
                ++rootCount;
            }
        }
        ++nodeIndex;
    }
    // Keep ALL active IDs, including fully replaced ancestors and incomplete
    // primitives, so sparse clearing cannot leave stale reachability next frame.
    state[sparse] = activeCount;
    state[sparse + 1u] = visitedNodes;
    state[sparse + 2u] = testedGroups;
    uint selectedCount = 0u;
    if (rootsReady) {
        for (uint remaining = activeCount; remaining != 0u; --remaining) {
            uint localGroup = state[sparse + 4u + remaining - 1u];
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
            uint mask = 0u;
            for (uint cluster = 0u; cluster < group.clusterCount; ++cluster) {
                uint refined = topology[group.clusterRefinedOffset + cluster];
                if (refined == kInvalidClusterIndex ||
                    state[base + (refined - primitive.groupOffset) * 2u] == 0u) {
                    mask |= 1u << cluster;
                }
            }
            state[base + localGroup * 2u + 1u] = mask;
            selectedCount += mask != 0u ? 1u : 0u;
        }
    }
    state[header] = selectedCount;
    state[header + 2u] = rootsReady ? rootCount : 0u;
    state[header + 3u] = rootsReady ? 1u : 0u;
}

// The LOD topology is resident even when page payloads are absent. State has
// four header words per instance (count, prefix, roots, ready), followed by
// compact per-instance pairs (active group, selected cluster mask).
void buildStreamLodFrontier(
    uint instanceIndex,
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances,
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> topology,
    RWStructuredBuffer<uint> state,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetPush push)
{
    uint header = instanceIndex * 4u;
    state[header] = 0u;
    state[header + 1u] = 0u;
    state[header + 2u] = 0u;
    state[header + 3u] = 0u;
    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        return;
    }
    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    // Frustum rejection is current-frame and independent of HZB history. Keep
    // the ordinary state clearing below, so a returning instance cannot reuse
    // stale parent reachability. Manual cut/debug and RT traversal stay intact.
    if (params.enableGpuLodSelection != 0u && !streamRayLodEnabled(params) && push.rasterBindingsBuffer != 0xffffffffu) {
        StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindings =
            resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));
        if ((bindings[0].cullingFlags & 1u) != 0u &&
            !streamSphereInsideFrustum(instance.boundsCenterRadius.xyz,
                max(instance.boundsCenterRadius.w, 0.000001) + bindings[0].displacementBound, currentStreamCamera(params))) {
            instance.visible = 0u;
        }
    }
    uint telemetryBase = topology[params.lodInstanceOffsetsOffset + instanceIndex];
    prepareStreamRayLodHistory(state, telemetryBase, primitive.groupCount, instance.visible != 0u, params);
    bool telemetryContinuous = beginStreamLodTelemetry(state, telemetryBase, primitive.groupCount,
        instance.visible != 0u, params);
    if (primitive.lodBvhNodeCount != 0u) {
        buildStreamLodBvhFrontier(instanceIndex, instance, primitive, groups,
            topology, state, pageTable, requests, params, telemetryContinuous);
        return;
    }
    // Metadata-free fixtures retain the brute-force traversal as an oracle.
    if (instance.visible == 0u) { return; }
    uint base = topology[params.lodInstanceOffsetsOffset + instanceIndex];
    bool rootsReady = true;
    uint rootCount = 0u;
    float3 delta = params.center.xyz - params.eye.xyz;
    float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
    float4 lodEye = float4(params.eye.xyz, params.clipOrtho.x);
    float4 lodForward = float4(forward, params.upProjection.w);
    float4 lodProjection = float4(params.viewport.z, tan(params.viewport.w * 0.5),
        params.clipOrtho.z, params.lodPixelError);
    uint manualLevel = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
    // Asset validation guarantees refined < owner. The reverse order therefore
    // evaluates every parent before its descendants, without a work queue cap.
    for (uint remaining = primitive.groupCount; remaining != 0u; --remaining) {
        uint localGroup = remaining - 1u;
        GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
        bool terminal = (group.flags & 1u) != 0u;
        bool parentsActive = true;
        for (uint parent = 0u; parent < group.parentCount; ++parent) {
            uint parentId = topology[group.parentOffset + parent];
            parentsActive = parentsActive && state[base + (parentId - primitive.groupOffset) * 2u] != 0u;
        }
        bool demanded = streamLodGroupDemand(group, instance, lodEye, lodForward, lodProjection,
            manualLevel, params, state, base, primitive.groupCount, localGroup);
        if (demanded) {
            if (streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) {
                markStreamPageDemand(pageTable, params, group.pageIndex);
            } else {
                requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
            }
        }
        bool drawable = groupPageDrawable(pageTable, params, group);
        bool active = parentsActive && demanded && drawable;
        observeStreamLodGroup(state, topology, primitive, group, instance, base, localGroup,
            demanded, drawable, active, telemetryContinuous, params);
        state[base + localGroup * 2u] = active ? 1u : 0u;
        state[base + localGroup * 2u + 1u] = 0u;
        if (terminal) {
            rootsReady = rootsReady && drawable;
            ++rootCount;
        }
    }
    // Publish only complete primitives. PendingUpload pages never replace
    // their coarse ancestors, and an incomplete terminal cut stays unready.
    uint selectedCount = 0u;
    if (rootsReady) {
        for (uint localGroup = 0u; localGroup < primitive.groupCount; ++localGroup) {
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
            uint mask = 0u;
            if (state[base + localGroup * 2u] != 0u) {
                for (uint cluster = 0u; cluster < group.clusterCount; ++cluster) {
                    uint refined = topology[group.clusterRefinedOffset + cluster];
                    if (refined == kInvalidClusterIndex ||
                        state[base + (refined - primitive.groupOffset) * 2u] == 0u) {
                        mask |= 1u << cluster;
                    }
                }
            }
            state[base + localGroup * 2u + 1u] = mask;
            selectedCount += mask != 0u ? 1u : 0u;
        }
    }
    state[header] = selectedCount;
    state[header + 2u] = rootsReady ? rootCount : 0u;
    state[header + 3u] = rootsReady ? 1u : 0u;
}

public groupshared uint2 g_StreamLodPrefix[64];

// Saturate before addition so even malformed/extreme counts cannot wrap.
uint2 scanStreamLodCounts(uint2 value, uint lane, uint limit)
{
    g_StreamLodPrefix[lane] = min(value, limit);
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 64u; stride *= 2u) {
        uint2 previous = lane >= stride ? g_StreamLodPrefix[max(lane, stride) - stride] : uint2(0u);
        GroupMemoryBarrierWithGroupSync();
        uint2 current = g_StreamLodPrefix[lane];
        g_StreamLodPrefix[lane] = current + min(previous, limit - current);
        GroupMemoryBarrierWithGroupSync();
    }
    return lane == 0u ? uint2(0u) : g_StreamLodPrefix[lane - 1u];
}

void prefixStreamLodFrontier(
    RWStructuredBuffer<uint> state,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    GPUDrivenStreamAssetParams params, uint lane)
{
    uint perLane = (params.sceneInstanceCount + 63u) / 64u;
    uint first = min(lane * perLane, params.sceneInstanceCount);
    uint end = min(first + perLane, params.sceneInstanceCount);
    uint limit = params.activeGroupCount + 1u;
    uint2 total = 0u;
    for (uint i = first; i < end; ++i) {
        uint selected = state[i * 4u];
        uint baseline = min(selected, state[i * 4u + 2u]);
        total += min(uint2(selected, baseline), limit - total);
    }
    uint2 before = scanStreamLodCounts(total, lane, limit);
    uint2 all = g_StreamLodPrefix[63];
    bool overflow = all.x > params.activeGroupCount;
    bool invalidCapacity = all.y > params.activeGroupCount;
    if (overflow && !invalidCapacity) {
        // Reserve complete coverage for every instance, then spend remaining
        // slots in stable instance order. Detail cuts cheaper than their roots
        // are always kept. One view-wide overflow no longer coarsens everyone.
        // A second scan compacts the mixed full-detail / terminal output.
        uint extra = 0u;
        for (uint i = first; i < end; ++i) {
            uint selected = state[i * 4u];
            uint baseline = min(selected, state[i * 4u + 2u]);
            extra += min(selected - baseline, limit - extra);
        }
        GroupMemoryBarrierWithGroupSync();
        uint used = scanStreamLodCounts(uint2(extra, 0u), lane, limit).x;
        uint spare = params.activeGroupCount - all.y;
        total = 0u;
        for (uint i = first; i < end; ++i) {
            uint header = i * 4u;
            uint selected = state[header];
            uint baseline = min(selected, state[header + 2u]);
            uint cost = selected - baseline;
            bool keepDetail = cost == 0u || (used <= spare && cost <= spare - used);
            used += min(cost, limit - used);
            // bit 0: complete roots ready; bit 1: emit the terminal cut.
            if (!keepDetail) { state[header + 3u] |= 2u; }
            total.x += keepDetail ? selected : baseline;
            total.y += !keepDetail ? 1u : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        before = scanStreamLodCounts(total, lane, limit);
    }
    uint prefix = before.x;
    for (uint i = first; i < end; ++i) {
        uint header = i * 4u;
        state[header + 1u] = prefix;
        prefix += (state[header + 3u] & 2u) != 0u ? state[header + 2u] : state[header];
    }
    if (lane == 63u) {
        activeHeader[0].activeGroupCount = invalidCapacity ? 0u : g_StreamLodPrefix[63].x;
        activeHeader[0].overflowCount = invalidCapacity ? 2u : (overflow ? 1u : 0u);
        activeHeader[0].padding0 = overflow ? 1u : 0u;
        activeHeader[0].padding1 = invalidCapacity ? 1u : 0u;
        activeHeader[0].padding2 = overflow && !invalidCapacity ? g_StreamLodPrefix[63].y : 0u;
    }
}

void emitStreamLodFrontier(
    uint instanceIndex,
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances,
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> topology,
    RWStructuredBuffer<uint> state,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    GPUDrivenStreamAssetParams params)
{
    uint header = instanceIndex * 4u;
    if (state[header + 3u] == 0u || activeHeader[0].padding1 != 0u) {
        return;
    }
    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    uint base = topology[params.lodInstanceOffsetsOffset + instanceIndex];
    uint outputIndex = state[header + 1u];
    bool fallback = (state[header + 3u] & 2u) != 0u;
    bool sparseTraversal = primitive.lodBvhNodeCount != 0u;
    uint sparse = base + primitive.groupCount * 2u;
    uint groupCount = sparseTraversal ? state[sparse] : primitive.groupCount;
    for (uint index = 0u; index < groupCount; ++index) {
        uint localGroup = sparseTraversal ? state[sparse + 4u + groupCount - 1u - index] : index;
        GPUDrivenStreamAssetGroup source = groups[primitive.groupOffset + localGroup];
        uint mask = state[base + localGroup * 2u + 1u];
        if (fallback) {
            mask = (source.flags & 1u) == 0u ? 0u :
                (source.clusterCount == 32u ? 0xffffffffu : (1u << source.clusterCount) - 1u);
        }
        if (mask == 0u) {
            continue;
        }
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
        group.padding1 = 0u;
        group.padding2 = 0u;
        group.world0 = instance.world0;
        group.world1 = instance.world1;
        group.world2 = instance.world2;
        group.world3 = instance.world3;
        activeGroups[outputIndex++] = group;
    }
}

// Bounded subtrees form a global queue across visible instances. The preceding
// clear/seed dispatch publishes all task payloads before workers claim them.
// Safe-cut state is computed later, after a dispatch barrier, in the original
// parent-first order. There is no producer polling inside this dispatch.
