module LumenScreenProbeSchedule;

import ShaderCore;
import ParameterRoot;
import LumenAdaptiveProbes;
import LumenScreenProbeWork;

using Metallic;
using Metallic.Lumen;

struct LumenScreenProbeScheduleParams
{
    ResourceHandle<StructuredBuffer<uint2>> probePositions;
    ResourceHandle<RWStructuredBuffer<uint>> workList;
    ResourceHandle<RWStructuredBuffer<uint>> workState;
    ResourceHandle<RWStructuredBuffer<uint>> indirectArgs;
    uint capacity;
#if defined(METALLIC_LUMEN_SPLIT_PROBE_TRACING)
    ResourceHandle<StructuredBuffer<float4>> traceRadiance;
    ResourceHandle<RWStructuredBuffer<uint>> rayWorkList;
    ResourceHandle<RWStructuredBuffer<uint>> rayWorkCount;
    ResourceHandle<RWStructuredBuffer<uint>> rayIndirectArgs;
#endif
};

#if defined(METALLIC_LUMEN_SPLIT_PROBE_TRACING)
static const uint kRayCompactThreadGroupSize = 16u;
static const uint kRayCompactThreads = kRayCompactThreadGroupSize * kRayCompactThreadGroupSize;
static const uint kRaysPerProbe = kLumenProbeTraceResolution * kLumenProbeTraceResolution;
#if defined(METALLIC_LUMEN_COMPACT_DETAIL_RAYS)
static const uint kRayWorkCountIndex = 1u;
static const uint kRayTraceDispatchOffset = 6u;
#else
static const uint kRayWorkCountIndex = 0u;
static const uint kRayTraceDispatchOffset = 3u;
#endif

void writeScreenProbeRayDispatch(RWStructuredBuffer<uint> args, uint offset, uint groups)
{
    args[offset] = min(groups, kLumenProbeDispatchWidth);
    args[offset + 1u] = (groups + kLumenProbeDispatchWidth - 1u) / kLumenProbeDispatchWidth;
    args[offset + 2u] = 1u;
}
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void lumenResetScreenProbeScheduleMain()
{
    let params = getParameters<LumenScreenProbeScheduleParams>();
    let state = resolveUniform(params.workState);
    state[0] = 0u;
    state[1] = 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenCompactScreenProbesMain(uint3 thread : SV_DispatchThreadID)
{
    let params = getParameters<LumenScreenProbeScheduleParams>();
    const uint slot = thread.x;
    if (slot >= params.capacity) { return; }
    const bool active = resolveUniform(params.probePositions)[slot].x != kLumenInvalidProbe;
    let state = resolveUniform(params.workState);
    let list = resolveUniform(params.workList);

#if defined(METALLIC_LUMEN_PROBE_COMPACTION_WAVE_OPS)
    // UE CompactTraces reduces allocator contention by reserving a range for
    // multiple lanes. The fixed atlas needs a separate list for invalid slots.
    const uint activeCount = WaveActiveCountBits(active);
    const uint inactiveCount = WaveActiveCountBits(!active);
    const uint activeOffset = WavePrefixCountBits(active);
    const uint inactiveOffset = WavePrefixCountBits(!active);
    uint activeBase = 0u;
    uint inactiveBase = 0u;
    if (WaveIsFirstLane()) {
        if (activeCount != 0u) { InterlockedAdd(state[0], activeCount, activeBase); }
        if (inactiveCount != 0u) { InterlockedAdd(state[1], inactiveCount, inactiveBase); }
    }
    activeBase = WaveReadLaneFirst(activeBase);
    inactiveBase = WaveReadLaneFirst(inactiveBase);
    const uint index = active ? activeBase + activeOffset : params.capacity + inactiveBase + inactiveOffset;
#else
    uint rank;
    InterlockedAdd(state[active ? 0u : 1u], 1u, rank);
    const uint index = rank + (active ? 0u : params.capacity);
#endif
    list[index] = slot;
}

// UE SetupAdaptiveProbeIndirectArgsCS derives the dispatch from the actual
// probe count. The fixed-width rows keep large lists within Vulkan limits.
[shader("compute")]
[numthreads(1, 1, 1)]
void lumenBuildScreenProbeIndirectArgsMain()
{
    let params = getParameters<LumenScreenProbeScheduleParams>();
    let state = resolveUniform(params.workState);
    let args = resolveUniform(params.indirectArgs);
    for (uint index = 0u; index < 2u; ++index) {
        const uint count = state[index];
        args[index * 3u] = min(count, kLumenProbeDispatchWidth);
        args[index * 3u + 1u] = (count + kLumenProbeDispatchWidth - 1u) / kLumenProbeDispatchWidth;
        args[index * 3u + 2u] = 1u;
    }
#if defined(METALLIC_LUMEN_SPLIT_PROBE_TRACING)
    const uint rayCount = state[0] * kRaysPerProbe;
    let rayWorkCount = resolveUniform(params.rayWorkCount);
    rayWorkCount[0] = 0u;
#if defined(METALLIC_LUMEN_SPLIT_DETAIL_TRACING)
    rayWorkCount[1] = 0u;
#endif
    writeScreenProbeRayDispatch(resolveUniform(params.rayIndirectArgs), 0u,
        (rayCount + kRayCompactThreads - 1u) / kRayCompactThreads);
#endif
}

#if defined(METALLIC_LUMEN_SPLIT_PROBE_TRACING)
groupshared uint SharedGlobalTraceTexelStartOffset;
#if defined(METALLIC_LUMEN_PROBE_COMPACTION_WAVE_OPS)
groupshared uint SharedGroupSum;
#else
groupshared uint SharedTraceTexelAllocator;
groupshared uint SharedTraceTexels[kRayCompactThreads];
#endif

// UE ScreenProbeCompactTracesCS, using Metallic's placed probe indices and pending trace markers.
[shader("compute")]
[numthreads(kRayCompactThreadGroupSize, kRayCompactThreadGroupSize, 1)]
void lumenCompactScreenProbeRaysMain(uint3 group : SV_GroupID, uint linearThreadIndex : SV_GroupIndex)
{
    let params = getParameters<LumenScreenProbeScheduleParams>();
#if defined(METALLIC_LUMEN_PROBE_COMPACTION_WAVE_OPS)
    if (linearThreadIndex == 0u) { SharedGroupSum = 0u; }
#else
    if (linearThreadIndex == 0u) { SharedTraceTexelAllocator = 0u; }
    GroupMemoryBarrierWithGroupSync();
#endif

    bool bTraceValid = false;
    uint traceTexelForThisThread = 0u;
    const uint rayIndex = (group.y * kLumenProbeDispatchWidth + group.x) * kRayCompactThreads + linearThreadIndex;
    const uint activeProbeIndex = rayIndex / kRaysPerProbe;
    if (activeProbeIndex < resolveUniform(params.workState)[0]) {
        const uint probeIndex = resolveUniform(params.workList)[activeProbeIndex];
        traceTexelForThisThread = probeIndex * kRaysPerProbe + rayIndex % kRaysPerProbe;
        const float traceState = resolveUniform(params.traceRadiance)[traceTexelForThisThread].w;
#if defined(METALLIC_LUMEN_COMPACT_DETAIL_RAYS)
        bTraceValid = traceState == kLumenProbeRayPendingDetail;
#elif defined(METALLIC_LUMEN_SPLIT_DETAIL_TRACING)
        bTraceValid = traceState == kLumenProbeRayPending || traceState == kLumenProbeRayPendingDetail;
#else
        bTraceValid = traceState == kLumenProbeRayPending;
#endif
    }
#if !defined(METALLIC_LUMEN_PROBE_COMPACTION_WAVE_OPS)
    if (bTraceValid) {
        uint sharedTexelOffset;
        InterlockedAdd(SharedTraceTexelAllocator, 1u, sharedTexelOffset);
        SharedTraceTexels[sharedTexelOffset] = traceTexelForThisThread;
    }
#endif

    GroupMemoryBarrierWithGroupSync();

    let compactedTraceTexelAllocator = resolveUniform(params.rayWorkCount);
    let compactedTraceTexelData = resolveUniform(params.rayWorkList);
#if defined(METALLIC_LUMEN_PROBE_COMPACTION_WAVE_OPS)
    const uint lastLaneIndex = WaveGetLaneCount() - 1u;
    const uint laneIndex = WaveGetLaneIndex();
    const uint offsetInWave = WavePrefixCountBits(bTraceValid);
    uint offsetInGroup = 0u;
    if (laneIndex == lastLaneIndex) {
        const uint thisWaveSum = offsetInWave + (bTraceValid ? 1u : 0u);
        InterlockedAdd(SharedGroupSum, thisWaveSum, offsetInGroup);
    }
    offsetInGroup = WaveReadLaneAt(offsetInGroup, lastLaneIndex) + offsetInWave;

    GroupMemoryBarrierWithGroupSync();

    if (linearThreadIndex == 0u) {
        InterlockedAdd(compactedTraceTexelAllocator[kRayWorkCountIndex], SharedGroupSum, SharedGlobalTraceTexelStartOffset);
    }

    GroupMemoryBarrierWithGroupSync();

    if (bTraceValid) {
        compactedTraceTexelData[SharedGlobalTraceTexelStartOffset + offsetInGroup] = traceTexelForThisThread;
    }
#else
    if (linearThreadIndex == 0u) {
        InterlockedAdd(compactedTraceTexelAllocator[kRayWorkCountIndex], SharedTraceTexelAllocator, SharedGlobalTraceTexelStartOffset);
    }

    GroupMemoryBarrierWithGroupSync();

    if (linearThreadIndex < SharedTraceTexelAllocator) {
        compactedTraceTexelData[SharedGlobalTraceTexelStartOffset + linearThreadIndex] = SharedTraceTexels[linearThreadIndex];
    }
#endif
}

// UE SetupCompactedTracesIndirectArgsCS for the 64-thread software trace.
[shader("compute")]
[numthreads(1, 1, 1)]
void lumenBuildScreenProbeRayIndirectArgsMain()
{
    let params = getParameters<LumenScreenProbeScheduleParams>();
    const uint rayCount = resolveUniform(params.rayWorkCount)[kRayWorkCountIndex];
    writeScreenProbeRayDispatch(resolveUniform(params.rayIndirectArgs), kRayTraceDispatchOffset,
        (rayCount + kLumenProbeFallbackGroupSize - 1u) / kLumenProbeFallbackGroupSize);
}
#endif
