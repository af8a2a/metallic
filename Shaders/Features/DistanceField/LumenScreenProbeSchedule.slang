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
};

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
}
