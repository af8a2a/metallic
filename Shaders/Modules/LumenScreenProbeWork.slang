module LumenScreenProbeWork;

import LumenAdaptiveProbes;

namespace Metallic.Lumen
{
public static const uint kLumenProbeDispatchWidth = 1024u;

public uint2 lumenProbeTileForIndex(uint probeIndex, uint probeWidth)
{
    const uint tile = probeIndex / kLumenProbeSlotsPerTile;
    return uint2(tile % probeWidth, tile / probeWidth);
}

public bool resolveLumenActiveProbe(uint3 group, StructuredBuffer<uint> workList,
    StructuredBuffer<uint> workState, out uint probeIndex)
{
    const uint index = group.y * kLumenProbeDispatchWidth + group.x;
    probeIndex = 0u;
    if (index >= workState[0]) { return false; }
    probeIndex = workList[index];
    return true;
}
} // namespace Metallic.Lumen
