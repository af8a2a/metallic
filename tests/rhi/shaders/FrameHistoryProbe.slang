#include "../FrameHistoryProbeParameters.h"

[[vk::push_constant]] ConstantBuffer<FrameHistoryProbeParameters> gProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void accumulateHistory(uint3 thread : SV_DispatchThreadID)
{
    if (!gProbe.output.contains(gProbe.index)) { return; }
    float value = gProbe.index == 0 ? 41.0 : resolveDescriptor(gProbe.previous)[uint2(0, 0)].x + 1.0;
    resolveDescriptor(gProbe.current)[uint2(0, 0)] = float4(value, 0.0, 0.0, 1.0);
    gProbe.output.data[gProbe.index] = uint(value);
}
