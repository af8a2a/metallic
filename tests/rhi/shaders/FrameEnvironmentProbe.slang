#include "../FrameEnvironmentProbeParameters.h"

[[vk::push_constant]] ConstantBuffer<FrameEnvironmentProbeParameters> gProbe;

[shader("compute")]
[numthreads(1, 1, 1)]
void readEnvironment(uint3 thread : SV_DispatchThreadID)
{
    if (!gProbe.output.contains(0u)) { return; }
    gProbe.output.data[0] = resolveDescriptor(gProbe.environment).Load(int3(0, 0, 0));
}
