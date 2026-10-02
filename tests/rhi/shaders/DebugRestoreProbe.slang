#include "../DebugRestoreProbeParameters.h"
[[vk::push_constant]] ConstantBuffer<DebugRestoreProbeParameters> gDebugRestore;

[shader("compute")]
[numthreads(1, 1, 1)]
void debugRestoreMain(uint3 id : SV_DispatchThreadID)
{
    DebugRestoreProbeParameters p = gDebugRestore;
    if (p.output.contains(p.index)) {
        p.output.data[p.index] = resolveDescriptor(p.input)[0];
    }
}
