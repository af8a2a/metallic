#if defined(BATCH_PROBE_INLINE)
#include "../BatchBarrierProbeParameters.h"
[[vk::push_constant]] ConstantBuffer<BatchBarrierProbeParameters> gProbe;
#define gOutput gProbe.output.data
#else
import Core;
using Metallic;
#define gOutput getResource<RWStructuredBuffer<uint>>(0)

#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void batchBarrierMain(uint3 thread : SV_DispatchThreadID)
{
#if defined(BATCH_PROBE_INLINE)
    if (!gProbe.output.contains(0u)) { return; }
#endif
    gOutput[0] += 1;
}
