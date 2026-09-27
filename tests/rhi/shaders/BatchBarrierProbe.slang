#ifndef METALLIC_LEGACY_BINDINGS
#define METALLIC_LEGACY_BINDINGS 0
#endif

#if METALLIC_LEGACY_BINDINGS
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> gOutput;
#else
import Core;
using Metallic;
#define gOutput getResource<RWStructuredBuffer<uint>>(0)
#endif

[shader("compute")]
[numthreads(1, 1, 1)]
void batchBarrierMain(uint3 thread : SV_DispatchThreadID)
{
    gOutput[0] += 1;
}
