import Core;
using Metallic;
#define gOutput getResource<RWStructuredBuffer<uint>>(0)

[shader("compute")]
[numthreads(1, 1, 1)]
void batchBarrierMain(uint3 thread : SV_DispatchThreadID)
{
    gOutput[0] += 1;
}
