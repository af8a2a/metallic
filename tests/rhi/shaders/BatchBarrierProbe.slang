import Core;
using Metallic;
#include "../TestResourceParameters.h"
#define gOutput resolveBuffer<RWStructuredBuffer<uint>>(getResourceParameters<BatchBarrierProbeResources>().output)

[shader("compute")]
[numthreads(1, 1, 1)]
void batchBarrierMain(uint3 thread : SV_DispatchThreadID)
{
    gOutput[0] += 1;
}
