module ShaderSpecializationProbe;

import Core;
using Metallic;
#include "../TestResourceParameters.h"

[SpecializationConstant]
const int kProbeFixedMetalness = -1;
[SpecializationConstant]
const bool kProbeOpaquePrimary = false;

[shader("compute")]
[numthreads(1, 1, 1)]
void specializationProbeMain(uint3 id : SV_DispatchThreadID)
{
    RWStructuredBuffer<uint> output = resolveBuffer<RWStructuredBuffer<uint>>(
        getResourceParameters<BatchBarrierProbeResources>().output);
    output[0] = uint(kProbeFixedMetalness);
    output[1] = kProbeOpaquePrimary ? 1u : 0u;
    output[2] = uint(kProbeFixedMetalness) ^ 0x6D3A9F27u;
    output[3] = (kProbeOpaquePrimary ? 1u : 0u) ^ 0x6D3A9F27u;
}
