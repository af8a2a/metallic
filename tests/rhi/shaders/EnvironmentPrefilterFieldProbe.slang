import Core;
using Metallic;
#include "../TestResourceParameters.h"
import Lighting;
using Metallic.Lighting;

[shader("compute")]
[numthreads(64, 1, 1)]
void environmentPrefilterFieldProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint count = kEnvironmentFilterWidth * kEnvironmentFilterHeight * kEnvironmentFilterLayers;
    if (id.x < count) {
        resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<EnvironmentPrefilterFieldProbeResources>().output)[id.x] =
            resolveBuffer<StructuredBuffer<float4>>(getResourceParameters<EnvironmentPrefilterFieldProbeResources>().coefficients)[id.x];
    } else if (id.x < count + 2u) {
        uint width, height, levels;
        resolveUniform(getResourceParameters<EnvironmentPrefilterFieldProbeResources>().environment).GetDimensions(0, width, height, levels);
        resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<EnvironmentPrefilterFieldProbeResources>().output)[id.x] = id.x == count
            ? float4(resolveUniform(getResourceParameters<EnvironmentPrefilterFieldProbeResources>().environment).Load(int3(0, 0, levels - 1)).rgb, float(levels))
            : float4(resolveUniform(getResourceParameters<EnvironmentPrefilterFieldProbeResources>().environment).Load(int3(width / 2, height / 2, 0)).r,
                resolveUniform(getResourceParameters<EnvironmentPrefilterFieldProbeResources>().environment).Load(int3(0, 0, 0)).r, 0.0, 0.0);
    }
}
