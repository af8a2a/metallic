#include "../EnvironmentPrefilterProbeParameters.h"
using Metallic;
import Lighting;
using Metallic.Lighting;

[[vk::push_constant]] ConstantBuffer<EnvironmentPrefilterProbeParameters> gProbe;

[shader("compute")]
[numthreads(64, 1, 1)]
void environmentPrefilterFieldProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint count = kEnvironmentFilterWidth * kEnvironmentFilterHeight * kEnvironmentFilterLayers;
    if (!gProbe.output.contains(id.x)) { return; }
    if (id.x < count) {
        if (!gProbe.prefiltered.contains(id.x)) { return; }
        gProbe.output.data[id.x] =
            gProbe.prefiltered.data[id.x];
    } else if (id.x < count + 2u) {
        uint width, height, levels;
        resolveDescriptor(gProbe.radiance).GetDimensions(0, width, height, levels);
        gProbe.output.data[id.x] = id.x == count
            ? float4(resolveDescriptor(gProbe.radiance).Load(int3(0, 0, levels - 1)).rgb, float(levels))
            : float4(resolveDescriptor(gProbe.radiance).Load(int3(width / 2, height / 2, 0)).r,
                resolveDescriptor(gProbe.radiance).Load(int3(0, 0, 0)).r, 0.0, 0.0);
    }
}
