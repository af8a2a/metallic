import Core;
using Metallic;
import Lighting;
using Metallic.Lighting;

[shader("compute")]
[numthreads(64, 1, 1)]
void environmentPrefilterFieldProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint count = kEnvironmentFilterWidth * kEnvironmentFilterHeight * kEnvironmentFilterLayers;
    if (id.x < count) {
        getResource<RWStructuredBuffer<float4>>(0)[id.x] =
            getResource<StructuredBuffer<float4>>(1)[id.x];
    } else if (id.x < count + 2u) {
        uint width, height, levels;
        getResource<Texture2D<float4>>(2).GetDimensions(0, width, height, levels);
        getResource<RWStructuredBuffer<float4>>(0)[id.x] = id.x == count
            ? float4(getResource<Texture2D<float4>>(2).Load(int3(0, 0, levels - 1)).rgb, float(levels))
            : float4(getResource<Texture2D<float4>>(2).Load(int3(width / 2, height / 2, 0)).r,
                getResource<Texture2D<float4>>(2).Load(int3(0, 0, 0)).r, 0.0, 0.0);
    }
}
