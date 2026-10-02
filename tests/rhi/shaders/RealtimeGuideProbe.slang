#include "../RealtimeProbeParameters.h"
using Metallic;
import Lighting;
using Metallic.Lighting;

[[vk::push_constant]] ConstantBuffer<RealtimeProbeParameters> gProbe;
#define gMotion resolveDescriptor(gProbe.motion)
#define gDepth resolveDescriptor(gProbe.depth)
#define gData gProbe.data.data

#define gPrefilter resolveDescriptor(gProbe.prefilter)

[shader("compute")]
[numthreads(8, 8, 1)]
void realtimeGuideProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    gDepth.GetDimensions(width, height);
    if (id.x >= width || id.y >= height || !gProbe.data.contains(id.y * width + id.x)) { return; }
    gData[id.y * width + id.x] = float4(gMotion.Load(int3(id.xy, 0)), gDepth.Load(int3(id.xy, 0)), 1.0);
}

[shader("compute")]
[numthreads(8, 1, 1)]
void environmentPrefilterProbeMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= 8u || !gProbe.data.contains(id.x)) { return; }
    // Both poles and both sides of the lat-long wrap, across all roughnesses.
    float3 direction = environmentFilterDirection(float2(float(id.x) / 7.0, float(id.x % 3u) * 0.5));
    gData[id.x] = float4(environmentFilterLayer(gPrefilter, direction, id.x), 1.0);
}
