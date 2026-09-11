#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"
#define gMotion METALLIC_RESOURCE(Texture2D<float2>, 0)
#define gDepth METALLIC_RESOURCE(Texture2D<float>, 1)
#define gData METALLIC_RESOURCE(RWStructuredBuffer<float4>, 2)
#include "../../../Shaders/Libraries/Lighting/EnvironmentPrefilter.slang"
#define gPrefilter METALLIC_RESOURCE(StructuredBuffer<float4>, 3)

[shader("compute")]
[numthreads(8, 8, 1)]
void realtimeGuideProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    gDepth.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) { return; }
    gData[id.y * width + id.x] = float4(gMotion.Load(int3(id.xy, 0)), gDepth.Load(int3(id.xy, 0)), 1.0);
}

[shader("compute")]
[numthreads(8, 1, 1)]
void environmentPrefilterProbeMain(uint3 id : SV_DispatchThreadID)
{
    // Both poles and both sides of the lat-long wrap, across all roughnesses.
    float3 direction = environmentFilterDirection(float2(float(id.x) / 7.0, float(id.x % 3u) * 0.5));
    gData[id.x] = float4(environmentFilterLayer(gPrefilter, direction, id.x), 1.0);
}
