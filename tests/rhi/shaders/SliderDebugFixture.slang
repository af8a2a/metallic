#include "../../../Shaders/Libraries/Resources/ComputeResources.slang"

#define gOutput METALLIC_RESOURCE(RWTexture2D<float4>, 0)
#define gInput METALLIC_RESOURCE(Texture2D<float4>, 1)
#define gReadback METALLIC_RESOURCE(RWStructuredBuffer<float4>, 2)
struct FixturePush { uint path; };
#define gPush METALLIC_CONSTANTS(FixturePush)

[shader("compute")]
[numthreads(8, 8, 1)]
void sliderFixtureMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) { return; }
    gOutput[id.xy] = float4(float(id.x) + 0.125, -float(id.y) - 0.25,
        gPush.path == 0 ? 4.5 : 12.75, gPush.path == 0 ? 0.125 : 0.75);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sliderReadbackMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    gInput.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) { return; }
    gReadback[id.y * width + id.x] = gInput.Load(int3(id.xy, 0));
}
