#include "../SliderFixtureParameters.h"

[[vk::push_constant]] ConstantBuffer<SliderFixtureParameters> gPush;
#define gOutput resolveDescriptor(gPush.output)
#define gInput resolveDescriptor(gPush.input)

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
    uint index = id.y * width + id.x;
    if (!gPush.readback.contains(index)) { return; }
    gPush.readback.data[index] = gInput.Load(int3(id.xy, 0));
}
