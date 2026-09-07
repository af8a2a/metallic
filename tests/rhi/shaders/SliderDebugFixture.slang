[[vk::binding(0, 0)]] RWTexture2D<float4> gOutput;
[[vk::binding(1, 0)]] Texture2D<float4> gInput;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> gReadback;
struct FixturePush { uint path; };
[[vk::push_constant]] ConstantBuffer<FixturePush> gPush;

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
