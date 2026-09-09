struct SliderDebugPush {
    float splitPosition;
    uint horizontal;
    uint swapSides;
};

[[vk::push_constant]] ConstantBuffer<SliderDebugPush> gPush;
[[vk::binding(0, 0)]] Texture2D<float4> gSourceA;
[[vk::binding(1, 0)]] Texture2D<float4> gSourceB;
[[vk::binding(2, 0)]] RWTexture2D<float4> gOutput;

[shader("compute")]
[numthreads(8, 8, 1)]
void sliderDebugMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) { return; }
    float coordinate = gPush.horizontal != 0 ? float(pixel.y) + 0.5 : float(pixel.x) + 0.5;
    float extent = gPush.horizontal != 0 ? float(height) : float(width);
    bool useA = coordinate < gPush.splitPosition * extent;
    if (gPush.swapSides != 0) { useA = !useA; }
    // Keep full-image coordinates: the divider reveals each image without resizing it.
    // UI draws the handle separately so captures retain unmodified source pixels.
    gOutput[pixel] = useA ? gSourceA.Load(int3(pixel, 0)) : gSourceB.Load(int3(pixel, 0));
}
