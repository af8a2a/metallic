#include "../../Libraries/Resources/ComputeResources.slang"

struct SliderDebugPush {
    float splitPosition;
    uint horizontal;
    uint swapSides;
};

#define gPush METALLIC_CONSTANTS(SliderDebugPush)
#define gSourceA METALLIC_RESOURCE(Texture2D<float4>, 0)
#define gSourceB METALLIC_RESOURCE(Texture2D<float4>, 1)
#define gOutput METALLIC_RESOURCE(RWTexture2D<float4>, 2)

bool sliderDebugUsesSourceA(uint2 pixel, uint width, uint height)
{
    float coordinate = gPush.horizontal != 0 ? float(pixel.y) + 0.5 : float(pixel.x) + 0.5;
    float extent = gPush.horizontal != 0 ? float(height) : float(width);
    bool useA = coordinate < gPush.splitPosition * extent;
    if (gPush.swapSides != 0) { useA = !useA; }
    return useA;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sliderDebugMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) { return; }
    // Keep full-image coordinates: the divider reveals each image without resizing it.
    // UI draws the handle separately so captures retain unmodified source pixels.
    gOutput[pixel] = sliderDebugUsesSourceA(pixel, width, height)
        ? gSourceA.Load(int3(pixel, 0)) : gSourceB.Load(int3(pixel, 0));
}

// The processed B image is already in gOutput. Reveal A without sampling the
// writable output or copying the processed image to an intermediate texture.
[shader("compute")]
[numthreads(8, 8, 1)]
void sliderDebugOverlayMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) { return; }
    if (sliderDebugUsesSourceA(pixel, width, height)) {
        gOutput[pixel] = gSourceA.Load(int3(pixel, 0));
    }
}
