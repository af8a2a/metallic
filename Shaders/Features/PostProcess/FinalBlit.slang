#include "../../Libraries/Resources/ComputeResources.slang"
#include "../../Libraries/Color/DisplayOutput.slang"

struct DisplayOutputPush
{
    uint hdr;
    uint inputEncoding;
    uint calibration;
    uint sampledSrgb;
    float paperWhiteNits;
    float peakNits;
    float exposure;
    float padding;
};

float3 displayOutput(float3 color, DisplayOutputPush push)
{
    if (push.hdr != 0u) {
        if (push.inputEncoding == 2u) { return color; }
        if (push.inputEncoding == 0u) {
            return (push.sampledSrgb != 0u ? color : displaySrgbToLinear(color)) * (push.paperWhiteNits / 80.0);
        }
        return displayMapToScRgb(color * push.exposure, push.paperWhiteNits, push.peakNits);
    }
    if (push.inputEncoding == 2u) { return displayLinearToSrgb(saturate(color * (80.0 / push.paperWhiteNits))); }
    if (push.inputEncoding == 1u) {
        color = max(color * push.exposure, 0.0);
        return displayLinearToSrgb(color / (color + 1.0));
    }
    return push.sampledSrgb != 0u ? displayLinearToSrgb(color) : color;
}

#define gOutput METALLIC_RESOURCE(RWTexture2D<float4>, 0)

#define gSource METALLIC_RESOURCE(Texture2D<float4>, 1)

[shader("compute")]
[numthreads(8, 8, 1)]
void finalBlitUvMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) {
        return;
    }
    float2 uv = (float2(pixel) + 0.5) / float2(width, height);
    DisplayOutputPush push = METALLIC_CONSTANTS(DisplayOutputPush);
    float3 color;
    if (push.calibration != 0u) {
        float3 nits = displayCalibrationNits(uv);
        color = push.hdr != 0u ? nits / 80.0 : displayLinearToSrgb(saturate(nits / push.paperWhiteNits));
    } else {
        push.inputEncoding = 0u;
        color = displayOutput(float3(uv, 0.0), push);
    }
    gOutput[pixel] = float4(color, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void finalBlitMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint width, height;
    gOutput.GetDimensions(width, height);
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= width || pixel.y >= height) {
        return;
    }

    uint sourceWidth, sourceHeight;
    gSource.GetDimensions(sourceWidth, sourceHeight);
    float2 uv = (float2(pixel) + 0.5) / float2(width, height);
    float2 sourcePixel = uv * float2(sourceWidth, sourceHeight) - 0.5;
    int2 origin = int2(floor(sourcePixel));
    int2 maximum = int2(sourceWidth, sourceHeight) - 1;
    float2 weight = frac(sourcePixel);
    // Explicit bilinear filtering also supports float RT formats without a
    // hardware linear-filter feature requirement.
    float3 c00 = gSource.Load(int3(clamp(origin, int2(0), maximum), 0)).rgb;
    float3 c10 = gSource.Load(int3(clamp(origin + int2(1, 0), int2(0), maximum), 0)).rgb;
    float3 c01 = gSource.Load(int3(clamp(origin + int2(0, 1), int2(0), maximum), 0)).rgb;
    float3 c11 = gSource.Load(int3(clamp(origin + int2(1, 1), int2(0), maximum), 0)).rgb;
    float3 color = lerp(lerp(c00, c10, weight.x), lerp(c01, c11, weight.x), weight.y);
    gOutput[pixel] = float4(displayOutput(color, METALLIC_CONSTANTS(DisplayOutputPush)), 1.0);
}
