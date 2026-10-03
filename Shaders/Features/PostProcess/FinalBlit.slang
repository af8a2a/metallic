import ShaderCore;
import ColorGrading;
using Metallic;
using Metallic.ColorGrading;

#include "../../../Source/Runtime/Render/Core/PostProcessParameters.h"

[[vk::push_constant]] ConstantBuffer<FinalBlitParams> gParams;

float3 displayOutput(float3 color, DisplayOutputPush push)
{
#if FINAL_USE_LUT
    if ((push.inputEncoding == 1u || push.inputEncoding == 3u) && push.hasLut != 0u) {
        color = float3(isfinite(color.x) ? max(color.x, 0.0) : 0.0,
            isfinite(color.y) ? max(color.y, 0.0) : 0.0, isfinite(color.z) ? max(color.z, 0.0) : 0.0);
        return applyGradingLut(resolveUniform(gParams.lut), resolveUniform(gParams.lutSampler), color * push.exposure);
    }
#endif
    if (push.hdr != 0u) {
        if (push.inputEncoding == 2u) { return color; }
        if (push.inputEncoding == 0u) {
            return (push.sampledSrgb != 0u ? color : displaySrgbToLinear(color)) * (push.paperWhiteNits / 80.0);
        }
        return displayMapToScRgb(color * push.exposure, push.paperWhiteNits, push.peakNits);
    }
    if (push.inputEncoding == 2u) { return displayLinearToSrgb(saturate(color * (80.0 / push.paperWhiteNits))); }
    if (push.inputEncoding == 1u || push.inputEncoding == 3u) {
        color = max(color * push.exposure, 0.0);
        color = push.toneCurve == 2u ? saturate(color) :
            (push.toneCurve == 1u ? 1.0 - exp(-color) : color / (color + 1.0));
        return displayLinearToSrgb(color);
    }
    return push.sampledSrgb != 0u ? displayLinearToSrgb(color) : color;
}

#define gOutput resolveUniform(gParams.output)

#define gSource resolveUniform(gParams.source)

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
    DisplayOutputPush push = gParams.display;
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
    gOutput[pixel] = float4(displayOutput(color, gParams.display), 1.0);
}
