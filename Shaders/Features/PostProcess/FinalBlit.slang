module FinalBlit;

import ShaderCore;
import ColorGrading;
using Metallic;
using Metallic.ColorGrading;

#include "../../../Source/Runtime/Render/Core/PostProcessParameters.h"

[[vk::push_constant]] ConstantBuffer<FinalBlitParams> gParams;

float3 displayOutput(float3 color, DisplayOutputPush push)
{
    if (push.inputEncoding == 4u) {
        return push.hdr != 0u ? color * (push.paperWhiteNits / 80.0)
            : displayLinearToSrgb(saturate(color));
    }
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
        return displayMapToScRgb(Metallic.WorkingColor::toLinearRec709(color) * push.exposure, push.paperWhiteNits, push.peakNits);
    }
    if (push.inputEncoding == 2u) { return displayLinearToSrgb(saturate(color * (80.0 / push.paperWhiteNits))); }
    if (push.inputEncoding == 1u || push.inputEncoding == 3u) {
        color = max(Metallic.WorkingColor::toLinearRec709(color) * push.exposure, 0.0);
        color = push.toneCurve == 2u ? saturate(color) :
            (push.toneCurve == 1u ? 1.0 - exp(-color) : color / (color + 1.0));
        return displayLinearToSrgb(color);
    }
    return push.sampledSrgb != 0u ? displayLinearToSrgb(color) : color;
}

#define gOutput resolveUniform(gParams.output)

#define gSource resolveUniform(gParams.source)

float3 sampleSourceBilinear(uint2 pixel, uint2 sourceSize, uint2 outputSize)
{
    float2 uv = (float2(pixel) + 0.5) / float2(outputSize);
    float2 sourcePixel = uv * float2(sourceSize) - 0.5;
    int2 origin = int2(floor(sourcePixel));
    int2 maximum = int2(sourceSize) - 1;
    float2 weight = frac(sourcePixel);
    // Explicit loads support float RTs without a hardware linear-filter requirement.
    float3 c00 = gSource.Load(int3(clamp(origin, int2(0), maximum), 0)).rgb;
    float3 c10 = gSource.Load(int3(clamp(origin + int2(1, 0), int2(0), maximum), 0)).rgb;
    float3 c01 = gSource.Load(int3(clamp(origin + int2(0, 1), int2(0), maximum), 0)).rgb;
    float3 c11 = gSource.Load(int3(clamp(origin + int2(1, 1), int2(0), maximum), 0)).rgb;
    return lerp(lerp(c00, c10, weight.x), lerp(c01, c11, weight.x), weight.y);
}

float3 filterSourceArea(uint2 pixel, uint2 sourceSize, uint2 outputSize, bool decodeSrgb)
{
    float2 scale = float2(sourceSize) / float2(outputSize);
    // Unit-width coverage is bilinear on an axis that is being enlarged.
    float2 footprintSize = max(scale, 1.0);
    float2 center = (float2(pixel) + 0.5) * scale;
    float2 lower = clamp(center - footprintSize * 0.5, 0.0, float2(sourceSize) - footprintSize);
    float2 upper = lower + footprintSize;
    int2 first = int2(floor(lower));
    int2 end = min(int2(ceil(upper)), int2(sourceSize));

    float3 sum = 0.0;
    for (int y = first.y; y < end.y; ++y) {
        float rowWeight = min(upper.y, float(y + 1)) - max(lower.y, float(y));
        for (int x = first.x; x < end.x; ++x) {
            float columnWeight = min(upper.x, float(x + 1)) - max(lower.x, float(x));
            float3 sample = gSource.Load(int3(x, y, 0)).rgb;
            if (decodeSrgb) { sample = displaySrgbToLinear(sample); }
            sum += sample * (columnWeight * rowWeight);
        }
    }
    return sum / (footprintSize.x * footprintSize.y);
}

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

    uint2 sourceSize;
    gSource.GetDimensions(sourceSize.x, sourceSize.y);
    uint2 outputSize = uint2(width, height);
    DisplayOutputPush push = gParams.display;
    float3 color;
    if (all(sourceSize == outputSize)) {
        color = gSource.Load(int3(pixel, 0)).rgb;
    } else if (any(sourceSize > outputSize)) {
        // sRGB image views already decode on load. Other sRGB inputs decode here before blending.
        bool decodeSrgb = push.inputEncoding == 0u && push.sampledSrgb == 0u;
        color = filterSourceArea(pixel, sourceSize, outputSize, decodeSrgb);
        if (decodeSrgb) { push.sampledSrgb = 1u; }
    } else {
        color = sampleSourceBilinear(pixel, sourceSize, outputSize);
    }
    gOutput[pixel] = float4(displayOutput(color, push), 1.0);
}
