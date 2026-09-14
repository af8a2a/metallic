#ifndef METALLIC_DISPLAY_OUTPUT_SLANG
#define METALLIC_DISPLAY_OUTPUT_SLANG

float3 displaySrgbToLinear(float3 color)
{
    return select(color <= 0.04045, color / 12.92, pow(max((color + 0.055) / 1.055, 0.0), 2.4));
}

float3 displayLinearToSrgb(float3 color)
{
    return select(color <= 0.0031308, color * 12.92, 1.055 * pow(max(color, 0.0), 1.0 / 2.4) - 0.055);
}

// Exposed scene-linear 1 maps to paper white. A continuous shoulder above
// paper white approaches the display peak while preserving RGB ratios.
float3 displayMapToScRgb(float3 color, float paperWhiteNits, float peakNits)
{
    color = float3(isfinite(color.x) ? max(color.x, 0.0) : 0.0,
        isfinite(color.y) ? max(color.y, 0.0) : 0.0,
        isfinite(color.z) ? max(color.z, 0.0) : 0.0);
    float maximum = max(color.x, max(color.y, color.z));
    float headroom = max(peakNits - paperWhiteNits, 0.0);
    float nits = maximum * paperWhiteNits;
    if (maximum > 1.0) {
        nits = paperWhiteNits + headroom *
            (1.0 - exp(-(maximum - 1.0) * paperWhiteNits / max(headroom, 0.001)));
    }
    return maximum > 0.0 ? (color / maximum) * (nits / 80.0) : float3(0.0);
}

float3 displayCalibrationNits(float2 uv)
{
    if (uv.y < 0.5) {
        const float levels[4] = {80.0, 203.0, 400.0, 1000.0};
        return float3(levels[min(uint(uv.x * 4.0), 3u)]);
    }
    // Lower half: neutral and RGB ramps from black to 1000 nits.
    uint band = min(uint((uv.y - 0.5) * 8.0), 3u);
    float3 tint = band == 0u ? float3(1.0) :
        (band == 1u ? float3(1.0, 0.0, 0.0) : (band == 2u ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0)));
    return tint * uv.x * 1000.0;
}

#endif
