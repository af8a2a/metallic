#language slang 2026
implementing Core;

namespace Metallic
{
public float3 displaySrgbToLinear(float3 color)
{
    return select(color <= 0.04045, color / 12.92, pow(max((color + 0.055) / 1.055, 0.0), 2.4));
}

public float3 displayLinearToSrgb(float3 color)
{
    return select(color <= 0.0031308, color * 12.92, 1.055 * pow(max(color, 0.0), 1.0 / 2.4) - 0.055);
}

// Exposed scene-linear 1 maps to paper white. A continuous shoulder above
// paper white approaches the display peak while preserving RGB ratios.
public float3 displayMapToScRgb(float3 color, float paperWhiteNits, float peakNits)
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

// ST 2084 uses absolute luminance (10000 nits is code value 1).
public float3 displayNitsToPQ(float3 nits)
{
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    float3 p = pow(saturate(nits / 10000.0), m1);
    return pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
}

public float3 displayScRgbToHDR10(float3 color)
{
    // Linear Rec.709/D65 -> linear BT.2020/D65, then absolute PQ.
    float3 rec2020 = float3(
        dot(color, float3(0.627403896, 0.329283038, 0.043313066)),
        dot(color, float3(0.069097289, 0.919540395, 0.011362316)),
        dot(color, float3(0.016391439, 0.088013308, 0.895595253)));
    return displayNitsToPQ(rec2020 * 80.0);
}

public float3 displayCalibrationNits(float2 uv)
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
} // namespace Metallic
