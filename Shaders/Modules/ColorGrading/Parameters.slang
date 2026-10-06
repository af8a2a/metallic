#language slang 2026
implementing ColorGrading;

namespace Metallic.ColorGrading
{
public struct ColorGradingParameters
{
    public float4 saturation, contrast, gamma, gain, offset;
    public float4 film, filmExtra, lutWeights;
};

public struct GradingResources
{
    public Texture2D<float4> lut1, lut2, lut3, lut4;
    public Texture2D<float> reach, gamut, gamma;
    public SamplerState sampler;
};

static const float PI = 3.14159265358979323846;
float square(float x) { return x * x; }

// The shaper resolves black exactly and covers the finite FP16 radiance domain.
// Grading receives exposed scene-linear working RGB; display exposure is applied
// by the consumer before this shaper, not burned into the cube.
public float3 gradingLutEncode(float3 linear)
{
    return log2(1.0 + clamp(linear, 0.0, 65504.0) / 0.01) / log2(1.0 + 65504.0 / 0.01);
}

public float3 gradingLutDecode(float3 encoded)
{
    return (exp2(encoded * log2(1.0 + 65504.0 / 0.01)) - 1.0) * 0.01;
}

public float3 applyGradingLut(Texture3D<float4> lut, SamplerState sampler, float3 linear)
{
    uint width, height, depth;
    lut.GetDimensions(width, height, depth);
    float3 size = float3(width, height, depth);
    float3 uvw = (gradingLutEncode(linear) * (size - 1.0) + 0.5) / size;
    return lut.SampleLevel(sampler, uvw, 0).rgb;
}
} // namespace Metallic.ColorGrading
