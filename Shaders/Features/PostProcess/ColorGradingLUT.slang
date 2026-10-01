import Core;
import ColorGrading;
using Metallic;
using Metallic.ColorGrading;

struct GradingPush
{
    uint transform, hdr;
    float peak, paperWhite;
    ColorGradingParameters grade;
};

[shader("compute")]
[numthreads(4, 4, 4)]
void composeColorGradingLUT(uint3 id : SV_DispatchThreadID)
{
    RWTexture3D<float4> lut = getResource<RWTexture3D<float4>>(0);
    GradingPush p = getConstants<GradingPush>();
    GradingResources resources = {
        getResource<Texture2D<float4>>(1), getResource<Texture2D<float4>>(2),
        getResource<Texture2D<float4>>(3), getResource<Texture2D<float4>>(4),
        getResource<Texture2D<float>>(5), getResource<Texture2D<float>>(6),
        getResource<Texture2D<float>>(7), getResource<SamplerState>(8)
    };
    float3 linear = gradingLutDecode(float3(id) / 63.0);
    lut[id] = float4(evaluateColorGrading(linear, p.grade, resources, p.transform,
        p.hdr != 0u, p.peak, p.paperWhite), 1.0);
}
