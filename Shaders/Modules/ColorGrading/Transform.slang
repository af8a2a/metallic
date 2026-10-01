// Adapted from Unreal Engine 5.7.4. Copyright Epic Games, Inc. All Rights Reserved.
// See Shaders/Licenses/ColorGrading. Explicit parameters replace engine globals.
#language slang 2026
implementing ColorGrading;
namespace Metallic.ColorGrading
{
float3 applyCustomLuts(float3 gamma, ColorGradingParameters p, GradingResources resources)
{
    float3 uvw = gamma * (15.0 / 16.0) + 0.5 / 16.0;
    float3 result = max(1.0 - dot(p.lutWeights, float4(1.0)), 0.0) * gamma;
    result += p.lutWeights.x * UnwrappedTexture3DSample(resources.lut1, resources.sampler, uvw, 16, 1.0/16.0).rgb;
    result += p.lutWeights.y * UnwrappedTexture3DSample(resources.lut2, resources.sampler, uvw, 16, 1.0/16.0).rgb;
    result += p.lutWeights.z * UnwrappedTexture3DSample(resources.lut3, resources.sampler, uvw, 16, 1.0/16.0).rgb;
    result += p.lutWeights.w * UnwrappedTexture3DSample(resources.lut4, resources.sampler, uvw, 16, 1.0/16.0).rgb;
    return result;
}
// Returns sRGB display code values for SDR, absolute scRGB (1=80 nits) for HDR.
// transform: 0 Reinhard, 1 exponential, 2 passthrough, 3 UE Film/ACES1, 4 ACES2.
public float3 evaluateColorGrading(float3 sceneLinear, ColorGradingParameters p,
    GradingResources resources, uint transform, bool hdr, float peak, float paperWhite)
{
    const float3x3 AP1_2_sRGB = mul(XYZ_2_sRGB_MAT, mul(D60_2_D65_CAT, AP1_2_XYZ_MAT));
    const float3x3 AP0_2_AP1 = mul(XYZ_2_AP1_MAT, AP0_2_XYZ_MAT);
    const float3x3 AP1_2_AP0 = mul(XYZ_2_AP0_MAT, AP1_2_XYZ_MAT);
    const float3x3 sRGB_2_AP1 = mul(XYZ_2_AP1_MAT, mul(D65_2_D60_CAT, sRGB_2_XYZ_MAT));
    float3 ColorAP1 = mul(sRGB_2_AP1, sceneLinear);
    if (all(sceneLinear == 0.0)) { ColorAP1 = float3(1e-10); }

    float  LumaAP1 = dot( ColorAP1, AP1_RGB2Y );
    float3 ChromaAP1 = ColorAP1 / LumaAP1;

    float ChromaDistSqr = dot( ChromaAP1 - 1, ChromaAP1 - 1 );
    float ExpandAmount = ( 1 - exp2( -4 * ChromaDistSqr ) ) * ( 1 - exp2( -4 * p.filmExtra.z * LumaAP1*LumaAP1 ) );

    const float3x3 Wide_2_XYZ_MAT =
    {
        0.5441691,  0.2395926,  0.1666943,
        0.2394656,  0.7021530,  0.0583814,
        -0.0023439,  0.0361834,  1.0552183,
    };

    const float3x3 Wide_2_AP1 = mul( XYZ_2_AP1_MAT, Wide_2_XYZ_MAT );
    const float3x3 ExpandMat = mul( Wide_2_AP1, AP1_2_sRGB );

    float3 ColorExpand = mul( ExpandMat, ColorAP1 );
    ColorAP1 = lerp( ColorAP1, ColorExpand, ExpandAmount );

    ColorAP1 = ColorCorrect( ColorAP1, p.saturation, p.contrast, p.gamma, p.gain, p.offset );

    if (transform < 3u) {
        float3 linear = max(mul(AP1_2_sRGB, ColorAP1), 0.0);
        if (hdr) {
            float maximum = max(linear.x, max(linear.y, linear.z));
            float headroom = max(peak - paperWhite, 0.0);
            float nits = maximum * paperWhite;
            if (maximum > 1.0) {
                nits = paperWhite + headroom * (1.0-exp(-(maximum-1.0)*paperWhite/max(headroom,0.001)));
            }
            return maximum > 0.0 ? (linear/maximum)*(nits/80.0) : float3(0.0);
        }
        linear = transform == 2u ? saturate(linear) : (transform == 1u ? 1.0-exp(-linear) : linear/(linear+1.0));
        return applyCustomLuts(LinearToSrgb(linear), p, resources);
    }
    if (hdr || transform == 4u) {
        float3 aces = mul(AP1_2_AP0, ColorAP1 * 1.5);
        float3 nitsAP1;
        if (transform == 4u) {
            ODTParams params = init_ODTParams(peak, AP1_2_XYZ_MAT, XYZ_2_AP1_MAT,
                AP1_2_XYZ_MAT, XYZ_2_AP1_MAT, resources.reach, resources.gamut, resources.gamma);
            float3 xyz = outputTransform_fwd(aces, peak, params);
            // AP1 white limiting; encoding and limiting white are both D60.
            nitsAP1 = clamp(mul(XYZ_2_AP1_MAT, xyz), 0.0, peak/referenceLuminance) * referenceLuminance;
        } else {
            nitsAP1 = aces1Nits(aces, peak);
        }
        float3 nits = mul(AP1_2_sRGB, nitsAP1);
        if (hdr) { return nits / 80.0; }
        return applyCustomLuts(LinearToSrgb(saturate(nits / 100.0)), p, resources);
    }

    const float3x3 BlueCorrect =
    {
        0.9404372683, -0.0183068787, 0.0778696104,
        0.0083786969,  0.8286599939, 0.1629613092,
        0.0005471261, -0.0008833746, 1.0003362486
    };
    const float3x3 BlueCorrectInv =
    {
        1.06318,     0.0233956, -0.0865726,
        -0.0106337,   1.20632,   -0.19569,
        -0.000590887, 0.00105248, 0.999538
    };
    const float3x3 BlueCorrectAP1    = mul( AP0_2_AP1, mul( BlueCorrect,    AP1_2_AP0 ) );
    const float3x3 BlueCorrectInvAP1 = mul( AP0_2_AP1, mul( BlueCorrectInv, AP1_2_AP0 ) );

    ColorAP1 = lerp( ColorAP1, mul( BlueCorrectAP1, ColorAP1 ), p.filmExtra.y );

    float3 ToneMappedColorAP1 = FilmToneMap( ColorAP1, p );
    ColorAP1 = lerp(ColorAP1, ToneMappedColorAP1, p.filmExtra.w);

    ColorAP1 = lerp( ColorAP1, mul( BlueCorrectInvAP1, ColorAP1 ), p.filmExtra.y );

    float3 FilmColor = max(0, mul( AP1_2_sRGB, ColorAP1 ));

    return applyCustomLuts(LinearToSrgb(saturate(FilmColor)), p, resources);
}
} // namespace Metallic.ColorGrading
