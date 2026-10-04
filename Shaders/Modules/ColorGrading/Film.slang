// Adapted from Unreal Engine 5.7.4 / Academy ACES. Copyright Epic Games, Inc.
// All Rights Reserved. See Shaders/Licenses/ColorGrading. Forward rendering only.
#language slang 2026
implementing ColorGrading;
namespace Metallic.ColorGrading
{
float3 FilmToneMap( float3 LinearColor, ColorGradingParameters p )
{
    const float3x3 sRGB_2_AP0 = mul( XYZ_2_AP0_MAT, mul( D65_2_D60_CAT, sRGB_2_XYZ_MAT ) );

    const float3x3 AP0_2_sRGB = mul( XYZ_2_sRGB_MAT, mul( D60_2_D65_CAT, AP0_2_XYZ_MAT ) );
    const float3x3 AP1_2_sRGB = mul( XYZ_2_sRGB_MAT, mul( D60_2_D65_CAT, AP1_2_XYZ_MAT ) );

    const float3x3 AP0_2_AP1 = mul( XYZ_2_AP1_MAT, AP0_2_XYZ_MAT );
    const float3x3 AP1_2_AP0 = mul( XYZ_2_AP0_MAT, AP1_2_XYZ_MAT );

    float3 ColorAP1 = LinearColor;

    float3 ColorAP0 = mul( AP1_2_AP0, ColorAP1 );

    const float RRT_GLOW_GAIN = 0.05;
    const float RRT_GLOW_MID = 0.08;

    float saturation = rgb_2_saturation( ColorAP0 );
    float ycIn = rgb_2_yc( ColorAP0 );
    float s = sigmoid_shaper( (saturation - 0.4) / 0.2);
    float addedGlow = 1 + glow_fwd( ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
    ColorAP0 *= addedGlow;

    const float RRT_RED_SCALE = 0.82;
    const float RRT_RED_PIVOT = 0.03;
    const float RRT_RED_HUE = 0;
    const float RRT_RED_WIDTH = 135;
    float hue = rgb_2_hue( ColorAP0 );
    float centeredHue = center_hue( hue, RRT_RED_HUE );
    float hueWeight = square( smoothstep( 0, 1, 1 - abs( 2 * centeredHue / RRT_RED_WIDTH ) ) );

    ColorAP0.r += hueWeight * saturation * (RRT_RED_PIVOT - ColorAP0.r) * (1. - RRT_RED_SCALE);

    float3 WorkingColor = mul( AP0_2_AP1_MAT, ColorAP0 );

    WorkingColor = max( 0, WorkingColor );

    WorkingColor = lerp( dot( WorkingColor, AP1_RGB2Y ), WorkingColor, 0.96 );

    const float ToeScale            = 1 + p.film.w - p.film.y;
    const float ShoulderScale   = 1 + p.filmExtra.x - p.film.z;

    const float InMatch = 0.18;
    const float OutMatch = 0.18;

    float ToeMatch;
    if( p.film.y > 0.8 )
    {

        ToeMatch = ( 1 - p.film.y  - OutMatch ) / p.film.x + log10( InMatch );
    }
    else
    {

        const float bt = ( OutMatch + p.film.w ) / ToeScale - 1;
        ToeMatch = log10( InMatch ) - 0.5 * log( (1+bt)/(1-bt) ) * (ToeScale / p.film.x);
    }

    float StraightMatch = ( 1 - p.film.y ) / p.film.x - ToeMatch;
    float ShoulderMatch = p.film.z / p.film.x - StraightMatch;

    float3 LogColor = log10( WorkingColor );
    float3 StraightColor = p.film.x * ( LogColor + StraightMatch );

    float3 ToeColor     = (    -p.film.w ) + (2 *      ToeScale) / ( 1 + exp( (-2 * p.film.x /      ToeScale) * ( LogColor -      ToeMatch ) ) );
    float3 ShoulderColor    = ( 1 + p.filmExtra.x ) - (2 * ShoulderScale) / ( 1 + exp( ( 2 * p.film.x / ShoulderScale) * ( LogColor - ShoulderMatch ) ) );

    ToeColor        = select(LogColor <      ToeMatch,      ToeColor, StraightColor);
    ShoulderColor   = select(LogColor > ShoulderMatch, ShoulderColor, StraightColor);

    float3 t = saturate( ( LogColor - ToeMatch ) / ( ShoulderMatch - ToeMatch ) );
    t = ShoulderMatch < ToeMatch ? 1 - t : t;
    t = (3-2*t)*t*t;
    float3 ToneColor = lerp( ToeColor, ShoulderColor, t );

    ToneColor = lerp( dot( float3(ToneColor), AP1_RGB2Y ), ToneColor, 0.93 );

    return max( 0, ToneColor );
}

float3 ColorCorrect( float3 WorkingColor,
    float4 ColorSaturation,
    float4 ColorContrast,
    float4 ColorGamma,
    float4 ColorGain,
    float4 ColorOffset )
{

    float Luma = dot( WorkingColor, AP1_RGB2Y );
    WorkingColor = max( 0, lerp( Luma.xxx, WorkingColor, ColorSaturation.xyz*ColorSaturation.w ) );
    WorkingColor = pow( WorkingColor * (1.0 / 0.18), ColorContrast.xyz*ColorContrast.w ) * 0.18;
    WorkingColor = pow( WorkingColor, 1.0 / (ColorGamma.xyz*ColorGamma.w) );
    WorkingColor = WorkingColor * (ColorGain.xyz * ColorGain.w) + (ColorOffset.xyz + ColorOffset.w);
    return WorkingColor;
}

float4 UnwrappedTexture3DSample( Texture2D<float4> Texture, SamplerState Sampler, float3 UVW, float Size, float InvSize )
{

    float IntW = floor( UVW.z * Size - 0.5 );
    float FracW = UVW.z * Size - 0.5 - IntW;

    float U = ( UVW.x + IntW ) * InvSize;
    float V = UVW.y;

    float4 RG0 = Texture.SampleLevel(Sampler, float2(U, V), 0);
    float4 RG1 = Texture.SampleLevel(Sampler, float2(U + InvSize, V), 0);

    return lerp(RG0, RG1, FracW);
}
}
