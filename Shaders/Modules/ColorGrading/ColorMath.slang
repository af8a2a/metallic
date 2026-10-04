// Adapted from Unreal Engine 5.7.4 / Academy ACES. Copyright Epic Games, Inc.
// All Rights Reserved. See Shaders/Licenses/ColorGrading. Forward rendering only.
#language slang 2026
implementing ColorGrading;
namespace Metallic.ColorGrading
{
static const uint WhiteStandard_D60 = 1;

struct DisplayPrimaries
{
    uint White;
    float3x3 XYZtoRGB;
    float3x3 RGBtoXYZ;
};

static const float3x3 AP0_2_XYZ_MAT = Metallic.ColorSpace::kAP0ToXYZ;

static const float3x3 XYZ_2_AP0_MAT = Metallic.ColorSpace::kXYZToAP0;

static const float3x3 AP1_2_XYZ_MAT = Metallic.ColorSpace::kAP1ToXYZ;

static const float3x3 XYZ_2_AP1_MAT = Metallic.ColorSpace::kXYZToAP1;

static const float3x3 AP0_2_AP1_MAT =
{
     1.4514393161, -0.2365107469, -0.2149285693,
    -0.0765537734,  1.1762296998, -0.0996759264,
     0.0083161484, -0.0060324498,  0.9977163014,
};

static const float3 AP1_RGB2Y =
{
    0.2722287168,
    0.6740817658,
    0.0536895174,
};

static const float3x3 XYZ_2_sRGB_MAT = Metallic.ColorSpace::kXYZToRec709;

static const float3x3 sRGB_2_XYZ_MAT = Metallic.ColorSpace::kRec709ToXYZ;

static const float3x3 D65_2_D60_CAT = Metallic.ColorSpace::kD65ToD60;

static const float3x3 D60_2_D65_CAT = Metallic.ColorSpace::kD60ToD65;

static const float3x3 CAM16_2_XYZ_MAT =
{
     2.0512756811, -1.1400313439,  0.0887556628,
     0.4269389763,  0.7005835277, -0.1275225040,
    -0.0174712779, -0.0384725929,  1.0589468739
};

static const float3x3 XYZ_2_CAM16_MAT =
{
     0.3640744835,  0.5947008156, 0.04110127349,
    -0.2222450987,  1.0738554823, 0.14794533610,
    -0.0020676190,  0.0488260453, 0.95038755696
};

float rgb_2_saturation(float3 rgb)
{
    float minrgb = min(min(rgb.r, rgb.g), rgb.b);
    float maxrgb = max(max(rgb.r, rgb.g), rgb.b);
    return (max(maxrgb, 1e-10) - max(minrgb, 1e-10)) / max(maxrgb, 1e-2);
}

float rgb_2_hue(float3 rgb)
{

    float hue;
    if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) {

        hue = 0;
    }
    else {
        hue = (180. / PI) * atan2(sqrt(3.0) * (rgb[1] - rgb[2]), 2 * rgb[0] - rgb[1] - rgb[2]);
    }

    if (hue < 0.) hue = hue + 360;

    return clamp(hue, 0, 360);
}

float center_hue(float hue, float centerH)
{
    float hueCentered = hue - centerH;
    if (hueCentered < -180.) hueCentered += 360;
    else if (hueCentered > 180.) hueCentered -= 360;
    return hueCentered;
}

float glow_fwd(float ycIn, float glowGainIn, float glowMid)
{
    float glowGainOut;

    if (ycIn <= 2. / 3. * glowMid) {
        glowGainOut = glowGainIn;
    }
    else if (ycIn >= 2 * glowMid) {
        glowGainOut = 0;
    }
    else {
        glowGainOut = glowGainIn * (glowMid / ycIn - 0.5);
    }

    return glowGainOut;
}

float rgb_2_yc(float3 rgb, float ycRadiusWeight = 1.75)
{

    float r = rgb[0];
    float g = rgb[1];
    float b = rgb[2];

    float chroma = sqrt(b * (b - g) + g * (g - r) + r * (r - b));

    return (b + g + r + ycRadiusWeight * chroma) / 3.;
}

float sigmoid_shaper(float x)
{

    float t = max(1 - abs(0.5 * x), 0);
    float y = 1 + sign(x) * (1 - t * t);
    return 0.5 * y;
}

float LinearToSrgbBranchingChannel(float lin)
{
    if(lin < 0.00313067) return lin * 12.92;
    return pow(lin, (1.0/2.4)) * 1.055 - 0.055;
}

float3 LinearToSrgbBranching(float3 lin)
{
    return float3(
        LinearToSrgbBranchingChannel(lin.r),
        LinearToSrgbBranchingChannel(lin.g),
        LinearToSrgbBranchingChannel(lin.b));
}

float3 LinearToSrgb(float3 lin)
{

    return LinearToSrgbBranching(lin);
}


}
