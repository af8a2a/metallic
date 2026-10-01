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

static const float3x3 AP0_2_XYZ_MAT =
{
    0.9525523959, 0.0000000000, 0.0000936786,
    0.3439664498, 0.7281660966,-0.0721325464,
    0.0000000000, 0.0000000000, 1.0088251844,
};

static const float3x3 XYZ_2_AP0_MAT =
{
     1.0498110175, 0.0000000000,-0.0000974845,
    -0.4959030231, 1.3733130458, 0.0982400361,
     0.0000000000, 0.0000000000, 0.9912520182,
};

static const float3x3 AP1_2_XYZ_MAT =
{
     0.6624541811, 0.1340042065, 0.1561876870,
     0.2722287168, 0.6740817658, 0.0536895174,
    -0.0055746495, 0.0040607335, 1.0103391003,
};

static const float3x3 XYZ_2_AP1_MAT =
{
     1.6410233797, -0.3248032942, -0.2364246952,
    -0.6636628587,  1.6153315917,  0.0167563477,
     0.0117218943, -0.0082844420,  0.9883948585,
};

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

static const float3x3 XYZ_2_sRGB_MAT =
{
     3.2409699419, -1.5373831776, -0.4986107603,
    -0.9692436363,  1.8759675015,  0.0415550574,
     0.0556300797, -0.2039769589,  1.0569715142,
};

static const float3x3 sRGB_2_XYZ_MAT =
{
    0.4123907993, 0.3575843394, 0.1804807884,
    0.2126390059, 0.7151686788, 0.0721923154,
    0.0193308187, 0.1191947798, 0.9505321522,
};

static const float3x3 D65_2_D60_CAT =
{
     1.0130349146, 0.0061052578, -0.0149709436,
     0.0076982301, 0.9981633521, -0.0050320385,
    -0.0028413174, 0.0046851567,  0.9245061375,
};

static const float3x3 D60_2_D65_CAT =
{
     0.9872240087, -0.0061132286, 0.0159532883,
    -0.0075983718,  1.0018614847, 0.0053300358,
     0.0030725771, -0.0050959615, 1.0816806031,
};

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
