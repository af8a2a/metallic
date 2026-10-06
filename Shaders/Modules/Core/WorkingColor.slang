#language slang 2026
implementing WorkingColor;

#ifndef METALLIC_WORKING_SPACE_ACESCG
#define METALLIC_WORKING_SPACE_ACESCG 1
#endif
namespace Metallic.WorkingColor
{
// AP1 vector weights retain FP32: AP1 -> display RGB can amplify small
// channel errors near gamut boundaries. Scalar energy weights may remain FP16.
#if METALLIC_WORKING_SPACE_ACESCG
public typedef float3 BoundedEnergyColor;
public typedef float BoundedEnergyInput;
#else
public typedef half3 BoundedEnergyColor;
public typedef half BoundedEnergyInput;
#endif

public float3 fromLinearRec709(float3 rgb)
{
#if METALLIC_WORKING_SPACE_ACESCG
    return ColorSpace::rec709ToACEScg(rgb);
#else
    return rgb;
#endif
}
public float3 toLinearRec709(float3 rgb)
{
#if METALLIC_WORKING_SPACE_ACESCG
    return ColorSpace::ACEScgToRec709(rgb);
#else
    return rgb;
#endif
}
public float3 toACEScg(float3 rgb)
{
#if METALLIC_WORKING_SPACE_ACESCG
    return rgb;
#else
    return ColorSpace::rec709ToACEScg(rgb);
#endif
}
public float3 fromACEScg(float3 rgb)
{
#if METALLIC_WORKING_SPACE_ACESCG
    return rgb;
#else
    return ColorSpace::ACEScgToRec709(rgb);
#endif
}
// XYZ is relative to the current working white (D60 in ACEScg, D65 in Rec.709).
public float3 fromXYZ(float3 xyz)
{
#if METALLIC_WORKING_SPACE_ACESCG
    return mul(ColorSpace::kXYZToAP1, xyz);
#else
    return mul(ColorSpace::kXYZToRec709, xyz);
#endif
}
public float3 toXYZ(float3 rgb)
{
#if METALLIC_WORKING_SPACE_ACESCG
    return mul(ColorSpace::kAP1ToXYZ, rgb);
#else
    return mul(ColorSpace::kRec709ToXYZ, rgb);
#endif
}
public float luminance(float3 rgb)
{
#if METALLIC_WORKING_SPACE_ACESCG
    return dot(rgb, ColorSpace::kAP1ToXYZ[1]);
#else
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
#endif
}
// Non-finite values are invalid radiance. Preserve finite negative gamut values.
public float3 sanitizeRadiance(float3 rgb)
{
    return float3(isfinite(rgb.x) ? rgb.x : 0.0, isfinite(rgb.y) ? rgb.y : 0.0, isfinite(rgb.z) ? rgb.z : 0.0);
}
} // namespace Metallic.WorkingColor
