#language slang 2026
implementing ColorSpace;

namespace Metallic.ColorSpace
{
#define METALLIC_COLOR_MATRIX(name, ...) public static const float3x3 name = {__VA_ARGS__};
#include "../../../Source/Runtime/Render/Core/ColorSpaceMatrices.h"
#undef METALLIC_COLOR_MATRIX
public float3 rec709ToACEScg(float3 rgb)
{
    return mul(kXYZToAP1, mul(kD65ToD60, mul(kRec709ToXYZ, rgb)));
}
public float3 ACEScgToRec709(float3 rgb)
{
    return mul(kXYZToRec709, mul(kD60ToD65, mul(kAP1ToXYZ, rgb)));
}
} // namespace Metallic.ColorSpace
