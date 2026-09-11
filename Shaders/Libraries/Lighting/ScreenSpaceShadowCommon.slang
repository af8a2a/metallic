#ifndef METALLIC_SCREEN_SPACE_SHADOW_COMMON
#define METALLIC_SCREEN_SPACE_SHADOW_COMMON
#include "../Camera/ViewConstants.slang"
#include "PhotometricLighting.slang"

struct ShadowParameters
{
    ViewConstants view;
    GpuPunctualLight light;
    float4 trace; // ray length, reserved, normal bias, tan(angular radius)
    uint4 control; // enabled, reserved, stable light slot, debug
    float4 shape; // local light radius, denoise enabled, reserved, reserved
};
#endif
