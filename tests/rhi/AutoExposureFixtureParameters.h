#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using ExposureFixtureOutput = render::ShaderStorageImage;
#else
import ShaderCore;
using Metallic;
typealias ExposureFixtureOutput = DescriptorHandle<RWTexture2D<float4>>;
#endif
struct AutoExposureFixtureParameters
{
    ExposureFixtureOutput output;
    uint32_t width, height;
    float luminance;
    uint32_t outliers;
};
#ifdef __cplusplus
inline constexpr uint64_t kExposureFixtureABI = 0x4145464958540001ull;
static_assert(sizeof(AutoExposureFixtureParameters) == 24);
} // namespace metallic::tests
#endif
