#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using SliderFixtureOutput = render::ShaderStorageImage;
using SliderFixtureInput = render::ShaderSampledImage;
using SliderFixtureReadback = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias SliderFixtureOutput = DescriptorHandle<RWTexture2D<float4>>;
typealias SliderFixtureInput = DescriptorHandle<Texture2D<float4>>;
typealias SliderFixtureReadback = DataSpan<float4>;
#endif
struct SliderFixtureParameters
{
    SliderFixtureOutput output;
    SliderFixtureInput input;
    SliderFixtureReadback readback;
    uint32_t path;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kSliderFixtureABI = 0x534c444649580001ull;
static_assert(sizeof(SliderFixtureParameters) == 40);
} // namespace metallic::tests
#endif
