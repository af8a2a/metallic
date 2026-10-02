#pragma once
#ifdef __cplusplus
#include "ResourceRegistry.h"
namespace metallic::render {
using BindlessSmokeImage = ShaderSampledImage;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias BindlessSmokeImage = DescriptorHandle<Texture2D<float4>>;
#endif
struct BindlessSmokeParameters
{
    BindlessSmokeImage sourceImage;
};
#ifdef __cplusplus
inline constexpr uint64_t kBindlessSmokeABI = 0x534D4F4B45490001ull;
static_assert(sizeof(BindlessSmokeParameters) == 8);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
