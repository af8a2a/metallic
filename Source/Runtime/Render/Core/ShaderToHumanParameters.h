#pragma once
#ifdef __cplusplus
#include "ResourceRegistry.h"
namespace metallic::render {
using ShaderToHumanOutput = ShaderStorageImage;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias ShaderToHumanOutput = DescriptorHandle<RWTexture2D<float4>>;
#endif
struct ShaderToHumanParameters {
    ShaderToHumanOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kShaderToHumanABI = 0x533248554d4e0001ull;
static_assert(sizeof(ShaderToHumanParameters) == 8);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
