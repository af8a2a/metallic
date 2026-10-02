#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using MaterialErrorOutput = ShaderStorageImage;
using MaterialErrorUInt = uint32_t;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias MaterialErrorOutput = DescriptorHandle<RWTexture2D<float4>>;
typealias MaterialErrorUInt = uint;
#endif
struct MaterialErrorParams {
    MaterialErrorOutput output;
    MaterialErrorUInt color;
    MaterialErrorUInt padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kMaterialErrorABI = 0x4d41544552520001ull;
static_assert(sizeof(MaterialErrorParams) == 16 && offsetof(MaterialErrorParams, color) == 8);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
