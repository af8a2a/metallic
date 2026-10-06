#pragma once

#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
struct ColorResizeParams {
    GPUResourceHandle<ResourceViewKind::SampledImage> source;
    GPUResourceHandle<ResourceViewKind::StorageImage> output;
};
inline constexpr uint64_t kColorResizeABI = 0x434f4c5253000001ull;
static_assert(sizeof(ColorResizeParams) == 8 && offsetof(ColorResizeParams, source) == 0 &&
    offsetof(ColorResizeParams, output) == 4);
#else
import ShaderCore;
using Metallic;
namespace Metallic {
struct ColorResizeParams {
    ResourceHandle<Texture2D<float4>> source;
    ResourceHandle<RWTexture2D<float4>> output;
};
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
