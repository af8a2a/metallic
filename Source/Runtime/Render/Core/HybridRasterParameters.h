#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using HybridRasterQueue = ShaderBuffer;
using HybridRasterPixels = ShaderBuffer;
using HybridRasterArguments = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias HybridRasterQueue = DescriptorHandle<RWStructuredBuffer<uint4>>;
typealias HybridRasterPixels = DescriptorHandle<RWStructuredBuffer<uint64_t>>;
typealias HybridRasterArguments = DataSpan<uint>;
#endif
struct HybridRasterParameters
{
    HybridRasterQueue queue;
    HybridRasterPixels pixels;
    HybridRasterArguments arguments;
    uint32_t width;
    uint32_t height;
    uint32_t capacity;
    float maxPixels;
    uint32_t reversedZ;
    uint32_t subpixelBits;
};
#ifdef __cplusplus
inline constexpr uint64_t kHybridRasterABI = 0x4859425241530001ull;
static_assert(sizeof(HybridRasterParameters) == 56);
static_assert(offsetof(HybridRasterParameters, width) == 32);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
