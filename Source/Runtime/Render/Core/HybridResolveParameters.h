#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using HybridResolvePixels = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias HybridResolvePixels = DataSpan<uint64_t>;
#endif
struct HybridResolveParameters
{
    HybridResolvePixels pixels;
    uint32_t width;
    uint32_t reversedZ;
};
#ifdef __cplusplus
inline constexpr uint64_t kHybridResolveABI = 0x4859425245530001ull;
static_assert(sizeof(HybridResolveParameters) == 24);
static_assert(offsetof(HybridResolveParameters, width) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
