#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using HZBData = ShaderDataSpan;
using HZBDepth = ShaderSampledImage;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias HZBData = DataSpan<float>;
typealias HZBDepth = DescriptorHandle<Texture2D<float>>;
#endif
struct HZBParameters
{
    HZBData hzb;
    HZBDepth depth;
    uint32_t width;
    uint32_t height;
    uint32_t sourceWidth;
    uint32_t sourceHeight;
    uint32_t sourceOffset;
    uint32_t destinationOffset;
    uint32_t reversedZ;
    uint32_t mipLevel;
};
#ifdef __cplusplus
inline constexpr uint64_t kHZBABI = 0x535452485a420001ull;
static_assert(sizeof(HZBParameters) == 56);
static_assert(offsetof(HZBParameters, depth) == 16);
static_assert(offsetof(HZBParameters, width) == 24);
static_assert(offsetof(HZBParameters, mipLevel) == 52);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
