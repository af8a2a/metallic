#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using HZBSPDDepth = ShaderSampledImage;
using HZBSPDAtomicBuffer = ShaderBuffer;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias HZBSPDDepth = DescriptorHandle<Texture2D<float>>;
typealias HZBSPDAtomicBuffer = DescriptorHandle<RWStructuredBuffer<Atomic<uint>>>;
#endif
struct HZBSPDParameters
{
    HZBSPDDepth depthImage;
    HZBSPDAtomicBuffer hzbBuffer;
    HZBSPDAtomicBuffer counterBuffer;
    uint32_t width;
    uint32_t height;
    uint32_t mipCount;
    uint32_t reversedZ;
};
#ifdef __cplusplus
inline constexpr uint64_t kHZBSPDABI = 0x485a425350440001ull;
static_assert(sizeof(HZBSPDParameters) == 40);
static_assert(offsetof(HZBSPDParameters, width) == 24);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
