#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using HybridBinBuffer = ShaderBuffer;
using HybridBinArguments = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias HybridBinBuffer = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias HybridBinArguments = DataSpan<uint>;
#endif
struct HybridBinParameters
{
    HybridBinBuffer bins;
    HybridBinArguments arguments;
    uint32_t width;
    uint32_t height;
    uint32_t clusterCapacity;
    float maxPixels;
    uint32_t reversedZ;
    uint32_t subpixelBits;
    uint32_t inputClusterCount;
    uint32_t streamMode;
};
#ifdef __cplusplus
inline constexpr uint64_t kHybridBinABI = 0x48594242494e0002ull;
static_assert(sizeof(HybridBinParameters) == 56);
static_assert(offsetof(HybridBinParameters, arguments) == 8);
static_assert(offsetof(HybridBinParameters, width) == 24);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
