#pragma once
#include "StreamRasterParameters.h"
#ifdef __cplusplus
namespace metallic::render {
using StreamWorkloadCounters = ShaderBuffer;
#else
namespace Metallic {
typealias StreamWorkloadCounters = DescriptorHandle<RWStructuredBuffer<uint64_t>>;
#endif
struct StreamWorkloadParameters
{
    // raster.pixels is unused and left invalid; diagnostics only write counters.
    StreamRasterParameters raster;
    StreamWorkloadCounters counters;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamWorkloadABI = 0x535452574f520001ull;
static_assert(sizeof(StreamWorkloadParameters) == 96);
static_assert(offsetof(StreamWorkloadParameters, counters) == 88);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
