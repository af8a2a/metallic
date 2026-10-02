#pragma once
#include "PathTraceParameters.h"
#ifdef __cplusplus
namespace metallic::render {
using PathTraceResourceAddress = uint64_t;
#else
namespace Metallic {
typealias PathTraceResourceAddress = PathTraceParameters*;
#endif
struct PathTraceInlineParameters
{
    PathTraceResourceAddress resources;
    PathTraceSettings settings;
    PathTraceOutput output;
    PathTraceHistoryCurrent historyCurrent;
    PathTraceHistoryPrevious historyPrevious;
};
#ifdef __cplusplus
inline constexpr uint64_t kPathTraceInlineABI = 0x5054494e4c4e0001ull;
static_assert(sizeof(PathTraceInlineParameters) == 40);
static_assert(offsetof(PathTraceInlineParameters, output) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
