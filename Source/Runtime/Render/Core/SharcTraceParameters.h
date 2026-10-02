#pragma once
#include "PathTraceInlineParameters.h"
#include "PathTraceStageParameters.h"
#ifdef __cplusplus
namespace metallic::render {
#else
namespace Metallic {
#endif
struct SharcTraceParameters
{
    PathTraceInlineParameters path;
    PathTraceSettings cacheSettings;
    SharcHashHandle hashEntries;
    SharcAccumulationHandle accumulation;
    SharcResolvedHandle resolved;
};
#ifdef __cplusplus
inline constexpr uint64_t kSharcTraceABI = 0x5348525452430001ull;
static_assert(sizeof(SharcTraceParameters) == 72);
static_assert(offsetof(SharcTraceParameters, cacheSettings) == 40);
static_assert(offsetof(SharcTraceParameters, hashEntries) == 48);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
