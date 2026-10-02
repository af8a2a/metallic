#pragma once
#include "PathTraceInlineParameters.h"
#ifdef __cplusplus
namespace metallic::render {
#else
namespace Metallic {
#endif
struct PathTraceGuidesInlineParameters
{
    PathTraceInlineParameters path;
    PathTraceAlbedo albedo;
    PathTraceSpecularAlbedo specularAlbedo;
    PathTraceNormalRoughness normalRoughness;
    PathTraceMotionVectors motionVectors;
    PathTraceLinearDepth linearDepth;
    PathTraceSpecularHitDistance specularHitDistance;
    PathTraceDepth depth;
};
#ifdef __cplusplus
inline constexpr uint64_t kPathTraceGuidesInlineABI = 0x5054475549440001ull;
static_assert(sizeof(PathTraceGuidesInlineParameters) == 96);
static_assert(offsetof(PathTraceGuidesInlineParameters, albedo) == 40);
static_assert(offsetof(PathTraceGuidesInlineParameters, depth) == 88);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
