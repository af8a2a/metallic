#pragma once
#include "PathTraceInlineParameters.h"
#ifdef __cplusplus
namespace metallic::render {
using RealtimeIrradianceHandle = ShaderBuffer;
#else
namespace Metallic {
typealias RealtimeIrradianceHandle = DescriptorHandle<StructuredBuffer<float4>>;
#endif
struct RealtimeLightingParameters
{
    PathTraceInlineParameters path;
    RealtimeIrradianceHandle irradiance;
};
#ifdef __cplusplus
inline constexpr uint64_t kRealtimeLightingABI = 0x52544c4954470001ull;
static_assert(sizeof(RealtimeLightingParameters) == 48);
static_assert(offsetof(RealtimeLightingParameters, irradiance) == 40);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
