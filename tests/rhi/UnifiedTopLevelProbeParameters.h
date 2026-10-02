#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using RayProbeStructure = render::ShaderAccelerationStructure;
using RayProbeOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
struct RayObservation
{
    uint hit, instance, primitive, front;
    float distance, u, v, padding;
};
typealias RayProbeStructure = DescriptorHandle<RaytracingAccelerationStructure>;
typealias RayProbeOutput = DataSpan<RayObservation>;
#endif
struct UnifiedTopLevelProbeParameters
{
    RayProbeStructure structure;
    RayProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kUnifiedRayProbeABI = 0x554e495241590001ull;
static_assert(sizeof(UnifiedTopLevelProbeParameters) == 24);
} // namespace metallic::tests
#endif
