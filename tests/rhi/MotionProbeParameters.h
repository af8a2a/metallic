#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using MotionProbeOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias MotionProbeOutput = DataSpan<float4>;
#endif
struct MotionProbeParameters
{
    MotionProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kMotionProbeABI = 0x4d4f545052420001ull;
static_assert(sizeof(MotionProbeParameters) == 16);
} // namespace metallic::tests
#endif
