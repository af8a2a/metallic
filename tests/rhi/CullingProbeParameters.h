#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using CullingProbeOutput = render::ShaderDataSpan;
using CullingProbeHZB = render::ShaderBuffer;
#else
import ShaderCore;
using Metallic;
typealias CullingProbeOutput = DataSpan<float4>;
typealias CullingProbeHZB = DescriptorHandle<RWStructuredBuffer<float>>;
#endif
struct ConeProbeParameters
{
    CullingProbeOutput output;
};
struct OcclusionProbeParameters
{
    CullingProbeOutput output;
    CullingProbeHZB history;
    CullingProbeHZB current;
};
#ifdef __cplusplus
inline constexpr uint64_t kConeProbeABI = 0x434f4e4550520001ull;
inline constexpr uint64_t kOcclusionProbeABI = 0x4f43434c50520001ull;
static_assert(sizeof(ConeProbeParameters) == 16);
static_assert(sizeof(OcclusionProbeParameters) == 32);
} // namespace metallic::tests
#endif
