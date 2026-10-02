#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using SHProbeInput = render::ShaderBuffer;
using SHProbeOutput = render::ShaderBuffer;
#else
import ShaderCore;
using Metallic;
typealias SHProbeInput = DescriptorHandle<StructuredBuffer<float4>>;
typealias SHProbeOutput = DescriptorHandle<RWStructuredBuffer<float4>>;
#endif
struct SphericalHarmonicsProbeParameters
{
    SHProbeInput input;
    SHProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kSHProbeABI = 0x534850524f420001ull;
static_assert(sizeof(SphericalHarmonicsProbeParameters) == 16);
} // namespace metallic::tests
#endif
