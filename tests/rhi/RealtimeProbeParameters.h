#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using RealtimeProbeMotion = render::ShaderSampledImage;
using RealtimeProbeDepth = render::ShaderSampledImage;
using RealtimeProbeData = render::ShaderDataSpan;
using RealtimeProbePrefilter = render::ShaderBuffer;
#else
import ShaderCore;
using Metallic;
typealias RealtimeProbeMotion = DescriptorHandle<Texture2D<float2>>;
typealias RealtimeProbeDepth = DescriptorHandle<Texture2D<float>>;
typealias RealtimeProbeData = DataSpan<float4>;
typealias RealtimeProbePrefilter = DescriptorHandle<StructuredBuffer<float4>>;
#endif
struct RealtimeProbeParameters
{
    RealtimeProbeMotion motion;
    RealtimeProbeDepth depth;
    RealtimeProbeData data;
    RealtimeProbePrefilter prefilter;
};
#ifdef __cplusplus
inline constexpr uint64_t kRealtimeProbeABI = 0x525450524f420001ull;
static_assert(sizeof(RealtimeProbeParameters) == 40);
} // namespace metallic::tests
#endif
