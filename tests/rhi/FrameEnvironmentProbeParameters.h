#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using FrameEnvironmentImage = render::ShaderSampledImage;
using FrameEnvironmentOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias FrameEnvironmentImage = DescriptorHandle<Texture2D<float4>>;
typealias FrameEnvironmentOutput = DataSpan<float4>;
#endif
struct FrameEnvironmentProbeParameters
{
    FrameEnvironmentImage environment;
    FrameEnvironmentOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kFrameEnvironmentProbeABI = 0x46454e5650520001ull;
static_assert(sizeof(FrameEnvironmentProbeParameters) == 24);
} // namespace metallic::tests
#endif
