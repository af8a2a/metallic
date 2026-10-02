#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using EnvironmentProbeData = render::ShaderDataSpan;
using EnvironmentProbeImage = render::ShaderSampledImage;
#else
import ShaderCore;
using Metallic;
typealias EnvironmentProbeData = DataSpan<float4>;
typealias EnvironmentProbeImage = DescriptorHandle<Texture2D<float4>>;
#endif
struct EnvironmentPrefilterProbeParameters
{
    EnvironmentProbeData output;
    EnvironmentProbeData prefiltered;
    EnvironmentProbeImage radiance;
};
#ifdef __cplusplus
inline constexpr uint64_t kEnvironmentProbeABI = 0x454e565052420001ull;
static_assert(sizeof(EnvironmentPrefilterProbeParameters) == 40);
} // namespace metallic::tests
#endif
