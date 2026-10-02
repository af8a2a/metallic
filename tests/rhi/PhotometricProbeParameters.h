#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using PhotometricProbeOutput = render::ShaderDataSpan;
using PhotometricProbeSH = render::ShaderBuffer;
using PhotometricProbeLights = render::ShaderBuffer;
#else
import ShaderCore;
import Lighting;
using Metallic;
using Metallic.Lighting;
typealias PhotometricProbeOutput = DataSpan<float4>;
typealias PhotometricProbeSH = DescriptorHandle<StructuredBuffer<float4>>;
typealias PhotometricProbeLights = DescriptorHandle<StructuredBuffer<GPUPunctualLight>>;
#endif
struct PhotometricProbeParameters
{
    PhotometricProbeOutput output;
    PhotometricProbeSH sphericalHarmonics;
    PhotometricProbeLights lights;
};
#ifdef __cplusplus
inline constexpr uint64_t kPhotometricProbeABI = 0x50484f5052420001ull;
static_assert(sizeof(PhotometricProbeParameters) == 32);
} // namespace metallic::tests
#endif
