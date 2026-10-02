#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using ReGIRProbeOutput = render::ShaderBuffer;
using ReGIRProbeLights = render::ShaderBuffer;
using ReGIRProbeGrid = render::ShaderBuffer;
using ReGIRProbePdf = render::ShaderSampledImage;
#else
import ShaderCore;
import Lighting;
using Metallic;
using Metallic.Lighting;
typealias ReGIRProbeOutput = DescriptorHandle<RWStructuredBuffer<float4>>;
typealias ReGIRProbeLights = DescriptorHandle<StructuredBuffer<GPUPunctualLight>>;
typealias ReGIRProbeGrid = DescriptorHandle<StructuredBuffer<uint4>>;
typealias ReGIRProbePdf = DescriptorHandle<Texture2D<float>>;
#endif
struct ReGIRProbeParameters
{
    ReGIRProbeOutput output;
    ReGIRProbeLights lights;
    ReGIRProbeGrid grid;
    ReGIRProbePdf pdf;
#ifdef __cplusplus
    float position[3];
#else
    float3 position;
#endif
    uint32_t sampleCount;
    uint32_t seedOffset;
    uint32_t padding0, padding1, padding2;
};
#ifdef __cplusplus
inline constexpr uint64_t kReGIRProbeABI = 0x5247495250520001ull;
static_assert(sizeof(ReGIRProbeParameters) == 64);
} // namespace metallic::tests
#endif
