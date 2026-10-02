#pragma once
#ifdef __cplusplus
#include "ResourceRegistry.h"
namespace metallic::render {
using TextureProbeImage = ShaderSampledImage;
using TextureProbeSampler = ShaderSampler;
using TextureProbeOutput = ShaderDataSpan;
using TextureProbeFeedback = ShaderDataSpan;
using TextureProbeUInt = uint32_t;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias TextureProbeImage = DescriptorHandle<Texture2D<float4>>;
typealias TextureProbeSampler = DescriptorHandle<SamplerState>;
typealias TextureProbeOutput = DataSpan<float4>;
typealias TextureProbeFeedback = DataSpan<uint>;
typealias TextureProbeUInt = uint;
#endif
struct TextureResourceProbeParameters {
    TextureProbeImage texture;
    TextureProbeOutput output;
    TextureProbeUInt mip, flags;
};
struct TextureStreamingProbeParameters {
    TextureProbeImage texture;
    TextureProbeSampler sampler;
    TextureProbeFeedback feedback;
    TextureProbeOutput output;
    TextureProbeUInt slot, visible;
    float nearSourceLod, farSourceLod;
};
#ifdef __cplusplus
inline constexpr uint64_t kTextureResourceProbeABI = 0x5458524553500002ull;
inline constexpr uint64_t kTextureStreamingProbeABI = 0x5458535452500001ull;
static_assert(sizeof(TextureResourceProbeParameters) == 32);
static_assert(sizeof(TextureStreamingProbeParameters) == 64);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
