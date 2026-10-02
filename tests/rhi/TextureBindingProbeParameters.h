#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using ProbeStorageUintImage = render::ShaderStorageImage;
using ProbeUintImages = uint64_t;
using ProbeFloatImages = uint64_t;
using ProbeSampler = render::ShaderSampler;
using ProbeUintBuffer = render::ShaderBuffer;
using ProbeUint4Buffer = render::ShaderBuffer;
#else
import ShaderCore;
using Metallic;
typealias ProbeStorageUintImage = DescriptorHandle<RWTexture2D<uint>>;
typealias ProbeUintImages = DescriptorHandle<Texture2D<uint>>*;
typealias ProbeFloatImages = DescriptorHandle<Texture2D<float4>>*;
typealias ProbeSampler = DescriptorHandle<SamplerState>;
typealias ProbeUintBuffer = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias ProbeUint4Buffer = DescriptorHandle<RWStructuredBuffer<uint4>>;
#endif
struct RegistryTextureParams {
    ProbeStorageUintImage image;
    ProbeUintImages samples;
    ProbeUintBuffer output;
};
struct NonuniformTextureParams {
    ProbeFloatImages images;
    ProbeSampler samplers[2];
    ProbeUint4Buffer output;
};
#ifdef __cplusplus
inline constexpr uint64_t kRegistryTextureABI = 0x5245475445580001ull;
inline constexpr uint64_t kNonuniformTextureABI = 0x4e554e4954580001ull;
static_assert(sizeof(RegistryTextureParams) == 24);
static_assert(sizeof(NonuniformTextureParams) == 32);
} // namespace metallic::tests
#endif
