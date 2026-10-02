#pragma once
#ifdef __cplusplus
#include "ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using NRDSampledHandle = ShaderSampledImage;
using NRDStorageHandle = ShaderStorageImage;
using NRDSamplerHandle = ShaderSampler;
#else
// Image element types vary per NRD recipe. Keep the complete wire handle here;
// generated bindings specialize DescriptorHandle<T> at each use.
typealias NRDSampledHandle = uint2;
typealias NRDStorageHandle = uint2;
typealias NRDSamplerHandle = uint2;
#endif
struct NRDResourceHandles
{
    NRDSampledHandle sampled[32];
    NRDStorageHandle storage[16];
    NRDSamplerHandle samplers[2];
};
#ifdef __cplusplus
using NRDConstantsAddress = uint64_t;
using NRDResourcesAddress = uint64_t;
static_assert(sizeof(NRDResourceHandles) == 400);
static_assert(offsetof(NRDResourceHandles, storage) == 256);
static_assert(offsetof(NRDResourceHandles, samplers) == 384);
#else
typealias NRDConstantsAddress = uint*;
typealias NRDResourcesAddress = NRDResourceHandles*;
#endif
// The dispatch root is inline; both immutable snapshots belong to its packet.
struct NRDPushData
{
    NRDConstantsAddress constants;
    NRDResourcesAddress resources;
};
#ifdef __cplusplus
inline constexpr uint64_t kNRDABI = 0x4e52445000000003ull;
static_assert(sizeof(NRDPushData) == 16);
} // namespace metallic::render
#endif
