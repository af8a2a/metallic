#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using HybridProbeVertices = render::ShaderDataSpan;
using HybridProbePixels = render::ShaderBuffer;
using HybridProbeQueue = render::ShaderBuffer;
using HybridProbeCandidates = render::ShaderDataSpan;
using HybridProbeBins = render::ShaderBuffer;
#else
import ShaderCore;
using Metallic;
typealias HybridProbeVertices = DataSpan<float4>;
typealias HybridProbePixels = DescriptorHandle<RWStructuredBuffer<uint64_t>>;
typealias HybridProbeQueue = DescriptorHandle<RWStructuredBuffer<uint4>>;
typealias HybridProbeCandidates = DataSpan<uint2>;
typealias HybridProbeBins = DescriptorHandle<RWStructuredBuffer<uint>>;
#endif
struct HybridClusterProbeParameters
{
    HybridProbeCandidates input;
    HybridProbeBins bins;
    uint32_t count, padding;
};
struct HybridMeshProbeParameters
{
    HybridProbeVertices vertices;
    HybridProbeQueue queue;
    uint32_t doubleSided;
    uint32_t enabled;
};
struct PreparedRasterProbeParameters
{
    HybridProbeVertices vertices;
    HybridProbePixels reference;
    HybridProbePixels prepared;
    uint32_t width, height, reversed, doubleSided, bits, incrementalDepth, triangleCount, padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kHybridClusterProbeABI = 0x4859435052420001ull;
static_assert(sizeof(HybridClusterProbeParameters) == 32);
inline constexpr uint64_t kHybridMeshProbeABI = 0x48594d5052420001ull;
inline constexpr uint64_t kPreparedRasterProbeABI = 0x5052505241530001ull;
static_assert(sizeof(HybridMeshProbeParameters) == 32);
static_assert(sizeof(PreparedRasterProbeParameters) == 64);
} // namespace metallic::tests
#endif
