#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using GridProbeParams = render::ShaderBuffer;
using GridProbeLights = render::ShaderBuffer;
using GridProbeIndices = render::ShaderBuffer;
using GridProbeCells = render::ShaderBuffer;
using GridProbeOutput = render::ShaderDataSpan;
#else
import ShaderCore;
import Lighting;
using Metallic;
using Metallic.Lighting;
typealias GridProbeParams = DescriptorHandle<StructuredBuffer<ClusterLightGridParams>>;
typealias GridProbeLights = DescriptorHandle<StructuredBuffer<ClusterLightData>>;
typealias GridProbeIndices = DescriptorHandle<StructuredBuffer<uint>>;
typealias GridProbeCells = DescriptorHandle<StructuredBuffer<ClusterLightGridCell>>;
typealias GridProbeOutput = DataSpan<uint4>;
#endif
struct ClusterLightGridProbeParameters
{
    GridProbeParams params;
    GridProbeLights lights;
    GridProbeIndices candidates;
    GridProbeCells cells;
    GridProbeIndices lightIndices;
    GridProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kGridProbeABI = 0x434c475052420001ull;
static_assert(sizeof(ClusterLightGridProbeParameters) == 56);
} // namespace metallic::tests
#endif
