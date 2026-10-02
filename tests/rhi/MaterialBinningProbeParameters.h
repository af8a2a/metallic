#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using ProbeBins = render::ShaderDataSpan;
using ProbeTiles = render::ShaderDataSpan;
using ProbeWords = render::ShaderDataSpan;
using ProbeUInt = uint32_t;
#else
import ShaderCore;
import GPUDriven;
using Metallic;
using Metallic.GPUDriven;
typealias ProbeBins = DataSpan<MaterialBin>;
typealias ProbeTiles = DataSpan<MaterialTile>;
typealias ProbeWords = DataSpan<uint>;
typealias ProbeUInt = uint;
#endif
struct MaterialProbeParams {
    ProbeBins bins;
    ProbeTiles tiles;
    ProbeWords arguments, output;
    ProbeUInt width, height, binCount, bin;
};
#ifdef __cplusplus
inline constexpr uint64_t kProbeABI = 0x4d42505200000003ull;
static_assert(sizeof(MaterialProbeParams) == 80);
} // namespace metallic::tests
#endif
