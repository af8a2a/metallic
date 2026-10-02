#pragma once
#include "../../Source/Runtime/Render/Core/PathTraceParameters.h"
#ifdef __cplusplus
namespace metallic::tests {
using ValueProbeInstances = render::ShaderDataSpan;
using ValueProbeOutput = render::ShaderDataSpan;
#else
using Metallic;
typealias ValueProbeInstances = DataSpan<MaterialValueInstance>;
typealias ValueProbeOutput = DataSpan<float>;
#endif
struct MaterialValueProbeParameters
{
    ValueProbeInstances instances;
    ValueProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kMaterialValueProbeABI = 0x4d5650524f420001ull;
static_assert(sizeof(MaterialValueProbeParameters) == 32);
} // namespace metallic::tests
#endif
