#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using MaterialProbeInput = render::ShaderBuffer;
using MaterialProbeOutput = render::ShaderBuffer;
#else
import ShaderCore;
import Material;
using Metallic;
using Metallic.Material;
typealias MaterialProbeInput = DescriptorHandle<StructuredBuffer<PathTraceMaterial>>;
typealias MaterialProbeOutput = DescriptorHandle<RWStructuredBuffer<uint>>;
#endif
struct MaterialProbeParameters
{
    MaterialProbeInput materials;
    MaterialProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kMaterialProbeABI = 0x4d41545052420001ull;
static_assert(sizeof(MaterialProbeParameters) == 16);
} // namespace metallic::tests
#endif
