#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using DiagnosticInput = render::ShaderBuffer;
#else
import ShaderCore;
using Metallic;
typealias DiagnosticInput = DescriptorHandle<ByteAddressBuffer>;
#endif
struct ShaderDiagnosticParameters
{
    DiagnosticInput input;
    uint32_t cookie;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kShaderDiagnosticABI = 0x4449414750520001ull;
static_assert(sizeof(ShaderDiagnosticParameters) == 16);
} // namespace metallic::tests
#endif
