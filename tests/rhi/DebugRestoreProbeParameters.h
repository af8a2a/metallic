#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using DebugRestoreInput = render::ShaderBuffer;
using DebugRestoreOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias DebugRestoreInput = DescriptorHandle<StructuredBuffer<uint>>;
typealias DebugRestoreOutput = DataSpan<uint>;
#endif
struct DebugRestoreProbeParameters
{
    DebugRestoreInput input;
    DebugRestoreOutput output;
    uint32_t index;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kDebugRestoreProbeABI = 0x4442475253540001ull;
static_assert(sizeof(DebugRestoreProbeParameters) == 32);
static_assert(offsetof(DebugRestoreProbeParameters, output) == 8);
static_assert(offsetof(DebugRestoreProbeParameters, index) == 24);
} // namespace metallic::tests
#endif
