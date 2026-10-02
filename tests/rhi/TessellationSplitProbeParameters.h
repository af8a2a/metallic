#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using TessellationSplitInput = render::ShaderDataSpan;
using TessellationSplitOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
struct TessellationProbeCase { float4 position[3]; float4 eyeNear; float4 forwardScale; float4 options; };
typealias TessellationSplitInput = DataSpan<TessellationProbeCase>;
typealias TessellationSplitOutput = DataSpan<uint4>;
#endif
struct TessellationSplitProbeParameters
{
    TessellationSplitInput input;
    TessellationSplitOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kTessellationSplitProbeABI = 0x5445535052420001ull;
static_assert(sizeof(TessellationSplitProbeParameters) == 32);
static_assert(offsetof(TessellationSplitProbeParameters, output) == 16);
} // namespace metallic::tests
#endif
