#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using WaveWorkInput = render::ShaderDataSpan;
using WaveWorkOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias WaveWorkInput = DataSpan<uint>;
typealias WaveWorkOutput = DataSpan<uint4>;
#endif
struct WaveWorkProbeParameters
{
    WaveWorkInput input;
    WaveWorkOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kWaveWorkProbeABI = 0x5741565052420001ull;
static_assert(sizeof(WaveWorkProbeParameters) == 32);
static_assert(offsetof(WaveWorkProbeParameters, output) == 16);
} // namespace metallic::tests
#endif
