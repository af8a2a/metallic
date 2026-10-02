#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using BatchProbeOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias BatchProbeOutput = DataSpan<uint>;
#endif
struct BatchBarrierProbeParameters
{
    BatchProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kBatchProbeABI = 0x4241545052420001ull;
static_assert(sizeof(BatchBarrierProbeParameters) == 16);
} // namespace metallic::tests
#endif
