#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using DataProbeSpan = render::ShaderDataSpan;
using DataProbeUInt = uint32_t;
#else
import ShaderCore;
using Metallic;
typealias DataProbeSpan = DataSpan<uint>;
typealias DataProbeUInt = uint;
#endif
struct DataProbeParams {
    DataProbeSpan source, output, arguments;
    DataProbeUInt add, padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kDataProbeABI = 0x4441544150520001ull;
static_assert(sizeof(DataProbeParams) == 56);
} // namespace metallic::tests
#endif
