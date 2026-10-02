#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using ViewProbeData = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias ViewProbeData = DataSpan<ViewConstants>;
#endif
struct ViewProbeParameters
{
    ViewProbeData output;
    ViewProbeData view;
};
#ifdef __cplusplus
inline constexpr uint64_t kViewProbeABI = 0x5649455750520001ull;
static_assert(sizeof(ViewProbeParameters) == 32);
} // namespace metallic::tests
#endif
