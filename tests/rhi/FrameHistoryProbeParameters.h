#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using HistoryProbeImage = render::ShaderStorageImage;
using HistoryProbeOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias HistoryProbeImage = DescriptorHandle<RWTexture2D<float4>>;
typealias HistoryProbeOutput = DataSpan<uint>;
#endif
struct FrameHistoryProbeParameters
{
    HistoryProbeImage previous;
    HistoryProbeImage current;
    HistoryProbeOutput output;
    uint32_t index;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kFrameHistoryProbeABI = 0x4648495350520001ull;
static_assert(sizeof(FrameHistoryProbeParameters) == 40);
} // namespace metallic::tests
#endif
