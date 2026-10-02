#pragma once
#ifdef __cplusplus
#include "ResourceRegistry.h"
namespace metallic::render {
using ResidencyFeedback = ShaderDataSpan;
using ResidencyUInt = uint32_t;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias ResidencyFeedback = DataSpan<uint>;
typealias ResidencyUInt = uint;
#endif
struct TextureResidencyProbeParameters {
    ResidencyFeedback feedback;
    ResidencyUInt slot, wantedMip, samples, padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kTextureResidencyProbeABI = 0x5458524553500001ull;
static_assert(sizeof(TextureResidencyProbeParameters) == 32);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
