#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamCompositeColors = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamCompositeColors = DataSpan<uint>;
#endif
struct StreamCompositeParameters
{
    StreamCompositeColors colors;
    uint32_t width;
    uint32_t height;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamCompositeABI = 0x535452434f4d0001ull;
static_assert(sizeof(StreamCompositeParameters) == 24);
static_assert(offsetof(StreamCompositeParameters, width) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
