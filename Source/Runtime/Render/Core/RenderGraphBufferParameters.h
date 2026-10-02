#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using GraphBufferData = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias GraphBufferData = DataSpan<uint>;
#endif
struct RenderGraphBufferParams {
    GraphBufferData source;
    GraphBufferData output;
};
#ifdef __cplusplus
inline constexpr uint64_t kRenderGraphBufferABI = 0x5247425546460001ull;
static_assert(sizeof(RenderGraphBufferParams) == 32 && offsetof(RenderGraphBufferParams, output) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
