#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamPageTableSpan = ShaderDataSpan;
using StreamPagePatchSpan = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamPageTableSpan = DataSpan<uint2>;
typealias StreamPagePatchSpan = DataSpan<uint2>;
#endif
struct StreamPageTableParameters
{
    StreamPageTableSpan pages;
    StreamPagePatchSpan patches;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamPageTableABI = 0x5354525041470001ull;
static_assert(sizeof(StreamPageTableParameters) == 32);
static_assert(offsetof(StreamPageTableParameters, patches) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
