#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamCullSettings = ShaderDataSpan;
using StreamCullInstances = ShaderDataSpan;
using StreamCullWords = ShaderDataSpan;
using StreamCullCounter = ShaderBuffer;
using StreamCullHZB = ShaderBuffer;
#else
// Included after streaming GPU record declarations.
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamCullSettings = DataSpan<GPUDrivenStreamAssetParams>;
typealias StreamCullInstances = DataSpan<GPUDrivenStreamAssetInstance>;
typealias StreamCullWords = DataSpan<uint>;
typealias StreamCullCounter = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias StreamCullHZB = DescriptorHandle<StructuredBuffer<float>>;
#endif
struct StreamInstanceCullParameters
{
    StreamCullSettings settings;
    StreamCullInstances instances;
    StreamCullWords visibility;
    StreamCullWords visibleIds;
    StreamCullCounter counter;
    StreamCullHZB hzb;
    uint32_t phase;
    uint32_t width;
    uint32_t height;
    uint32_t mipCount;
    uint32_t hzbValid;
    uint32_t cullingFlags;
    float displacementBound;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamInstanceCullABI = 0x53545243554c0001ull;
static_assert(sizeof(StreamInstanceCullParameters) == 112);
static_assert(offsetof(StreamInstanceCullParameters, counter) == 64);
static_assert(offsetof(StreamInstanceCullParameters, phase) == 80);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
