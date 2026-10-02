#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamClassifySettings = ShaderDataSpan;
using StreamClassifyGroups = ShaderDataSpan;
using StreamClassifyPages = ShaderDataSpan;
using StreamClassifyInstances = ShaderDataSpan;
using StreamClassifyBins = ShaderBuffer;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamClassifySettings = DataSpan<GPUDrivenStreamAssetParams>;
typealias StreamClassifyGroups = DataSpan<GPUDrivenStreamAssetActiveGroup>;
typealias StreamClassifyPages = DataSpan<uint>;
typealias StreamClassifyInstances = DataSpan<GPUDrivenPreviewInstance>;
typealias StreamClassifyBins = DescriptorHandle<RWStructuredBuffer<uint>>;
#endif
struct StreamClassifyParameters
{
    StreamClassifySettings settings;
    StreamClassifyGroups groups;
    StreamClassifyPages pages;
    StreamClassifyInstances instances;
    StreamClassifyBins bins;
    uint32_t tessellationEnabled;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamClassifyABI = 0x535452434c530001ull;
static_assert(sizeof(StreamClassifyParameters) == 80);
static_assert(offsetof(StreamClassifyParameters, bins) == 64);
static_assert(offsetof(StreamClassifyParameters, tessellationEnabled) == 72);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
