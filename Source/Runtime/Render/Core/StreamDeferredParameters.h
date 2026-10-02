#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamDeferredSettings = ShaderDataSpan;
using StreamDeferredRecords = ShaderDataSpan;
using StreamDeferredGroups = ShaderDataSpan;
using StreamDeferredPageTable = ShaderDataSpan;
using StreamDeferredHeader = ShaderDataSpan;
using StreamDeferredOutput = ShaderDataSpan;
using StreamDeferredPages = ShaderDataSpan;
using StreamDeferredVisibility = ShaderSampledImage;
#else
// Included after the streaming record declarations in GPUDrivenStreamAsset.
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamDeferredSettings = DataSpan<GPUDrivenStreamAssetParams>;
typealias StreamDeferredRecords = DataSpan<CompactStreamVisibleRecord>;
typealias StreamDeferredGroups = DataSpan<GPUDrivenStreamAssetActiveGroup>;
typealias StreamDeferredPageTable = DataSpan<GPUDrivenStreamAssetPageTableEntry>;
typealias StreamDeferredHeader = DataSpan<GPUDrivenStreamAssetActiveHeader>;
typealias StreamDeferredOutput = DataSpan<uint>;
typealias StreamDeferredPages = DataSpan<uint>;
typealias StreamDeferredVisibility = DescriptorHandle<Texture2D<uint>>;
#endif
struct StreamDeferredParameters
{
    StreamDeferredSettings settings;
    StreamDeferredRecords records;
    StreamDeferredGroups groups;
    StreamDeferredPageTable pageTable;
    StreamDeferredHeader header;
    StreamDeferredOutput output;
    StreamDeferredPages pages;
    StreamDeferredVisibility visibility;
    uint32_t width;
    uint32_t height;
    uint32_t recordBase;
    uint32_t recordCapacity;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamDeferredABI = 0x5354524445460002ull;
static_assert(sizeof(StreamDeferredParameters) == 136);
static_assert(offsetof(StreamDeferredParameters, pages) == 96);
static_assert(offsetof(StreamDeferredParameters, width) == 120);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
