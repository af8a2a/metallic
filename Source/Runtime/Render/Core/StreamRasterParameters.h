#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamRasterSettings = ShaderDataSpan;
using StreamRasterPages = ShaderBuffer;
using StreamRasterGroups = ShaderBuffer;
using StreamRasterHeader = ShaderBuffer;
using StreamRasterPageTable = ShaderBuffer;
using StreamRasterInstances = ShaderBuffer;
using StreamRasterBins = ShaderBuffer;
using StreamRasterPixels = ShaderBuffer;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamRasterSettings = DataSpan<GPUDrivenStreamAssetParams>;
typealias StreamRasterPages = DescriptorHandle<StructuredBuffer<uint>>;
typealias StreamRasterGroups = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>;
typealias StreamRasterHeader = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>;
typealias StreamRasterPageTable = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>;
typealias StreamRasterInstances = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>;
typealias StreamRasterBins = DescriptorHandle<StructuredBuffer<uint>>;
typealias StreamRasterPixels = DescriptorHandle<RWStructuredBuffer<uint64_t>>;
#endif
struct StreamRasterParameters
{
    StreamRasterSettings settings;
    StreamRasterPages pages;
    StreamRasterGroups groups;
    StreamRasterHeader header;
    StreamRasterPageTable pageTable;
    StreamRasterInstances instances;
    StreamRasterBins bins;
    StreamRasterPixels pixels;
    uint32_t visibleRecordBase;
    uint32_t visibleRecordCapacity;
    uint32_t hasInstances;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamRasterABI = 0x5354525241530001ull;
static_assert(sizeof(StreamRasterParameters) == 88);
static_assert(offsetof(StreamRasterParameters, visibleRecordBase) == 72);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
