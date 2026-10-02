#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamClusterSettings = uint64_t;
using StreamClusterRasterSettings = uint64_t;
using StreamClusterPages = ShaderBuffer;
using StreamClusterGroups = ShaderBuffer;
using StreamClusterHeader = ShaderBuffer;
using StreamClusterPageTable = ShaderBuffer;
using StreamClusterRequests = ShaderBuffer;
using StreamClusterInstances = ShaderBuffer;
using StreamClusterVisibility = ShaderBuffer;
using StreamClusterRecords = ShaderBuffer;
using StreamClusterPreviousHZB = ShaderBuffer;
using StreamClusterCurrentHZB = ShaderBuffer;
using StreamClusterBins = ShaderBuffer;
using StreamClusterArguments = ShaderBuffer;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamClusterSettings = GPUDrivenStreamAssetParams*;
typealias StreamClusterRasterSettings = GPUDrivenStreamAssetRasterBindings*;
typealias StreamClusterPages = DescriptorHandle<StructuredBuffer<uint>>;
typealias StreamClusterGroups = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>;
typealias StreamClusterHeader = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>;
typealias StreamClusterPageTable = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>;
typealias StreamClusterRequests = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias StreamClusterInstances = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>;
typealias StreamClusterVisibility = DescriptorHandle<StructuredBuffer<uint>>;
typealias StreamClusterRecords = DescriptorHandle<RWStructuredBuffer<CompactStreamVisibleRecord>>;
typealias StreamClusterPreviousHZB = DescriptorHandle<StructuredBuffer<float>>;
typealias StreamClusterCurrentHZB = DescriptorHandle<StructuredBuffer<float>>;
typealias StreamClusterBins = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias StreamClusterArguments = DescriptorHandle<RWStructuredBuffer<uint>>;
#endif
struct StreamClusterCullParameters
{
    StreamClusterSettings settings;
    StreamClusterRasterSettings rasterSettings;
    StreamClusterPages pages;
    StreamClusterGroups groups;
    StreamClusterHeader header;
    StreamClusterPageTable pageTable;
    StreamClusterRequests requests;
    StreamClusterInstances instances;
    StreamClusterVisibility visibility;
    StreamClusterRecords records;
    StreamClusterPreviousHZB previousHZB;
    StreamClusterCurrentHZB currentHZB;
    StreamClusterBins bins;
    StreamClusterArguments arguments;
    uint32_t phase;
    uint32_t stage;
    uint32_t flags; // bit 0: instance flags available; bit 1: tessellation enabled
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamClusterCullABI = 0x5354524343430004ull;
static_assert(sizeof(StreamClusterCullParameters) == 128);
static_assert(offsetof(StreamClusterCullParameters, phase) == 112);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
