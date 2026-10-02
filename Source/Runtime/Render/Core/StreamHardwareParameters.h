#pragma once
#ifdef __cplusplus
#include "ResourceRegistry.h"
namespace metallic::render {
using HardwarePageBuffer = ShaderBuffer;
using HardwareActiveGroupBuffer = ShaderBuffer;
using HardwarePageTableBuffer = ShaderBuffer;
using HardwareSettings = uint64_t;
using HardwareRequestBuffer = ShaderBuffer;
using HardwareActiveHeaderBuffer = ShaderBuffer;
using HardwareRasterSettings = uint64_t;
using HardwareHybridQueueBuffer = ShaderBuffer;
using HardwareHybridClusterBuffer = ShaderBuffer;
using HardwareWritableBins = ShaderBuffer;
using HardwareUInt = uint32_t;
#else
import ShaderCore;
using Metallic;
typealias HardwarePageBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias HardwareActiveGroupBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>;
typealias HardwarePageTableBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>;
typealias HardwareSettings = GPUDrivenStreamAssetParams*;
typealias HardwareRequestBuffer = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias HardwareActiveHeaderBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>;
typealias HardwareRasterSettings = GPUDrivenStreamAssetRasterBindings*;
typealias HardwareHybridQueueBuffer = DescriptorHandle<RWStructuredBuffer<uint4>>;
typealias HardwareHybridClusterBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias HardwareWritableBins = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias HardwareUInt = uint;
#endif
struct StreamHardwareParameters
{
    HardwarePageBuffer pageBuffer;
    HardwareActiveGroupBuffer activeGroupBuffer;
    HardwarePageTableBuffer pageTableBuffer;
    HardwareSettings settings;
    HardwareRequestBuffer requestBuffer;
    HardwareActiveHeaderBuffer activeHeaderBuffer;
    HardwareRasterSettings rasterSettings;
    HardwareHybridQueueBuffer hybridQueueBuffer;
    HardwareHybridClusterBuffer hybridClusterBuffer;
    HardwareWritableBins writableBins;
    HardwareUInt traversalPhase;
    float tessellationEdgePixels;
    HardwareUInt tessellationMaxFactor, tessellationMaxSplitDepth;
    HardwareUInt hasQueue, hasBins;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamHardwareABI = 0x5354524841520007ull;
static_assert(sizeof(StreamHardwareParameters) == 104);
} // namespace metallic::render
#endif
