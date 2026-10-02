#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamActiveSettings = ShaderDataSpan;
using StreamActiveActiveGroupBuffer = ShaderBuffer;
using StreamActiveActiveHeaderBuffer = ShaderBuffer;
using StreamActiveDemandBuffer = ShaderBuffer;
using StreamActiveDemandStatsBuffer = ShaderBuffer;
using StreamActiveDrawIndirectBuffer = ShaderBuffer;
using StreamActiveGroupBuffer = ShaderBuffer;
using StreamActiveInstanceBuffer = ShaderBuffer;
using StreamActiveLODLevelBuffer = ShaderBuffer;
using StreamActiveLODStateBuffer = ShaderBuffer;
using StreamActiveLODTopologyBuffer = ShaderBuffer;
using StreamActiveNodeBuffer = ShaderBuffer;
using StreamActivePageBuffer = ShaderBuffer;
using StreamActivePageTableBuffer = ShaderBuffer;
using StreamActivePrimitiveBuffer = ShaderBuffer;
using StreamActiveRasterBindingsBuffer = ShaderBuffer;
using StreamActiveRequestBuffer = ShaderBuffer;
using StreamActiveTraversalHeaderBuffer = ShaderBuffer;
using StreamActiveTraversalWorkBuffer = ShaderBuffer;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamActiveSettings = DataSpan<GPUDrivenStreamAssetParams>;
typealias StreamActiveActiveGroupBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup>>;
typealias StreamActiveActiveHeaderBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader>>;
typealias StreamActiveDemandBuffer = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias StreamActiveDemandStatsBuffer = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias StreamActiveDrawIndirectBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetDrawIndirect>>;
typealias StreamActiveGroupBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>;
typealias StreamActiveInstanceBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>;
typealias StreamActiveLODLevelBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetLODLevel>>;
typealias StreamActiveLODStateBuffer = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias StreamActiveLODTopologyBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias StreamActiveNodeBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetNode>>;
typealias StreamActivePageBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias StreamActivePageTableBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>;
typealias StreamActivePrimitiveBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>;
typealias StreamActiveRasterBindingsBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>;
typealias StreamActiveRequestBuffer = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias StreamActiveTraversalHeaderBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader>>;
typealias StreamActiveTraversalWorkBuffer = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>>;
#endif
struct StreamActiveBuildParameters
{
    StreamActiveSettings settings;
    StreamActiveActiveGroupBuffer activeGroupBuffer;
    StreamActiveActiveHeaderBuffer activeHeaderBuffer;
    StreamActiveDemandBuffer demandBuffer;
    StreamActiveDemandStatsBuffer demandStatsBuffer;
    StreamActiveDrawIndirectBuffer drawIndirectBuffer;
    StreamActiveGroupBuffer groupBuffer;
    StreamActiveInstanceBuffer instanceBuffer;
    StreamActiveLODLevelBuffer lodLevelBuffer;
    StreamActiveLODStateBuffer lodStateBuffer;
    StreamActiveLODTopologyBuffer lodTopologyBuffer;
    StreamActiveNodeBuffer nodeBuffer;
    StreamActivePageBuffer pageBuffer;
    StreamActivePageTableBuffer pageTableBuffer;
    StreamActivePrimitiveBuffer primitiveBuffer;
    StreamActiveRasterBindingsBuffer rasterBindingsBuffer;
    StreamActiveRequestBuffer requestBuffer;
    StreamActiveTraversalHeaderBuffer traversalHeaderBuffer;
    StreamActiveTraversalWorkBuffer traversalWorkBuffer;
    uint32_t activeBuildPhase;
    uint32_t flags;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamActiveBuildABI = 0x5354524143540001ull;
static_assert(sizeof(StreamActiveBuildParameters) == 168);
static_assert(offsetof(StreamActiveBuildParameters, activeBuildPhase) == 160);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
