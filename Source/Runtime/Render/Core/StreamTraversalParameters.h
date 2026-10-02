#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamTraversalSettings = ShaderDataSpan;
using StreamTraversalInstances = ShaderDataSpan;
using StreamTraversalPages = ShaderDataSpan;
using StreamTraversalPrimitives = ShaderBuffer;
using StreamTraversalGroups = ShaderBuffer;
using StreamTraversalNodes = ShaderBuffer;
using StreamTraversalPageTable = ShaderBuffer;
using StreamTraversalRequests = ShaderBuffer;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias StreamTraversalSettings = DataSpan<GPUDrivenStreamAssetParams>;
typealias StreamTraversalInstances = DataSpan<GPUDrivenStreamAssetInstance>;
typealias StreamTraversalPages = DataSpan<uint>;
typealias StreamTraversalPrimitives = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>;
typealias StreamTraversalGroups = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>;
typealias StreamTraversalNodes = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetNode>>;
typealias StreamTraversalPageTable = DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>;
typealias StreamTraversalRequests = DescriptorHandle<RWStructuredBuffer<uint>>;
#endif
struct StreamTraversalParameters
{
    StreamTraversalSettings settings;
    StreamTraversalInstances instances;
    StreamTraversalPages residentPages;
    StreamTraversalPrimitives primitives;
    StreamTraversalGroups groups;
    StreamTraversalNodes nodes;
    StreamTraversalPageTable pageTable;
    StreamTraversalRequests requests;
    uint32_t phase;
    uint32_t threadCount;
};
#ifdef __cplusplus
inline constexpr uint64_t kStreamTraversalABI = 0x5354525452560001ull;
static_assert(sizeof(StreamTraversalParameters) == 96);
static_assert(offsetof(StreamTraversalParameters, primitives) == 48);
static_assert(offsetof(StreamTraversalParameters, phase) == 88);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
