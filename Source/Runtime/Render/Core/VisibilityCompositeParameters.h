#pragma once
#ifdef __cplusplus
#include "ResourceRegistry.h"
namespace metallic::render {
using CompositeBuffer = ShaderBuffer;
using CompositeImage = ShaderSampledImage;
using CompositeDepth = ShaderSampledImage;
using CompositeUInt = uint32_t;
#else
import ShaderCore;
import GPUDriven;
using Metallic;
using Metallic.GPUDriven;
// Matches MeshletStreamGPUActiveGroup. Logical pageIndex survives compaction
// and physical page eviction/reload; dataIndex and pageDeviceOffset do not.
struct VisibilityDebugActiveGroup
{
    uint pageDeviceOffset, pageIndex, clusterCount, primitiveIndex;
    uint lodLevel, materialIndex, mask, flags;
    uint instanceIndex, gpuSceneInstanceIndex, padding0, padding1;
    float4 world0, world1, world2, world3;
};

typealias CompositeParams = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>;
typealias CompositeResident = DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>;
typealias CompositeMeshlets = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>;
typealias CompositeStream = DescriptorHandle<StructuredBuffer<CompactStreamVisibleRecord>>;
typealias CompositeGroups = DescriptorHandle<StructuredBuffer<VisibilityDebugActiveGroup>>;
typealias CompositeImage = DescriptorHandle<Texture2D<uint>>;
typealias CompositeDepth = DescriptorHandle<Texture2D<float>>;
typealias CompositeUInt = uint;
#endif
#ifdef __cplusplus
using CompositeParams = CompositeBuffer;
using CompositeResident = CompositeBuffer;
using CompositeMeshlets = CompositeBuffer;
using CompositeStream = CompositeBuffer;
using CompositeGroups = CompositeBuffer;
#endif
struct VisibilityCompositeParameters {
    CompositeParams paramsBuffer;
    CompositeImage visibilityImage;
    CompositeDepth depthImage;
    CompositeResident residentRecords;
    CompositeMeshlets meshletBuffer;
    CompositeStream streamRecords;
    CompositeGroups streamGroups;
    CompositeUInt residentRecordCapacity, shadedColors, streamEnabled, padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kVisibilityCompositeABI = 0x564953434f4d0001ull;
static_assert(sizeof(VisibilityCompositeParameters) == 72);
} // namespace metallic::render
#endif
