#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using VisibilityMaterialUInt = uint32_t;
using VisibilityMaterialOutput = ShaderStorageImage;
using VisibilityMaterialIds = ShaderSampledImage;
using VisibilityMaterialInstances = ShaderBuffer;
using VisibilityMaterialMaterials = ShaderBuffer;
using VisibilityMaterialRecords = ShaderBuffer;
using VisibilityMaterialMeshlets = ShaderBuffer;
using VisibilityMaterialVertices = ShaderBuffer;
using VisibilityMaterialVertexIndices = ShaderBuffer;
using VisibilityMaterialTriangles = ShaderBuffer;
using VisibilityMaterialGeometries = ShaderBuffer;
using VisibilityMaterialStreamRecords = ShaderBuffer;
using VisibilityMaterialGroups = ShaderBuffer;
using VisibilityMaterialPages = ShaderBuffer;
using VisibilityMaterialPageTable = ShaderBuffer;
using VisibilityMaterialStreamParams = ShaderBuffer;
#else
import ShaderCore;
import GPUDriven;
using Metallic;
using Metallic.GPUDriven;
namespace Metallic {
typealias VisibilityMaterialUInt = uint;
typealias VisibilityMaterialOutput = DescriptorHandle<RWTexture2D<float4>>;
typealias VisibilityMaterialIds = DescriptorHandle<Texture2D<uint>>;
typealias VisibilityMaterialInstances = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>;
typealias VisibilityMaterialMaterials = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>>;
typealias VisibilityMaterialRecords = DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>;
typealias VisibilityMaterialMeshlets = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>;
typealias VisibilityMaterialVertices = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>>;
typealias VisibilityMaterialVertexIndices = DescriptorHandle<StructuredBuffer<uint>>;
typealias VisibilityMaterialTriangles = DescriptorHandle<StructuredBuffer<uint>>;
typealias VisibilityMaterialGeometries = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewGeometry>>;
typealias VisibilityMaterialStreamRecords = DescriptorHandle<StructuredBuffer<CompactStreamVisibleRecord>>;
typealias VisibilityMaterialGroups = DescriptorHandle<StructuredBuffer<StreamActiveGroup>>;
typealias VisibilityMaterialPages = DescriptorHandle<StructuredBuffer<uint>>;
typealias VisibilityMaterialPageTable = DescriptorHandle<StructuredBuffer<uint2>>;
typealias VisibilityMaterialStreamParams = DescriptorHandle<ByteAddressBuffer>;
#endif
struct VisibilityMaterialSettings {
    VisibilityMaterialUInt width, height, residentCount, streamCount, mode;
    float eyeX, eyeY, eyeZ;
};
struct VisibilityMaterialParams {
    VisibilityMaterialOutput output;
    VisibilityMaterialIds visibility;
    VisibilityMaterialInstances instances;
    VisibilityMaterialMaterials materials;
    VisibilityMaterialRecords records;
    VisibilityMaterialMeshlets meshlets;
    VisibilityMaterialVertices vertices;
    VisibilityMaterialVertexIndices vertexIndices;
    VisibilityMaterialTriangles triangles;
    VisibilityMaterialGeometries geometries;
    VisibilityMaterialStreamRecords streamRecords;
    VisibilityMaterialGroups groups;
    VisibilityMaterialPages pages;
    VisibilityMaterialPageTable pageTable;
    VisibilityMaterialStreamParams streamParams;
    VisibilityMaterialSettings settings;
};
#ifdef __cplusplus
inline constexpr uint64_t kVisibilityMaterialABI = 0x56424d4154000001ull;
static_assert(sizeof(VisibilityMaterialSettings) == 32);
static_assert(sizeof(VisibilityMaterialParams) == 152 && offsetof(VisibilityMaterialParams, settings) == 120);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
