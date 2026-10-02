#pragma once
#ifdef __cplusplus
#include "ResourceRegistry.h"
namespace metallic::render {
using ResidentPositionBuffer = ShaderBuffer;
using ResidentMeshletBuffer = ShaderBuffer;
using ResidentMeshletDrawBuffer = ShaderBuffer;
using ResidentMeshletVertexBuffer = ShaderBuffer;
using ResidentMeshletTriangleBuffer = ShaderBuffer;
using ResidentParamsBuffer = ShaderBuffer;
using ResidentTransformBuffer = ShaderBuffer;
using ResidentInstanceBuffer = ShaderBuffer;
using ResidentInstanceVisibilityBuffer = ShaderBuffer;
using ResidentMaterialBuffer = ShaderBuffer;
using ResidentMaterialTextureRemapBuffer = ShaderBuffer;
using ResidentStreamOwnerMaskBuffer = ShaderBuffer;
using ResidentBins = ShaderBuffer;
using ResidentHZB = ShaderBuffer;
using ResidentWritableBins = ShaderBuffer;
using ResidentQueue = ShaderBuffer;
#else
import ShaderCore;
import GPUDriven;
import Material;
using Metallic;
using Metallic.GPUDriven;
using Metallic.Material;
typealias ResidentPositionBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewVertex>>;
typealias ResidentMeshletBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMeshlet>>;
typealias ResidentMeshletDrawBuffer = DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>;
typealias ResidentMeshletVertexBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias ResidentMeshletTriangleBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias ResidentParamsBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewParams>>;
typealias ResidentTransformBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewGeometry>>;
typealias ResidentInstanceBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>;
typealias ResidentInstanceVisibilityBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias ResidentMaterialBuffer = DescriptorHandle<StructuredBuffer<GPUDrivenPreviewMaterial>>;
typealias ResidentMaterialTextureRemapBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias ResidentStreamOwnerMaskBuffer = DescriptorHandle<StructuredBuffer<uint>>;
typealias ResidentBins = DescriptorHandle<StructuredBuffer<uint>>;
typealias ResidentHZB = DescriptorHandle<RWStructuredBuffer<float>>;
typealias ResidentWritableBins = DescriptorHandle<RWStructuredBuffer<uint>>;
typealias ResidentQueue = DescriptorHandle<RWStructuredBuffer<uint4>>;
#endif
struct ResidentRasterResources
{
    ResidentPositionBuffer positionBuffer;
    ResidentMeshletBuffer meshletBuffer;
    ResidentMeshletDrawBuffer meshletDrawBuffer;
    ResidentMeshletVertexBuffer meshletVertexBuffer;
    ResidentMeshletTriangleBuffer meshletTriangleBuffer;
    ResidentParamsBuffer paramsBuffer;
    ResidentTransformBuffer transformBuffer;
    ResidentInstanceBuffer instanceBuffer;
    ResidentInstanceVisibilityBuffer instanceVisibilityBuffer;
    ResidentMaterialBuffer materialBuffer;
    ResidentMaterialTextureRemapBuffer materialTextureRemapBuffer;
    ResidentStreamOwnerMaskBuffer streamOwnerMaskBuffer;
    ResidentBins bins;
    ResidentHZB previousHZB;
    ResidentHZB currentHZB;
    ResidentWritableBins writableBins;
    ResidentQueue queue;
};
#ifdef __cplusplus
using ResidentRasterAddress = uint64_t;
using ResidentUInt = uint32_t;
#else
typealias ResidentRasterAddress = ResidentRasterResources*;
typealias ResidentUInt = uint;
#endif
struct ResidentRasterParameters
{
    ResidentRasterAddress resources;
    ResidentUInt passIndex, mipLevel, projectWithCullingCamera;
    float tessellationEdgePixels;
    ResidentUInt tessellationMaxFactor, tessellationMaxSplitDepth;
    ResidentUInt hasBins, hasQueue, hasStreamOwnerMask, padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kResidentRasterABI = 0x5253545252410001ull;
static_assert(sizeof(ResidentRasterResources) == 136);
static_assert(sizeof(ResidentRasterParameters) == 48);
} // namespace metallic::render
#endif
