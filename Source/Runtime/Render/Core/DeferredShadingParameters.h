#pragma once
#include "PathTraceInlineParameters.h"
#ifndef __cplusplus
import GPUDriven;
using Metallic.GPUDriven;
#endif
#ifdef __cplusplus
namespace metallic::render {
#else
namespace Metallic {
#endif
#ifdef __cplusplus
using DeferredVertices = ShaderDataSpan;
#else
typealias DeferredVertices = DataSpan<GPUDrivenPreviewVertex>;
#endif
#ifdef __cplusplus
using DeferredMeshlets = ShaderDataSpan;
#else
typealias DeferredMeshlets = DataSpan<GPUDrivenPreviewMeshlet>;
#endif
#ifdef __cplusplus
using DeferredRecords = ShaderDataSpan;
#else
typealias DeferredRecords = DataSpan<VisibleClusterRecord>;
#endif
#ifdef __cplusplus
using DeferredMeshletVertices = ShaderDataSpan;
#else
typealias DeferredMeshletVertices = DataSpan<uint>;
#endif
#ifdef __cplusplus
using DeferredTriangles = ShaderDataSpan;
#else
typealias DeferredTriangles = DataSpan<uint>;
#endif
#ifdef __cplusplus
using DeferredGeometries = ShaderDataSpan;
#else
typealias DeferredGeometries = DataSpan<GPUDrivenPreviewGeometry>;
#endif
#ifdef __cplusplus
using DeferredInstances = ShaderDataSpan;
#else
typealias DeferredInstances = DataSpan<GPUDrivenPreviewInstance>;
#endif
#ifdef __cplusplus
using DeferredMaterials = ShaderDataSpan;
#else
typealias DeferredMaterials = DataSpan<GPUDrivenPreviewMaterial>;
#endif
#ifdef __cplusplus
using DeferredBins = ShaderDataSpan;
#else
typealias DeferredBins = DataSpan<MaterialBin>;
#endif
#ifdef __cplusplus
using DeferredTiles = ShaderDataSpan;
#else
typealias DeferredTiles = DataSpan<MaterialTile>;
#endif
#ifdef __cplusplus
using DeferredIrradiance = ShaderBuffer;
#else
typealias DeferredIrradiance = DescriptorHandle<StructuredBuffer<float4>>;
#endif
#ifdef __cplusplus
using DeferredSpecular = ShaderBuffer;
#else
typealias DeferredSpecular = DescriptorHandle<StructuredBuffer<float4>>;
#endif
#ifdef __cplusplus
using DeferredGridParams = ShaderBuffer;
#else
typealias DeferredGridParams = DescriptorHandle<StructuredBuffer<ClusterLightGridParams>>;
#endif
#ifdef __cplusplus
using DeferredGridLights = ShaderBuffer;
#else
typealias DeferredGridLights = DescriptorHandle<StructuredBuffer<GPUPunctualLight>>;
#endif
#ifdef __cplusplus
using DeferredGridCandidates = ShaderBuffer;
#else
typealias DeferredGridCandidates = DescriptorHandle<StructuredBuffer<uint>>;
#endif
#ifdef __cplusplus
using DeferredGridCells = ShaderBuffer;
#else
typealias DeferredGridCells = DescriptorHandle<StructuredBuffer<ClusterLightGridCell>>;
#endif
#ifdef __cplusplus
using DeferredGridIndices = ShaderBuffer;
#else
typealias DeferredGridIndices = DescriptorHandle<StructuredBuffer<uint>>;
#endif
#ifdef __cplusplus
using DeferredShadow = ShaderSampledImage;
#else
typealias DeferredShadow = DescriptorHandle<Texture2D<float>>;
#endif
#ifdef __cplusplus
using DeferredShadowParams = ShaderBuffer;
#else
typealias DeferredShadowParams = DescriptorHandle<StructuredBuffer<ShadowParameters>>;
#endif
#ifdef __cplusplus
using DeferredView = ShaderBuffer;
#else
typealias DeferredView = DescriptorHandle<StructuredBuffer<ViewConstants>>;
#endif
#ifdef __cplusplus
using DeferredStreamRecords = ShaderBuffer;
#else
typealias DeferredStreamRecords = DescriptorHandle<StructuredBuffer<CompactStreamVisibleRecord>>;
#endif
#ifdef __cplusplus
using DeferredStreamGroups = ShaderBuffer;
#else
typealias DeferredStreamGroups = DescriptorHandle<StructuredBuffer<StreamActiveGroup>>;
#endif
#ifdef __cplusplus
using DeferredStreamPages = ShaderBuffer;
#else
typealias DeferredStreamPages = DescriptorHandle<StructuredBuffer<uint>>;
#endif
#ifdef __cplusplus
using DeferredStreamTable = ShaderBuffer;
#else
typealias DeferredStreamTable = DescriptorHandle<StructuredBuffer<uint2>>;
#endif
#ifdef __cplusplus
using DeferredFrameInfo = ShaderBuffer;
#else
typealias DeferredFrameInfo = DescriptorHandle<ByteAddressBuffer>;
#endif
#ifdef __cplusplus
using DeferredStreamParams = ShaderBuffer;
#else
typealias DeferredStreamParams = DescriptorHandle<ByteAddressBuffer>;
#endif
#ifdef __cplusplus
using DeferredFeedback = ShaderBuffer;
#else
typealias DeferredFeedback = DescriptorHandle<RWStructuredBuffer<uint>>;
#endif
#ifdef __cplusplus
using DeferredSampler = ShaderSampler;
#else
typealias DeferredSampler = DescriptorHandle<SamplerState>;
#endif
struct DeferredShadingResources
{
    DeferredVertices vertices;
    DeferredMeshlets meshlets;
    DeferredRecords records;
    DeferredMeshletVertices meshletVertices;
    DeferredTriangles triangles;
    DeferredGeometries geometries;
    DeferredInstances instances;
    DeferredMaterials materials;
    DeferredBins bins;
    DeferredTiles tiles;
    DeferredIrradiance irradiance;
    DeferredSpecular specular;
    DeferredGridParams gridParams;
    DeferredGridLights gridLights;
    DeferredGridCandidates gridCandidates;
    DeferredGridCells gridCells;
    DeferredGridIndices gridIndices;
    DeferredShadow shadow;
    DeferredShadowParams shadowParams;
    DeferredView view;
    DeferredStreamRecords streamRecords;
    DeferredStreamGroups streamGroups;
    DeferredStreamPages streamPages;
    DeferredStreamTable streamTable;
    DeferredFrameInfo frameInfo;
    DeferredStreamParams streamParams;
    DeferredFeedback feedback;
    DeferredSampler sampler;
};
#ifdef __cplusplus
using DeferredResourceAddress = uint64_t;
#else
typealias DeferredResourceAddress = DeferredShadingResources*;
#endif
struct DeferredShadingParameters
{
    PathTraceInlineParameters path;
    DeferredResourceAddress resources;
#ifdef __cplusplus
    ShaderSampledImage visibility;
#else
    DescriptorHandle<Texture2D<uint>> visibility;
#endif
    PathTraceEnvironmentPdf depth;
    PathTraceEnvironment domain;
    PathTraceMotionVectors motion;
    PathTraceDepth deviceDepth;
    uint32_t binIndex;
    uint32_t padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kDeferredShadingABI = 0x4445465348440001ull;
static_assert(sizeof(DeferredShadingParameters) == 96);
static_assert(sizeof(DeferredShadingResources) == 304);
static_assert(offsetof(DeferredShadingParameters, resources) == 40);
static_assert(offsetof(DeferredShadingParameters, binIndex) == 88);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
