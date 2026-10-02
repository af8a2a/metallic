#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::render {
using BinningUInt = uint32_t;
using BinningVisibility = ShaderSampledImage;
#else
import ShaderCore;
import GPUDriven;
import Material;
using Metallic;
using Metallic.GPUDriven;
using Metallic.Material;
namespace Metallic {
typealias BinningUInt = uint;
typealias BinningVisibility = DescriptorHandle<Texture2D<uint>>;
#endif
#ifdef __cplusplus
using BinningRecords = ShaderDataSpan;
#else
typealias BinningRecords = DataSpan<VisibleClusterRecord>;
#endif
#ifdef __cplusplus
using BinningInstances = ShaderDataSpan;
#else
typealias BinningInstances = DataSpan<GPUDrivenPreviewInstance>;
#endif
#ifdef __cplusplus
using BinningMaterials = ShaderDataSpan;
#else
typealias BinningMaterials = DataSpan<GPUDrivenPreviewMaterial>;
#endif
#ifdef __cplusplus
using BinningShadingMaterials = ShaderDataSpan;
#else
typealias BinningShadingMaterials = DataSpan<PathTraceMaterial>;
#endif
#ifdef __cplusplus
using BinningBins = ShaderDataSpan;
#else
typealias BinningBins = DataSpan<MaterialBin>;
#endif
#ifdef __cplusplus
using BinningTiles = ShaderDataSpan;
#else
typealias BinningTiles = DataSpan<MaterialTile>;
#endif
#ifdef __cplusplus
using BinningArguments = ShaderDataSpan;
#else
typealias BinningArguments = DataSpan<uint>;
#endif
#ifdef __cplusplus
using BinningStreamRecords = ShaderDataSpan;
#else
typealias BinningStreamRecords = DataSpan<CompactStreamVisibleRecord>;
#endif
#ifdef __cplusplus
using BinningStreamGroups = ShaderDataSpan;
#else
typealias BinningStreamGroups = DataSpan<StreamActiveGroup>;
#endif

// Immutable input spans, shared by all three classification stages.
struct MaterialBinningResources {
    BinningRecords records;
    BinningInstances instances;
    BinningMaterials materials;
    BinningShadingMaterials shadingMaterials;
    BinningStreamRecords streamRecords;
    BinningStreamGroups streamGroups;
};
#ifdef __cplusplus
using BinningResources = uint64_t;
#else
typealias BinningResources = MaterialBinningResources*;
#endif
struct MaterialBinningParams {
    BinningVisibility visibility;
    BinningResources resources;
    BinningBins bins;
    BinningTiles tiles;
    BinningArguments arguments;
    BinningUInt width, height, tileCount, residentRecordCount;
};
#ifdef __cplusplus
inline constexpr uint64_t kMaterialBinningABI = 0x4d42494e00000004ull;
static_assert(sizeof(MaterialBinningResources) == 96);
static_assert(sizeof(MaterialBinningParams) == 80);
static_assert(offsetof(MaterialBinningParams, width) == 64);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
