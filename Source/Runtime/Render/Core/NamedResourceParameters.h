#pragma once

// Shared CPU/Slang wire declarations. Resource identities are direct descriptor
// handles; arrays and ordinary data are descriptor-relative spans.
#ifdef __cplusplus
#include "Runtime/Render/Core/ShaderResourceABI.h"
#include <cstddef>
namespace metallic::render {
#define DR_PUBLIC
#define DR_BUFFER GPUResourceHandle<ResourceViewKind::RawBuffer>
#define DR_SAMPLED(T) GPUResourceHandle<ResourceViewKind::SampledImage>
#define DR_STORAGE(T) GPUResourceHandle<ResourceViewKind::StorageImage>
#define DR_SAMPLER GPUSamplerHandle
#define DR_SPAN GPUBufferSpan
#else
#define DR_PUBLIC public
#define DR_BUFFER ResourceHandle<ByteAddressBuffer>
#define DR_SAMPLED(T) ResourceHandle<Texture##T>
#define DR_STORAGE(T) ResourceHandle<RWTexture##T>
#define DR_SAMPLER SamplerHandle
#define DR_SPAN BufferSpan<uint>
#endif

DR_PUBLIC struct GPUProbeResourceParameters
{
    DR_PUBLIC DR_BUFFER input;
    DR_PUBLIC DR_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(GPUProbeResourceParameters) == 8);
#endif

DR_PUBLIC struct EnvironmentResourceParameters
{
    // Fixed celestial domain: Sun at slot 0, Moon at slot 1; no count header.
    DR_PUBLIC DR_BUFFER celestialLights;
    DR_PUBLIC DR_BUFFER atmosphereParameters;
    DR_PUBLIC DR_SAMPLED(2D<float4>) transmittance;
    DR_PUBLIC DR_SAMPLED(2D<float4>) multiScattering;
    DR_PUBLIC DR_SAMPLED(2D<float4>) skyView;
    DR_PUBLIC DR_BUFFER aerialPerspective;
    DR_PUBLIC DR_SAMPLED(2D<float4>) cloudShadow;
};
#ifdef __cplusplus
static_assert(sizeof(EnvironmentResourceParameters) == 28);
#endif

// Reserved for explicit NTC/VT experiments; never embedded in ordinary scene packets.
DR_PUBLIC struct NeuralTextureResourceParameters
{
    DR_PUBLIC DR_BUFFER ntcConstants;
    DR_PUBLIC DR_BUFFER ntcInfo;
    DR_PUBLIC DR_SPAN ntcLatents;
    DR_PUBLIC DR_SAMPLER ntcSampler;
    DR_PUBLIC DR_BUFFER ntcWeights;
};
#ifdef __cplusplus
static_assert(sizeof(NeuralTextureResourceParameters) == 28);
#endif

DR_PUBLIC struct SceneResourceParameters
{
    DR_PUBLIC uint64_t scene;
    DR_PUBLIC DR_STORAGE(2D<float4>) accumulationCurrent;
    DR_PUBLIC DR_STORAGE(2D<float4>) accumulationPrevious;
    DR_PUBLIC DR_STORAGE(2D<float4>) baseColorMetalness;
    DR_PUBLIC DR_BUFFER cacheParams;
    DR_PUBLIC DR_STORAGE(2D<float4>) emissive;
    DR_PUBLIC DR_BUFFER environmentIrradianceSH;
    DR_PUBLIC DR_SAMPLED(2D<float4>) environmentMap;
    DR_PUBLIC DR_SAMPLED(2D<float>) environmentPdf;
    DR_PUBLIC DR_STORAGE(2D<float4>) guideAlbedo;
    DR_PUBLIC DR_STORAGE(2D<float>) guideDepth;
    DR_PUBLIC DR_STORAGE(2D<float>) guideLinearDepth;
    DR_PUBLIC DR_STORAGE(2D<float2>) guideMotion;
    DR_PUBLIC DR_STORAGE(2D<float4>) guideNormalRoughness;
    DR_PUBLIC DR_STORAGE(2D<float4>) guideSpecularAlbedo;
    DR_PUBLIC DR_STORAGE(2D<float>) guideSpecularHitDistance;
    DR_PUBLIC DR_BUFFER lightGridCandidates;
    DR_PUBLIC DR_BUFFER lightGridCells;
    DR_PUBLIC DR_BUFFER lightGridIndices;
    DR_PUBLIC DR_BUFFER lightGridLights;
    DR_PUBLIC DR_BUFFER lightGridParams;
    DR_PUBLIC DR_BUFFER lights;
    DR_PUBLIC DR_SAMPLED(2D<float>) lightsPdf;
    DR_PUBLIC DR_SPAN materialBins;
    DR_PUBLIC DR_SAMPLER materialSampler;
    DR_PUBLIC DR_SPAN materialTextures;
    DR_PUBLIC DR_SPAN materialTiles;
    DR_PUBLIC DR_BUFFER materialValues;
    DR_PUBLIC DR_BUFFER materials;
    DR_PUBLIC DR_STORAGE(2D<float4>) noisyDiffuse;
    DR_PUBLIC DR_STORAGE(2D<float4>) noisySpecular;
    DR_PUBLIC DR_STORAGE(2D<float4>) normalCurrent;
    DR_PUBLIC DR_STORAGE(2D<float4>) normalPrevious;
    DR_PUBLIC DR_STORAGE(2D<float4>) nrdMotion;
    DR_PUBLIC DR_STORAGE(2D<float4>) nrdNormalRoughness;
    DR_PUBLIC DR_STORAGE(2D<float>) nrdViewZ;
    DR_PUBLIC DR_SPAN openPBRLut2D;
    DR_PUBLIC DR_SPAN openPBRLut3D;
    DR_PUBLIC DR_SAMPLER openPBRLutSampler;
    DR_PUBLIC DR_STORAGE(2D<float4>) output;
    DR_PUBLIC DR_STORAGE(2D<float>) penumbra;
    DR_PUBLIC DR_STORAGE(2D<float4>) positionCurrent;
    DR_PUBLIC DR_STORAGE(2D<float4>) positionPrevious;
    DR_PUBLIC DR_BUFFER previousEnvironmentIrradianceSH;
    DR_PUBLIC DR_BUFFER rayStreamChanges;
    DR_PUBLIC DR_BUFFER probeOutput;
    DR_PUBLIC DR_SAMPLED(2D<float4>) rasterDomain;
    DR_PUBLIC DR_BUFFER rasterFrameInfo;
    DR_PUBLIC DR_BUFFER rasterInstances;
    DR_PUBLIC DR_BUFFER rasterMaterials;
    DR_PUBLIC DR_BUFFER rayStreamHeader;
    DR_PUBLIC DR_BUFFER rayStreamInstances;
    DR_PUBLIC DR_BUFFER rayStreamPageTable;
    DR_PUBLIC DR_BUFFER rayStreamPages;
    DR_PUBLIC DR_BUFFER regirGrid;
    DR_PUBLIC DR_STORAGE(2D<uint4>) reservoirCurrent;
    DR_PUBLIC DR_STORAGE(2D<uint4>) reservoirPrevious;
    DR_PUBLIC DR_SAMPLED(2D<float>) shadowDepth;
    DR_PUBLIC DR_STORAGE(2D<float4>) shadowMotion;
    DR_PUBLIC DR_STORAGE(2D<float4>) shadowNormal;
    DR_PUBLIC DR_STORAGE(2D<float>) shadowOutput;
    DR_PUBLIC DR_BUFFER shadowParameters;
    DR_PUBLIC DR_STORAGE(2D<float>) shadowViewZ;
    DR_PUBLIC DR_SAMPLED(2D<float>) shadowVisibility;
    DR_PUBLIC DR_BUFFER sharcAccumulation;
    DR_PUBLIC DR_BUFFER sharcHashEntries;
    DR_PUBLIC DR_BUFFER sharcResolved;
    DR_PUBLIC DR_BUFFER streamGroups;
    DR_PUBLIC DR_BUFFER streamPageCount;
    DR_PUBLIC DR_BUFFER streamPageTable;
    DR_PUBLIC DR_BUFFER streamPages;
    DR_PUBLIC DR_BUFFER streamRecords;
    DR_PUBLIC DR_BUFFER textureSamplingMetadata;
    DR_PUBLIC DR_STORAGE(2D<float>) upscalerDepth;
    DR_PUBLIC DR_STORAGE(2D<float2>) upscalerMotion;
    DR_PUBLIC DR_BUFFER view;
    DR_PUBLIC DR_SAMPLED(2D<uint>) visibility;
    DR_PUBLIC DR_SAMPLED(2D<float>) visibilityDepth;
    DR_PUBLIC EnvironmentResourceParameters environment;
    DR_PUBLIC DR_BUFFER textureDemandMip;
    DR_PUBLIC DR_BUFFER textureDemandHits;
};
#ifdef __cplusplus
static_assert(sizeof(SceneResourceParameters) == 392);
static_assert(offsetof(SceneResourceParameters, environment) == 356);
static_assert(offsetof(SceneResourceParameters, environment) + sizeof(EnvironmentResourceParameters) ==
    offsetof(SceneResourceParameters, textureDemandMip));
#endif

// Read-only views of the existing scene packet; these do not change its wire layout.
DR_PUBLIC struct SceneMaterialTextureResourceParameters
{
    DR_PUBLIC DR_SAMPLER materialSampler;
    DR_PUBLIC DR_SPAN materialTextures;
};
#ifdef __cplusplus
static_assert(sizeof(SceneMaterialTextureResourceParameters) == 16);
static_assert(offsetof(SceneResourceParameters, materialSampler) == 108 + offsetof(SceneMaterialTextureResourceParameters, materialSampler));
static_assert(offsetof(SceneResourceParameters, materialTextures) == 108 + offsetof(SceneMaterialTextureResourceParameters, materialTextures));
#endif

DR_PUBLIC struct SceneOpenPBRLUTResourceParameters
{
    DR_PUBLIC DR_SPAN openPBRLut2D;
    DR_PUBLIC DR_SPAN openPBRLut3D;
    DR_PUBLIC DR_SAMPLER openPBRLutSampler;
};
#ifdef __cplusplus
static_assert(sizeof(SceneOpenPBRLUTResourceParameters) == 28);
static_assert(offsetof(SceneResourceParameters, openPBRLut2D) == 172 + offsetof(SceneOpenPBRLUTResourceParameters, openPBRLut2D));
static_assert(offsetof(SceneResourceParameters, openPBRLut3D) == 172 + offsetof(SceneOpenPBRLUTResourceParameters, openPBRLut3D));
static_assert(offsetof(SceneResourceParameters, openPBRLutSampler) == 172 + offsetof(SceneOpenPBRLUTResourceParameters, openPBRLutSampler));
#endif

DR_PUBLIC struct SceneRayStreamResourceParameters
{
    DR_PUBLIC DR_BUFFER rayStreamHeader;
    DR_PUBLIC DR_BUFFER rayStreamInstances;
    DR_PUBLIC DR_BUFFER rayStreamPageTable;
    DR_PUBLIC DR_BUFFER rayStreamPages;
};
#ifdef __cplusplus
static_assert(sizeof(SceneRayStreamResourceParameters) == 16);
static_assert(offsetof(SceneResourceParameters, rayStreamHeader) == 244 + offsetof(SceneRayStreamResourceParameters, rayStreamHeader));
static_assert(offsetof(SceneResourceParameters, rayStreamInstances) == 244 + offsetof(SceneRayStreamResourceParameters, rayStreamInstances));
static_assert(offsetof(SceneResourceParameters, rayStreamPageTable) == 244 + offsetof(SceneRayStreamResourceParameters, rayStreamPageTable));
static_assert(offsetof(SceneResourceParameters, rayStreamPages) == 244 + offsetof(SceneRayStreamResourceParameters, rayStreamPages));
#endif

DR_PUBLIC struct SceneLightResourceParameters
{
    DR_PUBLIC DR_BUFFER lights;
    DR_PUBLIC DR_SAMPLED(2D<float>) lightsPdf;
};
#ifdef __cplusplus
static_assert(sizeof(SceneLightResourceParameters) == 8);
static_assert(offsetof(SceneResourceParameters, lights) == 88 + offsetof(SceneLightResourceParameters, lights));
static_assert(offsetof(SceneResourceParameters, lightsPdf) == 88 + offsetof(SceneLightResourceParameters, lightsPdf));
#endif

DR_PUBLIC struct SceneGuideResourceParameters
{
    DR_PUBLIC DR_STORAGE(2D<float4>) guideAlbedo;
    DR_PUBLIC DR_STORAGE(2D<float>) guideDepth;
    DR_PUBLIC DR_STORAGE(2D<float>) guideLinearDepth;
    DR_PUBLIC DR_STORAGE(2D<float2>) guideMotion;
    DR_PUBLIC DR_STORAGE(2D<float4>) guideNormalRoughness;
    DR_PUBLIC DR_STORAGE(2D<float4>) guideSpecularAlbedo;
    DR_PUBLIC DR_STORAGE(2D<float>) guideSpecularHitDistance;
};
#ifdef __cplusplus
static_assert(sizeof(SceneGuideResourceParameters) == 28);
static_assert(offsetof(SceneResourceParameters, guideAlbedo) == 40 + offsetof(SceneGuideResourceParameters, guideAlbedo));
static_assert(offsetof(SceneResourceParameters, guideDepth) == 40 + offsetof(SceneGuideResourceParameters, guideDepth));
static_assert(offsetof(SceneResourceParameters, guideLinearDepth) == 40 + offsetof(SceneGuideResourceParameters, guideLinearDepth));
static_assert(offsetof(SceneResourceParameters, guideMotion) == 40 + offsetof(SceneGuideResourceParameters, guideMotion));
static_assert(offsetof(SceneResourceParameters, guideNormalRoughness) == 40 + offsetof(SceneGuideResourceParameters, guideNormalRoughness));
static_assert(offsetof(SceneResourceParameters, guideSpecularAlbedo) == 40 + offsetof(SceneGuideResourceParameters, guideSpecularAlbedo));
static_assert(offsetof(SceneResourceParameters, guideSpecularHitDistance) == 40 + offsetof(SceneGuideResourceParameters, guideSpecularHitDistance));
#endif

DR_PUBLIC struct SceneSharcResourceParameters
{
    DR_PUBLIC DR_BUFFER sharcAccumulation;
    DR_PUBLIC DR_BUFFER sharcHashEntries;
    DR_PUBLIC DR_BUFFER sharcResolved;
};
#ifdef __cplusplus
static_assert(sizeof(SceneSharcResourceParameters) == 12);
static_assert(offsetof(SceneResourceParameters, sharcAccumulation) == 300 + offsetof(SceneSharcResourceParameters, sharcAccumulation));
static_assert(offsetof(SceneResourceParameters, sharcHashEntries) == 300 + offsetof(SceneSharcResourceParameters, sharcHashEntries));
static_assert(offsetof(SceneResourceParameters, sharcResolved) == 300 + offsetof(SceneSharcResourceParameters, sharcResolved));
#endif

// Keep shader byte offsets checked against the CPU packet declaration.
#ifdef __cplusplus
#define DR_SCENE_OFFSET(Name, Field, Offset) inline constexpr uint32_t Name = Offset; static_assert(offsetof(SceneResourceParameters, Field) == Name)
#else
#define DR_SCENE_OFFSET(Name, Field, Offset) public static const uint Name = Offset
#endif
DR_SCENE_OFFSET(kSceneSceneOffset, scene, 0);
DR_SCENE_OFFSET(kSceneAccumulationCurrentOffset, accumulationCurrent, 8);
DR_SCENE_OFFSET(kSceneAccumulationPreviousOffset, accumulationPrevious, 12);
DR_SCENE_OFFSET(kSceneCacheParamsOffset, cacheParams, 20);
DR_SCENE_OFFSET(kSceneEnvironmentIrradianceSHOffset, environmentIrradianceSH, 28);
DR_SCENE_OFFSET(kSceneEnvironmentMapOffset, environmentMap, 32);
DR_SCENE_OFFSET(kSceneEnvironmentPdfOffset, environmentPdf, 36);
DR_SCENE_OFFSET(kSceneGuidesOffset, guideAlbedo, 40);
DR_SCENE_OFFSET(kSceneLightGridCandidatesOffset, lightGridCandidates, 68);
DR_SCENE_OFFSET(kSceneLightGridCellsOffset, lightGridCells, 72);
DR_SCENE_OFFSET(kSceneLightGridIndicesOffset, lightGridIndices, 76);
DR_SCENE_OFFSET(kSceneLightGridLightsOffset, lightGridLights, 80);
DR_SCENE_OFFSET(kSceneLightGridParamsOffset, lightGridParams, 84);
DR_SCENE_OFFSET(kSceneLightsOffset, lights, 88);
DR_SCENE_OFFSET(kSceneLightsPdfOffset, lightsPdf, 92);
DR_SCENE_OFFSET(kSceneMaterialBinsOffset, materialBins, 96);
DR_SCENE_OFFSET(kSceneMaterialSamplerOffset, materialSampler, 108);
DR_SCENE_OFFSET(kSceneMaterialTexturesOffset, materialTextures, 112);
DR_SCENE_OFFSET(kSceneMaterialTilesOffset, materialTiles, 124);
DR_SCENE_OFFSET(kSceneMaterialValuesOffset, materialValues, 136);
DR_SCENE_OFFSET(kSceneMaterialsOffset, materials, 140);
DR_SCENE_OFFSET(kSceneOpenPBRLUTsOffset, openPBRLut2D, 172);
DR_SCENE_OFFSET(kScenePreviousEnvironmentIrradianceSHOffset, previousEnvironmentIrradianceSH, 216);
DR_SCENE_OFFSET(kSceneRayStreamChangesOffset, rayStreamChanges, 220);
DR_SCENE_OFFSET(kSceneRasterDomainOffset, rasterDomain, 228);
DR_SCENE_OFFSET(kSceneRasterFrameInfoOffset, rasterFrameInfo, 232);
DR_SCENE_OFFSET(kSceneRasterInstancesOffset, rasterInstances, 236);
DR_SCENE_OFFSET(kSceneRasterMaterialsOffset, rasterMaterials, 240);
DR_SCENE_OFFSET(kSceneRayStreamOffset, rayStreamHeader, 244);
DR_SCENE_OFFSET(kSceneRayStreamHeaderOffset, rayStreamHeader, 244);
DR_SCENE_OFFSET(kSceneRayStreamInstancesOffset, rayStreamInstances, 248);
DR_SCENE_OFFSET(kSceneRayStreamPageTableOffset, rayStreamPageTable, 252);
DR_SCENE_OFFSET(kSceneRayStreamPagesOffset, rayStreamPages, 256);
DR_SCENE_OFFSET(kSceneReGIRGridOffset, regirGrid, 260);
DR_SCENE_OFFSET(kSceneShadowParametersOffset, shadowParameters, 288);
DR_SCENE_OFFSET(kSceneShadowVisibilityOffset, shadowVisibility, 296);
DR_SCENE_OFFSET(kSceneSharcOffset, sharcAccumulation, 300);
DR_SCENE_OFFSET(kSceneStreamGroupsOffset, streamGroups, 312);
DR_SCENE_OFFSET(kSceneStreamPageCountOffset, streamPageCount, 316);
DR_SCENE_OFFSET(kSceneStreamPageTableOffset, streamPageTable, 320);
DR_SCENE_OFFSET(kSceneStreamPagesOffset, streamPages, 324);
DR_SCENE_OFFSET(kSceneStreamRecordsOffset, streamRecords, 328);
DR_SCENE_OFFSET(kSceneTextureSamplingMetadataOffset, textureSamplingMetadata, 332);
DR_SCENE_OFFSET(kSceneTextureDemandMipOffset, textureDemandMip, 384);
DR_SCENE_OFFSET(kSceneTextureDemandHitsOffset, textureDemandHits, 388);
DR_SCENE_OFFSET(kSceneUpscalerDepthOffset, upscalerDepth, 336);
DR_SCENE_OFFSET(kSceneUpscalerMotionOffset, upscalerMotion, 340);
DR_SCENE_OFFSET(kSceneViewOffset, view, 344);
DR_SCENE_OFFSET(kSceneVisibilityOffset, visibility, 348);
DR_SCENE_OFFSET(kSceneVisibilityDepthOffset, visibilityDepth, 352);
DR_SCENE_OFFSET(kSceneEnvironmentOffset, environment, 356);
DR_SCENE_OFFSET(kSceneOutputOffset, output, 200);
#undef DR_SCENE_OFFSET

// Experimental primary-guide split extends the packet without changing the
// scene ABI used by existing pipelines.
DR_PUBLIC struct ScenePrimarySplitResourceParameters
{
    DR_PUBLIC SceneResourceParameters scene;
    DR_PUBLIC DR_BUFFER primaryHits;
#ifdef __cplusplus
    uint32_t reserved = 0;
#else
    public uint reserved;
#endif
};
#ifdef __cplusplus
inline constexpr uint32_t kOpenPBRPrimaryHitStride = 100;
inline constexpr uint32_t kScenePrimaryHitsOffset = 392;
static_assert(offsetof(ScenePrimarySplitResourceParameters, scene) == 0);
static_assert(offsetof(ScenePrimarySplitResourceParameters, primaryHits) == kScenePrimaryHitsOffset);
static_assert(offsetof(ScenePrimarySplitResourceParameters, reserved) == 396);
static_assert(sizeof(ScenePrimarySplitResourceParameters) == 400);
#else
public static const uint kOpenPBRPrimaryHitStride = 100;
public static const uint kScenePrimaryHitsOffset = 392;
#endif

// Opt-in path diagnostics extend only their own packet; production offsets stay fixed.
DR_PUBLIC struct ScenePathDiagnosticResourceParameters
{
    DR_PUBLIC SceneResourceParameters scene;
    DR_PUBLIC DR_STORAGE(2D<float4>) pathDiagnostics;
};
#ifdef __cplusplus
static_assert(offsetof(ScenePathDiagnosticResourceParameters, pathDiagnostics) == 392);
static_assert(sizeof(ScenePathDiagnosticResourceParameters) == 400);
#endif

DR_PUBLIC struct OutputImageResourceParameters
{
    DR_PUBLIC DR_STORAGE(2D<float4>) output;
};
#ifdef __cplusplus
static_assert(sizeof(OutputImageResourceParameters) == 4);
#endif

DR_PUBLIC struct TextureFeedbackResourceParameters
{
    DR_PUBLIC DR_BUFFER samplingMetadata;
    DR_PUBLIC DR_BUFFER demandMip;
    DR_PUBLIC DR_BUFFER demandHits;
};
#ifdef __cplusplus
static_assert(sizeof(TextureFeedbackResourceParameters) == 12);
#endif

DR_PUBLIC struct TextureProbeResourceParameters
{
    DR_PUBLIC DR_BUFFER output;
    DR_PUBLIC DR_SPAN textures;
};
#ifdef __cplusplus
static_assert(sizeof(TextureProbeResourceParameters) == 16);
#endif

DR_PUBLIC struct UpscalerGuideResourceParameters
{
    DR_PUBLIC DR_SAMPLED(2D<float>) inputDepth;
    DR_PUBLIC DR_SAMPLED(2D<float2>) inputMotion;
    DR_PUBLIC DR_STORAGE(2D<float>) outputDepth;
    DR_PUBLIC DR_STORAGE(2D<float2>) outputMotion;
};
#ifdef __cplusplus
static_assert(sizeof(UpscalerGuideResourceParameters) == 16);
#endif

DR_PUBLIC struct VisibilityMaterialResourceParameters
{
    DR_PUBLIC DR_BUFFER groups;
    DR_PUBLIC DR_BUFFER instances;
    DR_PUBLIC DR_BUFFER materials;
    DR_PUBLIC DR_STORAGE(2D<float4>) output;
    DR_PUBLIC DR_BUFFER pageCount;
    DR_PUBLIC DR_BUFFER pageTable;
    DR_PUBLIC DR_BUFFER pages;
    DR_PUBLIC DR_BUFFER streamRecords;
    DR_PUBLIC DR_SAMPLED(2D<uint>) visibility;
};
#ifdef __cplusplus
static_assert(sizeof(VisibilityMaterialResourceParameters) == 36);
#endif

#undef DR_PUBLIC
#undef DR_BUFFER
#undef DR_SAMPLED
#undef DR_STORAGE
#undef DR_SAMPLER
#undef DR_SPAN
#ifdef __cplusplus
} // namespace metallic::render
#endif
