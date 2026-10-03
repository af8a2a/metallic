#pragma once

// Shared CPU/Slang wire declarations. Resource identities are direct descriptor
// handles; arrays and ordinary data are descriptor-relative spans.
#ifdef __cplusplus
#include "Runtime/Render/Core/ShaderResourceABI.h"
namespace metallic::render {
#define DR_PUBLIC
#define DR_PADDING uint32_t
#define DR_BUFFER GPUResourceHandle<ResourceViewKind::RawBuffer>
#define DR_SAMPLED(T) GPUResourceHandle<ResourceViewKind::SampledImage>
#define DR_STORAGE(T) GPUResourceHandle<ResourceViewKind::StorageImage>
#define DR_SAMPLER GPUSamplerHandle
#define DR_SPAN GPUBufferSpan
#else
#define DR_PUBLIC public
#define DR_PADDING uint
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
    DR_PUBLIC DR_BUFFER indices;
    DR_PUBLIC DR_BUFFER instances;
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
    DR_PUBLIC DR_BUFFER nrcCounters;
    DR_PUBLIC DR_BUFFER nrcQueryPathInfo;
    DR_PUBLIC DR_BUFFER nrcQueryRadianceParams;
    DR_PUBLIC DR_BUFFER nrcTrainingPathInfo;
    DR_PUBLIC DR_BUFFER nrcTrainingPathVertices;
    DR_PUBLIC DR_STORAGE(2D<float4>) nrdMotion;
    DR_PUBLIC DR_STORAGE(2D<float4>) nrdNormalRoughness;
    DR_PUBLIC DR_STORAGE(2D<float>) nrdViewZ;
    DR_PUBLIC DR_BUFFER ntcConstants;
    DR_PUBLIC DR_BUFFER ntcInfo;
    DR_PUBLIC DR_SPAN ntcLatents;
    DR_PUBLIC DR_SAMPLER ntcSampler;
    DR_PUBLIC DR_BUFFER ntcWeights;
    DR_PUBLIC DR_SPAN openPBRLut2D;
    DR_PUBLIC DR_SPAN openPBRLut3D;
    DR_PUBLIC DR_STORAGE(2D<float4>) output;
    DR_PUBLIC DR_STORAGE(2D<float>) penumbra;
    DR_PUBLIC DR_STORAGE(2D<float4>) positionCurrent;
    DR_PUBLIC DR_STORAGE(2D<float4>) positionPrevious;
    DR_PUBLIC DR_BUFFER positions;
    DR_PUBLIC DR_BUFFER previousEnvironmentIrradianceSH;
    DR_PUBLIC DR_BUFFER primitives;
    DR_PUBLIC DR_BUFFER probeOutput;
    DR_PUBLIC DR_SAMPLED(2D<float4>) rasterDomain;
    DR_PUBLIC DR_BUFFER rasterFrameInfo;
    DR_PUBLIC DR_BUFFER rasterGeometries;
    DR_PUBLIC DR_BUFFER rasterInstances;
    DR_PUBLIC DR_BUFFER rasterMaterials;
    DR_PUBLIC DR_BUFFER rasterMeshlets;
    DR_PUBLIC DR_BUFFER rasterRecords;
    DR_PUBLIC DR_BUFFER rasterTriangles;
    DR_PUBLIC DR_BUFFER rasterVertexIndices;
    DR_PUBLIC DR_BUFFER rasterVertices;
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
    DR_PUBLIC DR_BUFFER textureFeedback;
    DR_PUBLIC DR_STORAGE(2D<float>) upscalerDepth;
    DR_PUBLIC DR_STORAGE(2D<float2>) upscalerMotion;
    DR_PUBLIC DR_BUFFER vertices;
    DR_PUBLIC DR_BUFFER view;
    DR_PUBLIC DR_SAMPLED(2D<uint>) visibility;
    DR_PUBLIC DR_SAMPLED(2D<float>) visibilityDepth;
};
#ifdef __cplusplus
static_assert(sizeof(SceneResourceParameters) == 440);
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
    DR_PUBLIC DR_BUFFER feedback;
};
#ifdef __cplusplus
static_assert(sizeof(TextureFeedbackResourceParameters) == 4);
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
    DR_PUBLIC DR_BUFFER geometries;
    DR_PUBLIC DR_BUFFER groups;
    DR_PUBLIC DR_BUFFER instances;
    DR_PUBLIC DR_BUFFER materials;
    DR_PUBLIC DR_BUFFER meshlets;
    DR_PUBLIC DR_STORAGE(2D<float4>) output;
    DR_PUBLIC DR_BUFFER pageCount;
    DR_PUBLIC DR_BUFFER pageTable;
    DR_PUBLIC DR_BUFFER pages;
    DR_PUBLIC DR_BUFFER records;
    DR_PUBLIC DR_BUFFER streamRecords;
    DR_PUBLIC DR_BUFFER triangles;
    DR_PUBLIC DR_BUFFER vertexIndices;
    DR_PUBLIC DR_BUFFER vertices;
    DR_PUBLIC DR_SAMPLED(2D<uint>) visibility;
};
#ifdef __cplusplus
static_assert(sizeof(VisibilityMaterialResourceParameters) == 60);
#endif

#undef DR_PUBLIC
#undef DR_PADDING
#undef DR_BUFFER
#undef DR_SAMPLED
#undef DR_STORAGE
#undef DR_SAMPLER
#undef DR_SPAN
#ifdef __cplusplus
} // namespace metallic::render
#endif
