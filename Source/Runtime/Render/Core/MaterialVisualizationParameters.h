#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using VisualizeFloat4 = float[4];
using VisualizeScene = ShaderAccelerationStructure;
using VisualizeOutput = ShaderStorageImage;
using VisualizeVertices = ShaderBuffer;
using VisualizeIndices = ShaderBuffer;
using VisualizePrimitives = ShaderBuffer;
using VisualizeInstances = ShaderBuffer;
using VisualizeMaterials = ShaderBuffer;
using VisualizeTextures = uint64_t;
using VisualizePositions = ShaderBuffer;
using VisualizeNtcLatents = uint64_t;
using VisualizeNtcConstants = ShaderBuffer;
using VisualizeNtcWeights = ShaderBuffer;
using VisualizeNtcInfo = ShaderBuffer;
using VisualizeNtcSampler = ShaderSampler;
#else
import ShaderCore;
import NeuralTextures;
using Metallic;
using Metallic.Interop.NeuralTextures;
struct MaterialVisualizePrimitive
{
    uint firstVertex;
    uint vertexCount;
    uint firstIndex;
    uint indexCount;
    uint flags;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct MaterialVisualizeInstance
{
    uint primitiveIndex;
    uint materialIndex;
    uint flags;
    uint padding;
};

struct MaterialVisualizeTextureInfo
{
    uint textureIndex;
    uint texCoord;
    uint ntcTextureSetIndex;
    uint ntcChannelMapping;
    float4 transform0;
    float4 transform1;
};

struct MaterialVisualizeMaterial
{
    float4 baseColor;
    float4 emissive;
    float4 params;
    float4 textureParams;
    float4 glassParams;
    float4 attenuationColor;
    float4 diffuseTransmission;
    float4 rtxcrHairBaseColor;
    float4 rtxcrHairParams0;
    float4 rtxcrHairParams1;
    float4 rtxcrHairDiffuseTint;
    MaterialVisualizeTextureInfo baseColorTexture;
    MaterialVisualizeTextureInfo metallicRoughnessTexture;
    MaterialVisualizeTextureInfo normalTexture;
    MaterialVisualizeTextureInfo occlusionTexture;
    MaterialVisualizeTextureInfo emissiveTexture;
    MaterialVisualizeTextureInfo transmissionTexture;
    MaterialVisualizeTextureInfo thicknessTexture;
    MaterialVisualizeTextureInfo diffuseTransmissionTexture;
    MaterialVisualizeTextureInfo diffuseTransmissionColorTexture;
    float4 specular;
    MaterialVisualizeTextureInfo specularTexture;
    MaterialVisualizeTextureInfo specularColorTexture;
};

namespace Metallic {
typealias VisualizeFloat4 = float4;
typealias VisualizeScene = DescriptorHandle<RaytracingAccelerationStructure>;
typealias VisualizeOutput = DescriptorHandle<RWTexture2D<float4>>;
typealias VisualizeVertices = DescriptorHandle<StructuredBuffer<SceneShadingVertex>>;
typealias VisualizeIndices = DescriptorHandle<StructuredBuffer<uint>>;
typealias VisualizePrimitives = DescriptorHandle<StructuredBuffer<MaterialVisualizePrimitive>>;
typealias VisualizeInstances = DescriptorHandle<StructuredBuffer<MaterialVisualizeInstance>>;
typealias VisualizeMaterials = DescriptorHandle<StructuredBuffer<MaterialVisualizeMaterial>>;
typealias VisualizeTextures = DescriptorHandle<Texture2D<float4>>*;
typealias VisualizePositions = DescriptorHandle<StructuredBuffer<SceneVertexPosition>>;
typealias VisualizeNtcLatents = DescriptorHandle<Texture2DArray<float4>>*;
typealias VisualizeNtcConstants = DescriptorHandle<NeuralTextureConstantBuffer>;
typealias VisualizeNtcWeights = DescriptorHandle<ByteAddressBuffer>;
typealias VisualizeNtcInfo = DescriptorHandle<StructuredBuffer<uint4>>;
typealias VisualizeNtcSampler = DescriptorHandle<SamplerState>;
#endif
struct SceneMaterialVisualizationPush
{
    VisualizeFloat4 eye;
    VisualizeFloat4 center;
    VisualizeFloat4 upProjection;
    VisualizeFloat4 viewport;
    VisualizeFloat4 clipOrtho;
    uint32_t width;
    uint32_t height;
    uint32_t mode;
    uint32_t materialTextureCount;
    float bitangentFlip;
    uint32_t ntcTextureSetCount;
    uint32_t padding1;
    uint32_t padding2;
};

struct MaterialVisualizationParameters
{
    VisualizeScene scene;
    VisualizeOutput output;
    VisualizeVertices vertices;
    VisualizeIndices indices;
    VisualizePrimitives primitives;
    VisualizeInstances instances;
    VisualizeMaterials materials;
    VisualizeTextures textures;
    VisualizePositions positions;
    VisualizeNtcLatents ntcLatents;
    VisualizeNtcConstants ntcConstants;
    VisualizeNtcWeights ntcWeights;
    VisualizeNtcInfo ntcInfo;
    VisualizeNtcSampler ntcSampler;
    SceneMaterialVisualizationPush settings;
};
#ifdef __cplusplus
inline constexpr uint64_t kMaterialVisualizationABI = 0x4d41545649530001ull;
static_assert(sizeof(SceneMaterialVisualizationPush) == 112);
static_assert(sizeof(MaterialVisualizationParameters) == 224);
static_assert(offsetof(MaterialVisualizationParameters, settings) == 112);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
