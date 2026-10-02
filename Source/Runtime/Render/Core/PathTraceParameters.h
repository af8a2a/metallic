#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::render {
#else
import ShaderCore;
import Material;
import Lighting;
import NeuralTextures;
using Metallic.Interop.NeuralTextures;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
#define METALLIC_MATERIAL_VALUE_INSTANCE_DEFINED 1
struct MaterialValueInstance { uint programId; uint reserved0; uint reserved1; uint reserved2; float4 parameters[4]; };

struct PathTracePrimitive
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

struct PathTraceInstance
{
    uint primitiveIndex;
    uint materialIndex;
    uint flags;
    float rayConeLodConstant;
};

namespace Metallic {
#endif
#ifdef __cplusplus
using PathTraceSettings = uint64_t;
using PathTraceScene = ShaderAccelerationStructure;
using PathTraceOutput = ShaderStorageImage;
using PathTraceVertices = ShaderDataSpan;
using PathTraceIndices = ShaderDataSpan;
using PathTracePrimitives = ShaderDataSpan;
using PathTraceInstances = ShaderDataSpan;
using PathTracePositions = ShaderDataSpan;
using PathTraceMaterials = ShaderBuffer;
using PathTraceHistoryCurrent = ShaderStorageImage;
using PathTraceHistoryPrevious = ShaderStorageImage;
using PathTraceMaterialTextures = uint64_t;
using PathTraceEnvironment = ShaderSampledImage;
using PathTraceEnvironmentPdf = ShaderSampledImage;
using PathTraceLUT2D = uint64_t;
using PathTraceLUT3D = uint64_t;
using PathTraceLights = ShaderBuffer;
using PathTraceReGIR = ShaderBuffer;
using PathTracePunctualPdf = ShaderSampledImage;
using PathTraceAlbedo = ShaderStorageImage;
using PathTraceSpecularAlbedo = ShaderStorageImage;
using PathTraceNormalRoughness = ShaderStorageImage;
using PathTraceMotionVectors = ShaderStorageImage;
using PathTraceLinearDepth = ShaderStorageImage;
using PathTraceSpecularHitDistance = ShaderStorageImage;
using PathTraceDepth = ShaderStorageImage;
using PathTraceMaterialValues = ShaderBuffer;
using PathTraceNTCLatents = uint64_t;
using PathTraceNTCConstants = ShaderBuffer;
using PathTraceNTCWeights = ShaderBuffer;
using PathTraceNTCInfo = ShaderBuffer;
using PathTraceNTCSampler = ShaderSampler;
#else
typealias PathTraceSettings = uint*;
typealias PathTraceScene = DescriptorHandle<RaytracingAccelerationStructure>;
typealias PathTraceOutput = DescriptorHandle<RWTexture2D<float4>>;
typealias PathTraceVertices = DataSpan<SceneShadingVertex>;
typealias PathTraceIndices = DataSpan<uint>;
typealias PathTracePrimitives = DataSpan<PathTracePrimitive>;
typealias PathTraceInstances = DataSpan<PathTraceInstance>;
typealias PathTracePositions = DataSpan<SceneVertexPosition>;
typealias PathTraceMaterials = DescriptorHandle<StructuredBuffer<PathTraceMaterial>>;
typealias PathTraceHistoryCurrent = DescriptorHandle<RWTexture2D<float4>>;
typealias PathTraceHistoryPrevious = DescriptorHandle<RWTexture2D<float4>>;
typealias PathTraceMaterialTextures = DescriptorHandle<Texture2D<float4>>*;
typealias PathTraceEnvironment = DescriptorHandle<Texture2D<float4>>;
typealias PathTraceEnvironmentPdf = DescriptorHandle<Texture2D<float>>;
typealias PathTraceLUT2D = DescriptorHandle<Texture2D<float4>>*;
typealias PathTraceLUT3D = DescriptorHandle<Texture3D<float4>>*;
typealias PathTraceLights = DescriptorHandle<StructuredBuffer<GPUPunctualLight>>;
typealias PathTraceReGIR = DescriptorHandle<StructuredBuffer<uint4>>;
typealias PathTracePunctualPdf = DescriptorHandle<Texture2D<float>>;
typealias PathTraceAlbedo = DescriptorHandle<RWTexture2D<float4>>;
typealias PathTraceSpecularAlbedo = DescriptorHandle<RWTexture2D<float4>>;
typealias PathTraceNormalRoughness = DescriptorHandle<RWTexture2D<float4>>;
typealias PathTraceMotionVectors = DescriptorHandle<RWTexture2D<float2>>;
typealias PathTraceLinearDepth = DescriptorHandle<RWTexture2D<float>>;
typealias PathTraceSpecularHitDistance = DescriptorHandle<RWTexture2D<float>>;
typealias PathTraceDepth = DescriptorHandle<RWTexture2D<float>>;
typealias PathTraceMaterialValues = DescriptorHandle<StructuredBuffer<MaterialValueInstance>>;
typealias PathTraceNTCLatents = DescriptorHandle<Texture2DArray<float4>>*;
typealias PathTraceNTCConstants = DescriptorHandle<NeuralTextureConstantBuffer>;
typealias PathTraceNTCWeights = DescriptorHandle<ByteAddressBuffer>;
typealias PathTraceNTCInfo = DescriptorHandle<StructuredBuffer<uint4>>;
typealias PathTraceNTCSampler = DescriptorHandle<SamplerState>;
#endif
// Canonical resource handles and bounded BDA spans. settings points to an
// immutable ScenePathTracePush snapshot owned by the same encoded packet.
struct PathTraceParameters {
    PathTraceSettings settings;
    PathTraceScene scene;
    PathTraceOutput output;
    PathTraceVertices vertices;
    PathTraceIndices indices;
    PathTracePrimitives primitives;
    PathTraceInstances instances;
    PathTracePositions positions;
    PathTraceMaterials materials;
    PathTraceHistoryCurrent historyCurrent;
    PathTraceHistoryPrevious historyPrevious;
    PathTraceMaterialTextures materialTextures;
    PathTraceEnvironment environment;
    PathTraceEnvironmentPdf environmentPdf;
    PathTraceLUT2D lut2D;
    PathTraceLUT3D lut3D;
    PathTraceLights lights;
    PathTraceReGIR reGIR;
    PathTracePunctualPdf punctualPdf;
    PathTraceAlbedo albedo;
    PathTraceSpecularAlbedo specularAlbedo;
    PathTraceNormalRoughness normalRoughness;
    PathTraceMotionVectors motionVectors;
    PathTraceLinearDepth linearDepth;
    PathTraceSpecularHitDistance specularHitDistance;
    PathTraceDepth depth;
    PathTraceMaterialValues materialValues;
    PathTraceNTCLatents ntcLatents;
    PathTraceNTCConstants ntcConstants;
    PathTraceNTCWeights ntcWeights;
    PathTraceNTCInfo ntcInfo;
    PathTraceNTCSampler ntcSampler;
};
#ifdef __cplusplus
static_assert(sizeof(PathTraceParameters) == 296);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
