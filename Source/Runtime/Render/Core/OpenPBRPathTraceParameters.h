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
using OpenPBRSettings = uint64_t;
using OpenPBRScene = ShaderAccelerationStructure;
using OpenPBROutput = ShaderStorageImage;
using OpenPBRVertices = ShaderDataSpan;
using OpenPBRIndices = ShaderDataSpan;
using OpenPBRPrimitives = ShaderDataSpan;
using OpenPBRInstances = ShaderDataSpan;
using OpenPBRPositions = ShaderDataSpan;
using OpenPBRMaterials = ShaderBuffer;
using OpenPBRHistoryCurrent = ShaderStorageImage;
using OpenPBRHistoryPrevious = ShaderStorageImage;
using OpenPBRMaterialTextures = uint64_t;
using OpenPBREnvironment = ShaderSampledImage;
using OpenPBREnvironmentPdf = ShaderSampledImage;
using OpenPBRLUT2D = uint64_t;
using OpenPBRLUT3D = uint64_t;
using OpenPBRLights = ShaderBuffer;
using OpenPBRReGIR = ShaderBuffer;
using OpenPBRPunctualPdf = ShaderSampledImage;
using OpenPBRAlbedo = ShaderStorageImage;
using OpenPBRSpecularAlbedo = ShaderStorageImage;
using OpenPBRNormalRoughness = ShaderStorageImage;
using OpenPBRMotionVectors = ShaderStorageImage;
using OpenPBRLinearDepth = ShaderStorageImage;
using OpenPBRSpecularHitDistance = ShaderStorageImage;
using OpenPBRDepth = ShaderStorageImage;
using OpenPBRMaterialValues = ShaderBuffer;
using OpenPBRNTCLatents = uint64_t;
using OpenPBRNTCConstants = ShaderBuffer;
using OpenPBRNTCWeights = ShaderBuffer;
using OpenPBRNTCInfo = ShaderBuffer;
using OpenPBRNTCSampler = ShaderSampler;
#else
typealias OpenPBRSettings = uint*;
typealias OpenPBRScene = DescriptorHandle<RaytracingAccelerationStructure>;
typealias OpenPBROutput = DescriptorHandle<RWTexture2D<float4>>;
typealias OpenPBRVertices = DataSpan<SceneShadingVertex>;
typealias OpenPBRIndices = DataSpan<uint>;
typealias OpenPBRPrimitives = DataSpan<PathTracePrimitive>;
typealias OpenPBRInstances = DataSpan<PathTraceInstance>;
typealias OpenPBRPositions = DataSpan<SceneVertexPosition>;
typealias OpenPBRMaterials = DescriptorHandle<StructuredBuffer<PathTraceMaterial>>;
typealias OpenPBRHistoryCurrent = DescriptorHandle<RWTexture2D<float4>>;
typealias OpenPBRHistoryPrevious = DescriptorHandle<RWTexture2D<float4>>;
typealias OpenPBRMaterialTextures = DescriptorHandle<Texture2D<float4>>*;
typealias OpenPBREnvironment = DescriptorHandle<Texture2D<float4>>;
typealias OpenPBREnvironmentPdf = DescriptorHandle<Texture2D<float>>;
typealias OpenPBRLUT2D = DescriptorHandle<Texture2D<float4>>*;
typealias OpenPBRLUT3D = DescriptorHandle<Texture3D<float4>>*;
typealias OpenPBRLights = DescriptorHandle<StructuredBuffer<GPUPunctualLight>>;
typealias OpenPBRReGIR = DescriptorHandle<StructuredBuffer<uint4>>;
typealias OpenPBRPunctualPdf = DescriptorHandle<Texture2D<float>>;
typealias OpenPBRAlbedo = DescriptorHandle<RWTexture2D<float4>>;
typealias OpenPBRSpecularAlbedo = DescriptorHandle<RWTexture2D<float4>>;
typealias OpenPBRNormalRoughness = DescriptorHandle<RWTexture2D<float4>>;
typealias OpenPBRMotionVectors = DescriptorHandle<RWTexture2D<float2>>;
typealias OpenPBRLinearDepth = DescriptorHandle<RWTexture2D<float>>;
typealias OpenPBRSpecularHitDistance = DescriptorHandle<RWTexture2D<float>>;
typealias OpenPBRDepth = DescriptorHandle<RWTexture2D<float>>;
typealias OpenPBRMaterialValues = DescriptorHandle<StructuredBuffer<MaterialValueInstance>>;
typealias OpenPBRNTCLatents = DescriptorHandle<Texture2DArray<float4>>*;
typealias OpenPBRNTCConstants = DescriptorHandle<NeuralTextureConstantBuffer>;
typealias OpenPBRNTCWeights = DescriptorHandle<ByteAddressBuffer>;
typealias OpenPBRNTCInfo = DescriptorHandle<StructuredBuffer<uint4>>;
typealias OpenPBRNTCSampler = DescriptorHandle<SamplerState>;
#endif
// Canonical resource handles and bounded BDA spans. settings points to an
// immutable ScenePathTracePush snapshot owned by the same encoded packet.
struct OpenPBRPathTraceParameters {
    OpenPBRSettings settings;
    OpenPBRScene scene;
    OpenPBROutput output;
    OpenPBRVertices vertices;
    OpenPBRIndices indices;
    OpenPBRPrimitives primitives;
    OpenPBRInstances instances;
    OpenPBRPositions positions;
    OpenPBRMaterials materials;
    OpenPBRHistoryCurrent historyCurrent;
    OpenPBRHistoryPrevious historyPrevious;
    OpenPBRMaterialTextures materialTextures;
    OpenPBREnvironment environment;
    OpenPBREnvironmentPdf environmentPdf;
    OpenPBRLUT2D lut2D;
    OpenPBRLUT3D lut3D;
    OpenPBRLights lights;
    OpenPBRReGIR reGIR;
    OpenPBRPunctualPdf punctualPdf;
    OpenPBRAlbedo albedo;
    OpenPBRSpecularAlbedo specularAlbedo;
    OpenPBRNormalRoughness normalRoughness;
    OpenPBRMotionVectors motionVectors;
    OpenPBRLinearDepth linearDepth;
    OpenPBRSpecularHitDistance specularHitDistance;
    OpenPBRDepth depth;
    OpenPBRMaterialValues materialValues;
    OpenPBRNTCLatents ntcLatents;
    OpenPBRNTCConstants ntcConstants;
    OpenPBRNTCWeights ntcWeights;
    OpenPBRNTCInfo ntcInfo;
    OpenPBRNTCSampler ntcSampler;
};
#ifdef __cplusplus
inline constexpr uint64_t kOpenPBRPathTraceABI = 0x4f50425250540001ull;
static_assert(sizeof(OpenPBRPathTraceParameters) == 296);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)
