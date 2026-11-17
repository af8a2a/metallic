implementing SceneSurface;

import MaterialTextureDecode;
import AlphaCoverage;
import Core;
import Material;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import MaterialProgram;
import Lighting;
import RTXCRHair;
import ScenePathTypes;
import ScenePathMath;
import RTXCRMaterialAdapter;
import MaterialValueTextureSampling;
import RayMaterialExecution;
import StreamTextureSampling;
import WorkingColor;

namespace Metallic.SceneSurface
{
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.ScenePath;

#if SHARC_UPDATE
public RWTexture2D<float4> sceneOutput()
{
    return resolveUniform(getSceneResourceParameters().output);
}
#else
public RWTexture2D<float4> sceneOutput()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<RWTexture2D<float4>>>(kSceneOutputOffset));
}
#endif

public struct PathTracePrimitive
{
    public uint firstVertex;
    public uint vertexCount;
    public uint firstIndex;
    public uint indexCount;
    public uint flags;
    public uint padding0;
    public uint padding1;
    public uint padding2;
};

public struct PathTraceInstance
{
    public uint primitiveIndex;
    public uint materialIndex;
    public uint flags;
    public float rayConeLodConstant;
};


#if SHARC_UPDATE
public StructuredBuffer<SceneMaterialData> sceneMaterials()
{
    return resolveBuffer<StructuredBuffer<SceneMaterialData>>(getSceneResourceParameters().materials);
}
#else
public StructuredBuffer<SceneMaterialData> sceneMaterials()
{
    return resolveBuffer<StructuredBuffer<SceneMaterialData>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneMaterialsOffset));
}
#endif

#if SHARC_UPDATE
public RWTexture2D<float4> sceneHistoryCurrent()
{
    return resolveUniform(getSceneResourceParameters().accumulationCurrent);
}
#else
public RWTexture2D<float4> sceneHistoryCurrent()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<RWTexture2D<float4>>>(kSceneAccumulationCurrentOffset));
}
#endif

#if SHARC_UPDATE
public RWTexture2D<float4> sceneHistoryPrevious()
{
    return resolveUniform(getSceneResourceParameters().accumulationPrevious);
}
#else
public RWTexture2D<float4> sceneHistoryPrevious()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<RWTexture2D<float4>>>(kSceneAccumulationPreviousOffset));
}
#endif

#if SHARC_UPDATE
[ForceInline]
public Texture2D<float4> sceneMaterialTexture(uint index)
{
    return resolveNonUniform(ResourceHandle<Texture2D<float4>>(getSceneResourceParameterField<BufferSpan<uint>>(kSceneMaterialTexturesOffset).load(index)));
}
[ForceInline]
public SamplerState sceneMaterialSampler()
{
    return resolveUniform(getSceneResourceParameters().materialSampler);
}
[ForceInline]
public StructuredBuffer<TextureSamplingMetadata> sceneTextureSamplingMetadata()
{
    return resolveBuffer<StructuredBuffer<TextureSamplingMetadata>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneTextureSamplingMetadataOffset));
}
[ForceInline]
public ByteAddressBuffer sceneMaterialValues()
{
    return resolveUniform(getSceneResourceParameters().materialValues);
}
[ForceInline]
public Texture2D<float4> sceneEnvironmentMap()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float4>>>(kSceneEnvironmentMapOffset));
}
[ForceInline]
public Texture2D<float> sceneEnvironmentImportancePdf()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float>>>(kSceneEnvironmentPdfOffset));
}
#else
[ForceInline]
public Texture2D<float4> sceneMaterialTexture(uint index)
{
    return resolveNonUniform(ResourceHandle<Texture2D<float4>>(getSceneResourceParameterField<BufferSpan<uint>>(kSceneMaterialTexturesOffset).load(index)));
}
[ForceInline]
public SamplerState sceneMaterialSampler()
{
    return resolveUniform(getSceneResourceParameterField<SamplerHandle>(kSceneMaterialSamplerOffset));
}
[ForceInline]
public StructuredBuffer<TextureSamplingMetadata> sceneTextureSamplingMetadata()
{
    return resolveBuffer<StructuredBuffer<TextureSamplingMetadata>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneTextureSamplingMetadataOffset));
}
[ForceInline]
public ByteAddressBuffer sceneMaterialValues()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneMaterialValuesOffset));
}
[ForceInline]
public Texture2D<float4> sceneEnvironmentMap()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float4>>>(kSceneEnvironmentMapOffset));
}
[ForceInline]
public Texture2D<float> sceneEnvironmentImportancePdf()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float>>>(kSceneEnvironmentPdfOffset));
}
#endif

[ForceInline]
public RWStructuredBuffer<uint> sceneTextureDemandMip()
{
    return resolveBuffer<RWStructuredBuffer<uint>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneTextureDemandMipOffset));
}

[ForceInline]
public RWStructuredBuffer<uint> sceneTextureDemandHits()
{
    return resolveBuffer<RWStructuredBuffer<uint>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneTextureDemandHitsOffset));
}

#if SHARC_UPDATE
public StructuredBuffer<GPUPunctualLight> sceneLights()
{
    return resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getSceneResourceParameters().lights);
}
public Texture2D<float> sceneLightsPdf()
{
    return resolveUniform(getSceneResourceParameters().lightsPdf);
}
public StructuredBuffer<uint4> sceneReGIRGrid()
{
    return resolveBuffer<StructuredBuffer<uint4>>(getSceneResourceParameters().regirGrid);
}
#else
public StructuredBuffer<GPUPunctualLight> sceneLights()
{
    return resolveBuffer<StructuredBuffer<GPUPunctualLight>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneLightsOffset));
}
public Texture2D<float> sceneLightsPdf()
{
    return resolveUniform(getSceneResourceParameterField<ResourceHandle<Texture2D<float>>>(kSceneLightsPdfOffset));
}
public StructuredBuffer<uint4> sceneReGIRGrid()
{
    return resolveBuffer<StructuredBuffer<uint4>>(getSceneResourceParameterField<ResourceHandle<ByteAddressBuffer>>(kSceneReGIRGridOffset));
}
#endif

public EnvironmentResourceParameters getSceneEnvironmentResources()
{
#if SHARC_UPDATE
    return getSceneResourceParameters().environment;
#else
    return getSceneResourceParameterField<EnvironmentResourceParameters>(kSceneEnvironmentOffset);
#endif
}


public static const uint kInvalidMaterialTextureIndex = 0xffffffffu;
public static const uint kPrimitiveHasAuthoredTangents = 1u;
public static const uint kEnvironmentModeProcedural = 0u;
public static const uint kEnvironmentModeMap = 1u;
public static const uint kEnvironmentModeDisabled = 2u;


public SceneMaterialData loadSceneMaterial(uint materialIndex)
{
    return sceneMaterials()[materialIndex];
}

public OpenPBRMaterialParameters loadSceneOpenPBRMaterial(uint materialIndex)
{
    return sceneOpenPBRParameters(loadSceneMaterial(materialIndex));
}

public RTXCRHairMaterialParameters loadSceneFiberMaterial(uint materialIndex)
{
    return sceneFiberParameters(loadSceneMaterial(materialIndex));
}

} // namespace Metallic.SceneSurface
