#language slang 2026
implementing Material;

namespace Metallic.Material
{
// Shared upload ABI for path tracing and deferred material classification.
public struct MaterialTextureInfo
{
    public uint textureIndex;
    public uint texCoord;
    // Required for float4 alignment in StructuredBuffer std430 layout.
    public uint2 alignmentPadding;
    public float4 transform0;
    public float4 transform1;
};

public struct SceneMaterialData
{
    public float4 baseColor;
    public float4 emissive;
    public float4 params;
    public float4 textureParams;
    public float4 glassParams;
    public float4 attenuationColor;
    public float4 diffuseTransmission;
    public float4 rtxcrHairBaseColor;
    public float4 rtxcrHairParams0;
    public float4 rtxcrHairParams1;
    public float4 rtxcrHairDiffuseTint;
    public MaterialTextureInfo baseColorTexture;
    public MaterialTextureInfo metallicRoughnessTexture;
    public MaterialTextureInfo normalTexture;
    public MaterialTextureInfo occlusionTexture;
    public MaterialTextureInfo emissiveTexture;
    public MaterialTextureInfo transmissionTexture;
    public MaterialTextureInfo thicknessTexture;
    public MaterialTextureInfo diffuseTransmissionTexture;
    public MaterialTextureInfo diffuseTransmissionColorTexture;
    public float4 specular;
    public MaterialTextureInfo specularTexture;
    public MaterialTextureInfo specularColorTexture;
};

// Value expressions consume authoring inputs and texture slots, independently
// of the scattering model selected by the scene. This is not a BSDF payload.
public struct MaterialValueParameters
{
    public float4 baseColor;
    public float4 emissive;
    public float4 params;
    public MaterialTextureInfo baseColorTexture;
    public MaterialTextureInfo metallicRoughnessTexture;
    public MaterialTextureInfo normalTexture;
    public MaterialTextureInfo occlusionTexture;
    public MaterialTextureInfo emissiveTexture;
    public MaterialTextureInfo specularTexture;
};

public MaterialValueParameters materialValueParameters(MaterialValueParameters parameters)
{
    return parameters;
}

public MaterialValueParameters sceneMaterialValueParameters(SceneMaterialData data)
{
    MaterialValueParameters result;
    result.baseColor = data.baseColor;
    result.emissive = data.emissive;
    result.params = data.params;
    result.baseColorTexture = data.baseColorTexture;
    result.metallicRoughnessTexture = data.metallicRoughnessTexture;
    result.normalTexture = data.normalTexture;
    result.occlusionTexture = data.occlusionTexture;
    result.emissiveTexture = data.emissiveTexture;
    result.specularTexture = data.specularTexture;
    return result;
}

} // namespace Metallic.Material
