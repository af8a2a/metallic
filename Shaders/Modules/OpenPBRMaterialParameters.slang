#language slang 2026
module OpenPBRMaterialParameters;

import Material;

namespace Metallic.Material
{
// Only the surface model's inputs enter its program. Scene storage is decoded
// at the resource boundary; no Fiber fields or other closures are retained.
public struct OpenPBRMaterialParameters
{
    public float4 baseColor;
    public float4 emissive;
    public float4 params;
    public float4 textureParams;
    public float4 glassParams;
    public float4 attenuationColor;
    public float4 diffuseTransmission;
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

public OpenPBRMaterialParameters sceneOpenPBRParameters(SceneMaterialData data)
{
    OpenPBRMaterialParameters result;
    result.baseColor = data.baseColor;
    result.emissive = data.emissive;
    result.params = data.params;
    result.textureParams = data.textureParams;
    result.glassParams = data.glassParams;
    result.attenuationColor = data.attenuationColor;
    result.diffuseTransmission = data.diffuseTransmission;
    result.baseColorTexture = data.baseColorTexture;
    result.metallicRoughnessTexture = data.metallicRoughnessTexture;
    result.normalTexture = data.normalTexture;
    result.occlusionTexture = data.occlusionTexture;
    result.emissiveTexture = data.emissiveTexture;
    result.transmissionTexture = data.transmissionTexture;
    result.thicknessTexture = data.thicknessTexture;
    result.diffuseTransmissionTexture = data.diffuseTransmissionTexture;
    result.diffuseTransmissionColorTexture = data.diffuseTransmissionColorTexture;
    result.specular = data.specular;
    result.specularTexture = data.specularTexture;
    result.specularColorTexture = data.specularColorTexture;
    return result;
}

public MaterialValueParameters materialValueParameters(OpenPBRMaterialParameters parameters)
{
    MaterialValueParameters result;
    result.baseColor = parameters.baseColor;
    result.emissive = parameters.emissive;
    result.params = parameters.params;
    result.baseColorTexture = parameters.baseColorTexture;
    result.metallicRoughnessTexture = parameters.metallicRoughnessTexture;
    result.normalTexture = parameters.normalTexture;
    result.occlusionTexture = parameters.occlusionTexture;
    result.emissiveTexture = parameters.emissiveTexture;
    result.specularTexture = parameters.specularTexture;
    return result;
}

public MaterialCoverageParameters materialCoverageParameters(OpenPBRMaterialParameters parameters)
{
    MaterialCoverageParameters result;
    result.baseAlpha = parameters.baseColor.a;
    result.cutoff = parameters.params.z;
    result.alphaMode = parameters.textureParams.w;
    result.hasProgram = parameters.attenuationColor.w > 0.5;
    result.baseColorTexture = parameters.baseColorTexture;
    return result;
}
} // namespace Metallic.Material
