#language slang 2026
implementing Material;

namespace Metallic.Material
{
// Shared upload ABI for path tracing and deferred material classification.
public struct PathTraceTextureInfo
{
    public uint textureIndex;
    public uint texCoord;
    // Preserve the 48-byte texture descriptor ABI; ordinary materials use textureIndex only.
    public uint reserved0;
    public uint reserved1;
    public float4 transform0;
    public float4 transform1;
};

public struct PathTraceMaterial
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
    public PathTraceTextureInfo baseColorTexture;
    public PathTraceTextureInfo metallicRoughnessTexture;
    public PathTraceTextureInfo normalTexture;
    public PathTraceTextureInfo occlusionTexture;
    public PathTraceTextureInfo emissiveTexture;
    public PathTraceTextureInfo transmissionTexture;
    public PathTraceTextureInfo thicknessTexture;
    public PathTraceTextureInfo diffuseTransmissionTexture;
    public PathTraceTextureInfo diffuseTransmissionColorTexture;
    public float4 specular;
    public PathTraceTextureInfo specularTexture;
    public PathTraceTextureInfo specularColorTexture;
};

} // namespace Metallic.Material
