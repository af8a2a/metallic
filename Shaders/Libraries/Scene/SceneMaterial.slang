#ifndef METALLIC_SCENE_MATERIAL
#define METALLIC_SCENE_MATERIAL

// Shared upload ABI for path tracing and deferred material classification.
struct PathTraceTextureInfo
{
    uint textureIndex;
    uint texCoord;
    uint ntcTextureSetIndex;
    uint ntcChannelMapping;
    float4 transform0;
    float4 transform1;
};

struct PathTraceMaterial
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
    PathTraceTextureInfo baseColorTexture;
    PathTraceTextureInfo metallicRoughnessTexture;
    PathTraceTextureInfo normalTexture;
    PathTraceTextureInfo occlusionTexture;
    PathTraceTextureInfo emissiveTexture;
    PathTraceTextureInfo transmissionTexture;
    PathTraceTextureInfo thicknessTexture;
    PathTraceTextureInfo diffuseTransmissionTexture;
    PathTraceTextureInfo diffuseTransmissionColorTexture;
};

#endif // METALLIC_SCENE_MATERIAL
