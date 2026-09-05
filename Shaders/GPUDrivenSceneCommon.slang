#ifndef GPU_DRIVEN_SCENE_COMMON_SLANG
#define GPU_DRIVEN_SCENE_COMMON_SLANG

#include "GPUDrivenRasterCommon.slang"

// Shared GPUScene ABI for culling, visibility rasterization and material resolve.
static const uint kInvalidTextureIndex = 0xffffffffu;

struct GPUDrivenPreviewPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint positionBuffer;
    uint meshletBuffer;
    uint meshletDrawBuffer;
    uint meshletVertexBuffer;
    uint meshletTriangleBuffer;
    uint paramsBuffer;
    uint transformBuffer;
    uint instanceBuffer;
    uint instanceVisibilityBuffer;
    uint visibleInstanceIdsBuffer;
    uint visibleInstanceCounterBuffer;
    uint visibleMeshletBuffer0;
    uint visibleMeshletBuffer1;
    uint indirectBuffer0;
    uint indirectBuffer1;
    uint hzbBuffer0;
    uint hzbBuffer1;
    uint deferredColorBuffer;
    uint depthImage;
    uint visibilityImage;
    uint passIndex;
    uint mipLevel;
    uint projectWithCullingCamera;
    uint materialBuffer;
    uint materialTextureRemapBuffer;
    uint environmentImage;
    uint environmentSHBuffer;
    uint streamDeferredBindingsBuffer;
    uint residentRecordCapacity;
    uint streamOwnerMaskBuffer;
};

struct GPUDrivenPreviewVertex
{
    float4 position;
    float4 normal;
    float4 tangent;
    float2 texcoord;
    uint flags;
    uint reserved;
};

struct SceneTransform
{
    float4 world0;
    float4 world1;
    float4 world2;
    float4 world3;
};

struct GPUDrivenPreviewMeshlet
{
    uint4 ranges;
    uint4 lod;
    float4 boundingSphere;
    float4 coneApexCutoff;
    float4 coneAxisLodError;
};

struct GPUDrivenPreviewGeometry
{
    uint4 source;
    uint4 counts;
    uint4 payload;
    uint4 meshletPayload;
    float4 localBoundingSphere;
    uint4 identity;
};

struct GPUDrivenPreviewInstance
{
    float4 world0;
    float4 world1;
    float4 world2;
    float4 world3;
    float4 previousWorld0;
    float4 previousWorld1;
    float4 previousWorld2;
    float4 previousWorld3;
    float4 localBoundingSphere;
    uint4 identity;
};

SceneTransform instanceTransform(GPUDrivenPreviewInstance instance)
{
    SceneTransform transform;
    transform.world0 = instance.world0;
    transform.world1 = instance.world1;
    transform.world2 = instance.world2;
    transform.world3 = instance.world3;
    return transform;
}

uint loadPackedMeshletTriangleIndex(
    StructuredBuffer<uint> triangleWords,
    uint firstWord,
    uint triangleIndex,
    uint triangleCorner)
{
    uint byteOffset = triangleIndex * 3u + triangleCorner;
    uint word = triangleWords[firstWord + (byteOffset >> 2u)];
    return (word >> ((byteOffset & 3u) * 8u)) & 0xffu;
}

struct GPUDrivenPreviewTextureInfo
{
    uint textureIndex;
    uint texCoord;
    uint padding0;
    uint padding1;
    float4 transform0;
    float4 transform1;
};

struct GPUDrivenPreviewMaterial
{
    float4 baseColor;
    float4 emissive;
    float4 params;
    float4 textureParams;
    float4 glassParams;
    float4 attenuationColor;
    float4 diffuseTransmission;
    GPUDrivenPreviewTextureInfo baseColorTexture;
    GPUDrivenPreviewTextureInfo metallicRoughnessTexture;
    GPUDrivenPreviewTextureInfo normalTexture;
    GPUDrivenPreviewTextureInfo occlusionTexture;
    GPUDrivenPreviewTextureInfo emissiveTexture;
    GPUDrivenPreviewTextureInfo transmissionTexture;
    GPUDrivenPreviewTextureInfo thicknessTexture;
    GPUDrivenPreviewTextureInfo diffuseTransmissionTexture;
    GPUDrivenPreviewTextureInfo diffuseTransmissionColorTexture;
    uint4 identity;
};

struct GPUDrivenPreviewParams
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    float4 clearColor;
    float4 previousEye;
    float4 previousCenter;
    float4 previousUpProjection;
    float4 previousViewport;
    float4 previousClipOrtho;
    float4 renderEye;
    float4 renderCenter;
    float4 renderUpProjection;
    float4 renderViewport;
    float4 renderClipOrtho;
    uint mode;
    uint meshletOffset;
    uint meshletCount;
    uint selectedLodLevel;
    uint instanceCount;
    uint width;
    uint height;
    uint hzbMipCount;
    uint frameIndex;
    uint hzbValid;
    uint cullingFlags;
    uint materialTextureCount;
    float environmentIntensity;
    float environmentRotationRadians;
    uint environmentMode;
    uint environmentVisible;
    uint materialCount;
    uint visibleMeshletCapacity;
    uint shadingPadding1;
    uint shadingPadding2;
};

#endif
