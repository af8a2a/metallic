#language slang 2026
implementing GPUDriven;

namespace Metallic.GPUDriven
{
// Shared GPUScene ABI for culling, visibility rasterization and material resolve.
public static const uint kInvalidTextureIndex = 0xffffffffu;

public struct GPUDrivenPreviewPush
{
    public uint positionBuffer;
    public uint meshletBuffer;
    public uint meshletDrawBuffer;
    public uint meshletVertexBuffer;
    public uint meshletTriangleBuffer;
    public uint paramsBuffer;
    public uint transformBuffer;
    public uint instanceBuffer;
    public uint instanceVisibilityBuffer;
    public uint visibleInstanceIdsBuffer;
    public uint visibleInstanceCounterBuffer;
    public uint visibleMeshletBuffer0;
    public uint visibleMeshletBuffer1;
    public uint indirectBuffer0;
    public uint indirectBuffer1;
    public uint hzbBuffer0;
    public uint hzbBuffer1;
    public uint deferredColorBuffer;
    public uint depthImage;
    public uint visibilityImage;
    public uint passIndex;
    public uint mipLevel;
    public uint projectWithCullingCamera;
    public uint materialBuffer;
    public uint materialTextureRemapBuffer;
    public uint environmentImage;
    public uint environmentSHBuffer;
    public uint streamDeferredBindingsBuffer;
    public uint residentRecordCapacity;
    public uint streamOwnerMaskBuffer;
    public float tessellationEdgePixels;
    public uint tessellationMaxFactor;
    public uint tessellationMaxSplitDepth;
};

public struct GPUDrivenPreviewVertex
{
    public float4 position;
    public float4 normal;
    public float4 tangent;
    public float2 texcoord;
    public uint flags;
    public uint reserved;
};

public struct SceneTransform
{
    public float4 world0;
    public float4 world1;
    public float4 world2;
    public float4 world3;
};

public struct GPUDrivenPreviewMeshlet
{
    public uint4 ranges;
    public uint4 lod;
    public float4 boundingSphere;
    public float4 coneApexCutoff;
    public float4 coneAxisLodError;
};

public struct GPUDrivenPreviewGeometry
{
    public uint4 source;
    public uint4 counts;
    public uint4 payload;
    public uint4 meshletPayload;
    public float4 localBoundingSphere;
    public uint4 identity;
};

public struct GPUDrivenPreviewInstance
{
    public float4 world0;
    public float4 world1;
    public float4 world2;
    public float4 world3;
    public float4 previousWorld0;
    public float4 previousWorld1;
    public float4 previousWorld2;
    public float4 previousWorld3;
    public float4 localBoundingSphere;
    public uint4 identity;
};

public SceneTransform instanceTransform(GPUDrivenPreviewInstance instance)
{
    SceneTransform transform;
    transform.world0 = instance.world0;
    transform.world1 = instance.world1;
    transform.world2 = instance.world2;
    transform.world3 = instance.world3;
    return transform;
}

public uint loadPackedMeshletTriangleIndex(
    StructuredBuffer<uint> triangleWords,
    uint firstWord,
    uint triangleIndex,
    uint triangleCorner)
{
    uint byteOffset = triangleIndex * 3u + triangleCorner;
    uint word = triangleWords[firstWord + (byteOffset >> 2u)];
    return (word >> ((byteOffset & 3u) * 8u)) & 0xffu;
}

public struct GPUDrivenPreviewTextureInfo
{
    public uint textureIndex;
    public uint texCoord;
    public uint padding0;
    public uint padding1;
    public float4 transform0;
    public float4 transform1;
};

public struct GPUDrivenPreviewMaterial
{
    public float4 baseColor;
    public float4 emissive;
    public float4 params;
    public float4 textureParams;
    public float4 glassParams;
    public float4 attenuationColor;
    public float4 diffuseTransmission;
    public GPUDrivenPreviewTextureInfo baseColorTexture;
    public GPUDrivenPreviewTextureInfo metallicRoughnessTexture;
    public GPUDrivenPreviewTextureInfo normalTexture;
    public GPUDrivenPreviewTextureInfo occlusionTexture;
    public GPUDrivenPreviewTextureInfo emissiveTexture;
    public GPUDrivenPreviewTextureInfo transmissionTexture;
    public GPUDrivenPreviewTextureInfo thicknessTexture;
    public GPUDrivenPreviewTextureInfo diffuseTransmissionTexture;
    public GPUDrivenPreviewTextureInfo diffuseTransmissionColorTexture;
    public uint4 identity;
};

public struct GPUDrivenPreviewParams
{
    public float4 eye;
    public float4 center;
    public float4 upProjection;
    public float4 viewport;
    public float4 clipOrtho;
    public float4 clearColor;
    public float4 previousEye;
    public float4 previousCenter;
    public float4 previousUpProjection;
    public float4 previousViewport;
    public float4 previousClipOrtho;
    public float4 renderEye;
    public float4 renderCenter;
    public float4 renderUpProjection;
    public float4 renderViewport;
    public float4 renderClipOrtho;
    public uint mode;
    public uint meshletOffset;
    public uint meshletCount;
    public uint selectedLodLevel;
    public uint instanceCount;
    public uint width;
    public uint height;
    public uint hzbMipCount;
    public uint frameIndex;
    public uint hzbValid;
    public uint cullingFlags;
    public uint materialTextureCount;
    public float environmentIntensity;
    public float environmentRotationRadians;
    public uint environmentMode;
    public uint environmentVisible;
    public uint materialCount;
    public uint visibleMeshletCapacity;
    public uint lodSelectionBuffer;
    public uint lodSelectionEnabled;
    public uint tessellationBuffer;
    public float displacementBound;
    public uint2 tessellationPadding;
};

public uint residentLodRecordIndex(uint candidate, GPUDrivenPreviewParams params, GPUDrivenPreviewPush push)
{
    if (params.lodSelectionEnabled == 0u) { return params.meshletOffset + candidate; }
    DescriptorHandle<StructuredBuffer<uint4>> selections =
        DescriptorHandle<StructuredBuffer<uint4>>(uint2(params.lodSelectionBuffer, 0));
    uint count = min(selections[0].x, selections[0].y);
    return candidate < count ? selections[candidate + 1u].z : 0xffffffffu;
}
} // namespace Metallic.GPUDriven
