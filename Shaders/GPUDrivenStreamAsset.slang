#include "GPUDrivenRasterCommon.slang"

static const uint kMaxMeshletVertices = 128;
static const uint kMaxMeshletTriangles = 128;
static const uint kMeshletTriangleChunkSize = 64u;
static const uint kMeshletTriangleChunkCount = 2u;
static const uint kMaxMeshOutputVertices = kMeshletTriangleChunkSize * 3u;
static const uint kPayloadHeaderBytes = 112u;
static const uint kPayloadClusterBytes = 96u;
static const uint kPayloadClusterWords = kPayloadClusterBytes / 4u;
static const uint kPayloadAttributePosition = 1u << 0u;
static const uint kPayloadAttributeNormal = 1u << 1u;
static const uint kPayloadAttributeTexcoord0 = 1u << 2u;
static const uint kPayloadAttributeTangent = 1u << 4u;
static const uint kPayloadFormatFloat32x2 = 1u;
static const uint kPayloadFormatFloat32x4 = 2u;
static const uint kInvalidStreamDeviceOffset = 0xffffffffu;
static const uint kInvalidClusterIndex = 0xffffffffu;
static const uint kUnloadClusterIndex = 0xfffffffeu;
static const uint kStreamStatePendingUpload = 1u;
static const uint kStreamStateResident = 2u;
static const uint kStreamStateLockedFallback = 3u;
static const uint kStreamPageStateMask = 0x00000007u;
static const uint kStreamPageDeviceOffsetMask = 0xfffffff8u;
static const uint kNoDebugLodOverride = 0xffffffffu;
static const uint kActiveGroupResident = 1u << 0u;
static const uint kActiveGroupLoadRequest = 1u << 1u;
static const uint kActiveGroupUnloadRequest = 1u << 2u;
static const uint kTraversalPhaseLoad = 0u;
static const uint kTraversalPhaseUnload = 1u;
static const uint kActiveBuildPhaseReset = 0u;
static const uint kActiveBuildPhaseBuild = 1u;
static const uint kActiveBuildPhaseFinalize = 2u;
static const uint kActiveBuildPhaseSeed = 3u;
static const uint kActiveBuildPhaseRun = 4u;
static const uint kBlasInputPhaseReset = 0u;
static const uint kBlasInputPhaseCount = 1u;
static const uint kBlasInputPhaseSetup = 2u;
static const uint kBlasInputPhaseInsert = 3u;
static const uint kClasPageStateActive = 1u;
static const uint kClasPageAddressOffsetMask = 0x3fffffffu;
static const uint kClasPageStateShift = 30u;
static const uint kBlasInstanceFallback = 1u << 0u;
static const uint kBlasInstanceDynamic = 1u << 1u;
static const uint kBlasInstanceOverflow = 1u << 2u;
static const uint kTraversalStackCapacity = 128u;
static const uint kStreamRequestHeaderWords = 16u;
static const uint kStreamUpdateHeaderWords = 16u;
static const uint kStreamRequestMaxLoadsWord = 0u;
static const uint kStreamRequestMaxUnloadsWord = 1u;
static const uint kStreamRequestLoadCounterWord = 2u;
static const uint kStreamRequestUnloadCounterWord = 3u;
static const uint kStreamRequestFrameIndexWord = 4u;
static const uint kStreamRequestLoadOverflowCounterWord = 5u;
static const uint kStreamRequestUnloadOverflowCounterWord = 6u;
static const uint kStreamRequestInvalidPageCounterWord = 7u;
static const uint kStreamRequestLastLoadOverflowFrameWord = 8u;
static const uint kStreamRequestLastUnloadOverflowFrameWord = 9u;
static const uint kStreamRequestLastInvalidPageFrameWord = 10u;

struct GPUDrivenStreamAssetPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint pageBuffer;
    uint activeGroupBuffer;
    uint pageTableBuffer;
    uint paramsBuffer;
    uint requestBuffer;
    uint residentPageBuffer;
    uint updateBuffer;
    uint activeHeaderBuffer;
    uint instanceBuffer;
    uint primitiveBuffer;
    uint lodLevelBuffer;
    uint groupBuffer;
    uint nodeBuffer;
    uint drawIndirectBuffer;
    uint traversalHeaderBuffer;
    uint traversalWorkBuffer;
    uint clasAddressBuffer;
    uint clasPageTableBuffer;
    uint blasHeaderBuffer;
    uint instanceBlasBuffer;
    uint blasBuildInfoBuffer;
    uint blasClusterReferenceBuffer;
    uint fallbackBlasAddressBuffer;
    uint dynamicBlasAddressBuffer;
    uint tlasInstanceBuffer;
    uint traversalPhase;
    uint activeBuildPhase;
    uint rasterBindingsBuffer;
};

struct GPUDrivenStreamAssetRasterBindings
{
    uint visibleClusterBuffer;
    uint instanceVisibilityBuffer;
    uint hzbBuffer0;
    uint hzbBuffer1;
    uint depthImage;
    uint visibilityImage;
    uint deferredColorBuffer;
    uint visibleInstanceIdsBuffer;
    uint visibleRecordBase;
    uint visibleRecordCapacity;
    uint hzbMipCount;
    uint hzbValid;
    uint cullingFlags;
    uint width;
    uint height;
    uint visibleInstanceCounterBuffer;
};

struct GPUDrivenStreamAssetActiveHeader
{
    uint activeGroupCount;
    uint activeGroupCapacity;
    uint maxActiveGroupClusters;
    uint overflowCount;
    uint frameIndex;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct GPUDrivenStreamAssetActiveGroup
{
    uint pageDeviceOffsetBytes;
    uint pageIndex;
    uint clusterCount;
    uint primitiveIndex;
    uint lodLevel;
    uint materialIndex;
    uint clusterSelectionMask;
    uint flags;
    uint instanceIndex;
    uint gpuSceneInstanceIndex;
    uint padding1;
    uint padding2;
    float4 world0;
    float4 world1;
    float4 world2;
    float4 world3;
};

struct GPUDrivenStreamAssetInstance
{
    uint primitiveIndex;
    uint materialIndex;
    uint visible;
    uint gpuSceneInstanceIndex;
    float4 world0;
    float4 world1;
    float4 world2;
    float4 world3;
    float4 boundsCenterRadius;
};

struct GPUDrivenStreamAssetPrimitive
{
    uint lodLevelOffset;
    uint lodLevelCount;
    uint pageOffset;
    uint pageCount;
    uint fallbackPageOffset;
    uint fallbackPageCount;
    uint groupOffset;
    uint groupCount;
    uint fallbackGroupOffset;
    uint fallbackGroupCount;
    uint materialIndex;
    uint nodeOffset;
    uint nodeCount;
    uint padding2;
    uint padding3;
    uint padding4;
};

struct GPUDrivenStreamAssetLodLevel
{
    uint pageOffset;
    uint pageCount;
    uint lodLevel;
    uint clusterCount;
    float minBoundingSphereRadius;
    float minMaxQuadricError;
    uint padding0;
    uint padding1;
};

struct GPUDrivenStreamAssetGroup
{
    uint primitiveIndex;
    uint pageIndex;
    uint lodLevel;
    uint clusterCount;
    float boundsCenterX;
    float boundsCenterY;
    float boundsCenterZ;
    float boundsRadius;
    float maxQuadricError;
};

struct GPUDrivenStreamAssetNode
{
    uint primitiveIndex;
    uint childOffset;
    uint childCount;
    uint groupIndex;
    float4 boundsCenterRadius;
    float maxQuadricError;
    uint lodLevel;
    uint padding0;
    uint padding1;
};

struct GPUDrivenStreamAssetDrawIndirect
{
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

struct GPUDrivenStreamAssetTraversalHeader
{
    uint readCounter;
    uint writeCounter;
    uint taskCounter;
    uint overflowCount;
    uint frameIndex;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct GPUDrivenStreamAssetTraversalWorkItem
{
    uint instanceIndex;
    uint nodeIndex;
    uint readyFrame;
    uint padding0;
};

struct GPUDrivenStreamAssetClasPageEntry
{
    uint addressOffsetAndState;
};

uint clasPageAddressOffset(GPUDrivenStreamAssetClasPageEntry entry)
{
    return entry.addressOffsetAndState & kClasPageAddressOffsetMask;
}

uint clasPageState(GPUDrivenStreamAssetClasPageEntry entry)
{
    return entry.addressOffsetAndState >> kClasPageStateShift;
}

struct GPUDrivenStreamAssetBlasHeader
{
    uint clusterReferenceCount;
    uint blasBuildCount;
    uint clusterReferenceCapacity;
    uint overflowCount;
    uint frameIndex;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct GPUDrivenStreamAssetInstanceBlas
{
    uint clusterReferenceOffset;
    uint clusterReferenceCapacity;
    uint selectedClusterCount;
    uint insertedClusterCount;
    uint blasBuildIndex;
    uint flags;
    uint padding0;
    uint padding1;
};

struct GPUDrivenStreamAssetBlasBuildInfo
{
    uint clusterReferencesCount;
    uint clusterReferencesStride;
    uint clusterReferencesAddressLow;
    uint clusterReferencesAddressHigh;
};

struct GPUDrivenStreamAssetTlasInstance
{
    float4 transform0;
    float4 transform1;
    float4 transform2;
    uint customIndexMask;
    uint shaderBindingTableFlags;
    uint accelerationStructureReferenceLow;
    uint accelerationStructureReferenceHigh;
};

struct GPUDrivenStreamAssetPageTableEntry
{
    uint deviceOffsetAndState;
    uint lastRequestFrame;
};

uint streamPageDeviceOffset(GPUDrivenStreamAssetPageTableEntry entry)
{
    return (entry.deviceOffsetAndState & kStreamPageStateMask) == 0u
        ? kInvalidStreamDeviceOffset
        : entry.deviceOffsetAndState & kStreamPageDeviceOffsetMask;
}

uint streamPageState(GPUDrivenStreamAssetPageTableEntry entry)
{
    return entry.deviceOffsetAndState & kStreamPageStateMask;
}

struct GPUDrivenStreamAssetRequestHeader
{
    uint maxLoadRequests;
    uint maxUnloadRequests;
    uint loadCounter;
    uint unloadCounter;
    uint frameIndex;
    uint loadOverflowCounter;
    uint unloadOverflowCounter;
    uint invalidPageCounter;
    uint lastLoadOverflowFrame;
    uint lastUnloadOverflowFrame;
    uint lastInvalidPageFrame;
    uint padding0;
    uint padding1;
    uint padding2;
    uint padding3;
    uint padding4;
};

struct GPUDrivenStreamAssetUpdateHeader
{
    uint patchUnloadPageCount;
    uint patchPageCount;
    uint frameIndex;
    uint patchOverflowCounter;
    uint padding0;
    uint padding1;
    uint padding2;
    uint padding3;
    uint padding4;
    uint padding5;
    uint padding6;
    uint padding7;
    uint padding8;
    uint padding9;
    uint padding10;
    uint padding11;
};

struct GPUDrivenStreamAssetParams
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    float4 clearColor;
    uint debugColorMode;
    uint pageBufferBytes;
    uint drawTaskCount;
    uint frameIndex;
    uint maxGpuPageRequests;
    uint maxGpuPageUnloadRequests;
    uint activeGroupCount;
    uint maxActiveGroupClusters;
    uint sceneInstanceCount;
    uint scenePrimitiveCount;
    uint sceneLodLevelCount;
    uint scenePageCount;
    uint selectedLodLevel;
    uint enableGpuLodSelection;
    uint enableGpuUnloadRequests;
    uint sceneGroupCount;
    uint maxPrimitiveGroupCount;
    uint sceneNodeCount;
    uint traversalWorkerCount;
    uint traversalWorkCapacity;
    uint blasClusterReferenceAddressLow;
    uint blasClusterReferenceAddressHigh;
    uint blasClusterReferenceCapacity;
    uint blasBuildCapacity;
    float4 previousEye;
    float4 previousCenter;
    float4 previousUpProjection;
    float4 previousViewport;
    float4 previousClipOrtho;
};

struct GPUDrivenStreamAssetVertexOutput
{
    float4 position : SV_Position;
    nointerpolation uint visibilityId : TEXCOORD0;
};

struct GPUDrivenStreamAssetFullscreenOutput
{
    float4 position : SV_Position;
};

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

bool streamPayloadRangeValid(
    uint byteOffset,
    uint byteCount,
    uint payloadBytes)
{
    return byteOffset <= payloadBytes && byteCount <= payloadBytes - byteOffset;
}

bool streamPayloadArrayRangeValid(
    uint byteOffset,
    uint elementCount,
    uint elementStride,
    uint payloadBytes)
{
    return elementStride != 0u &&
        byteOffset <= payloadBytes &&
        elementCount <= (payloadBytes - byteOffset) / elementStride;
}

float3 transformStreamDirection(
    GPUDrivenStreamAssetActiveGroup group,
    float3 localDirection)
{
    return group.world0.xyz * localDirection.x +
        group.world1.xyz * localDirection.y +
        group.world2.xyz * localDirection.z;
}

float3 transformStreamNormal(
    GPUDrivenStreamAssetActiveGroup group,
    float3 localNormal,
    float3 fallback)
{
    float3 cofactor0 = cross(group.world1.xyz, group.world2.xyz);
    float3 cofactor1 = cross(group.world2.xyz, group.world0.xyz);
    float3 cofactor2 = cross(group.world0.xyz, group.world1.xyz);
    return safeNormalize(
        cofactor0 * localNormal.x +
            cofactor1 * localNormal.y +
            cofactor2 * localNormal.z,
        fallback);
}

float4 projectPosition(float3 worldPosition, GPUDrivenStreamAssetParams params)
{
    float3 forward = safeNormalize(params.center.xyz - params.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(params.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));

    float3 cameraVector = worldPosition - params.eye.xyz;
    float3 viewPosition = float3(
        dot(cameraVector, right),
        dot(cameraVector, up),
        dot(cameraVector, forward));

    float zNear = max(params.clipOrtho.x, 0.0001);
    float zFar = max(params.clipOrtho.y, zNear + 0.0001);
    float ndcZ = (viewPosition.z - zNear) / (zFar - zNear);
    if (params.upProjection.w <= 0.5) {
        float safeViewZ = abs(viewPosition.z) > 0.000001
            ? viewPosition.z
            : (viewPosition.z < 0.0 ? -0.000001 : 0.000001);
        ndcZ = zFar / (zFar - zNear) -
            (zFar * zNear) / ((zFar - zNear) * safeViewZ);
    }
    ndcZ = params.clipOrtho.w > 0.5 ? 1.0 - ndcZ : ndcZ;
    float aspect = max(params.viewport.x, 0.001);

    if (params.upProjection.w > 0.5) {
        float halfHeight = max(params.clipOrtho.z * 0.5, 0.0001);
        return float4(
            viewPosition.x / (halfHeight * aspect),
            viewPosition.y / halfHeight,
            ndcZ,
            1.0);
    }

    float fovRadians = clamp(params.viewport.w, 0.017453292, 3.12413936);
    float yScale = 1.0 / tan(fovRadians * 0.5);
    float w = viewPosition.z;
    return float4(
        viewPosition.x * yScale / aspect,
        viewPosition.y * yScale,
        ndcZ * w,
        w);
}

struct GPUDrivenStreamAssetCameraData
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
};

GPUDrivenStreamAssetCameraData currentStreamCamera(GPUDrivenStreamAssetParams params)
{
    GPUDrivenStreamAssetCameraData camera;
    camera.eye = params.eye;
    camera.center = params.center;
    camera.upProjection = params.upProjection;
    camera.viewport = params.viewport;
    camera.clipOrtho = params.clipOrtho;
    return camera;
}

GPUDrivenStreamAssetCameraData previousStreamCamera(GPUDrivenStreamAssetParams params)
{
    GPUDrivenStreamAssetCameraData camera;
    camera.eye = params.previousEye;
    camera.center = params.previousCenter;
    camera.upProjection = params.previousUpProjection;
    camera.viewport = params.previousViewport;
    camera.clipOrtho = params.previousClipOrtho;
    return camera;
}

float3 streamWorldToView(float3 worldPosition, GPUDrivenStreamAssetCameraData camera)
{
    float3 forward = safeNormalize(camera.center.xyz - camera.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(camera.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float3 cameraVector = worldPosition - camera.eye.xyz;
    return float3(
        dot(cameraVector, right),
        dot(cameraVector, up),
        dot(cameraVector, forward));
}

float streamProjectDepth(float viewZ, GPUDrivenStreamAssetCameraData camera)
{
    float zNear = max(camera.clipOrtho.x, 0.0001);
    float zFar = max(camera.clipOrtho.y, zNear + 0.0001);
    float depth = (viewZ - zNear) / (zFar - zNear);
    if (camera.upProjection.w <= 0.5) {
        float safeViewZ = abs(viewZ) > 0.000001
            ? viewZ
            : (viewZ < 0.0 ? -0.000001 : 0.000001);
        depth = zFar / (zFar - zNear) -
            (zFar * zNear) / ((zFar - zNear) * safeViewZ);
    }
    return camera.clipOrtho.w > 0.5 ? 1.0 - depth : depth;
}

bool streamSphereInsideFrustum(
    float3 center,
    float radius,
    GPUDrivenStreamAssetCameraData camera)
{
    float3 viewCenter = streamWorldToView(center, camera);
    float zNear = max(camera.clipOrtho.x, 0.0001);
    float zFar = max(camera.clipOrtho.y, zNear + 0.0001);
    if (viewCenter.z + radius < zNear || viewCenter.z - radius > zFar) {
        return false;
    }
    float aspect = max(camera.viewport.x, 0.001);
    if (camera.upProjection.w > 0.5) {
        float halfHeight = max(camera.clipOrtho.z * 0.5, 0.0001);
        float halfWidth = halfHeight * aspect;
        return abs(viewCenter.x) <= halfWidth + radius &&
            abs(viewCenter.y) <= halfHeight + radius;
    }
    if (viewCenter.z + radius <= 0.0) {
        return false;
    }
    float tanY = tan(clamp(camera.viewport.w, 0.017453292, 3.12413936) * 0.5);
    float tanX = tanY * aspect;
    float radiusX = radius * sqrt(1.0 + tanX * tanX);
    float radiusY = radius * sqrt(1.0 + tanY * tanY);
    return abs(viewCenter.x) <= viewCenter.z * tanX + radiusX &&
        abs(viewCenter.y) <= viewCenter.z * tanY + radiusY;
}

uint streamMipDimension(uint baseDimension, uint mipLevel)
{
    uint dimension = max(baseDimension, 1u);
    for (uint level = 0u; level < mipLevel; ++level) {
        dimension = max(1u, (dimension + 1u) / 2u);
    }
    return dimension;
}

uint streamHzbMipOffset(
    GPUDrivenStreamAssetRasterBindings bindings,
    uint mipLevel)
{
    uint offset = 0u;
    uint width = max(bindings.width, 1u);
    uint height = max(bindings.height, 1u);
    for (uint level = 0u; level < mipLevel; ++level) {
        offset += width * height;
        width = max(1u, (width + 1u) / 2u);
        height = max(1u, (height + 1u) / 2u);
    }
    return offset;
}

float loadStreamHzb(
    GPUDrivenStreamAssetPush push,
    GPUDrivenStreamAssetRasterBindings bindings,
    uint hzbIndex,
    uint elementIndex)
{
    uint handleIndex = hzbIndex == 0u ? bindings.hzbBuffer0 : bindings.hzbBuffer1;
    DescriptorHandle<StructuredBuffer<float>> hzbHandle =
        DescriptorHandle<StructuredBuffer<float>>(uint2(
            push.bufferShaderIndexBase + handleIndex,
            0));
    StructuredBuffer<float> hzb = hzbHandle;
    return hzb[elementIndex];
}

bool streamSphereOccluded(
    float3 center,
    float radius,
    GPUDrivenStreamAssetCameraData camera,
    GPUDrivenStreamAssetRasterBindings bindings,
    GPUDrivenStreamAssetPush push,
    uint hzbIndex)
{
    if (bindings.hzbMipCount == 0u) {
        return false;
    }
    float3 viewCenter = streamWorldToView(center, camera);
    float zNear = max(camera.clipOrtho.x, 0.0001);
    if (viewCenter.z - radius <= zNear) {
        return false;
    }
    float2 centerPixels;
    float2 radiusPixels;
    if (camera.upProjection.w > 0.5) {
        float halfHeight = max(camera.clipOrtho.z * 0.5, 0.0001);
        float halfWidth = halfHeight * max(camera.viewport.x, 0.001);
        centerPixels = float2(
            (viewCenter.x / halfWidth * 0.5 + 0.5) * float(bindings.width),
            (-viewCenter.y / halfHeight * 0.5 + 0.5) * float(bindings.height));
        radiusPixels = float2(
            radius / halfWidth * 0.5 * float(bindings.width),
            radius / halfHeight * 0.5 * float(bindings.height));
    } else {
        float tanY = tan(clamp(camera.viewport.w, 0.017453292, 3.12413936) * 0.5);
        float tanX = tanY * max(camera.viewport.x, 0.001);
        float closestZ = max(viewCenter.z - radius, zNear);
        centerPixels = float2(
            (viewCenter.x / (viewCenter.z * tanX) * 0.5 + 0.5) * float(bindings.width),
            (-viewCenter.y / (viewCenter.z * tanY) * 0.5 + 0.5) * float(bindings.height));
        radiusPixels = float2(
            radius / (closestZ * tanX) * 0.5 * float(bindings.width),
            radius / (closestZ * tanY) * 0.5 * float(bindings.height));
    }
    float2 limit = float2(max(bindings.width, 1u) - 1u, max(bindings.height, 1u) - 1u);
    float2 minPixels = clamp(centerPixels - radiusPixels, 0.0, limit);
    float2 maxPixels = clamp(centerPixels + radiusPixels, 0.0, limit);
    float maxExtent = max(maxPixels.x - minPixels.x, maxPixels.y - minPixels.y);
    uint mipLevel = min(
        bindings.hzbMipCount - 1u,
        maxExtent > 1.0 ? uint(ceil(log2(maxExtent))) : 0u);
    uint mipWidth = streamMipDimension(bindings.width, mipLevel);
    uint mipHeight = streamMipDimension(bindings.height, mipLevel);
    float mipScale = exp2(-float(mipLevel));
    uint2 minTexel = min(uint2(minPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    uint2 maxTexel = min(uint2(maxPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    uint mipOffset = streamHzbMipOffset(bindings, mipLevel);
    float depth00 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + minTexel.y * mipWidth + minTexel.x);
    float depth10 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + minTexel.y * mipWidth + maxTexel.x);
    float depth01 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + maxTexel.y * mipWidth + minTexel.x);
    float depth11 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + maxTexel.y * mipWidth + maxTexel.x);
    float nearestZ = max(viewCenter.z - radius, zNear);
    float depthBias = max(0.001, radius * 0.01);
    float nearestDepth = streamProjectDepth(max(nearestZ - depthBias, zNear), camera);
    if (camera.clipOrtho.w > 0.5) {
        float occluderDepth = min(min(depth00, depth10), min(depth01, depth11));
        return nearestDepth < occluderDepth;
    }
    float occluderDepth = max(max(depth00, depth10), max(depth01, depth11));
    return nearestDepth > occluderDepth;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void gpuDrivenStreamAssetCullResetMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    // Late culling appends the instances recovered by the current-frame HZB to
    // the early-pass list, matching the common GPUScene two-phase contract.
    if (min(push.traversalPhase, 1u) != 0u) {
        return;
    }
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> bindingsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(
            push.bufferShaderIndexBase + push.rasterBindingsBuffer,
            0));
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingsBuffer = bindingsHandle;
    GPUDrivenStreamAssetRasterBindings bindings = bindingsBuffer[0];
    DescriptorHandle<RWStructuredBuffer<uint>> counterHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + bindings.visibleInstanceCounterBuffer,
            0));
    RWStructuredBuffer<uint> counter = counterHandle;
    counter[0] = 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetInstanceCullMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(
            push.bufferShaderIndexBase + push.paramsBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> bindingsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(
            push.bufferShaderIndexBase + push.rasterBindingsBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>> instanceHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(uint2(
            push.bufferShaderIndexBase + push.instanceBuffer,
            0));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingsBuffer = bindingsHandle;
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    GPUDrivenStreamAssetRasterBindings bindings = bindingsBuffer[0];
    uint rasterPhase = min(push.traversalPhase, 1u);
    uint instanceIndex = dispatchThreadID.x;
    if (instanceIndex >= params.sceneInstanceCount) {
        return;
    }

    DescriptorHandle<RWStructuredBuffer<uint>> visibilityHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + bindings.instanceVisibilityBuffer,
            0));
    RWStructuredBuffer<uint> visibility = visibilityHandle;
    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    uint gpuSceneInstanceIndex = instance.gpuSceneInstanceIndex;
    if (gpuSceneInstanceIndex == kInvalidClusterIndex) {
        return;
    }
    if (rasterPhase != 0u && visibility[gpuSceneInstanceIndex] != 2u) {
        return;
    }
    if (instance.visible == 0u) {
        visibility[gpuSceneInstanceIndex] = 0u;
        return;
    }

    float3 center = instance.boundsCenterRadius.xyz;
    float radius = max(instance.boundsCenterRadius.w, 0.000001);
    GPUDrivenStreamAssetCameraData currentCamera = currentStreamCamera(params);
    if ((bindings.cullingFlags & (1u << 0u)) != 0u &&
        !streamSphereInsideFrustum(center, radius, currentCamera)) {
        visibility[gpuSceneInstanceIndex] = 0u;
        return;
    }

    bool occluded = false;
    if ((bindings.cullingFlags & (1u << 1u)) != 0u &&
        (rasterPhase != 0u || bindings.hzbValid != 0u)) {
        uint currentHzb = params.frameIndex & 1u;
        uint hzbIndex = rasterPhase == 0u ? (currentHzb ^ 1u) : currentHzb;
        GPUDrivenStreamAssetCameraData hzbCamera = rasterPhase == 0u
            ? previousStreamCamera(params)
            : currentCamera;
        occluded = streamSphereOccluded(
            center,
            radius,
            hzbCamera,
            bindings,
            push,
            hzbIndex);
    }
    uint visibleState = rasterPhase == 0u ? 1u : 3u;
    visibility[gpuSceneInstanceIndex] = occluded
        ? (rasterPhase == 0u ? 2u : 0u)
        : visibleState;
    if (!occluded) {
        DescriptorHandle<RWStructuredBuffer<uint>> visibleIdsHandle =
            DescriptorHandle<RWStructuredBuffer<uint>>(uint2(
                push.bufferShaderIndexBase + bindings.visibleInstanceIdsBuffer,
                0));
        DescriptorHandle<RWStructuredBuffer<uint>> counterHandle =
            DescriptorHandle<RWStructuredBuffer<uint>>(uint2(
                push.bufferShaderIndexBase + bindings.visibleInstanceCounterBuffer,
                0));
        RWStructuredBuffer<uint> visibleIds = visibleIdsHandle;
        RWStructuredBuffer<uint> counter = counterHandle;
        uint outputIndex = 0u;
        InterlockedAdd(counter[0], 1u, outputIndex);
        if (outputIndex < params.sceneInstanceCount) {
            visibleIds[outputIndex] = gpuSceneInstanceIndex;
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gpuDrivenStreamAssetHzbMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(
            push.bufferShaderIndexBase + push.paramsBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> bindingsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(
            push.bufferShaderIndexBase + push.rasterBindingsBuffer,
            0));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> bindingsBuffer = bindingsHandle;
    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    GPUDrivenStreamAssetRasterBindings bindings = bindingsBuffer[0];
    uint mipLevel = push.activeBuildPhase;
    if (mipLevel >= bindings.hzbMipCount) {
        return;
    }
    uint mipWidth = streamMipDimension(bindings.width, mipLevel);
    uint mipHeight = streamMipDimension(bindings.height, mipLevel);
    if (dispatchThreadID.x >= mipWidth || dispatchThreadID.y >= mipHeight) {
        return;
    }
    uint currentHzb = params.frameIndex & 1u;
    uint hzbHandleIndex = currentHzb == 0u ? bindings.hzbBuffer0 : bindings.hzbBuffer1;
    DescriptorHandle<RWStructuredBuffer<float>> hzbHandle =
        DescriptorHandle<RWStructuredBuffer<float>>(uint2(
            push.bufferShaderIndexBase + hzbHandleIndex,
            0));
    RWStructuredBuffer<float> hzb = hzbHandle;
    uint destinationIndex = streamHzbMipOffset(bindings, mipLevel) +
        dispatchThreadID.y * mipWidth + dispatchThreadID.x;
    if (mipLevel == 0u) {
        DescriptorHandle<Texture2D<float>> depthHandle =
            DescriptorHandle<Texture2D<float>>(uint2(
                push.imageShaderIndexBase + bindings.depthImage,
                0));
        Texture2D<float> depth = depthHandle;
        hzb[destinationIndex] = depth.Load(int3(dispatchThreadID.xy, 0));
        return;
    }
    uint sourceMip = mipLevel - 1u;
    uint sourceWidth = streamMipDimension(bindings.width, sourceMip);
    uint sourceHeight = streamMipDimension(bindings.height, sourceMip);
    uint sourceOffset = streamHzbMipOffset(bindings, sourceMip);
    uint2 sourceBase = dispatchThreadID.xy * 2u;
    uint2 limit = uint2(sourceWidth - 1u, sourceHeight - 1u);
    uint2 source0 = min(sourceBase, limit);
    uint2 source1 = min(sourceBase + uint2(1u, 0u), limit);
    uint2 source2 = min(sourceBase + uint2(0u, 1u), limit);
    uint2 source3 = min(sourceBase + uint2(1u, 1u), limit);
    float depth0 = hzb[sourceOffset + source0.y * sourceWidth + source0.x];
    float depth1 = hzb[sourceOffset + source1.y * sourceWidth + source1.x];
    float depth2 = hzb[sourceOffset + source2.y * sourceWidth + source2.x];
    float depth3 = hzb[sourceOffset + source3.y * sourceWidth + source3.x];
    hzb[destinationIndex] = params.clipOrtho.w > 0.5
        ? min(min(depth0, depth1), min(depth2, depth3))
        : max(max(depth0, depth1), max(depth2, depth3));
}

float3 hashColor(uint seed)
{
    float3 value = frac(sin(float3(seed, seed * 17u + 13u, seed * 31u + 7u) * 0.017453292) * 43758.5453);
    return lerp(float3(0.16, 0.39, 0.70), float3(0.92, 0.78, 0.26), value);
}

uint readPageByte(StructuredBuffer<uint> pageWords, uint absoluteByteOffset)
{
    uint word = pageWords[absoluteByteOffset / 4u];
    uint shift = (absoluteByteOffset & 3u) * 8u;
    return (word >> shift) & 0xffu;
}

void requestStreamPageLoad(
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params,
    uint pageIndex)
{
    uint oldRequestFrame = 0;
    InterlockedMax(pageTable[pageIndex].lastRequestFrame, params.frameIndex, oldRequestFrame);
    if (oldRequestFrame == params.frameIndex) {
        return;
    }

    uint requestOffset = 0;
    uint maxLoads = requests[kStreamRequestMaxLoadsWord];
    InterlockedAdd(requests[kStreamRequestLoadCounterWord], 1u, requestOffset);
    if (requestOffset < maxLoads) {
        requests[kStreamRequestHeaderWords + requestOffset] = pageIndex;
    } else {
        InterlockedAdd(requests[kStreamRequestLoadOverflowCounterWord], 1u);
        InterlockedMax(requests[kStreamRequestLastLoadOverflowFrameWord], params.frameIndex);
    }
}

void markStreamPageDemand(
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    uint pageIndex)
{
    uint oldRequestFrame = 0;
    InterlockedMax(pageTable[pageIndex].lastRequestFrame, params.frameIndex, oldRequestFrame);
}

void requestStreamPageUnload(
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params,
    uint pageIndex)
{
    uint requestOffset = 0;
    uint maxLoads = requests[kStreamRequestMaxLoadsWord];
    uint maxUnloads = requests[kStreamRequestMaxUnloadsWord];
    InterlockedAdd(requests[kStreamRequestUnloadCounterWord], 1u, requestOffset);
    if (requestOffset < maxUnloads) {
        requests[kStreamRequestHeaderWords + maxLoads + requestOffset] = pageIndex;
    } else {
        InterlockedAdd(requests[kStreamRequestUnloadOverflowCounterWord], 1u);
        InterlockedMax(requests[kStreamRequestLastUnloadOverflowFrameWord], params.frameIndex);
    }
}

bool streamPageAvailable(uint state)
{
    return state == kStreamStatePendingUpload ||
        state == kStreamStateResident ||
        state == kStreamStateLockedFallback;
}

bool streamPageDrawable(uint state)
{
    return state == kStreamStateResident ||
        state == kStreamStateLockedFallback;
}

uint selectTraversalLod(
    GPUDrivenStreamAssetParams params,
    StructuredBuffer<GPUDrivenStreamAssetLodLevel> lodLevels,
    GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetPrimitive primitive)
{
    if (primitive.lodLevelCount == 0u) {
        return 0u;
    }
    if (params.enableGpuLodSelection == 0u && params.selectedLodLevel != kNoDebugLodOverride) {
        return min(params.selectedLodLevel, primitive.lodLevelCount - 1u);
    }

    float3 center = instance.boundsCenterRadius.xyz;
    float radius = max(instance.boundsCenterRadius.w, 0.001);
    float distanceToCamera = max(length(center - params.eye.xyz) - radius, 0.001);
    float fovRadians = clamp(params.viewport.w, 0.017453292, 3.12413936);
    float pixelScale = max(params.viewport.z, 1.0) / max(2.0 * tan(fovRadians * 0.5), 0.001);
    const float targetPixelError = 1.5;

    for (uint reverse = 0u; reverse < primitive.lodLevelCount; ++reverse) {
        uint localLod = primitive.lodLevelCount - 1u - reverse;
        uint lodIndex = primitive.lodLevelOffset + localLod;
        if (lodIndex >= params.sceneLodLevelCount) {
            continue;
        }
        GPUDrivenStreamAssetLodLevel lod = lodLevels[lodIndex];
        float lodError = max(lod.minMaxQuadricError, radius * 0.0005);
        float pixelError = lodError * pixelScale / distanceToCamera;
        if (pixelError <= targetPixelError || localLod == 0u) {
            return localLod;
        }
    }
    return 0u;
}

float instanceUniformScale(GPUDrivenStreamAssetInstance instance)
{
    return max(
        max(length(instance.world0.xyz), length(instance.world1.xyz)),
        max(length(instance.world2.xyz), 0.000001));
}

float groupPixelError(
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetGroup group)
{
    float3 localCenter = float3(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ);
    float3 worldCenter =
        instance.world0.xyz * localCenter.x +
        instance.world1.xyz * localCenter.y +
        instance.world2.xyz * localCenter.z +
        instance.world3.xyz;
    float uniformScale = instanceUniformScale(instance);
    float worldRadius = max(group.boundsRadius * uniformScale, 0.000001);
    float distanceToCamera = max(length(worldCenter - params.eye.xyz) - worldRadius, 0.001);
    float fovRadians = clamp(params.viewport.w, 0.017453292, 3.12413936);
    float pixelScale = max(params.viewport.z, 1.0) / max(2.0 * tan(fovRadians * 0.5), 0.001);
    return group.maxQuadricError * uniformScale * pixelScale / distanceToCamera;
}

bool groupNeedsTraversal(
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetPrimitive primitive,
    GPUDrivenStreamAssetGroup group)
{
    if (params.enableGpuLodSelection == 0u && params.selectedLodLevel != kNoDebugLodOverride) {
        uint selectedLod = primitive.lodLevelCount == 0u
            ? 0u
            : min(params.selectedLodLevel, primitive.lodLevelCount - 1u);
        return group.lodLevel == selectedLod;
    }
    const float targetPixelError = 1.5;
    return groupPixelError(params, instance, group) > targetPixelError;
}

float nodePixelError(
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetNode node)
{
    float3 localCenter = node.boundsCenterRadius.xyz;
    float3 worldCenter =
        instance.world0.xyz * localCenter.x +
        instance.world1.xyz * localCenter.y +
        instance.world2.xyz * localCenter.z +
        instance.world3.xyz;
    float uniformScale = instanceUniformScale(instance);
    float worldRadius = max(node.boundsCenterRadius.w * uniformScale, 0.000001);
    float distanceToCamera = max(length(worldCenter - params.eye.xyz) - worldRadius, 0.001);
    float fovRadians = clamp(params.viewport.w, 0.017453292, 3.12413936);
    float pixelScale = max(params.viewport.z, 1.0) / max(2.0 * tan(fovRadians * 0.5), 0.001);
    return node.maxQuadricError * uniformScale * pixelScale / distanceToCamera;
}

bool nodeNeedsTraversal(
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetPrimitive primitive,
    GPUDrivenStreamAssetNode node)
{
    if (node.lodLevel == kInvalidClusterIndex) {
        return true;
    }
    if (params.enableGpuLodSelection == 0u && params.selectedLodLevel != kNoDebugLodOverride) {
        uint selectedLod = primitive.lodLevelCount == 0u
            ? 0u
            : min(params.selectedLodLevel, primitive.lodLevelCount - 1u);
        return node.lodLevel == selectedLod;
    }
    const float targetPixelError = 1.5;
    return nodePixelError(params, instance, node) > targetPixelError;
}

bool updateGroupPageDemand(
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetPrimitive primitive,
    GPUDrivenStreamAssetGroup group)
{
    if (group.pageIndex >= params.scenePageCount ||
        !groupNeedsTraversal(params, instance, primitive, group)) {
        return false;
    }
    uint state = streamPageState(pageTable[group.pageIndex]);
    if (streamPageAvailable(state)) {
        markStreamPageDemand(pageTable, params, group.pageIndex);
    } else {
        requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
    }
    return true;
}

void reportInvalidTraversal(
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params)
{
    InterlockedAdd(requests[kStreamRequestInvalidPageCounterWord], 1u);
    InterlockedMax(requests[kStreamRequestLastInvalidPageFrameWord], params.frameIndex);
}

bool groupPageDrawable(
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetGroup group)
{
    if (group.pageIndex >= params.scenePageCount) {
        return false;
    }
    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[group.pageIndex];
    return streamPageDrawable(streamPageState(pageEntry)) &&
        streamPageDeviceOffset(pageEntry) != kInvalidStreamDeviceOffset;
}

uint buildGroupClusterSelectionMask(
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> pageWords,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetPrimitive primitive,
    GPUDrivenStreamAssetGroup group)
{
    if (!groupNeedsTraversal(params, instance, primitive, group)) {
        return 0u;
    }

    uint clusterCount = min(group.clusterCount, 32u);
    if (params.enableGpuLodSelection == 0u && params.selectedLodLevel != kNoDebugLodOverride) {
        return clusterCount == 32u ? 0xffffffffu : ((1u << clusterCount) - 1u);
    }

    GPUDrivenStreamAssetPageTableEntry currentPage = pageTable[group.pageIndex];
    uint pageWord = 0u;
    uint payloadClusterCount = 0u;
    uint clusterOffsetBytes = 0u;
    uint payloadBytes = 0u;
    bool clusterDirectoryAvailable = false;
    uint currentPageOffset = streamPageDeviceOffset(currentPage);
    if (currentPageOffset != kInvalidStreamDeviceOffset &&
        params.pageBufferBytes >= kPayloadHeaderBytes &&
        currentPageOffset <= params.pageBufferBytes - kPayloadHeaderBytes) {
        pageWord = currentPageOffset / 4u;
        payloadClusterCount = pageWords[pageWord + 2u];
        clusterOffsetBytes = pageWords[pageWord + 9u];
        payloadBytes = pageWords[pageWord + 12u];
        clusterDirectoryAvailable = payloadBytes >= kPayloadHeaderBytes &&
            payloadBytes <= params.pageBufferBytes &&
            currentPageOffset <= params.pageBufferBytes - payloadBytes &&
            (clusterOffsetBytes & 3u) == 0u &&
            streamPayloadArrayRangeValid(
                clusterOffsetBytes,
                payloadClusterCount,
                kPayloadClusterBytes,
                payloadBytes);
    }

    uint mask = 0u;
    for (uint clusterChild = 0u; clusterChild < 32u; ++clusterChild) {
        if (clusterChild >= clusterCount) {
            break;
        }
        uint refinedGroupIndex = kInvalidClusterIndex;
        if (clusterDirectoryAvailable) {
            uint refinedGroupByteOffset =
                clusterOffsetBytes + clusterChild * kPayloadClusterBytes + 32u;
            if (clusterChild < payloadClusterCount &&
                refinedGroupByteOffset >= clusterOffsetBytes &&
                refinedGroupByteOffset + 4u >= refinedGroupByteOffset &&
                refinedGroupByteOffset + 4u <= payloadBytes) {
                refinedGroupIndex = pageWords[pageWord + refinedGroupByteOffset / 4u];
            }
        }
        bool renderCluster = refinedGroupIndex == kInvalidClusterIndex;
        if (!renderCluster) {
            if (refinedGroupIndex >= params.sceneGroupCount) {
                renderCluster = true;
            } else {
                GPUDrivenStreamAssetGroup refinedGroup = groups[refinedGroupIndex];
                renderCluster = refinedGroup.primitiveIndex != group.primitiveIndex ||
                    !groupPageDrawable(pageTable, params, refinedGroup) ||
                    !groupNeedsTraversal(params, instance, primitive, refinedGroup);
            }
        }
        if (renderCluster) {
            mask |= 1u << clusterChild;
        }
    }
    return mask;
}

void appendActivePage(
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    uint instanceIndex,
    GPUDrivenStreamAssetGroup sourceGroup,
    uint clusterSelectionMask)
{
    uint pageIndex = sourceGroup.pageIndex;
    if (pageIndex >= params.scenePageCount) {
        return;
    }

    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[pageIndex];
    uint pageDeviceOffset = streamPageDeviceOffset(pageEntry);
    if (!streamPageDrawable(streamPageState(pageEntry)) ||
        pageDeviceOffset == kInvalidStreamDeviceOffset) {
        return;
    }

    uint activeIndex = 0u;
    InterlockedAdd(activeHeader[0].activeGroupCount, 1u, activeIndex);
    if (activeIndex >= activeHeader[0].activeGroupCapacity) {
        InterlockedAdd(activeHeader[0].overflowCount, 1u);
        return;
    }

    GPUDrivenStreamAssetActiveGroup group;
    group.pageDeviceOffsetBytes = pageDeviceOffset;
    group.pageIndex = pageIndex;
    group.clusterCount = sourceGroup.clusterCount;
    group.primitiveIndex = sourceGroup.primitiveIndex;
    group.lodLevel = sourceGroup.lodLevel;
    group.materialIndex = instance.materialIndex;
    group.clusterSelectionMask = clusterSelectionMask;
    group.flags = kActiveGroupResident;
    group.instanceIndex = instanceIndex;
    group.gpuSceneInstanceIndex = instance.gpuSceneInstanceIndex;
    group.padding1 = 0u;
    group.padding2 = 0u;
    group.world0 = instance.world0;
    group.world1 = instance.world1;
    group.world2 = instance.world2;
    group.world3 = instance.world3;
    activeGroups[activeIndex] = group;
}

bool pageRangeDrawable(
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    uint pageOffset,
    uint pageCount)
{
    for (uint page = 0u; page < pageCount; ++page) {
        uint pageIndex = pageOffset + page;
        if (pageIndex >= params.scenePageCount ||
            !streamPageDrawable(streamPageState(pageTable[pageIndex]))) {
            return false;
        }
    }
    return true;
}

void traverseInstancePageDemand(
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance)
{
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        reportInvalidTraversal(requests, params);
        return;
    }

    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    if (primitive.nodeCount == 0u ||
        primitive.nodeOffset >= params.sceneNodeCount ||
        primitive.nodeCount > params.sceneNodeCount - primitive.nodeOffset) {
        reportInvalidTraversal(requests, params);
        return;
    }

    uint stack[kTraversalStackCapacity];
    uint stackCount = 1u;
    stack[0] = primitive.nodeOffset;
    while (stackCount != 0u) {
        uint nodeIndex = stack[--stackCount];
        if (nodeIndex < primitive.nodeOffset ||
            nodeIndex >= primitive.nodeOffset + primitive.nodeCount ||
            nodeIndex >= params.sceneNodeCount) {
            reportInvalidTraversal(requests, params);
            return;
        }

        GPUDrivenStreamAssetNode node = nodes[nodeIndex];
        if (node.primitiveIndex != instance.primitiveIndex ||
            !nodeNeedsTraversal(params, instance, primitive, node)) {
            continue;
        }
        if (node.childCount != 0u) {
            if (node.childOffset < primitive.nodeOffset ||
                node.childOffset >= primitive.nodeOffset + primitive.nodeCount ||
                node.childCount > primitive.nodeOffset + primitive.nodeCount - node.childOffset ||
                node.childCount > kTraversalStackCapacity - stackCount) {
                reportInvalidTraversal(requests, params);
                return;
            }
            for (uint child = 0u; child < node.childCount; ++child) {
                stack[stackCount++] = node.childOffset + child;
            }
            continue;
        }

        if (node.groupIndex >= params.sceneGroupCount) {
            reportInvalidTraversal(requests, params);
            return;
        }
        GPUDrivenStreamAssetGroup group = groups[node.groupIndex];
        if (group.primitiveIndex != instance.primitiveIndex ||
            group.pageIndex >= params.scenePageCount ||
            !groupNeedsTraversal(params, instance, primitive, group)) {
            if (group.primitiveIndex != instance.primitiveIndex || group.pageIndex >= params.scenePageCount) {
                reportInvalidTraversal(requests, params);
                return;
            }
            continue;
        }

        uint state = streamPageState(pageTable[group.pageIndex]);
        if (streamPageAvailable(state)) {
            markStreamPageDemand(pageTable, params, group.pageIndex);
        } else {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
    }
}

void buildInstanceActiveGroups(
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> pageWords,
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    uint instanceIndex)
{
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        InterlockedAdd(activeHeader[0].overflowCount, 1u);
        return;
    }

    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    if (primitive.nodeCount == 0u ||
        primitive.nodeOffset >= params.sceneNodeCount ||
        primitive.nodeCount > params.sceneNodeCount - primitive.nodeOffset) {
        InterlockedAdd(activeHeader[0].overflowCount, 1u);
        return;
    }

    uint stack[kTraversalStackCapacity];
    uint stackCount = 1u;
    stack[0] = primitive.nodeOffset;
    while (stackCount != 0u) {
        uint nodeIndex = stack[--stackCount];
        if (nodeIndex < primitive.nodeOffset ||
            nodeIndex >= primitive.nodeOffset + primitive.nodeCount ||
            nodeIndex >= params.sceneNodeCount) {
            InterlockedAdd(activeHeader[0].overflowCount, 1u);
            return;
        }

        GPUDrivenStreamAssetNode node = nodes[nodeIndex];
        if (node.primitiveIndex != instance.primitiveIndex ||
            !nodeNeedsTraversal(params, instance, primitive, node)) {
            continue;
        }
        if (node.childCount != 0u) {
            if (node.childOffset < primitive.nodeOffset ||
                node.childOffset >= primitive.nodeOffset + primitive.nodeCount ||
                node.childCount > primitive.nodeOffset + primitive.nodeCount - node.childOffset ||
                node.childCount > kTraversalStackCapacity - stackCount) {
                InterlockedAdd(activeHeader[0].overflowCount, 1u);
                return;
            }
            for (uint child = 0u; child < node.childCount; ++child) {
                stack[stackCount++] = node.childOffset + child;
            }
            continue;
        }

        if (node.groupIndex >= params.sceneGroupCount) {
            InterlockedAdd(activeHeader[0].overflowCount, 1u);
            return;
        }
        GPUDrivenStreamAssetGroup group = groups[node.groupIndex];
        if (group.primitiveIndex != instance.primitiveIndex ||
            group.clusterCount == 0u ||
            group.clusterCount > 32u ||
            !groupPageDrawable(pageTable, params, group)) {
            continue;
        }
        uint clusterSelectionMask = buildGroupClusterSelectionMask(
            groups,
            pageWords,
            pageTable,
            params,
            instance,
            primitive,
            group);
        if (clusterSelectionMask != 0u) {
            appendActivePage(
                activeGroups,
                activeHeader,
                pageTable,
                params,
                instance,
                instanceIndex,
                group,
                clusterSelectionMask);
        }
    }
}

bool enqueueActiveTraversalWork(
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader> traversalHeader,
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem> traversalWork,
    GPUDrivenStreamAssetParams params,
    uint instanceIndex,
    uint nodeIndex)
{
    uint writeIndex = 0u;
    InterlockedAdd(traversalHeader[0].writeCounter, 1u, writeIndex);
    if (writeIndex >= params.traversalWorkCapacity) {
        InterlockedAdd(traversalHeader[0].overflowCount, 1u);
        return false;
    }

    InterlockedAdd(traversalHeader[0].taskCounter, 1u);
    traversalWork[writeIndex].instanceIndex = instanceIndex;
    traversalWork[writeIndex].nodeIndex = nodeIndex;
    traversalWork[writeIndex].padding0 = 0u;
    DeviceMemoryBarrier();
    uint oldReadyFrame = 0u;
    InterlockedExchange(traversalWork[writeIndex].readyFrame, params.frameIndex + 1u, oldReadyFrame);
    return true;
}

void appendInstanceFallbackGroups(
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> pageWords,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    uint instanceIndex,
    GPUDrivenStreamAssetPrimitive primitive)
{
    if (primitive.fallbackGroupOffset > params.sceneGroupCount ||
        primitive.fallbackGroupCount > params.sceneGroupCount - primitive.fallbackGroupOffset) {
        InterlockedAdd(activeHeader[0].overflowCount, 1u);
        return;
    }
    for (uint fallbackChild = 0u; fallbackChild < primitive.fallbackGroupCount; ++fallbackChild) {
        uint groupIndex = primitive.fallbackGroupOffset + fallbackChild;
        GPUDrivenStreamAssetGroup group = groups[groupIndex];
        if (group.primitiveIndex != instance.primitiveIndex ||
            group.clusterCount == 0u ||
            group.clusterCount > 32u ||
            group.pageIndex >= params.scenePageCount) {
            InterlockedAdd(activeHeader[0].overflowCount, 1u);
            reportInvalidTraversal(requests, params);
            continue;
        }
        uint state = streamPageState(pageTable[group.pageIndex]);
        if (streamPageAvailable(state)) {
            markStreamPageDemand(pageTable, params, group.pageIndex);
        } else {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
        if (!groupPageDrawable(pageTable, params, group)) {
            continue;
        }
        uint clusterSelectionMask = 0u;
        if (!groupNeedsTraversal(params, instance, primitive, group)) {
            uint clusterCount = min(group.clusterCount, 32u);
            clusterSelectionMask = clusterCount == 32u
                ? 0xffffffffu
                : ((1u << clusterCount) - 1u);
        } else {
            clusterSelectionMask = buildGroupClusterSelectionMask(
                groups,
                pageWords,
                pageTable,
                params,
                instance,
                primitive,
                group);
        }
        if (clusterSelectionMask != 0u) {
            appendActivePage(
                activeGroups,
                activeHeader,
                pageTable,
                params,
                instance,
                instanceIndex,
                group,
                clusterSelectionMask);
        }
    }
}

void runActiveTraversalWorker(
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader> traversalHeader,
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem> traversalWork,
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances,
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> pageWords,
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params)
{
    uint readIndex = 0xffffffffu;
    for (;;) {
        if (WaveActiveAllTrue(readIndex == 0xffffffffu)) {
            uint waveReadIndex = 0u;
            if (WaveIsFirstLane()) {
                InterlockedAdd(traversalHeader[0].readCounter, WaveGetLaneCount(), waveReadIndex);
            }
            waveReadIndex = WaveReadLaneFirst(waveReadIndex) + WaveGetLaneIndex();
            readIndex = waveReadIndex < params.traversalWorkCapacity
                ? waveReadIndex
                : 0xffffffffu;
            if (WaveActiveAllTrue(readIndex == 0xffffffffu)) {
                return;
            }
        }

        bool runnable = false;
        GPUDrivenStreamAssetTraversalWorkItem work;
        for (;;) {
            if (readIndex != 0xffffffffu) {
                uint readyFrame = 0u;
                InterlockedAdd(traversalWork[readIndex].readyFrame, 0u, readyFrame);
                runnable = readyFrame == params.frameIndex + 1u;
                if (runnable) {
                    DeviceMemoryBarrier();
                    work = traversalWork[readIndex];
                }
            }
            if (WaveActiveAnyTrue(runnable)) {
                break;
            }
            uint taskCount = 0u;
            InterlockedAdd(traversalHeader[0].taskCounter, 0u, taskCount);
            if (WaveActiveAnyTrue(taskCount == 0u)) {
                return;
            }
            DeviceMemoryBarrier();
        }

        bool validWork = runnable &&
            work.instanceIndex < params.sceneInstanceCount &&
            work.nodeIndex < params.sceneNodeCount;
        if (runnable && validWork) {
            GPUDrivenStreamAssetInstance instance = instances[work.instanceIndex];
            validWork = instance.visible != 0u && instance.primitiveIndex < params.scenePrimitiveCount;
            if (validWork) {
                GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
                validWork = primitive.nodeCount != 0u &&
                    primitive.nodeOffset < params.sceneNodeCount &&
                    primitive.nodeCount <= params.sceneNodeCount - primitive.nodeOffset &&
                    work.nodeIndex >= primitive.nodeOffset &&
                    work.nodeIndex < primitive.nodeOffset + primitive.nodeCount;
                if (validWork) {
                    GPUDrivenStreamAssetNode node = nodes[work.nodeIndex];
                    validWork = node.primitiveIndex == instance.primitiveIndex;
                    if (validWork && nodeNeedsTraversal(params, instance, primitive, node)) {
                        if (node.childCount != 0u) {
                            bool validChildren = node.childOffset >= primitive.nodeOffset &&
                                node.childOffset < primitive.nodeOffset + primitive.nodeCount &&
                                node.childCount <= primitive.nodeOffset + primitive.nodeCount - node.childOffset;
                            if (validChildren) {
                                for (uint child = node.childCount; child != 0u; --child) {
                                    enqueueActiveTraversalWork(
                                        traversalHeader,
                                        traversalWork,
                                        params,
                                        work.instanceIndex,
                                        node.childOffset + child - 1u);
                                }
                            } else {
                                validWork = false;
                            }
                        } else if (node.groupIndex < params.sceneGroupCount) {
                            bool fallbackGroup = node.groupIndex >= primitive.fallbackGroupOffset &&
                                node.groupIndex - primitive.fallbackGroupOffset < primitive.fallbackGroupCount;
                            GPUDrivenStreamAssetGroup group = groups[node.groupIndex];
                            bool validGroup = group.primitiveIndex == instance.primitiveIndex &&
                                group.clusterCount != 0u &&
                                group.clusterCount <= 32u &&
                                group.pageIndex < params.scenePageCount;
                            if (!validGroup) {
                                validWork = false;
                                reportInvalidTraversal(requests, params);
                            } else if (updateGroupPageDemand(
                                    pageTable,
                                    requests,
                                    params,
                                    instance,
                                    primitive,
                                    group) &&
                                !fallbackGroup &&
                                groupPageDrawable(pageTable, params, group)) {
                                uint clusterSelectionMask = buildGroupClusterSelectionMask(
                                    groups,
                                    pageWords,
                                    pageTable,
                                    params,
                                    instance,
                                    primitive,
                                    group);
                                if (clusterSelectionMask != 0u) {
                                    appendActivePage(
                                        activeGroups,
                                        activeHeader,
                                        pageTable,
                                        params,
                                        instance,
                                        work.instanceIndex,
                                        group,
                                        clusterSelectionMask);
                                }
                            }
                        } else {
                            validWork = false;
                        }
                    }
                }
            }
        }
        if (runnable) {
            if (!validWork) {
                InterlockedAdd(traversalHeader[0].overflowCount, 1u);
            }
            InterlockedAdd(traversalHeader[0].taskCounter, 0xffffffffu);
            readIndex = 0xffffffffu;
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetTraversalMain(uint3 dispatchThreadID : SV_DispatchThreadID, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>> instanceHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(uint2(push.bufferShaderIndexBase + push.instanceBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>> primitiveHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>(uint2(push.bufferShaderIndexBase + push.primitiveBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetLodLevel>> lodLevelHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetLodLevel>>(uint2(push.bufferShaderIndexBase + push.lodLevelBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>> groupHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>(uint2(push.bufferShaderIndexBase + push.groupBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetNode>> nodeHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetNode>>(uint2(push.bufferShaderIndexBase + push.nodeBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTableHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> requestHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.requestBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));

    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives = primitiveHandle;
    StructuredBuffer<GPUDrivenStreamAssetLodLevel> lodLevels = lodLevelHandle;
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups = groupHandle;
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes = nodeHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    RWStructuredBuffer<uint> requests = requestHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    if (push.traversalPhase == kTraversalPhaseUnload) {
        uint residentPageIndex = dispatchThreadID.x;
        if (residentPageIndex >= push.activeBuildPhase || params.enableGpuUnloadRequests == 0u) {
            return;
        }
        DescriptorHandle<StructuredBuffer<uint>> residentPageHandle =
            DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.residentPageBuffer, 0));
        StructuredBuffer<uint> residentPages = residentPageHandle;
        uint pageIndex = residentPages[residentPageIndex];
        if (pageIndex >= params.scenePageCount) {
            reportInvalidTraversal(requests, params);
            return;
        }
        GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[pageIndex];
        if (streamPageState(pageEntry) == kStreamStateResident &&
            pageEntry.lastRequestFrame != params.frameIndex) {
            requestStreamPageUnload(requests, params, pageIndex);
        }
        return;
    }

    if (params.sceneNodeCount != 0u) {
        uint workerCount = max(params.traversalWorkerCount, 1u);
        if (dispatchThreadID.x >= workerCount) {
            return;
        }
        for (uint instanceIndex = dispatchThreadID.x;
             instanceIndex < params.sceneInstanceCount;
             instanceIndex += workerCount) {
            GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
            if (instance.visible != 0u) {
                traverseInstancePageDemand(
                    primitives,
                    groups,
                    nodes,
                    pageTable,
                    requests,
                    params,
                    instance);
            }
        }
        return;
    }

    if (params.maxPrimitiveGroupCount == 0u) {
        return;
    }
    uint workItemIndex = dispatchThreadID.x;
    uint instanceIndex = workItemIndex / params.maxPrimitiveGroupCount;
    uint localGroupIndex = workItemIndex - instanceIndex * params.maxPrimitiveGroupCount;
    if (instanceIndex >= params.sceneInstanceCount) {
        return;
    }

    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    if (instance.visible == 0u) {
        return;
    }
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        InterlockedAdd(requests[kStreamRequestInvalidPageCounterWord], 1u);
        InterlockedMax(requests[kStreamRequestLastInvalidPageFrameWord], params.frameIndex);
        return;
    }

    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    if (primitive.groupCount == 0u ||
        localGroupIndex >= primitive.groupCount ||
        primitive.groupOffset >= params.sceneGroupCount ||
        primitive.groupCount > params.sceneGroupCount - primitive.groupOffset) {
        if (localGroupIndex >= primitive.groupCount) {
            return;
        }
        InterlockedAdd(requests[kStreamRequestInvalidPageCounterWord], 1u);
        InterlockedMax(requests[kStreamRequestLastInvalidPageFrameWord], params.frameIndex);
        return;
    }

    uint groupIndex = primitive.groupOffset + localGroupIndex;
    GPUDrivenStreamAssetGroup group = groups[groupIndex];
    if (group.primitiveIndex != instance.primitiveIndex ||
        group.pageIndex >= params.scenePageCount) {
        InterlockedAdd(requests[kStreamRequestInvalidPageCounterWord], 1u);
        InterlockedMax(requests[kStreamRequestLastInvalidPageFrameWord], params.frameIndex);
        return;
    }
    if (!groupNeedsTraversal(params, instance, primitive, group)) {
        return;
    }
    uint state = streamPageState(pageTable[group.pageIndex]);
    if (streamPageAvailable(state)) {
        markStreamPageDemand(pageTable, params, group.pageIndex);
    } else {
        requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetBuildActiveMain(uint3 dispatchThreadID : SV_DispatchThreadID, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<uint>> pageHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.pageBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup>> activeGroupHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(push.bufferShaderIndexBase + push.activeGroupBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader>> activeHeaderHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(uint2(push.bufferShaderIndexBase + push.activeHeaderBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>> instanceHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(uint2(push.bufferShaderIndexBase + push.instanceBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>> primitiveHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>(uint2(push.bufferShaderIndexBase + push.primitiveBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetLodLevel>> lodLevelHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetLodLevel>>(uint2(push.bufferShaderIndexBase + push.lodLevelBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>> groupHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>(uint2(push.bufferShaderIndexBase + push.groupBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetNode>> nodeHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetNode>>(uint2(push.bufferShaderIndexBase + push.nodeBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetDrawIndirect>> drawIndirectHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetDrawIndirect>>(uint2(push.bufferShaderIndexBase + push.drawIndirectBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader>> traversalHeaderHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader>>(uint2(push.bufferShaderIndexBase + push.traversalHeaderBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>> traversalWorkHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>>(uint2(push.bufferShaderIndexBase + push.traversalWorkBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTableHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> requestHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.requestBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));

    StructuredBuffer<uint> pageWords = pageHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader = activeHeaderHandle;
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives = primitiveHandle;
    StructuredBuffer<GPUDrivenStreamAssetLodLevel> lodLevels = lodLevelHandle;
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups = groupHandle;
    StructuredBuffer<GPUDrivenStreamAssetNode> nodes = nodeHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetDrawIndirect> drawIndirect = drawIndirectHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalHeader> traversalHeader = traversalHeaderHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem> traversalWork = traversalWorkHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    RWStructuredBuffer<uint> requests = requestHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    if (push.activeBuildPhase == kActiveBuildPhaseReset) {
        if (dispatchThreadID.x == 0u) {
            GPUDrivenStreamAssetActiveHeader header;
            header.activeGroupCount = 0u;
            header.activeGroupCapacity = params.activeGroupCount;
            header.maxActiveGroupClusters = params.maxActiveGroupClusters;
            header.overflowCount = 0u;
            header.frameIndex = params.frameIndex;
            header.padding0 = 0u;
            header.padding1 = 0u;
            header.padding2 = 0u;
            activeHeader[0] = header;
            if (params.sceneNodeCount != 0u) {
                GPUDrivenStreamAssetDrawIndirect draw;
                draw.groupCountX = 0u;
                draw.groupCountY = 1u;
                draw.groupCountZ = 1u;
                drawIndirect[0] = draw;

                GPUDrivenStreamAssetTraversalHeader traversal;
                traversal.readCounter = 0u;
                traversal.writeCounter = 0u;
                traversal.taskCounter = 0u;
                traversal.overflowCount = 0u;
                traversal.frameIndex = params.frameIndex;
                traversal.padding0 = 0u;
                traversal.padding1 = 0u;
                traversal.padding2 = 0u;
                traversalHeader[0] = traversal;
            }
        }
        return;
    }

    if (push.activeBuildPhase == kActiveBuildPhaseFinalize) {
        if (dispatchThreadID.x == 0u) {
            GPUDrivenStreamAssetActiveHeader header = activeHeader[0];
            if (params.sceneNodeCount != 0u) {
                header.overflowCount += traversalHeader[0].overflowCount;
                activeHeader[0] = header;
            }
            uint activeCount = min(header.activeGroupCount, header.activeGroupCapacity);
            GPUDrivenStreamAssetDrawIndirect draw;
            draw.groupCountX = min(
                activeCount * header.maxActiveGroupClusters * kMeshletTriangleChunkCount,
                params.drawTaskCount);
            draw.groupCountY = 1u;
            draw.groupCountZ = 1u;
            drawIndirect[0] = draw;
        }
        return;
    }

    if (push.activeBuildPhase == kActiveBuildPhaseSeed) {
        uint workerCount = max(params.traversalWorkerCount, 1u);
        if (dispatchThreadID.x >= workerCount) {
            return;
        }
        for (uint instanceIndex = dispatchThreadID.x;
             instanceIndex < params.sceneInstanceCount;
             instanceIndex += workerCount) {
            GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
            if (instance.visible == 0u) {
                continue;
            }
            if (instance.primitiveIndex >= params.scenePrimitiveCount) {
                InterlockedAdd(traversalHeader[0].overflowCount, 1u);
                continue;
            }
            GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
            appendInstanceFallbackGroups(
                activeGroups,
                activeHeader,
                groups,
                pageWords,
                pageTable,
                requests,
                params,
                instance,
                instanceIndex,
                primitive);
            if (primitive.nodeCount == 0u ||
                primitive.nodeOffset >= params.sceneNodeCount ||
                primitive.nodeCount > params.sceneNodeCount - primitive.nodeOffset) {
                InterlockedAdd(traversalHeader[0].overflowCount, 1u);
                continue;
            }
            enqueueActiveTraversalWork(
                traversalHeader,
                traversalWork,
                params,
                instanceIndex,
                primitive.nodeOffset);
        }
        return;
    }

    if (push.activeBuildPhase == kActiveBuildPhaseRun) {
        uint workerCount = max(params.traversalWorkerCount, 1u);
        if (dispatchThreadID.x < workerCount) {
            runActiveTraversalWorker(
                activeGroups,
                activeHeader,
                traversalHeader,
                traversalWork,
                instances,
                primitives,
                groups,
                pageWords,
                nodes,
                pageTable,
                requests,
                params);
        }
        return;
    }

    if (params.sceneNodeCount != 0u) {
        uint workerCount = max(params.traversalWorkerCount, 1u);
        if (dispatchThreadID.x >= workerCount) {
            return;
        }
        for (uint instanceIndex = dispatchThreadID.x;
             instanceIndex < params.sceneInstanceCount;
             instanceIndex += workerCount) {
            GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
            if (instance.visible != 0u) {
                buildInstanceActiveGroups(
                    activeGroups,
                    activeHeader,
                    primitives,
                    groups,
                    pageWords,
                    nodes,
                    pageTable,
                    params,
                    instance,
                    instanceIndex);
            }
        }
        return;
    }

    if (params.maxPrimitiveGroupCount == 0u) {
        return;
    }
    uint workItemIndex = dispatchThreadID.x;
    uint instanceIndex = workItemIndex / params.maxPrimitiveGroupCount;
    uint localGroupIndex = workItemIndex - instanceIndex * params.maxPrimitiveGroupCount;
    if (instanceIndex >= params.sceneInstanceCount) {
        return;
    }

    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    if (instance.visible == 0u || instance.primitiveIndex >= params.scenePrimitiveCount) {
        return;
    }

    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    if (primitive.groupCount == 0u ||
        localGroupIndex >= primitive.groupCount ||
        primitive.groupOffset >= params.sceneGroupCount ||
        primitive.groupCount > params.sceneGroupCount - primitive.groupOffset) {
        return;
    }

    uint groupIndex = primitive.groupOffset + localGroupIndex;
    GPUDrivenStreamAssetGroup group = groups[groupIndex];
    if (group.primitiveIndex != instance.primitiveIndex ||
        group.clusterCount == 0u ||
        group.clusterCount > 32u ||
        !groupPageDrawable(pageTable, params, group)) {
        return;
    }
    uint clusterSelectionMask = buildGroupClusterSelectionMask(
        groups,
        pageWords,
        pageTable,
        params,
        instance,
        primitive,
        group);
    if (clusterSelectionMask != 0u) {
        appendActivePage(
            activeGroups,
            activeHeader,
            pageTable,
            params,
            instance,
            instanceIndex,
            group,
            clusterSelectionMask);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetBuildBlasInputMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>> activeGroupHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(push.bufferShaderIndexBase + push.activeGroupBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>> activeHeaderHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(uint2(push.bufferShaderIndexBase + push.activeHeaderBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetClasPageEntry>> clasPageTableHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetClasPageEntry>>(uint2(push.bufferShaderIndexBase + push.clasPageTableBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint2>> clasAddressHandle =
        DescriptorHandle<StructuredBuffer<uint2>>(uint2(push.bufferShaderIndexBase + push.clasAddressBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetBlasHeader>> blasHeaderHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetBlasHeader>>(uint2(push.bufferShaderIndexBase + push.blasHeaderBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetInstanceBlas>> instanceBlasHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetInstanceBlas>>(uint2(push.bufferShaderIndexBase + push.instanceBlasBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetBlasBuildInfo>> blasBuildInfoHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetBlasBuildInfo>>(uint2(push.bufferShaderIndexBase + push.blasBuildInfoBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint2>> blasClusterReferenceHandle =
        DescriptorHandle<RWStructuredBuffer<uint2>>(uint2(push.bufferShaderIndexBase + push.blasClusterReferenceBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));

    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader = activeHeaderHandle;
    StructuredBuffer<GPUDrivenStreamAssetClasPageEntry> clasPageTable = clasPageTableHandle;
    StructuredBuffer<uint2> clasAddresses = clasAddressHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetBlasHeader> blasHeader = blasHeaderHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetInstanceBlas> instanceBlas = instanceBlasHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetBlasBuildInfo> blasBuildInfos = blasBuildInfoHandle;
    RWStructuredBuffer<uint2> blasClusterReferences = blasClusterReferenceHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    uint itemIndex = dispatchThreadID.x;

    if (push.activeBuildPhase == kBlasInputPhaseReset) {
        if (itemIndex == 0u) {
            GPUDrivenStreamAssetBlasHeader header;
            header.clusterReferenceCount = 0u;
            header.blasBuildCount = 0u;
            header.clusterReferenceCapacity = params.blasClusterReferenceCapacity;
            header.overflowCount = 0u;
            header.frameIndex = params.frameIndex;
            header.padding0 = 0u;
            header.padding1 = 0u;
            header.padding2 = 0u;
            blasHeader[0] = header;
        }
        if (itemIndex < params.sceneInstanceCount) {
            GPUDrivenStreamAssetInstanceBlas info = instanceBlas[itemIndex];
            info.selectedClusterCount = 0u;
            info.insertedClusterCount = 0u;
            info.blasBuildIndex = kInvalidClusterIndex;
            info.flags = 0u;
            info.padding0 = 0u;
            info.padding1 = 0u;
            instanceBlas[itemIndex] = info;
        }
        return;
    }

    uint activeCount = min(activeHeader[0].activeGroupCount, activeHeader[0].activeGroupCapacity);
    if (push.activeBuildPhase == kBlasInputPhaseCount) {
        if (itemIndex >= activeCount) {
            return;
        }
        GPUDrivenStreamAssetActiveGroup group = activeGroups[itemIndex];
        if (group.instanceIndex >= params.sceneInstanceCount ||
            group.pageIndex >= params.scenePageCount ||
            group.clusterCount == 0u ||
            group.clusterCount > 32u) {
            InterlockedAdd(blasHeader[0].overflowCount, 1u);
            return;
        }
        GPUDrivenStreamAssetClasPageEntry page = clasPageTable[group.pageIndex];
        if (clasPageState(page) != kClasPageStateActive) {
            InterlockedOr(instanceBlas[group.instanceIndex].flags, kBlasInstanceFallback);
            return;
        }
        uint validMask = group.clusterCount == 32u
            ? 0xffffffffu
            : ((1u << group.clusterCount) - 1u);
        uint selectedCount = countbits(group.clusterSelectionMask & validMask);
        if (selectedCount != 0u) {
            InterlockedAdd(instanceBlas[group.instanceIndex].selectedClusterCount, selectedCount);
        }
        return;
    }

    if (push.activeBuildPhase == kBlasInputPhaseSetup) {
        if (itemIndex >= params.sceneInstanceCount) {
            return;
        }
        GPUDrivenStreamAssetInstanceBlas info = instanceBlas[itemIndex];
        if (info.selectedClusterCount == 0u ||
            (info.flags & kBlasInstanceFallback) != 0u) {
            InterlockedOr(instanceBlas[itemIndex].flags, kBlasInstanceFallback);
            return;
        }
        if (info.clusterReferenceCapacity == 0u ||
            info.selectedClusterCount > info.clusterReferenceCapacity ||
            info.clusterReferenceOffset >= params.blasClusterReferenceCapacity ||
            info.selectedClusterCount > params.blasClusterReferenceCapacity - info.clusterReferenceOffset) {
            InterlockedOr(
                instanceBlas[itemIndex].flags,
                kBlasInstanceFallback | kBlasInstanceOverflow);
            InterlockedAdd(blasHeader[0].overflowCount, 1u);
            return;
        }

        uint buildIndex;
        InterlockedAdd(blasHeader[0].blasBuildCount, 1u, buildIndex);
        uint ignored;
        InterlockedAdd(blasHeader[0].clusterReferenceCount, info.selectedClusterCount, ignored);

        uint byteAddressLow = info.clusterReferenceOffset << 3u;
        uint byteAddressHigh = info.clusterReferenceOffset >> 29u;
        uint referenceAddressLow = params.blasClusterReferenceAddressLow + byteAddressLow;
        uint carry = referenceAddressLow < params.blasClusterReferenceAddressLow ? 1u : 0u;
        GPUDrivenStreamAssetBlasBuildInfo buildInfo;
        buildInfo.clusterReferencesCount = info.selectedClusterCount;
        buildInfo.clusterReferencesStride = 8u;
        buildInfo.clusterReferencesAddressLow = referenceAddressLow;
        buildInfo.clusterReferencesAddressHigh =
            params.blasClusterReferenceAddressHigh + byteAddressHigh + carry;
        blasBuildInfos[buildIndex] = buildInfo;
        instanceBlas[itemIndex].blasBuildIndex = buildIndex;
        InterlockedOr(instanceBlas[itemIndex].flags, kBlasInstanceDynamic);
        return;
    }

    if (push.activeBuildPhase != kBlasInputPhaseInsert || itemIndex >= activeCount) {
        return;
    }

    GPUDrivenStreamAssetActiveGroup group = activeGroups[itemIndex];
    if (group.instanceIndex >= params.sceneInstanceCount ||
        group.pageIndex >= params.scenePageCount ||
        group.clusterCount == 0u ||
        group.clusterCount > 32u) {
        return;
    }
    GPUDrivenStreamAssetInstanceBlas info = instanceBlas[group.instanceIndex];
    if ((info.flags & kBlasInstanceDynamic) == 0u ||
        (info.flags & kBlasInstanceFallback) != 0u) {
        return;
    }
    GPUDrivenStreamAssetClasPageEntry page = clasPageTable[group.pageIndex];
    if (clasPageState(page) != kClasPageStateActive) {
        InterlockedOr(instanceBlas[group.instanceIndex].flags, kBlasInstanceFallback);
        InterlockedAdd(blasHeader[0].overflowCount, 1u);
        return;
    }

    uint validMask = group.clusterCount == 32u
        ? 0xffffffffu
        : ((1u << group.clusterCount) - 1u);
    uint selectedMask = group.clusterSelectionMask & validMask;
    for (uint clusterIndex = 0u; clusterIndex < group.clusterCount; ++clusterIndex) {
        if ((selectedMask & (1u << clusterIndex)) == 0u) {
            continue;
        }
        uint insertIndex;
        InterlockedAdd(instanceBlas[group.instanceIndex].insertedClusterCount, 1u, insertIndex);
        if (insertIndex >= info.selectedClusterCount || insertIndex >= info.clusterReferenceCapacity) {
            InterlockedOr(instanceBlas[group.instanceIndex].flags, kBlasInstanceOverflow);
            InterlockedAdd(blasHeader[0].overflowCount, 1u);
            continue;
        }
        blasClusterReferences[info.clusterReferenceOffset + insertIndex] =
            clasAddresses[clasPageAddressOffset(page) + clusterIndex];
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetBuildTlasInputMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>> instanceHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(uint2(push.bufferShaderIndexBase + push.instanceBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstanceBlas>> instanceBlasHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstanceBlas>>(uint2(push.bufferShaderIndexBase + push.instanceBlasBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint2>> fallbackBlasAddressHandle =
        DescriptorHandle<StructuredBuffer<uint2>>(uint2(push.bufferShaderIndexBase + push.fallbackBlasAddressBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint2>> dynamicBlasAddressHandle =
        DescriptorHandle<StructuredBuffer<uint2>>(uint2(push.bufferShaderIndexBase + push.dynamicBlasAddressBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTlasInstance>> tlasInstanceHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTlasInstance>>(uint2(push.bufferShaderIndexBase + push.tlasInstanceBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));

    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    StructuredBuffer<GPUDrivenStreamAssetInstanceBlas> instanceBlas = instanceBlasHandle;
    StructuredBuffer<uint2> fallbackBlasAddresses = fallbackBlasAddressHandle;
    StructuredBuffer<uint2> dynamicBlasAddresses = dynamicBlasAddressHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetTlasInstance> tlasInstances = tlasInstanceHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    uint instanceIndex = dispatchThreadID.x;
    if (instanceIndex >= params.sceneInstanceCount) {
        return;
    }

    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    GPUDrivenStreamAssetInstanceBlas blas = instanceBlas[instanceIndex];
    uint2 accelerationStructureReference = uint2(0u, 0u);
    bool dynamicReady =
        (blas.flags & kBlasInstanceDynamic) != 0u &&
        (blas.flags & (kBlasInstanceFallback | kBlasInstanceOverflow)) == 0u &&
        blas.selectedClusterCount != 0u &&
        blas.insertedClusterCount == blas.selectedClusterCount &&
        blas.blasBuildIndex < params.blasBuildCapacity;
    if (dynamicReady) {
        accelerationStructureReference = dynamicBlasAddresses[blas.blasBuildIndex];
    } else if (instance.primitiveIndex < params.scenePrimitiveCount) {
        accelerationStructureReference = fallbackBlasAddresses[instance.primitiveIndex];
    }

    bool addressValid =
        accelerationStructureReference.x != 0u ||
        accelerationStructureReference.y != 0u;
    uint visibilityMask = instance.visible != 0u && addressValid ? 0xffu : 0u;

    GPUDrivenStreamAssetTlasInstance output;
    output.transform0 = float4(instance.world0.x, instance.world1.x, instance.world2.x, instance.world3.x);
    output.transform1 = float4(instance.world0.y, instance.world1.y, instance.world2.y, instance.world3.y);
    output.transform2 = float4(instance.world0.z, instance.world1.z, instance.world2.z, instance.world3.z);
    output.customIndexMask = (instanceIndex & 0x00ffffffu) | (visibilityMask << 24u);
    output.shaderBindingTableFlags = 0u;
    output.accelerationStructureReferenceLow = accelerationStructureReference.x;
    output.accelerationStructureReferenceHigh = accelerationStructureReference.y;
    tlasInstances[instanceIndex] = output;
}

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void gpuDrivenStreamAssetMeshMain(
    uint3 groupID : SV_GroupID,
    out vertices GPUDrivenStreamAssetVertexOutput vertices[kMaxMeshOutputVertices],
    out indices uint3 triangles[kMeshletTriangleChunkSize],
    uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<uint>> pageHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.pageBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>> activeGroupHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(push.bufferShaderIndexBase + push.activeGroupBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>> activeHeaderHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(uint2(push.bufferShaderIndexBase + push.activeHeaderBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTableHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> requestHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.requestBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> rasterBindingsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(
            push.bufferShaderIndexBase + push.rasterBindingsBuffer,
            0));

    StructuredBuffer<uint> pageWords = pageHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader = activeHeaderHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    RWStructuredBuffer<uint> requests = requestHandle;
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rasterBindingsBuffer = rasterBindingsHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    GPUDrivenStreamAssetRasterBindings rasterBindings = rasterBindingsBuffer[0];
    uint drawIndex = groupID.x;
    if (drawIndex >= params.drawTaskCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }
    uint localRecordIndex = drawIndex / kMeshletTriangleChunkCount;
    uint chunkIndex = drawIndex - localRecordIndex * kMeshletTriangleChunkCount;
    uint recordCapacity = rasterBindings.visibleRecordCapacity != 0u
        ? rasterBindings.visibleRecordCapacity
        : params.drawTaskCount / kMeshletTriangleChunkCount;
    uint recordIndex = 0u;
    if (!resolveVisibilityRecordIndex(
            localRecordIndex,
            rasterBindings.visibleRecordBase,
            recordCapacity,
            recordIndex)) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    GPUDrivenStreamAssetActiveHeader header = activeHeader[0];
    uint maxActiveGroupClusters = max(header.maxActiveGroupClusters, 1u);
    uint activeGroupIndex = localRecordIndex / maxActiveGroupClusters;
    uint clusterIndex = localRecordIndex - activeGroupIndex * maxActiveGroupClusters;
    uint activeGroupCount = min(header.activeGroupCount, header.activeGroupCapacity);
    if (activeGroupIndex >= activeGroupCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    GPUDrivenStreamAssetActiveGroup group = activeGroups[activeGroupIndex];
    if (group.pageIndex >= params.scenePageCount ||
        group.gpuSceneInstanceIndex == kInvalidClusterIndex ||
        clusterIndex >= group.clusterCount ||
        clusterIndex >= 32u) {
        SetMeshOutputCounts(0, 0);
        return;
    }
    if ((group.flags & kActiveGroupLoadRequest) != 0u) {
        if (clusterIndex == 0u) {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
        SetMeshOutputCounts(0, 0);
        return;
    }
    if ((group.flags & kActiveGroupUnloadRequest) != 0u) {
        if (clusterIndex == 0u) {
            requestStreamPageUnload(requests, params, group.pageIndex);
        }
        SetMeshOutputCounts(0, 0);
        return;
    }

    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[group.pageIndex];
    uint pageState = streamPageState(pageEntry);
    uint pageDeviceOffset = streamPageDeviceOffset(pageEntry);
    if (((group.clusterSelectionMask >> clusterIndex) & 1u) == 0u ||
        ((pageState != kStreamStateResident && pageState != kStreamStateLockedFallback) ||
        pageDeviceOffset == kInvalidStreamDeviceOffset)) {
        if (clusterIndex == 0u) {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
        SetMeshOutputCounts(0, 0);
        return;
    }
    if (params.pageBufferBytes < kPayloadHeaderBytes ||
        pageDeviceOffset > params.pageBufferBytes - kPayloadHeaderBytes) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    uint pageWord = pageDeviceOffset / 4u;
    uint payloadBytes = pageWords[pageWord + 12u];
    if (payloadBytes < kPayloadHeaderBytes ||
        payloadBytes > params.pageBufferBytes ||
        pageDeviceOffset > params.pageBufferBytes - payloadBytes) {
        SetMeshOutputCounts(0, 0);
        return;
    }
    uint clusterCount = pageWords[pageWord + 2u];
    uint payloadVertexCount = pageWords[pageWord + 3u];
    uint triangleIndexCount = pageWords[pageWord + 4u];
    uint attributeFlags = pageWords[pageWord + 14u];
    uint positionFormat = pageWords[pageWord + 20u];
    if (clusterIndex >= clusterCount ||
        (attributeFlags & kPayloadAttributePosition) == 0u ||
        positionFormat != kPayloadFormatFloat32x4) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    uint clusterOffsetBytes = pageWords[pageWord + 9u];
    uint positionOffsetBytes = pageWords[pageWord + 10u];
    uint triangleOffsetBytes = pageWords[pageWord + 11u];
    if ((clusterOffsetBytes & 3u) != 0u ||
        (positionOffsetBytes & 3u) != 0u ||
        !streamPayloadArrayRangeValid(
            clusterOffsetBytes,
            clusterCount,
            kPayloadClusterBytes,
            payloadBytes) ||
        !streamPayloadArrayRangeValid(
            positionOffsetBytes,
            payloadVertexCount,
            16u,
            payloadBytes) ||
        !streamPayloadRangeValid(
            triangleOffsetBytes,
            triangleIndexCount,
            payloadBytes)) {
        SetMeshOutputCounts(0, 0);
        return;
    }
    uint clusterWord = pageWord + clusterOffsetBytes / 4u +
        clusterIndex * kPayloadClusterWords;
    uint vertexOffset = pageWords[clusterWord + 0u];
    uint vertexCount = pageWords[clusterWord + 1u];
    uint triangleOffset = pageWords[clusterWord + 2u];
    uint clusterTriangleCount = pageWords[clusterWord + 3u];
    if (vertexCount == 0u ||
        vertexCount > kMaxMeshletVertices ||
        vertexOffset > payloadVertexCount ||
        vertexCount > payloadVertexCount - vertexOffset ||
        clusterTriangleCount > kMaxMeshletTriangles ||
        triangleOffset > triangleIndexCount ||
        clusterTriangleCount > (triangleIndexCount - triangleOffset) / 3u) {
        SetMeshOutputCounts(0, 0);
        return;
    }
    uint triangleStart = chunkIndex * kMeshletTriangleChunkSize;
    if (triangleStart >= clusterTriangleCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    float3 localSphereCenter = float3(
        asfloat(pageWords[clusterWord + 12u]),
        asfloat(pageWords[clusterWord + 13u]),
        asfloat(pageWords[clusterWord + 14u]));
    float localSphereRadius = max(asfloat(pageWords[clusterWord + 15u]), 0.0);
    float3 worldSphereCenter =
        group.world0.xyz * localSphereCenter.x +
        group.world1.xyz * localSphereCenter.y +
        group.world2.xyz * localSphereCenter.z +
        group.world3.xyz;
    float worldScale = max(
        max(length(group.world0.xyz), length(group.world1.xyz)),
        max(length(group.world2.xyz), 0.000001));
    float worldSphereRadius = localSphereRadius * worldScale;
    GPUDrivenStreamAssetCameraData rasterCamera = currentStreamCamera(params);
    if ((rasterBindings.cullingFlags & (1u << 2u)) != 0u &&
        !streamSphereInsideFrustum(
            worldSphereCenter,
            worldSphereRadius,
            rasterCamera)) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    float coneCutoff = asfloat(pageWords[clusterWord + 19u]);
    if ((rasterBindings.cullingFlags & (1u << 3u)) != 0u && coneCutoff < 0.9999) {
        float3 localConeApex = float3(
            asfloat(pageWords[clusterWord + 16u]),
            asfloat(pageWords[clusterWord + 17u]),
            asfloat(pageWords[clusterWord + 18u]));
        float3 localConeAxis = float3(
            asfloat(pageWords[clusterWord + 20u]),
            asfloat(pageWords[clusterWord + 21u]),
            asfloat(pageWords[clusterWord + 22u]));
        float3 worldConeApex =
            group.world0.xyz * localConeApex.x +
            group.world1.xyz * localConeApex.y +
            group.world2.xyz * localConeApex.z +
            group.world3.xyz;
        float3 worldConeAxis = transformStreamNormal(
            group,
            localConeAxis,
            float3(0.0, 0.0, 1.0));
        float3 viewDirection = safeNormalize(
            worldConeApex - rasterCamera.eye.xyz,
            -worldConeAxis);
        if (dot(viewDirection, worldConeAxis) >= coneCutoff) {
            SetMeshOutputCounts(0, 0);
            return;
        }
    }

    DescriptorHandle<StructuredBuffer<uint>> visibilityStateHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + rasterBindings.instanceVisibilityBuffer,
            0));
    StructuredBuffer<uint> visibilityStates = visibilityStateHandle;
    uint rasterPhase = min(push.traversalPhase, 1u);
    uint expectedState = rasterPhase == 0u ? 1u : 3u;
    if (visibilityStates[group.gpuSceneInstanceIndex] != expectedState) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    DescriptorHandle<RWStructuredBuffer<VisibleClusterRecord>> visibleClusterHandle =
        DescriptorHandle<RWStructuredBuffer<VisibleClusterRecord>>(uint2(
            push.bufferShaderIndexBase + rasterBindings.visibleClusterBuffer,
            0));
    RWStructuredBuffer<VisibleClusterRecord> visibleClusters = visibleClusterHandle;
    if (chunkIndex == 0u) {
        VisibleClusterRecord record;
        record.clusterIndex = clusterIndex;
        record.instanceIndex = group.gpuSceneInstanceIndex;
        record.dataIndex = activeGroupIndex;
        record.flags = kVisibleClusterSourceStreamPage << kVisibleClusterSourceShift;
        visibleClusters[localRecordIndex] = record;
    }

    uint triangleCount = min(
        clusterTriangleCount - triangleStart,
        kMeshletTriangleChunkSize);
    uint outputVertexCount = triangleCount * 3u;
    SetMeshOutputCounts(outputVertexCount, triangleCount);

    uint triangleBaseByte = pageDeviceOffset + triangleOffsetBytes + triangleOffset;
    for (uint vertexIndex = 0; vertexIndex < kMaxMeshOutputVertices; ++vertexIndex) {
        if (vertexIndex < outputVertexCount) {
            uint chunkTriangleIndex = vertexIndex / 3u;
            uint triangleCorner = vertexIndex % 3u;
            uint meshletTriangleIndex = triangleStart + chunkTriangleIndex;
            uint localVertexIndex = readPageByte(
                pageWords,
                triangleBaseByte + meshletTriangleIndex * 3u + triangleCorner);
            if (localVertexIndex >= vertexCount) {
                localVertexIndex = 0u;
            }
            uint positionWord = pageWord + positionOffsetBytes / 4u +
                (vertexOffset + localVertexIndex) * 4u;
            float3 localPosition = float3(
                asfloat(pageWords[positionWord + 0u]),
                asfloat(pageWords[positionWord + 1u]),
                asfloat(pageWords[positionWord + 2u]));
            float3 worldPosition =
                group.world0.xyz * localPosition.x +
                group.world1.xyz * localPosition.y +
                group.world2.xyz * localPosition.z +
                group.world3.xyz;
            float4 projected = projectPosition(worldPosition, params);
            projected.y = -projected.y;
            vertices[vertexIndex].position = projected;
            vertices[vertexIndex].visibilityId =
                packVisibilityId(recordIndex, meshletTriangleIndex);
        }
    }

    for (uint triangleIndex = 0; triangleIndex < kMeshletTriangleChunkSize; ++triangleIndex) {
        if (triangleIndex < triangleCount) {
            triangles[triangleIndex] = uint3(
                triangleIndex * 3u,
                triangleIndex * 3u + 1u,
                triangleIndex * 3u + 2u);
        }
    }
}

float edgeFunction(float2 a, float2 b, float2 point)
{
    float2 edge = b - a;
    float2 offset = point - a;
    return edge.x * offset.y - edge.y * offset.x;
}

uint packStreamColor(float3 color)
{
    uint3 encoded = uint3(round(saturate(color) * 255.0));
    return encoded.x | (encoded.y << 8u) | (encoded.z << 16u) | 0xff000000u;
}

float4 unpackStreamColor(uint packed)
{
    return float4(
        float(packed & 0xffu),
        float((packed >> 8u) & 0xffu),
        float((packed >> 16u) & 0xffu),
        float((packed >> 24u) & 0xffu)) / 255.0;
}

float3 streamPerspectiveBarycentrics(
    float2 pixelCenter,
    float4 clip0,
    float4 clip1,
    float4 clip2,
    uint width,
    uint height)
{
    float2 viewport = float2(max(width, 1u), max(height, 1u));
    float2 screen0 = (clip0.xy / max(abs(clip0.w), 0.000001) * 0.5 + 0.5) * viewport;
    float2 screen1 = (clip1.xy / max(abs(clip1.w), 0.000001) * 0.5 + 0.5) * viewport;
    float2 screen2 = (clip2.xy / max(abs(clip2.w), 0.000001) * 0.5 + 0.5) * viewport;
    float area = edgeFunction(screen0, screen1, screen2);
    if (abs(area) < 0.000001) {
        return float3(1.0, 0.0, 0.0);
    }
    float3 affine = float3(
        edgeFunction(screen1, screen2, pixelCenter),
        edgeFunction(screen2, screen0, pixelCenter),
        edgeFunction(screen0, screen1, pixelCenter)) / area;
    float3 weighted = affine / float3(
        max(abs(clip0.w), 0.000001),
        max(abs(clip1.w), 0.000001),
        max(abs(clip2.w), 0.000001));
    float weight = weighted.x + weighted.y + weighted.z;
    return abs(weight) > 0.000001 ? weighted / weight : affine;
}

float4 loadPayloadFloat4(
    StructuredBuffer<uint> pageWords,
    uint pageWord,
    uint byteOffset,
    uint elementIndex)
{
    uint word = pageWord + byteOffset / 4u + elementIndex * 4u;
    return float4(
        asfloat(pageWords[word + 0u]),
        asfloat(pageWords[word + 1u]),
        asfloat(pageWords[word + 2u]),
        asfloat(pageWords[word + 3u]));
}

float2 loadPayloadFloat2(
    StructuredBuffer<uint> pageWords,
    uint pageWord,
    uint byteOffset,
    uint elementIndex)
{
    uint word = pageWord + byteOffset / 4u + elementIndex * 2u;
    return float2(
        asfloat(pageWords[word + 0u]),
        asfloat(pageWords[word + 1u]));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gpuDrivenStreamAssetDeferredMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(
            push.bufferShaderIndexBase + push.paramsBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> rasterBindingsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(
            push.bufferShaderIndexBase + push.rasterBindingsBuffer,
            0));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rasterBindingsBuffer = rasterBindingsHandle;
    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    GPUDrivenStreamAssetRasterBindings rasterBindings = rasterBindingsBuffer[0];
    uint width = max(rasterBindings.width, 1u);
    uint height = max(rasterBindings.height, 1u);
    if (dispatchThreadID.x >= width || dispatchThreadID.y >= height) {
        return;
    }

    DescriptorHandle<Texture2D<uint>> visibilityHandle =
        DescriptorHandle<Texture2D<uint>>(uint2(
            push.imageShaderIndexBase + rasterBindings.visibilityImage,
            0));
    DescriptorHandle<RWStructuredBuffer<uint>> outputHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + rasterBindings.deferredColorBuffer,
            0));
    Texture2D<uint> visibility = visibilityHandle;
    RWStructuredBuffer<uint> output = outputHandle;
    uint outputIndex = dispatchThreadID.y * width + dispatchThreadID.x;
    uint packedVisibility = visibility.Load(int3(dispatchThreadID.xy, 0));
    uint recordIndex = 0u;
    uint triangleIndex = 0u;
    if (!unpackVisibilityId(packedVisibility, recordIndex, triangleIndex)) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    uint visibleRecordCapacity = rasterBindings.visibleRecordCapacity != 0u
        ? rasterBindings.visibleRecordCapacity
        : params.drawTaskCount / kMeshletTriangleChunkCount;
    if (!visibilityRecordRangeValid(
            rasterBindings.visibleRecordBase,
            visibleRecordCapacity) ||
        recordIndex < rasterBindings.visibleRecordBase ||
        recordIndex - rasterBindings.visibleRecordBase >= visibleRecordCapacity) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }

    DescriptorHandle<StructuredBuffer<VisibleClusterRecord>> visibleClusterHandle =
        DescriptorHandle<StructuredBuffer<VisibleClusterRecord>>(uint2(
            push.bufferShaderIndexBase + rasterBindings.visibleClusterBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>> activeGroupHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(
            push.bufferShaderIndexBase + push.activeGroupBuffer,
            0));
    DescriptorHandle<StructuredBuffer<uint>> pageHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + push.pageBuffer,
            0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTableHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(
            push.bufferShaderIndexBase + push.pageTableBuffer,
            0));
    StructuredBuffer<VisibleClusterRecord> visibleClusters = visibleClusterHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    StructuredBuffer<uint> pageWords = pageHandle;
    StructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;

    uint localRecordIndex = recordIndex - rasterBindings.visibleRecordBase;
    VisibleClusterRecord record = visibleClusters[localRecordIndex];
    if (visibleClusterSource(record) != kVisibleClusterSourceStreamPage) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    GPUDrivenStreamAssetActiveHeader activeHeader;
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>> activeHeaderHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(uint2(
            push.bufferShaderIndexBase + push.activeHeaderBuffer,
            0));
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeaders = activeHeaderHandle;
    activeHeader = activeHeaders[0];
    uint activeGroupCount = min(activeHeader.activeGroupCount, activeHeader.activeGroupCapacity);
    if (record.dataIndex >= activeGroupCount) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    GPUDrivenStreamAssetActiveGroup group = activeGroups[record.dataIndex];
    if (record.instanceIndex == kInvalidClusterIndex ||
        record.instanceIndex != group.gpuSceneInstanceIndex ||
        group.pageIndex >= params.scenePageCount) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[group.pageIndex];
    uint pageDeviceOffset = streamPageDeviceOffset(pageEntry);
    if (!streamPageDrawable(streamPageState(pageEntry)) ||
        pageDeviceOffset == kInvalidStreamDeviceOffset ||
        params.pageBufferBytes < kPayloadHeaderBytes ||
        pageDeviceOffset > params.pageBufferBytes - kPayloadHeaderBytes) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }

    uint pageWord = pageDeviceOffset / 4u;
    uint payloadBytes = pageWords[pageWord + 12u];
    uint clusterCount = pageWords[pageWord + 2u];
    uint payloadVertexCount = pageWords[pageWord + 3u];
    uint triangleIndexCount = pageWords[pageWord + 4u];
    if (payloadBytes < kPayloadHeaderBytes ||
        payloadBytes > params.pageBufferBytes ||
        pageDeviceOffset > params.pageBufferBytes - payloadBytes ||
        record.clusterIndex >= clusterCount) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }

    uint clusterOffsetBytes = pageWords[pageWord + 9u];
    uint positionOffsetBytes = pageWords[pageWord + 10u];
    uint triangleOffsetBytes = pageWords[pageWord + 11u];
    uint attributeFlags = pageWords[pageWord + 14u];
    uint normalOffsetBytes = pageWords[pageWord + 16u];
    uint texcoordOffsetBytes = pageWords[pageWord + 17u];
    uint positionFormat = pageWords[pageWord + 20u];
    uint normalFormat = pageWords[pageWord + 21u];
    uint texcoordFormat = pageWords[pageWord + 22u];
    uint tangentOffsetBytes = pageWords[pageWord + 24u];
    uint tangentFormat = pageWords[pageWord + 25u];
    bool hasNormals = (attributeFlags & kPayloadAttributeNormal) != 0u;
    bool hasTexcoords = (attributeFlags & kPayloadAttributeTexcoord0) != 0u;
    bool hasTangents = (attributeFlags & kPayloadAttributeTangent) != 0u;
    if ((attributeFlags & kPayloadAttributePosition) == 0u ||
        positionFormat != kPayloadFormatFloat32x4 ||
        (clusterOffsetBytes & 3u) != 0u ||
        (positionOffsetBytes & 3u) != 0u ||
        !streamPayloadArrayRangeValid(
            clusterOffsetBytes,
            clusterCount,
            kPayloadClusterBytes,
            payloadBytes) ||
        !streamPayloadArrayRangeValid(
            positionOffsetBytes,
            payloadVertexCount,
            16u,
            payloadBytes) ||
        !streamPayloadRangeValid(
            triangleOffsetBytes,
            triangleIndexCount,
            payloadBytes) ||
        (hasNormals &&
            (normalFormat != kPayloadFormatFloat32x4 ||
             (normalOffsetBytes & 3u) != 0u ||
             !streamPayloadArrayRangeValid(
                 normalOffsetBytes,
                 payloadVertexCount,
                 16u,
                 payloadBytes))) ||
        (hasTexcoords &&
            (texcoordFormat != kPayloadFormatFloat32x2 ||
             (texcoordOffsetBytes & 3u) != 0u ||
             !streamPayloadArrayRangeValid(
                 texcoordOffsetBytes,
                 payloadVertexCount,
                 8u,
                 payloadBytes))) ||
        (hasTangents &&
            (tangentFormat != kPayloadFormatFloat32x4 ||
             (tangentOffsetBytes & 3u) != 0u ||
             !streamPayloadArrayRangeValid(
                 tangentOffsetBytes,
                 payloadVertexCount,
                 16u,
                 payloadBytes)))) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    uint clusterWord = pageWord + clusterOffsetBytes / 4u +
        record.clusterIndex * kPayloadClusterWords;
    uint vertexOffset = pageWords[clusterWord + 0u];
    uint vertexCount = pageWords[clusterWord + 1u];
    uint localTriangleOffset = pageWords[clusterWord + 2u];
    uint triangleCount = pageWords[clusterWord + 3u];
    if (triangleIndex >= triangleCount ||
        triangleCount > kMaxMeshletTriangles ||
        vertexCount == 0u ||
        vertexCount > kMaxMeshletVertices ||
        vertexOffset > payloadVertexCount ||
        vertexCount > payloadVertexCount - vertexOffset ||
        localTriangleOffset > triangleIndexCount ||
        triangleCount > (triangleIndexCount - localTriangleOffset) / 3u) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }

    uint triangleByte = pageDeviceOffset + triangleOffsetBytes +
        localTriangleOffset + triangleIndex * 3u;
    uint3 localIndices = uint3(
        readPageByte(pageWords, triangleByte + 0u),
        readPageByte(pageWords, triangleByte + 1u),
        readPageByte(pageWords, triangleByte + 2u));
    if (any(localIndices >= vertexCount)) {
        output[outputIndex] = packStreamColor(params.clearColor.xyz);
        return;
    }
    uint3 vertexIndices = vertexOffset + localIndices;
    float3 local0 = loadPayloadFloat4(pageWords, pageWord, positionOffsetBytes, vertexIndices.x).xyz;
    float3 local1 = loadPayloadFloat4(pageWords, pageWord, positionOffsetBytes, vertexIndices.y).xyz;
    float3 local2 = loadPayloadFloat4(pageWords, pageWord, positionOffsetBytes, vertexIndices.z).xyz;
    float3 world0 = group.world0.xyz * local0.x + group.world1.xyz * local0.y +
        group.world2.xyz * local0.z + group.world3.xyz;
    float3 world1 = group.world0.xyz * local1.x + group.world1.xyz * local1.y +
        group.world2.xyz * local1.z + group.world3.xyz;
    float3 world2 = group.world0.xyz * local2.x + group.world1.xyz * local2.y +
        group.world2.xyz * local2.z + group.world3.xyz;
    float4 clip0 = projectPosition(world0, params);
    float4 clip1 = projectPosition(world1, params);
    float4 clip2 = projectPosition(world2, params);
    clip0.y = -clip0.y;
    clip1.y = -clip1.y;
    clip2.y = -clip2.y;
    float3 barycentrics = streamPerspectiveBarycentrics(
        float2(dispatchThreadID.xy) + 0.5,
        clip0,
        clip1,
        clip2,
        width,
        height);

    float3 geometryNormal = safeNormalize(
        cross(world1 - world0, world2 - world0),
        float3(0.0, 1.0, 0.0));
    float3 authoredNormal = geometryNormal;
    if (hasNormals) {
        float3 normal0 = loadPayloadFloat4(pageWords, pageWord, normalOffsetBytes, vertexIndices.x).xyz;
        float3 normal1 = loadPayloadFloat4(pageWords, pageWord, normalOffsetBytes, vertexIndices.y).xyz;
        float3 normal2 = loadPayloadFloat4(pageWords, pageWord, normalOffsetBytes, vertexIndices.z).xyz;
        float3 localNormal = normal0 * barycentrics.x + normal1 * barycentrics.y +
            normal2 * barycentrics.z;
        authoredNormal = transformStreamNormal(
            group,
            localNormal,
            geometryNormal);
    }
    if (dot(authoredNormal, geometryNormal) < 0.0) {
        authoredNormal = -authoredNormal;
    }

    float3 tangent = safeNormalize(
        cross(abs(authoredNormal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0), authoredNormal),
        float3(1.0, 0.0, 0.0));
    float handedness = 1.0;
    if (hasTangents) {
        float4 tangent0 = loadPayloadFloat4(pageWords, pageWord, tangentOffsetBytes, vertexIndices.x);
        float4 tangent1 = loadPayloadFloat4(pageWords, pageWord, tangentOffsetBytes, vertexIndices.y);
        float4 tangent2 = loadPayloadFloat4(pageWords, pageWord, tangentOffsetBytes, vertexIndices.z);
        float4 localTangent = tangent0 * barycentrics.x + tangent1 * barycentrics.y +
            tangent2 * barycentrics.z;
        float3 worldTangent = transformStreamDirection(group, localTangent.xyz);
        tangent = safeNormalize(
            worldTangent - authoredNormal * dot(authoredNormal, worldTangent),
            tangent);
        handedness = localTangent.w < 0.0 ? -1.0 : 1.0;
    }
    float3 bitangent = safeNormalize(
        cross(authoredNormal, tangent) * handedness,
        cross(authoredNormal, tangent));
    float3 shadingNormal = authoredNormal;
    float2 uv = float2(0.0, 0.0);
    if (hasTexcoords) {
        float2 uv0 = loadPayloadFloat2(pageWords, pageWord, texcoordOffsetBytes, vertexIndices.x);
        float2 uv1 = loadPayloadFloat2(pageWords, pageWord, texcoordOffsetBytes, vertexIndices.y);
        float2 uv2 = loadPayloadFloat2(pageWords, pageWord, texcoordOffsetBytes, vertexIndices.z);
        uv = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;
    }

    uint colorSeed = group.pageIndex;
    if (params.debugColorMode == 1u) {
        colorSeed = group.lodLevel;
    } else if (params.debugColorMode == 2u) {
        colorSeed = group.primitiveIndex;
    } else if (params.debugColorMode == 3u) {
        colorSeed = group.instanceIndex;
    } else if (params.debugColorMode == 4u) {
        colorSeed = ((group.pageIndex + 1u) * 0x9e3779b9u) ^ record.clusterIndex;
    }
    float3 color = hashColor(colorSeed + 1u);
    if (params.debugColorMode == 5u) {
        float3 lightDirection = safeNormalize(float3(0.35, 0.85, 0.4), float3(0.0, 1.0, 0.0));
        float diffuse = 0.18 + 0.82 * max(dot(shadingNormal, lightDirection), 0.0);
        float tangentPattern = 0.92 + 0.08 * sin((dot(uv, float2(1.0, 0.0)) + dot(bitangent, tangent)) * 40.0);
        color = float3(0.18, 0.42, 0.12) * diffuse * tangentPattern;
    }
    output[outputIndex] = packStreamColor(color);
}

[shader("vertex")]
GPUDrivenStreamAssetFullscreenOutput gpuDrivenStreamAssetCompositeVertexMain(
    uint vertexID : SV_VertexID)
{
    GPUDrivenStreamAssetFullscreenOutput output;
    float2 position = vertexID == 0u
        ? float2(-1.0, -1.0)
        : (vertexID == 1u ? float2(3.0, -1.0) : float2(-1.0, 3.0));
    output.position = float4(position, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 gpuDrivenStreamAssetCompositeFragmentMain(
    GPUDrivenStreamAssetFullscreenOutput input,
    uniform GPUDrivenStreamAssetPush push) : SV_Target0
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> rasterBindingsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(
            push.bufferShaderIndexBase + push.rasterBindingsBuffer,
            0));
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rasterBindingsBuffer = rasterBindingsHandle;
    GPUDrivenStreamAssetRasterBindings rasterBindings = rasterBindingsBuffer[0];
    uint width = max(rasterBindings.width, 1u);
    uint height = max(rasterBindings.height, 1u);
    uint2 pixel = min(uint2(input.position.xy), uint2(width - 1u, height - 1u));
    DescriptorHandle<StructuredBuffer<uint>> colorHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + rasterBindings.deferredColorBuffer,
            0));
    StructuredBuffer<uint> colors = colorHandle;
    return unpackStreamColor(colors[pixel.y * width + pixel.x]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetInitializePageTableMain(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform GPUDrivenStreamAssetPush push)
{
    static const uint kPageTableInitRowThreads = 65535u * 64u;
    uint pageIndex = dispatchThreadID.x + dispatchThreadID.y * kPageTableInitRowThreads;
    uint pageCount = push.activeBuildPhase;
    if (pageIndex >= pageCount) {
        return;
    }

    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTableHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;

    GPUDrivenStreamAssetPageTableEntry entry;
    entry.deviceOffsetAndState = 0u;
    entry.lastRequestFrame = 0u;
    pageTable[pageIndex] = entry;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetApplyUpdatesMain(uint3 dispatchThreadID : SV_DispatchThreadID, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTableHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> updateHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.updateBuffer, 0));

    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    StructuredBuffer<uint> updates = updateHandle;

    uint patchIndex = dispatchThreadID.x;
    uint patchCount = updates[1];
    if (patchIndex >= patchCount) {
        return;
    }

    uint patchWord = kStreamUpdateHeaderWords + patchIndex * 2u;
    uint pageId = updates[patchWord + 0u];
    pageTable[pageId].deviceOffsetAndState = updates[patchWord + 1u];
}

[shader("fragment")]
uint gpuDrivenStreamAssetFragmentMain(GPUDrivenStreamAssetVertexOutput input, uniform GPUDrivenStreamAssetPush push) : SV_Target0
{
    return input.visibilityId;
}
