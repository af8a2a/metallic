import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;
#include "WaveWorkDistribution.slang"
#define VISIBILITY_TESSELLATION_STREAM 1
#include "../VisibilityBuffer/VisibilityTessellation.slang"

static const uint kMaxMeshletVertices = 128;
static const uint kMaxMeshletTriangles = 128;
static const uint kMeshletTriangleChunkSize = 64u;
static const uint kMeshletTriangleChunkCount = 2u;
static const uint kPayloadHeaderBytes = 112u;
static const uint kPayloadClusterBytes = 96u;
static const uint kPayloadClusterWords = kPayloadClusterBytes / 4u;
static const uint kPayloadAttributePosition = 1u << 0u;
static const uint kPayloadAttributeNormal = 1u << 1u;
static const uint kPayloadAttributeTexcoord0 = 1u << 2u;
static const uint kPayloadAttributeTangent = 1u << 4u;
static const uint kPayloadFormatFloat32x2 = 1u;
static const uint kPayloadFormatFloat32x4 = 2u;
static const uint kPayloadFormatFloat32x3 = 4u;
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
static const uint kTraversalPhaseClearPriority = 2u;
static const uint kTraversalPhaseGatherPriority = 3u;
static const uint kActiveBuildPhaseReset = 0u;
static const uint kActiveBuildPhaseBuild = 1u;
static const uint kActiveBuildPhaseFinalize = 2u;
static const uint kActiveBuildPhaseSeed = 3u;
static const uint kActiveBuildPhaseRun = 4u;
static const uint kActiveBuildPhaseFrontier = 5u;
static const uint kActiveBuildPhasePrefix = 6u;
static const uint kActiveBuildPhaseEmit = 7u;
static const uint kActiveBuildPhaseInitializeLodState = 8u;
static const uint kActiveBuildPhasePrefetch = 9u;
static const uint kActiveBuildPhaseClear = 10u;
static const uint kActiveBuildPhaseMask = 11u;
static const uint kActiveBuildPhaseDemandReset = 12u;
static const uint kActiveBuildPhaseDemand = 13u;
static const uint kStreamDemandStatsWords = 32u;
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
static const uint kStreamRequestLoadPriorityOffsetWord = 11u;
static const uint kStreamRequestPriorityTableOffsetWord = 12u;
static const uint kStreamRequestPrefetchLimitWord = 13u;
static const uint kStreamRequestPrefetchCounterWord = 14u;
static const uint kStreamRequestPrefetchDroppedWord = 15u;
static const uint kStreamPrefetchPageTag = 1u << 31u;

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
    uint hybridQueueBuffer;
    uint hybridClusterBuffer;
    uint clasPublicationRevision;
    float tessellationEdgePixels;
    uint tessellationMaxFactor;
    uint tessellationMaxSplitDepth;
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
    uint gpuSceneInstanceBuffer;
    uint tessellationBuffer;
    float displacementBound;
    uint padding2;
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
    uint lodBvhOffset;
    uint lodBvhNodeCount;
    uint lodTileOffset;
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
    uint clusterRefinedOffset;
    uint flags;
    uint parentOffset;
    uint parentCount;
    float demandMinX;
    float demandMinY;
    float demandMinZ;
    float demandMaxX;
    float demandMaxY;
    float demandMaxZ;
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
    float lodPixelError;
    uint lodTopologyBuffer;
    uint lodStateBuffer;
    uint lodInstanceOffsetsOffset;
    float4 renderEye;
    float4 renderCenter;
    float4 renderUpProjection;
    float4 renderViewport;
    float4 renderClipOrtho;
    float4 prefetchParams;
    uint demandBuffer;
    uint demandTaskOffset;
    uint demandTaskCount;
    uint splitFrontier;
    uint demandInstanceOffsetsOffset;
    uint demandStatsBuffer;
    uint2 demandPadding;
};

struct GPUDrivenStreamAssetVertexOutput
{
    float4 position : SV_Position;
    nointerpolation uint doubleSided : TEXCOORD1;
};

// IDs vary per triangle, so they must not use a shared vertex's flat attribute.
struct GPUDrivenStreamAssetPrimitiveOutput
{
    uint visibilityId : SV_PrimitiveID;
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
    if (params.renderViewport.y > 0.0) {
        params.eye = params.renderEye;
        params.center = params.renderCenter;
        params.upProjection = params.renderUpProjection;
        params.viewport = params.renderViewport;
        params.clipOrtho = params.renderClipOrtho;
    }
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
    float ndcZ = conservativeProjectDepth(viewPosition.z, zNear, zFar,
        params.upProjection.w > 0.5, params.clipOrtho.w > 0.5);
    float aspect = max(params.viewport.x, 0.001);

    if (params.upProjection.w > 0.5) {
        float halfHeight = max(params.clipOrtho.z * 0.5, 0.0001);
        float4 clip = float4(
            viewPosition.x / (halfHeight * aspect),
            viewPosition.y / halfHeight,
            ndcZ,
            1.0);
        if (params.renderViewport.y > 0.0) {
            clip.xy -= 2.0 * float2(params.renderEye.w, -params.renderCenter.w) / params.renderViewport.yz * clip.w;
        }
        return clip;
    }

    float fovRadians = clamp(params.viewport.w, 0.017453292, 3.12413936);
    float yScale = 1.0 / tan(fovRadians * 0.5);
    float w = viewPosition.z;
    float4 clip = float4(
        viewPosition.x * yScale / aspect,
        viewPosition.y * yScale,
        ndcZ * w,
        w);
    if (params.renderViewport.y > 0.0) {
        clip.xy -= 2.0 * float2(params.renderEye.w, -params.renderCenter.w) / params.renderViewport.yz * clip.w;
    }
    return clip;
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
    return conservativeProjectDepth(viewZ, zNear, zFar,
        camera.upProjection.w > 0.5, camera.clipOrtho.w > 0.5);
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

bool streamRefinementDemandVisible(GPUDrivenStreamAssetGroup group, GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetParams params)
{
    if (params.enableGpuLodSelection == 0u || (group.flags & 1u) != 0u) { return true; }
    float3 low = float3(group.demandMinX, group.demandMinY, group.demandMinZ);
    float3 high = float3(group.demandMaxX, group.demandMaxY, group.demandMaxZ);
    if (any(low > high) || !all(isfinite(low)) || !all(isfinite(high))) { return true; }
    float3 center = low * 0.5 + high * 0.5, extent = high * 0.5 - low * 0.5;
    float3 a = instance.world0.xyz, b = instance.world1.xyz, c = instance.world2.xyz;
    float3 delta = a * center.x + b * center.y + c * center.z + instance.world3.xyz - params.eye.xyz;
    float3 forward = safeNormalize(params.center.xyz - params.eye.xyz, float3(0,0,-1));
    float3 right = safeNormalize(cross(forward, params.upProjection.xyz), float3(1,0,0));
    float3 up = cross(right, forward);
    float3 magnitude = abs(a) * (abs(center.x) + extent.x) + abs(b) * (abs(center.y) + extent.y) +
        abs(c) * (abs(center.z) + extent.z) + abs(instance.world3.xyz) + abs(params.eye.xyz);
    if (!all(isfinite(delta)) || !isfinite(dot(magnitude, float3(1, 1, 1)))) { return true; }
    float padding = dot(magnitude, float3(1,1,1)) * (32.0 * 1.1920928955078125e-7);
    float z = dot(delta, forward);
    float rz = dot(abs(float3(dot(a, forward), dot(b, forward), dot(c, forward))), extent) + padding;
    if (z + rz < params.clipOrtho.x || z - rz > params.clipOrtho.y) { return false; }
    // One pixel guard plus transform roundoff. Demand must be conservative at
    // viewport edges even while temporal jitter moves the raster samples.
    float guard = 1.0 + 2.0 / max(min(params.viewport.y, params.viewport.z), 1.0);
    float ty = tan(params.viewport.w * 0.5) * guard, tx = ty * params.viewport.x;
    if (params.upProjection.w != 0.0) {
        float halfHeight = params.clipOrtho.z * 0.5 * guard;
        float rx = dot(abs(float3(dot(a, right), dot(b, right), dot(c, right))), extent) + padding;
        float ry = dot(abs(float3(dot(a, up), dot(b, up), dot(c, up))), extent) + padding;
        return abs(dot(delta, right)) <= halfHeight * params.viewport.x + rx && abs(dot(delta, up)) <= halfHeight + ry;
    }
    float3 planes[4] = {forward * tx + right, forward * tx - right, forward * ty + up, forward * ty - up};
    for (uint i = 0u; i < 4u; ++i) {
        float3 normal = planes[i];
        float radius = dot(abs(float3(dot(a, normal), dot(b, normal), dot(c, normal))), extent);
        if (dot(delta, normal) + radius + padding * length(normal) < 0.0) { return false; }
    }
    return true;
}

// A scheduling heuristic, not a replacement for the conservative LOD cut.
// Offscreen dependencies remain requested with zero benefit. Shared pages use
// the maximum instance benefit, independently of request-list deduplication.
float streamGroupScreenBenefit(GPUDrivenStreamAssetGroup group, GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetParams params, float4 eye, float4 direction, float4 projection)
{
    float3 a = instance.world0.xyz, b = instance.world1.xyz, c = instance.world2.xyz;
    float ab = abs(dot(a, b)), ac = abs(dot(a, c)), bc = abs(dot(b, c));
    float scale = sqrt(max(dot(a, a) + ab + ac, max(dot(b, b) + ab + bc, dot(c, c) + ac + bc)));
    float4 sphere = float4(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ, group.boundsRadius);
    float3 center = a * sphere.x + b * sphere.y + c * sphere.z + instance.world3.xyz;
    float radius = (sphere.w + group.maxQuadricError) * scale;
    if (!streamSphereInsideFrustum(center, radius, currentStreamCamera(params))) { return 0.0; }
    float pixelsPerUnit = direction.w != 0.0 ? projection.x / max(projection.z, 1e-6) :
        projection.x / max(2.0 * projection.y, 1e-6) /
            max(dot(center - eye.xyz, direction.xyz) - radius, max(eye.w, 1e-6));
    float pixelRadius = min(radius * pixelsPerUnit, max(params.viewport.y, params.viewport.z));
    float area = min(3.14159265 * pixelRadius * pixelRadius, params.viewport.y * params.viewport.z);
    float error = meshletLodProjectedError(sphere, group.maxQuadricError, instance.world0, instance.world1,
        instance.world2, instance.world3, eye, direction, projection);
    float excess = clamp(error / max(projection.w, 1e-6) - 1.0, 1.0, 64.0);
    return clamp(max(area, 1.0) * excess, 0.0, 1e9);
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
    float4 rect;
    if (!conservativeSphereScreenRect(viewCenter, radius, float2(bindings.width, bindings.height),
        max(camera.viewport.x, 0.001), clamp(camera.viewport.w, 0.017453292, 3.12413936),
        camera.upProjection.w > 0.5 ? max(camera.clipOrtho.z, 0.0002) : 0.0, zNear, rect)) {
        return false;
    }
    float2 minPixels = rect.xy, maxPixels = rect.zw;
    float maxExtent = max(maxPixels.x - minPixels.x, maxPixels.y - minPixels.y);
    uint mipLevel = min(
        bindings.hzbMipCount - 1u,
        maxExtent > 1.0 ? uint(ceil(log2(maxExtent))) : 0u);
    uint mipWidth = streamMipDimension(bindings.width, mipLevel);
    uint mipHeight = streamMipDimension(bindings.height, mipLevel);
    float mipScale = exp2(-float(mipLevel));
    uint2 minTexel = min(uint2(minPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    uint2 maxTexel = min(uint2(maxPixels * mipScale), uint2(mipWidth - 1u, mipHeight - 1u));
    if (any(maxTexel - minTexel > 1u)) { return false; }
    uint mipOffset = streamHzbMipOffset(bindings, mipLevel);
    float depth00 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + minTexel.y * mipWidth + minTexel.x);
    float depth10 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + minTexel.y * mipWidth + maxTexel.x);
    float depth01 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + maxTexel.y * mipWidth + minTexel.x);
    float depth11 = loadStreamHzb(push, bindings, hzbIndex, mipOffset + maxTexel.y * mipWidth + maxTexel.x);
    float nearestZ = max(viewCenter.z - radius, zNear);
    float depthBias = max(0.001, radius * 0.01);
    float nearestDepth = streamProjectDepth(max(nearestZ - depthBias, zNear), camera);
    return conservativeDepthOccluded(nearestDepth, float4(depth00, depth10, depth01, depth11),
        camera.clipOrtho.w > 0.5);
}

bool streamClusterVisibleInOcclusionPhase(float3 center, float radius, uint instanceState,
    GPUDrivenStreamAssetParams params, GPUDrivenStreamAssetRasterBindings bindings, GPUDrivenStreamAssetPush push, out bool retryLate)
{
    retryLate = false;
    bool early = push.traversalPhase == 0u;
    if ((bindings.cullingFlags & (1u << 4u)) == 0u) {
        return instanceState == (early ? 1u : 3u);
    }
    if (early ? instanceState != 1u : (instanceState != 1u && instanceState != 3u)) { return false; }
    uint currentHzb = params.frameIndex & 1u;
    bool deferred = instanceState == 3u || (bindings.hzbValid != 0u &&
        streamSphereOccluded(center, radius, previousStreamCamera(params), bindings, push, currentHzb ^ 1u));
    if (early) { retryLate = deferred; return !deferred; }
    return deferred && !streamSphereOccluded(center, radius, currentStreamCamera(params), bindings, push, currentHzb);
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
    float radius = max(instance.boundsCenterRadius.w, 0.000001) + bindings.displacementBound;
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
    uint rowBase = dispatchThreadID.y * (65535u * 64u);
    if (dispatchThreadID.x > 0xffffffffu - rowBase) { return; }
    uint threadIndex = dispatchThreadID.x + rowBase;
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
    if (push.traversalPhase == kTraversalPhaseClearPriority) {
        uint table = requests[kStreamRequestPriorityTableOffsetWord];
        if (table != 0u && threadIndex < params.scenePageCount) { requests[table + threadIndex] = 0u; }
        return;
    }
    if (push.traversalPhase == kTraversalPhaseGatherPriority) {
        uint table = requests[kStreamRequestPriorityTableOffsetWord];
        uint output = requests[kStreamRequestLoadPriorityOffsetWord];
        uint count = min(requests[kStreamRequestLoadCounterWord], requests[kStreamRequestMaxLoadsWord]);
        if (table != 0u && output != 0u && threadIndex < count) {
            uint page = requests[kStreamRequestHeaderWords + threadIndex];
            if (requests[kStreamRequestPrefetchLimitWord] != 0u) { page &= ~kStreamPrefetchPageTag; }
            requests[output + threadIndex] = page < params.scenePageCount ? requests[table + page] : 0u;
        }
        return;
    }
    if (push.traversalPhase == kTraversalPhaseUnload) {
        uint residentPageIndex = threadIndex;
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
        if (threadIndex >= workerCount) {
            return;
        }
        for (uint instanceIndex = threadIndex;
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
    uint workItemIndex = threadIndex;
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

// MeshletLodBvhNode is 10 scalar words in the resident topology buffer. Nodes
// are preorder, leaves cover descending group IDs, and escape skips a subtree.
// This preserves parent-before-child evaluation without a stack/queue limit.
bool streamLodBvhNeedsFine(StructuredBuffer<uint> topology, uint offset,
    GPUDrivenStreamAssetInstance instance, float4 eye, float4 forward,
    float4 projection, uint manualLevel)
{
    if ((topology[offset + 6u] & 1u) != 0u) { return true; }
    if (manualLevel != 0xffffffffu) { return topology[offset + 5u] >= manualLevel; }
    float4 sphere = float4(asfloat(topology[offset]), asfloat(topology[offset + 1u]),
        asfloat(topology[offset + 2u]), asfloat(topology[offset + 3u]));
    // Computing parent and child centers independently in float can break
    // containment after large-coordinate cancellation. Bound both affine/view
    // roundoff errors using all possible centers inside this object sphere.
    float worldPadding = 0.0;
    if (forward.w == 0.0) {
        float3 extent = abs(sphere.xyz) + sphere.www;
        float3 magnitude = abs(instance.world0.xyz) * extent.x + abs(instance.world1.xyz) * extent.y +
            abs(instance.world2.xyz) * extent.z + abs(instance.world3.xyz) + abs(eye.xyz);
        worldPadding = (32.0 * 1.1920928955078125e-7) * (magnitude.x + magnitude.y + magnitude.z);
    }
    float upperError = meshletLodProjectedError(sphere, asfloat(topology[offset + 4u]),
        instance.world0, instance.world1, instance.world2, instance.world3, eye, forward, projection, worldPadding);
    // Outward-rounded node spheres plus a small conservative comparison guard
    // prevent floating point cancellation at an LOD threshold from dropping a
    // desired leaf. Non-finite upper bounds also descend; leaves use exact error.
    return !(upperError < projection.w - max(abs(projection.w) * 1e-4, 1e-5));
}

void buildStreamLodBvhFrontier(uint instanceIndex,
    GPUDrivenStreamAssetInstance instance, GPUDrivenStreamAssetPrimitive primitive,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups, StructuredBuffer<uint> topology,
    RWStructuredBuffer<uint> state,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests, GPUDrivenStreamAssetParams params)
{
    uint header = instanceIndex * 4u;
    uint base = topology[params.lodInstanceOffsetsOffset + instanceIndex];
    // After the existing active/mask pairs: all-active count, node visits,
    // group tests, reserved; then groupCount sparse local group IDs.
    uint sparse = base + primitive.groupCount * 2u;
    uint previousCount = state[sparse];
    for (uint index = 0u; index < previousCount; ++index) {
        uint localGroup = state[sparse + 4u + index];
        state[base + localGroup * 2u] = 0u;
        state[base + localGroup * 2u + 1u] = 0u;
    }
    state[sparse] = 0u;
    state[sparse + 1u] = 0u;
    state[sparse + 2u] = 0u;
    state[sparse + 3u] = 0u;
    if (instance.visible == 0u) { return; }

    float3 delta = params.center.xyz - params.eye.xyz;
    float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
    float4 lodEye = float4(params.eye.xyz, params.clipOrtho.x);
    float4 lodForward = float4(forward, params.upProjection.w);
    float4 lodProjection = float4(params.viewport.z, tan(params.viewport.w * 0.5),
        params.clipOrtho.z, params.lodPixelError);
    uint manualLevel = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
    uint activeCount = 0u, rootCount = 0u, visitedNodes = 0u, testedGroups = 0u;
    bool rootsReady = true;
    uint nodeIndex = 0u;
    while (nodeIndex < primitive.lodBvhNodeCount) {
        uint offset = primitive.lodBvhOffset + nodeIndex * 10u;
        ++visitedNodes;
        if (!streamLodBvhNeedsFine(topology, offset, instance,
                lodEye, lodForward, lodProjection, manualLevel)) {
            nodeIndex = topology[offset + 7u];
            continue;
        }
        uint firstGroup = topology[offset + 8u];
        uint groupCount = topology[offset + 9u];
        for (uint remaining = groupCount; remaining != 0u; --remaining) {
            uint localGroup = firstGroup + remaining - 1u;
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
            ++testedGroups;
            bool terminal = (group.flags & 1u) != 0u;
            bool parentsActive = true;
            for (uint parent = 0u; parent < group.parentCount; ++parent) {
                uint parentId = topology[group.parentOffset + parent];
                parentsActive = parentsActive && state[base + (parentId - primitive.groupOffset) * 2u] != 0u;
            }
            float4 sphere = float4(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ, group.boundsRadius);
            bool desired = meshletLodGroupNeedsFine(sphere, group.maxQuadricError,
                group.lodLevel, group.flags, instance.world0, instance.world1, instance.world2,
                instance.world3, lodEye, lodForward, lodProjection, manualLevel);
            // Demand comes from resident metadata, independently of the safe cut.
            bool demanded = terminal || (desired && streamRefinementDemandVisible(group, instance, params));
            if (demanded) {
                if (streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) {
                    markStreamPageDemand(pageTable, params, group.pageIndex);
                } else {
                    requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
                }
            }
            bool drawable = groupPageDrawable(pageTable, params, group);
            if (parentsActive && demanded && drawable) {
                state[base + localGroup * 2u] = 1u;
                state[sparse + 4u + activeCount++] = localGroup;
            }
            if (terminal) {
                rootsReady = rootsReady && drawable;
                ++rootCount;
            }
        }
        ++nodeIndex;
    }
    // Keep ALL active IDs, including fully replaced ancestors and incomplete
    // primitives, so sparse clearing cannot leave stale reachability next frame.
    state[sparse] = activeCount;
    state[sparse + 1u] = visitedNodes;
    state[sparse + 2u] = testedGroups;
    uint selectedCount = 0u;
    if (rootsReady) {
        for (uint remaining = activeCount; remaining != 0u; --remaining) {
            uint localGroup = state[sparse + 4u + remaining - 1u];
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
            uint mask = 0u;
            for (uint cluster = 0u; cluster < group.clusterCount; ++cluster) {
                uint refined = topology[group.clusterRefinedOffset + cluster];
                if (refined == kInvalidClusterIndex ||
                    state[base + (refined - primitive.groupOffset) * 2u] == 0u) {
                    mask |= 1u << cluster;
                }
            }
            state[base + localGroup * 2u + 1u] = mask;
            selectedCount += mask != 0u ? 1u : 0u;
        }
    }
    state[header] = selectedCount;
    state[header + 2u] = rootsReady ? rootCount : 0u;
    state[header + 3u] = rootsReady ? 1u : 0u;
}

// The LOD topology is resident even when page payloads are absent. State has
// four header words per instance (count, prefix, roots, ready), followed by
// compact per-instance pairs (active group, selected cluster mask).
void buildStreamLodFrontier(
    uint instanceIndex,
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances,
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> topology,
    RWStructuredBuffer<uint> state,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<uint> requests,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetPush push)
{
    uint header = instanceIndex * 4u;
    state[header] = 0u;
    state[header + 1u] = 0u;
    state[header + 2u] = 0u;
    state[header + 3u] = 0u;
    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        return;
    }
    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    // Frustum rejection is current-frame and independent of HZB history. Keep
    // the ordinary state clearing below, so a returning instance cannot reuse
    // stale parent reachability. Manual cut/debug and RT traversal stay intact.
    if (params.enableGpuLodSelection != 0u && push.rasterBindingsBuffer != 0xffffffffu) {
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> bindings =
            DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(push.bufferShaderIndexBase + push.rasterBindingsBuffer, 0));
        if ((bindings[0].cullingFlags & 1u) != 0u &&
            !streamSphereInsideFrustum(instance.boundsCenterRadius.xyz,
                max(instance.boundsCenterRadius.w, 0.000001) + bindings[0].displacementBound, currentStreamCamera(params))) {
            instance.visible = 0u;
        }
    }
    if (primitive.lodBvhNodeCount != 0u) {
        buildStreamLodBvhFrontier(instanceIndex, instance, primitive, groups,
            topology, state, pageTable, requests, params);
        return;
    }
    // Metadata-free fixtures retain the brute-force traversal as an oracle.
    if (instance.visible == 0u) { return; }
    uint base = topology[params.lodInstanceOffsetsOffset + instanceIndex];
    bool rootsReady = true;
    uint rootCount = 0u;
    float3 delta = params.center.xyz - params.eye.xyz;
    float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
    float4 lodEye = float4(params.eye.xyz, params.clipOrtho.x);
    float4 lodForward = float4(forward, params.upProjection.w);
    float4 lodProjection = float4(params.viewport.z, tan(params.viewport.w * 0.5),
        params.clipOrtho.z, params.lodPixelError);
    uint manualLevel = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
    // Asset validation guarantees refined < owner. The reverse order therefore
    // evaluates every parent before its descendants, without a work queue cap.
    for (uint remaining = primitive.groupCount; remaining != 0u; --remaining) {
        uint localGroup = remaining - 1u;
        GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
        bool terminal = (group.flags & 1u) != 0u;
        bool parentsActive = true;
        for (uint parent = 0u; parent < group.parentCount; ++parent) {
            uint parentId = topology[group.parentOffset + parent];
            parentsActive = parentsActive && state[base + (parentId - primitive.groupOffset) * 2u] != 0u;
        }
        float4 sphere = float4(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ, group.boundsRadius);
        bool desired = meshletLodGroupNeedsFine(sphere, group.maxQuadricError,
            group.lodLevel, group.flags, instance.world0, instance.world1, instance.world2,
            instance.world3, lodEye, lodForward, lodProjection, manualLevel);
        bool demanded = terminal || (desired && streamRefinementDemandVisible(group, instance, params));
        if (demanded) {
            if (streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) {
                markStreamPageDemand(pageTable, params, group.pageIndex);
            } else {
                requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
            }
        }
        bool drawable = groupPageDrawable(pageTable, params, group);
        bool active = parentsActive && demanded && drawable;
        state[base + localGroup * 2u] = active ? 1u : 0u;
        state[base + localGroup * 2u + 1u] = 0u;
        if (terminal) {
            rootsReady = rootsReady && drawable;
            ++rootCount;
        }
    }
    // Publish only complete primitives. PendingUpload pages never replace
    // their coarse ancestors, and an incomplete terminal cut stays unready.
    uint selectedCount = 0u;
    if (rootsReady) {
        for (uint localGroup = 0u; localGroup < primitive.groupCount; ++localGroup) {
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
            uint mask = 0u;
            if (state[base + localGroup * 2u] != 0u) {
                for (uint cluster = 0u; cluster < group.clusterCount; ++cluster) {
                    uint refined = topology[group.clusterRefinedOffset + cluster];
                    if (refined == kInvalidClusterIndex ||
                        state[base + (refined - primitive.groupOffset) * 2u] == 0u) {
                        mask |= 1u << cluster;
                    }
                }
            }
            state[base + localGroup * 2u + 1u] = mask;
            selectedCount += mask != 0u ? 1u : 0u;
        }
    }
    state[header] = selectedCount;
    state[header + 2u] = rootsReady ? rootCount : 0u;
    state[header + 3u] = rootsReady ? 1u : 0u;
}

groupshared uint2 g_StreamLodPrefix[64];

// Saturate before addition so even malformed/extreme counts cannot wrap.
uint2 scanStreamLodCounts(uint2 value, uint lane, uint limit)
{
    g_StreamLodPrefix[lane] = min(value, limit);
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 64u; stride *= 2u) {
        uint2 previous = lane >= stride ? g_StreamLodPrefix[max(lane, stride) - stride] : uint2(0u);
        GroupMemoryBarrierWithGroupSync();
        uint2 current = g_StreamLodPrefix[lane];
        g_StreamLodPrefix[lane] = current + min(previous, limit - current);
        GroupMemoryBarrierWithGroupSync();
    }
    return lane == 0u ? uint2(0u) : g_StreamLodPrefix[lane - 1u];
}

void prefixStreamLodFrontier(
    RWStructuredBuffer<uint> state,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    GPUDrivenStreamAssetParams params, uint lane)
{
    uint perLane = (params.sceneInstanceCount + 63u) / 64u;
    uint first = min(lane * perLane, params.sceneInstanceCount);
    uint end = min(first + perLane, params.sceneInstanceCount);
    uint limit = params.activeGroupCount + 1u;
    uint2 total = 0u;
    for (uint i = first; i < end; ++i) {
        uint selected = state[i * 4u];
        uint baseline = min(selected, state[i * 4u + 2u]);
        total += min(uint2(selected, baseline), limit - total);
    }
    uint2 before = scanStreamLodCounts(total, lane, limit);
    uint2 all = g_StreamLodPrefix[63];
    bool overflow = all.x > params.activeGroupCount;
    bool invalidCapacity = all.y > params.activeGroupCount;
    if (overflow && !invalidCapacity) {
        // Reserve complete coverage for every instance, then spend remaining
        // slots in stable instance order. Detail cuts cheaper than their roots
        // are always kept. One view-wide overflow no longer coarsens everyone.
        // A second scan compacts the mixed full-detail / terminal output.
        uint extra = 0u;
        for (uint i = first; i < end; ++i) {
            uint selected = state[i * 4u];
            uint baseline = min(selected, state[i * 4u + 2u]);
            extra += min(selected - baseline, limit - extra);
        }
        GroupMemoryBarrierWithGroupSync();
        uint used = scanStreamLodCounts(uint2(extra, 0u), lane, limit).x;
        uint spare = params.activeGroupCount - all.y;
        total = 0u;
        for (uint i = first; i < end; ++i) {
            uint header = i * 4u;
            uint selected = state[header];
            uint baseline = min(selected, state[header + 2u]);
            uint cost = selected - baseline;
            bool keepDetail = cost == 0u || (used <= spare && cost <= spare - used);
            used += min(cost, limit - used);
            // bit 0: complete roots ready; bit 1: emit the terminal cut.
            if (!keepDetail) { state[header + 3u] |= 2u; }
            total.x += keepDetail ? selected : baseline;
            total.y += !keepDetail ? 1u : 0u;
        }
        GroupMemoryBarrierWithGroupSync();
        before = scanStreamLodCounts(total, lane, limit);
    }
    uint prefix = before.x;
    for (uint i = first; i < end; ++i) {
        uint header = i * 4u;
        state[header + 1u] = prefix;
        prefix += (state[header + 3u] & 2u) != 0u ? state[header + 2u] : state[header];
    }
    if (lane == 63u) {
        activeHeader[0].activeGroupCount = invalidCapacity ? 0u : g_StreamLodPrefix[63].x;
        activeHeader[0].overflowCount = invalidCapacity ? 2u : (overflow ? 1u : 0u);
        activeHeader[0].padding0 = overflow ? 1u : 0u;
        activeHeader[0].padding1 = invalidCapacity ? 1u : 0u;
        activeHeader[0].padding2 = overflow && !invalidCapacity ? g_StreamLodPrefix[63].y : 0u;
    }
}

void emitStreamLodFrontier(
    uint instanceIndex,
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances,
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> topology,
    RWStructuredBuffer<uint> state,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    GPUDrivenStreamAssetParams params)
{
    uint header = instanceIndex * 4u;
    if (state[header + 3u] == 0u || activeHeader[0].padding1 != 0u) {
        return;
    }
    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    uint base = topology[params.lodInstanceOffsetsOffset + instanceIndex];
    uint outputIndex = state[header + 1u];
    bool fallback = (state[header + 3u] & 2u) != 0u;
    bool sparseTraversal = primitive.lodBvhNodeCount != 0u;
    uint sparse = base + primitive.groupCount * 2u;
    uint groupCount = sparseTraversal ? state[sparse] : primitive.groupCount;
    for (uint index = 0u; index < groupCount; ++index) {
        uint localGroup = sparseTraversal ? state[sparse + 4u + groupCount - 1u - index] : index;
        GPUDrivenStreamAssetGroup source = groups[primitive.groupOffset + localGroup];
        uint mask = state[base + localGroup * 2u + 1u];
        if (fallback) {
            mask = (source.flags & 1u) == 0u ? 0u :
                (source.clusterCount == 32u ? 0xffffffffu : (1u << source.clusterCount) - 1u);
        }
        if (mask == 0u) {
            continue;
        }
        GPUDrivenStreamAssetActiveGroup group;
        group.pageDeviceOffsetBytes = streamPageDeviceOffset(pageTable[source.pageIndex]);
        group.pageIndex = source.pageIndex;
        group.clusterCount = source.clusterCount;
        group.primitiveIndex = source.primitiveIndex;
        group.lodLevel = source.lodLevel;
        group.materialIndex = instance.materialIndex;
        group.clusterSelectionMask = mask;
        group.flags = kActiveGroupResident;
        group.instanceIndex = instanceIndex;
        group.gpuSceneInstanceIndex = instance.gpuSceneInstanceIndex;
        group.padding1 = 0u;
        group.padding2 = 0u;
        group.world0 = instance.world0;
        group.world1 = instance.world1;
        group.world2 = instance.world2;
        group.world3 = instance.world3;
        activeGroups[outputIndex++] = group;
    }
}

// Bounded subtrees form a global queue across visible instances. The preceding
// clear/seed dispatch publishes all task payloads before workers claim them.
// Safe-cut state is computed later, after a dispatch barrier, in the original
// parent-first order. There is no producer polling inside this dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void streamDistributedDemandMain(uint3 workgroup : SV_GroupID, uint lane : SV_GroupIndex,
    uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    GPUDrivenStreamAssetParams params = paramsHandle[0];
    if (params.demandBuffer == 0xffffffffu) { return; }
    DescriptorHandle<RWStructuredBuffer<uint>> demand =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + params.demandBuffer, 0));
    if (push.activeBuildPhase == kActiveBuildPhaseDemandReset) {
        RWStructuredBuffer<uint> words = demand;
        uint size, stride;
        words.GetDimensions(size, stride);
        uint index = (workgroup.x + workgroup.y * 65535u) * 64u + lane;
        if (index < size) { demand[index] = 0u; }
        return;
    }
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>> instances =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(uint2(push.bufferShaderIndexBase + push.instanceBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>> primitives =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>(uint2(push.bufferShaderIndexBase + push.primitiveBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>> groups =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>(uint2(push.bufferShaderIndexBase + push.groupBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> topology =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + params.lodTopologyBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTable =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> requests =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.requestBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>> tasks =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>>(uint2(push.bufferShaderIndexBase + push.traversalWorkBuffer, 0));
    uint taskCount = demand[16];
    float3 delta = params.center.xyz - params.eye.xyz;
    float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
    float4 eye = float4(params.eye.xyz, params.clipOrtho.x);
    float4 direction = float4(forward, params.upProjection.w);
    float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5), params.clipOrtho.z, params.lodPixelError);
    uint manual = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
    uint processed = 0u, visitedTotal = 0u, testedTotal = 0u, maxVisited = 0u, maxTested = 0u, nonempty = 0u, slots = 0u;
    uint histogram[8];
    [unroll] for (uint bin = 0u; bin < 8u; ++bin) { histogram[bin] = 0u; }
    uint instanceIndex = 0u, tile = 0u, end = 0u, visited = 0u, tested = 0u;
    bool haveTask = false, exhausted = false;
    GPUDrivenStreamAssetInstance instance = {};
    GPUDrivenStreamAssetPrimitive primitive = {};
    for (;;) {
        uint firstGroup = 0u, count = 0u, leaf = 0u;
        // Fill productive sources before flattening their children. Refill
        // empty lanes across task boundaries; cap the search so one lane cannot
        // hold a ready wave while walking the remaining empty task tail.
        for (uint refill = 0u; refill < 8u && WaveActiveAnyTrue(count == 0u && !exhausted); ++refill) {
            bool needTask = count == 0u && !exhausted && tile == end;
            uint claims = WaveActiveCountBits(needTask);
            uint rank = WavePrefixCountBits(needTask);
            uint firstTask = 0u;
            if (WaveIsFirstLane() && claims != 0u) { InterlockedAdd(demand[0], claims, firstTask); }
            firstTask = WaveReadLaneFirst(firstTask);
            if (needTask) {
                if (haveTask) {
                    visitedTotal += visited; testedTotal += tested;
                    maxVisited = max(maxVisited, visited); maxTested = max(maxTested, tested);
                    nonempty += tested != 0u ? 1u : 0u;
                    uint bin = tested == 0u ? 0u : tested <= 8u ? 1u : min(7u, uint(firstbithigh(tested - 1u)) - 1u);
                    ++histogram[bin];
                }
                uint task = firstTask + rank;
                haveTask = task < taskCount;
                exhausted = !haveTask;
                visited = 0u; tested = 0u;
                if (haveTask) {
                    ++processed;
                    GPUDrivenStreamAssetTraversalWorkItem work = tasks[task];
                    instanceIndex = work.instanceIndex;
                    instance = instances[instanceIndex];
                    primitive = primitives[instance.primitiveIndex];
                    tile = work.nodeIndex;
                    end = topology[primitive.lodTileOffset + 1u + tile * 10u + 7u];
                }
            }
            if (count == 0u && !exhausted) {
                while (tile < end) {
                    uint offset = primitive.lodTileOffset + 1u + tile * 10u;
                    ++visited;
                    if (!streamLodBvhNeedsFine(topology, offset, instance, eye, direction, projection, manual)) {
                        tile = topology[offset + 7u];
                        continue;
                    }
                    ++tile;
                    count = topology[offset + 9u];
                    if (count != 0u) {
                        leaf = tile - 1u;
                        firstGroup = primitive.groupOffset + topology[offset + 8u] + count - 1u;
                        break;
                    }
                }
                tested += count;
            }
        }
        if (!WaveActiveAnyTrue(count != 0u) && WaveActiveAllTrue(exhausted)) { break; }
        WaveWorkDistribution work = waveWorkBegin(count);
        for (uint batch = 0u; batch < work.total; batch += WaveGetLaneCount()) {
            uint item = batch + WaveGetLaneIndex();
            uint localItem;
            uint source = waveWorkNext(work, batch, localItem);
            uint groupIndex = WaveReadLaneAt(firstGroup, source) - localItem;
            uint owner = WaveReadLaneAt(instanceIndex, source);
            uint ownerLeaf = WaveReadLaneAt(leaf, source);
            if (item < work.total) {
                GPUDrivenStreamAssetGroup group = groups[groupIndex];
                GPUDrivenStreamAssetInstance ownerInstance = instances[owner];
                bool wanted = (group.flags & 1u) != 0u ||
                    (meshletLodGroupNeedsFine(float4(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ, group.boundsRadius),
                        group.maxQuadricError, group.lodLevel, group.flags, ownerInstance.world0, ownerInstance.world1,
                        ownerInstance.world2, ownerInstance.world3, eye, direction, projection, manual) &&
                        streamRefinementDemandVisible(group, ownerInstance, params));
                if (wanted) {
                    GPUDrivenStreamAssetPrimitive ownerPrimitive = primitives[ownerInstance.primitiveIndex];
                    uint groupBits = topology[params.demandInstanceOffsetsOffset + owner * 4u];
                    uint bit = groupBits + groupIndex - ownerPrimitive.groupOffset;
                    InterlockedOr(demand[kStreamDemandStatsWords + (bit >> 5u)], 1u << (bit & 31u));
                    // Publish the demand path once per leaf. Other lanes may
                    // stop at an already marked ancestor: its winner will
                    // finish propagating before the dispatch barrier.
                    uint tileBits = topology[params.demandInstanceOffsetsOffset + owner * 4u + 1u];
                    uint parentTable = ownerPrimitive.lodTileOffset + 1u + topology[ownerPrimitive.lodTileOffset] * 10u;
                    for (uint node = ownerLeaf; node != 0xffffffffu; node = topology[parentTable + node]) {
                        uint nodeBit = tileBits + node;
                        uint mask = 1u << (nodeBit & 31u), previous;
                        InterlockedOr(demand[kStreamDemandStatsWords + (nodeBit >> 5u)], mask, previous);
                        if ((previous & mask) != 0u) { break; }
                    }
                    if (streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) {
                        markStreamPageDemand(pageTable, params, group.pageIndex);
                    } else {
                        uint table = requests[kStreamRequestPriorityTableOffsetWord];
                        if (table != 0u) {
                            float benefit = streamGroupScreenBenefit(group, ownerInstance, params, eye, direction, projection);
                            InterlockedMax(requests[table + group.pageIndex], asuint(benefit));
                        }
                        requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
                    }
                }
            }
            if (WaveIsFirstLane()) { slots += WaveGetLaneCount(); }
        }
    }
    // Reduce once per worker wave, not per task. Histograms count all admitted
    // tasks and make the remaining task tail observable.
    uint p = WaveActiveSum(processed), v = WaveActiveSum(visitedTotal), t = WaveActiveSum(testedTotal);
    uint mv = WaveActiveMax(maxVisited), mt = WaveActiveMax(maxTested), n = WaveActiveSum(nonempty);
    if (WaveIsFirstLane()) {
        InterlockedAdd(demand[1], p); InterlockedAdd(demand[2], v); InterlockedAdd(demand[3], t);
        InterlockedMax(demand[4], mv); InterlockedMax(demand[5], mt); InterlockedAdd(demand[6], n);
        InterlockedAdd(demand[7], slots);
        InterlockedAdd(demand[18], t);
    }
    if (workgroup.x == 0u && workgroup.y == 0u && lane == 0u) { demand[17] = 1u; }
    [unroll] for (uint bin = 0u; bin < 8u; ++bin) {
        uint value = WaveActiveSum(histogram[bin]);
        if (WaveIsFirstLane()) { InterlockedAdd(demand[8u + bin], value); }
    }
}

groupshared uint g_CooperativeWaveCounts[64];
groupshared uint g_CooperativeRoots;
groupshared uint g_CooperativeMissingRoots;

// Stable workgroup scan, with no fixed subgroup size assumption.
uint streamCooperativePrefix(uint value, uint lane, out uint total)
{
    uint wave = lane / WaveGetLaneCount();
    uint waves = (64u + WaveGetLaneCount() - 1u) / WaveGetLaneCount();
    uint prefix = WavePrefixSum(value);
    uint sum = WaveActiveSum(value);
    if (WaveIsFirstLane()) { g_CooperativeWaveCounts[wave] = sum; }
    GroupMemoryBarrierWithGroupSync();
    total = 0u;
    for (uint i = 0u; i < waves; ++i) {
        uint count = g_CooperativeWaveCounts[i];
        if (i < wave) { prefix += count; }
        total += count;
    }
    GroupMemoryBarrierWithGroupSync();
    return prefix;
}

void streamCooperativeMasks(StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> topology, RWStructuredBuffer<uint> state,
    GPUDrivenStreamAssetPrimitive primitive, uint header, uint base, uint sparse,
    uint count, bool ready, uint lane)
{
    uint selected = 0u;
    if (ready) {
        for (uint index = lane; index < count; index += 64u) {
            uint localGroup = state[sparse + 4u + index];
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
            uint mask = 0u;
            for (uint cluster = 0u; cluster < group.clusterCount; ++cluster) {
                uint refined = topology[group.clusterRefinedOffset + cluster];
                if (refined == kInvalidClusterIndex || state[base + (refined - primitive.groupOffset) * 2u] == 0u) { mask |= 1u << cluster; }
            }
            state[base + localGroup * 2u + 1u] = mask;
            selected += mask != 0u ? 1u : 0u;
        }
    }
    uint totalSelected;
    streamCooperativePrefix(selected, lane, totalSelected);
    if (lane == 0u) { state[header] = totalSelected; }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void streamCooperativeLodMain(uint3 workgroup : SV_GroupID, uint lane : SV_GroupIndex,
    uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    GPUDrivenStreamAssetParams params = paramsHandle[0];
    uint instanceIndex = workgroup.x + workgroup.y * 65535u;
    if (instanceIndex >= params.sceneInstanceCount) { return; }
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>> instances =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetInstance>>(uint2(push.bufferShaderIndexBase + push.instanceBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>> primitives =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPrimitive>>(uint2(push.bufferShaderIndexBase + push.primitiveBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>> groups =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>(uint2(push.bufferShaderIndexBase + push.groupBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> topology =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + params.lodTopologyBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> state =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + params.lodStateBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTable =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> requests =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.requestBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup>> activeGroups =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(push.bufferShaderIndexBase + push.activeGroupBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader>> activeHeader =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(uint2(push.bufferShaderIndexBase + push.activeHeaderBuffer, 0));
    GPUDrivenStreamAssetInstance instance = instances[instanceIndex];
    uint header = instanceIndex * 4u;
    if (instance.primitiveIndex >= params.scenePrimitiveCount) {
        if (lane < 4u && push.activeBuildPhase != kActiveBuildPhasePrefetch) { state[header + lane] = 0u; }
        return;
    }
    GPUDrivenStreamAssetPrimitive primitive = primitives[instance.primitiveIndex];
    // Request-only lookahead: parent residency still gates the actual frontier.
    // Run after all demand requests so speculation cannot consume their slots.
    if (push.activeBuildPhase == kActiveBuildPhasePrefetch) {
        uint limit = requests[kStreamRequestPrefetchLimitWord];
        if (params.enableGpuLodSelection == 0u || params.prefetchParams.z == 0.0 ||
            instance.visible == 0u || primitive.lodTileOffset == 0xffffffffu || limit == 0u) { return; }
        GPUDrivenStreamAssetParams forecast = params;
        forecast.viewport.w = 2.0 * atan(tan(params.viewport.w * 0.5) * params.prefetchParams.x);
        forecast.clipOrtho.z *= params.prefetchParams.x;
        forecast.lodPixelError *= params.prefetchParams.y;
        if (!streamSphereInsideFrustum(instance.boundsCenterRadius.xyz,
                max(instance.boundsCenterRadius.w, 0.000001), currentStreamCamera(forecast))) { return; }
        float3 delta = params.center.xyz - params.eye.xyz;
        float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
        float4 eye = float4(params.eye.xyz, params.clipOrtho.x);
        float4 direction = float4(forward, params.upProjection.w);
        float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5), params.clipOrtho.z, forecast.lodPixelError);
        uint table = requests[kStreamRequestPriorityTableOffsetWord];
        uint tiles = topology[primitive.lodTileOffset];
        for (uint tile = 0u; tile < tiles;) {
            // The CPU publishes only spare admission capacity. Stop searching
            // once it is filled instead of repeatedly reporting deferred work.
            if (requests[kStreamRequestPrefetchCounterWord] >= limit) { return; }
            uint offset = primitive.lodTileOffset + 1u + tile * 10u;
            if (!streamLodBvhNeedsFine(topology, offset, instance, eye, direction, projection, 0xffffffffu)) {
                tile = topology[offset + 7u];
                continue;
            }
            ++tile;
            uint count = topology[offset + 9u];
            if (count == 0u) { continue; }
            if (lane >= count) { continue; }
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + topology[offset + 8u] + count - 1u - lane];
            if ((group.flags & 1u) != 0u || group.pageIndex >= params.scenePageCount ||
                streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) { continue; }
            if (!meshletLodGroupNeedsFine(float4(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ, group.boundsRadius),
                    group.maxQuadricError, group.lodLevel, group.flags, instance.world0, instance.world1,
                    instance.world2, instance.world3, eye, direction, projection, 0xffffffffu) ||
                !streamRefinementDemandVisible(group, instance, forecast)) { continue; }
            if (table != 0u) {
                float benefit = streamGroupScreenBenefit(group, instance, forecast, eye, direction, projection);
                InterlockedMax(requests[table + group.pageIndex], asuint(benefit));
            }
            uint previous;
            InterlockedMax(pageTable[group.pageIndex].lastRequestFrame, params.frameIndex, previous);
            if (previous >= params.frameIndex) { continue; }
            uint index;
            InterlockedAdd(requests[kStreamRequestPrefetchCounterWord], 1u, index);
            if (index >= limit) {
                InterlockedAdd(requests[kStreamRequestPrefetchDroppedWord], 1u);
                continue;
            }
            for (;;) {
                uint count = requests[kStreamRequestLoadCounterWord];
                if (count >= requests[kStreamRequestMaxLoadsWord]) {
                    InterlockedAdd(requests[kStreamRequestPrefetchDroppedWord], 1u);
                    break;
                }
                uint observed;
                InterlockedCompareExchange(requests[kStreamRequestLoadCounterWord], count, count + 1u, observed);
                if (observed == count) {
                    requests[kStreamRequestHeaderWords + count] = group.pageIndex | kStreamPrefetchPageTag;
                    break;
                }
            }
        }
        return;
    }
    if (primitive.lodTileOffset == 0xffffffffu) {
        if (push.activeBuildPhase == kActiveBuildPhaseClear || push.activeBuildPhase == kActiveBuildPhaseMask) { return; }
        if (lane == 0u) {
            if (push.activeBuildPhase == kActiveBuildPhaseFrontier) {
                buildStreamLodFrontier(instanceIndex, instances, primitives, groups, topology, state, pageTable, requests, params, push);
            } else {
                emitStreamLodFrontier(instanceIndex, instances, primitives, groups, topology, state, pageTable, activeGroups, activeHeader, params);
            }
        }
        return;
    }
    uint base = topology[params.lodInstanceOffsetsOffset + instanceIndex];
    uint sparse = base + primitive.groupCount * 2u;
    if (push.activeBuildPhase == kActiveBuildPhaseMask) {
        streamCooperativeMasks(groups, topology, state, primitive, header, base, sparse,
            state[sparse], state[header + 3u] != 0u, lane);
        return;
    }
    if (push.activeBuildPhase == kActiveBuildPhaseEmit) {
        if (state[header + 3u] == 0u || activeHeader[0].padding1 != 0u) { return; }
        uint count = state[sparse];
        uint output = state[header + 1u];
        bool fallback = (state[header + 3u] & 2u) != 0u;
        for (uint first = 0u; first < count; first += 64u) {
            uint index = first + lane;
            GPUDrivenStreamAssetGroup source;
            uint mask = 0u;
            if (index < count) {
                uint localGroup = state[sparse + 4u + count - 1u - index];
                source = groups[primitive.groupOffset + localGroup];
                mask = state[base + localGroup * 2u + 1u];
                if (fallback) { mask = (source.flags & 1u) == 0u ? 0u :
                    (source.clusterCount == 32u ? 0xffffffffu : (1u << source.clusterCount) - 1u); }
            }
            uint emitted;
            uint offset = streamCooperativePrefix(mask != 0u ? 1u : 0u, lane, emitted);
            if (mask != 0u) {
                GPUDrivenStreamAssetActiveGroup group;
                group.pageDeviceOffsetBytes = streamPageDeviceOffset(pageTable[source.pageIndex]);
                group.pageIndex = source.pageIndex;
                group.clusterCount = source.clusterCount;
                group.primitiveIndex = source.primitiveIndex;
                group.lodLevel = source.lodLevel;
                group.materialIndex = instance.materialIndex;
                group.clusterSelectionMask = mask;
                group.flags = kActiveGroupResident;
                group.instanceIndex = instanceIndex;
                group.gpuSceneInstanceIndex = instance.gpuSceneInstanceIndex;
                group.padding1 = 0u; group.padding2 = 0u;
                group.world0 = instance.world0; group.world1 = instance.world1;
                group.world2 = instance.world2; group.world3 = instance.world3;
                activeGroups[output + offset] = group;
            }
            output += emitted;
        }
        return;
    }

    if (push.activeBuildPhase == kActiveBuildPhaseClear || params.splitFrontier == 0u) {
        uint previousCount = state[sparse];
        GroupMemoryBarrierWithGroupSync();
        for (uint index = lane; index < previousCount; index += 64u) {
            uint localGroup = state[sparse + 4u + index];
            state[base + localGroup * 2u] = 0u;
            state[base + localGroup * 2u + 1u] = 0u;
        }
        if (lane < 4u) { state[header + lane] = 0u; state[sparse + lane] = 0u; }
    }
    if (push.activeBuildPhase != kActiveBuildPhaseClear) {
        if (lane == 0u) { g_CooperativeRoots = 0u; g_CooperativeMissingRoots = 0u; }
        AllMemoryBarrierWithGroupSync();
    }
    if (params.enableGpuLodSelection != 0u && push.rasterBindingsBuffer != 0xffffffffu) {
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> bindings =
            DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(push.bufferShaderIndexBase + push.rasterBindingsBuffer, 0));
        if ((bindings[0].cullingFlags & 1u) != 0u && !streamSphereInsideFrustum(instance.boundsCenterRadius.xyz,
                max(instance.boundsCenterRadius.w, 0.000001) + bindings[0].displacementBound, currentStreamCamera(params))) { instance.visible = 0u; }
    }
    if (instance.visible == 0u) { return; }
    if (push.activeBuildPhase == kActiveBuildPhaseClear) {
        if (params.demandBuffer != 0xffffffffu) {
            DescriptorHandle<RWStructuredBuffer<uint>> demand = DescriptorHandle<RWStructuredBuffer<uint>>(
                uint2(push.bufferShaderIndexBase + params.demandBuffer, 0));
            DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>> tasks =
                DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetTraversalWorkItem>>(uint2(push.bufferShaderIndexBase + push.traversalWorkBuffer, 0));
            uint meta = params.demandInstanceOffsetsOffset + instanceIndex * 4u;
            uint first = topology[meta + 2u], count = topology[meta + 3u];
            float3 delta = params.center.xyz - params.eye.xyz;
            float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
            float4 eye = float4(params.eye.xyz, params.clipOrtho.x);
            float4 direction = float4(forward, params.upProjection.w);
            float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5), params.clipOrtho.z, params.lodPixelError);
            uint manual = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
            for (uint batch = 0u; batch < count; batch += 64u) {
                uint index = batch + lane, root = 0u;
                bool wanted = false;
                if (index < count) {
                    root = topology[params.demandTaskOffset + first + index];
                    wanted = streamLodBvhNeedsFine(topology, primitive.lodTileOffset + 1u + root * 10u,
                        instance, eye, direction, projection, manual);
                }
                // Task order has no cut/budget semantics. Reserve independently
                // per wave; a dispatch barrier publishes all payloads to workers.
                uint total = WaveActiveCountBits(wanted);
                uint rank = WavePrefixCountBits(wanted);
                uint firstTask = 0u;
                if (WaveIsFirstLane() && total != 0u) { InterlockedAdd(demand[16], total, firstTask); }
                firstTask = WaveReadLaneFirst(firstTask);
                if (wanted) {
                    GPUDrivenStreamAssetTraversalWorkItem work;
                    work.instanceIndex = instanceIndex; work.nodeIndex = root;
                    work.readyFrame = 0u; work.padding0 = 0u;
                    tasks[firstTask + rank] = work;
                }
            }
            // The lane that cleared this word also publishes the statistic;
            // no wave operation publishes another lane's device-memory stores.
            if (lane == 3u) { state[sparse + 3u] = count; }
        }
        return;
    }
    float3 delta = params.center.xyz - params.eye.xyz;
    float3 forward = dot(delta, delta) > 1e-12 ? normalize(delta) : float3(0.0, 0.0, -1.0);
    float4 eye = float4(params.eye.xyz, params.clipOrtho.x);
    float4 direction = float4(forward, params.upProjection.w);
    float4 projection = float4(params.viewport.z, tan(params.viewport.w * 0.5), params.clipOrtho.z, params.lodPixelError);
    uint manual = params.enableGpuLodSelection != 0u ? 0xffffffffu : params.selectedLodLevel;
    uint count = 0u, tested = 0u, visited = 0u;
    uint previousLevel = 0xffffffffu;
    uint tiles = topology[primitive.lodTileOffset];
    for (uint tile = 0u; tile < tiles;) {
        uint offset = primitive.lodTileOffset + 1u + tile * 10u;
        ++visited;
        bool visit;
        if (params.demandBuffer != 0xffffffffu) {
            DescriptorHandle<StructuredBuffer<uint>> demand = DescriptorHandle<StructuredBuffer<uint>>(
                uint2(push.bufferShaderIndexBase + params.demandBuffer, 0));
            uint bit = topology[params.demandInstanceOffsetsOffset + instanceIndex * 4u + 1u] + tile;
            visit = (demand[kStreamDemandStatsWords + (bit >> 5u)] & (1u << (bit & 31u))) != 0u;
        } else {
            visit = streamLodBvhNeedsFine(topology, offset, instance, eye, direction, projection, manual);
        }
        if (!visit) { tile = topology[offset + 7u]; continue; }
        ++tile;
        uint tileCount = topology[offset + 9u];
        if (tileCount == 0u) { continue; }
        // Ordered leaves contain one LOD level. Sibling tiles are independent;
        // only a finer level can read the active flags written by this level.
        // Prefix scans still synchronize LDS, but device state only needs to be
        // published at the level boundary (and before the final mask pass).
        uint level = topology[offset + 5u];
        if (previousLevel != 0xffffffffu && level != previousLevel) {
            AllMemoryBarrierWithGroupSync();
        }
        previousLevel = level;
        uint localGroup = topology[offset + 8u] + tileCount - 1u - min(lane, tileCount - 1u);
        bool active = false;
        if (lane < tileCount) {
            GPUDrivenStreamAssetGroup group = groups[primitive.groupOffset + localGroup];
            bool terminal = (group.flags & 1u) != 0u;
            bool parentsActive = true;
            for (uint parent = 0u; parent < group.parentCount; ++parent) {
                uint id = topology[group.parentOffset + parent];
                parentsActive = parentsActive && state[base + (id - primitive.groupOffset) * 2u] != 0u;
            }
            bool demanded;
            if (params.demandBuffer != 0xffffffffu) {
                DescriptorHandle<StructuredBuffer<uint>> demand = DescriptorHandle<StructuredBuffer<uint>>(
                    uint2(push.bufferShaderIndexBase + params.demandBuffer, 0));
                uint bit = topology[params.demandInstanceOffsetsOffset + instanceIndex * 4u] + localGroup;
                demanded = (demand[kStreamDemandStatsWords + (bit >> 5u)] & (1u << (bit & 31u))) != 0u;
            } else {
                bool desired = meshletLodGroupNeedsFine(float4(group.boundsCenterX, group.boundsCenterY, group.boundsCenterZ, group.boundsRadius),
                    group.maxQuadricError, group.lodLevel, group.flags, instance.world0, instance.world1,
                    instance.world2, instance.world3, eye, direction, projection, manual);
                demanded = terminal || (desired && streamRefinementDemandVisible(group, instance, params));
                if (demanded) {
                    if (streamPageAvailable(streamPageState(pageTable[group.pageIndex]))) { markStreamPageDemand(pageTable, params, group.pageIndex); }
                    else {
                        uint table = requests[kStreamRequestPriorityTableOffsetWord];
                        if (table != 0u) {
                            float benefit = streamGroupScreenBenefit(group, instance, params, eye, direction, projection);
                            InterlockedMax(requests[table + group.pageIndex], asuint(benefit));
                        }
                        requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
                    }
                }
            }
            bool drawable = groupPageDrawable(pageTable, params, group);
            active = parentsActive && demanded && drawable;
            if (active) { state[base + localGroup * 2u] = 1u; }
            if (terminal) {
                InterlockedAdd(g_CooperativeRoots, 1u);
                if (!drawable) { InterlockedOr(g_CooperativeMissingRoots, 1u); }
            }
        }
        uint added;
        uint rank = streamCooperativePrefix(active ? 1u : 0u, lane, added);
        if (active) { state[sparse + 4u + count + rank] = localGroup; }
        count += added;
        tested += tileCount;
    }
    AllMemoryBarrierWithGroupSync();
    if (params.splitFrontier == 0u) {
        streamCooperativeMasks(groups, topology, state, primitive, header, base, sparse,
            count, g_CooperativeMissingRoots == 0u, lane);
    }
    if (lane == 0u) {
        state[sparse] = count; state[sparse + 1u] = visited; state[sparse + 2u] = tested;
        state[header + 2u] = g_CooperativeMissingRoots == 0u ? g_CooperativeRoots : 0u;
        state[header + 3u] = g_CooperativeMissingRoots == 0u ? 1u : 0u;
        if (params.demandBuffer == 0xffffffffu && params.demandStatsBuffer != 0xffffffffu) {
            DescriptorHandle<RWStructuredBuffer<uint>> stats = DescriptorHandle<RWStructuredBuffer<uint>>(
                uint2(push.bufferShaderIndexBase + params.demandStatsBuffer, 0));
            InterlockedAdd(stats[18], tested);
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void gpuDrivenStreamAssetBuildActiveMain(uint3 dispatchThreadID : SV_DispatchThreadID, uniform GPUDrivenStreamAssetPush push)
{
    uint rowBase = dispatchThreadID.y * (65535u * 64u);
    // The final padded row for a near-32-bit state allocation can wrap the
    // flattened index. Those lanes must not clear or build the first entries.
    if (dispatchThreadID.x > 0xffffffffu - rowBase) { return; }
    uint threadIndex = dispatchThreadID.x + rowBase;
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
    if (push.activeBuildPhase >= kActiveBuildPhaseFrontier && push.activeBuildPhase <= kActiveBuildPhaseInitializeLodState) {
        DescriptorHandle<StructuredBuffer<uint>> topologyHandle =
            DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + params.lodTopologyBuffer, 0));
        DescriptorHandle<RWStructuredBuffer<uint>> stateHandle =
            DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + params.lodStateBuffer, 0));
        StructuredBuffer<uint> topology = topologyHandle;
        RWStructuredBuffer<uint> state = stateHandle;
        if (push.activeBuildPhase == kActiveBuildPhaseInitializeLodState) {
            uint wordCount, stride;
            state.GetDimensions(wordCount, stride);
            if (threadIndex < wordCount) { state[threadIndex] = 0u; }
        } else if (push.activeBuildPhase == kActiveBuildPhasePrefix) {
            prefixStreamLodFrontier(state, activeHeader, params, threadIndex);
        } else if (threadIndex < params.sceneInstanceCount) {
            if (push.activeBuildPhase == kActiveBuildPhaseFrontier) {
                buildStreamLodFrontier(threadIndex, instances, primitives, groups,
                    topology, state, pageTable, requests, params, push);
            } else {
                emitStreamLodFrontier(threadIndex, instances, primitives, groups,
                    topology, state, pageTable, activeGroups, activeHeader, params);
            }
        }
        return;
    }
    if (push.activeBuildPhase == kActiveBuildPhaseReset) {
        if (params.demandStatsBuffer != 0xffffffffu && threadIndex < kStreamDemandStatsWords) {
            DescriptorHandle<RWStructuredBuffer<uint>> stats = DescriptorHandle<RWStructuredBuffer<uint>>(
                uint2(push.bufferShaderIndexBase + params.demandStatsBuffer, 0));
            stats[threadIndex] = 0u;
        }
        if (threadIndex == 0u) {
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
                // Separate tessellation arguments: eight 16-triangle tasks per
                // cluster instead of two ordinary 64-triangle mesh workgroups.
                uint tessGroups = (draw.groupCountX / kMeshletTriangleChunkCount) * 8u;
                draw.groupCountX = min(tessGroups, 65535u);
                draw.groupCountY = max(1u, (tessGroups + 65534u) / 65535u);
                drawIndirect[1] = draw;

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
        if (threadIndex == 0u) {
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
            // Separate tessellation arguments: eight 16-triangle tasks per
            // cluster instead of two ordinary 64-triangle mesh workgroups.
            uint tessGroups = (draw.groupCountX / kMeshletTriangleChunkCount) * 8u;
            draw.groupCountX = min(tessGroups, 65535u);
            draw.groupCountY = max(1u, (tessGroups + 65534u) / 65535u);
            drawIndirect[1] = draw;
        }
        return;
    }

    if (push.activeBuildPhase == kActiveBuildPhaseSeed) {
        uint workerCount = max(params.traversalWorkerCount, 1u);
        if (threadIndex >= workerCount) {
            return;
        }
        for (uint instanceIndex = threadIndex;
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
        if (threadIndex < workerCount) {
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
        if (threadIndex >= workerCount) {
            return;
        }
        for (uint instanceIndex = threadIndex;
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
    uint workItemIndex = threadIndex;
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

    uint activeCount = min(activeHeader[0].activeGroupCount, activeHeader[0].activeGroupCapacity);

    if (push.activeBuildPhase == kBlasInputPhaseReset) {
        if (itemIndex == 0u) {
            GPUDrivenStreamAssetBlasHeader previous = (GPUDrivenStreamAssetBlasHeader)0;
            if (push.traversalPhase == 0u) { previous = blasHeader[0]; }
            GPUDrivenStreamAssetBlasHeader header;
            header.clusterReferenceCount = 0u;
            header.blasBuildCount = 0u;
            header.clusterReferenceCapacity = params.blasClusterReferenceCapacity;
            header.overflowCount = 0u;
            header.frameIndex = params.frameIndex;
            header.padding0 = push.traversalPhase != 0u || previous.padding1 != activeCount ||
                previous.padding2 != push.clasPublicationRevision ? 1u : 0u;
            header.padding1 = activeCount;
            header.padding2 = push.clasPublicationRevision;
            blasHeader[0] = header;
        }
        return;
    }

    // Stable active-group order makes this an exact comparison, with no hashes
    // or readback. Transforms do not change object-space BLAS geometry.
    if (push.activeBuildPhase == 4u) {
        if (itemIndex < activeCount) {
            DescriptorHandle<RWStructuredBuffer<uint4>> cacheHandle =
                DescriptorHandle<RWStructuredBuffer<uint4>>(uint2(push.bufferShaderIndexBase + push.blasHeaderBuffer, 0));
            RWStructuredBuffer<uint4> cache = cacheHandle;
            GPUDrivenStreamAssetActiveGroup group = activeGroups[itemIndex];
            uint4 key = uint4(group.instanceIndex, group.pageIndex, group.clusterSelectionMask, group.clusterCount);
            if (push.traversalPhase == 0u && any(cache[2u + itemIndex] != key)) {
                InterlockedOr(blasHeader[0].padding0, 1u);
            }
            cache[2u + itemIndex] = key;
        }
        return;
    }
    if (blasHeader[0].padding0 == 0u) { return; }
    if (push.activeBuildPhase == 5u) {
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

struct StreamRasterCluster
{
    GPUDrivenStreamAssetParams params;
    GPUDrivenStreamAssetActiveGroup group;
    uint pageWord;
    uint positionOffsetBytes;
    uint positionStrideWords;
    uint vertexOffset;
    uint vertexCount;
    uint triangleCount;
    uint triangleBaseByte;
    uint recordIndex;
    uint clusterIndex;
    uint instanceFlags;
};

bool loadStreamRasterCluster(uint localRecordIndex, bool testVisibility, bool publishRecord,
    GPUDrivenStreamAssetPush push, out StreamRasterCluster output, out bool retryLate)
{
    retryLate = false;
    output = (StreamRasterCluster)0;
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
    if (localRecordIndex >= params.drawTaskCount / kMeshletTriangleChunkCount) { return false; }
    uint recordCapacity = rasterBindings.visibleRecordCapacity != 0u
        ? rasterBindings.visibleRecordCapacity
        : params.drawTaskCount / kMeshletTriangleChunkCount;
    uint recordIndex = 0u;
    if (!resolveVisibilityRecordIndex(
            localRecordIndex,
            rasterBindings.visibleRecordBase,
            recordCapacity,
            recordIndex)) {
        return false;
    }

    GPUDrivenStreamAssetActiveHeader header = activeHeader[0];
    uint maxActiveGroupClusters = max(header.maxActiveGroupClusters, 1u);
    uint activeGroupIndex = localRecordIndex / maxActiveGroupClusters;
    uint clusterIndex = localRecordIndex - activeGroupIndex * maxActiveGroupClusters;
    uint activeGroupCount = min(header.activeGroupCount, header.activeGroupCapacity);
    if (activeGroupIndex >= activeGroupCount) {
        return false;
    }

    GPUDrivenStreamAssetActiveGroup group = activeGroups[activeGroupIndex];
    uint instanceFlags = 0u;
    if (rasterBindings.gpuSceneInstanceBuffer != 0xffffffffu) {
        DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>> instances =
            DescriptorHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(uint2(
                push.bufferShaderIndexBase + rasterBindings.gpuSceneInstanceBuffer, 0));
        StructuredBuffer<GPUDrivenPreviewInstance> instanceBuffer = instances;
        uint count, stride;
        instanceBuffer.GetDimensions(count, stride);
        if (group.gpuSceneInstanceIndex >= count) { return false; }
        instanceFlags = instances[group.gpuSceneInstanceIndex].identity.w;
    }
    if (group.pageIndex >= params.scenePageCount ||
        group.gpuSceneInstanceIndex == kInvalidClusterIndex ||
        clusterIndex >= group.clusterCount ||
        clusterIndex >= 32u) {
        return false;
    }
    if ((group.flags & kActiveGroupLoadRequest) != 0u) {
        if (testVisibility && clusterIndex == 0u) {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
        return false;
    }
    if ((group.flags & kActiveGroupUnloadRequest) != 0u) {
        if (testVisibility && clusterIndex == 0u) {
            requestStreamPageUnload(requests, params, group.pageIndex);
        }
        return false;
    }

    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[group.pageIndex];
    uint pageState = streamPageState(pageEntry);
    uint pageDeviceOffset = streamPageDeviceOffset(pageEntry);
    if (((group.clusterSelectionMask >> clusterIndex) & 1u) == 0u ||
        ((pageState != kStreamStateResident && pageState != kStreamStateLockedFallback) ||
        pageDeviceOffset == kInvalidStreamDeviceOffset)) {
        if (testVisibility && clusterIndex == 0u) {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
        return false;
    }
    if (params.pageBufferBytes < kPayloadHeaderBytes ||
        pageDeviceOffset > params.pageBufferBytes - kPayloadHeaderBytes) {
        return false;
    }

    uint pageWord = pageDeviceOffset / 4u;
    uint payloadBytes = pageWords[pageWord + 12u];
    if (payloadBytes < kPayloadHeaderBytes ||
        payloadBytes > params.pageBufferBytes ||
        pageDeviceOffset > params.pageBufferBytes - payloadBytes) {
        return false;
    }
    uint clusterCount = pageWords[pageWord + 2u];
    uint payloadVertexCount = pageWords[pageWord + 3u];
    uint triangleIndexCount = pageWords[pageWord + 4u];
    uint attributeFlags = pageWords[pageWord + 14u];
    uint positionFormat = pageWords[pageWord + 20u];
    if (clusterIndex >= clusterCount ||
        (attributeFlags & kPayloadAttributePosition) == 0u ||
        (positionFormat != kPayloadFormatFloat32x4 && positionFormat != kPayloadFormatFloat32x3)) {
        return false;
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
            positionFormat == kPayloadFormatFloat32x3 ? 12u : 16u,
            payloadBytes) ||
        !streamPayloadRangeValid(
            triangleOffsetBytes,
            triangleIndexCount,
            payloadBytes)) {
        return false;
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
        return false;
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
    float worldScale = conservativeTransformScale(group.world0.xyz, group.world1.xyz, group.world2.xyz);
    float worldSphereRadius = localSphereRadius * worldScale + rasterBindings.displacementBound;
    GPUDrivenStreamAssetCameraData rasterCamera = currentStreamCamera(params);
    if (testVisibility && (rasterBindings.cullingFlags & (1u << 2u)) != 0u &&
        !streamSphereInsideFrustum(
            worldSphereCenter,
            worldSphereRadius,
            rasterCamera)) {
        return false;
    }

    float coneCutoff = asfloat(pageWords[clusterWord + 19u]);
    if (testVisibility && rasterBindings.displacementBound == 0.0 && (instanceFlags & 2u) == 0u &&
        (rasterBindings.cullingFlags & (1u << 3u)) != 0u && coneCutoff < 0.9999) {
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
        // Cofactors include the determinant sign. Match the corrected raster
        // winding without changing authored normals used by material shading.
        if (dot(cross(group.world0.xyz, group.world1.xyz), group.world2.xyz) < 0.0) {
            worldConeAxis = -worldConeAxis;
        }
        float3 viewDirection = safeNormalize(
            worldConeApex - rasterCamera.eye.xyz,
            -worldConeAxis);
        if (dot(viewDirection, worldConeAxis) >= coneCutoff) {
            return false;
        }
    }

    DescriptorHandle<StructuredBuffer<uint>> visibilityStateHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(
            push.bufferShaderIndexBase + rasterBindings.instanceVisibilityBuffer,
            0));
    StructuredBuffer<uint> visibilityStates = visibilityStateHandle;
    if (testVisibility && !streamClusterVisibleInOcclusionPhase(worldSphereCenter, worldSphereRadius,
        visibilityStates[group.gpuSceneInstanceIndex], params, rasterBindings, push, retryLate)) {
        return false;
    }

    DescriptorHandle<RWStructuredBuffer<VisibleClusterRecord>> visibleClusterHandle =
        DescriptorHandle<RWStructuredBuffer<VisibleClusterRecord>>(uint2(
            push.bufferShaderIndexBase + rasterBindings.visibleClusterBuffer,
            0));
    RWStructuredBuffer<VisibleClusterRecord> visibleClusters = visibleClusterHandle;
    if (publishRecord) {
        VisibleClusterRecord record;
        record.clusterIndex = clusterIndex;
        record.instanceIndex = group.gpuSceneInstanceIndex;
        record.dataIndex = activeGroupIndex;
        record.flags = kVisibleClusterSourceStreamPage << kVisibleClusterSourceShift;
        record.flags |= ((instanceFlags & 2u) != 0u ? 1u : 0u) | ((instanceFlags & 4u) != 0u ? 2u : 0u);
        visibleClusters[localRecordIndex] = record;
    }

    output.params = params;
    output.group = group;
    output.pageWord = pageWord;
    output.positionOffsetBytes = positionOffsetBytes;
    output.positionStrideWords = positionFormat == kPayloadFormatFloat32x3 ? 3u : 4u;
    output.vertexOffset = vertexOffset;
    output.vertexCount = vertexCount;
    output.triangleCount = clusterTriangleCount;
    output.triangleBaseByte = pageDeviceOffset + triangleOffsetBytes + triangleOffset;
    output.recordIndex = recordIndex;
    output.clusterIndex = clusterIndex;
    output.instanceFlags = instanceFlags;
    return true;
}

float4 streamClusterClip(StreamRasterCluster c, uint vertex, GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<uint>> pages =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.pageBuffer, 0));
    uint word = c.pageWord + c.positionOffsetBytes / 4u + (c.vertexOffset + vertex) * c.positionStrideWords;
    float3 local = float3(asfloat(pages[word]), asfloat(pages[word + 1u]), asfloat(pages[word + 2u]));
    float3 world = c.group.world0.xyz * local.x + c.group.world1.xyz * local.y +
        c.group.world2.xyz * local.z + c.group.world3.xyz;
    float4 clip = projectPosition(world, c.params);
    clip.y = -clip.y;
    return clip;
}

uint3 streamClusterTriangle(StreamRasterCluster c, uint triangle, GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<uint>> pages =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.pageBuffer, 0));
    uint address = c.triangleBaseByte + triangle * 3u;
    uint3 indices = uint3(readPageByte(pages, address), readPageByte(pages, address + 1u), readPageByte(pages, address + 2u));
    // Retain the existing malformed-index fallback in the stream HW decoder.
    indices = uint3(indices.x < c.vertexCount ? indices.x : 0u,
        indices.y < c.vertexCount ? indices.y : 0u, indices.z < c.vertexCount ? indices.z : 0u);
    // A reflected instance reverses projected winding. Keep authored front
    // faces front-facing in both hardware and software rasterization; the
    // visibility ID still names the original payload triangle for shading.
    bool reflected = dot(cross(c.group.world0.xyz, c.group.world1.xyz), c.group.world2.xyz) < 0.0;
    return reflected ? indices.xzy : indices;
}

groupshared StreamRasterCluster g_StreamMeshCluster;
groupshared uint g_StreamMeshValid;

// Prebinned clusters emit their complete indexed mesh in one workgroup.
// The legacy unbinned queue path retains its two 64-triangle draw slots.
groupshared float4 g_StreamMeshClip[kMaxMeshletVertices];

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void gpuDrivenStreamAssetMeshMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices GPUDrivenStreamAssetVertexOutput vertices[kMaxMeshletVertices],
    out indices uint3 triangles[kMaxMeshletTriangles],
    out primitives GPUDrivenStreamAssetPrimitiveOutput primitives[kMaxMeshletTriangles],
    uniform GPUDrivenStreamAssetPush push)
{
    uint draw = hybridFlatGroup(groupID);
    bool prebinned = push.hybridClusterBuffer != 0xffffffffu;
    uint record = prebinned ? draw : draw / kMeshletTriangleChunkCount;
    uint chunk = prebinned ? 0u : draw % kMeshletTriangleChunkCount;
    if (prebinned) {
        DescriptorHandle<StructuredBuffer<uint>> bins =
            DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
        uint item = record;
        if (!hybridBinRecord(bins, 0u, item, record)) { SetMeshOutputCounts(0, 0); return; }
    }
    if (lane == 0u) {
        bool retryLate;
        g_StreamMeshValid = loadStreamRasterCluster(record, !prebinned, !prebinned && chunk == 0u,
            push, g_StreamMeshCluster, retryLate) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamMeshValid == 0u) { SetMeshOutputCounts(0, 0); return; }
    StreamRasterCluster c = g_StreamMeshCluster;
    uint start = chunk * kMeshletTriangleChunkSize;
    if (start >= c.triangleCount) { SetMeshOutputCounts(0, 0); return; }
    uint count = prebinned ? c.triangleCount : min(c.triangleCount - start, kMeshletTriangleChunkSize);
    SetMeshOutputCounts(c.vertexCount, count);
    bool queueTriangles = !prebinned && push.hybridQueueBuffer != 0xffffffffu;
    for (uint vertex = lane; vertex < c.vertexCount; vertex += 64u) {
        float4 clip = streamClusterClip(c, vertex, push);
        vertices[vertex].position = clip;
        vertices[vertex].doubleSided = (c.instanceFlags & 2u) != 0u;
        if (queueTriangles) { g_StreamMeshClip[vertex] = clip; }
    }
    if (queueTriangles) { GroupMemoryBarrierWithGroupSync(); }
    for (uint i = lane; i < count; i += 64u) {
        uint3 indices = streamClusterTriangle(c, start + i, push);
        uint id = packVisibilityId(c.recordIndex, start + i);
        bool queued = false;
        if (queueTriangles) {
            queued = hybridEnqueueTriangle(g_StreamMeshClip[indices.x], g_StreamMeshClip[indices.y],
                g_StreamMeshClip[indices.z], id, (c.instanceFlags & 2u) != 0u,
                push.hybridQueueBuffer, push.bufferShaderIndexBase);
        }
        triangles[i] = queued ? uint3(0u) : indices;
        primitives[i].visibilityId = id;
    }
}

groupshared StreamRasterCluster g_StreamRasterCluster;
groupshared float4 g_StreamRasterClip[128];
groupshared uint g_StreamBinVisible;
groupshared uint g_StreamBinHardware;

groupshared uint g_StreamCandidatePrefix[128];

uint streamPhaseCandidateMask(GPUDrivenStreamAssetActiveGroup group, uint index,
    RWStructuredBuffer<uint> bins, StructuredBuffer<uint> visibility, bool early)
{
    uint state = visibility[group.gpuSceneInstanceIndex];
    if (early) {
        bins[hybridRetryBase(bins[5]) + index] = 0u;
        return state == 1u ? group.clusterSelectionMask : 0u;
    }
    if (state == 3u) { return group.clusterSelectionMask; }
    return state == 1u ? group.clusterSelectionMask & bins[hybridRetryBase(bins[5]) + index] : 0u;
}

uint streamCandidateInclusiveScan(uint value, uint lane)
{
    g_StreamCandidatePrefix[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 128u; stride *= 2u) {
        uint previous = lane >= stride ? g_StreamCandidatePrefix[max(lane, stride) - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        g_StreamCandidatePrefix[lane] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    return g_StreamCandidatePrefix[lane];
}

// Setup/count/prefix/scatter use activeBuildPhase, independently of the
// early/late traversalPhase. Count and scatter scale with the actual active
// group count, not the capacity. Original group/cluster order and IDs survive.
// Before classification, bin 0 stores group masks, rank storage holds each
// group's block-local offset, and block storage holds counts/global offsets.
// All three regions are dead until stable binning, which overwrites them.
[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterPrepareMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>> headers =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(uint2(push.bufferShaderIndexBase + push.activeHeaderBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>> groups =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(push.bufferShaderIndexBase + push.activeGroupBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> bins =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> args =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridQueueBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> bindingHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(push.bufferShaderIndexBase + push.rasterBindingsBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> visibility =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + bindingHandle[0].instanceVisibilityBuffer, 0));
    uint capacity = bins[5];
    if (push.activeBuildPhase == 0u) {
        if (lane == 0u) {
            GPUDrivenStreamAssetActiveHeader header = headers[0];
            // The stream producer guarantees activeGroupCapacity <= capacity.
            uint count = min(header.activeGroupCount, header.activeGroupCapacity);
            bins[13] = header.maxActiveGroupClusters;
            bins[15] = count;
            uint blocks = (count + 127u) / 128u;
            args[6] = min(blocks, kHybridDispatchWidth);
            args[7] = max(1u, (blocks + kHybridDispatchWidth - 1u) / kHybridDispatchWidth);
            args[8] = 1u;
        }
        return;
    }

    uint count = bins[15];
    uint blocks = (count + 127u) / 128u;
    uint blockBase = hybridBlockBase(capacity);
    if (push.activeBuildPhase == 2u) {
        uint perLane = (blocks + 127u) / 128u;
        uint first = min(lane * perLane, blocks), end = min(first + perLane, blocks);
        uint total = 0u;
        for (uint block = first; block < end; ++block) { total += bins[blockBase + block]; }
        uint offset = streamCandidateInclusiveScan(total, lane) - total;
        for (uint block = first; block < end; ++block) {
            uint blockCount = bins[blockBase + block];
            bins[blockBase + block] = offset;
            offset += blockCount;
        }
        if (lane == 127u) {
            uint candidates = min(g_StreamCandidatePrefix[127], capacity);
            bins[12] = candidates;
            bins[14] = g_StreamCandidatePrefix[127] - candidates;
            args[0] = min(candidates, kHybridDispatchWidth);
            args[1] = max(1u, (candidates + kHybridDispatchWidth - 1u) / kHybridDispatchWidth);
            args[2] = 1u;
            uint candidateBlocks = (candidates + 127u) / 128u;
            args[3] = min(candidateBlocks, kHybridDispatchWidth);
            args[4] = max(1u, (candidateBlocks + kHybridDispatchWidth - 1u) / kHybridDispatchWidth);
            args[5] = 1u;
        }
        return;
    }

    uint block = hybridFlatGroup(groupID);
    if (block >= blocks) { return; }
    uint index = block * 128u + lane;
    if (push.activeBuildPhase == 1u) {
        uint mask = 0u;
        if (index < count) {
            mask = streamPhaseCandidateMask(groups[index], index, bins, visibility, push.traversalPhase == 0u);
        }
        uint localCount = countbits(mask);
        uint offset = streamCandidateInclusiveScan(localCount, lane) - localCount;
        if (index < count) {
            bins[kHybridBinHeaderWords + index] = mask;
            bins[hybridRankBase(capacity) + index] = offset;
        }
        if (lane == 127u) { bins[blockBase + block] = g_StreamCandidatePrefix[127]; }
    } else if (push.activeBuildPhase == 3u && index < count) {
        uint mask = bins[kHybridBinHeaderWords + index];
        uint offset = bins[blockBase + block] + bins[hybridRankBase(capacity) + index];
        while (mask != 0u) {
            uint cluster = firstbitlow(mask);
            if (offset < capacity) {
                uint slot = hybridCandidateBase(capacity) + offset * 2u;
                bins[slot] = index * bins[13] + cluster;
                bins[slot + 1u] = 0xffffffffu;
            }
            ++offset;
            mask &= mask - 1u;
        }
    }
}

// One lane validates/culls one candidate. Survivors carry only the geometry
// decode fields needed by classification, in the still-unused bin list storage.
// A wave reserves one range; its scheduling order is intentionally irrelevant:
// classifiers write back to original candidate slots before stable binning.
[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterCullMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<RWStructuredBuffer<uint>> handle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
    RWStructuredBuffer<uint> bins = handle;
    if (push.activeBuildPhase == 1u) {
        if (lane == 0u) {
            DescriptorHandle<RWStructuredBuffer<uint>> args =
                DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridQueueBuffer, 0));
            uint count = bins[0];
            args[0] = min(count, kHybridDispatchWidth);
            args[1] = max(1u, (count + kHybridDispatchWidth - 1u) / kHybridDispatchWidth);
            args[2] = 1u;
        }
        return;
    }
    uint candidate = hybridFlatGroup(group) * 128u + lane;
    StreamRasterCluster cluster = (StreamRasterCluster)0;
    bool visible = false;
    if (candidate < bins[12]) {
        uint record = bins[hybridCandidateBase(bins[5]) + candidate * 2u];
        bool retryLate;
        visible = loadStreamRasterCluster(record, true, true, push, cluster, retryLate) && cluster.triangleCount != 0u;
        if (push.traversalPhase == 0u && retryLate) {
            InterlockedOr(bins[hybridRetryBase(bins[5]) + record / bins[13]], 1u << (record % bins[13]));
        }
    }
    uint count = WaveActiveCountBits(visible);
    uint offset = WavePrefixCountBits(visible);
    uint base = 0u;
    if (WaveIsFirstLane() && count != 0u) { InterlockedAdd(bins[0], count, base); }
    base = WaveReadLaneFirst(base);
    if (visible) {
        uint item = hybridStreamClassifyBase(base + offset);
        bins[item] = candidate;
        bins[item + 1u] = cluster.pageWord + cluster.positionOffsetBytes / 4u + cluster.vertexOffset * cluster.positionStrideWords;
        bins[item + 2u] = cluster.vertexCount | (cluster.triangleCount << 16u) | (cluster.positionStrideWords == 3u ? 0x80000000u : 0u);
        bins[item + 3u] = cluster.triangleBaseByte;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterBinMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<RWStructuredBuffer<uint>> handle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
    RWStructuredBuffer<uint> bins = handle;
    uint work = hybridFlatGroup(group);
    if (work >= bins[0]) { return; }
    uint item = hybridStreamClassifyBase(work);
    uint candidate = bins[item];
    uint record = bins[hybridCandidateBase(bins[5]) + candidate * 2u];
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>> raster = DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(uint2(push.bufferShaderIndexBase + push.rasterBindingsBuffer, 0));
    if (raster[0].tessellationBuffer != 0xffffffffu) {
        if (lane == 0u) { hybridClassifyCluster(bins, 0u, record, candidate); }
        return;
    }
    if (lane == 0u) {
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> params =
            DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>> groups =
            DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(uint2(push.bufferShaderIndexBase + push.activeGroupBuffer, 0));
        g_StreamRasterCluster = (StreamRasterCluster)0;
        g_StreamRasterCluster.params = params[0];
        g_StreamRasterCluster.group = groups[record / bins[13]];
        g_StreamRasterCluster.pageWord = bins[item + 1u];
        g_StreamRasterCluster.vertexCount = bins[item + 2u] & 0xffffu;
        g_StreamRasterCluster.triangleCount = (bins[item + 2u] >> 16u) & 0x7fffu;
        g_StreamRasterCluster.positionStrideWords = (bins[item + 2u] & 0x80000000u) != 0u ? 3u : 4u;
        g_StreamRasterCluster.triangleBaseByte = bins[item + 3u];
        g_StreamBinHardware = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < g_StreamRasterCluster.vertexCount) { g_StreamRasterClip[lane] = streamClusterClip(g_StreamRasterCluster, lane, push); }
    GroupMemoryBarrierWithGroupSync();
    bool hardware = false;
    if (lane < g_StreamRasterCluster.triangleCount) {
        uint3 tri = streamClusterTriangle(g_StreamRasterCluster, lane, push);
        hardware = !hybridTriangleFits(g_StreamRasterClip[tri.x], g_StreamRasterClip[tri.y], g_StreamRasterClip[tri.z],
            float2(bins[6], bins[7]), asfloat(bins[8]));
    }
    bool waveHardware = WaveActiveAnyTrue(hardware);
    if (WaveIsFirstLane() && waveHardware) { InterlockedOr(g_StreamBinHardware, 1u); }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0u) { hybridClassifyCluster(bins, g_StreamBinHardware != 0u ? 0u : kHybridSoftwareBin, record, candidate); }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void streamClusterRasterMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform GPUDrivenStreamAssetPush push)
{
    DescriptorHandle<StructuredBuffer<uint>> bins =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
    uint record;
    if (!hybridBinRecord(bins, kHybridSoftwareBin, hybridFlatGroup(group), record)) { return; }
    if (lane == 0u) { bool retryLate; g_StreamBinVisible = loadStreamRasterCluster(record, false, false, push, g_StreamRasterCluster, retryLate) ? 1u : 0u; }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamBinVisible == 0u) { return; }
    if (lane < g_StreamRasterCluster.vertexCount) { g_StreamRasterClip[lane] = streamClusterClip(g_StreamRasterCluster, lane, push); }
    GroupMemoryBarrierWithGroupSync();
    if (lane < g_StreamRasterCluster.triangleCount) {
        uint3 tri = streamClusterTriangle(g_StreamRasterCluster, lane, push);
        hybridRasterTriangle(g_StreamRasterClip[tri.x], g_StreamRasterClip[tri.y], g_StreamRasterClip[tri.z],
            packVisibilityId(g_StreamRasterCluster.recordIndex, lane), (g_StreamRasterCluster.instanceFlags & 2u) != 0u,
            bins[6], bins[7], bins[9], bins[10], bins[11], push.bufferShaderIndexBase);
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

float3 loadPayloadPosition(StructuredBuffer<uint> words, uint pageWord, uint offsetBytes, uint index, uint format)
{
    uint base = pageWord + offsetBytes / 4u + index * (format == kPayloadFormatFloat32x3 ? 3u : 4u);
    return asfloat(uint3(words[base], words[base + 1u], words[base + 2u]));
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
        (positionFormat != kPayloadFormatFloat32x4 && positionFormat != kPayloadFormatFloat32x3) ||
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
            positionFormat == kPayloadFormatFloat32x3 ? 12u : 16u,
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
    float3 local0 = loadPayloadPosition(pageWords, pageWord, positionOffsetBytes, vertexIndices.x, positionFormat);
    float3 local1 = loadPayloadPosition(pageWords, pageWord, positionOffsetBytes, vertexIndices.y, positionFormat);
    float3 local2 = loadPayloadPosition(pageWords, pageWord, positionOffsetBytes, vertexIndices.z, positionFormat);
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
uint gpuDrivenStreamAssetFragmentMain(GPUDrivenStreamAssetVertexOutput input, uint visibilityId : SV_PrimitiveID,
    bool frontFace : SV_IsFrontFace,
    uniform GPUDrivenStreamAssetPush push) : SV_Target0
{
    if (!frontFace && input.doubleSided == 0u) { discard; }
    return visibilityId;
}

#include "../VisibilityBuffer/VisibilityTessellationStream.slang"
