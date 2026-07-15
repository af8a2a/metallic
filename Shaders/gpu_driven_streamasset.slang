static const uint kMaxMeshletVertices = 128;
static const uint kMaxMeshletTriangles = 128;
static const uint kInvalidStreamDeviceOffset = 0xffffffffu;
static const uint kInvalidClusterIndex = 0xffffffffu;
static const uint kUnloadClusterIndex = 0xfffffffeu;
static const uint kStreamStatePendingUpload = 1u;
static const uint kStreamStateResident = 2u;
static const uint kStreamStateLockedFallback = 3u;
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
    uint updateBuffer;
    uint activeHeaderBuffer;
    uint instanceBuffer;
    uint primitiveBuffer;
    uint lodLevelBuffer;
    uint pageInfoBuffer;
    uint groupBuffer;
    uint clusterRefBuffer;
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
    uint blasClusterReferenceAddressLow;
    uint blasClusterReferenceAddressHigh;
    uint blasClusterReferenceCapacity;
    uint traversalPhase;
    uint activeBuildPhase;
    uint blasInputPhase;
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
    uint padding0;
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
    uint padding0;
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

struct GPUDrivenStreamAssetPageInfo
{
    uint primitiveIndex;
    uint lodLevel;
    uint pageIndex;
    uint clusterCount;
};

struct GPUDrivenStreamAssetGroup
{
    uint primitiveIndex;
    uint pageIndex;
    uint lodLevel;
    uint clusterRefOffset;
    uint clusterCount;
    uint padding0;
    uint padding1;
    uint padding2;
    float4 boundsCenterRadius;
    float maxQuadricError;
    uint padding3;
    uint padding4;
    uint padding5;
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
    uint addressOffset;
    uint clusterCount;
    uint state;
    uint generation;
};

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

struct GPUDrivenStreamAssetPageTableEntry
{
    uint deviceOffsetBytes;
    uint deviceSizeBytes;
    uint state;
    uint lastRequestFrame;
    uint lodLevel;
    uint payloadBytes;
    uint padding0;
    uint padding1;
};

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

struct GPUDrivenStreamAssetPatch
{
    uint pageId;
    uint deviceOffsetBytes;
    uint deviceSizeBytes;
    uint state;
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
};

struct GPUDrivenStreamAssetVertexOutput
{
    float4 position : SV_Position;
    nointerpolation float3 color : COLOR0;
};

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
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
    if (params.clipOrtho.w > 0.5) {
        ndcZ = 1.0 - ndcZ;
    }
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
    float3 localCenter = group.boundsCenterRadius.xyz;
    float3 worldCenter =
        instance.world0.xyz * localCenter.x +
        instance.world1.xyz * localCenter.y +
        instance.world2.xyz * localCenter.z +
        instance.world3.xyz;
    float uniformScale = instanceUniformScale(instance);
    float worldRadius = max(group.boundsCenterRadius.w * uniformScale, 0.000001);
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
    uint state = pageTable[group.pageIndex].state;
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
    return streamPageDrawable(pageEntry.state) &&
        pageEntry.deviceOffsetBytes != kInvalidStreamDeviceOffset &&
        pageEntry.deviceSizeBytes != 0u &&
        pageEntry.payloadBytes != 0u;
}

uint buildGroupClusterSelectionMask(
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> clusterRefs,
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

    uint mask = 0u;
    for (uint clusterChild = 0u; clusterChild < 32u; ++clusterChild) {
        if (clusterChild >= clusterCount) {
            break;
        }
        uint refinedGroupIndex = clusterRefs[group.clusterRefOffset + clusterChild];
        bool renderCluster = refinedGroupIndex == kInvalidClusterIndex;
        if (!renderCluster) {
            if (refinedGroupIndex >= params.sceneGroupCount) {
                renderCluster = true;
            } else {
                GPUDrivenStreamAssetGroup refinedGroup = groups[refinedGroupIndex];
                renderCluster = !groupPageDrawable(pageTable, params, refinedGroup) ||
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
    StructuredBuffer<GPUDrivenStreamAssetPageInfo> pageInfos,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    uint instanceIndex,
    uint pageIndex,
    uint clusterSelectionMask)
{
    if (pageIndex >= params.scenePageCount) {
        return;
    }

    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[pageIndex];
    if (!streamPageDrawable(pageEntry.state) ||
        pageEntry.deviceOffsetBytes == kInvalidStreamDeviceOffset ||
        pageEntry.deviceSizeBytes == 0u ||
        pageEntry.payloadBytes == 0u) {
        return;
    }

    uint activeIndex = 0u;
    InterlockedAdd(activeHeader[0].activeGroupCount, 1u, activeIndex);
    if (activeIndex >= activeHeader[0].activeGroupCapacity) {
        InterlockedAdd(activeHeader[0].overflowCount, 1u);
        return;
    }

    GPUDrivenStreamAssetPageInfo pageInfo = pageInfos[pageIndex];
    GPUDrivenStreamAssetActiveGroup group;
    group.pageDeviceOffsetBytes = pageEntry.deviceOffsetBytes;
    group.pageIndex = pageIndex;
    group.clusterCount = pageInfo.clusterCount;
    group.primitiveIndex = pageInfo.primitiveIndex;
    group.lodLevel = pageInfo.lodLevel;
    group.materialIndex = instance.materialIndex;
    group.clusterSelectionMask = clusterSelectionMask;
    group.flags = kActiveGroupResident;
    group.instanceIndex = instanceIndex;
    group.padding0 = 0u;
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
            !streamPageDrawable(pageTable[pageIndex].state)) {
            return false;
        }
    }
    return true;
}

void appendActivePageRange(
    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups,
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader,
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable,
    StructuredBuffer<GPUDrivenStreamAssetPageInfo> pageInfos,
    GPUDrivenStreamAssetParams params,
    GPUDrivenStreamAssetInstance instance,
    uint instanceIndex,
    uint pageOffset,
    uint pageCount)
{
    for (uint page = 0u; page < pageCount; ++page) {
        appendActivePage(
            activeGroups,
            activeHeader,
            pageTable,
            pageInfos,
            params,
            instance,
            instanceIndex,
            pageOffset + page,
            0xffffffffu);
    }
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

        uint state = pageTable[group.pageIndex].state;
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
    StructuredBuffer<GPUDrivenStreamAssetPageInfo> pageInfos,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> clusterRefs,
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
            clusterRefs,
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
                pageInfos,
                params,
                instance,
                instanceIndex,
                group.pageIndex,
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
    StructuredBuffer<GPUDrivenStreamAssetPageInfo> pageInfos,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> clusterRefs,
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
        if (!updateGroupPageDemand(pageTable, requests, params, instance, primitive, group) ||
            !groupPageDrawable(pageTable, params, group)) {
            continue;
        }
        uint clusterSelectionMask = buildGroupClusterSelectionMask(
            groups,
            clusterRefs,
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
                pageInfos,
                params,
                instance,
                instanceIndex,
                group.pageIndex,
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
    StructuredBuffer<GPUDrivenStreamAssetPageInfo> pageInfos,
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups,
    StructuredBuffer<uint> clusterRefs,
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
                                    clusterRefs,
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
                                        pageInfos,
                                        params,
                                        instance,
                                        work.instanceIndex,
                                        group.pageIndex,
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
        uint pageIndex = dispatchThreadID.x;
        if (pageIndex >= params.scenePageCount || params.enableGpuUnloadRequests == 0u) {
            return;
        }
        GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[pageIndex];
        if (pageEntry.state == kStreamStateResident && pageEntry.lastRequestFrame != params.frameIndex) {
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
    uint state = pageTable[group.pageIndex].state;
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
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPageInfo>> pageInfoHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetPageInfo>>(uint2(push.bufferShaderIndexBase + push.pageInfoBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>> groupHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetGroup>>(uint2(push.bufferShaderIndexBase + push.groupBuffer, 0));
    DescriptorHandle<StructuredBuffer<uint>> clusterRefHandle =
        DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.clusterRefBuffer, 0));
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

    RWStructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader = activeHeaderHandle;
    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives = primitiveHandle;
    StructuredBuffer<GPUDrivenStreamAssetLodLevel> lodLevels = lodLevelHandle;
    StructuredBuffer<GPUDrivenStreamAssetPageInfo> pageInfos = pageInfoHandle;
    StructuredBuffer<GPUDrivenStreamAssetGroup> groups = groupHandle;
    StructuredBuffer<uint> clusterRefs = clusterRefHandle;
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
            draw.groupCountX = min(activeCount * header.maxActiveGroupClusters, params.drawTaskCount);
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
                pageInfos,
                groups,
                clusterRefs,
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
                pageInfos,
                groups,
                clusterRefs,
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
                    pageInfos,
                    groups,
                    clusterRefs,
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
        clusterRefs,
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
            pageInfos,
            params,
            instance,
            instanceIndex,
            group.pageIndex,
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

    if (push.blasInputPhase == kBlasInputPhaseReset) {
        if (itemIndex == 0u) {
            GPUDrivenStreamAssetBlasHeader header;
            header.clusterReferenceCount = 0u;
            header.blasBuildCount = 0u;
            header.clusterReferenceCapacity = push.blasClusterReferenceCapacity;
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
    if (push.blasInputPhase == kBlasInputPhaseCount) {
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
        if (page.state != kClasPageStateActive ||
            page.addressOffset == kInvalidClusterIndex ||
            group.clusterCount > page.clusterCount) {
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

    if (push.blasInputPhase == kBlasInputPhaseSetup) {
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
            info.clusterReferenceOffset >= push.blasClusterReferenceCapacity ||
            info.selectedClusterCount > push.blasClusterReferenceCapacity - info.clusterReferenceOffset) {
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
        uint referenceAddressLow = push.blasClusterReferenceAddressLow + byteAddressLow;
        uint carry = referenceAddressLow < push.blasClusterReferenceAddressLow ? 1u : 0u;
        GPUDrivenStreamAssetBlasBuildInfo buildInfo;
        buildInfo.clusterReferencesCount = info.selectedClusterCount;
        buildInfo.clusterReferencesStride = 8u;
        buildInfo.clusterReferencesAddressLow = referenceAddressLow;
        buildInfo.clusterReferencesAddressHigh =
            push.blasClusterReferenceAddressHigh + byteAddressHigh + carry;
        blasBuildInfos[buildIndex] = buildInfo;
        instanceBlas[itemIndex].blasBuildIndex = buildIndex;
        InterlockedOr(instanceBlas[itemIndex].flags, kBlasInstanceDynamic);
        return;
    }

    if (push.blasInputPhase != kBlasInputPhaseInsert || itemIndex >= activeCount) {
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
    if (page.state != kClasPageStateActive ||
        page.addressOffset == kInvalidClusterIndex ||
        group.clusterCount > page.clusterCount) {
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
            clasAddresses[page.addressOffset + clusterIndex];
    }
}

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void gpuDrivenStreamAssetMeshMain(
    uint3 groupID : SV_GroupID,
    out vertices GPUDrivenStreamAssetVertexOutput vertices[kMaxMeshletVertices],
    out indices uint3 triangles[kMaxMeshletTriangles],
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

    StructuredBuffer<uint> pageWords = pageHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader = activeHeaderHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    RWStructuredBuffer<uint> requests = requestHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    uint drawIndex = groupID.x;
    if (drawIndex >= params.drawTaskCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    GPUDrivenStreamAssetActiveHeader header = activeHeader[0];
    uint maxActiveGroupClusters = max(header.maxActiveGroupClusters, 1u);
    uint activeGroupIndex = drawIndex / maxActiveGroupClusters;
    uint clusterIndex = drawIndex - activeGroupIndex * maxActiveGroupClusters;
    uint activeGroupCount = min(header.activeGroupCount, header.activeGroupCapacity);
    if (activeGroupIndex >= activeGroupCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    GPUDrivenStreamAssetActiveGroup group = activeGroups[activeGroupIndex];
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
    if (clusterIndex >= group.clusterCount ||
        clusterIndex >= 32u ||
        ((group.clusterSelectionMask >> clusterIndex) & 1u) == 0u ||
        ((pageEntry.state != kStreamStateResident && pageEntry.state != kStreamStateLockedFallback) ||
        pageEntry.deviceOffsetBytes == kInvalidStreamDeviceOffset ||
        pageEntry.deviceSizeBytes == 0u ||
        pageEntry.payloadBytes == 0u)) {
        if (clusterIndex == 0u) {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
        SetMeshOutputCounts(0, 0);
        return;
    }
    if (pageEntry.deviceOffsetBytes + pageEntry.payloadBytes < pageEntry.deviceOffsetBytes ||
        pageEntry.deviceOffsetBytes + pageEntry.payloadBytes > params.pageBufferBytes ||
        pageEntry.payloadBytes > pageEntry.deviceSizeBytes) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    uint pageWord = pageEntry.deviceOffsetBytes / 4u;
    uint clusterCount = pageWords[pageWord + 2u];
    if (clusterIndex >= clusterCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    uint clusterOffsetBytes = pageWords[pageWord + 9u];
    uint positionOffsetBytes = pageWords[pageWord + 10u];
    uint triangleOffsetBytes = pageWords[pageWord + 11u];
    uint clusterWord = pageWord + clusterOffsetBytes / 4u + clusterIndex * 8u;
    uint vertexOffset = pageWords[clusterWord + 0u];
    uint vertexCount = min(pageWords[clusterWord + 1u], kMaxMeshletVertices);
    uint triangleOffset = pageWords[clusterWord + 2u];
    uint triangleCount = min(pageWords[clusterWord + 3u], kMaxMeshletTriangles);

    SetMeshOutputCounts(vertexCount, triangleCount);

    uint colorSeed = group.pageIndex;
    if (params.debugColorMode == 1u) {
        colorSeed = group.lodLevel;
    } else if (params.debugColorMode == 2u) {
        colorSeed = group.primitiveIndex;
    } else if (params.debugColorMode == 3u) {
        colorSeed = group.instanceIndex;
    } else if (params.debugColorMode == 4u) {
        colorSeed = ((group.pageIndex + 1u) * 0x9e3779b9u) ^ clusterIndex;
    }
    float3 color = hashColor(colorSeed + 1u);

    for (uint vertexIndex = 0; vertexIndex < kMaxMeshletVertices; ++vertexIndex) {
        if (vertexIndex < vertexCount) {
            uint positionWord = pageWord + positionOffsetBytes / 4u + (vertexOffset + vertexIndex) * 4u;
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
            vertices[vertexIndex].color = color;
        }
    }

    uint triangleBaseByte = pageEntry.deviceOffsetBytes + triangleOffsetBytes + triangleOffset;
    for (uint triangleIndex = 0; triangleIndex < kMaxMeshletTriangles; ++triangleIndex) {
        if (triangleIndex < triangleCount) {
            uint baseByte = triangleBaseByte + triangleIndex * 3u;
            triangles[triangleIndex] = uint3(
                readPageByte(pageWords, baseByte + 0u),
                readPageByte(pageWords, baseByte + 1u),
                readPageByte(pageWords, baseByte + 2u));
        }
    }
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

    uint patchWord = kStreamUpdateHeaderWords + patchIndex * 4u;
    uint pageId = updates[patchWord + 0u];
    pageTable[pageId].deviceOffsetBytes = updates[patchWord + 1u];
    pageTable[pageId].deviceSizeBytes = updates[patchWord + 2u];
    pageTable[pageId].state = updates[patchWord + 3u];
}

[shader("fragment")]
float4 gpuDrivenStreamAssetFragmentMain(GPUDrivenStreamAssetVertexOutput input, uniform GPUDrivenStreamAssetPush push) : SV_Target0
{
    return float4(input.color, 1.0);
}
