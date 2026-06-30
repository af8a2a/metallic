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
    uint instanceBuffer;
    uint primitiveBuffer;
    uint lodLevelBuffer;
    uint pageInfoBuffer;
    uint traversalPhase;
};

struct GPUDrivenStreamAssetActiveGroup
{
    uint pageDeviceOffsetBytes;
    uint pageIndex;
    uint clusterCount;
    uint primitiveIndex;
    uint lodLevel;
    uint materialIndex;
    uint colorSeed;
    uint flags;
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
    uint materialIndex;
    uint padding0;
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
    uint padding0;
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
    uint padding0;
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
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTableHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> requestHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.requestBuffer, 0));

    StructuredBuffer<GPUDrivenStreamAssetInstance> instances = instanceHandle;
    StructuredBuffer<GPUDrivenStreamAssetPrimitive> primitives = primitiveHandle;
    StructuredBuffer<GPUDrivenStreamAssetLodLevel> lodLevels = lodLevelHandle;
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

    uint instanceIndex = dispatchThreadID.x;
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
    if (primitive.lodLevelCount == 0u ||
        primitive.lodLevelOffset >= params.sceneLodLevelCount ||
        primitive.lodLevelCount > params.sceneLodLevelCount - primitive.lodLevelOffset) {
        InterlockedAdd(requests[kStreamRequestInvalidPageCounterWord], 1u);
        InterlockedMax(requests[kStreamRequestLastInvalidPageFrameWord], params.frameIndex);
        return;
    }

    uint localLod = selectTraversalLod(params, lodLevels, instance, primitive);
    uint selectedLodIndex = primitive.lodLevelOffset + localLod;
    GPUDrivenStreamAssetLodLevel selectedLod = lodLevels[selectedLodIndex];
    for (uint page = 0u; page < selectedLod.pageCount; ++page) {
        uint pageIndex = selectedLod.pageOffset + page;
        if (pageIndex >= params.scenePageCount) {
            InterlockedAdd(requests[kStreamRequestInvalidPageCounterWord], 1u);
            InterlockedMax(requests[kStreamRequestLastInvalidPageFrameWord], params.frameIndex);
            continue;
        }
        uint state = pageTable[pageIndex].state;
        if (streamPageAvailable(state)) {
            markStreamPageDemand(pageTable, params, pageIndex);
        } else {
            requestStreamPageLoad(pageTable, requests, params, pageIndex);
        }
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
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTableHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> requestHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.requestBuffer, 0));

    StructuredBuffer<uint> pageWords = pageHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    RWStructuredBuffer<uint> requests = requestHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    uint drawIndex = groupID.x;
    if (drawIndex >= params.drawTaskCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    uint maxActiveGroupClusters = max(params.maxActiveGroupClusters, 1u);
    uint activeGroupIndex = drawIndex / maxActiveGroupClusters;
    uint clusterIndex = drawIndex - activeGroupIndex * maxActiveGroupClusters;
    if (activeGroupIndex >= params.activeGroupCount) {
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
