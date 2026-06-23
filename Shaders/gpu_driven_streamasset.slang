static const uint kMaxMeshletVertices = 128;
static const uint kMaxMeshletTriangles = 128;
static const uint kInvalidStreamSlot = 0xffffffffu;
static const uint kInvalidClusterIndex = 0xffffffffu;
static const uint kStreamStateResident = 2u;
static const uint kStreamStateLockedFallback = 3u;

struct GPUDrivenStreamAssetPush
{
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint pageBuffer;
    uint drawItemBuffer;
    uint pageTableBuffer;
    uint paramsBuffer;
    uint requestBuffer;
    uint updateBuffer;
    uint padding0;
    uint padding1;
};

struct GPUDrivenStreamAssetDrawItem
{
    uint pageSlot;
    uint clusterIndex;
    uint pageIndex;
    uint primitiveIndex;
    uint lodLevel;
    uint materialIndex;
    uint colorSeed;
    uint padding0;
    float4 world0;
    float4 world1;
    float4 world2;
    float4 world3;
};

struct GPUDrivenStreamAssetPageTableEntry
{
    uint slot;
    uint state;
    uint lastRequestFrame;
    uint lodLevel;
    uint payloadBytes;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct GPUDrivenStreamAssetRequestHeader
{
    uint loadCounter;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct GPUDrivenStreamAssetPatch
{
    uint pageId;
    uint slot;
    uint state;
    uint padding0;
};

struct GPUDrivenStreamAssetUpdateHeader
{
    uint patchCounter;
    uint padding0;
    uint padding1;
    uint padding2;
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
    uint pageStrideWords;
    uint drawItemCount;
    uint frameIndex;
    uint maxGpuPageRequests;
    uint padding0;
    uint padding1;
    uint padding2;
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

void requestStreamPage(
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
    InterlockedAdd(requests[0], 1u, requestOffset);
    if (requestOffset < params.maxGpuPageRequests) {
        requests[4u + requestOffset] = pageIndex;
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
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetDrawItem>> drawItemHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetDrawItem>>(uint2(push.bufferShaderIndexBase + push.drawItemBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>> pageTableHandle =
        DescriptorHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(uint2(push.bufferShaderIndexBase + push.pageTableBuffer, 0));
    DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>> paramsHandle =
        DescriptorHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(uint2(push.bufferShaderIndexBase + push.paramsBuffer, 0));
    DescriptorHandle<RWStructuredBuffer<uint>> requestHandle =
        DescriptorHandle<RWStructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.requestBuffer, 0));

    StructuredBuffer<uint> pageWords = pageHandle;
    StructuredBuffer<GPUDrivenStreamAssetDrawItem> drawItems = drawItemHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    RWStructuredBuffer<uint> requests = requestHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    uint drawIndex = groupID.x;
    if (drawIndex >= params.drawItemCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    GPUDrivenStreamAssetDrawItem item = drawItems[drawIndex];
    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[item.pageIndex];
    if (item.clusterIndex == kInvalidClusterIndex ||
        ((pageEntry.state != kStreamStateResident && pageEntry.state != kStreamStateLockedFallback) ||
        pageEntry.slot == kInvalidStreamSlot ||
        pageEntry.payloadBytes == 0u)) {
        requestStreamPage(pageTable, requests, params, item.pageIndex);
        SetMeshOutputCounts(0, 0);
        return;
    }

    uint pageWord = pageEntry.slot * params.pageStrideWords;
    uint clusterCount = pageWords[pageWord + 2u];
    if (item.clusterIndex >= clusterCount) {
        SetMeshOutputCounts(0, 0);
        return;
    }

    uint clusterOffsetBytes = pageWords[pageWord + 9u];
    uint positionOffsetBytes = pageWords[pageWord + 10u];
    uint triangleOffsetBytes = pageWords[pageWord + 11u];
    uint clusterWord = pageWord + clusterOffsetBytes / 4u + item.clusterIndex * 8u;
    uint vertexOffset = pageWords[clusterWord + 0u];
    uint vertexCount = min(pageWords[clusterWord + 1u], kMaxMeshletVertices);
    uint triangleOffset = pageWords[clusterWord + 2u];
    uint triangleCount = min(pageWords[clusterWord + 3u], kMaxMeshletTriangles);

    SetMeshOutputCounts(vertexCount, triangleCount);

    uint colorSeed = item.pageIndex;
    if (params.debugColorMode == 1u) {
        colorSeed = item.lodLevel;
    } else if (params.debugColorMode == 2u) {
        colorSeed = item.primitiveIndex;
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
                item.world0.xyz * localPosition.x +
                item.world1.xyz * localPosition.y +
                item.world2.xyz * localPosition.z +
                item.world3.xyz;
            float4 projected = projectPosition(worldPosition, params);
            projected.y = -projected.y;
            vertices[vertexIndex].position = projected;
            vertices[vertexIndex].color = color;
        }
    }

    uint triangleBaseByte = pageEntry.slot * params.pageStrideWords * 4u + triangleOffsetBytes + triangleOffset;
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
    uint patchCount = updates[0];
    if (patchIndex >= patchCount) {
        return;
    }

    uint patchWord = 4u + patchIndex * 4u;
    uint pageId = updates[patchWord + 0u];
    pageTable[pageId].slot = updates[patchWord + 1u];
    pageTable[pageId].state = updates[patchWord + 2u];
}

[shader("fragment")]
float4 gpuDrivenStreamAssetFragmentMain(GPUDrivenStreamAssetVertexOutput input, uniform GPUDrivenStreamAssetPush push) : SV_Target0
{
    return float4(input.color, 1.0);
}
