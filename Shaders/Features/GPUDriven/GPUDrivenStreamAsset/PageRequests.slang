#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Page residency requests, traversal predicates and page-table updates.
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
    StructuredBuffer<GPUDrivenStreamAssetLODLevel> lodLevels,
    GPUDrivenStreamAssetInstance instance,
    GPUDrivenStreamAssetPrimitive primitive)
{
    if (primitive.lodLevelCount == 0u) {
        return 0u;
    }
    if (params.enableGpuLodSelection == 0u && params.selectedLodLevel != kNoDebugLODOverride) {
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
        GPUDrivenStreamAssetLODLevel lod = lodLevels[lodIndex];
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
    if (params.enableGpuLodSelection == 0u && params.selectedLodLevel != kNoDebugLODOverride) {
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
    if (params.enableGpuLodSelection == 0u && params.selectedLodLevel != kNoDebugLODOverride) {
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
    if (!streamPageDrawable(streamPageState(pageEntry)) ||
        streamPageDeviceOffset(pageEntry) == kInvalidStreamDeviceOffset) { return false; }
    // The same predicate controls activation and complementary parent masks.
    // Geometry arriving before its CLAS must leave the legal ancestor cut in
    // place, rather than select a fine cut that sends the whole BLAS to its root.
    if (params.rayClasPageTableBuffer != 0xffffffffu) {
        StructuredBuffer<GPUDrivenStreamAssetCLASPageEntry> clasPages =
            resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetCLASPageEntry>>(params.rayClasPageTableBuffer));
        if (clasPageState(clasPages[group.pageIndex]) != kCLASPageStateActive) { return false; }
    }
    return true;
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

    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTableHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(push.pageTableBuffer));
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
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTableHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(push.pageTableBuffer));
    StructuredBuffer<uint> updateHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.updateBuffer));

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

