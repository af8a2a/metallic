module LumenRadianceCacheUpdate;

import ShaderCore;
import ParameterRoot;
import LumenAdaptiveProbes;
import LumenRadianceCache;

using Metallic;
using Metallic.Lumen;

struct LumenRadianceCacheUpdateParams
{
    ResourceHandle<Texture2D<float4>> position;
    ResourceHandle<StructuredBuffer<uint2>> screenProbePositions;
    ResourceHandle<StructuredBuffer<LumenRadianceCacheClipmap>> clipmaps;
    ResourceHandle<StructuredBuffer<LumenRadianceCacheClipmap>> previousClipmaps;
    ResourceHandle<RWStructuredBuffer<uint>> indirection;
    ResourceHandle<RWStructuredBuffer<uint>> previousIndirection;
    ResourceHandle<RWStructuredBuffer<uint>> freeList;
    ResourceHandle<RWStructuredBuffer<uint2>> probeFrames;
    ResourceHandle<RWStructuredBuffer<uint>> updateState;
    ResourceHandle<RWStructuredBuffer<float4>> traceData;
    ResourceHandle<RWStructuredBuffer<uint>> indirectArgs;
    uint gridResolution;
    uint clipmapCount;
    uint maxProbes;
    uint screenProbeCount;
    uint frameNumber;
    uint historyValid;
    uint probeTraceBudget;
    uint probeResolution;
    float3 cameraPosition;
    uint historyFirstFrame;
};

struct LumenCompactRadianceCacheMarkParams
{
    LumenRadianceCacheUpdateParams update;
    ResourceHandle<StructuredBuffer<uint>> adaptiveCount;
    uint uniformCount;
};

struct LumenRadianceCachePriorityParams
{
    ResourceHandle<RWStructuredBuffer<uint>> updateState;
    uint probeTraceBudget;
    uint globalLightingChanged;
};

// Compact buffer adaptation of the UE 5.7.4 allocator, update and histogram buffers.
static const uint kProbeAllocator = 0u;
static const uint kFreeListCount = 1u;
static const uint kTraceProbeCount = 2u;
static const uint kMaxUpdateBucket = 3u;
static const uint kLastBucketBudget = 4u;
static const uint kSelectedTraceCost = 5u;
static const uint kSelectedLastBucketCost = 6u;
static const uint kMinNewTraceCost = 7u;
static const uint kPriorityHistogram = 8u;
static const uint kPriorityHistogramSize = 16u;
static const uint kMarkedProbes = 24u;
static const uint kReusedMarkedProbes = 25u;
static const uint kAllocatedProbes = 26u;
static const uint kEvictedProbes = 27u;
static const uint kOverflowProbes = 28u;
static const uint kEffectivePriorityBudget = 30u;
static const uint kGlobalLightingChange = 31u;
static const uint kUpdateStateCount = 32u;

// UE 5.7.4 LumenRadianceCacheUpdate.ush and LumenRadianceCache.cpp defaults.
static const uint kNeverTraced = 0u;
static const uint kDownsampledTraceCost = 1u;
static const uint kNormalTraceCost = 4u;
static const uint kSupersampledTraceCost = 16u;
static const uint kKeepCachedProbesFrames = 8u;
static const float kDownsampleDistanceSq = 40.0f * 40.0f;

uint cacheGridCellCount(LumenRadianceCacheUpdateParams params)
{
    return params.gridResolution * params.gridResolution * params.gridResolution * params.clipmapCount;
}

uint3 cacheCellForAddress(uint address, LumenRadianceCacheUpdateParams params, out uint clipmapIndex)
{
    uint packedWidth = params.gridResolution * params.clipmapCount;
    uint packedX = address % packedWidth;
    clipmapIndex = packedX / params.gridResolution;
    return uint3(packedX % params.gridResolution, address / packedWidth % params.gridResolution,
        address / packedWidth / params.gridResolution);
}

bool cacheCellInBounds(int3 cell, uint gridResolution)
{
    return all(cell >= 0) && all(cell < int(gridResolution));
}

uint getProbeTraceCost(float3 cameraPosition, float3 probeWorldPosition, float supersampleDistanceSq)
{
    // GetDistanceToCameraFromViewVector, perspective branch.
    float cameraDistance = length(cameraPosition - probeWorldPosition);
    float distanceFromCameraSq = cameraDistance * cameraDistance;
    if (distanceFromCameraSq >= kDownsampleDistanceSq) {
        return kDownsampledTraceCost;
    }
    if (distanceFromCameraSq < supersampleDistanceSq) {
        return kSupersampledTraceCost;
    }
    return kNormalTraceCost;
}

uint priorityBucketIndex(uint lastTracedFrame, uint lastUsedFrame, uint clipmapIndex)
{
    uint bucketIndex = 0u;
    if (lastTracedFrame != kNeverTraced) {
        uint framesBetweenTracedAndUsed = lastUsedFrame > lastTracedFrame ? lastUsedFrame - lastTracedFrame : 1u;
        float updateImportance = float(framesBetweenTracedAndUsed) / (float(clipmapIndex) + 1.0f);
        bucketIndex = uint(float(kPriorityHistogramSize - 1u) -
            clamp(log2(updateImportance), 0.0f, float(kPriorityHistogramSize - 2u)));
    }
    return bucketIndex;
}

// UE ClearProbeIndirectionCS and ClearRadianceCacheUpdateResourcesCS.
[shader("compute")]
[numthreads(64, 1, 1)]
void lumenClearRadianceCacheUpdateMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadianceCacheUpdateParams params = getParameters<LumenRadianceCacheUpdateParams>();
    uint index = thread.x;
    if (index < cacheGridCellCount(params)) {
        resolveUniform(params.indirection)[index] = kLumenRadianceInvalidProbe;
    }
    if (index < kUpdateStateCount && (index >= kTraceProbeCount || params.historyValid == 0u)) {
        resolveUniform(params.updateState)[index] = 0u;
    }
}

// UE MarkRadianceProbesUsedByScreenProbesCS, with world meters from the Deferred receiver texture.
void markRadianceCacheProbe(LumenRadianceCacheUpdateParams params, uint index)
{
    uint2 pixel = resolveUniform(params.screenProbePositions)[index];
    if (pixel.x == kLumenInvalidProbe) {
        return;
    }
    float4 receiver = resolveUniform(params.position).Load(int3(pixel, 0));
    if (receiver.w <= 0.0f) {
        return;
    }

    StructuredBuffer<LumenRadianceCacheClipmap> clipmaps = resolveUniform(params.clipmaps);
    uint clipmapIndex = lumenRadianceClipmapForPosition(receiver.xyz, clipmaps, 0.0f);
    if (clipmapIndex >= params.clipmapCount) {
        return;
    }
    float3 probeCoord = lumenRadianceProbeCoordFloat(receiver.xyz, clipmaps[clipmapIndex]);
    int3 bottomCorner = int3(floor(probeCoord - 0.5f));
    RWStructuredBuffer<uint> indirection = resolveUniform(params.indirection);
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    [unroll]
    for (uint corner = 0u; corner < 8u; ++corner) {
        int3 cell = bottomCorner + int3(corner >> 2u, (corner >> 1u) & 1u, corner & 1u);
        if (cacheCellInBounds(cell, params.gridResolution)) {
            uint address = lumenRadianceIndirectionAddress(uint3(cell), clipmapIndex, params.gridResolution, params.clipmapCount);
            uint old;
            InterlockedExchange(indirection[address], kLumenRadianceUsedProbe, old);
            if (old == kLumenRadianceInvalidProbe) {
                InterlockedAdd(state[kMarkedProbes], 1u);
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenMarkRadianceCacheProbesMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadianceCacheUpdateParams params = getParameters<LumenRadianceCacheUpdateParams>();
    if (thread.x < params.screenProbeCount) {
        markRadianceCacheProbe(params, thread.x);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenMarkCompactRadianceCacheProbesMain(uint3 thread : SV_DispatchThreadID)
{
    LumenCompactRadianceCacheMarkParams params = getParameters<LumenCompactRadianceCacheMarkParams>();
    uint adaptiveCapacity = params.update.screenProbeCount - params.uniformCount;
    uint probeCount = params.uniformCount + min(resolveUniform(params.adaptiveCount)[0], adaptiveCapacity);
    if (thread.x < probeCount) {
        markRadianceCacheProbe(params.update, thread.x);
    }
}

// UE UpdateCacheForUsedProbesCS, retaining positions still used or younger than
// the cache lifetime and returning all other physical slots to the free list.
[shader("compute")]
[numthreads(64, 1, 1)]
void lumenRemapRadianceCacheProbesMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadianceCacheUpdateParams params = getParameters<LumenRadianceCacheUpdateParams>();
    if (params.historyValid == 0u || thread.x >= cacheGridCellCount(params)) {
        return;
    }
    uint lastProbe = resolveUniform(params.previousIndirection)[thread.x];
    if (lastProbe == kLumenRadianceInvalidProbe) {
        return;
    }
    uint clipmapIndex;
    uint3 lastCell = cacheCellForAddress(thread.x, params, clipmapIndex);
    float3 worldPosition = lumenRadianceProbeWorldPosition(lastCell, resolveUniform(params.previousClipmaps)[clipmapIndex]);
    int3 cell = int3(floor(lumenRadianceProbeCoordFloat(worldPosition, resolveUniform(params.clipmaps)[clipmapIndex])));
    RWStructuredBuffer<uint> indirection = resolveUniform(params.indirection);
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    if (cacheCellInBounds(cell, params.gridResolution)) {
        uint address = lumenRadianceIndirectionAddress(uint3(cell), clipmapIndex, params.gridResolution, params.clipmapCount);
        uint marker = indirection[address];
        RWStructuredBuffer<uint2> probeFrames = resolveUniform(params.probeFrames);
        uint2 frames = probeFrames[lastProbe];
        if (marker == kLumenRadianceUsedProbe || params.frameNumber - frames.y < kKeepCachedProbesFrames) {
            if (marker == kLumenRadianceUsedProbe) {
                frames.y = params.frameNumber;
                probeFrames[lastProbe] = frames;
                InterlockedAdd(state[kReusedMarkedProbes], 1u);
            }
            indirection[address] = lastProbe;
            return;
        }
    }
    uint freeIndex;
    InterlockedAdd(state[kFreeListCount], 1u, freeIndex);
    resolveUniform(params.freeList)[freeIndex] = lastProbe;
    InterlockedAdd(state[kEvictedProbes], 1u);
}

// UE AllocateUsedProbesCS, persistent cache permutation.
[shader("compute")]
[numthreads(64, 1, 1)]
void lumenAllocateRadianceCacheProbesMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadianceCacheUpdateParams params = getParameters<LumenRadianceCacheUpdateParams>();
    if (thread.x >= cacheGridCellCount(params)) {
        return;
    }
    RWStructuredBuffer<uint> indirection = resolveUniform(params.indirection);
    uint marker = indirection[thread.x];
    if (marker == kLumenRadianceInvalidProbe) {
        return;
    }
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    RWStructuredBuffer<uint2> probeFrames = resolveUniform(params.probeFrames);
    uint probeIndex = marker;
    uint2 frames;
    if (marker == kLumenRadianceUsedProbe) {
        uint freeCountBits;
        InterlockedAdd(state[kFreeListCount], 0xffffffffu, freeCountBits);
        int freeCount = asint(freeCountBits);
        if (freeCount > 0) {
            probeIndex = resolveUniform(params.freeList)[freeCount - 1];
        } else {
            InterlockedAdd(state[kProbeAllocator], 1u, probeIndex);
        }
        if (probeIndex >= params.maxProbes) {
            indirection[thread.x] = kLumenRadianceInvalidProbe;
            InterlockedAdd(state[kOverflowProbes], 1u);
            return;
        }
        frames = uint2(kNeverTraced, params.frameNumber);
        probeFrames[probeIndex] = frames;
        indirection[thread.x] = probeIndex;
        InterlockedAdd(state[kAllocatedProbes], 1u);
    } else {
        frames = probeFrames[probeIndex];
    }

    uint clipmapIndex;
    uint3 cell = cacheCellForAddress(thread.x, params, clipmapIndex);
    LumenRadianceCacheClipmap clipmap = resolveUniform(params.clipmaps)[clipmapIndex];
    float3 worldPosition = lumenRadianceProbeWorldPosition(cell, clipmap);
    uint bucket = priorityBucketIndex(frames.x, frames.y, clipmapIndex);
    InterlockedAdd(state[kPriorityHistogram + bucket], getProbeTraceCost(params.cameraPosition, worldPosition, clipmap.traceSettings.w));
    if (bucket == 0u) {
        InterlockedAdd(state[kMinNewTraceCost], kDownsampledTraceCost);
    }
}

// UE SelectMaxPriorityBucketCS.
[shader("compute")]
[numthreads(1, 1, 1)]
void lumenSelectRadianceCachePriorityMain()
{
    LumenRadianceCachePriorityParams params = getParameters<LumenRadianceCachePriorityParams>();
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    // Clamp the normalized cost before multiplying. The saturated value exceeds all representable probe work.
    uint budget = min(params.probeTraceBudget, 0xffffffffu / kNormalTraceCost) * kNormalTraceCost;
    uint traceSum = 0u;
    uint bucketIndex = 0u;
    uint lastBucketBudget = budget;
    for (; bucketIndex < kPriorityHistogramSize; ++bucketIndex) {
        uint bucketCost = state[kPriorityHistogram + bucketIndex];
        if (traceSum + bucketCost >= budget) {
            lastBucketBudget = budget - traceSum;
            break;
        }
        traceSum += bucketCost;
    }
    state[kMaxUpdateBucket] = bucketIndex;
    state[kLastBucketBudget] = lastBucketBudget;
    state[kEffectivePriorityBudget] = params.probeTraceBudget;
    state[kGlobalLightingChange] = params.globalLightingChanged;
}

// UE AllocateProbeTracesCS. Newly mapped slots are always traced before a
// consumer can sample them, with reference downsampling when over budget.
[shader("compute")]
[numthreads(64, 1, 1)]
void lumenSelectRadianceCacheTraceProbesMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadianceCacheUpdateParams params = getParameters<LumenRadianceCacheUpdateParams>();
    if (thread.x >= cacheGridCellCount(params)) {
        return;
    }
    uint probeIndex = resolveUniform(params.indirection)[thread.x];
    if (probeIndex >= params.maxProbes) {
        return;
    }
    uint clipmapIndex;
    uint3 cell = cacheCellForAddress(thread.x, params, clipmapIndex);
    LumenRadianceCacheClipmap clipmap = resolveUniform(params.clipmaps)[clipmapIndex];
    float3 worldPosition = lumenRadianceProbeWorldPosition(cell, clipmap);
    uint probeCost = getProbeTraceCost(params.cameraPosition, worldPosition, clipmap.traceSettings.w);
    RWStructuredBuffer<uint2> probeFrames = resolveUniform(params.probeFrames);
    uint2 frames = probeFrames[probeIndex];
    uint bucket = priorityBucketIndex(frames.x, frames.y, clipmapIndex);
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    uint maxBucket = state[kMaxUpdateBucket];
    if (bucket > maxBucket) {
        return;
    }
    bool forceDownsample = false;
    if (bucket == maxBucket && maxBucket > 0u) {
        uint prevLastBucketCost;
        InterlockedAdd(state[kSelectedLastBucketCost], probeCost, prevLastBucketCost);
        if (prevLastBucketCost + probeCost > state[kLastBucketBudget]) {
            return;
        }
    } else if (bucket == 0u) {
        uint prevMinNewCost;
        uint extraCost = probeCost - kDownsampledTraceCost;
        InterlockedAdd(state[kMinNewTraceCost], extraCost, prevMinNewCost);
        forceDownsample = prevMinNewCost + extraCost > params.probeTraceBudget * kNormalTraceCost;
    }

    InterlockedAdd(state[kSelectedTraceCost], probeCost);
    uint traceIndex;
    InterlockedAdd(state[kTraceProbeCount], 1u, traceIndex);
    // UE PackProbeTraceData, LumenRadianceCacheTracingCommon.ush.
    uint packedTrace = (clipmapIndex << 24u) | probeIndex;
    packedTrace |= forceDownsample ? 1u << 31u : 0u;
#if defined(METALLIC_LUMEN_WRC_PROGRESSIVE)
    if (frames.x < params.historyFirstFrame) {
        packedTrace |= kLumenRadianceResetDirectionHistory;
    }
#endif
    resolveUniform(params.traceData)[traceIndex] = float4(worldPosition, asfloat(packedTrace));
    frames.x = params.frameNumber;
    probeFrames[probeIndex] = frames;
}

void writeCacheDispatch(RWStructuredBuffer<uint> indirectArgs, uint offset, uint x, uint y, uint z)
{
    indirectArgs[offset] = x;
    indirectArgs[offset + 1u] = y;
    indirectArgs[offset + 2u] = z;
}

// UE SetupProbeIndirectArgsCS, restricting arguments to the directional-radiance path.
[shader("compute")]
[numthreads(1, 1, 1)]
void lumenSetupRadianceCacheDispatchMain()
{
    LumenRadianceCacheUpdateParams params = getParameters<LumenRadianceCacheUpdateParams>();
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    state[kFreeListCount] = uint(clamp(asint(state[kFreeListCount]), 0, int(params.maxProbes)));
    state[kProbeAllocator] = min(state[kProbeAllocator], params.maxProbes);
    uint traceProbeCount = state[kTraceProbeCount];
    RWStructuredBuffer<uint> indirectArgs = resolveUniform(params.indirectArgs);
    uint filterGroups = (params.probeResolution + 7u) / 8u;
    uint borderGroups = (params.probeResolution + 2u + 7u) / 8u;
    writeCacheDispatch(indirectArgs, 0u, 1u, 1u, traceProbeCount);
    writeCacheDispatch(indirectArgs, 3u, filterGroups, filterGroups, traceProbeCount);
    writeCacheDispatch(indirectArgs, 6u, borderGroups, borderGroups, traceProbeCount);
}

// Materialize UE ExtractRadianceCacheState's indirection rollover in the
// graph's persistent previous grid.
[shader("compute")]
[numthreads(64, 1, 1)]
void lumenSaveRadianceCacheIndirectionMain(uint3 thread : SV_DispatchThreadID)
{
    LumenRadianceCacheUpdateParams params = getParameters<LumenRadianceCacheUpdateParams>();
    if (thread.x < cacheGridCellCount(params)) {
        RWStructuredBuffer<uint> indirection = resolveUniform(params.indirection);
        uint probeIndex = indirection[thread.x];
        resolveUniform(params.previousIndirection)[thread.x] = probeIndex;
#if defined(METALLIC_LUMEN_WRC_PROGRESSIVE)
        if (probeIndex < params.maxProbes &&
            resolveUniform(params.probeFrames)[probeIndex].x < params.historyFirstFrame) {
            indirection[thread.x] = kLumenRadianceInvalidProbe;
        }
#endif
    }
}
