module LumenRadiositySchedule;

import ShaderCore;
import ParameterRoot;
import GlobalDistanceField;
import LumenSurfaceCache;
import LumenRadiosity;

using Metallic;
using Metallic.DistanceField;
using Metallic.Lumen;

static const uint kPriorityHistogramSize = 16u;
static const uint kNumReset = 2u;
static const uint kNumActiveReset = 3u;
static const uint kMaxUpdateBucket = 4u;
static const uint kMaxUpdateBucketQuota = 5u;
static const uint kAllocatedInMaxUpdateBucket = 6u;
static const uint kHistogramOffset = 8u;
static const uint kHistoryResetMask = kLumenCardCaptured | kLumenCardResetLighting;

struct RadiosityScheduleParams
{
    ResourceHandle<StructuredBuffer<SurfaceCacheCard>> cards;
    ResourceHandle<StructuredBuffer<uint>> cardScene;
    ResourceHandle<StructuredBuffer<GlobalDistanceFieldDomain>> domains;
    ResourceHandle<StructuredBuffer<uint>> cardUpdates;
    ResourceHandle<RWStructuredBuffer<uint2>> cardState;
    ResourceHandle<RWStructuredBuffer<uint2>> scheduledCards;
    ResourceHandle<RWStructuredBuffer<uint2>> refreshCards;
    ResourceHandle<RWStructuredBuffer<uint>> updateState;
    ResourceHandle<RWStructuredBuffer<uint>> indirectArgs;
    ResourceHandle<RWStructuredBuffer<uint>> updateStatistics;
    uint capacity;
    uint maxCardsPerFrame;
    uint tileResolution;
    uint cacheInitialized;
    uint frameIndex;
    uint hasCamera;
    float4 cameraEye;
    float4 cameraPlanes[6];
};

// UE 5.7.4 LumenSceneLighting.usf, GetPriorityBucketIndex. Each physical
// tile is the whole card, so the card's local box is also the page bounds.
uint getPriorityBucketIndex(RadiosityScheduleParams params, uint cardIndex, uint lastLightingUpdate)
{
    uint framesSinceLastUpdated = params.frameIndex - lastLightingUpdate;
    float updateSpeed = 1.0f;
    if (params.hasCamera != 0u) {
        SurfaceCacheCard card = resolveUniform(params.cards)[cardIndex];
        float3 relativeEye = params.cameraEye.xyz - card.origin.xyz;
        float3 cardSpaceViewPosition = float3(dot(relativeEye, card.axisX.xyz),
            dot(relativeEye, card.axisY.xyz), dot(relativeEye, card.axisZ.xyz));
        const float3 extent = float3(card.axisX.w, card.axisY.w, card.axisZ.w);
        const float3 delta = max(abs(cardSpaceViewPosition) - extent, 0.0f);
        float distanceFromViewer = sqrt(dot(delta, delta));
        GlobalDistanceFieldDomain domain = resolveUniform(params.domains)[0];
        const float firstClipmapWorldExtentRcp = 2.0f / (domain.maxAndInfluenceRadius.x - domain.minAndVoxelSize.x);
        updateSpeed = 1.0f / (1.0f + max(distanceFromViewer * firstClipmapWorldExtentRcp, 0.0f));

        const float3 cardWorldExtent = abs(card.axisX.xyz) * extent.x + abs(card.axisY.xyz) * extent.y +
            abs(card.axisZ.xyz) * extent.z;
        const float maxDistanceFromFrustum = 5.0f; // UE's 500 cm default.
        bool nearFrustum = true;
        for (uint planeIndex = 0; planeIndex < 6; ++planeIndex) {
            const float4 plane = params.cameraPlanes[planeIndex];
            const float distanceFromPageCenterToPlane = dot(plane, float4(card.origin.xyz, 1.0f));
            const float radius = dot(cardWorldExtent, abs(plane.xyz));
            nearFrustum = nearFrustum && (distanceFromPageCenterToPlane <= radius + maxDistanceFromFrustum);
        }
        if (nearFrustum) {
            updateSpeed *= 2.0f;
        }
    }

    if (lastLightingUpdate == 0u) {
        framesSinceLastUpdated = 2048u;
    }
    uint bucketIndex = uint(kPriorityHistogramSize - 1u -
        clamp(log2(max(4.0f * framesSinceLastUpdated * updateSpeed, 1.0f)), 0, kPriorityHistogramSize - 1u));
    return bucketIndex;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void lumenRadiosityClearScheduleMain(uint3 thread : SV_DispatchThreadID)
{
    RadiosityScheduleParams params = getParameters<RadiosityScheduleParams>();
    if (thread.x < kHistogramOffset + kPriorityHistogramSize) {
        resolveUniform(params.updateState)[thread.x] = 0u;
    }
}

struct CardUpdate
{
    bool active;
    bool reset;
    bool refresh;
    uint2 state;
    uint flags;
};

CardUpdate getCardUpdate(RadiosityScheduleParams params, uint slot, uint resident)
{
    CardUpdate update = {};
    update.active = slot < resident;
    update.state = params.cacheInitialized != 0u ? resolveUniform(params.cardState)[slot] : uint2(0u);
    update.flags = update.active ? resolveUniform(params.cardUpdates)[slot] : 0u;
    const bool wasActive = update.state.x != 0u;
    update.reset = params.cacheInitialized == 0u || update.active != wasActive ||
        (update.flags & kHistoryResetMask) != 0u;
    update.refresh = update.reset || (update.flags & kLumenCardRelit) != 0u;
    return update;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenRadiosityBuildPriorityMain(uint3 thread : SV_DispatchThreadID)
{
    RadiosityScheduleParams params = getParameters<RadiosityScheduleParams>();
    if (thread.x >= params.capacity) { return; }

    const uint resident = resolveUniform(params.cardScene)[0];
    CardUpdate update = getCardUpdate(params, thread.x, resident);
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    if (update.active) {
        if (update.reset) {
            InterlockedAdd(state[kNumActiveReset], 1u);
        } else {
            uint bucketIndex = getPriorityBucketIndex(params, thread.x, update.state.x);
            InterlockedAdd(state[kHistogramOffset + bucketIndex], 1u);
        }
    }
}

// UE 5.7.4 LumenSceneLighting.usf, SelectMaxUpdateBucket. One card is one
// budget unit in Metallic. Subtract mandatory resets from the normal budget.
[shader("compute")]
[numthreads(1, 1, 1)]
void lumenRadiositySelectPriorityMain()
{
    RadiosityScheduleParams params = getParameters<RadiosityScheduleParams>();
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    const uint maxCardsToUpdate = params.maxCardsPerFrame - min(params.maxCardsPerFrame, state[kNumActiveReset]);
    uint updateTileSum = 0;
    uint priorityBucketIndex = 0;
    uint priorityBucketMaxTiles = maxCardsToUpdate;
    for (; priorityBucketIndex < kPriorityHistogramSize; ++priorityBucketIndex) {
        uint tilesPerBucket = state[kHistogramOffset + priorityBucketIndex];
        if (updateTileSum + tilesPerBucket >= maxCardsToUpdate) {
            priorityBucketMaxTiles = maxCardsToUpdate - updateTileSum;
            break;
        }
        updateTileSum += tilesPerBucket;
    }
    state[kMaxUpdateBucket] = priorityBucketIndex;
    state[kMaxUpdateBucketQuota] = priorityBucketMaxTiles;
}

// UE 5.7.4 LumenSceneLighting.usf, AddCardPageIndex. The reset/relit combine
// uses a separate list so it cannot consume the trace allocation counter.
[shader("compute")]
[numthreads(64, 1, 1)]
void lumenRadiosityBuildCardListsMain(uint3 thread : SV_DispatchThreadID)
{
    RadiosityScheduleParams params = getParameters<RadiosityScheduleParams>();
    const uint slot = thread.x;
    if (slot >= params.capacity) { return; }

    const uint resident = resolveUniform(params.cardScene)[0];
    CardUpdate update = getCardUpdate(params, slot, resident);
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    if (update.refresh) {
        uint refreshIndex;
        InterlockedAdd(state[kRadiosityRefreshCount], 1u, refreshIndex);
        resolveUniform(params.refreshCards)[refreshIndex] = uint2(slot, uint(update.reset) | (uint(update.active) << 1u));
    }
    if (update.reset) {
        if (update.active || update.state.x != 0u) {
            InterlockedAdd(state[kNumReset], 1u);
        }
        resolveUniform(params.cardState)[slot] = update.active ? uint2(params.frameIndex, 1u) : uint2(0u);
        return;
    }
    if (!update.active) { return; }

    const uint bucketIndex = getPriorityBucketIndex(params, slot, update.state.x);
    const uint maxUpdateBucket = state[kMaxUpdateBucket];
    bool updateThisCard = bucketIndex <= maxUpdateBucket;
    if (updateThisCard && bucketIndex == maxUpdateBucket) {
        uint allocatedInMaxUpdateBucket;
        InterlockedAdd(state[kAllocatedInMaxUpdateBucket], 1u, allocatedInMaxUpdateBucket);
        updateThisCard = allocatedInMaxUpdateBucket < state[kMaxUpdateBucketQuota];
    }
    if (updateThisCard) {
        uint listIndex;
        InterlockedAdd(state[kRadiosityScheduledCount], 1u, listIndex);
        const uint temporalIndex = update.state.y + 1u;
        resolveUniform(params.scheduledCards)[listIndex] = uint2(slot, temporalIndex);
        resolveUniform(params.cardState)[slot] = uint2(params.frameIndex, temporalIndex);
    }
}

void writeDispatchArgs(RWStructuredBuffer<uint> args, uint offset, uint edge, uint count)
{
    const uint groups = (edge + 7u) / 8u;
    args[offset + 0u] = groups * min(count, kRadiosityDispatchCardsPerRow);
    args[offset + 1u] = groups;
    args[offset + 2u] = (count + kRadiosityDispatchCardsPerRow - 1u) / kRadiosityDispatchCardsPerRow;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lumenRadiosityFinishScheduleMain()
{
    RadiosityScheduleParams params = getParameters<RadiosityScheduleParams>();
    RWStructuredBuffer<uint> args = resolveUniform(params.indirectArgs);
    RWStructuredBuffer<uint> state = resolveUniform(params.updateState);
    const uint traceCount = state[kRadiosityScheduledCount];
    writeDispatchArgs(args, 0u, params.tileResolution, state[kRadiosityRefreshCount]);
    writeDispatchArgs(args, 3u, params.tileResolution, traceCount);
    writeDispatchArgs(args, 6u, params.tileResolution / kRadiosityProbeSpacing, traceCount);
    RWStructuredBuffer<uint> stats = resolveUniform(params.updateStatistics);
    stats[0] = traceCount + state[kNumActiveReset];
    stats[1] = state[kNumReset];
    stats[2] = resolveUniform(params.cardScene)[0];
    stats[3] = params.maxCardsPerFrame;
}
