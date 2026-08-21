// SHaRC maintenance passes (reference: RTXGI v2 SDK SharcResolve.hlsl).
//
// sharcClearMain   - clears hash entries / accumulation / resolved buffers so
//                    the cache can be rebuilt from scratch.
// sharcResolveMain - one thread per cache entry; combines the per-frame
//                    accumulation data with previously resolved data, handles
//                    temporal accumulation, stale eviction and adjacent-level
//                    blending, then clears the accumulation buffer.

#include "Sharc/SharcCommon.h"

#ifndef SHARC_MAINTENANCE_BLOCK_SIZE
#define SHARC_MAINTENANCE_BLOCK_SIZE 256
#endif

struct SceneSharcMaintenanceParams
{
    float4 cameraPosition;
    float4 cameraPositionPrev;
    float sceneScale;
    uint entriesNum;
    uint accumulationFrameNum;
    uint staleFrameNumMax;
    uint frameIndex;
};

[[vk::binding(0, 0)]]
RWStructuredBuffer<uint64_t> gHashEntries;

[[vk::binding(1, 0)]]
RWStructuredBuffer<SharcAccumulationData> gAccumulation;

[[vk::binding(2, 0)]]
RWStructuredBuffer<SharcPackedData> gResolved;

[[vk::binding(3, 0)]]
StructuredBuffer<SceneSharcMaintenanceParams> gParams;

[shader("compute")]
[numthreads(SHARC_MAINTENANCE_BLOCK_SIZE, 1, 1)]
void sharcClearMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint entryIndex = dispatchThreadId.x;
    SceneSharcMaintenanceParams params = gParams[0];
    if (entryIndex >= params.entriesNum) {
        return;
    }

    gHashEntries[entryIndex] = HASH_GRID_INVALID_HASH_KEY;
    gAccumulation[entryIndex] = SharcZeroAccumulationData();
    gResolved[entryIndex] = SharcZeroPackedData();
}

[shader("compute")]
[numthreads(SHARC_MAINTENANCE_BLOCK_SIZE, 1, 1)]
void sharcResolveMain(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint entryIndex = dispatchThreadId.x;
    SceneSharcMaintenanceParams params = gParams[0];

    SharcParameters sharcParameters;
    sharcParameters.hashGridParameters.cameraPosition = params.cameraPosition.xyz;
    sharcParameters.hashGridParameters.logarithmBase = SHARC_GRID_LOGARITHM_BASE;
    sharcParameters.hashGridParameters.sceneScale = max(params.sceneScale, 0.0001);
    sharcParameters.hashGridParameters.levelBias = SHARC_GRID_LEVEL_BIAS;

    sharcParameters.hashGridData.capacity = params.entriesNum;
    sharcParameters.hashGridData.hashEntriesBuffer = gHashEntries;
    sharcParameters.accumulationBuffer = gAccumulation;
    sharcParameters.resolvedBuffer = gResolved;
    sharcParameters.radianceScale = 1.0e3;

    SharcResolveParameters resolveParameters;
    resolveParameters.cameraPositionPrev = params.cameraPositionPrev.xyz;
    resolveParameters.accumulationFrameNum = params.accumulationFrameNum;
    resolveParameters.staleFrameNumMax = params.staleFrameNumMax;
    resolveParameters.frameIndex = params.frameIndex;

    SharcResolveEntry(entryIndex, sharcParameters, resolveParameters);
}
