#include "../Shared/gpu_driven_helpers.slang"

struct StreamingAgeFilterUniforms {
    uint activeResidentGroupCount;
    uint ageThreshold;
    uint requestFrameIndex;
    uint unloadRequestCapacity;
    uint groupResidencyCount;
    uint groupAgeCount;
    uint compactDispatch;
    uint reserved1;
};

ConstantBuffer<StreamingAgeFilterUniforms> streamingUniforms; // buffer(GPU_DRIVEN_STREAMING_AGE_UNIFORMS_BINDING)
RWByteAddressBuffer                        groupResidency;    // buffer(GPU_DRIVEN_STREAMING_AGE_GROUP_RESIDENCY_BINDING)
RWStructuredBuffer<uint>                   groupAgeBuffer;    // buffer(GPU_DRIVEN_STREAMING_AGE_GROUP_AGE_BINDING)
RWStructuredBuffer<ClusterUnloadRequest>   unloadRequests;    // buffer(GPU_DRIVEN_STREAMING_AGE_UNLOAD_REQUEST_OUTPUT_BINDING)
RWByteAddressBuffer                        unloadRequestState; // buffer(GPU_DRIVEN_STREAMING_AGE_UNLOAD_REQUEST_STATE_BINDING)
RWByteAddressBuffer                        streamingStats;    // buffer(GPU_DRIVEN_STREAMING_AGE_STATS_BINDING)
StructuredBuffer<uint>                     activeResidentGroups; // buffer(GPU_DRIVEN_STREAMING_AGE_ACTIVE_RESIDENT_GROUPS_BINDING)

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    const bool compactDispatch = streamingUniforms.compactDispatch != 0u;
    const uint dispatchIndex = dispatchThreadID.x;
    const uint dispatchCount = compactDispatch
                                   ? streamingUniforms.activeResidentGroupCount
                                   : min(streamingUniforms.groupResidencyCount,
                                         streamingUniforms.groupAgeCount);
    if (dispatchIndex >= dispatchCount) {
        return;
    }

    const uint groupIndex =
        compactDispatch ? activeResidentGroups[dispatchIndex] : dispatchIndex;

    if (groupIndex >= streamingUniforms.groupResidencyCount ||
        groupIndex >= streamingUniforms.groupAgeCount) {
        clusterStreamingIncrementErrorAgeFilter(streamingStats);
        return;
    }

    const uint residencyWordAddress = groupIndex * 4u;
    uint residencyState = 0u;
    groupResidency.InterlockedAnd(
        residencyWordAddress,
        ~(kClusterLodGroupResidencyTouched | kClusterLodGroupResidencyRequested),
        residencyState);
    const bool resident = (residencyState & kClusterLodGroupResidencyResident) != 0u;
    const bool alwaysResident = (residencyState & kClusterLodGroupResidencyAlwaysResident) != 0u;
    const bool touchedThisFrame = (residencyState & kClusterLodGroupResidencyTouched) != 0u;

    if (!resident || alwaysResident) {
        groupAgeBuffer[groupIndex] = 0u;
        return;
    }

    if (touchedThisFrame) {
        groupAgeBuffer[groupIndex] = 0u;
        return;
    }

    uint age = groupAgeBuffer[groupIndex];
    age = min(age + 1u, 0xFFFFu);
    groupAgeBuffer[groupIndex] = age;

    if (age <= streamingUniforms.ageThreshold) {
        return;
    }

    const uint slot = gpuDrivenAppendWorkItemSlot(unloadRequestState);
    if (slot >= streamingUniforms.unloadRequestCapacity) {
        clusterStreamingIncrementErrorAgeFilter(streamingStats);
        return;
    }

    ClusterUnloadRequest request;
    request.targetGroupIndex = groupIndex;
    request.requestFrameIndex = streamingUniforms.requestFrameIndex;
    unloadRequests[slot] = request;
    clusterStreamingAccumulateUnloadStats(streamingStats, age);
}
