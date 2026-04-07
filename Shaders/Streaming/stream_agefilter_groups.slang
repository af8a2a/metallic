#include "../Shared/gpu_driven_helpers.slang"

struct StreamingAgeFilterUniforms {
    uint groupCount;
    uint ageThreshold;
    uint reserved0;
    uint reserved1;
};

ConstantBuffer<StreamingAgeFilterUniforms> streamingUniforms; // buffer(GPU_DRIVEN_STREAMING_AGE_UNIFORMS_BINDING)
ByteAddressBuffer                          groupResidency;    // buffer(GPU_DRIVEN_STREAMING_AGE_GROUP_RESIDENCY_BINDING)
RWStructuredBuffer<uint>                   groupAgeBuffer;    // buffer(GPU_DRIVEN_STREAMING_AGE_GROUP_AGE_BINDING)
RWStructuredBuffer<uint>                   unloadRequests;    // buffer(GPU_DRIVEN_STREAMING_AGE_UNLOAD_REQUEST_OUTPUT_BINDING)
RWByteAddressBuffer                        unloadRequestState; // buffer(GPU_DRIVEN_STREAMING_AGE_UNLOAD_REQUEST_STATE_BINDING)

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    const uint groupIndex = dispatchThreadID.x;
    if (groupIndex >= streamingUniforms.groupCount) {
        return;
    }

    const uint residencyState = groupResidency.Load(groupIndex * 4u);
    const bool resident = (residencyState & kClusterLodGroupResidencyResident) != 0u;
    const bool alwaysResident = (residencyState & kClusterLodGroupResidencyAlwaysResident) != 0u;
    const bool touchedThisFrame = (residencyState & kClusterLodGroupResidencyRequested) != 0u;

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
    unloadRequests[slot] = groupIndex;
}
