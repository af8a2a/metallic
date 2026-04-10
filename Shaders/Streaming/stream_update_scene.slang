#include "../Shared/gpu_driven_helpers.slang"

struct StreamingUpdateUniforms {
    uint patchCount;
    uint copySourceData;
    uint sourceGroupMeshletIndexCount;
    uint residentGroupMeshletIndexCount;
    uint groupPageTableCount;
    uint reserved1;
    uint reserved2;
    uint reserved3;
};

ConstantBuffer<StreamingUpdateUniforms> updateUniforms; // buffer(GPU_DRIVEN_STREAMING_UPDATE_UNIFORMS_BINDING)
StructuredBuffer<uint>                  sourceGroupMeshletIndices; // buffer(GPU_DRIVEN_STREAMING_UPDATE_SOURCE_GROUP_MESHLET_INDICES_BINDING)
RWStructuredBuffer<uint>                residentGroupMeshletIndices; // buffer(GPU_DRIVEN_STREAMING_UPDATE_RESIDENT_GROUP_MESHLET_INDICES_BINDING)
RWStructuredBuffer<uint64_t>            lodGroupPageTable; // buffer(GPU_DRIVEN_STREAMING_UPDATE_GROUP_PAGE_TABLE_BINDING)
StructuredBuffer<StreamingPatch>        patches; // buffer(GPU_DRIVEN_STREAMING_UPDATE_PATCHES_BINDING)
RWByteAddressBuffer                     streamingStats; // buffer(GPU_DRIVEN_STREAMING_UPDATE_STATS_BINDING)

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID) {
    const uint patchIndex = dispatchThreadID.x;
    if (patchIndex >= updateUniforms.patchCount) {
        return;
    }

    const StreamingPatch patch = patches[patchIndex];
    if (patch.groupIndex == 0xFFFFFFFFu) {
        return;
    }

    if (patch.groupIndex >= updateUniforms.groupPageTableCount) {
        clusterStreamingIncrementErrorUpdate(streamingStats);
        clusterStreamingIncrementErrorPageTable(streamingStats);
        return;
    }

    if (!clusterLodGroupPageAddressIsValid(patch.residentHeapOffset)) {
        clusterStreamingIncrementAppliedPatchCount(streamingStats);
        lodGroupPageTable[patch.groupIndex] = patch.residentHeapOffset;
        return;
    }

    const uint64_t residentHeapOffset = patch.residentHeapOffset;
    const uint64_t clusterStart = patch.clusterStart;
    const uint64_t clusterCount = patch.clusterCount;

    if ((residentHeapOffset >> 32u) != 0u ||
        residentHeapOffset + clusterCount >
            ((uint64_t)updateUniforms.residentGroupMeshletIndexCount)) {
        clusterStreamingIncrementErrorUpdate(streamingStats);
        clusterStreamingIncrementErrorAllocation(streamingStats);
        return;
    }

    if (updateUniforms.copySourceData != 0u &&
        clusterStart + clusterCount >
            ((uint64_t)updateUniforms.sourceGroupMeshletIndexCount)) {
        clusterStreamingIncrementErrorUpdate(streamingStats);
        return;
    }

    const uint residentHeapOffsetUint = (uint)residentHeapOffset;
    clusterStreamingIncrementAppliedPatchCount(streamingStats);
    if (updateUniforms.copySourceData != 0u) {
        for (uint clusterIndex = 0u; clusterIndex < patch.clusterCount; ++clusterIndex) {
            residentGroupMeshletIndices[residentHeapOffsetUint + clusterIndex] =
                sourceGroupMeshletIndices[patch.clusterStart + clusterIndex];
        }
    }
    lodGroupPageTable[patch.groupIndex] = patch.residentHeapOffset;
}
