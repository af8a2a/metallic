#include "../Shared/gpu_driven_helpers.slang"

struct StreamingUpdateUniforms {
    uint patchCount;
    uint copySourceData;
    uint reserved1;
    uint reserved2;
};

ConstantBuffer<StreamingUpdateUniforms> updateUniforms; // buffer(GPU_DRIVEN_STREAMING_UPDATE_UNIFORMS_BINDING)
StructuredBuffer<uint>                  sourceGroupMeshletIndices; // buffer(GPU_DRIVEN_STREAMING_UPDATE_SOURCE_GROUP_MESHLET_INDICES_BINDING)
RWStructuredBuffer<uint>                residentGroupMeshletIndices; // buffer(GPU_DRIVEN_STREAMING_UPDATE_RESIDENT_GROUP_MESHLET_INDICES_BINDING)
RWStructuredBuffer<uint64_t>            lodGroupPageTable; // buffer(GPU_DRIVEN_STREAMING_UPDATE_GROUP_PAGE_TABLE_BINDING)
StructuredBuffer<StreamingPatch>        patches; // buffer(GPU_DRIVEN_STREAMING_UPDATE_PATCHES_BINDING)

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

    if (!clusterLodGroupPageAddressIsValid(patch.residentHeapOffset)) {
        lodGroupPageTable[patch.groupIndex] = patch.residentHeapOffset;
        return;
    }

    const uint residentHeapOffset = (uint)patch.residentHeapOffset;
    if (updateUniforms.copySourceData != 0u) {
        for (uint clusterIndex = 0u; clusterIndex < patch.clusterCount; ++clusterIndex) {
            residentGroupMeshletIndices[residentHeapOffset + clusterIndex] =
                sourceGroupMeshletIndices[patch.clusterStart + clusterIndex];
        }
    }
    lodGroupPageTable[patch.groupIndex] = patch.residentHeapOffset;
}
