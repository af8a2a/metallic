#include "../../Source/Rendering/gpu_driven_constants.h"

struct InstanceData {
    float4x4 worldMatrix;
    float4x4 prevWorldMatrix;
    uint     geometryIndex;
    uint     dispatchStart;
    uint     sceneNodeIndex;
    uint     visibilityFlags;
};

struct GeometryData {
    uint   meshletStart;
    uint   meshletCount;
    uint   primitiveGroupStart;
    uint   primitiveGroupCount;
    uint   indexStart;
    uint   indexCount;
    uint   materialIndex;
    uint   lodRootNode;
    float4 boundsCenterRadius;
};

struct MeshletDrawInfo {
    uint instanceID;
    uint globalMeshletID;
    uint meshletSource;
    uint lodLevel;
};

struct VisibleInstanceInfo {
    uint   sceneInstanceID;
    uint   geometryIndex;
    uint   lodRootNode;
    uint   classificationFlags;
    float4 boundsCenterRadius;
    float4 lodMetric;
};

static const uint kGpuSceneInstanceVisible = 1u << 0;
static const uint kGpuSceneInstanceHasLod = 1u << 1;
static const uint kVisibleInstanceClassificationVisible = 1u << 0;
static const uint kVisibleInstanceClassificationHasLod = 1u << 1;
static const uint kMeshletDrawSourceScene = 0u;
static const uint kMeshletDrawSourceClusterLod = 1u;
static const uint kClusterTraversalStatsHistogramSize = 8u;

uint gpuDrivenLoadWorklistWriteCursor(RWByteAddressBuffer worklistStateBuffer) {
    return worklistStateBuffer.Load(GPU_DRIVEN_WORKLIST_WRITE_CURSOR_OFFSET_BYTES);
}

uint gpuDrivenLoadWorklistProducedCount(RWByteAddressBuffer worklistStateBuffer) {
    return worklistStateBuffer.Load(GPU_DRIVEN_WORKLIST_PRODUCED_COUNT_OFFSET_BYTES);
}

uint gpuDrivenLoadWorklistConsumedCount(RWByteAddressBuffer worklistStateBuffer) {
    return worklistStateBuffer.Load(GPU_DRIVEN_WORKLIST_CONSUMED_COUNT_OFFSET_BYTES);
}

uint gpuDrivenAppendWorkItemSlot(RWByteAddressBuffer worklistStateBuffer) {
    uint slot = 0u;
    worklistStateBuffer.InterlockedAdd(GPU_DRIVEN_WORKLIST_WRITE_CURSOR_OFFSET_BYTES, 1u, slot);
    return slot;
}

uint gpuDrivenConsumeWorkItemSlot(RWByteAddressBuffer worklistStateBuffer) {
    uint slot = 0u;
    worklistStateBuffer.InterlockedAdd(GPU_DRIVEN_WORKLIST_CONSUMED_COUNT_OFFSET_BYTES, 1u, slot);
    return slot;
}

void gpuDrivenStoreWorklistProducedCount(RWByteAddressBuffer worklistStateBuffer, uint count) {
    worklistStateBuffer.Store(GPU_DRIVEN_WORKLIST_PRODUCED_COUNT_OFFSET_BYTES, count);
}

void gpuDrivenResetWorklistWriteCursor(RWByteAddressBuffer worklistStateBuffer) {
    worklistStateBuffer.Store(GPU_DRIVEN_WORKLIST_WRITE_CURSOR_OFFSET_BYTES, 0u);
}

void gpuDrivenResetWorklistProducedCount(RWByteAddressBuffer worklistStateBuffer) {
    worklistStateBuffer.Store(GPU_DRIVEN_WORKLIST_PRODUCED_COUNT_OFFSET_BYTES, 0u);
}

void gpuDrivenResetWorklistConsumedCount(RWByteAddressBuffer worklistStateBuffer) {
    worklistStateBuffer.Store(GPU_DRIVEN_WORKLIST_CONSUMED_COUNT_OFFSET_BYTES, 0u);
}

void gpuDrivenStoreDispatchIndirectArgs1D(RWByteAddressBuffer worklistStateBuffer, uint dispatchX) {
    worklistStateBuffer.Store(GPU_DRIVEN_WORKLIST_DISPATCH_X_OFFSET_BYTES, dispatchX);
    worklistStateBuffer.Store(GPU_DRIVEN_WORKLIST_DISPATCH_Y_OFFSET_BYTES, 1u);
    worklistStateBuffer.Store(GPU_DRIVEN_WORKLIST_DISPATCH_Z_OFFSET_BYTES, 1u);
}

void gpuDrivenStoreMeshDispatchIndirectArgs1D(RWByteAddressBuffer worklistStateBuffer, uint dispatchX) {
    gpuDrivenStoreDispatchIndirectArgs1D(worklistStateBuffer, dispatchX);
}

void gpuDrivenResetDispatchIndirectArgs1D(RWByteAddressBuffer worklistStateBuffer) {
    gpuDrivenStoreDispatchIndirectArgs1D(worklistStateBuffer, 0u);
}

void gpuDrivenResetMeshDispatchIndirectArgs1D(RWByteAddressBuffer worklistStateBuffer) {
    gpuDrivenResetDispatchIndirectArgs1D(worklistStateBuffer);
}

void gpuDrivenResetWorklistState1D(RWByteAddressBuffer worklistStateBuffer) {
    gpuDrivenResetWorklistWriteCursor(worklistStateBuffer);
    gpuDrivenResetWorklistProducedCount(worklistStateBuffer);
    gpuDrivenResetWorklistConsumedCount(worklistStateBuffer);
    gpuDrivenResetDispatchIndirectArgs1D(worklistStateBuffer);
}

void gpuDrivenPublishDispatchIndirectArgs1DFromWriteCursor(RWByteAddressBuffer worklistStateBuffer) {
    uint producedCount = gpuDrivenLoadWorklistWriteCursor(worklistStateBuffer);
    gpuDrivenStoreWorklistProducedCount(worklistStateBuffer, producedCount);
    gpuDrivenResetWorklistConsumedCount(worklistStateBuffer);
    gpuDrivenStoreDispatchIndirectArgs1D(worklistStateBuffer, producedCount);
    gpuDrivenResetWorklistWriteCursor(worklistStateBuffer);
}

void gpuDrivenPublishMeshDispatchIndirectArgs1DFromWriteCursor(RWByteAddressBuffer worklistStateBuffer) {
    gpuDrivenPublishDispatchIndirectArgs1DFromWriteCursor(worklistStateBuffer);
}

uint gpuDrivenLoadWorklistCount(RWByteAddressBuffer counterBuffer) {
    return gpuDrivenLoadWorklistWriteCursor(counterBuffer);
}

void gpuDrivenStoreIndirectGridArgs1D(RWByteAddressBuffer counterBuffer, uint dispatchX) {
    gpuDrivenStoreDispatchIndirectArgs1D(counterBuffer, dispatchX);
}

void gpuDrivenResetWorklistCount(RWByteAddressBuffer counterBuffer) {
    gpuDrivenResetWorklistWriteCursor(counterBuffer);
}

void gpuDrivenBuildIndirectGridArgs1DFromCount(RWByteAddressBuffer counterBuffer) {
    gpuDrivenPublishDispatchIndirectArgs1DFromWriteCursor(counterBuffer);
}

uint gpuDrivenLoadCompactionCount(RWByteAddressBuffer counterBuffer) {
    return gpuDrivenLoadWorklistCount(counterBuffer);
}

uint gpuDrivenAppendCompactionSlot(RWByteAddressBuffer counterBuffer) {
    return gpuDrivenAppendWorkItemSlot(counterBuffer);
}

void gpuDrivenStoreDispatchArgs1D(RWByteAddressBuffer counterBuffer, uint dispatchX) {
    gpuDrivenStoreDispatchIndirectArgs1D(counterBuffer, dispatchX);
}

void gpuDrivenResetCompactionCount(RWByteAddressBuffer counterBuffer) {
    gpuDrivenResetWorklistCount(counterBuffer);
}

void gpuDrivenBuildDispatchArgs1DFromCount(RWByteAddressBuffer counterBuffer) {
    gpuDrivenBuildIndirectGridArgs1DFromCount(counterBuffer);
}

uint gpuDrivenFindInstanceIndexByDispatchOffset(StructuredBuffer<InstanceData> instances,
                                                uint instanceCount,
                                                uint dispatchIndex) {
    uint left = 0u;
    uint right = instanceCount;

    while (left + 1u < right) {
        uint mid = (left + right) >> 1u;
        if (instances[mid].dispatchStart <= dispatchIndex) {
            left = mid;
        } else {
            right = mid;
        }
    }

    return left;
}
