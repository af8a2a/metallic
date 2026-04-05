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
};

static const uint kGpuSceneInstanceVisible = 1u << 0;
static const uint kGpuSceneInstanceHasLod = 1u << 1;

uint gpuDrivenLoadWorklistCount(RWByteAddressBuffer counterBuffer) {
    return counterBuffer.Load(GPU_DRIVEN_COUNT_OFFSET_BYTES);
}

uint gpuDrivenAppendWorkItemSlot(RWByteAddressBuffer counterBuffer) {
    uint slot = 0u;
    counterBuffer.InterlockedAdd(GPU_DRIVEN_COUNT_OFFSET_BYTES, 1u, slot);
    return slot;
}

void gpuDrivenStoreIndirectGridArgs1D(RWByteAddressBuffer counterBuffer, uint dispatchX) {
    counterBuffer.Store(GPU_DRIVEN_DISPATCH_X_OFFSET_BYTES, dispatchX);
    counterBuffer.Store(GPU_DRIVEN_DISPATCH_Y_OFFSET_BYTES, 1u);
    counterBuffer.Store(GPU_DRIVEN_DISPATCH_Z_OFFSET_BYTES, 1u);
}

void gpuDrivenResetWorklistCount(RWByteAddressBuffer counterBuffer) {
    counterBuffer.Store(GPU_DRIVEN_COUNT_OFFSET_BYTES, 0u);
}

void gpuDrivenBuildIndirectGridArgs1DFromCount(RWByteAddressBuffer counterBuffer) {
    uint count = gpuDrivenLoadWorklistCount(counterBuffer);
    gpuDrivenStoreIndirectGridArgs1D(counterBuffer, count);
    gpuDrivenResetWorklistCount(counterBuffer);
}

uint gpuDrivenLoadCompactionCount(RWByteAddressBuffer counterBuffer) {
    return gpuDrivenLoadWorklistCount(counterBuffer);
}

uint gpuDrivenAppendCompactionSlot(RWByteAddressBuffer counterBuffer) {
    return gpuDrivenAppendWorkItemSlot(counterBuffer);
}

void gpuDrivenStoreDispatchArgs1D(RWByteAddressBuffer counterBuffer, uint dispatchX) {
    gpuDrivenStoreIndirectGridArgs1D(counterBuffer, dispatchX);
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
