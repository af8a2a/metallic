#include "../../Source/Rendering/gpu_driven_constants.h"

struct InstanceData {
    float4x4 mvp;
    float4x4 modelView;
    float4x4 worldMatrix;
    uint     meshletStart;
    uint     meshletCount;
    uint     instanceID;
    uint     pad;
};

struct MeshletDrawInfo {
    uint instanceID;
    uint globalMeshletID;
};

uint gpuDrivenLoadCompactionCount(RWByteAddressBuffer counterBuffer) {
    return counterBuffer.Load(GPU_DRIVEN_COUNT_OFFSET_BYTES);
}

uint gpuDrivenAppendCompactionSlot(RWByteAddressBuffer counterBuffer) {
    uint slot = 0u;
    counterBuffer.InterlockedAdd(GPU_DRIVEN_COUNT_OFFSET_BYTES, 1u, slot);
    return slot;
}

void gpuDrivenStoreDispatchArgs1D(RWByteAddressBuffer counterBuffer, uint dispatchX) {
    counterBuffer.Store(GPU_DRIVEN_DISPATCH_X_OFFSET_BYTES, dispatchX);
    counterBuffer.Store(GPU_DRIVEN_DISPATCH_Y_OFFSET_BYTES, 1u);
    counterBuffer.Store(GPU_DRIVEN_DISPATCH_Z_OFFSET_BYTES, 1u);
}

void gpuDrivenResetCompactionCount(RWByteAddressBuffer counterBuffer) {
    counterBuffer.Store(GPU_DRIVEN_COUNT_OFFSET_BYTES, 0u);
}

void gpuDrivenBuildDispatchArgs1DFromCount(RWByteAddressBuffer counterBuffer) {
    uint count = gpuDrivenLoadCompactionCount(counterBuffer);
    gpuDrivenStoreDispatchArgs1D(counterBuffer, count);
    gpuDrivenResetCompactionCount(counterBuffer);
}
