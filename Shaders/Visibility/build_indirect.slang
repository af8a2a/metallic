// Publish a mesh-dispatch command from the worklist write cursor.
// The cursor is preserved so a second cull pass can append into the same
// worklist after the first pass has already emitted indirect draw args.

#include "../Shared/gpu_driven_helpers.slang"

RWByteAddressBuffer worklistState; // buffer(GPU_DRIVEN_BUILD_DISPATCH_COUNTER_BINDING)

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain() {
    gpuDrivenPublishMeshDispatchIndirectArgs1DFromWriteCursor(worklistState);
}
