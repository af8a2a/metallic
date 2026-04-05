// Build indirect dispatch arguments from atomic counter.
// Dispatched as a single thread after meshlet culling.

#include "../Shared/gpu_driven_helpers.slang"

RWByteAddressBuffer counters; // buffer(GPU_DRIVEN_BUILD_DISPATCH_COUNTER_BINDING)

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain() {
    gpuDrivenBuildDispatchArgs1DFromCount(counters);
}
