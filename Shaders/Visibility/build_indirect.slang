// Build indirect dispatch arguments from atomic counter.
// Dispatched as a single thread after meshlet culling.

RWByteAddressBuffer counters; // buffer(0): [atomicCounter(4B), indirectArgs(12B)]

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain() {
    uint count = counters.Load(0);
    // Write MTLDispatchThreadgroupsIndirectArguments at offset 4
    counters.Store(4, count);   // threadgroupsPerGrid.x = visible meshlet count
    counters.Store(8, 1u);      // threadgroupsPerGrid.y = 1
    counters.Store(12, 1u);     // threadgroupsPerGrid.z = 1
    // Reset atomic counter for next frame
    counters.Store(0, 0u);
}
