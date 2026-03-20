// Build indirect dispatch arguments from atomic counter.
// Dispatched as a single thread after meshlet culling.
// Phase 0 (after phase1 cull): build indirect args for visible (offset 0→4)
//   AND for occlusion-failed (offset 16→20), then reset counter[0].
// Phase 1 (after phase2 cull): build indirect args for visible (offset 0→4),
//   then reset counter[0].

struct BuildIndirectUniforms {
    uint phase; // 0 = after phase1, 1 = after phase2
};

ConstantBuffer<BuildIndirectUniforms> buildUniforms; // buffer(0) — constant buffer
RWByteAddressBuffer counters;                        // buffer(1)

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain() {
    // Always build indirect args from visible counter at offset 0
    uint visibleCount = counters.Load(0);
    counters.Store(4, visibleCount);   // threadgroupsPerGrid.x
    counters.Store(8, 1u);             // threadgroupsPerGrid.y
    counters.Store(12, 1u);            // threadgroupsPerGrid.z
    // Reset visible counter for next dispatch
    counters.Store(0, 0u);

    if (buildUniforms.phase == 0u) {
        // Phase 0: also build indirect args for occlusion-failed meshlets
        uint failedCount = counters.Load(16);
        counters.Store(20, failedCount); // threadgroupsPerGrid.x for phase2 dispatch
        counters.Store(24, 1u);          // threadgroupsPerGrid.y
        counters.Store(28, 1u);          // threadgroupsPerGrid.z
        // Do NOT reset failed counter here — phase 2 cull reads it as dispatch size
    }
}
