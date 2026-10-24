# Single-wave LOD frontier

The complete 32-thread, per-instance Frontier path is enabled by default on
supported configurations. Set the render-pass runtime property
`streamFrontierWave32=false` to retain the 64-thread path explicitly.
`applyStreamRayLODSettings` consumes this property; direct callers can set
`MeshletStreamFrameDesc::frontierWave32`, which also defaults to true.

Selection requires distributed demand with a split Frontier/Mask dispatch and
a device exposing fixed subgroup size 32 with compute ballot/arithmetic support.
Other configurations retain `streamCooperativeLodMain`. Clear, Mask, Emit and
Prefetch always retain their 64-thread entry; the new entry is only dispatched
for Frontier. Both pipelines are created during initialization on supported
devices, so switching modes does not compile a pipeline during measurement.

`streamCooperativeLod32Main` uses the same traversal body and per-instance state.
A leaf with more than 32 groups is processed in two ordered batches. Wave
prefix/sum replaces only Frontier's cross-wave LDS scan. The sparse list keeps
its original descending local-group order, and the second batch starts after
the first batch's selected count. Groups within a leaf have the same LOD and
cannot depend on one another. Cross-LOD and final device-memory publication
barriers remain intact. Generic Mask/Emit/RT prefix helpers are unchanged.

The mode neither resets history nor changes RT ownership, residency, budgets or
cache invalidation keys. Command recording uses the existing submission and
resource-retention path; no additional allocation or CPU instance scan is
required. Runtime reset clears the selected-mode diagnostic flag.

`SceneStreamingProfile` and the full-roam benchmark export the effective mode,
workgroup size, device subgroup size and selected shader module content hash.
The hash is the RHI's length-prefixed device SPIR-V identity, not a SHA-256 file
digest. `detailDemandComparison.frontier32=true` with `candidate="C3"` and
`packingOnly=true` runs fixed late-history A1/B1/B2/A2 arms. It requires packed
history and sparse history writes to remain disabled. Deferred diagnostics
restore the corresponding execution mode without resetting owner/history state.

The `meshlet_lod_stream_frontier32` RHI test compares both entries with independent
per-leaf reference demand and continuous-frame state. It covers production BVH
leaves and 1/8/31/32/33/63/64-group boundaries, missing pages/terminal groups,
visibility transitions, manual LOD, history token reset/wrap and an unsubmitted
recording followed by retry. Unsupported fixed Wave32 devices report a skip.
