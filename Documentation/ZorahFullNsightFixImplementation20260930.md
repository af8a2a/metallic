# ZorahFull Nsight repair implementation - 2026-09-30

The renderer now accumulates texture demand in Device storage, reads it back
asynchronously at the RenderGraph epilogue, and defaults immutable meshlet groups
and LOD topology to Device storage in RenderGraph stream sessions. Nsight's
SDK-default injection policy is retained. The resource changes are applied to
production sources; the temporary automatic capture-export harness is removed.
The earlier [evaluation](ZorahFullNsightFixEvaluation20260930.md) records the
first feedback candidate and its restoration before this implementation.

## Runtime contracts

Texture feedback retains the existing eight-word layout. The first consumer
copies a HostUpload seed into Device storage; subsequent consumers share that
buffer and texture generation. A single epilogue copies accumulated demand to
HostReadback after every graph segment. CPU consumes only completed, accepted
readbacks on later frames. The readback has an independent submission transaction:
an accepted prefix with a cancelled copy tail cannot publish stale demand.
In-flight ownership, frozen publication and sampling mip transitions are retained.
Both executor entry points include the epilogue, including streamer-only graphs.
Feedback consumers retain the existing Graphics/opaque scheduling contract, which
orders seed initialization and subsequent consumers even without a graph edge.
Async feedback across different queues is outside the supported opt-in contract
and was not validated by these tests.

Immutable groups and topology are uploaded once through the initial loader,
using at most 64 MiB of staging in each batch. Accepted copy transactions advance
the offset; cancelled transactions retry it. Staging cannot be overwritten until
completion, and allocation leases survive submission. TransferWrite-to-ShaderRead
barriers and the final completion dependency cover graphics/compute consumers.
Traversal and scene readiness wait for initialization. CPU seed vectors and
staging are released after their final accepted upload/completion, respectively.
With `initialLoad=false`, metadata-only initialization preserves lazy root pages.

Both stream descriptor factories and the editor roam benchmark default
`deviceImmutableMetadata` to true. Explicit false remains available for Host
comparison. Raw `MeshletStreamRuntimeDesc` defaults to false to preserve direct
callers' initialize-to-frame contract; Device callers must first pump the initial
loader, with metadata-only mode when appropriate. Dynamic instances, per-frame
parameters and the separate nodes worker path retain their existing storage.
The supplementary Deferred path-tracing variant remains disabled in these runs.

## Same-binary measurements

Evidence: `build/nsight-fix-implementation-20260930-04/`. All seven candidate
experiments used binary SHA256
`557BAD7E573EA61176550FB2669F245CFF0308C5CC67DD0B6C07097B4EA4862A`
and identical shader-source digest. Each paired group differed only in
`deviceImmutableMetadata`, verified against manifests and capture settings.
Both variants already contained the permanent texture-feedback repair.

Fixed RenderGraph output **2560 x 1440**, DLSS Quality internal **1707 x 960**,
fixed sample camera, temporal jitter, live streaming, VSync, hidden editor,
cached assets/shaders, 40 seconds post-readiness warmup and 20 seconds measurement.
GPU experiments ran serially, with fresh processes and directories and frozen
source/binaries. Hardware: RTX 5070 Ti, driver 616.92, 16,303 MiB reported memory.

| Mode / metadata | Frames | Frame mean / P95 ms | Graph GPU ms | Deferred shading ms | Traversal ms |
|---|---:|---:|---:|---:|---:|
| Nsight / Host | 393 | 50.933 / 68.403 | 49.097 | 3.342 | 35.761 |
| Nsight / Device | 483 | 41.449 / 66.296 | 35.746 | 3.378 | 22.334 |
| Nsight / Host repeat | 351 | 56.981 / 78.950 | 47.891 | 3.250 | 35.137 |
| Normal / Host | 1346 | 14.861 / 17.292 | 11.959 | 0.914 | 2.695 |
| Normal / Device | 1333 | 15.010 / 17.541 | 11.846 | 0.896 | 2.669 |

LOD frontier fell from 7.053/7.038 ms in the Host runs to 1.431 ms with Device;
LOD mask from 4.677/4.548 to 0.609 ms, emit from 3.629/3.459 to 0.585 ms.
Residual injected traversal is dominated by BLAS count/setup/reset at
7.584/3.823/2.371 ms. These scopes did not improve with this migration. The
injected sample still exceeds the 33.33 ms frame threshold.

Frame time is the editor start-to-start loop and includes recording, waits,
Reflex pacing and present. Graph GPU excludes editor composition/present and
independent uploads. GPU scopes are inclusive and must not be summed. The Host
repeat had much more observed Reflex sleep, explaining why a similar Graph GPU
interval does not imply similar frame time. Different fresh processes can refine
different per-image mips and experience different background work; no precise
causal frame-speedup ratio or confidence interval is claimed. Normal Device
frame mean was 0.149 ms higher, while its Graph GPU interval was 0.114 ms lower;
these observations show no substantial GPU regression in this workload.

## Memory and live correctness

Actual groups plus topology payload and allocation requirements were
**298,294,832 bytes (284.476 MiB)**, exceeding the earlier refinement-only lower
bound. Every measured Device frame reported metadata ready, all bytes submitted,
five upload batches, and zero remaining staging bytes. This is requested storage
and allocator accounting, not independent proof of physical residency. Loader
and driver budget logs are retained for placement and peak inspection.

Live feedback, texture migration, upload GPU timing and request/submit/completion
ordering remained active. Required Deferred and RTAS GPU scope coverage was
complete. No load failures, request overflows, invalid BLAS groups, BLAS budget
rejections or device loss were observed. Fixed-view runs reported one repeated
allocation/admission denial per maintenance frame with a full resident budget;
this counter resets each frame and counts repeated attempts, not unique pages or
fatal failures. It combines budget denials and allocation failures, so the
individual branch cannot be identified from the counter alone. The moving
validation run reported zero such denials. Missing-CLAS
and publication-invalidation state counters remain visible in the raw report.

Card-memory and GPU Engine CSVs include load, warmup, measurement and teardown.
Reports lack an exact UTC origin for the relative measurement timestamps; cached
driver-status timestamps cannot supply one. Whole-runtime peaks are therefore
reported as whole-runtime peaks. Missing thresholded engine rows are not zero,
and unrelated process utilization is not added across engines. Detailed audits
and counters are in `ImplementationComparison.json/.md`.

## Validation and actual capture

Final MSVC/Ninja Release builds succeeded for `Metallic`,
`MetallicGPUDrivenSample` and `MetallicRhiTests` in their existing configured trees.
Seven real Vulkan tests passed with bindless descriptors and core validation,
without skips:

- `ktx2_texture_streaming`, `ktx2_texture_streaming_sampling_stability`, and
  `ktx2_texture_feedback_submission_contract`.
- `render_graph_texture_feedback_epilogue`, including both executor entry points.
- `stream_initial_loading`, including cancellation, accepted-prefix/cancelled-tail
  retry, metadata-only lazy roots, staging limits and exact Host/Device GPU cuts.
- `render_graph_gpu_driven_preview_pass_smoke` and
  `render_graph_gpu_driven_streamasset_pass_smoke`, covering both default factories.

The first added test run exposed two fixture errors: mip 2-to-0 correctly requires
two upgrades, and debug readback must be enabled before buffer creation to obtain
TransferSource usage. Both fixtures were corrected while retaining strict checks.
Raw failed and successful logs are preserved. Loader warnings about two missing
unrelated layer manifests remain; older validation-layer OMM limitations select
the supported shader path. No separate synchronization-validation run is claimed.

A full-scene moving route ran for 30 seconds after 20 seconds warmup, with 95
measured frames under debug control/core validation. Metadata and live streaming
audits passed without VUID or device loss. Validation/debug overhead makes its
frame times unsuitable as a production baseline. This bounded run does not
establish long-session or scene-reload stability.

The candidate exported one SDK frame after a separate five-second measurement:
`Captures/NsightGraphics/MetallicGPUDrivenSample_2026_09_30_20_13_22.ngfx-capture`,
**13,217,387,928 bytes**, SHA256
`6694D321107DF08A82462A4B066B0622271DB5AD04F27A12F091389C4A40DA1D`.
Export took 26.626 seconds. Official Nsight Graphics 2026.3.1 completed three
hidden replay loops with exit 0; error metadata found no messages at severity >=2.
Initialization took 99.7 seconds and process runtime 102.106 seconds, which are
resource reconstruction/runtime values, not GPU frame time. Whole-replay sampled
card-memory peak was 12,661 MiB; replay resource data was 8.31 GiB with 325 MiB
reset data. All logs and diagnostics are preserved in `device-replay/`.

The embedded final-present screenshot was inspected and contains the rendered
scene. Its **2400 x 1350 swapchain** differs from the fixed 2560 x 1440 RenderGraph
output. It is captured pixels, not newly rendered replay pixels. The replayer
emitted version/extension/NvAPI diagnostics and its NGX SuperSampling artifact
warning; successful replay does not establish pixel-correct NGX replay.

NVIDIA documents system-memory demotion as the default HVVM workaround. That
policy remains in place, with no CPU-hash or capture/replay-memory workaround.
[Official capture/replay CLI reference](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-cli.html)

The final production binary is rebuilt after enabling factory defaults and
removing the automatic export harness. Its independent default-configuration
acceptance evidence is in `final-default-normal/`; it is not pooled into the
same-binary A/B group. Without a metadata override, it reported Device metadata
active and completed 1,328 measured frames: mean **15.069 ms**, P95 **17.659 ms**,
maximum **53.250 ms**. One frame exceeded 33.33 ms, so the strict sampled-route
30-FPS flag is false; mean/P95 do not establish an uninterrupted frame guarantee.
Its Graph GPU mean was 11.995 ms, Deferred shading 0.905 ms and traversal
2.712 ms. Independent manifest, counter and test audits are in
`FinalAcceptance.json/.md`. A separate three-process measurement of the established
production roam route is recorded in the
[post-repair 1440p baseline](ZorahFullPerformanceBaselineAfterNsightFix20260930.md).
Previous benchmark/freeze changes, asset caches and the
unrelated External working tree remain preserved. Generated captures and
measurement files remain local output.
