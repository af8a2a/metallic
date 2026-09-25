# ZorahFull: pre-pacing streaming maintenance (2026-09-25)

Implemented camera-independent maintenance before Reflex Sleep and incremental request priority scoring. The measured benefit is a shorter post-Sleep submission path. These captures do **not** establish improved whole-frame throughput or elimination of GPU scheduling gaps.

## Implementation and synchronization

- `EditorApplication::waitForFrameSlotBeforeInput()` calls `StreamerSubsystem::prepareBeforePacing()` after the existing frame-slot wait, before Reflex Sleep and input sampling. The separate `Streamer maintenance before pacing` CPU scope includes its nested maintenance timings in the editor profiler; the work remains counted in Frame.
- `MeshletStreamRuntime::prepareMaintenance()` polls completed residency tasks, consumes completed GPU demand, refreshes/adopts requests, makes joint cold-page reclamation decisions, and discards obsolete CPU CLAS plans. It is idempotent until `cmdBeginFrame()` consumes the preparation. Direct/non-editor callers retain a fallback at the old point.
- GPU-visible CLAS completion publication/retirement, page-table patches, staging uploads and command recording remain behind the original RenderGraph dependencies. Texture command preparation remains there too. No camera-dependent traversal or GPU-visible write is moved across those dependencies.
- Tracked request feedback uses a bounded `queuedFrameCount + 1` readback ring. Each copy has a submission transaction and frame completion point. Consume only the latest submitted, completed, non-cancelled feedback, never a merely recorded/cancelled buffer. Do not wait when the ring is full; skip that copy. Legacy untracked callers retain the original readback path under their existing synchronization contract. Full adds about 3.15 MiB of host readback buffers.
- Minimized editor and benchmark streaming freeze bypass the early phase. Runtime freeze also prevents maintenance. No render pass acquires new loading responsibilities.
- Priority cache is sparse and keyed by stable asset page ID. Only admissible candidates are scored. Recalculate when screen benefit or the existing saturated age changes; the immutable page byte size is valid for the cache's asset lifetime. Reset destroys the cache. Demand/prefetch ordering, prefetch LOD ordering, age saturation at 120 frames and page-ID tie-breaks are unchanged.
- Liveness/eligibility still require a scan, and the eligible heap is still rebuilt. This is incremental **score computation**, not a persistent heap or elimination of all request traversal. `priorityRecomputed` / `priorityReused` are exported to `Frames.jsonl`.

Diagnostic switch, read once at process startup: `METALLIC_STREAM_MAINTENANCE_BEFORE_PACING=0` restores the old maintenance phase; absent or `1` enables the new default. Both cases retain the safe feedback ring and priority cache, so the following A/B isolates phase placement, not the independent speedup of caching.

## Same-binary phase comparison

Four runs in order: control, enabled, enabled-repeat, control-repeat. All use Release executable SHA256 `BAB81C81B14BB690B1025B5BB361F6B6FFC9E4FC537CD53B66D390D997171BD3` and shader SHA256 `82D734E5F689F06CBF4954CE71B62B2DFD275D2FE5B98C444D34CACD984F3927`.

RTX 5070 Ti / driver 616.92; Full material pipeline, LOD 1.5 px, SW threshold 8 px; output 1797x660, render 1198x440, DLSS Quality, Reflex on, FIFO, two frame slots. Each hidden-editor run has 10 s ready warmup and 30 s wall-time fixed-route measurement (6 units forward, turns, return). Matching route/settings are verified in the manifests and captures; this is not a frozen camera/cut/residency workload and sample counts differ.

All values below are milliseconds; average unless explicitly P95.

| Metric | Control | Enabled | Enabled repeat | Control repeat |
|---|---:|---:|---:|---:|
| Frames | 1489 | 1382 | 1385 | 1416 |
| Pre-Sleep maintenance | — | 4.488 | 4.565 | — |
| Post-Sleep Stream Begin CPU | 4.011 | 0.436 | 0.390 | 4.876 |
| Record RenderGraph CPU | 9.329 | 6.452 | 6.641 | 10.784 |
| Reflex Sleep | 7.758 | 7.357 | 7.203 | 7.137 |
| RenderGraph GPU envelope | 19.731 | 21.248 | 21.236 | 20.684 |
| Whole frame average | 20.153 | 21.710 | 21.665 | 21.192 |
| Whole frame P95 | 25.779 | 27.826 | 27.721 | 27.418 |
| Whole frame maximum | 48.489 | 51.118 | 48.943 | 63.700 |
| Feedback age, min/mean/max frames | 2 / 2 / 2 | 2 / 2 / 2 | 2 / 2 / 2 | 2 / 2 / 2 |
| Eligible priority reuse | 24.38% | 23.78% | 24.00% | 24.31% |
| Load failures | 0 | 0 | 0 | 0 |

The early phase removes roughly 3.6–4.5 ms from Stream Begin in these runs. It moves the cost, rather than eliminating it. Whole-frame means were slower in both enabled runs, and P95 was also higher. Reverse control slowed relative to the first control, so the data cannot assign the full difference to this change. GPU envelopes also increase in enabled runs; clocks alone do not explain the first pair (both approximately 2.9 GHz late in the run). Application-exit GPU utilization remained about 4–9%, with about 3 GiB background memory. A future throughput experiment must isolate background activity and align GPU work before attributing the remaining difference. No FPS improvement is claimed.

Driver reports, deduplicated by reported frame ID:

| Driver interval | Control | Enabled | Enabled repeat | Control repeat |
|---|---:|---:|---:|---:|
| Simulation start to GPU start, average | 10.048 | 6.156 | 7.075 | 12.943 |
| Render submit interval, average | 9.394 | 6.428 | 6.302 | 11.043 |
| Report count | 26 | 24 | 25 | 24 |

The first enabled capture has only 22 valid nonnegative Simulation-to-GPU intervals out of 24 reports; the analyzer excludes the two invalid timestamp intervals. Reports are periodically refreshed and cached, not every-frame scheduling traces. These values support shorter input-to-submission delay but cannot quantify GPU idle time or establish input-to-display latency. The driver's `gpuActiveRenderTimeUs` field is a timestamp span in this backend, not a hardware busy-time counter.

## Validation

Release `MetallicGPUDrivenSample` and `MetallicRhiTests` builds succeeded. Eight selected tests passed with Vulkan validation: request selection, screen priority, prefetch admission, budget admission, joint cold reclaim, upload completion, ordered publication retry, and BLAS cut cache. Additional changed-benefit priority invalidation test passed after rebuilding the test executable. Main benchmark executable did not change between the four captures.

The request-selection oracle covers ordering, duplicate promotion, blocked candidates, queued keepalive and reset; the added assertions verify unchanged-score reuse and exactly one candidate invalidated by one benefit change. BLAS lifecycle assertions cover repeated preparation, completed-feedback consumption and cancelled recording rejection. Validation's older layer disables KHR OMM for these focused tests; this is not OMM validation of the Full production path.

Four Full captures completed, with zero streaming load failures and unchanged two-frame feedback age. These are hidden-editor runtime/profile checks; no new interactive image-quality acceptance is claimed. `git diff --check` passed.

## Evidence and reproduction

Local evidence, relative to repository root:

- `build-release/PrePacingComparison20260925.json`: aggregated timings, counters, conditions and driver intervals.
- `build-release/full-prepace-{control,enabled,enabled-repeat,control-repeat}-0925/`: manifests plus `run1/Capture.json`, `Frames.jsonl`, `ReflexSummary.json`, logs and GPU telemetry.
- `build-release/prepace-final-build-0925.log`, `prepace-rhi-0925.log`, `prepace-priority-test-build.log`, `prepace-priority-0925.log`.
- The earlier `full-prepace-before-0925` capture used the old executable and had a 719 ms cold outlier. It is not used to claim tail-latency improvement.

Use a fresh output directory for each run:

```powershell
$env:METALLIC_REFLEX_MODE = 'on'
$env:METALLIC_STREAM_MAINTENANCE_BEFORE_PACING = '1' # '0' for phase control
./Tools/RunZorahFullRoam.ps1 -OutputRoot E:/metallic/build-release/full-prepace-new -Width 1797 -Height 660 -DurationSeconds 30 -WarmupSeconds 10 -Runs 1
python -B Tools/AnalyzeZorahFullReflex.py build-release/full-prepace-new/run1
```

Next optimization should target remaining request liveness/state lookup work (about 0.84–1.06 ms in the first pair), preserving demand keepalive and cold-page correctness. Do not optimize Sleep based on its duration alone: it is pacing, and the current evidence does not isolate a new GPU scheduling gap.
