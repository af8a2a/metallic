# ZorahFull Nsight repair evaluation - 2026-09-30

This is the historical first-candidate evaluation and restoration record. The
subsequent authorized implementation, including immutable Device metadata, is
documented in [the implementation report](ZorahFullNsightFixImplementation20260930.md).

The recommended first repair is **Device texture-demand accumulation with asynchronous HostReadback staging at the RenderGraph epilogue**. A working candidate was built and tested against live ZorahFull texture streaming, real Vulkan regression tests, and an actual Nsight Graphics capture/replay. The candidate and temporary capture harness are archived as patches; the production sources and executables are restored after evaluation.

This addresses the dominant Deferred resource path identified in [the root-cause investigation](ZorahFullNsightRootCause20260930.md). It does not eliminate the separate streaming-traversal cost associated with Nsight's default host-visible memory policy. A second, unimplemented candidate is described below.

## First candidate and its contracts

Each in-flight feedback entry owns three buffers:

1. CPU initializes the existing eight-word-per-texture metadata in a HostUpload / TransferSource seed.
2. The first consumer copies the seed into Device Storage / TransferSource / TransferDestination, then exposes it to shader reads and atomics. Later consumers in the same frame share the initialized Device buffer and accumulated demand.
3. After all declared consumers finish, the graph epilogue copies Device feedback into HostReadback / TransferDestination. CPU consumes it on a later frame, after accepted GPU completion.

The shader layout, source dimensions, demand atomics, sampling LOD floor, texture generations, refinement policy and budget stay intact. There is no new render-loop wait, per-pass early readback, or forced feedback disable. Initialization and readback have explicit HostWrite/TransferRead, TransferWrite/ShaderReadWrite, ShaderReadWrite/TransferRead and TransferWrite/HostRead dependencies.

An accepted frame prefix is insufficient proof that the readback tail was submitted. The candidate attaches a separate SubmissionTransaction to the copy command buffer and requires both its acceptance and aggregate frame completion before CPU consumption. Cancelled or absent copies never publish stale staging contents. Buffer reuse waits for aggregate completion; frame retention protects seed, Device, readback and texture-generation lifetimes. Frozen publication still retains valid delayed demand and immutable sampling metadata.

Streamer records one epilogue copy per shared scene-resource owner and retains the original begin frame index. Both RenderGraph entry points include this epilogue in the GPU envelope. The submitted entry point creates subsystem boundaries even for a graph whose only requirement is streamer texture feedback, and joins earlier segments before the copy.

This implementation preserves the current graphics-queue consumers. Extending texture feedback to an asynchronous compute consumer requires additional initialization dependencies and queue-access declarations; graph-end joining alone is insufficient. The streamer-only scheduling branch was reviewed statically; full-scene runtime exercised the production Deferred graph, and the regression test exercised independent command-buffer consumers.

## Workload and measurement limits

Evidence root: `build/nsight-fix-evaluation-20260930-03/`. Output **2560 x 1440**, DLSS Quality internal **1707 x 960**, fixed sample camera, temporal jitter, VSync, two frame slots, hidden window, validation off for performance runs, cached assets/shaders, 40-second post-readiness warmup and 20-second measurement. Supplementary Deferred path tracing remains disabled. GPU experiments ran serially in fresh processes and output directories. Shaders and binaries did not change during a run.

Frame ms is the start-to-start editor loop, including waits, CPU recording, pacing and presentation. GPU scopes are inclusive RenderGraph intervals; the envelope excludes editor composition/present and independent texture-upload submissions. Do not add nested timings. Live processes refine different per-image mip working sets, so cross-process differences are descriptive samples, not a precise isolated speedup ratio.

The initial baseline had sustained Edge video/3D and DWM activity; the candidate had substantially less background competition. Approximate measurement-window clocks were comparable (2917 versus 2915 MHz), and temperatures were 58-60 C. The monitored process counters are thresholded per-engine observations: missing rows are not zero, and percentages across engines/PIDs must not be added. See `EvidenceEnvironment.md/.json`. The earlier same-process Host/Device/off repeat experiment supplies the stronger causal evidence for the feedback path.

The final table and counter audit are generated from raw reports in `Comparison.md/.json` by `AnalyzeCandidate.py`. All five runs completed with zero report-integrity errors.

| Implementation / mode | Frames | Frame mean / P95 ms | Graph GPU mean ms | Deferred shading GPU mean ms | Traversal GPU mean ms |
|---|---:|---:|---:|---:|---:|
| Original, initial Nsight baseline | 159 | 126.133 / 141.633 | 125.773 | 82.264 | 33.982 |
| Original, restored Nsight baseline | 181 | 110.620 / 130.657 | 110.334 | 72.369 | 29.532 |
| Candidate, Nsight | 431 | 46.470 / 59.537 | 41.688 | 2.795 | 30.556 |
| Original, restored normal | 648 | 30.891 / 33.498 | 20.286 | 10.734 | 2.397 |
| Candidate, normal | 1776 | 11.263 / 12.974 | 11.099 | 0.811 | 2.741 |

The restored injected run again had high Deferred cost, with little Edge/DWM activity and no qualifying Edge video-decode rows, unlike the initial baseline. Its System copy-engine observations included spikes, whose attribution is unresolved. This repeat supports the feedback-path conclusion while retaining the cross-process limitations. The two normal runs also have different heat/power states and minor geometry/texture working-set differences; the faster candidate was hotter and clocked slightly lower. None of these measurements is converted into a precise causal speedup ratio.

The injected candidate still fails the sampled 30-FPS threshold because its live traversal remains approximately 30.6 ms. Its 46.5 ms loop is a partial repair of capture-ready slowdown, not a return to ordinary-mode performance. The candidate normal sample meets the sampled threshold; the restored normal P95 is 33.498 ms and does not meet it.

## Live correctness

The injected candidate consumed 430 additional feedback frames during its 431-frame measurement and published 125 additional texture upgrades. Refined images progressed from 779 to 829; texture resident bytes progressed from 289,075,712 to 310,931,968. Retired bytes ended at zero; the maximum resident + retired + pending allocation was 311,045,632 bytes against the 536,870,912-byte budget. Geometry remained at 61,661 resident pages / 3,758,094,848 bytes, with no load failures or request overflows. Freeze was disabled.

The candidate normal run consumed 1,775 additional feedback frames and continued upgrades/downgrades, with 848 refined images and no budget violations. The restored normal run also retained live feedback/publication: 647 additional feedback frames, 17 upgrades, refined images 839 to 848, and retired/pending bytes zero at the measurement end. All measured cases had accepted upload frame ordering and complete upload GPU timing. Neither optional path-tracing-on variants nor long moving-camera/scene-reload stress were exercised by this fixed-camera evaluation.

All observed upload records maintained requestFrame <= submitFrame <= completionFrame and had GPU timing. These counters show functioning demand consumption, decode/upload/publication and retirement. They do not prove identical per-image mip residency or pixel equality. Upload request-to-publication frame latency excludes pre-admission budget waiting; host-observed completion latency is not GPU duration. Frame-number fields from different report domains have different origins and must not be subtracted across domains.

Three real Vulkan tests passed without skips, using bindless descriptors and Vulkan validation:

- `RHIRendering.ktx2_texture_streaming`: cancelled unsubmitted work, shared budget, visible refinement, MASK policy, logical IDs, frozen publication/resume and cold downgrade/retirement.
- `RHIRendering.ktx2_texture_streaming_sampling_stability`: actual GPU texture sampling through mip transitions and reveal.
- `RHIRendering.ktx2_texture_feedback_submission_contract`: accepted prefix with cancelled readback tail does not trigger refinement; two independent consumers share demand and the epilogue includes the later consumer's finer request.

Builds used the existing x64 MSVC/Ninja Release configurations: `Metallic` and `MetallicGPUDrivenSample` in `build-release`, and `MetallicRHITests` in `build-scheduling-release`. Test command:

```powershell
.\build-scheduling-release\tests\MetallicRHITests.exe --rhi-bindless --rhi-validation `
  '--gtest_filter=*ktx2_texture_streaming*:*ktx2_texture_feedback_submission_contract*' `
  --output-dir build/nsight-fix-evaluation-20260930-03/rhi-normal
```

The Vulkan loader warned about two unrelated missing layer manifests; no VUID/validation error was recorded. This run enabled core validation, not a separate synchronization-validation experiment.

## Actual capture and replay

The temporary, default-off benchmark harness requests one SDK frame after measurement, while the fixed graph output extent remains active. Nsight SDK/default injection policy was retained; CPU-hash and capture/replay-memory policy workarounds were absent.

Capture: `Captures/NsightGraphics/MetallicGPUDrivenSample_2026_09_30_18_58_03.ngfx-capture`, 13,299,273,672 bytes, SHA256 `1D5FFC4725C3DC2D918015263B0B7EC5672604A2D4B9DFFAEA2A4B67A66F6BD4`. SDK export completed in approximately 49 seconds. The same installed Nsight Graphics 2026.3.1 replayer completed three hidden replay loops with exit code 0 and no DeviceLost. Capture error metadata reported no messages at severity >= 2. Replay initialization took about 99 seconds; this is resource reconstruction time, not a GPU-frame timing or performance baseline.

`candidate-replay/CapturedFinalPresent.png` is the embedded capture-side final-present image. It was inspected and contains the rendered scene. It is not an image newly rendered by replay, and replay pixels were not compared. The replayer emitted its NGX SuperSampling artifact warning; successful API replay does not establish pixel-correct NGX replay. Logs, identity and return codes are retained in `candidate-replay/`.

NVIDIA documents HVVM demotion to system memory as the default workaround for capture/update-tracking limitations. Keep this policy during repair qualification, since the earlier CPU-hash workaround failed replay. [Official Graphics Capture CLI reference](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-cli.html)

## Second candidate: immutable traversal metadata

The residual injected cost remains dominated by streaming traversal. The next single-variable experiment should move immutable **groups + LOD topology** into Device storage and upload them once before traversal begins. Current cooperative Frontier/Emit/Prefetch repeatedly read these tables. The nodes table is used by another worker path and should be evaluated separately. Keep dynamically mapped instances and per-frame parameters on their existing paths.

Current asset counts and verified C++ strides give groups **106.078 MiB** and LOD refinement words alone **134.729 MiB**. The final topology also contains parents/BVH/tile/task data, so **240.807 MiB is only a payload lower bound** for groups plus topology. Adding nodes gives a lower bound of 318.549 MiB. Query actual buffer sizes/allocation requirements and peak budgets before acceptance; these values are not a guaranteed net VRAM increase.

Use the initial-loader handoff, bounded 64 MiB staging batches, completion-held leases and submission transactions, TransferWrite-to-ShaderRead synchronization, and graphics/compute queue access. Cancellation must retry initialization rather than expose undefined contents. Account for old/new scene overlap, device allocations and capture-tool memory pressure. Detailed source pointers and design are in `GeometryCandidate.md`.

This candidate has not been implemented, measured or captured/replayed. It is a follow-up experiment, not a validated remaining-speedup claim.

## Reviewable artifacts and final state

- `Candidate.patch`: five runtime files plus the Vulkan regression test; applies cleanly to the restored sources.
- `CaptureHarness.patch`: separate temporary benchmark export harness; excludes the previously retained freeze/analyzer changes.
- `CandidateBuildManifest.json`, `RestoredSourceManifest.json`: candidate binary/patch identity and restored-source backup checks.
- `rhi-normal/`, `candidate-injected-capture/`, `candidate-replay/`: real test/capture/replay evidence.
- Each live run retains Manifest/Config/Capture, Frames/Uploads, stdout/stderr and GPU/process samples.

The candidate is archived for review rather than left in the default renderer. Previous user/editor benchmark changes, offline analyzer changes, asset caches and the unrelated External working tree are preserved. Both production executables and the RHI test executable are rebuilt after restoration. Restoration build logs and a final binary-string check are retained in the evidence root; neither production executable nor the test binary contains the candidate feedback/profile or temporary capture-harness strings. Runtime/test source diffs are empty, restored hashes match every backup, and both patches pass `git apply --check`. The final restored normal run completed with injection false, shader mode disabled, freeze false and diagnostic false. No test, renderer or replayer process remains running.
