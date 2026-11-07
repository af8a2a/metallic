# SuperSponza CPU jobs production acceptance

[ReviewFollowup.md](ReviewFollowup.md) records subsequent PR review fixes and
their separate validation. The historical measurements below were not rerun
and do not establish the revised code's speed or resource peaks.

This report completes the production phase of [issue #73](https://github.com/af8a2a/metallic/issues/73)
after the [M0–M2 evidence](../CPUJobs20261106/Validation.md), within the user's revised sampling budget.
The implementation lives in the isolated `codex/issue-73-cpu-jobs` worktree.
Source/executable identities, invocation order and original records accompany
each experiment. The [fixed protocol](Protocol.json) was declared before the
corresponding acceptance samples; failed and excluded pilots remain separate.

## Workload and comparison

- Source: `E:/metallic/Asset/SuperSponza/NewSponza_Main_glTF_003.gltf`, 405
  primitives, 3,747,018 source triangles, 28 materials and 72 full 4096² PNGs.
  [Assets.json](Assets.json) hashes the scene, buffer, images and prebuilt stream.
- Fixed authored `PhysCamera001`, DLSS-SR Quality: **1280×720 rendering,
  1920×1080 output**. RT main-light shadows, SIGMA, material binning, auto
  exposure and ACES2 remain enabled. Optional DLSS-NR remains disabled by the
  preset. This is a fixed view, without scene edits or a camera route.
- Both executables use the same final Release renderer, compiler, SDKs,
  TaskSystem, graph, settings and caches. Only the benchmark-linked
  `MeshletBuildParallel` adapter differs: original dedicated workers versus the
  shared range runtime. This is **not** an executable comparison against the
  entire issue-start revision, or a production SIMD substitution.
- The renderer reuses the prebuilt stream; it does not cook LODs during these
  frames. The separate fresh-output cooker experiment exercises the changed
  real geometry path. See [the source/window audit](Notes.md).
- Windows 11, Ryzen 7 7800X3D, RTX 5070 Ti 16 GiB, NVIDIA 621.01,
  MSVC 19.44 x64/Ninja Release. [BuildEnvironment.json](BuildEnvironment.json)
  retains build features and dependency roots. No clock or power policy was
  changed. The desktop's observed pre-run whole-GPU memory usage was 6452 MiB,
  GPU utilization 1%, with no other Metallic/Nsight workload running.

The Zorah template's 512 MiB material budget rejected this PNG dataset before
rendering; the [first failure](Pilots/TextureBudget512MiB.log) is retained.
The final SuperSponza configuration explicitly allows 8192 MiB for approximately
6 GiB of full mip allocations. Its geometry pool is 512 MiB: all 3781 cooked
pages need 314,864,384 aligned device bytes and fit together. This replaces the
unrelated Zorah 3.5 GiB capacity in both arms. Payload, LOD error, traversal,
upload batch/concurrency and image quality are unchanged. These are benchmark
dataset settings, not changed application defaults or a quality fallback.

## Validation and output

| Check | Completed result |
| --- | --- |
| Production Task / Scene / kernel regressions | 33 Task tests; 19 range tests ×100 repetitions; 160 Scene passed with 9 explicit asset/USD skips; 6 kernel tests passed |
| Core RHI regression | 26/26 passed, including simultaneous allocation accounting, alias/sparse lifetimes, recording, preparation, joined/pipelined submission and serial/parallel pixels; zero VUID |
| Independent RHI synchronization validation | 18/18 passed in fresh core/async profiles without Streamline; actual validation/messenger/profile evidence retained |
| Full SuperSponza core validation | Four ABBA processes, 60 warmup + 120 measured frames each; actual core validation and messenger active, zero validation warnings, all 480 GPU frame envelopes available |
| Output | Four finite RGBA16F/scRGB images inspected; geometry, materials and shadows agree visually; mean absolute differences are no larger than same-backend repeats, not bitwise identity |
| Runner evidence contracts | 18/18 tests passed, including core-only/sync state, null telemetry, non-overlapping frame intervals and independent process/block statistics |

[ValidationPlots/Report.md](ValidationPlots/Report.md) contains the actual
images and instrumented diagnostics. Cross-backend mean absolute RGB differences
were 0.000032624 and 0.000042759; same-backend repeats were 0.000055593 and
0.000043322. Local maximum differences reach 0.3047, versus up to 0.3008 in
same-backend repeats. These are descriptive comparisons of fixed-view temporal outputs,
not a universal rendering tolerance or proof that every frame is identical.

The full Streamline + synchronization-validation attempt failed at instance
creation: the SDK interposer checks only global extensions and rejects the
validation layer's `VK_EXT_layer_settings`. No scene frame executed in that
attempt. The complete DLSS/SIGMA workload therefore uses core validation;
separate non-Streamline RHI cases establish their own synchronization coverage.
The sparse case has core coverage only. Windows ThreadSanitizer and physical
unsupported-ISA hardware remain unavailable, as in the earlier evidence.

The first complete 600-frame core pilot passed in the application with zero
validation messages, but the old runner incorrectly required synchronization
validation whenever core validation was requested. That schema failure was
preserved, the mode check was corrected, and the final four processes ran in
a new directory. Validation times are excluded from performance conclusions.
An attached debug observer also forces the validation renderer to one effective
preparation lane; its four-worker request is not a four-worker scaling result.

## Final performance and resource results

The user requested fewer repetitions to avoid a long task. The original
56-process frame plan was stopped before inspecting comparative results.
Eight completed prewarm processes remain excluded; the complete W1 ABBA
processes 8–11 are the formal population. The later W2 legacy-only process 12
and interrupted shared process 13 are retained in the original records and
excluded from paired conclusions. No further renderer or test process was
started to fill that partial block. This external budget revision is recorded
in [the reduced metadata](Data/FrameReduced/Metadata.json) and
[the original release evidence](Storage.md), not hidden by claiming
that the original three-block matrix completed.

### Ordinary whole-frame latency

Each process has 180 excluded warmup frames and 600 measured frames, with
validation and optional scheduling diagnostics off. All 2400 formal frame
envelopes are available. There is **one independent ABBA block, two processes
per backend**; frames within a process are correlated.

| Process / backend | Frame mean / P50 / P95 / P99 (ms) | GPU envelope P95 / P99 (ms) |
| --- | --- | --- |
| 8 / legacy | 5.5871 / 5.5279 / 6.1239 / 6.4502 | 4.1883 / 4.4860 |
| 9 / shared | 5.8405 / 5.7130 / 6.6274 / 7.0597 | 4.6492 / 4.8495 |
| 10 / shared | 5.9968 / 5.8290 / 7.1159 / 8.0692 | 4.4963 / 4.8612 |
| 11 / legacy | 17.6344 / 6.5399 / 59.1116 / 71.1660 | 45.0520 / 61.9444 |

The two-process mean P50 is 6.0339 ms legacy versus 5.7710 ms shared
(legacy/shared 1.0456). The last legacy process has a substantial **CPU and
GPU** tail. It is retained without trimming or an outcome-dependent retry.
This small, dispersed population does not establish a stable FPS improvement.
The prebuilt-stream frame path does not execute the replaced meshlet builder
or experimental SIMD kernel, so these frame differences are not credited to
either change. [Full distributions and time series](FinalPlots/Report.html)
make the outlier and sample counts visible.

The captured coordinator path is dominated by the render/submit/Present
interval, including synchronization performed inside that renderer path.
The following are arithmetic means of disjoint intervals from the **same**
frames, not sums of inclusive scope medians:

| Process | Render/submit/Present (ms) | Profiler-end capture work (ms) | Frame-slot wait (ms) |
| --- | ---: | ---: | ---: |
| 8 | 5.2682 | 0.2567 | 0.0313 |
| 9 | 5.4952 | 0.2822 | 0.0315 |
| 10 | 5.5248 | 0.3987 | 0.0352 |
| 11 | 16.9897 | 0.5350 | 0.0533 |

[CPUFramePath.json](CPUFramePath.json) and
[its offline summarizer](SummarizeCPUFramePath.py) retain every segment mean.
The profiler capture work is explicit measurement overhead in both arms;
these are captured editor-loop timings, not an uninstrumented FPS promise.
Wall intervals can contain waits and do not measure CPU instruction occupancy.

Post-initialization scene-ready loading took 30.7445–32.9522 seconds; whole
process lifetimes were 40.4196–48.0689 seconds. These have different boundaries.
The [loading scope figure](FinalPlots/ProductionLoadingScopes.png) also retains
the separately logged Editor initialization. Shared application/OS/driver
caches remain warm; no flushed-cache claim or inferred decode/mipmap/upload
breakdown is made.

### Real source geometry cooking

All **48 fresh-output cooks and 48 separate validation processes passed**:
three ABBA/BAAB blocks, six processes per backend and worker count. Every
405-primitive, 3781-page, 110,435,932-byte stream has the same SHA256 as the
original provided stream:
`f5a1cf492d3f9ee538a160a41322e3fcfb173e9eae3dc3dcd9d4514d67a57c28`.
Every page decoded and the source attributes matched reference precision.
Source/OS caches are retained; "cold" means a new output without resume.

| Workers | Legacy / shared mean process seconds | Legacy / shared sampled thread peak | Shared wall change |
| --- | --- | --- | --- |
| 1 | 7.0698 / 7.4121 | 5 / 5 | +4.84% |
| 2 | 5.9011 / 5.7328 | 7 / 6 | -2.85% |
| 4 | 4.9983 / 5.1105 | 11 / 8 | +2.25% |
| 8 | 4.6608 / 4.7648 | 19 / 12 | +2.23% |

The extra worker pool is removed, with seven fewer sampled threads at eight
workers. There is **no general cooking speedup**: timing varies by worker
count and block, and four-worker regression is present in all three blocks.
The [cook figures](FinalPlots/ProductionCookProcesses.png) and
[block ratios](PlotReadout.md) retain that tradeoff. At eight workers, maximum
observed OS commit was 40.02/39.50 MiB and working set 41.23/40.18 MiB
(legacy/shared); four-worker commit increased 32.74 to 34.77 MiB. There is no
general memory-footprint reduction claim. This is CPU-only cooking, not VRAM
measurement; the separate validation subprocesses are excluded from its times.

### Resource peaks and longer steady execution

| Ordinary W1 process-lifetime metric | Legacy | Shared |
| --- | ---: | ---: |
| RHI tracked device-local event high-water | 7.1083 GiB | 7.1083 GiB |
| RHI tracked host+device event high-water | 7.1134 GiB | 7.1134 GiB |
| Observed RHI-reported device-local heap usage peak | 7.6898 GiB | 7.6916 GiB |
| Observed OS commit high-water | 8.1959 GiB | 8.2025 GiB |
| Observed OS working-set high-water | 0.6986 GiB | 0.6971 GiB |
| Sampled whole-process thread peak | 48 | 48 |

These quantities are separate and are not added. The renderer does not create
the legacy cooker pool on this prebuilt-stream path, so its thread count does
not inherit the cooker's 19-to-12 reduction. See the
[resource figure](FinalPlots/ProductionResourcePeaks.png) and
[RHI event highs](FinalPlots/ProductionRHIHighWaters.png).

Four additional diagnostic processes each ran **6000 frames after 180 warmup
frames**. Across all 24,000 captured frames, load/allocation failures,
pending geometry pages, pending texture bytes/images and sampled allocation
denials were zero. All nine tracked/residency metrics have equal observed
first/last values and constant last-600-frame ranges. Four final raw images
are finite with alpha one. [SoakAudit](SoakAudit/Report.md) includes the
reproducible checks, hashes and images; this establishes bounded fixed-view
resource stability, without claiming camera-route coverage or every-frame
pixel equivalence.

The normal diagnostic mode actually executes one `Prepare VBuffer` TaskGraph
per frame: 6000 graphs overlap each measured window. The 1683 stream-load graphs
all precede that window. CPU command recording remains Serial by pass policy;
the two independent preparation callbacks use the shared runtime and join
before owner-side publication. [Coordinator and DAG figures](SoakPlots/Report.html)
retain actual frame segments, task intervals and scheduling delays. The core
validation observer's forced-one-lane case is not used to infer this behavior.

## Metric ownership

Frame wall time follows consecutive coordinator loop starts, including existing
frame-slot/pass-overlap waits, rendering, submission, Present and telemetry.
The final interval ends after its final telemetry. The benchmark adds no
per-frame drain/readback; normal renderer synchronization contracts remain.
GPU query backfill and optional image readback happen after the capture.

Coordinator segments are non-overlapping wall intervals covering the same
frames. Nested CPU scopes, parallel worker totals, scheduling phase walls and
GPU envelopes are separate quantities, not additive layers of a critical path.
Diagnostic TaskGraph DAGs retain their actual time windows: startup/stream
graphs cannot be relabeled as measured-frame range work. CPU wall durations
also include waits and do not establish OS thread running time.

Windows process CPU, commit/working-set high-water counters and sampled thread
counts cover launch, loading, warmup, capture, export and teardown. They are
not measured-frame CPU utilization or GPU memory. OS high-water observations
extend through the last successful query; raw polling gaps/errors remain.

The new simultaneous RHI counters update under the existing allocation ledger
mutex and retain highs across tracked allocation/free events. They cover the
existing RHI tracker, excluding external SDK/driver allocations, VMA block
slack and temporary failed allocations never published to that tracker.
VMA allocation/block bytes and driver heap usage are separately sampled.
Per-domain highs may occur at different times and are never summed. NVML is
whole-device telemetry, with timestamp/age/interval; cached observations are
not independent per-frame sensor updates or renderer VRAM.

## Decisions and reproducibility

The optional SoA SIMD kernel retains the C++ default. Its numerical contract,
packing costs and workload-dependent ISPC results are described in
[SIMDAudit.md](SIMDAudit.md); no inactive kernel is credited with frame gains.
The M4 custom-deque gate is not met. Keep stdexec and the C++ automatic kernel
default. The shared meshlet adapter stays for its single worker owner and
measured thread-count reduction, with the cooking-latency tradeoffs above
reported as an architectural/resource change rather than a speedup claim.

[AcceptanceChecklist.md](AcceptanceChecklist.md) maps the milestones and limits;
[Plotting.md](Plotting.md) documents the Python figures. [Data/Evidence.json](Data/Evidence.json)
is the unchanged original record/archive hash manifest. The repository keeps
small metadata and per-run `CompactSummary.json` files; full Raw/Summary records,
frames, OS samples, TaskGraphs and raw images are retained byte-for-byte in the
[versioned evidence release](Storage.md). Directory attributes preserve exact
source-catalog and as-run snapshot bytes across Git checkouts. Original inner
run ZIPs use LZMA and are restored by the helper below. Generated cook payload binaries stay in the
original build output and are represented by validated hashes; the provided
source asset has the identical payload hash.

Reproduce plots offline, without running the renderer or tests:

```powershell
gh release download pr-74-cpu-jobs-evidence-20261107 --repo af8a2a/metallic --pattern metallic-cpu-jobs-production-20261107.zip --dir build-issue73-evidence-download
python -B Documentation/Measurements/CPUJobsProduction20261107/RestoreEvidence.py --archive build-issue73-evidence-download/metallic-cpu-jobs-production-20261107.zip --output build-issue73-evidence-restored --bundle FrameReduced --bundle ColdCook
python -B Documentation/Measurements/CPUJobsProduction20261107/FinalPlots/PlotProductionCPUJobs.py --production build-issue73-evidence-restored/Data/FrameReduced --production-cook build-issue73-evidence-restored/Data/ColdCook --output build-issue73-replot
```

[PackageEvidence.py](PackageEvidence.py) creates new archives without modifying
the original records in local/release staging. [Storage.md](Storage.md) records
retention, asset SHA256 and the original-history mapping. Migration changes
storage only; historical measurements are not relabeled as timings of later
code review fixes. The [historical plots](HistoricalPlotsFinal/Report.html)
separately visualize the archived M0/M2/generated-builder populations.
