# Production attribution notes

This is a read-only source and completed-record audit on 2026-11-07. No build,
GPU workload, timing protocol or runtime source was changed for this audit.
The final ordinary-performance and resource conclusions belong to the complete
production report; validation timing below is not a performance population.

## What actually executes in the fixed-camera graph

The completed core-validation captures use the requested
`E:/metallic/Asset/SuperSponza/NewSponza_Main_glTF_003.gltf` and its prebuilt
`.gltf.meshstream.bin`. The configured graph contains VBuffer, Shadows,
Deferred, DLSSSR, AutoExposure, DLSSNR, FinalBlit and ColorGrading. Actual
properties retain DLSSSR Quality, ray-traced shadows/SIGMA and the recorded
streaming/texture budgets. DLSSNR is configured with `enabled=false`; a node's
presence does not establish that its optional SDK feature executes.

The production pass command-recording path is serial on the CPU. All eight
pass implementations inherit `CPURecordingPolicy::Serial`. The explicit
`ParallelJoined` overrides found in runtime source are ClearColor, CopyColor
and the two buffer utility passes, which are absent from this scene graph.
This CPU policy is separate from GPU queue selection, GPU work within a pass,
and keeping multiple frames in flight.

| Owning source | Meaning |
| --- | --- |
| [`RenderGraphTypes.h:593`](../../../Source/Runtime/Render/RenderGraph/RenderGraphTypes.h#L593), `:599` | Pipelined-submission support defaults to false; CPU recording defaults to Serial. |
| [`RenderGraphExecutor.cpp:3528`](../../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L3528) | Per-pass opt-in and scene/subsystem dependencies also gate submission pipelining. A requested Pipelined mode alone is not proof that the scene uses it. |
| [`RenderGraphExecutor.cpp:4162`](../../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L4162) | Worker recording requires ParallelJoined, a worker limit above one, no prepared scene, no scene dependency, no required subsystem and no first-feature reservation. |
| [`VisibilityBufferPass.cpp:3678`](../../../Source/Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp#L3678) | Two independent CPU preparation callbacks compute camera parameters and a resource packet. They join before owner-side publication. This does not opt the pass's command recording into worker execution. |
| [`RenderGraph.cpp:613`](../../../Source/Runtime/Render/RenderGraph/RenderGraph.cpp#L613) | `prepareJoined()` uses the shared TaskSystem for multiple batches, otherwise runs inline. Its graphs are named `Prepare <passName>`. |
| [`RenderGraphExecutor.h:24`](../../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.h#L24) | Recording/preparation share the requested worker limit; preparation batch workload defaults to one. |

Changing the global worker count therefore does not automatically move this
scene's command recording onto workers. In ordinary mode, the two VBuffer
preparation jobs can use at most two simultaneous lanes; workers four and
eight do not create additional declared preparation work. Streaming load
concurrency is another independent bound. Any complete-frame result must be
interpreted with the actual executed work rather than the requested pool size.

## Validation deliberately has a narrower worker path

The validation runner enables debug control. The editor creates its debug
runtime at [`EditorApplication.cpp:2093`](../../../Source/Editor/EditorApplication.cpp#L2093)
and attaches it as the graph observer at `:2616`. Both
[`RenderGraphExecutor.cpp:4020`](../../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L4020)
and [`RenderGraph.cpp:620`](../../../Source/Runtime/Render/RenderGraph/RenderGraph.cpp#L620)
force the effective worker limit to one when an observer is attached.
Consequently the core-validation four-worker request validates a serial
observer-compatible rendering path. It does not validate four-worker recording
scalability. Ordinary performance runs disable that debug runtime; their
separate preparation behavior must be read from their own diagnostic records.

The final completed core-validation ABBA records are under
`build-issue73-production/validation-core-final-super-sponza/`. All four
Capture records report `passed`, each contains 120 measured frames, and the
corresponding Console logs contain no VUID/Validation Error lines. Each has
`validationRequested=true`, `validationEnabled=true`,
`validationMessengerActive=true`, `synchronizationValidationRequested=false`,
`synchronizationValidationEnabled=false` and `diagnostics=true`. These are
core-validation results, not synchronization validation results.

| Completed run | scenePrepareNs median (ms) | prepareNs median (ms) | serialExecuteNs median (ms) | workerExecuteNs / waitNs, every frame |
| --- | ---: | ---: | ---: | --- |
| `run-000-legacy-w4-measured` | 116.6641 | 0.3449 | 13.4862 | 0 / 0 |
| `run-001-shared-w4-measured` | 129.3518 | 0.3594 | 15.4185 | 0 / 0 |
| `run-002-shared-w4-measured` | 104.5165 | 0.3247 | 12.6136 | 0 / 0 |
| `run-003-legacy-w4-measured` | 91.4826 | 0.2989 | 12.2835 | 0 / 0 |

These phase medians describe the instrumented validation path. They are not a
backend speed comparison. `scenePrepareNs` surrounds `recordSceneBegin()` and
`recordSceneTraversal()` at
[`RenderGraphExecutor.cpp:2587`](../../../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L2587)
and `:2603`: it includes coordinator-side streaming/scene work and command
recording, plus any costs incurred in that scope. It is not worker occupancy
or proof that a range kernel dominates. `serialExecuteNs` surrounds pass
`execute()` at `:2610`. `workerExecuteNs` is measured around the separate
worker-recording path at `:2461`. Their sums and nested CPU scopes must not be
relabeled as a reconstructed dependency-graph critical path. Zero recording
join wait is not zero frame-slot/GPU wait or proof of idle/sleeping workers.

The earlier 600-frame core pilot in
`validation-core-super-sponza/run-000-legacy-w4-measured` also reports app
success with no VUID lines and zero worker-recording time. Its original runner
rejected the result because it incorrectly required synchronization validation
for a deliberately core-only configuration. Preserve it as an application
qualification record and rejected harness invocation; do not report a GPU
failure or include it in the corrected final population.

## TaskGraphs in these captures describe loading, not measured frames

Every final core-validation run has exactly 1,683 `TaskGraphs.jsonl` records.
All are named `MeshletStreamPageLoad`, and all finish before the first measured
frame's `originSteadyNs`. No recorded graph overlaps the measured window.

| Completed run | Task graphs overlapping measured frames | Last graph completion before first frame (s) |
| --- | ---: | ---: |
| `run-000-legacy-w4-measured` | 0 | 7.3738900 |
| `run-001-shared-w4-measured` | 0 | 7.4197586 |
| `run-002-shared-w4-measured` | 0 | 6.9599279 |
| `run-003-legacy-w4-measured` | 0 | 6.3287223 |

`pendingPages` is zero in every measured streaming snapshot of those runs.
The earlier 600-frame pilot likewise has 1,683 load graphs, none overlapping
its measured window; its last graph completes 22.1712305 seconds before the
first measured frame. These facts establish that the recorded graph work
occurred during launch/warmup. They do not establish absence of all CPU work.
The sink observes submitted TaskSystem graphs, not arbitrary application/SDK
threads or inline callbacks.

The owning submission is
[`MeshletStreamPageLoader.cpp:129`](../../../Source/Runtime/Render/Streamer/MeshletStreamPageLoader.cpp#L129).
Each graph has one Streaming node, with the page index as userTag; it calls
`loadPage()`. It is not the LOD builder or a frame command-recording batch.
Queue-delay endpoints can correlate this loading with other same-clock data;
they do not measure an OS worker wakeup transition.

## Why these frames do not exercise the new Meshlet range or SIMD kernel

[`SceneResourceManager.cpp:110`](../../../Source/Runtime/Render/Streamer/SceneResourceManager.cpp#L110)
selects metadata import when a compatible prebuilt glTF streamasset exists.
[`Scene.cpp:4008`](../../../Source/Runtime/Scene/Scene.cpp#L4008) routes that
request through `loadInternal(..., streamMetadata=true)`; the resident meshlet
cache/build block at `:4620` is skipped. Thus the Console summary
`meshletCacheLoaded=false` is expected here. The accurate workload description
is **prebuilt streamasset reuse**, not a resident meshlet cache hit.

[`MeshletStreamRuntime.cpp:1185`](../../../Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp#L1185)
opens the prebuilt asset and checks source compatibility. Failure is reported;
runtime auto-build is explicitly unsupported. The actual captured VBuffer
property is also `autoBuildStreamAsset=false`. The changed parallel adapter
is reached from LOD construction, for example
[`Scene.cpp:1229`](../../../Source/Runtime/Scene/Scene.cpp#L1229) and the offline
stream cook at
[`MeshletStreamAsset.cpp:4034`](../../../Source/Runtime/Scene/MeshletStreamAsset.cpp#L4034).
[`MeshletBuildParallel.cpp:76`](../../../Source/Runtime/Scene/MeshletBuildParallel.cpp#L76)
uses Dynamic partitioning with grain one on the shared TaskSystem. That
construction path is not repeated by fixed-camera frames using the prebuilt
streamasset.

Range diagnostics are opt-in. The production adapter supplies no diagnostics
object, so the absence of an operation record alone would not prove absence of
range execution; the source/lifecycle evidence above supplies that distinction.
The range identity example and generated-grid full-builder A/B remain separate
evidence with their own measured scopes.

`transformAABBs()` is still called only by the independent CPU-kernel library's
tests/benchmarks, not by production scene/visibility code. `Automatic` retains
the C++ auto-vectorized implementation. The optional ISPC target does not turn
these SuperSponza frames into an ISPC A/B. See [SIMD audit](SIMDAudit.md).

The legacy/shared editor comparison therefore checks complete-path correctness,
resource/lifecycle behavior and lack of an unintended regression in this
prebuilt fixed-camera workload. It cannot by itself establish an LOD-builder,
range-scheduler or SIMD speedup. A null frame result is meaningful evidence
for this scope, not a reason to attribute noise to inactive code.

## Regression and reporting boundaries

The completed production regression log reports Task 33/33, Scene 160 passed
with nine explicit asset/USD skips, and CPU kernels 6/6. The final selected
RHI core run in `.cache/issue73-production-rhi-final.log` reports 26/26 and
contains no VUID/Validation Error lines. The separate RHI synchronization
testbench evidence is under `build-issue73-production/rhi-sync-final/`; it must
retain its own actual feature/profile metadata. Neither those focused tests
nor the core scene run establish full-scene Streamline synchronization
validation. The earlier SDK/interposer startup failure remains a separate
retained limitation.

Final reporting must keep per-domain memory peaks separate from simultaneous
tracked allocation/device-local high-water counters. Those tracked peaks cover
existing RHI allocation events over the device lifetime, excluding external SDK
and driver allocations. NVML whole-device usage is sampled/cached sensor data;
its timestamps/age/interval are required to distinguish observations from
independent sensor updates. No exact whole-process VRAM peak is established by
adding these values together.

The worker/strategy matrix, generated builder workload and SIMD measurements
support their stated scopes. None supplies measured OS wakeup behavior, a
ThreadSanitizer pass on this Windows platform, or a reason to implement the
conditional M4 custom deque. Retaining the audited mature backend remains the
documented decision unless new owning-path evidence changes that gate.

The graph/window observation is reproducible without running the renderer.
From the repository root, the following read-only Python fragment prints the
population, graph names, overlap count and gap for each completed final run:

```python
import collections, json, pathlib
root = pathlib.Path("build-issue73-production/validation-core-final-super-sponza")
for run in sorted(root.glob("run-*")):
    frames = [json.loads(x) for x in (run / "Frames.jsonl").read_text().splitlines()]
    graphs = [json.loads(x) for x in (run / "TaskGraphs.jsonl").read_text().splitlines()]
    begin = frames[0]["originSteadyNs"]
    end = frames[-1]["originSteadyNs"] + int(frames[-1]["frameMs"] * 1e6)
    overlap = sum(g["completedNs"] >= begin and g["submittedNs"] <= end for g in graphs)
    gap = (begin - max(g["completedNs"] for g in graphs)) / 1e9
    print(run.name, len(frames), collections.Counter(g["name"] for g in graphs), overlap, gap)
```
