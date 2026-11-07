# Production CPU-job A/B runner

`MeasureProductionCPUJobs.py` runs the actual editor twice from the same source
and build configuration: the ordinary shared runtime and a benchmark-only link
using `LegacyMeshletSceneAdapter.cpp`. The adapter is the comparison variable;
the renderer, assets, graph, camera, settings and TaskSystem backend remain the
same. It does not infer production speedup from the synthetic job benchmark.

The editor exports `Capture.json` and measured `Frames.jsonl` through
`METALLIC_CPU_JOB_BENCH_CONFIG` / `METALLIC_CPU_JOB_BENCH_OUTPUT`. Configuration
controls the fixed frame route, scene, render graph, output extent, per-process
warmup and measured frame count. The runner only substitutes `workerCount`.

First run a bounded pilot in a fresh output directory. Reserve the machine/GPU
and wait for all compilation, downloads, readback captures and other benchmarks
to finish before collecting acceptance data. Choose the final block/frame
budget before inspecting its comparative outcome.

```powershell
python -B Tools/CPUJobs/MeasureProductionCPUJobs.py `
  --shared <shared-editor.exe> --legacy <legacy-adapter-editor.exe> `
  --config <fixed-super-sponza-config.json> --output <fresh-output-directory> `
  --workers 1,2,4,8 --blocks 3 --prewarm 1 --sample-ms 100 `
  --asset <scene-file> --asset <render-graph-file>
```

Each independent block starts four fresh processes in ABBA order; alternate
blocks use BAAB. Worker ordering rotates between blocks. Both candidates run
excluded prewarm processes before measured blocks. Existing shared shader,
pipeline, content, driver and OS caches are retained; there is no claim of a
flushed-cache or first-ever-load result. Warmup frames are excluded by the
application and must not appear in measured `Frames.jsonl`.

`--diagnostics on` enables existing RenderGraph scheduling diagnostics in both
arms. Keep diagnostic and ordinary runs in separate evidence directories;
the additional instrumentation can perturb timing. Performance captures must
explicitly report Vulkan validation disabled. Run correctness/validation and
image inspection separately. The runner checks effective scene/camera/graph,
extent, worker and recording limits and rejects changed workloads.

Use `--purpose validation --diagnostics on` with a validation/capture-image
configuration for correctness ABBA runs through the same OS evidence pipeline.
Those captures have a distinct purpose and produce no performance comparisons.
The runner synchronizes config diagnostics/validation flags with the child
environment, clears inherited capture/debug/trace/full-roam flags, and fixes
VSync off, Reflex off and frame limit zero. Effective application conditions
and the present mode parsed from the swapchain log are retained and checked.

The Windows monitor samples the child through `GetProcessTimes`,
`GetProcessMemoryInfo(PROCESS_MEMORY_COUNTERS_EX)` and Toolhelp process thread
counts. OS samples cover launch, load, warmup, rendering, export and teardown.
Resource summaries therefore cover a broader interval than the measured-frame
distributions. Current PrivateUsage and WorkingSetSize peaks are sampled;
PeakPagefileUsage and PeakWorkingSetSize are OS high-water counters observed
through the last successful query. Post-exit memory queries may be unavailable;
the last observed interval and errors remain explicit. Process CPU is kernel
plus user time, with average active cores derived from creation/exit FILETIMEs
when available. Thread count is a sampled peak, not an exact lifetime peak.

GPU memory comes exclusively from application `DeviceMemoryBudget` snapshots.
Device-local allocation bytes, VMA block bytes and heap usage are separate
quantities, not additive layers. Per-domain allocation peaks may occur at
different times and must not be summed into a simultaneous total. Frame and
after-load/warmup/capture snapshots establish observed heap peaks, not unseen
transient lifetime VRAM peaks. Missing telemetry stays `null` with a reason;
process private memory is never substituted for VRAM.

The global `peakTrackedAllocationBytes` / `peakTrackedDeviceLocalBytes` counters
are simultaneous high-water totals from already published/tracked RHI allocation
events. They exclude temporary native allocations failing before tracking,
untracked external SDK allocations and driver memory. They remain distinct from
sampled VMA blocks, driver usage and independently timed domain peaks.

Each run directory retains its invocation/config, console, raw OS samples,
application capture/frames and process summary. Top-level `Raw.jsonl` has one
process summary per invocation; `Summary.json` keeps complete processes plus
paired independent-block comparisons. Within-process frame P50/P95/P99 use
nearest rank. Frames within a process are correlated; statistical inference
must use independent process/block units. Nested CPU/GPU scopes and parallel
worker totals remain independent and are not summed into a critical path.

No GPU work is needed to validate the evidence contracts:

```powershell
python -B Tools/CPUJobs/TestProductionCPUJobs.py
```

For CPU-only source geometry, `MeasureProductionMeshlets.py` creates a unique
fresh stream output for every process. It measures the cook separately from a
subsequent `--inspect --validate-payloads` process; optional
`--validate-attributes` also checks the source attribute contract. Whole-file
stream SHA256 and logical manifest fields must agree across all candidates,
workers and repetitions. It retains the first validated reference stream and
all reports/events/logs/OS samples; later matching generated streams are removed
after cross-candidate comparison to bound temporary storage. `--keep-payloads`
retains every binary. Source and OS caches are retained; “cold” means fresh
output, without a resume or existing cooked-output hit.

```powershell
python -B Tools/CPUJobs/MeasureProductionMeshlets.py `
  --shared <MetallicMeshletCook.exe> --legacy <MetallicMeshletCookLegacy.exe> `
  --source <production.gltf> --output <fresh-cook-evidence-directory> `
  --workers 1,2,4,8 --blocks 3 --sample-ms 100 --validate-attributes
```
