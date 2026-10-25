# CLAS template instantiation A/B

`MeasureCLASTessellation.py` compares the production `StreamCLASTessellation`
triangle-build and template-instantiation modes in the **same executable**. It
reuses `RHIRendering.material_phase0_baseline` to render the production graph and
`ExperimentRunner.block_gain` for v2 complete-ABBA log-ratio statistics. It has its
own `metallic-clas-tessellation-ab-v1` evidence protocol: the WorkRaster source-swap
runner does not support this experiment, and no source files are swapped.

```powershell
python -B Tools/Perf/MeasureCLASTessellation.py prepare --output build/clas-ab-fixture-new
python -B Tools/Perf/MeasureCLASTessellation.py run --exe build-pass-stages-nrd/tests/MetallicRHITests.exe --fixtures build/clas-ab-fixture-new --output build/clas-ab-qualification-new --diagnostics-only
python -B Tools/Perf/MeasureCLASTessellation.py run --exe build-pass-stages-nrd/tests/MetallicRHITests.exe --fixtures build/clas-ab-fixture-new --output build/clas-ab-new --blocks 3 --confirmation-blocks 5
python -B Tools/Perf/MeasureCLASTessellation.py verify --input build/clas-ab-new
python -B Tools/Perf/MeasureCLASTessellation.py report --input build/clas-ab-new --output build/clas-ab-report-new
```

Build the executable separately in its existing compatible Release configuration.
The runner neither builds nor changes SDK/compiler options. All fixture, evidence
and report directories must be new. No GPU collection runs concurrently. A shared
cooperative lock does not prove system-wide GPU exclusivity; whole-device telemetry
is retained and unrelated user processes are not stopped.

The generated case derives the authored displaced-plane sample's VBuffer,
RayTracedShadowPass and Deferred branch. It uses the original geometry, height,
material and camera, native 1444 x 694 resolution and the sample HDR environment.
It registers a fixed directional sun, manual exposure, zero angular light radius,
zero shadow RNG frame, disabled temporal jitter and disabled SIGMA. Upscalers,
auto exposure, color grading and presentation are outside this measured branch.
Both arms have identical settings; only `rtTessellationBuildMode` changes.
Explicit HDR, depth and shadow readbacks after timing verify that real ray consumers
render the same result. This is a fixed workload result, not editor FPS.

Each independent process renders 32 warmup frames, then 64 measured frames, then
captures correctness outputs. Source residency freezes at frame 16 while RT
selection, tessellation, CLAS, BLAS and TLAS continue to execute every frame.
`freezeStreamingFrame` cannot be substituted: it freezes the raster snapshot and
skips the work under test. Compilation, loading, template-pool initialization,
CPU waits and readbacks are excluded from measured GPU time. Disk caches remain
enabled and clocks are unchanged. Startup cost is not silently counted as steady
state; use a separate declared experiment to measure it.

Scopes are separately reported:

- `clasMs`: native CLAS triangle build or template instantiation and its AS barrier.
- `tessellationMs`: the enclosing record chain plus publication, without summing
  its nested CLAS section again. Final TLAS construction belongs to graph time.
- `graphMs`: the enclosing native-resolution production graph GPU interval.
- `shadowsMs`: the actual ray-traced shadow consumer pass.

Two separate diagnostic processes capture bound production shader recipes and
SPIR-V. Their timing data is never used for acceptance. Ordinary processes disable
validation, clear inherited `METALLIC_*` instrumentation and reject injected Vulkan
layers. Every process checks its fixed frame windows, queues, positive GPU scopes,
stable per-arm backend/shader identity, frozen same-process source identity and
finite HDR output. Source, interop headers, runtime DLLs, executable, CMake cache,
fixture and asset identities are retained. Across arms and stages all correctness
readbacks must be byte-identical. If diagnostic correctness differs, collection
stops before ordinary timings; do not loosen the tolerance after inspecting gains.
Both modes may legitimately bind identical shader SPIR-V; the mode/native operation
receipt establishes the intentional difference.
Each process also archives its actual cooked stream bytes. Runtime cache paths
contain a session identity and remain provenance rather than cross-process input
identity. For v9 runtime cooks with zero external dependency paths, the content
hash covers the entire file after clearing only the header's `sourceWriteTime`
(offset 32, 8 bytes) and `sourceDependencyFingerprint` (offset 40, 8 bytes). The
producer derives both from the temporary `.source` marker's metadata; neither
describes geometry. All geometry, instances, page payloads and other header bytes
remain exact. Raw hashes, original field values and the source marker are retained;
the marker must name an independently hashed authored asset. Other cooks use an
exact full-file hash. Unknown stream versions or external-dependency runtime cooks
fail this protocol instead of broadening normalization.
The final untimed frame copies the actual production `tessellationCounts` GPU
buffer. Nonzero patch and BLAS counts and zero rejected/invalid work are required;
the enabled setting alone is not proof of tessellation. The template arm must also
report an initialized pool and its allocation sizes. The report lists retained
template storage, template auxiliary storage, CLAS capacity and runtime scratch
separately, so excluded initialization time does not hide persistent memory cost.

Budgets are declared before collection: three discovery ABBA blocks and, only if
promising, five independent confirmation blocks by default. Both accept 2–10.
Each process has a 300-second timeout. One **complete four-process ABBA block** is
the statistical unit. The v2 helper computes a 95% Student-t interval across block
mean log(B/A) ratios and converts it to gain space. Frames and the two pairs inside
a block are not independent trials. No outliers are removed, no optional sample
extension is allowed, and discovery samples are not pooled into confirmation.

The primary target is at least 3% CLAS-stage gain, with at most 2% whole-graph
regression. Only confirmation can accept when both lower confidence bounds pass.
An upper bound below zero establishes regression; a positive upper bound below
3% is below target, not evidence of zero benefit. Discovery screens on its point
estimate after completing its budget. Inconclusive results retain their evidence.
Per-arm spread over 10% is a warning, not an early rejection of otherwise valid
paired evidence. The report exposes `evidenceValidity`, `effectAssessment` and
`nextAction` separately.

`verify` uses the current repository parser and statistics code, rehashes sealed
artifacts, rereads raw frames/images/recipes, checks independent process ordering,
and recomputes decisions. It never executes archived scripts. `report` writes
Python/matplotlib process-median and ABBA-interval charts, plotting data and version
metadata outside the sealed bundle. Open the charts and inspect axes/legends before
delivering performance claims. A diagnostic-only run has no ordinary performance
data and intentionally produces no timing chart.
