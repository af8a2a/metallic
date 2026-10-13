# Compute compilation and execution experiments

`RunComputePipelineExperiments.py` measures native creation separately from real
GPU work. Its `compile` command invokes `MetallicShaderPipelineBench`; `dispatch`
invokes the production RenderGraph harness in `MetallicRHITests`. Each command
records its schedule, inputs, process logs and results. It never edits shaders.
Run from the repository root using a current build and Python 3.11 or later;
dispatch readback comparisons also require numpy.

For exact SPIR-V CFG/callgraph analysis, opt-in Driver executable statistics,
ISA availability checks and the measured shadow-only noinline feasibility
experiment, see [Shader compilation complexity](ShaderCompilationComplexity.md).
For the optional two-pass SPIR-V cleanup, including Slang's integrated
downstream path, its cache/metrics contract and current rendering qualification,
see [Dead-branch cleanup](ShaderDeadBranchCleanup.md).

## Native compilation

First list the current catalog. Indices can change; retain the selected request
metadata rather than relying on a permanent index.

```powershell
$bench = '.\build-release\Source\MetallicShaderPipelineBench.exe'
& $bench --list --filter OpenPBRRayQueryPathTraceGuides
# Current catalog: 146 = Guides with SHARC off; 149 = Update; 152 = Query.
python -B Tools/Perf/RunComputePipelineExperiments.py compile `
  --executable $bench --request-index 146 `
  --optimizations O1 O0 O2 --backends pipeline `
  --driver-optimization both --pairs 3 `
  --output-dir build-release/bench-compute-optimization
```

Each Slang optimization level and postprocess mode first fills a private shader cache. Measured
processes must load those exact SPIR-V bytes; Slang compilation is excluded from
native creation. Different levels may produce different bytecode. The native
disable-optimization pair must retain the same input and final device SPIR-V,
descriptor ABI, debug/validation settings and physical device/driver. The report
retains both SPIR-V statistics and bytecode identities, including the device
rewrites. Statistics describe SPIR-V structure, not physical registers or final
driver machine code.

Compare `pipeline_load_and_create_wall_ms` for the complete pipeline loading
scope: application cache file/metadata/backend-cache preparation plus the RHI
creation call. `pipeline_cache_preparation_wall_ms` records that preparation
once, where it happened. Device creation, SPIR-V loading/module preparation and
cache saving are separate. `pipeline_wall_ms` alone excludes application-cache
initialization, but includes native binary loading performed inside the RHI.
`native_create_ms` measures the actual `vkCreateComputePipelines` or
`vkCreateShadersEXT` call. `native_binary_import_ms` measures preparation before
that call: shader-object file/CPU loading, or pipeline-binary file loading and
`vkCreatePipelineBinariesKHR` handle restoration. Export preparation has its own
`native_binary_export_ms`. A smaller final native call alone does not describe
the total binary-loading benefit.

All measured source/import processes request
`METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled`. Source samples use an empty
application cache; ordinary `.pso` imports use their primed application cache.
Experimental native binary backends receive a null `VkPipelineCache`. This does
not establish that every driver/compiler/OS state is cold. Report
`native_create_ms` separately from feedback `driver_ms`; the latter is null when
Vulkan creation feedback is unavailable. An application cache-hit flag does not
identify an internal driver-cache hit.

Compute shader objects and pipeline binaries have separate native source/import
arms. Extraction is excluded from measured import samples, recorded with its
time and bytes. Imports must report `native_binary_cache_hit=true`, accept the
binary without driver rejection, and preserve the extracted files' SHA-256
identities. A fallback to source is an experiment failure, with logs preserved.

```powershell
python -B Tools/Perf/RunComputePipelineExperiments.py compile `
  --executable $bench --request-index 146 --optimizations O1 `
  --backends shader-object pipeline-binary --driver-optimization normal `
  --native-binary-cache --pairs 3 `
  --output-dir build-release/bench-compute-native-binary
```

Shader objects expose no pipeline disable-optimization flag, so that combination
is omitted from mixed experiments and rejected when it is the only requested
condition. Native binary cache options require a binary-capable backend. Device
or extension failures remain failures; the runner does not substitute another
backend. Native binary files remain separate from ordinary `.pso` caches.

For a fair three-path source/import comparison, add the ordinary application
cache arm. Its prime is excluded; each measured reload must report a loaded
cache, a metadata hit and a valid native application-cache hit. Artifact hashes
must remain unchanged. Native object reuse and source fallback fail the run.

```powershell
python -B Tools/Perf/RunComputePipelineExperiments.py compile `
  --executable $bench --request-index 146 --optimizations O1 `
  --backends pipeline shader-object pipeline-binary --driver-optimization normal `
  --application-pipeline-cache --native-binary-cache --pairs 3 `
  --output-dir build-release/bench-compute-load-paths
```

The Benchmark's `--pipeline-cache-file` allows a separate `.pso` per condition
while retaining one fixed SPIR-V cache. Reports include per-call
`compilation_phases`: compiler setup, module load, entry-point composition,
component link, code generation, target finalization and dependency/cache work.
The code-generation API includes IR optimization and downstream SPIR-V tools;
component `link_ms` does not isolate the compiler's IR linking/optimization pass.
The Slang SDK's `compiler_reported_downstream_ms` overlaps code generation and
must not be added to wall phases. `compile_wall_ms` also includes local compiler
teardown and registry bookkeeping after the returned metrics' total timer.
Cache hits leave compiler phase counters at zero.

`--slang-perf-diagnostics` requests SDK detailed reporting during cache misses;
the component API does not guarantee a detailed pass report. Preserve actual
diagnostics rather than assuming the option produced one. A separate Slang CLI
profile is eligible only after its output matches the Benchmark SPIR-V byte for
byte. Diagnostic compiler/pipeline statistics runs are separate from ordinary
creation and GPU timing.

## Production GPU execution and correctness

The existing `PreparePathTraceQueries.py` generator provides deterministic opaque
and mixed material scenes with real geometry, textures, camera and HDR lighting.
The `OpaquePerf` workload uses the production OpenPBR PathTrace pass with guides,
two samples per frame, depth 12 and SHARC off. Keep those settings fixed across
arms. A small resolution is useful for a first correctness/execution check:

```powershell
python -B Tools/Perf/PreparePathTraceQueries.py build/compute-create-fixtures
python -B Tools/Perf/RunComputePipelineExperiments.py dispatch `
  --executable cmake-build-debug/tests/MetallicRHITests.exe `
  --cases build/compute-create-fixtures/Performance.json --case-id OpaquePerf `
  --resolution 64 --warmup-frames 2 --timing-frames 4 --capture-graph-outputs `
  --optimizations O1 O0 O2 --backends pipeline --driver-optimization both `
  --pairs 3 --output-dir build/compute-dispatch-exploration
```

The script changes only the selected case set, requested resolution and timing
window in an archived copy of `Cases.json`. Temporal/camera correctness fixtures
cannot have their frame window resized. The source graphs and assets are
preserved and hashed. Final HDR and optional graph outputs are read back after
the timing window. Every result is compared against the first measured arm using
predeclared absolute/relative tolerances (both 2e-5 by default). Nonfinite or
truncated readbacks, workload changes, missing timing frames and Vulkan
validation errors fail qualification. Failed numerical comparisons retain all
measured timings and differences; they cannot justify a correctness-preserving
optimization.

Runtime logs must confirm the effective Slang level and actual native creation
policy. Recipe capture archives the actual PSO states, input SPIR-V, device,
driver, enabled features and validation settings from each process. Within one
Slang level, those input bytes must remain identical across backend/driver-flag
arms. Ordinary dispatch timing uses the controllable driver cache enabled and a
private renderer `.pso` directory: all compilation and cache loading happens
outside the GPU measurement scope.

Reported `dispatch_scope_ms` comes from the declared production GPU scope (for
this case, `Path trace shading`). `pass_ms` and `graph_ms` are distinct timestamp
scopes. They exclude presentation, denoisers and upscaling. They are not
end-to-end frame rate, and a scope containing several dispatches is not one
dispatch's timing. The statistical unit is an independent-process median;
frames inside a process are not independent process repeats. Conditions rotate
and reverse between blocks. A short 64 x 64 run or one process per arm is
exploratory; use a current Release test build, the original 512 x 512 workload,
32 warmup/64 timed frames and sufficient process repeats before adopting an
optimization. Run validation as a separate correctness experiment, using
`--validation`, so its cost does not enter ordinary timings.

The runtime controls are `METALLIC_SLANG_OPTIMIZATION`, `METALLIC_SPIRV_POSTPROCESS`,
`METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION`, `METALLIC_VK_COMPUTE_BACKEND`,
`METALLIC_VK_NATIVE_BINARY_CACHE_DIR` and `METALLIC_VK_PIPELINE_BINARY`.
The runner sets them before process launch and retains them in evidence. It
clears inherited `METALLIC_*` controls and rejects injected Vulkan layers for
ordinary measurements. The production default remains Slang O1 with normal
pipeline optimization. Creation timing alone does not justify changing it.

## Opt-in SPIR-V cleanup comparison

Use the same current compute request, O1, mapped descriptors and normal Driver
optimization to compare `none`, `dead-branches` and `slang-dead-branches`.
Replace `N` with the current Guides request selected from `--list`; use a fresh
output directory. Both cleanup modes are opt-in and require the optional
installed SPIRV-Tools dependency. Requested cleanup fails before cache lookup
when it is unavailable or the request uses native descriptor heaps.

`dead-branches` runs an additional in-process optimizer after target finalization,
with input/output validation. `slang-dead-branches` appends the same two pass
flags to Slang's existing optimizer invocation through `DownstreamArgs`, then
performs target finalization and final Vulkan 1.4 scalar-layout validation.
The selected O0/O1/O2/O3 level is preserved; no external optimizer process or
second local optimizer is launched. The new runtime path is qualified against
the pinned Slang 2026.18.2 SDK, not arbitrary alternate `SLANG_ROOT` installs.

### Fresh shader-cache frontend costs

`MeasureShaderPostprocess.py` isolates source compilation from device/Pipeline
creation. It fixes requested/effective O1, mapped descriptors and disabled
shader debug/validation. Every independent process compiles one request using
`--shader-cache reuse` into a new private cache directory. A hit or skipped
compiler execution fails the experiment. This measures a fresh cache miss,
including cache writing; it does not disable caching or establish fully cold
OS/compiler state.

```powershell
python -B Tools/Perf/MeasureShaderPostprocess.py run `
  --executable .\build-release\Source\MetallicShaderPipelineBench.exe `
  --request-indices N --postprocesses none dead-branches slang-dead-branches `
  --pairs 3 --output-dir build-release/compute-cleanup-frontend-<fresh>
python -B Tools/Perf/MeasureShaderPostprocess.py verify `
  build-release/compute-cleanup-frontend-<fresh>
```

The frontend runner rotates three-arm blocks so every mode occupies each
position across three blocks; two arms alternate AB/BA. It archives the actual
schedule, process commands/logs, compiler phases, settings and exact `Output.spv`
bytes extracted from validated cache containers. Independent-process medians
are computed per request/mode; repeated output byte differences fail verification.

Schema v2 performs one excluded `none` compilation per request to discover
the compiler's actual source dependencies. It freezes their current hashes
before that request's measured samples, including vendor/import and interop
headers. It also freezes renderer/shader sources, the executable/DLLs and
imported tool helpers. Before/after checks reject input changes, and a newly
reported unfrozen dependency fails the run. Verification independently checks
discovery coverage, fresh misses, settings, schedule, raw metrics and cache/output
byte identity. Discovery timings are retained in `excluded_runs` and excluded
from aggregates. Existing sealed schema-v1 evidence remains readable and retains
the copied runner that actually collected it.

The [current frontend results](ShaderDeadBranchCleanup.md#slang-downstream-api-measurements)
contain three processes per mode for Guides request 146 and one exploratory
process per mode for SceneRTXDI/SHARC requests 137/149/152. Guides, SceneRTXDI and
SHARC Query have matching external/Slang cleanup SHA-256 values; SHARC Update
149 has equal output sizes but different SHA-256 even after compacting IDs,
requiring affected SHARC GPU qualification. These compile-only results establish
no whole-call cold startup benefit. Native timings below deliberately exclude
frontend work.

The separate [same-process cold comparison](ShaderDeadBranchCleanup.md#fresh-source-plus-native-creation-in-one-process)
includes source compilation and native creation for current Guides, with three
pairs and disabled controllable Driver caching. Its observed 7.72% paired-median
reduction applies to that single Bench request. The subsequent
[SHARC rendering batch](ShaderDeadBranchCleanup.md#sharc-rendering-qualification-for-both-cleanup-paths)
qualifies HDR/guides in the fixed MixedSharc scene, including Update request 149;
it enables validation and ordinary Driver caching and makes no performance claim.

### Native and production GPU comparisons

```powershell
python -B Tools/Perf/RunComputePipelineExperiments.py compile `
  --executable .\build-release\Source\MetallicShaderPipelineBench.exe `
  --request-index N --optimizations O1 --postprocesses none dead-branches slang-dead-branches `
  --descriptor-mode mapped --backends pipeline --driver-optimization normal `
  --pairs 3 --output-dir build-release/compute-cleanup-native-<fresh>
```

The runner sets `METALLIC_SPIRV_POSTPROCESS` explicitly for every arm and supplies
`--spirv-postprocess` to the Benchmark. Its default is `--postprocesses none`.
Original, separate-cleanup and Slang-cleanup bytecode use distinct shader/native
identity groups and cache fills, even when their bytes agree. `none` and the
existing separate-cleanup cache key remain unchanged; the new mode adds its own
revisioned policy, runtime Slang build tag and mode-specific tool version.
Backend/Driver flag comparisons retain identical code within each
Slang-level/postprocess group. Two cleanup conditions alternate AB, BA, AB
between blocks. Prefill compilation and cleanup are excluded from measured
native creation, with their per-call metrics retained separately.

For a separate production execution comparison, retain the real graph/settings
and capture its declared outputs:

```powershell
python -B Tools/Perf/RunComputePipelineExperiments.py dispatch `
  --executable .\build-release\tests\MetallicRHITests.exe `
  --cases build/compute-create-fixtures/Performance.json --case-id OpaquePerf `
  --optimizations O1 --postprocesses none dead-branches `
  --backends pipeline --driver-optimization normal `
  --resolution 512 --warmup-frames 32 --timing-frames 64 --capture-graph-outputs `
  --pairs 3 --output-dir build-release/compute-cleanup-dispatch-<fresh>
```

Use `--validation` in a distinct correctness batch. Runtime logs and captured
recipe stages must confirm the requested cleanup policy and corresponding tool
version. Missing or conflicting confirmation fails verification. A transformation that
passes readback tolerance is not necessarily byte-identical in HDR.
Recipe replay consumes finalized captured bytes and forbids an explicit new
`--spirv-postprocess` override. Both cleanup policies require a confirmed tool
version, and ambiguous source identity is rejected for performance comparison.

For `dead-branches`, `compilation_phases.postprocess_ms` includes input/output
validation and the two cleanup passes after target finalization. For
`slang-dead-branches`, it measures final validation only; its appended passes
are inside `code_generation_ms`. `raw_spirv_bytes` describes the returned Slang
blob before Metallic target finalization, so the new mode already includes its
transforms in that count. Zero on a cache hit means unavailable.
`output_spirv_bytes` describes the returned code, including on cache hits, which
have zero compilation/cleanup work. These are source SPIR-V sizes rather than
Driver ISA. `slangShaderPostprocessToolVersion(mode)` retains the old no-argument
behavior; the new mode identifies Slang's version plus the linked final validator
version, without claiming that validator version describes Slang's optimizer.
When assessing total uncached compilation, measure source compilation and
pipeline loading/creation in the same declared call scope. Adding independent
frontend/native medians is not an end-to-end startup measurement.
The [cleanup report](ShaderDeadBranchCleanup.md) defines the exact measured
scopes, identities and three-scene correctness limits. Production remains O1
with `METALLIC_SPIRV_POSTPROCESS=none` (also the unset default).

## Evidence and validation

`summary.json` checkpoints progress, failures, cache conditions and measured
scope. Successful results are sealed by `Manifest.json`; verify them using:

```powershell
python -B Tools/Perf/RunComputePipelineExperiments.py verify <experiment-directory>
python -B tests/perf/TestComputePipelineExperiments.py
```

`AnalyzeComputePipelineExperiments.py` verifies the sealed inputs before writing
per-process CSV, JSON and SVG distributions. It reports median/min/max and paired
block ratios, never treating individual GPU frames as independent repeats or
pooling different experiment directories. Nonoverlapping loading components use
means; driver duration remains inside native creation. Optional
`--statistics-log LABEL=PATH` retains driver-defined executable statistics in a
separate diagnostic section; local memory is not assumed to mean spills.

```powershell
python -B Tools/Perf/AnalyzeComputePipelineExperiments.py `
  build-release/bench-compute-load-paths build-release/bench-compute-optimization `
  --output-dir build-release/bench-compute-analysis
```

The verifier checks artifact hashes and recomputes native/GPU timing and
correctness evidence. Keep generated evidence outside source control. Existing
GPU experiment locks cause a failure: inspect the recorded owner and preserve
its work before starting another experiment. The lock prevents cooperating
Metallic runners from overlapping; it cannot establish global GPU exclusivity.
GPU clocks are unaltered and no other applications are closed by this runner.

## Repeated measurements, 2026-10-12

The current experiment uses Intel Core Ultra 5 245KF / NVIDIA RTX 5060 / driver
616.92, Windows build 26100, Slang 2026.18.2 and the reused Release build. The
power plan was already High Performance; clocks and other applications were not
changed. Ordinary timings disabled debug information and validation. Each native
condition used three independent processes, fixed request 146, mapped descriptors,
identical SPIR-V within each optimization level, an empty application cache and
disabled controllable internal cache. Input/final device SPIR-V, full PSO hash and
device/driver identities were retained. The ordinary O1 PSO hash was
`5aa0734d26f3cc25`, also captured from the real RenderPass.

| Slang | SPIR-V bytes | Instructions / functions / blocks | Frontend wall, n=1 (s) | Downstream compiler, contained in frontend (s) | Native median, n=3 (s) | Native min–max (s) |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| O0 | 709,664 | 42,673 / 574 / 4,594 | 1.726 | 0.000 | 56.160 | 53.029–64.302 |
| O1 | 6,940,612 | 447,484 / 1 / 58,220 | 19.816 | 17.999 | 63.109 | 60.501–79.019 |
| O2 | 1,992,720 | 115,049 / 1 / 17,277 | 86.387 | 84.625 | 50.303 | 49.254–51.833 |

These frontend values are one-time excluded cache fills, not three-process
distributions. Module loading took 0.718/0.783/0.811 s respectively. Code
generation took 0.833/18.775/85.353 s; downstream time is contained in that API
interval. O1's downstream compiler consumed about 96% of code generation, while
O2's downstream work consumed about 99%. The O1 module's largest function has
446,028 instructions. These are static IR counts, not executed instructions.
O2's smaller module reduces measured native creation but its first frontend
compile costs substantially more. Adding separate phase measurements estimates
component costs; it is not a measured end-to-end cold launch.

A separate pinned Slang CLI diagnostic produced **byte-for-byte identical O1
SPIR-V** (`ef751a428c7a9c48`, SHA-256
`414c4001b496f6e44cdd2e86faa57339d891acc4473e77f34da0b9d196f63b38`). Its
`compileInner` was 18.762 s, downstream compile 17.40 s,
`linkAndOptimizeIR` 0.532 s and semantic checking 0.671 s. Nested SDK counters
overlap and are not summed. This locates the high frontend cost in downstream
SPIR-V optimization rather than module parsing or Slang IR linking alone. It
does not identify an individual proprietary NVIDIA compiler pass. [Slang's
SPIR-V documentation](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/a2-01-spirv-target-specific.html)
describes O0 preserving the call graph; the observed O1 growth is consistent with
extensive inlining, not proof that source-level Slang inlining is the bottleneck.

```powershell
python -B Tools/Perf/ProfileComputeShaderFrontend.py `
  --input-root build-release/compute-cold-optimization-n3 `
  --optimizations O1 --pairs 1 --output-dir build-release/compute-frontend-cli-O1
python -B Tools/Perf/ProfileComputeShaderFrontend.py `
  --verify build-release/compute-frontend-cli-O1
```

The native optimization group reported valid feedback/application misses in all
nine processes. `driver_ms` closely matched native wall; lock waits were
0.0004–0.0011 ms. This establishes dominance of driver creation for this serial
workload, not absence of contention in parallel workloads. Native durations
varied materially, so retain the ranges and avoid conclusions from one sample.

### Source and complete cached loading

The second matrix held O1 bytecode and all other controls fixed. Each backend
had three source and three independent-process import samples. Priming/export
was excluded. Every import reported the actual intended cache hit without object
reuse or source fallback; file SHA-256 identities remained unchanged. The wall
scope includes file/cache preparation exactly once and excludes common device,
SPIR-V/module initialization and saving.

| Path | Source complete creation median (s), n=3 | Cached complete load median (ms), n=3 | Cached min–max (ms) | Native import call median (ms) |
| --- | ---: | ---: | --- | ---: |
| Pipeline / existing `.pso` | 66.765 | 31.377 | 31.270–34.182 | 18.898 |
| Shader Object / shader binary | 60.746 | 187.434 | 177.111–209.909 | 174.155 |
| Pipeline / `VK_KHR_pipeline_binary` | 74.530 | 36.634 | 34.714–37.041 | 20.644 |

`.pso` preparation median was 12.211 ms. Pipeline Binary preparation median was
14.605 ms; Shader Object file preparation was 10.777 ms, with aligned staging
and other RHI work in the remaining wall interval. Shader Object has no pipeline
creation feedback, so its `driver_ms` is unavailable rather than zero. Excluded
native extraction took 12.536 ms / 4,430,994 bytes (Shader Object) and
13.033 ms / 5,385,894 bytes (Pipeline Binary); `.pso` saving remained outside
loading samples.

All three caches avoid the expensive source creation. Existing `.pso` loading
was fastest in this matrix: Pipeline Binary was about 17% slower by median full
wall and Shader Object Binary about 6 times its duration. Source durations have
overlapping, wide ranges; changing to Shader Object still leaves roughly a minute
of source creation and does not remove the cold bottleneck. There is no measured
reason to replace the ordinary `.pso` path for this device/workload, and no result
here establishes behavior across driver updates.

### Real scene execution and startup attribution

`OpaquePerf` used fixed geometry/textures/HDR lighting, 512 x 512, two samples
per frame, depth 12, SHARC off, 32 warmup and 64 timed frames. Compilation and
cache loading precede GPU timing; captures follow it. Each row is the distribution
of three independent-process shading-scope medians.

| O1 backend | Shading median (ms) | Process min–max (ms) | Paired candidate/Pipeline median (range) |
| --- | ---: | --- | --- |
| Pipeline | 23.860 | 23.653–24.052 | 1.000 |
| Shader Object | 22.831 | 22.329–23.058 | 0.957 (0.928–0.975) |
| Pipeline Binary | 23.503 | 23.039–24.179 | 0.985 (0.958–1.022) |

All 72 HDR/guide readbacks were numerically identical. Logs and exact captured
recipes prove real Guides binary imports in blocks 1/2, following source export
in block 0. Runtime native import calls were 449–499 ms for Shader Object,
277–368 ms for Pipeline Binary (plus 16–19 ms preparation), and 282–324 ms for
`.pso`. These runtime conditions use the internal cache enabled and a populated
scene; they are distinct from the isolated cache-disabled loading matrix above.
Shader Object was faster in these three GPU pairs, but the small repeat count,
process-order/clock effects and one workload limit adoption conclusions.
Pipeline Binary's paired range crosses 1; a stable GPU benefit was not shown.

In each backend's first real-scene process, Guides accounted for **94.42–94.61%**
of the sum of native creation for 12 Compute PSOs. The first Pipeline process
spent 79.353 s on Guides out of 83.878 s on Compute native creation. This identifies
the workload to prioritize; the fraction excludes frontend work, graphics PSOs,
uploads and the rest of scene startup.

The separate optimization GPU group retained its original failure status:

| Slang | Shading median (ms), n=3 | Paired/O1 median | Readback qualification |
| --- | ---: | ---: | --- |
| O1 | 24.423 | 1.000 | Pass |
| O0 | 27.595 | 1.130 | Fail in all three processes |
| O2 | 24.305 | 0.998 | Pass; paired range crosses 1 |

O0 consistently differed at only two pixels / six RGB channels, with maximum HDR
absolute error 0.000480763 above the predeclared tolerance. Alpha, the other
262,142 pixels and all seven guides were exact. This is a deterministic variant
difference, not evidence of memory corruption or a specific shader root cause.
Its timing is diagnostic and does not qualify as an output-preserving optimization.
O2's maximum HDR error was 1.073e-6; all seven guides were exact. The tolerance
was not relaxed. Keep production O1 and normal driver optimization: O0 sacrifices
this correctness gate and GPU time, while O2 adds frontend cost without a stable
GPU gain. Driver source creation and the downstream SPIR-V optimization path are
the measured priorities for further structural shader experiments.

Current evidence is outside source control:

- `build-release/compute-cold-optimization-n3/`: sealed nine-process source matrix and frontend phases.
- `build-release/compute-cold-paths-n3/`: sealed eighteen-process source/import matrix, with primes excluded.
- `build-release/compute-frontend-cli-O1/`: sealed identical-bytecode diagnostic and raw nested SDK timings.
- `build-release/compute-gpu-backends-512-v2/`: sealed nine-process GPU matrix.
- `build-release/compute-gpu-backends-512-proof.json`: raw report/log/recipe/SPIR-V/readback hashes and actual import attribution.
- `build-release/compute-gpu-optimization-512/`: archived **failed** qualification group; its status remains failed.
- `build-release/compute-gpu-optimization-512-proof.json` and `compute-gpu-optimization-512-analysis/`: independently recomputed diagnostic timings, pixel differences and hashes.
- `build-release/compute-final-analysis/`: verified 36 process measurements, CSV/JSON, median/min–max and loading-component SVGs.
- `build-release/compute-final-preview/ComputeBackends512.png`: visually inspected common HDR previews; two-sample noise remains present.
- `build-release/ComputeExperimentHost.json`: CPU/GPU/OS/power-plan inventory; no system settings changed.

Release builds and incremental verification passed. Four focused shader metrics/
structural tests, two Vulkan-validation native backend tests and four CTest tool
suites (48 Python cases) passed. The Benchmark regression suite passed six checks
and skipped four opt-in GPU cases; the real-scene and independent-process groups
above provide the actual GPU/creation evidence. No native crash occurred in these
measured groups. Validation was kept out of performance runs, and neither this
fixed workload nor one-frame previews establish full-scene temporal/memory behavior.

## Historical single-process exploration

The single-process exploration below is historical. The subsequent Release
512 x 512 / three-process experiments supersede its adoption conclusions:
`build-release/compute-cold-optimization-n3`,
`build-release/compute-gpu-backends-512-v2`, and
`build-release/compute-gpu-optimization-512`. The optimization GPU group retains
`status=failed` because O0 exceeded its predeclared HDR tolerance; its diagnostic
results must not be reported as a qualified optimization.

The existing Release `build-release` Benchmark ran fixed catalog request 146,
`OpenPBRRayQueryPathTraceGuides`, SHARC off, mapped descriptors, disabled debug
information and validation. Hardware was NVIDIA RTX 5060 / driver 616.92.
Each row used prefilled SPIR-V and one independent process per creation arm,
an empty application cache and disabled controllable internal cache. All native
feedback was valid, application-cache misses, and lock waits were 0.0007–0.0011 ms.
Values are exploratory single samples, without a confidence interval:

| Slang | SPIR-V bytes | Instructions | Functions | Basic blocks | Frontend/cache wall (s) | Native normal (s) | Native disable optimization (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O0 | 709,664 | 42,673 | 574 | 4,594 | 1.790 | 58.595 | 58.465 |
| O1 | 6,940,612 | 447,484 | 1 | 58,220 | 21.989 | 66.471 | 69.624 |
| O2 | 1,992,720 | 115,049 | 1 | 17,277 | 74.888 | 52.852 | 52.460 |

Frontend wall includes compilation and shader-cache population. Input and final
device SPIR-V sizes/structural counts agreed, with their separate
hashes retained. The O1 module's single function contains 446,028 instructions;
this points toward investigating frontend inlining/control-flow growth. The
much smaller O0 input still takes nearly a minute in the driver. Disabling
optimization provided no clear creation benefit in this run. O2's reduced native
time comes with substantially more frontend work, so native time alone cannot
justify adopting O2. No production default or shader code structure was changed.

An O1 matrix used the same bytecode, compiler/debug settings, device and cache
control with separate source, extraction and independent-process import arms:

| Path | Native call (ms) | Creation path wall (ms) | Imported bytes |
| --- | ---: | ---: | ---: |
| Shader Object source | 54,414.774 | 54,415.486 | — |
| Shader Object binary import | 189.481 | 204.013 | 4,430,994 |
| Pipeline Binary source | 64,196.376 | 64,196.830 | — |
| Pipeline Binary import | 18.521 | 34.719 | 5,385,894 |
| Existing `.pso` reload, process 1 | 18.110 | 18.197 | — |
| Existing `.pso` reload, process 2 | 17.053 | 17.172 | — |

Native calls time `vkCreateShadersEXT` or `vkCreateComputePipelines`. Binary
creation-path wall includes file loading/handle restoration; `.pso` cache
initialization/loading precedes its pipeline wall timer. Thus these wall columns
do not establish which complete load path is faster. Both `.pso` reloads reported
valid **application** cache hits, while native binary imports reported their own
binary hits without source fallback. Pipeline Binary's API creation time is
comparable to the existing warm `.pso` result here; there is no demonstrated
advantage over `.pso` or evidence about reliability across driver updates.
Extraction wall was 14.300 ms (Shader Object) / 13.559 ms (Pipeline Binary),
excluded from import measurements. Native binary files and `.pso` remain separate.

Real execution used the existing Debug `cmake-build-debug` RHI build and fixed
`OpaquePerf` geometry/textures/HDR environment, 64 x 64, two samples per frame,
depth 12, SHARC off, two warmup frames and four timed frames. Each condition ran
in one independent process. Internal cache was enabled and private application
caches retained; all creation is outside the reported GPU timestamp scopes.

| Slang / Driver policy | Graph median (ms) | PathTrace pass median (ms) | Path trace shading scope median (ms) |
| --- | ---: | ---: | ---: |
| O1 / normal | 1.637504 | 1.611280 | 0.924976 |
| O1 / disable optimization | 1.346976 | 1.308256 | 0.839024 |
| O0 / normal | 1.790720 | 1.764256 | 1.059264 |
| O0 / disable optimization | 1.419632 | 1.358576 | 0.935360 |
| O2 / normal | 1.678016 | 1.653056 | 0.945024 |
| O2 / disable optimization | 1.311600 | 1.276784 | 0.843168 |

All final HDR and seven declared guide readbacks qualified. O0 and driver-flag
variants were numerically identical to the baseline; O2's maximum guide error
was 1.1920928955078125e-7, within the predeclared 2e-5 absolute/relative tolerance.
A separate O1 backend matrix qualified with identical readbacks for Pipeline,
Shader Object and Pipeline Binary; shading-scope medians were 0.971904, 1.190624
and 0.909088 ms respectively. These short Debug runs at small resolution do not
establish a GPU speedup, temporal stability, full-scene quality or memory behavior.
Use the larger Release workload and balanced process repeats before changing
the default policy. Linear HDR previews were inspected; their two-sample noise
is expected and the raw numeric comparisons remain authoritative.

Evidence, outside source control:

- `build-release/compute-guides-source/summary.json` and its sealed input/report artifacts.
- `build-release/compute-guides-binary/summary.json`; extraction excluded from the four measured samples.
- `build-release/compute-guides-pso/summary.json`; two independent warm reloads, priming excluded.
- `build/compute-dispatch-optimization/summary.json` and `build/compute-dispatch-backends/summary.json`.
- `build-release/compute-guides-plots/`: per-sample CSV and log-axis SVG distributions, separated by optimization/backend/import condition.
- `build/compute-dispatch-plots/OptimizationReferences.png`: O0/O1/O2 HDR display previews outside the sealed data.

Debug and Release builds passed. Validation covered seven RHI tests (12 initial
executions) plus the final extended Recipe roundtrip, with no VUIDs, errors or
skips. Native binary imports were exercised on independent devices/processes
with actual mapped/native buffer readbacks. Optimization/structural tests 7,
native binary container tests 4, experiment helper tests 21, legacy driver tools
tests 9 and Benchmark checks 10 passed. Three original/experimental Recipe
Benchmark replays and an explicit disable-optimization override passed with
validation; baseline hash is checked before any override. The full shader-request
suite hit its existing 300-second timeout during a CaptureSymbols/O1 OpenPBR
frontend compile; the remaining 26 cases passed separately. This timeout was
not a driver creation or Vulkan failure. Logs are retained under the reused
`cmake-build-debug` and `build-release` trees.
