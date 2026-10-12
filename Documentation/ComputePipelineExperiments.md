# Compute compilation and execution experiments

`RunComputePipelineExperiments.py` measures native creation separately from real
GPU work. Its `compile` command invokes `MetallicShaderPipelineBench`; `dispatch`
invokes the production RenderGraph harness in `MetallicRHITests`. Each command
records its schedule, inputs, process logs and results. It never edits shaders.
Run from the repository root using a current build and Python 3.11 or later;
dispatch readback comparisons also require numpy.

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

Each Slang optimization level first fills a private shader cache. Measured
processes must load those exact SPIR-V bytes; Slang compilation is excluded from
native creation. Different levels may produce different bytecode. The native
disable-optimization pair must retain the same input and final device SPIR-V,
descriptor ABI, debug/validation settings and physical device/driver. The report
retains both SPIR-V statistics and bytecode identities, including the device
rewrites. Statistics describe SPIR-V structure, not physical registers or final
driver machine code.

Compare `pipeline_wall_ms` for the total RHI creation/loading path.
`native_create_ms` measures the actual `vkCreateComputePipelines` or
`vkCreateShadersEXT` call. `native_binary_import_ms` measures preparation before
that call: shader-object file/CPU loading, or pipeline-binary file loading and
`vkCreatePipelineBinariesKHR` handle restoration. Export preparation has its own
`native_binary_export_ms`. A smaller final native call alone does not describe
the total binary-loading benefit.

All measured source/import processes request
`METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled` and an empty application cache.
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

The runtime controls are `METALLIC_SLANG_OPTIMIZATION`,
`METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION`, `METALLIC_VK_COMPUTE_BACKEND`,
`METALLIC_VK_NATIVE_BINARY_CACHE_DIR` and `METALLIC_VK_PIPELINE_BINARY`.
The runner sets them before process launch and retains them in evidence. It
clears inherited `METALLIC_*` controls and rejects injected Vulkan layers for
ordinary measurements. The production default remains Slang O1 with normal
pipeline optimization. Creation timing alone does not justify changing it.

## Evidence and validation

`summary.json` checkpoints progress, failures, cache conditions and measured
scope. Successful results are sealed by `Manifest.json`; verify them using:

```powershell
python -B Tools/Perf/RunComputePipelineExperiments.py verify <experiment-directory>
python -B tests/perf/TestComputePipelineExperiments.py
```

The verifier checks artifact hashes and recomputes native/GPU timing and
correctness evidence. Keep generated evidence outside source control. Existing
GPU experiment locks cause a failure: inspect the recorded owner and preserve
its work before starting another experiment. The lock prevents cooperating
Metallic runners from overlapping; it cannot establish global GPU exclusivity.
GPU clocks are unaltered and no other applications are closed by this runner.

## Initial measurements, 2026-10-12

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
