# Shader and pipeline timing harness

`MetallicShaderPipelineBench` measures shader compilation and compute pipeline
creation using Metallic's production shader request catalog. Requests run
serially by default; `--pipeline-workers N` enables concurrent creation with a
shared pipeline cache. Each result belongs to one request and one iteration. The harness
does not load a scene or render frames.

The executable is a manual target outside the default build and application
dependencies. Reuse a configured build tree and run from the repository root:

```powershell
cmake --build build-pass-stages-nrd --target MetallicShaderPipelineBench
.\build-pass-stages-nrd\Source\MetallicShaderPipelineBench.exe --help
.\build-pass-stages-nrd\Source\MetallicShaderPipelineBench.exe --list
```

The build requires the same x64 Visual Studio developer environment and SDK
configuration as the existing tree; see [Build.md](Build.md).

## Select requests

`--list` prints one JSON object per catalog request, including its index, module,
entry point, defines, capabilities and search paths. Listing does not compile
shaders or create a device. `--filter` matches a case-sensitive substring of
`module.entry`; it can select many permutations. `--request-index` selects one
catalog entry. Indices belong to the current catalog, so save the request metadata
with a measurement instead of treating an index as a permanent identifier.

```powershell
$bench = '.\build-pass-stages-nrd\Source\MetallicShaderPipelineBench.exe'
& $bench --list --filter GPUProbe
& $bench --filter GPUProbe --iterations 3 --output-dir build-pass-stages-nrd/bench-probe

# Inspect the exact production variants before selecting an index.
& $bench --list --filter OpenPBRRayQueryPathTrace
& $bench --list --filter SceneRTXDI
& $bench --list --filter VisibilityBufferDeferred
```

For one of those listed indices, run `--request-index N --iterations 3
--output-dir <new-directory>`, replacing `N` with the selected index. To measure
every matching OpenPBR or RTXDI variant:

```powershell
& $bench --filter OpenPBRRayQueryPathTrace --iterations 3 --output-dir build-pass-stages-nrd/bench-openpbr
& $bench --filter SceneRTXDI --iterations 3 --output-dir build-pass-stages-nrd/bench-rtxdi
```

The OpenPBR filter also selects its Guides module. These can involve long driver
compilations; a single request index is useful when investigating one change.
`--all` explicitly selects the entire catalog. A measurement requires a selector
and `--output-dir`; an existing `report.json` is rejected to preserve prior
evidence. By default each request completes all iterations before the next request starts.

## Cache and compilation modes

| Option | Default | Behavior |
| --- | --- | --- |
| `--iterations N` | `3` | Positive sample count per selected request; report iterations are zero-based. |
| `--pipeline-workers N` | Serial, unstaged | Explicitly enable staged batches with N positive workers, including N=1 for a comparable serial baseline. |
| `--compile-only` | Off | Compile or load SPIR-V, then skip module and pipeline creation. |
| `--shader-cache disabled\|reuse` | `disabled` | Disable shader disk-cache reads and writes, or reuse/populate the selected shader cache. |
| `--pipeline-cache empty\|reuse` | `empty` | Create a new in-memory application cache per sample (per batch with `--pipeline-workers`), or load/share/save one cache across the run. |
| `--cache-dir path` | `<output-dir>/cache` | Cache root, independent of the evidence directory. SPIR-V uses `spirv/`; pipelines use `pipeline.pso`. |
| `--descriptor-mode request\|mapped\|native` | `request` | Preserve the catalog request's descriptor mode, or explicitly override it. |
| `--debug-mode disabled\|capture\|debug` | `disabled` | Select the Slang debug mode, which also affects shader cache identity. |
| `--validation` | Off | Enable Vulkan validation and count error messages in the report. |

The default pipeline cache is explicit and isolated from normal renderer PSO
files. Empty caches are not persisted. Reuse mode loads an existing harness cache
when compatible and saves it after the samples, with save time recorded
separately. Shader cache reuse can produce a compile miss followed by cache-hit
samples within one run; check `shader_cache_hit` for every sample.

Use an independent cache root and fresh evidence directories to compare repeated
process runs without overwriting a report:

```powershell
& $bench --filter GPUProbe --shader-cache reuse --pipeline-cache reuse --cache-dir build-pass-stages-nrd/bench-cache --output-dir build-pass-stages-nrd/bench-reuse-1
& $bench --filter GPUProbe --shader-cache reuse --pipeline-cache reuse --cache-dir build-pass-stages-nrd/bench-cache --output-dir build-pass-stages-nrd/bench-reuse-2
& $bench --filter SceneRTXDI --compile-only --output-dir build-pass-stages-nrd/bench-rtxdi-compile
```

Compile-only runs do not create a logical Vulkan device. Native descriptor
requests still query a supported adapter for descriptor strides. The report
records each effective mode and stride; `request` preserves explicit catalog
modes and allows the normal resolution of auto modes. Device initialization is
timed separately from the first pipeline sample.

## Concurrent pipeline creation

Explicit `--pipeline-workers N` prepares every selected compute shader and module
serially, then creates their pipelines in one batch per iteration. The worker
threads synchronize at a barrier before the batch timer starts measuring work.
Every job in the batch uses the same cache. `empty` creates a fresh shared cache
per iteration; `reuse` shares the loaded cache across all batches and saves after
all workers finish. Compilation, module creation, cache creation/loading and
thread startup are outside the batch measurement. The measured batch includes
barrier release, scheduling, result bookkeeping, pipeline destruction and joins;
per-call `pipeline_wall_ms` ends when pipeline creation returns.

Use explicit workers=1 and workers=N with the same selection, iterations and
cache policy. This keeps batching and preparation order identical. For example:

```powershell
& $bench --filter AutoExposure --iterations 3 --pipeline-workers 1 --output-dir build-pass-stages-nrd/bench-workers-1
& $bench --filter AutoExposure --iterations 3 --pipeline-workers 3 --output-dir build-pass-stages-nrd/bench-workers-3
```

Actual worker count is capped by the number of successfully prepared compute
requests in each batch; select at least N compute requests to exercise N-way
concurrency. A single request index does not create simultaneous copies of the
same pipeline. `--pipeline-workers` rejects `--compile-only`. A failed job is
recorded without abandoning other jobs, and workers finish before any cache is
saved or destroyed. Only the main thread updates the report.

The cache uses Vulkan's internal synchronization. Metadata locking does not
cover native pipeline creation, but the driver may still serialize internally.
Batch elapsed time and per-call duration sums expose their measured scopes;
their ratio alone does not establish cold-driver compilation throughput.

## Measurement boundaries

Shader acquisition, shader module creation and pipeline creation are separate
operations. Shader acquisition can invoke Slang or read a cached shader binary,
depending on the selected shader cache mode. A cache hit is not a compiler timing.
Shader module creation measures the RHI operation that creates the Vulkan shader
module. Pipeline creation measures the RHI call independently of those steps.

Native Vulkan pipeline creation and optional driver feedback provide narrower
measurements inside the RHI operation. They need not equal its wall-clock time:
the wrapper can also perform layout preparation, synchronization and bookkeeping.
Unavailable or invalid feedback is not a zero-duration driver compilation.

An empty application pipeline cache does not make the driver cold. Driver-managed
disk and process caches, compiler state, filesystem caches, validation, device
initialization and earlier iterations can all affect results. Keep build,
GPU/driver, request selection, descriptor/debug modes and validation settings
identical when comparing changes. Retain individual iterations instead of
reporting only an average.

The harness does not establish rendering correctness, frame performance or
startup duration. Production graphics requests can be compiled, but graphics
pipeline creation requires pass-specific attachment and fixed-function state
that is outside this harness's compute pipeline scope.

## Read the evidence

`<output-dir>/report.json` has `schema_version: 2`, the configuration, selected
request indices and an ordered `samples` array. Each sample retains its full
request, effective descriptor mode, dependencies, SPIR-V size and FNV-1a hash,
diagnostics and status. Times are milliseconds.

| Field | Measured scope |
| --- | --- |
| `compile_wall_ms` | ShaderRegistry shader acquisition: Slang compilation or cached-binary acquisition. |
| `shader_module_wall_ms` | RHI shader module creation. |
| `pipeline_wall_ms` | RHI compute pipeline creation, including its wrapper work. |
| `pipeline.native_create_ms` | Native Vulkan compute pipeline creation call. |
| `pipeline.driver_ms` | Valid driver pipeline-creation feedback duration; otherwise `null`. |
| `pipeline.cache_lock_wait_ms` | Time waiting to acquire the short application cache metadata locks. |

Pipeline timings and feedback are immutable per-pipeline creation statistics,
not cache-wide `last*` values that another worker can overwrite.
`pipeline_batches` records requested/actual workers, sample indices, iteration,
`pipeline_batch_wall_ms`, `pipeline_wall_sum_ms`, successful native duration and
metadata lock wait sums, plus metadata/native hit/miss counter deltas. The sums
named `successful_*` exclude failed creates, whose wall time remains in samples.
Worker mode records preparation validation errors per sample and pipeline-phase
errors per batch, since concurrent validation callbacks cannot be attributed
reliably to an individual request. Root validation errors cover the whole run.

`pipeline.metadata_hit` identifies an application PSO metadata match.
`pipeline.native_cache_hit` reflects driver feedback and is `null` when feedback
is invalid; it is distinct from a metadata match. Neither a metadata miss nor a
native cache miss alone proves the driver compiled the shader from scratch.
`pipeline.feedback_valid` states whether the driver supplied valid feedback.

Root fields include device/driver identifiers when a device was created,
`device_initialization_wall_ms`, `validation_errors`, `failures` and overall
`status`, Slang build and shader target revision. `summary` groups successful
timings by exact request with count/min/median/max; raw samples retain their cache
states, so do not interpret a mixed cache-hit/miss median as uncached compilation.
`pipelines_created` and `pipelines_skipped` count completed pipeline phases.
The report is checkpointed before expensive phases and after samples.
An interrupted process can leave `running` status and a sample in `compiling`,
`creating_device`, `creating_pipeline_cache`, `creating_shader_module`, `prepared` or `creating_pipeline`; treat that
sample as incomplete. Checkpoint writes are not a guarantee against file
truncation if the process is interrupted during a write.

Completed compile-only samples use `pipeline_status: "compile_only"`. Graphics
requests use `pipeline_status: "skipped_non_compute"` in normal runs. Overall
`passed` means the attempted phases succeeded; these skip statuses do not certify
graphics pipeline creation or rendering. Validation errors make the run fail
even if pipeline creation returned success.
`validation_messages` retains up to 256 diagnostics with severity, category, ID,
name and text; `validation_messages_dropped` reports overflow. Per-sample
`validation_errors` includes errors during that sample's device initialization.
Loader errors such as missing layer manifests also fail a validation run and
remain visible in this array rather than being silently ignored.

Exit status `0` means all attempted phases passed; `1` means a runtime,
compilation, pipeline, validation or evidence-write failure; `2` means invalid
arguments, no matching request or an existing report. Per-sample failures include
`failed_phase` and `error`; inspect both the report and process exit code.

## Regression checks

Build the executable before running its CTest checks:

```powershell
ctest --test-dir build-pass-stages-nrd -R '^MetallicShaderPipelineBenchTests$' --output-on-failure
python -B tests/perf/TestShaderPipelineBench.py --executable build-pass-stages-nrd/Source/MetallicShaderPipelineBench.exe --gpu
```

The default checks use mapped compile-only requests without a GPU. `--gpu` adds
pipeline creation and verifies empty caches, within-process hits, persisted cache
reloads and timing fields; it does not render a scene or enable validation layers.

Validated on 2026-10-11 using the existing Release `build-pass-stages-nrd` tree:
six checks including the optional GPU case passed. GPUProbe also passed native
descriptor mode, an actual RTXDI request completed two compile/create iterations,
and a graphics request explicitly reported a skipped pipeline. Validation-mode
creation succeeded but the run correctly failed on two existing loader errors
for missing layer manifests; both were retained in JSON. Evidence is under the
build tree's `shader-bench-*` directories. These smoke measurements are not a
controlled performance comparison or rendering correctness test.

The metadata-lock follow-up on 2026-10-11 passed seven harness checks, including
explicit one-worker and three-worker batches, plus five RHI cache tests. The RHI
concurrency test holds the first native creation entry until another caller
enters, with a bounded failure guard; this checks application-side overlap
independently of driver compilation speed. It also validates 32 same-cache
creations against exact feedback/revision counts and concurrent save/reload.
The graphics and external graphics feedback/readback tests passed; no Vulkan
VUID errors appeared (the existing missing loader-manifest warnings remain).

A serial/parallel/parallel/serial run used the same three AutoExposure requests,
five batches per process, reused isolated SPIR-V cache, and a fresh shared empty
application pipeline cache per batch. Compilation/module preparation was outside
batch timing. The serial batch medians were 2.6430 and 2.5239 ms; three-worker
medians were 2.1352 and 2.3218 ms. Each parallel run created 15 pipelines with
total metadata lock waits of 0.0065 and 0.0041 ms. All native feedback reported
application-cache misses; driver internal/OS caches were not cleared or disabled.
These short batches include scheduling/join overhead and do not establish a
production shader compilation speedup. Evidence is in
`build-pass-stages-nrd/pipeline-concurrency-comparison.json`, the four
`pipeline-concurrency-{serial,parallel}-{a,b}/report.json` files, and
`pipeline-cache-concurrency-rhi.log` under the same build tree.
