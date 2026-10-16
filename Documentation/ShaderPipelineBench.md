# Shader and pipeline timing harness

`MetallicShaderPipelineBench` measures shader compilation and pipeline creation
using Metallic's production shader request catalog or a captured PSO recipe.
Requests run serially by default; explicit modes stage comparable creation batches.
Each result belongs to one request, iteration and replica. The harness
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
The manual catalog also appends the opt-in
[primary-guides split experiment](OpenPBRPrimaryGuidesSplit.md); these two requests
are excluded from normal application warmup.
`--all` explicitly selects the entire catalog. A measurement requires a selector
and `--output-dir`; an existing `report.json` is rejected to preserve prior
evidence. By default each request completes all iterations before the next request starts.

## Cache and compilation modes

| Option | Default | Behavior |
| --- | --- | --- |
| `--iterations N` | `3` | Positive sample count per selected request; report iterations are zero-based. |
| `--pipeline-workers N` | Serial, unstaged | Explicitly enable staged batches with N positive workers, including N=1 for a comparable serial baseline. |
| `--mode serial\|parallel-shared\|parallel-sharded` | Unstaged `serial` | Explicit modes prepare all jobs before measuring each iteration. Shared mode uses one cache; sharded mode uses one cache per actual worker. |
| `--workers N` | `1` | Worker count, capped by prepared jobs. Legacy `--pipeline-workers` also selects shared mode when no mode is specified. Serial mode requires one worker. |
| `--pipeline-replicas N` | `1` | Create N copies of each selected PSO per iteration; useful for testing a fixed request with 1/2/4/8 workers. |
| `--cache-lock-scope metadata\|creation` | `metadata` | Production short metadata lock or a controlled baseline holding the same lock through native creation. Both use the same Vulkan cache flags. |
| `--merge-shards` | Off | Merge quiescent shards after every batch; separately time merge and final persistence. Requires sharded mode. |
| `--recipe-file path` | Off | Replay one exported compute/graphics PSO from exact input SPIR-V and state, bypassing Slang. Replaces catalog selection. |
| `--compile-only` | Off | Compile or load SPIR-V, then skip module and pipeline creation. |
| `--shader-cache disabled\|reuse` | `disabled` | Disable shader disk-cache reads and writes, or reuse/populate the selected shader cache. |
| `--pipeline-cache empty\|reuse` | `empty` | Create a new in-memory application cache per sample (per batch with `--pipeline-workers`), or load/share/save one cache across the run. |
| `--cache-dir path` | `<output-dir>/cache` | Cache root, independent of the evidence directory. SPIR-V uses `spirv/`; pipelines use `pipeline.pso`. |
| `--descriptor-mode request\|mapped\|native` | `request` | Preserve the catalog request's descriptor mode, or explicitly override it. |
| `--debug-mode disabled\|capture\|debug` | `disabled` | Select the Slang debug mode, which also affects shader cache identity. |
| `--slang-optimization default\|O0\|O1\|O2\|O3` | `default` | Per-request frontend optimization; default honors `METALLIC_SLANG_OPTIMIZATION`, otherwise O1 except ShaderDebug's O0. Record the effective level and final SPIR-V hash. |
| `--disable-driver-optimization` | Off | Set Vulkan's disable-optimization creation flag for compute/graphics pipelines; uses a distinct PSO identity. |
| `--compute-backend pipeline\|shader-object\|pipeline-binary` | `pipeline` | Direct pipeline creation, compute `vkCreateShadersEXT`, or KHR pipeline-binary experiment. Experimental backends use no `VkPipelineCache`. |
| `--native-binary-cache-dir path` | Off | Persist/import native shader-object or pipeline binaries in an independent cache; prime and import are separate conditions. |
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

## Large compute compilation experiments

Use one catalog request and explicit optimization levels. The output includes
`effective_slang_optimization`, `spirv_statistics`, and the input SPIR-V hash.
Statistics count instructions, functions, calls, basic blocks, merges, branches,
switches, Phi nodes, and the largest functions in the final module. They describe
module structure, not GPU executable instruction counts or proof of a compiler
bottleneck. Recipe replay records statistics for each stage and uses the captured
bytecode, so it rejects frontend optimization overrides.

```powershell
& $bench --request-index N --iterations 1 --slang-optimization O1 `
  --shader-cache reuse --compile-only --cache-dir build-release/compile-cache `
  --output-dir build-release/compile-prefill

$env:METALLIC_VK_INTERNAL_PIPELINE_CACHE = 'disabled'
& $bench --request-index N --iterations 1 --slang-optimization O1 `
  --shader-cache reuse --pipeline-cache empty --cache-dir build-release/compile-cache `
  --disable-driver-optimization --output-dir build-release/compile-disable-opt
Remove-Item Env:METALLIC_VK_INTERNAL_PIPELINE_CACHE
```

Replace `N` with the listed request index. Prefill separately for every Slang
level. Driver A/B measurements at one level must reuse identical SPIR-V; O0/O1/O2
comparisons deliberately change the input module. Keep debug information,
validation, descriptor ABI, device/driver, scene and settings fixed. The Vulkan
flag may reduce creation time; it guarantees no particular speedup
([VkPipelineCreateFlagBits](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCreateFlagBits.html)).
Shader Object has no equivalent pipeline flag and rejects that combination.
Experimental compute backends require serial mode. Their native binary file
locks are distinct from the `VkPipelineCache` metadata lock measured by the
concurrency modes.

Binary import reports `native_binary_cache_hit`, `native_binary_driver_rejected`,
`native_binary_data_bytes`, and separate import/export time. This hit describes
an explicitly imported native blob. Pipeline creation feedback describes the
application pipeline cache; internal driver-cache hits remain unobserved. Compare
binary import to both source creation and existing `.pso` reuse with the same
shader and settings. Prime/export samples and persistence time are outside the
import sample distribution.

Creation-only runs never dispatch arbitrary production shaders with dummy
resources. Use the [compute experiment runners](ComputePipelineExperiments.md)
for process-isolated matrices and real RenderGraph dispatch/readback comparisons.
Evaluate GPU time and image differences before adopting a faster creation policy;
production defaults remain unchanged.

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

Actual worker count is capped by the number of successfully prepared jobs in
each batch. `--pipeline-replicas 8` allows one request to exercise eight workers.
Replicas intentionally repeat PSOs and may benefit from intra-batch driver/cache
reuse; a varied request selection measures a different workload. Keep selection,
replicas and iterations identical across comparisons. Explicit modes reject
`--compile-only`. A failed job is
recorded without abandoning other jobs, and workers finish before any cache is
saved or destroyed. Only the main thread updates the report.

The cache uses Vulkan's internal synchronization. Metadata locking does not
cover native pipeline creation, but the driver may still serialize internally.
Batch elapsed time and per-call duration sums expose their measured scopes;
their ratio alone does not establish cold-driver compilation throughput.

For explicit `serial`, `parallel-shared` and `parallel-sharded`, each iteration
prepares the same request/replica sequence. Preparation may acquire the shader
disk cache for each replica. Workers use fixed strided assignments, so shard
ownership repeats across iterations. `empty` creates one fresh cache per batch
or worker; `reuse` retains those caches and loads/saves `pipeline.pso` or
`pipeline-shard-N.pso`. A batch with preparation failures reports fewer actual
workers; reject it for an otherwise matched comparison.
Explicit `serial` uses the same per-batch cache policy as parallel modes.
The default unstaged serial path retains its historical fresh cache per sample
when `--pipeline-cache empty` is selected.

Use the matrix helper for 1/2/4/8 workers and both shared lock scopes. It prefills
SPIR-V, keeps job counts and input hashes fixed, and writes independent process
reports plus a combined summary. Repeats reverse the cell order on alternate
runs. Merge time is excluded from throughput and retained separately.

```powershell
python -B Tools/Perf/RunShaderPipelineMatrix.py --executable $bench `
  --filter AutoExposure --iterations 5 --repeats 2 --pipeline-replicas 8 `
  --merge-shards --output-dir build-release/bench-matrix
```

The `creation` lock scope emulates the broad application lock using the current
backend; it is an A/B control, not an archived binary of an older renderer.
The production default stays `metadata`. Vulkan only requires external cache
synchronization during creation when the cache has the externally synchronized
flag; see [vkCreateComputePipelines](https://docs.vulkan.org/refpages/latest/refpages/source/vkCreateComputePipelines.html).
The harness joins workers before merging because merge destination access has
its own external synchronization requirement; see
[vkMergePipelineCaches](https://docs.vulkan.org/refpages/latest/refpages/source/vkMergePipelineCaches.html).

## Independent-process driver cache experiments

`--pipeline-cache empty` resets the application cache only. To isolate a fixed
shader, first populate its SPIR-V cache, then start a fresh process for every
native creation. The helper below selects one exact request index, excludes
SPIR-V prefill and driver prime runs from timings, alternates controlled cache
disabled/enabled-primed samples, and rejects changed bytecode, request settings,
device/driver information, or a shader-cache miss during measurement.

```powershell
& $bench --list --filter OpenPBRRayQueryPathTrace
# Replace N with one actual compute request index from that list.
python -B Tools/Perf/RunShaderPipelineDriver.py --executable $bench `
  --request-index N --pairs 3 --cache-dir build-release/bench-driver-cache `
  --output-dir build-release/bench-driver
```

The helper explicitly uses `METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled` and
`enabled` so both sides use the same diagnostic device feature/extension chain.
Each measured process uses one iteration and an empty application cache.
`enabled-primed` means a preceding independent process created that same PSO;
the driver need not retain its internal cache across processes. `disabled`
disables the controllable cache path; it does not guarantee every compiler, OS,
or driver state is cold. The reports preserve SPIR-V hashes, debug/descriptor/
validation settings, device and driver identifiers, process arguments and logs.
Compare `native_create_ms` and valid `driver_ms` separately; an application-cache
hit bit does not report hidden driver-cache hits.

## Capture and replay real PSO recipes

Set `METALLIC_PIPELINE_RECIPE_DIRECTORY` while exercising the desired renderer
pass. ShaderRegistry exports successful RHI compute/graphics pipelines with
their exact input SPIR-V, stage entry points, attachment formats, topology,
raster/depth state, task subgroup requirements, bindless layout and indirect
binding state. The manifest records source device features and debug/descriptor
metadata. Content-addressed `.spv` files must remain beside the manifest.

```powershell
$env:METALLIC_PIPELINE_RECIPE_DIRECTORY = "$PWD/build-release/recipes"
try { & .\build-release\Source\Metallic.exe --smoke-test }
finally { Remove-Item Env:METALLIC_PIPELINE_RECIPE_DIRECTORY }

# Choose a captured graphics *.recipe.json from the pass being investigated.
& $bench --recipe-file .\build-release\recipes\graphics-<pso-hash>.recipe.json `
  --mode parallel-shared --workers 4 --pipeline-replicas 8 --iterations 3 `
  --pipeline-cache empty --output-dir build-release/bench-graphics

python -B Tools/Perf/RunShaderPipelineDriver.py --executable $bench `
  --recipe-file .\build-release\recipes\graphics-<pso-hash>.recipe.json `
  --pairs 3 --output-dir build-release/bench-graphics-driver
```

Replay recreates source-device feature selections, bypasses Slang, verifies
blob sizes/content hashes, and checks the resulting PSO hash against the
captured hash. A changed backend state contract or feature-dependent shader
rewrite fails the hash check instead of silently claiming an identical PSO.
Source and replay devices and validation settings are reported separately.
`--validation` controls replay validation. Graphics shader catalog entries
still compile alone because they lack actual pass state; recipe replay supplies
that state. External native graphics adapters with custom vertex/descriptor
ABIs and shader-object programs are outside this RHI recipe format.
Export requires modules created through ShaderRegistry while capture is enabled.
Recipes deduplicate by native PSO hash; equivalent input requests that produce
the same native PSO share a captured recipe rather than reporting their frequency.

```powershell
python -B Tools/Perf/PlotShaderPipelineBench.py --help
python -B Tools/Perf/PlotShaderPipelineBench.py build-release/bench-driver-openpbr `
  build-release/bench-driver-graphics build-release/bench-pipeline-matrix `
  --log-scale --output-dir build-release/bench-pipeline-plots-log
```

The plotting helper flattens successful native creates to CSV, retaining
pipeline type, per-stage module/entry metadata, SPIR-V bytes, driver feedback
hit/unknown status and separate native/driver/lock timings. It writes scatter,
CDF and module/entry distributions; graphics stages are retained as one PSO
identity, with total SPIR-V size. It also works with a standard-library SVG
fallback when Matplotlib is unavailable. Keep generated reports and charts in
build/output directories.
`--log-scale` makes microsecond-through-minute samples readable, using a display
floor of 1 ns for zero-duration samples; raw CSV values remain unchanged.

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
startup duration. Production graphics requests can be compiled alone; actual
graphics creation uses captured pass-specific recipes.

## Read the evidence

`<output-dir>/report.json` has `schema_version: 4`, the configuration, selected
request indices and an ordered `samples` array. Each sample retains its full
request, effective descriptor mode, dependencies, SPIR-V size and FNV-1a hash,
diagnostics and status. Times are milliseconds.

| Field | Measured scope |
| --- | --- |
| `compile_wall_ms` | ShaderRegistry shader acquisition: Slang compilation or cached-binary acquisition. |
| `shader_module_wall_ms` | RHI shader module creation. |
| `pipeline_wall_ms` | RHI compute pipeline creation, including its wrapper work. |
| `pipeline.native_create_ms` | Native Vulkan pipeline or shader-object creation calls, excluding binary-file I/O and pipeline-binary handle import/export. |
| `pipeline.driver_ms` | Valid driver pipeline-creation feedback duration; otherwise `null`. |
| `pipeline.cache_lock_wait_ms` | Time waiting to acquire the short application cache metadata locks. |

Pipeline timings and feedback are immutable per-pipeline creation statistics,
not cache-wide `last*` values that another worker can overwrite.
`pipeline_batches` records mode, cache count, requested/actual workers, sample indices, iteration,
`pipeline_batch_wall_ms`, `pipeline_wall_sum_ms`, successful native duration and
metadata lock wait sums, P95 (nearest-rank), throughput in successful pipelines
per second, plus per-call metadata/native hit/miss counts. The sums
named `successful_*` exclude failed creates, whose wall time remains in samples.
`pipeline_attempt` retains independent RHI metrics on native failures as well.
`attempted_*` sums include those attempts; driver sums are `null` without valid
feedback and retain the valid-feedback sample count. The P95 native/lock fields
cover successful creates, while wall P95 covers attempted calls. Cache merge
and save timings do not contribute to creation throughput.
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
Unhandled native failures keep their OS exit code. The Windows harness suppresses
modal crash dialogs; this does not convert an access violation into success.
The Python regression runner stops on the first native crash or timeout and
copies reports, arguments and captured output to `.cache/ShaderPipelineBenchFailures/`
before deleting its temporary directory. Driver/matrix helpers retain process
exit/crash/timeout metadata and logs in their evidence directories.

## Regression checks

Build the executable before running its CTest checks:

```powershell
ctest --test-dir build-pass-stages-nrd -R '^MetallicShaderPipelineBenchTests$' --output-on-failure
python -B tests/perf/TestShaderPipelineBench.py --executable build-pass-stages-nrd/Source/MetallicShaderPipelineBench.exe --gpu
```

The default checks use mapped compile-only requests without a GPU. `--gpu` adds
pipeline creation and verifies empty caches, within-process hits, persisted cache
reloads and timing fields; it does not render a scene or enable validation layers.

The additional helper integrity checks run without a GPU:

```powershell
python -B tests/perf/TestShaderPipelineDriverTools.py
```

On 2026-10-12 the Release `build-release` benchmark and Debug
`cmake-build-debug` RHI targets built successfully. The benchmark passed 9/9
checks with `--gpu`, including 13 explicit mode/worker/lock-scope cells and
shard merge/reload; helper integrity checks passed 9/9. Four focused RHI checks
passed with Vulkan validation: cache concurrency/deferred saves, native
compute/graphics feedback and recipe roundtrip. No Vulkan validation diagnostics
appeared; the save-failure fixture produced its expected write-retry warning.

Release validation exposed and fixed a Vulkan symbol collision with the
Streamline import library. The benchmark explicitly links `MetallicRuntimeRender`
then `metallic::volk`, resolving writable Volk PFN globals before the interposer's
same-name Vulkan exports. Before the fix, device initialization wrote an import
thunk in the executable's read-only code and raised `0xC0000005`, including in
serial runs. A debugger trace identified that write; the fixed executable's
import table has no Vulkan function imports from `sl.interposer.dll`, and its
native startup and complete GPU regression now pass. Other targets' link
configuration was not changed.

Independent-process measurements on RTX 5060 / NVIDIA 616.92 / Release used
mapped descriptors, disabled Slang debug output and no replay validation.
Each condition has three samples, fixed SPIR-V and an empty application cache;
Slang prefill and the enabled-driver prime process are excluded:

| PSO | SPIR-V bytes | Cache control | Native median (ms) | Driver feedback median (ms) |
| --- | ---: | --- | ---: | ---: |
| OpenPBR Request 145 | 6,664,180 | Disabled | 57,382.17 | 57,382.16 |
| OpenPBR Request 145 | 6,664,180 | Enabled, primed | 52,894.69 | 52,894.68 |
| Captured Triangle graphics recipe | 1,716 | Disabled | 14.66 | 14.64 |
| Captured Triangle graphics recipe | 1,716 | Enabled, primed | 14.39 | 14.38 |

OpenPBR input hash is `c4d29c8e5629f6a6`; all Graphics replay hashes matched.
Every sample reported valid feedback and an application-cache miss. Internal
driver-cache hits remain unobserved. Durations drifted downward across the
OpenPBR sequence, so the median difference is not sufficient evidence of a
driver-cache speedup. These two PSOs do not establish the full scene's PSO mix.
Evidence: `build-release/bench-driver-openpbr/summary.json` and
`build-release/bench-driver-graphics/summary.json`.

The AutoExposure matrix used three requests × eight replicas per batch, three
batches per process, two process repeats with reversed order, warm SPIR-V,
disabled controllable internal cache and fresh application caches. The rows
below are medians of six batches; native and lock values are per-batch sums,
P95 is the per-batch call-wall P95, and merge is excluded from throughput:

| Mode / lock scope | Workers | Total (ms) | Native sum (ms) | Lock sum (ms) | P95 (ms) | Pipelines/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Serial / metadata | 1 | 89.97 | 89.12 | 0.0029 | 14.64 | 266.75 |
| Shared / creation | 8 | 91.32 | 89.86 | 631.03 | 74.46 | 262.87 |
| Shared / metadata | 8 | 98.99 | 779.21 | 0.0624 | 64.56 | 242.47 |
| Sharded / metadata | 8 | 103.76 | 816.71 | 0.0116 | 64.88 | 231.76 |

The short application lock removes its large wait contribution in this workload,
but neither shared concurrency nor sharding improved batch throughput here.
Overlapping native duration sums exceed elapsed time and can include driver
synchronization; they are not additional elapsed time. Eight-shard merge median
was 0.0075 ms. Keep the other worker counts and raw cache-hit samples in
`build-release/bench-pipeline-matrix/summary.json`; the distribution export and
CSV are in `build-release/bench-pipeline-plots-log/`. These are pipeline creation
measurements without dispatch, frames, temporal or full-scene memory validation.

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
