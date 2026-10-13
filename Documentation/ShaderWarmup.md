# Shader cache warmup

For isolated shader compilation and compute pipeline creation measurements, use
the manual [ShaderPipelineBench harness](ShaderPipelineBench.md). It does not run
startup warmup or render frames and reports compilation and pipeline timings separately.

After configuring the project, run this target manually:

~~~powershell
cmake --build build-release --target MetallicShaderWarmup --config Release
~~~

Use your own build directory/configuration in place of the example. The target
builds the small MetallicShaderCompiler executable and warms the project's
.cache/shaders/spirv directory using the same SlangCompiler.cpp as the runtime.
It does not start Metallic, create a GPU device, load scenes, or build the editor.
Mapped-mode compilation requires no GPU. Native-mode compilation queries Vulkan
physical devices for descriptor strides, without creating a logical device or
submitting work. Supplying both strides explicitly also permits offline native
compilation.

The editor, LookDev, all rendering samples, and MetallicShaderPrintfProbe now
warm the complete catalog synchronously before initializing rendering. Warmup
failure aborts startup with a nonzero exit code. To explicitly skip it:

~~~powershell
.\build-release\Source\Metallic.exe --skip-shader-warmup
.\build-release\Source\LookDev.exe --skip-shader-warmup
.\build-release\Source\MetallicPathTracingSample.exe --skip-shader-warmup
~~~

The flag also works with smoke-test launches and the GPU printf probe. Help,
sample listing, offline cooking, CLI control tools, auxiliary launchers and test
executables do not automatically run this startup warmup.

The shared implementation runs inside each rendering process, after selecting
its shader debug mode and before GPU initialization. It uses the existing
parallel workers and progress display, then restores application logging.
When requests use native descriptor heaps, warmup queries a supported adapter's
resource/sampler stride pair before starting workers. Every request compiles once;
the pair is passed only to native requests. Listing requests and explicitly
skipping warmup do not query Vulkan.
There is no external compiler process to locate or deploy. Startup always uses
the complete catalog; METALLIC_SHADER_WARMUP_ARGS only configures the manual
target and cannot bypass or filter startup warmup.

The standalone compiler and manual warmup target remain outside default builds.
Rendering targets link the shared implementation; building them does not execute
warmup. Runtime compile-on-cache-miss remains available for unlisted variants,
source changes, and launches using --skip-shader-warmup.

Repeated invocations validate dependencies and reuse current entries. Compilation
errors or failure to read back a newly written cache entry produce a nonzero exit
code; other requests are still attempted. This warms SPIR-V, not driver PSO caches.

## Parallel compilation and progress

Warmup uses a worker pool. The default is the available CPU thread count capped
at 8 to limit simultaneous Slang compiler memory use. The environment variable
`METALLIC_SHADER_WARMUP_JOBS` overrides this default for both application startup
and the standalone compiler. `--jobs N` takes precedence over the environment
variable; `--jobs 1` restores serial compilation. Values must be positive integers;
an unset or empty environment variable uses the default. Invalid values fail
warmup before compilation. Listing, help, and explicitly skipped startup warmup
do not require a valid worker setting. The actual worker count is capped by the
number of selected requests.

More workers increase simultaneous compiler memory use. Compare cold-cache
elapsed time and peak memory before increasing beyond the default; cache-hit
runs do not measure compilation throughput.

~~~powershell
.\build-release\Source\MetallicShaderCompiler.exe --jobs 8
# Also applies when starting a sample from this shell (or its IDE environment).
$env:METALLIC_SHADER_WARMUP_JOBS = '12'
.\build-release\Source\MetallicPathTracingSample.exe
Remove-Item Env:METALLIC_SHADER_WARMUP_JOBS
cmake -S . -B build-release "-DMETALLIC_SHADER_WARMUP_ARGS=--jobs;8"
cmake --build build-release --target MetallicShaderWarmup
~~~

CMake's --parallel option controls the build, not shader compiler workers.
Each worker reuses its thread-local Slang global session (the standard library),
while every request gets a fresh compilation session. Source edits and macro
permutations therefore do not reuse imported module IR. SDK elapsed-time counters
are sampled before and after each request, rather than reporting accumulated
time from the worker's earlier compilations. Expensive path-tracing and Deferred
requests start first; listing and progress request indices retain catalog order.
Progress reports completed
requests/total, percentage, cache hits, failures, elapsed time, and the completed
shader name. Completion order may differ from request order. Errors are counted
per request and remaining requests are still processed before the final summary.
The progress percentage measures request count, not estimated remaining time.
Each completion also reports per-request total/setup/module/code-generation
seconds. The summary sums these overlapping worker intervals; those sums are
not complete warmup wall time.

### Complete cold-cache measurements

Use the evidence runner to measure the complete compiler process, including
catalog preparation, native stride queries, cache read-back and teardown:

~~~powershell
python -B Tools/Perf/MeasureShaderWarmup.py `
  --executable build-release/Source/MetallicShaderCompiler.exe `
  --output-dir .cache/shader-warmup-measurement-new --jobs 8 --repeats 3
~~~

The output directory must be new. Every repetition starts an independent process
with a new, empty SPIR-V cache; no shader IR disk cache is introduced. The runner
preserves compiler settings and records the inherited environment, full catalog,
source/EXE/DLL SHA256 identities, raw logs, phases and Windows process memory
peaks. It rejects cache hits, incomplete catalogs, compilation/read-back failures
and changing inputs. OS filesystem caches are not flushed. Listing prepares
generated material source before freezing inputs; the timed processes still
perform their own complete catalog preparation.

The default acceptance threshold is strictly below 10 seconds for **every**
repetition. Valid measurements above the threshold have `status: passed` and
`goal_reached: false`; the runner returns 1 until the goal is reached. A failed
measurement also returns 1, with `status: failed` and a diagnostic in the report.
Shader/pipeline creation and GPU dispatch performance remain separate scopes.

The first 2026-10-13 Release/O1, debug-disabled, eight-worker iteration measured
107.9 seconds of internal warmup time for 202 requests. Standard-library reuse,
vertex-request sharing and early heavy-program scheduling measured 93.7 seconds
for 187 requests. Deferred field reads and shared OpenPBR sparse-training
requests then measured 81.2 seconds internally (81.478 seconds for the complete
process), for 186 requests. All three used empty SPIR-V caches and reported zero
existing hits/failures; these are single exploratory runs, not repeated timing
distributions or evidence that the 10-second goal has been achieved. Local logs
are under `.cache/SlangCompileGoal/` and remain outside source control.

After refining the scheduling priorities, three fresh-process measurements of
the same 186-request catalog took **82.388, 65.226 and 65.990 seconds** (median
65.990). Every run had zero existing hits/failures and unchanged input SHA256
identities. Peak working sets were 6.24-6.37 GiB, with 7.08-7.27 GiB peak private
commit. OS filesystem caches were allowed to remain warm; the range is retained
rather than dropping the slower run. The initial 107.9-second result is a single
exploratory internal-timer baseline, not a matched repeated comparison against
this complete-process distribution. The evidence runner correctly reports
`status: passed`, `goal_reached: false`; the 10-second target remains open.
The sealed input identities and raw logs are in
`.cache/SlangCompileGoal/round3-jobs8/summary.json`.

The request sharing preserves runtime coverage: editor vertex code does not
consume fragment display macros, and OpenPBR `SHARC_UPDATE=1` Guides dispatches
the ordinary sparse-training function without writing guides. Only an
unambiguous single update definition is shared. The original Slang entrypoints
remain registered. The two former training binaries have identical executable
instructions after ID normalization, with only an `OpName` difference.

Release validation passed 49 shader/request tests (one unavailable-build-only
test skipped), 14 startup tests, 11 evidence-runner tests and the module audit.
The full shader suite was stopped at its large CaptureSymbols/O1 compilation
after exceeding the usual 300-second suite timeout; that path is not qualified
by this release-only performance measurement. Five Vulkan-validation GPU checks
passed: shadow history, async scene handoff, stream transmission, field views/
bounds and optional-resource sentinels. The raster/ray shading-normal and
ABeautifulGame FP16 checks failed with both the original and candidate shader
sources. Their three saved PNGs were byte-identical across the source A/B;
these pre-existing test failures were not reported as passes. The outputs were
also visually inspected. Full-scene temporal/memory behavior and an ordinary
paired GPU performance comparison remain outside this validation scope.

## Selection and shader debug modes

~~~powershell
# Build only the tool (does not warm any shaders).
cmake --build build-release --target MetallicShaderCompiler --config Release

# Inspect the request list without compiling.
.\build-release\Source\MetallicShaderCompiler.exe --list

# Warm only matching module/entry names.
.\build-release\Source\MetallicShaderCompiler.exe --filter FinalBlit

# Match the runtime shader debug policy when capturing/debugging shaders.
.\build-release\Source\MetallicShaderCompiler.exe --debug-mode capture
.\build-release\Source\MetallicShaderCompiler.exe --debug-mode debug
~~~

For multi-configuration generators the executable is under Source/<Config>/.
The default shader debug mode is disabled, independently of C++ Debug/Release.
capture corresponds to CaptureSymbols; debug corresponds to ShaderDebug.
Disabled and capture modes use Slang O1 with default backend optimization passes.
`METALLIC_SLANG_OPTIMIZATION=O0|O1|O2|O3` overrides that default for standalone
warmup and runtime requests. Unset it, or use `default`, to retain the existing
debug-mode policy. Set the same value before launching warmup and the renderer;
each effective optimization level has a separate shader-cache identity. Explicit
`ShaderRequest::optimizationLevel` values take precedence over the environment.
Without an optimization override, the explicit debug mode retains O0 for variable inspection. Runtime and warmup
share this policy and its versioned cache identity.
The tool also honors the runtime METALLIC_SLANG_DESCRIPTOR_MODE environment
variable. Cache entries for different modes are separate.
Native cache identity also includes the resource and sampler byte strides. The
compiler emits literal descriptor `ArrayStride` values instead of relying on
device-size specialization of the emitted SPIR-V. At runtime, device creation
publishes the selected adapter's actual strides before shader compilation.
This is the temporary `TO-REMOVE(VVL payload-size)` compiler policy; see
[the verified cause and removal conditions](NativeDescriptorHeapStrideWorkaround.md).

For offline native warmup, specify the two strides together. The resource stride
must match the runtime's common buffer/image descriptor slot stride; the sampler
stride must match its sampler descriptor stride. For example, if the target
adapter uses 64-byte resource slots and 16-byte sampler slots:

~~~powershell
$env:METALLIC_SLANG_DESCRIPTOR_MODE = "native"
.\build-release\Source\MetallicShaderCompiler.exe --resource-heap-stride 64 --sampler-heap-stride 16
~~~

Without these arguments, native warmup discovers a supported adapter automatically.
Failure to find a valid native stride pair aborts compilation. Explicit strides
produce cache entries for that ABI; another adapter can require different entries.

The stride query obtains Vulkan functions through its own local dispatch and
uses volk only for header declarations. Keep `MetallicShaderWarmupCore` free of
a volk implementation link dependency. Adding that dependency can move
`volk.lib` after `sl.interposer.lib` in application links on Windows: Streamline's
Vulkan function imports then replace volk's same-named global function pointers,
and `volkInitializeCustom` crashes while writing an import thunk in the code
section. This startup failure also affects mapped mode, even when every warmup
request is a cache hit and the native query never runs. Validate changes to this
dependency with a Streamline-enabled application startup; tests built with
Streamline disabled cannot cover the collision.

To pass arguments through the CMake target, configure with a semicolon-separated
list, for example:

~~~powershell
cmake -S . -B build-release "-DMETALLIC_SHADER_WARMUP_ARGS=--filter;FinalBlit"
cmake --build build-release --target MetallicShaderWarmup
# Clear the filter to restore the complete request list.
cmake -S . -B build-release "-DMETALLIC_SHADER_WARMUP_ARGS="
~~~

--cache-dir <path> redirects output for isolated validation. Such a directory
will not be used automatically by the application.

## Coverage and maintenance

Tools/ShaderWarmupRequests.h lists explicit runtime requests: editor display,
postprocessing, environment lighting, basic samples, GPU-driven/mesh/tessellation
rendering, ReSTIR, conventional-texture path tracing and guides, SHARC path
tracing variants, and the default global-view reference/realtime deferred paths.
The realtime deferred variants include upscaler guides and opaque/transmission
StreamAsset paths. RTXCR sample requests are added only when its SDK is configured.

This is a bounded warmup set, not every combination of runtime properties.
Optional NRD SDK permutations, local-view deferred variants, and other
unlisted combinations continue to compile on demand. No cache is required to run.

When adding or changing requests, copy the owning runtime pass's module, entry,
capabilities, additional search paths, and macro definitions **in the same order**.
The runtime cache key includes all of them, even explicit zero-valued defines.
For native requests it also includes both descriptor strides.
Compile include-file entry points through the same root module used by the pass.
The tool deliberately reuses runtime hashing, dependency validation, SPIR-V
processing, and cache serialization instead of duplicating that logic.
