# Faster Metallic builds

The normal CMake entry point keeps the existing feature defaults and source
dependencies. The presets require CMake 3.27+, Ninja, and (on Windows) an x64
Visual Studio Developer PowerShell/Command Prompt. Install Slang in
`External/slang`, or pass `-DSLANG_ROOT=<path>` when configuring the application.

The Vulkan backend requires `VK_KHR_device_address_commands` with
`deviceAddressCommands` and `bufferDeviceAddress` enabled, independently of
opacity micromap support. Unsupported devices fail initialization with
`Unsupported`. Indirect dispatch, indirect mesh draws, and buffer transfers use
device-address commands; buffers with these usages receive device addresses
without an explicit `ShaderDeviceAddress` request from the caller.

All build presets enable parallel builds by default ("jobs": 0), equivalent to
passing `-j` without a job count. Ninja chooses its default parallelism. To limit
concurrency for a particular build, pass an explicit count, for example
`cmake --build --preset metallic-dev -j 8`.

## Daily development

```powershell
cmake --preset metallic-dev
cmake --build --preset metallic-dev
```

This builds the editor with glTF, Streamline DLSS and NVIDIA NRC support when the
SDKs are available. OpenUSD/oneTBB, NRD denoising and NTC are disabled; tests are off.
USD files report an explicit unsupported-build error. The `metallic-ci` profile
disables Streamline and NRC and enables the scene, task and debug tests:

```powershell
cmake --preset metallic-ci
cmake --build --preset metallic-ci
ctest --preset metallic-ci
```

USD tests skip when USD is disabled; a separate test checks the disabled importer.
Closing optional integrations changes which render passes can run. Use the full
profile when working on those integrations.

## Runtime shader cache and warmup

Applications warm the shared `.cache/shaders/spirv` cache before initialization.
Use `--skip-shader-warmup` to compile requests on demand. The manual
`MetallicShaderWarmup` target stays outside default builds and application dependencies:

```powershell
cmake --build build-release --target MetallicShaderCompiler
.\build-release\Source\MetallicShaderCompiler.exe --list
.\build-release\Source\MetallicShaderCompiler.exe --debug-mode disabled --jobs 4
```

[ShaderRequests.h](../Source/Runtime/Render/Core/ShaderRequests.h) defines owning
requests and the factories used by runtime scene, ray-query and shadow passes.
[BuiltinShaderRequests.h](../Source/Runtime/Render/Core/BuiltinShaderRequests.h)
selects the production warmup variants; `Tools/ShaderWarmupRequests.h` only adapts
that catalog to the tool. Macro, capability and search-path order remain part of
cache identity. Scene requests include the runtime `METALLIC_CUSTOM_MATERIALS=0`
default so prewarmed OpenPBR and path-tracing binaries are reused.

The catalog deduplicates complete requests after generation. Conventional scene
variants cover resident/streamed materials, supported position-fetch settings,
Deferred global/local views, FP16/FP32, guides and material classes. Editor display
requests share a factory with runtime and include SDR/scRGB/HDR10 paths at the
standard 80/203-nit reference whites.

Installed Painter and WhiteStudio catalogs are also read before initialization.
Their scene sidecars and material assets are lowered through the runtime CPU
material compiler to generate the same content-keyed include, then their declared
graph shader requests are appended, including custom Surface/Closure programs.
This preparation loads no geometry or textures and requires no GPU. Optional
catalogs that do not exist add no requests. Changing generated material source
after warmup, a new external scene, SDK-specific NTC/NRD permutations or arbitrary
display reference whites can still require compilation on demand. Debug mode and
mapped/native descriptor mode remain separate cache identities.

Build and run `MetallicShaderRequestsTests` in an existing tests-enabled tree to
verify catalog coverage, warmup-to-runtime cache reuse, and compiled SPIR-V
equivalence for folded material classes. These checks do not require a GPU.

Scene path tracing also persists Vulkan PSOs in `.cache/pso/ScenePathTracePass.pso`.
Base/OpenPBR, SHaRC/NRC update/query and SHaRC clear/resolve/NRC tonemap use the
same pass cache. Realtime lighting and Deferred use separate
`RealtimeLightingPass.pso` and `VisibilityBufferDeferredPass.pso` files.
Shader bytes, pipeline state and device/backend compatibility still control
cache reuse; a new shader can require driver compilation even after SPIR-V warmup.
Cache `hits` in pass logs describe the application's PSO hash table, not Vulkan
creation feedback. The cross-process `MetallicLookDevPathTracePipelineCacheSmoke`
test disables the driver internal cache to verify application cache coverage.

## NVIDIA Neural Radiance Cache

`External/NRC` is a Git submodule of the official
[NVIDIA NRC SDK](https://github.com/NVIDIA-RTX/NRC), pinned by the parent repository.
Initialize it before configuring to fetch the headers, import libraries and runtime
DLLs together:

```powershell
git submodule update --init --recursive -- External/NRC
cmake --preset metallic-dev
cmake --build --preset metallic-dev
```

On Windows, CMake enables `METALLIC_HAS_NRC=1` when it finds the SDK headers,
`Lib/NRC_Vulkan.lib` and `Bin/NRC_Vulkan.dll`. The build copies the SDK runtime DLLs
next to the executable. Dev, Release, RelWithDebInfo and Full presets enable NRC
detection; CI disables it. Pass `-DMETALLIC_NRC_ROOT=<path>` to use another SDK
checkout, or `-DMETALLIC_ENABLE_NRC=OFF` to disable the integration.

## Release and optimized debugging

Native configure and build presets provide both optimized configurations:

| Preset | CMake build type | Build directory |
| --- | --- | --- |
| `metallic-release` | `Release` | `build-release` |
| `metallic-relwithdebinfo` | `RelWithDebInfo` | `build-relwithdebinfo` |

Both inherit the `metallic-dev` feature settings, including Streamline DLSS and NRC,
source dependencies and disabled tests. `RelWithDebInfo` enables optimization
and native debug symbols. Each configuration has its own CMake cache and output
directory.

All presets, including `metallic-relwithdebinfo`, leave Nsight capture injection
disabled unless explicitly requested with a launch mode or
`METALLIC_NSIGHT_GRAPHICS_CAPTURE=1`. The RelWithDebInfo preset makes the SDK
available and embeds shader source text and NonSemantic source/function/line
debug information (`-g2`) with the runtime cleanup passes (`capture-symbols` mode).
RelWithDebInfo also defaults to these symbols when internal capture injection is
disabled (`METALLIC_NSIGHT_GRAPHICS_CAPTURE=0`), for external Nsight launches.
Capture and GPU Trace launch modes do not change shader compiler options. Release
uses the same symbol-free shader compilation and cache requests with or
without Nsight capture. Set `METALLIC_SHADER_CAPTURE_SYMBOLS=1` explicitly when
source and function views are needed. `METALLIC_SHADER_CAPTURE_SYMBOLS=0` disables
the RelWithDebInfo symbol default, including during capture.

Runtime and manual warmup use direct SPIR-V generation with `-O0` and an explicit
lightweight `spirv-opt` pass sequence: dead-function elimination, local single-block
and single-store cleanup, instruction simplification, dead-branch elimination,
CFG cleanup, aggressive dead-code elimination and ID compaction. This avoids the
default preset's exhaustive entry-point inlining and its large debug-info cost.
Capture symbols retain embedded source and NonSemantic function/call-site records.
The optimization level, backend and pass sequence are included in cache identity;
old optimized cache entries are not reused. The Vulkan driver can still inline and
optimize during pipeline creation, so compare pipeline creation and GPU execution
separately from Slang compilation. Use
`--nsight-shader-debug` for shader debugging (`-g2 -O0`, without cleanup passes),
not representative runtime profiling. Capture-symbol profiling uses the runtime
cleanup policy; actual source/call-site correlation must be verified in Nsight.
The former `-g1` mode only emitted paths/lines and was insufficient for Nsight's
high-level source and function views. The cache request version has been bumped
so those old binaries are not reused. Restart the rebuilt executable and make a
new capture; existing captures cannot acquire the missing information retroactively.
See NVIDIA's [shader compilation requirements](https://docs.nvidia.com/nsight-graphics/UserGuide/configure-application.html#shader-compilation).
Export requires an installed Nsight Graphics SDK and runtime.

Select the activity before Vulkan initialization in the editor, LookDev or samples:

```powershell
.\build-relwithdebinfo\Source\MetallicGPUDrivenSample.exe --zorah-full --nsight-mode gputrace
.\build-relwithdebinfo\Source\MetallicGPUDrivenSample.exe --zorah-full --nsight-mode capture
```

`--nsight-gputrace` and `--nsight-capture` are aliases; `--nsight-mode=gputrace`
and `--nsight-mode=capture` also work. Missing/invalid values and conflicting
modes fail before graphics initialization. The Profiler export button displays
the selected activity and exports the next full main-viewport frame under
`Captures/NsightGraphics/`. GPU Trace produces only `.ngfx-gputrace`; Capture
produces `.ngfx-capture` for replay and frame debugging. Switching activities
requires restarting the process. RHI test captures keep their existing mode.

For a one-click collection, launch with `--nsight-capture` and press **Export
View Capture + GPU Trace** in the Profiler. Metallic first saves the
`.ngfx-capture`, then launches an owned, hidden `ngfx-replay` process through
`ngfx` to profile that capture. It selects single-pass metrics automatically;
no separate preset or `METALLIC_NSIGHT_GPU_TRACE_METRICS` setting is needed.
That environment override applies only to direct `--nsight-gputrace` sessions.
The matching `<capture-name>_Collected/` directory contains the GPU Trace,
metrics configuration and host log. The Profiler shows paths to both artifacts.
Keep `MetallicNsightReplay.exe` beside the editor executable when distributing
this feature. It starts before Graphics Capture injection and launches the
replay later, preventing inherited Capture injection from conflicting with
GPU Trace in the replay process. It is built automatically with SDK-enabled
Windows editor targets.

The editor drains its GPU work and pauses rendering while the replay is profiled,
then resumes. Replay collection is bounded to 180 seconds and its process tree
is cleaned up on completion or editor exit. A failed replay preserves the
original capture and reports an error without terminating the editor. The two
files remain separate Nsight artifacts; GPU Trace is not embedded into the
`.ngfx-capture`. These measurements describe the replayed workload on the current
GPU and driver, not the original application's CPU pacing or streaming behavior.
See NVIDIA's [Live Replay GPU Trace documentation](https://docs.nvidia.com/nsight-graphics/UserGuide/graphics-capture-ui.html#gpu-trace).

GPU Trace requires an attached host for metric preparation and file writing
([NVIDIA SDK guide](https://docs.nvidia.com/nsight-graphics/UserGuide/sdk.html)).
Metallic starts the installed `ngfx.exe` hidden, uses SDK start/stop boundaries,
and keeps the host connected for repeated exports. Host diagnostics are saved
as `GpuTraceHost-<pid>.log` in the output directory. The host is owned by this
process and is cleaned up at shutdown. GPU clocks remain unaltered. Neither
activity's live timings are a production performance baseline.

GPU Trace explicitly selects single-pass metrics for each supported architecture
(`Top-Level Triage`, or `Throughput Metrics` on Turing). The generated
`GpuTraceMetrics-<pid>.json` is passed to `ngfx` so saved host preferences cannot
leave the session without a metric set. `METALLIC_NSIGHT_GPU_TRACE_METRICS` can
point to an alternate ngfx per-architecture JSON configuration.

Launching with only `--nsight-gputrace` uses the **single-pass performance
overview** preset above. Startup logs identify that preset (or the custom JSON
path, when configured) and print detailed usage, including the Profiler export
button, output location and configuration override. The same usage is available
through `--help` without starting Vulkan. To customize the preset, copy a
generated `GpuTraceMetrics-<pid>.json`, edit the per-architecture entries, then run:

```powershell
$env:METALLIC_NSIGHT_GPU_TRACE_METRICS = 'E:/path/MyMetrics.json'
.\build-release\Source\LookDev.exe --nsight-gputrace
```

Unset `METALLIC_NSIGHT_GPU_TRACE_METRICS` to return to the default preset. An
invalid custom configuration reports an error instead of silently using defaults.

Before the application's first queue submission, Metallic logs a GPU Trace
startup handshake and activates the SDK on the rendering thread. Host failure
is reported in the application log, including the backend error when available.
If that native activation blocks, a startup watchdog exits the instrumented
process with code 1 after host failure or a 60-second timeout. It cannot safely
unwind an uncancellable call inside Nsight. Termination uses a private Windows
job because even `TerminateProcess` can block in the injected teardown; the
owned host job is also cleaned up by Windows. This check runs before scene
loading rather than waiting for an export request. Host liveness is also checked
before frame waits.

The opt-in `METALLIC_NSIGHT_INTEGRATION_TESTS` CMake option registers
`MetallicNsightGPUTrace.disabled`, `MetallicNsightGPUTrace.export` and
`MetallicNsightGPUTrace.invalid-metrics`, plus
`MetallicNsightGPUTrace.collection` for three complete capture/replay exports.
Build `LookDev` first. The regression can also run against an existing binary
without changing its build configuration:

```powershell
cmake -DEDITOR_EXECUTABLE=E:/metallic/build-release/Source/LookDev.exe -DTEST_DIRECTORY=E:/metallic/build/nsight-startup/export -DCASE=export -P tests/editor/RunNsightGPUTraceStartup.cmake
cmake -DEDITOR_EXECUTABLE=E:/metallic/build-release/Source/LookDev.exe -DTEST_DIRECTORY=E:/metallic/build/nsight-startup/invalid-metrics -DCASE=invalid-metrics -P tests/editor/RunNsightGPUTraceStartup.cmake
```

For the ZorahFull export memory-pressure fix and full-scene capture/replay
regression, see [the investigation](ZorahFullNsightCaptureMemory.md).
Graphics Capture injection changes live execution even before F11 is pressed.
For a production performance baseline, set both
`METALLIC_NSIGHT_GRAPHICS_CAPTURE=0` and `METALLIC_SHADER_CAPTURE_SYMBOLS=0`, and
do not pass `--nsight-mode`, `--nsight-gputrace`, `--nsight-capture` or `--nsight-shader-debug`.
Editor and sample executables retain Nsight's SDK-default demotion of
host-visible video memory to system memory during self-injected Graphics Capture.
An experimental CPU-hash mode reduced measured live overhead but its tested
ZorahFull capture failed actual replay with GPU device loss. The original SDK
path exported three captures and its first capture passed three-loop replay;
the HVVM experiment was removed from the runtime and benchmark launcher.
External `ngfx` launchers use their own capture settings. The remaining live
injection overhead is separate from ordinary renderer baseline performance. See the
[ZorahFull capture performance measurements](ZorahFullNsightPerformance20260930.md).

Texture streaming feedback accumulates in Device storage and is copied to
HostReadback after all RenderGraph consumers finish. RenderGraph stream sessions
also default immutable groups and LOD topology to Device storage, initialized
through bounded 64 MiB staging batches before traversal. These resource changes
retain the SDK-default capture policy. The runtime property
`deviceImmutableMetadata: false` selects the original HostUpload metadata path
for comparison. Raw `MeshletStreamRuntimeDesc` keeps that original default for
direct callers; opting into Device metadata requires pumping
`MeshletStreamInitialLoader` before frame recording, with `metadataOnly=true`
when root pages should remain lazy. See the
[implementation and validation](ZorahFullNsightFixImplementation20260930.md).

When Nsight Graphics capture injection is active, opacity micromaps use the
`VK_EXT_opacity_micromap` backend, including native EXT builds and shader support.
Ordinary execution uses `VK_KHR_opacity_micromap`. This avoids the current capture
interceptor's KHR OMM null-range crash while retaining OMM acceleration. See the
[investigation and workaround](NsightKhrOpacityMicromapInvestigation.md).
TODO(Nsight KHR OMM): Remove the temporary EXT path after Nsight Graphics supports
KHR OMM and the injected build/capture/replay checks in that document pass.
The RHI test runner can reproduce the capture path with
`--rhi-nsight-capture --gtest_filter='*opacity_micromap*'`.

The Nsight EXT path also supplies explicit UINT32 identity OMM indices to avoid
the replayer's null dereference when restoring implicit indices. Buffers are
retained by their OMM; ordinary KHR behavior is unchanged. Restart the rebuilt
application and create a fresh capture, since existing files keep their original
build parameters. See [replay evidence and removal TODO](NsightCaptureReplayInvestigation.md).
To export a warmed-up Sponza frame without a window, run
`MetallicRHITests --rhi-realtime --rhi-async-compute --rhi-aftermath --rhi-nsight-export --rhi-no-validation --gtest_filter='*gpu_driven_sponza_realtime_pipeline' --output-dir .tmp/nsight-replay`.
The runner prints the capture path under the output directory's `nsight` folder;
verify it with `ngfx-replay --present-hidden --loop-count 3 --no-block-on-incompatibility <capture>`.

Nsight Graphics injection also disables Aftermath automatic checkpoints by
default to avoid a separately reproduced GPU page fault during Sponza BLAS
compaction/TLAS preparation. Aftermath crash dumps, resource tracking and shader
debug information remain available; ordinary runs keep automatic checkpoints.
TODO(Nsight Aftermath): Restore checkpoints after a fixed capture runtime passes
the regression. `METALLIC_AFTERMATH_AUTOMATIC_CHECKPOINTS=1` forces them on for
that verification (`0` forces them off). The editor-equivalent headless check is
`MetallicRHITests --rhi-realtime --rhi-async-compute --rhi-aftermath --rhi-nsight-capture --rhi-no-validation --gtest_filter='*sponza_async_scene_rtas*:*gpu_driven_sponza_realtime_pipeline*'`.
See the investigation document for the before/after evidence and cleanup fix.

Nsight is opt-in in every build configuration. The removed CMake option
`METALLIC_DEFAULT_NSIGHT_CAPTURE` is ignored even when an old build cache still
contains `ON`; it cannot inject Nsight into an ordinary launch. The explicit
environment setting `METALLIC_NSIGHT_GRAPHICS_CAPTURE=1` enables capture and replay collection,
while `0` or an unset variable leaves injection disabled. An explicit mode
option takes precedence over the environment variable; `--nsight-capture`
selects Graphics Capture. Shader symbols remain independently available in
RelWithDebInfo.

```powershell
cmake --preset metallic-release
cmake --build --preset metallic-release

cmake --preset metallic-relwithdebinfo
cmake --build --preset metallic-relwithdebinfo
```

In CLion, reload the CMake project and enable the desired preset in CMake
settings. Select it as the active build profile; no manually derived Release
profile or `CMAKE_BUILD_TYPE` override is needed.

## Full development with reusable dependencies

Build and install SDL3, spdlog, oneTBB and monolithic OpenUSD once:

```powershell
cmake --preset metallic-deps-debug
cmake --build --preset metallic-deps-debug
cmake --preset metallic-full
cmake --build --preset metallic-full
```

The dependency build installs automatically; no separate `cmake --install` is
needed. It uses up to eight compiler jobs, with one dependency project active at
a time to bound peak memory. Override `METALLIC_DEPENDENCY_JOBS` during configure
if needed. The first build still compiles all four libraries.

Installed packages live in `.cache/dependencies/<sha256>`, outside `build-full`.
Deleting or cleaning the application build therefore keeps the dependency
binaries. A clean full build imports those four libraries with `find_package`;
their C/C++ sources do not enter its build graph. Smaller libraries, GoogleTest,
and NTC when enabled still build from source. NRC/Streamline retain their
existing SDK binary integration.

NRD now uses vendored shaders and Metallic-owned dispatch scheduling. Enabling
`METALLIC_ENABLE_NRD` does not build the NRD SDK or ShaderMake, fetch DXC, or compile
all shader permutations. Kernels compile on first use through the Slang disk
cache. This also allows `cmake --preset metallic-dev -DMETALLIC_ENABLE_NRD=ON`
without initializing the NRD submodule. See the
[NRD integration notes](../Shaders/Interop/Denoising/NRD/README.md) for ownership,
upgrades, and the `MetallicNRDTests` validation target.

The hash includes compiler identity/version/target, platform, Windows SDK,
CRT, configuration, compile/link flags, toolchain file, dependency Git revisions,
tracked and nonignored untracked changes, and the build recipes. A completion
manifest is written only after every dependency installs successfully. Re-running
the dependency configure/build with a matching completed package skips compilation.
Changing application code alone does not change this hash. Reconfigure both builds
after modifying a dependency checkout or toolchain. Builds for the same package
should run from one dependency build directory at a time.

`METALLIC_DEPENDENCY_CACHE=<absolute directory>` selects a shared cache, including
one outside this checkout. `METALLIC_DEPENDENCY_ROOT=<absolute package directory>`
selects a specific installed package; the consumer rejects mismatched manifests.
The application writes its expected inputs to
`<application-build>/metallic-dependencies-expected.txt` for comparison with the
package's `metallic-dependencies.txt` when diagnosing a mismatch. Equivalent
CMake booleans such as `ON` and `TRUE` produce the same key.
Use matching toolchain and configuration arguments for producer and consumer.
Prebuilt mode currently requires a single-config generator; Visual Studio's
multi-config generator continues to work in source mode.

Other configurations can use the standalone dependency project:

```powershell
cmake -S cmake/dependencies -B build-dependencies/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-dependencies/release
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DMETALLIC_DEPENDENCY_MODE=PREBUILT
cmake --build build-release --target Metallic --parallel 8
```

Both sides must use `-DMETALLIC_ENABLE_OPENUSD=OFF` for a package containing only
SDL3 and spdlog. The heavy submodules need initialization only in the dependency
producer/source profile; prebuilt consumers can identify uninitialized submodules
from the repository's Git locks. Small source/header dependencies still need their
usual checkouts (including SDL's vendored Vulkan headers used by volk).

To keep every dependency in the application's source build:

```powershell
cmake -S . -B build-source -DMETALLIC_DEPENDENCY_MODE=SOURCE
cmake --build build-source --target Metallic --config Debug --parallel 8
```

## Compiler cache

Install `sccache` on PATH, then add `-DMETALLIC_USE_SCCACHE=ON` to both application
and dependency configure commands. This is opt-in and supported with Ninja or
Makefiles. With MSVC, the launcher remains
`RunMsvcCompiler.cmake -> sccache -> cl.exe`, preserving localized `/showIncludes`
normalization. Debug information uses `/Z7` to avoid shared compile PDB contention.
An existing custom CMake launcher is preserved when sccache is off; conflicting
custom launchers are rejected when sccache is explicitly enabled.

Use `sccache --show-stats` to check actual hits. MSVC module compilation may bypass
the cache; dependency packages are the primary way to avoid those libraries being
rebuilt. See the [sccache documentation](https://github.com/mozilla/sccache) and
[CMake debug information settings](https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_DEBUG_INFORMATION_FORMAT.html).

Unity builds and PCH are not enabled globally: several vendors and module targets
require separate validation. CI artifact upload/download is also left to the
repository's eventual CI provider; the local package and manifest provide the
reusable output without introducing a remote service.

## Validation

Validated on Windows x64 with MSVC 19.51.36256 and CMake 4.2.1, Debug:

- Building the four dependency libraries into a fresh package took about 418 s
  with eight compiler jobs. Reconfiguring and building the completed package
  took about 1.8 s with no compilation. These timings cover the dependency stage.
- The full application compile database contains no source compilations for
  SDL3, spdlog, oneTBB or OpenUSD.
- `metallic-full` built the editor and scene/task/debug tests. Those tests passed,
  including USD, USDC and embedded-texture USDZ imports. The optional Super Sponza
  fixture remained environment-gated.
- The lightweight scene/task/debug tests, nine cache identity/rejection tests,
  and editor Vulkan smoke checks in both profiles passed.
- An MSVC build with a custom launcher preserved the launcher and recorded header
  dependencies in Ninja; changing the header scheduled recompilation. Actual
  sccache cache hits were not measured because sccache was not installed.

## Optional RHI diagnostics

`METALLIC_RHI_DIAGNOSTICS` defaults to `OFF`. Enable it in a compatible existing
build tree to compile the same Vulkan barrier/submit observation hooks into the
shared render runtime, editor, and tests. It does not change public RHI object
layouts or enable capture by itself. `MetallicRHITests --tb-trace` installs the
bounded recorder; requesting it in an OFF build fails explicitly. See
[RHI Testbench](RhiTestbench.md#m4-诊断与合法序列) for trace A/B, sequence replay,
and shrinking commands. The property and shrinker tests do not require this flag.
