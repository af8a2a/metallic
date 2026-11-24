# Faster Metallic builds

The normal CMake entry point keeps the existing feature defaults and source
dependencies. The presets require CMake 3.27+, Ninja, and (on Windows) an x64
Visual Studio Developer PowerShell/Command Prompt. During configuration, CMake
automatically installs the missing Slang 2026.18.2 SDK into `External/slang` on
native Windows x64 using PowerShell and the version- and SHA256-checked installer.
The archive is cached in `.cache/slang` and reused by subsequent installations.
Existing installations are not overwritten; `-DSLANG_ROOT=<path>` selects a
user-managed Slang 2026.18.2 SDK and bypasses automatic installation.

CMake also initializes missing required Git submodules (including OpenPBR LUTs,
ImGui, SDL3 and the enabled source dependencies) at the repository's locked
commits. Existing checkouts are preserved. Git and network access are required
for the first download. Optional SDKs retain their own setup policies; licensed
RTXCR reference assets are not downloaded by this bootstrap.

NVIDIA SDK integrations are optional. Missing Streamline, Aftermath, NVTX,
Nsight Graphics, NvPerf, NTC or RTXCR installations disable the corresponding
integration with a configure status message, without blocking ordinary builds.
To disable all of these integrations (including NRD and experimental DLSS-NR)
regardless of installed SDKs, configure with `-DMETALLIC_ENABLE_NVIDIA_SDKS=OFF`.
Individual `METALLIC_ENABLE_*` cache choices are preserved when this master
switch is off; turning it back on restores them. RTXCR discovery can also be
disabled independently with `-DMETALLIC_ENABLE_RTXCR=OFF`.
Bundled shader sources such as SHaRC, RTXDI and RTXTF remain part of the renderer;
this switch controls optional SDK integrations, not Vulkan device requirements.

For pre-provisioned/offline builds, set `METALLIC_SLANG_AUTO_DOWNLOAD=OFF` and
`METALLIC_SUBMODULE_AUTO_DOWNLOAD=OFF`. Streamline has a separate
`METALLIC_STREAMLINE_AUTO_DOWNLOAD` option. Missing required dependencies then
produce an installation hint. Manual setup remains available:

```powershell
.\scripts\InstallSlang.ps1
git submodule update --init --recursive -- External/openpbr-bsdf
```

### Locked NRD and microprofile revisions

Submodule versions are pinned by the parent repository's gitlinks, not by the
current upstream default branch. Keep these locks when initializing dependencies;
do not use `git submodule update --remote` to reproduce a build.

| Submodule | Upstream | Locked commit |
| --- | --- | --- |
| `External/nrd` | [NVIDIA-RTX/NRD](https://github.com/NVIDIA-RTX/NRD.git) | `36183520b006b65860bebc96d3dd7185e5edeb45` (NRD 4.16) |
| `External/microprofile` | [jonasmr/microprofile](https://github.com/jonasmr/microprofile.git) | `a469709c55b973099c1600b78a503912150e2c69` |

The NRD lock matches the vendored shader snapshot described in the
[NRD integration notes](../Shaders/Interop/Denoising/NRD/README.md).
`External/nrd` is upstream reference material for snapshot updates, not a build
input for the Metallic-owned NRD implementation. Its mapping is still required
for repository-wide recursive submodule operations, even with NRD disabled.
Microprofile is a required source dependency, including its nested submodules.

For a fresh checkout, initialize all locked dependencies and inspect their state:

```powershell
git clone https://github.com/af8a2a/metallic.git metallic-clean
Set-Location metallic-clean
git submodule update --init --recursive
git submodule status --recursive
git status --short
```

A clean initialized checkout has no `-` (uninitialized), `+` (different commit), or
`U` (conflict) prefixes in submodule status, and no submodule changes in Git status.
The full recursive initialization also downloads optional source/reference
repositories, including OpenUSD and RTXCR assets; it can be substantially larger
than initializing only the dependencies required by a selected build profile.
For an existing checkout, inspect and preserve local submodule changes before
initializing. After pulling a changed mapping, synchronize the affected URLs and
initialize the locked revisions as needed:

```powershell
git submodule sync --recursive -- External/nrd External/microprofile
git submodule update --init --recursive -- External/nrd External/microprofile
```

Do not reset, clean, or change gitlinks to hide existing local modifications or
untracked files. A dirty nested checkout can coexist with a parent submodule HEAD
that already matches its gitlink; check both before diagnosing version drift.

The Vulkan backend requires `VK_KHR_device_address_commands` with
`deviceAddressCommands` and `bufferDeviceAddress` enabled. Unsupported devices fail initialization with
`Unsupported`. Indirect dispatch, indirect mesh draws, and buffer transfers use
device-address commands; buffers with these usages receive device addresses
without an explicit `ShaderDeviceAddress` request from the caller.

All build presets enable parallel builds by default ("jobs": 0), equivalent to
passing `-j` without a job count. Ninja chooses its default parallelism. To limit
concurrency for a particular build, pass an explicit count, for example
`cmake --build --preset metallic-dev -j 8`.

MSVC/Ninja builds track the compiler binary in the configure dependencies and
compile command fingerprint. Updating Visual Studio in place therefore rebuilds
objects and precompiled headers, avoiding reuse of incompatible PCH files (C1853).

## Daily development

```powershell
cmake --preset metallic-dev
cmake --build --preset metallic-dev
```

This builds the editor with glTF, Streamline DLSS when the SDK is available, and
the bundled SHaRC radiance cache. OpenUSD/oneTBB, NRD denoising and NTC are disabled; tests are off.
USD files report an explicit unsupported-build error. The `metallic-ci` profile
disables Streamline, requires Python, and selects the `scene`, `task`, `debug`,
`build`, `tools` and `skip-policy` test labels:

```powershell
cmake --preset metallic-ci
cmake --build --preset metallic-ci
ctest --preset metallic-ci
```

CTest does not build test executables. The CI build preset includes
`MetallicSceneTests`, `MetallicTaskTests`, `MetallicDebugTests`, `MetallicCtl`,
`MetallicRHITests` and `MetallicShaderPipelineBench` for the selected jobs. The
RHI executable is needed by the CPU-only filter-selection regression; selecting
`tools` does not select the ordinary `rhi`, `editor` or `testbench` lanes. The
mixed CPU/GPU `MetallicGPUPageTests` bundle also carries `scene`, so the test
preset explicitly excludes it by name; its CPU subtests are not selected here
either. Run that bundle separately on a suitable GPU. The pipeline benchmark
regression checks its CLI and compile-only fixtures, not GPU performance; its
four opt-in GPU cases remain skipped unless explicitly run with `--gpu`.
The source-only `MetallicShaderModuleOwnershipAudit` runs
[the Slang ownership checker](../Tools/CheckShaderModules.py) without a GPU or
Slang compiler; its fixture regression `MetallicShaderModuleOwnershipTests` is
also selected through `tools`. Their existing `shader;architecture` labels are
retained. `MetallicCIPresetTests` requires both jobs, the compile-only benchmark
and `MetallicCPUWorkflowTests` to remain registered and selected, and checks the
preset policy and selected executable references in the configured CTest command
model (direct commands, CMake `-D...EXECUTABLE` arguments and Python `--executable`
arguments).

The [Windows CPU workflow](../.github/workflows/CPU.yml) defines a hosted candidate
on `windows-2025` for pushes, pull requests and manual dispatch. GitHub requires
the workflow on the default branch before manual dispatch is available; initial
feature-branch validation uses its push trigger. It uses Python 3.12 and
[the lane script](../scripts/RunCPULane.ps1), which discovers the x64 Visual Studio
environment and resolves one Python interpreter for package installation and
CMake. It installs the four directly pinned [CPU Python dependencies](../requirements-cpu.txt)
as wheels, checks package consistency, logs the resolved package list and actual
compiler/SDK/tool versions, and runs the configure/build/test presets. The direct
pins are not a complete transitive or hash lock; Pillow is provided by matplotlib. Unlike the base preset, this hosted variant
explicitly disables the NVIDIA SDK umbrella, SPIRV-Tools and ISPC. Its build
parallelism is 2; CMake bootstraps required submodules and the version/hash-pinned
Slang SDK. It does not require a separately installed Vulkan SDK. A native
failure stops subsequent lane steps; direct `pwsh -File` preserves its code,
whereas GitHub's outer `pwsh -Command` wrapper can report 1 instead. Logs under
`build-ci/ci-logs/`, CTest JUnit and available CTest diagnostic logs are uploaded
even after failure. `MetallicCPUWorkflowTests` checks the authored workflow/script
source contracts and negative mutations, not arbitrary YAML/PowerShell semantics.
It also runs before configuration in the workflow. The [first hosted run and
bounded repair](Issues/CPUHostedRegression-2026-11-14.md) records a successful
SDK-off configure/build but **46/61 passing tests**, not hosted acceptance. See
also the [historical local workflow evidence](Issues/CPUWorkflowFollowup-2026-11-14.md);
local regression or workflow source presence cannot replace a green hosted run.

This CPU lane does not select every `perf`/shader test or establish GPU, editor,
Nsight, or non-Windows cache validation. USD tests skip when USD is disabled; a
separate test checks the disabled importer. Keep skipped and failed cases separate
from passes; CTest JUnit counts outer jobs and does not erase inner native skips.
See the [required-run and skip-baseline policies](RhiTestbench.md#普通-rhi--编辑器-required-run).
Closing optional integrations changes which render passes can run. Use the full
profile when working on those integrations.

For the Sponza virtual shadow map showcase, build `MetallicVSMSample` explicitly
in the configured tree. Camera, lighting, controls and validation commands are
documented in [Sponza VSM Sample](VSMSponzaSample.md).

## Runtime shader cache and warmup

Applications compile the active graph/UI requests on demand by default, using
the shared `.cache/shaders/spirv` and native pipeline caches. The first frame
does not wait for unrelated samples or full-catalog permutations. Runtime pass
factories own the exact source request and native specialization, including
scene materials, SHARC, guides and device ABI. Use `--full-shader-warmup` to
explicitly warm the complete source/compute PSO catalog before initialization;
`--skip-shader-warmup` selects the default mode (the last flag wins). The manual
`MetallicShaderWarmup` target stays outside default builds and application dependencies.

Full-catalog startup reports live `[Warmup]` phases (source capture, compilation, validation,
and cache publication), completed request counts, and a heartbeat every two
seconds while the batch is running. `ready` requests are provisional: startup
accepts them only after the entire source batch passes validation and cache
read-back. Application startup prints phase progress and a final summary; the
standalone `MetallicShaderCompiler` retains per-request progress and timings for
diagnostics and performance tooling.

The device-level native warmup obtains its sources through one complete source
batch using the actual device's descriptor heap strides before creating native
pipelines. It retains the existing producer, source namespace/content, policy,
and persisted-cache read-back validation boundaries. This does not retain an
unvalidated process-wide SPIR-V cache or change ordinary runtime acquisition and
hot-reload invalidation. Explicit artifact overrides retain their ordinary
acquisition semantics using one override-owner snapshot for the source batch.
Warmup requests must declare all dependency roots in their search paths,
including parent directories used by relative includes. The batch rejects
dependencies outside its captured roots.

Source bytes and complete epoch validation are shared, while read/miss/cache-hit
provenance is isolated per root-module session. Session reuse stays within the
same root and effective compiler policy, including across its entry points.
Each persisted request carries that scoped proof; batch publication still
requires complete source/producer validation and actual file read-back before
acceptance. Ordinary requests use independent provenance views as well.
Directory namespace proofs remain conservative: unrelated file additions or
removals can still invalidate entries. A compiler rebuild changes the producer
identity, so the first run after this repair can recompile old cache entries.

For manual warmup:

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

`PathTraceShaderPolicy.h` owns the unsplit compute path tracer's and visibility-buffer
Deferred's final transport and trusted-resource macros together with their native
root layout. Runtime and warmup use this policy after the scene request factory selects specialization.
The catalog includes the default inline root (24-byte `ComputeResourceParameters`)
and the explicit legacy descriptor root (12-byte `GPUBufferSpan`). Deferred uses
checked scene access with either root; its builtin and custom-material catalogs
cover both transports. Regression
validation must include the DLSS RR sample with single-sample guides and SHARC;
matching entry-point names alone does not establish matching shader requests.


The catalog deduplicates complete requests after generation. Conventional scene
variants use streamed geometry and CLAS ray queries,
Deferred global/local views, FP16/FP32, guides and material classes. Editor display
requests share a factory with runtime and include SDR/scRGB/HDR10 paths at the
standard 80/203-nit reference whites.

Explicit full warmup also reads installed Painter and WhiteStudio catalogs.
Their scene sidecars and material assets are lowered through the runtime CPU
material compiler to generate the same content-keyed include, then their declared
graph shader requests are appended, including custom Surface/Closure programs.
This preparation loads no geometry or textures and requires no GPU. Optional
catalogs that do not exist add no requests. Changing generated material source
after warmup, a new external scene, SDK-specific NRD permutations or arbitrary
display reference whites can still require compilation on demand. Debug mode and
mapped/native descriptor mode remain separate cache identities.

Build and run `MetallicShaderRequestsTests` in an existing tests-enabled tree to
verify catalog coverage, warmup-to-runtime cache reuse, and compiled SPIR-V
equivalence for folded material classes. These checks do not require a GPU.

Scene path tracing also persists Vulkan PSOs in `.cache/pso/ScenePathTracePass.pso`.
Base/OpenPBR, SHaRC update/query and SHaRC clear/resolve use the
same device-owned [ShaderRegistry](ShaderRegistry.md) cache. Realtime lighting and Deferred use separate
`RealtimeLightingPass.pso` and `VisibilityBufferDeferredPass.pso` files.
Shader bytes, pipeline state and device/backend compatibility still control
cache reuse; a new shader can require driver compilation even after SPIR-V warmup.
All runtime shader and pipeline acquisition uses ShaderRegistry; default pipeline
requests automatically load/persist PSOs without pass-owned cache plumbing.
Linked vertex/fragment Shader Objects also persist driver binaries in
`.cache/shader-objects/`; compatible entries are loaded with Vulkan's BINARY
creation path, and invalid/rejected entries rebuild from SPIR-V automatically.
Cache `hits` in Registry logs describe the application's PSO hash table, not Vulkan
creation feedback. The cross-process `MetallicLookDevPathTracePipelineCacheSmoke`
test disables the driver internal cache to verify application cache coverage.
`MetallicShaderRegistryUsageAudit` rejects new acquisition paths bypassing the registry.

## SHaRC radiance cache

SHaRC is the supported path-tracing radiance cache. Its headers and license are
bundled in `Shaders/ThirdParty/RadianceCache/Sharc`; no separate SDK checkout or
runtime DLL is required. Select `cacheMode: "sharc"` in a `ScenePathTracePass`
graph node, or run the `pathtracing-sharc-meet-mat` sample. `cacheMode: "off"`
keeps uncached path tracing available.

OpenPBR also supports SHaRC with `exportDenoiserGuides: true`, including the
DLSS Ray Reconstruction F preset. Enable `cacheMode: "sharc"` on `PathTrace`;
the RR mode and guide connections stay the same. Runtime controls include:

| Property | Default | Meaning |
| --- | --- | --- |
| `sharc.updateStride` | 5 | Trace one cache-training path per stride × stride pixel block per frame. |
| `sharc.updateMaxDepth` | 12 | OpenPBR training-path depth, independent of the image path's `maxDepth`. |
| `sharc.queryMinDepth` | 1 | Earliest cache query; primary surfaces are depth 0 and always shaded. |
| `sharc.minRoughness` | 0.25 | Minimum roughness for directional specular caching (hard floor 0.2); diffuse caching is independent. |
| `sharc.lobeMask` | 3 | 1 = diffuse only, 2 = directional specular only, 3 = both (OpenPBR). |
| `sharc.entriesLog2` | 22 | Total entry budget, divided equally between the two OpenPBR caches. |
| `sharc.sceneScale` | 0 | Hash-grid scale; 0 derives it from scene bounds. |

The image path terminates when a mature cache entry can replace its indirect
tail. A cache miss continues the reference integrator up to `maxDepth`; lowering
that limit also truncates cold-cache and unsupported-material paths. Training
uses independent frame-varying samples even when `accumulate` is false, as in
RR graphs. The cache stores scene-linear radiance before exposure and primary
aerial perspective. Camera motion retains history; scene/lighting changes,
grid scale, component selection, training depth and material-debug policy changes invalidate it.

OpenPBR stores the native BSDF's diffuse and specular outgoing radiance in
independent hash-table partitions. Diffuse keys include world position, material,
shading normal and quantized material parameters. Specular keys additionally
include an octahedral outgoing-direction bin; angular cells become smaller as
roughness decreases. Metal multiple scattering stays in the specular component,
as defined by OpenPBR. The allocation remains 40 bytes per total entry: the
default 2^22 entries consume 160 MiB, split into two 80 MiB partitions.

Direct-light evaluation and BSDF sampling retain OpenPBR's two RGB weights and
the complete mixture PDF. A partial hit replaces only that component's direct
and indirect radiance; the missing component continues tracing with its original
mixture weight. Both caches must be available (or a component physically absent)
to terminate the whole path. Training registers two vertices and propagates the
complete suffix to both; a full-cache bootstrap may shorten that suffix, but a
miss or full tracking array never silently drops later contributions. Cached
samples are normalized once per registered component, including sky misses.

This is a spatial/angular approximation of outgoing radiance, not a universal
irradiance field: diffuse view dependence and finite voxel/direction bins remain
approximations. Metal and mixed diffuse/specular materials are supported; very
sharp specular lobes, transmission, media, subsurface, coat, fuzz, thin film,
anisotropy and emission retain reference transport. Delta incoming rays, close
contacts and footprints smaller than a voxel also continue tracing. At least
two diffuse or eight specular samples and two frame ages are required for reuse.
Primary RR guides and first specular hit distance still use actual geometry.

OpenPBR uses a separate resolve permutation with adjacent-level blending
disabled: its opaque material/direction keys cannot be decoded as SHaRC spatial
keys. Camera motion retains entries and naturally misses when grid levels or
angular cells change; stale entries expire normally. The standard BSDF keeps
the original SHaRC resolve. Neither path adds screen-space history reuse.

Build `MetallicRHITests` before running the focused checks:

```powershell
.\build-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.openpbr_sharc_guides_and_history:RHIRendering.openpbr_sharc_mixed_components:RHIRendering.openpbr_sharc_directional_metal --rhi-validation
.\build-release\tests\MetallicRHITests.exe --gtest_filter=RHIRendering.openpbr_sharc_dlss_rr_ab --rhi-streamline --rhi-no-validation
```

The room tests compare Off, NoQuery, both components, diffuse only and specular
only on rough dielectric, mixed and metal materials. They check exact miss/guide
equivalence, energy and image error, live camera motion and history invalidation.
The BSDF closure probe independently checks both component weights against native
OpenPBR evaluation and sampling. The RR test records matched 1080p DLSS Quality F
far/near AB/BA runs, with 32 warmup and 32 measured frames per case. Its PT
timings include cache update and resolve; they are not end-to-end frame times.
Generated JSON and images remain in the test output directory.

The component-separated room checks on RTX 5060 / Release with Vulkan validation
enabled used 96x96, 4 spp, depth 8, stride 2, 2^16 total entries, 32 warmup and
64 averaged frames. Both components enabled gave the following HDR errors versus
uncached tracing (these include Monte Carlo noise):

| Room material | Relative RMSE | Cached/reference energy | Moving-camera relative RMSE |
| --- | ---: | ---: | ---: |
| Rough dielectric | 4.01% | 0.9957 | 10.71% |
| Mixed diffuse/specular | 3.23% | 0.9938 | 9.42% |
| Metal | 2.52% | 0.9981 | 7.99% |

All three also passed separate diffuse-only and specular-only comparisons,
exact cache-miss and seven-guide equivalence, and scale/component/light history
invalidation. Pure metal with diffuse-only caching matched the reference exactly.
The moving check retained warmed caches over 32 frames while translating the
camera laterally and toward the scene. This small fixture does not establish
full-scene speedups or artifact-free motion for arbitrary materials.

On 2026-10-10, RTX 5060 / NVIDIA 616.92 / Release, the component-separated
implementation on the chess sample at 1080p output, DLSS Quality F, 2 spp and
depth 12 measured the following pooled medians. Each mode used two runs in
AB/BA order, 32 warmup and 32 measured frames per run, warm shaders/PSOs/cache,
validation disabled and the editor closed. PT includes update and resolve.

| View | Cache off PT | Split-cache PT | PT time change |
| --- | ---: | ---: | ---: |
| Far | 7.75 ms | 8.32 ms | +7.3% |
| Near | 41.80 ms | 42.02 ms | +0.5% |

RR remained approximately 3.5-3.6 ms. The separate caches expand material
coverage and pass correctness checks, but do not yet accelerate this workload.
The production RR sample therefore still defaults to cache off; these results
do not justify reducing reference fallback depth. Directional bins need more
samples to become reusable, and lookup/training/resolve costs remain even when
most tails must continue. Quantifying hit rates and the remaining material
execution cost is necessary before claiming a speedup.

The RR A/B body completed and saved static and moving-camera images and timing
JSON. This run again stalled at global teardown, consistent with the previously
isolated Streamline shutdown issue; its process was terminated after preserving
evidence. It is not a clean process-exit pass. The prior cache-off
`working_color_dlss_rr_history` test reproduced the same shutdown stall. The
benchmark presents through a hidden swapchain each frame for SDK bookkeeping.
Six GPU checks (three room fixtures, standard SHaRC lighting and history, and
the OpenPBR component probe), one shader-variant compile check and nine
shader-request checks completed normally. Generated evidence stays outside source control in
`.cache/OpenPBRSplitAcceptance`, `.cache/OpenPBRSplitClosure` and
`.cache/OpenPBRSplitRRPerf`.

## Release and optimized debugging

Native configure and build presets provide both optimized configurations:

| Preset | CMake build type | Build directory |
| --- | --- | --- |
| `metallic-release` | `Release` | `build-release` |
| `metallic-relwithdebinfo` | `RelWithDebInfo` | `build-relwithdebinfo` |

Both inherit the `metallic-dev` feature settings, including Streamline DLSS and SHaRC,
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

Runtime and manual warmup use direct SPIR-V generation with Slang's default
`-O1` optimization policy. The backend uses its default optimization passes;
Metallic does not override them with a custom `spirv-opt` sequence.
Capture symbols retain embedded source and NonSemantic function/call-site records.
The optimization level, backend and pass sequence are included in cache identity;
old O0/custom-pass cache entries are not reused. The Vulkan driver can still inline and
optimize during pipeline creation, so compare pipeline creation and GPU execution
separately from Slang compilation. Use
`--nsight-shader-debug` for explicit shader debugging (`-g2 -O0`),
not representative runtime profiling. Capture-symbol profiling uses the runtime
O1 policy; actual source/call-site correlation must be verified in Nsight.
For controlled compiler experiments, `SlangShaderDesc::optimizationLevel` and
the owning `ShaderRequest::optimizationLevel` accept `Default`, `O0`, `O1`, `O2`
and `O3`. `Default` uses `METALLIC_SLANG_OPTIMIZATION=O0|O1|O2|O3` when set;
an unset, empty or `default` value retains O1 for release/capture and O0 for shader
debugging. Explicit request levels override both the environment and the debug
mode's optimization default, allowing the same debug information to be held fixed.
Set the environment before launching each benchmark or renderer process. Invalid
values fail shader compilation with `InvalidArgument`. Slang's O2 is aggressive;
O3 includes optimizations that can take much longer or trade code size for speed.
Cache identity uses the effective optimization level and the debug mode, so an
explicit O1 and the existing release default share the same cache request while
O0/O2/O3 variants remain separate. Compile and cache-hit logs record `optimization=O0`
(or the selected level). These options are experimental settings; compare fixed
scene GPU dispatch times and output correctness before changing production policy.
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
`Captures/NsightGraphics/` by default. GPU Trace produces only `.ngfx-gputrace`; Capture
produces `.ngfx-capture` for replay and frame debugging. Switching activities
requires restarting the process. RHI test captures keep their existing mode.

Set `METALLIC_NSIGHT_OUTPUT_DIRECTORY` to choose a writable directory for metrics,
host logs, captures and replay collections. Relative paths resolve against the
process working directory. For example, run from the repository root:

```powershell
$env:METALLIC_NSIGHT_OUTPUT_DIRECTORY = '.cache/NsightGraphics'
.\build-relwithdebinfo\Source\MetallicVSMSample.exe --nsight-gputrace
```

Remove the override to restore the default project capture directory:

```powershell
Remove-Item Env:METALLIC_NSIGHT_OUTPUT_DIRECTORY -ErrorAction SilentlyContinue
```

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
Metallic injects the SDK before Vulkan initialization, then starts the installed
`ngfx.exe` hidden after device creation and native PSO warmup, immediately before
the startup activation handshake. This prevents a cold pipeline cache from
exhausting the host's target-ready timeout. It uses SDK start/stop boundaries
and keeps the host connected for repeated exports. Host diagnostics are saved
as `GpuTraceHost-<pid>.log` in the output directory. The host is owned by this
process and is cleaned up at shutdown. GPU clocks remain unaltered. Neither
activity's live timings are a production performance baseline.

For Nsight 2026.3.1 GPU Trace sessions, editor shutdown retains an already-loaded
`D3D12.dll` until process exit: the NVIDIA Vulkan driver otherwise unloads it
before Nsight's exit callback releases its remaining D3D12 objects. This does
not load D3D12 when absent or affect ordinary startup. See the
[startup and exit investigation](NsightGPUTraceStartupInvestigation.md).

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
without changing its build configuration. Each run sets
`METALLIC_NSIGHT_OUTPUT_DIRECTORY` to `<TEST_DIRECTORY>/Captures` and verifies
that the reported export files exist there and are nonempty:

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

Ray tracing uses streamed CLAS geometry exclusively. Traditional triangle BLAS,
position-fetch and opacity micromap paths, including the former Nsight KHR/EXT
OMM workaround, have been removed. An explicit `enableRayQuery` device request
also requires CLAS; unsupported hardware returns `Unsupported`. Raster-only
devices do not acquire this requirement. TLAS and PTLAS remain supported over
CLAS-built bottom levels; production scene streaming currently uses standard TLAS.

To export a warmed-up Sponza frame without a window, run
`MetallicRHITests --rhi-realtime --rhi-async-compute --rhi-aftermath --rhi-nsight-export --rhi-no-validation --gtest_filter='*gpu_driven_sponza_realtime_pipeline' --output-dir .tmp/nsight-replay`.
The runner prints the capture path under the output directory's `nsight` folder;
verify it with `ngfx-replay --present-hidden --loop-count 3 --no-block-on-incompatibility <capture>`.

When Aftermath is enabled, ordinary runs capture GPU crash dumps with resource
tracking and automatic checkpoints. Driver shader debug information is opt-in.
Set `METALLIC_AFTERMATH_SHADER_DEBUG_INFO=1` before launch when you need to map
GPU instructions to SPIR-V or source lines. On affected NVIDIA drivers, the
default avoids `ucode.elf` writes during pipeline creation.

Nsight Graphics injection disables Aftermath automatic checkpoints by
default to avoid a separately reproduced GPU page fault during Sponza BLAS
compaction/TLAS preparation.
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
and the explicitly requested reserved NTC target still build from source. Streamline retains its existing
SDK binary integration.

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

## Offline mesh distance field baker

`MetallicMeshDistanceFieldBake` is available when `METALLIC_EMBREE_ROOT` points to an Embree 4.3.3 SDK. The baker accepts a glTF/GLB source mesh index and writes a `.meshdf` with three sparse mips. Supply an outward-wound, closed mesh made of single-sided opaque sections. On Windows, keep Embree's matching TBB runtime on `PATH`. For the local Unreal 5.7.4 SDK this is included in `Engine/Binaries/Win64`.

From an x64 Visual Studio developer shell at the repository root, build the optional tool and bake the sample:

```powershell
$unrealEngine = 'F:/UnrealEngine/Engine'
cmake -S . -B build-release "-DMETALLIC_EMBREE_ROOT=$unrealEngine/Source/ThirdParty/Intel/Embree/Deploy/embree-4.3.3"
cmake --build build-release --target MetallicMeshDistanceFieldBake
$env:Path = "$unrealEngine/Binaries/Win64;$env:Path"
New-Item -ItemType Directory -Force .cache/mesh-distance-field | Out-Null
.\build-release\Source\MetallicMeshDistanceFieldBake.exe Asset/MeshDistanceField/BoxInstances.gltf 0 .cache/mesh-distance-field/BoxInstances.meshdf
```

Open **Lighting / Mesh Distance Fields** in the editor. The sample traces two instances of the same mesh, including a rotated and mirrored instance. The pass can display normals, trace steps, or instance IDs. Its `trace` texture stores hit distance in meters, step count, debug instance ID, and a hit flag. The `MeshDistanceField` checkpoint exposes the R8 atlas for slice capture. The supported instance transforms are rigid or uniformly scaled, including reflections.

### Global Distance Field

**Lighting / Global Distance Field** uses the same baked asset. The compose pass accepts `distanceFieldPaths`, a list of baked assets for source meshes in the bound world scene. Its `center` and `extent` define a fixed world-space cube in meters. `extent` is the half-width. A logical `resolution` of `N` produces an R32F volume with `N + 1` samples per axis to include both endpoints of the Unreal page lattice. Inspect slices of `GlobalDistanceField.distanceField` in the resource inspector.

The volume is rebuilt from the current scene instances each frame. It stores the minimum mesh distance clipped to plus or minus four voxel widths. The debug pass shows normals, step count, or coverage. Its RGBA32F `trace` contains hit distance in meters, the number of samples, coverage state, and surface expansion in meters. Coverage states are 0 for a ray segment outside the sampled domain, 1 for a covered segment without a hit, and 2 for a hit.

**Lighting / Global Distance Field / Sparse Pages** uses the same world-space lattice. Each virtual page covers seven cells per axis and stores eight samples along each axis so neighboring pages share their boundary. The R32Uint page table points to an R32F atlas page or contains one of three states. `0xffffffff` means the whole page equals the positive distance limit, `0xfffffffe` means it equals the negative limit, and `0xfffffffd` means its physical allocation exceeded `maxPhysicalPages`. The sparse trace reports coverage state 3 and displays orange if it reaches an unavailable page. Its `statistics` buffer lists required physical pages, resident pages, positive constant pages, negative constant pages and unavailable pages.

The sparse pass composes each page's samples on every frame and allocates atlas storage for pages with varying distances. It uses the same samples as the dense pass, including the clipped tail of a partial page. This provides a dense comparison for page sampling.

**Lighting / Global Distance Field / R8 and Coarse Mip** enables `quantized` on the sparse producer and debug pass. The page atlas stores UE's normalized distance encoding in R8 UNORM. The `coarseMip` volume has `ceil(N/4)` samples per axis and encodes a distance range four times wider than the fine pages. The first propagation pass reads the fine pages and applies the Eikonal update. Four more passes read the previous R8 result and propagate into the alternate volume.

Each coarse write applies the Direct3D reference FLOAT-to-UNORM rounding before storing the normalized value in Vulkan. This keeps the five propagation steps on the same rounding rule.

Tracing starts with the coarse sample and switches to an available fine page when the coarse encoded value is below 0.625. A negative constant page seeds the mip as a fine page filled with zero. Overflow pages use the coarse estimate when propagation reaches them and keep coverage state 3 when the coarse value remains saturated positive. Both R8 textures can be inspected as normalized values. The path depends on Vulkan's extended storage image formats.

**Lighting / Global Distance Field / Camera Clipmaps** enables multiple levels with `clipmapCount` on the quantized producer. Counts from 2 to 6 use the graph camera. `extent` sets the innermost half-width in meters. Each successive level doubles the half-width and voxel size, and independently snaps its center to the nearest page. The logical resolution rounds up to a multiple of 28. All levels share the R8 atlas and page budget, with the page table and coarse mip packed fine to coarse along Z. All levels are rebuilt each frame.

The **Clipmap Levels** debug mode colors hits by level. Its multi-level `trace` output stores hit distance, sample count, coverage state, and hit level (`-1` for a non-hit). State 0 means the nonempty segment missed all clipmaps, 1 means a fully covered miss, 2 means a hit, 3 means an unavailable page sample, 4 means a miss with partial coverage, 7 means that a clipmap exhausted its step budget, and 8 means an empty ray interval.

The opaque trace resumes at the clipped exit of the previous layer and gives each coarser clipmap its own step budget, following UE 5.7.4. Metallic also follows this order after a page becomes unavailable. A later hit identifies the hit clipmap and keeps the accumulated successful sample count. If the remaining layers produce no hit, the first unavailable sample or exhausted budget still determines the final state. An empty ray interval consumes zero samples and identifies no hit clipmap. UE uses a single no-hit result and counts loop iterations with a different terminal-sample convention.

### Lumen Mesh Cards and Surface Cache

The optional Embree SDK also enables the Mesh Cards baker. Bake a card set per source mesh. Supply a matching Mesh Distance Field to use its expanded local bounds, as in Unreal 5.7.4. The sample uses the same geometry as the Mesh Distance Field boxes.

```powershell
cmake --build build-release --target MetallicMeshCardsBake
New-Item -ItemType Directory -Force .cache/mesh-distance-field | Out-Null
.\build-release\Source\MetallicMeshCardsBake.exe Asset/Lumen/SurfaceCacheBoxes.gltf 0 .cache/mesh-distance-field/SurfaceCacheBoxes.meshcards --distance-field=.cache/mesh-distance-field/BoxInstances.meshdf
```

Open **Lighting / Lumen / Surface Cache** after baking. The pass consumes `meshCardPaths` in the bound world scene. `atlasSize` and `tileResolution` configure the fixed square atlas. The `cards` buffer lists world-space OBBs in tile order, with the scene object, source mesh and cardinal direction. The `scene` buffer records the live card count, tile resolution and the 64-bit scene resource identity.

Inspect `albedo`, `normal`, `emissive`, `depth` and `lighting`. Albedo stores the square root of the fully rough diffuse response. Normal stores card-space XY and coverage. Depth is R16 UNORM with 1 reserved for uncovered texels. Lighting stores outgoing scene-linear radiance from the world Sun with shadows disabled and material emissive. The pass captures each frame and supports single-sided, opaque static triangles with orthogonal instance transforms, including mirrors. Atlas residency and lighting feedback are subsequent stages of Lumen integration.

### Lumen Global Tracing

With both `BoxInstances.meshdf` and `SurfaceCacheBoxes.meshcards` baked, open **Lighting / Lumen / Global Tracing**. The graph builds quantized Global Distance Field clipmaps and the fixed Surface Cache from the same world. `Trace.rays` lists world-space ray origins, directions and `minDistance`/`maxDistance` in meters. The pass normalizes directions. `Trace.radiance` contains one scene-linear RGB texel per ray.

Capture `Trace.results` at `AfterPass` to inspect the 96-byte records in ray order. Each record contains six four-component lanes.

| Lane | Components |
| --- | --- |
| `float4 metrics` | Hit distance, surface expansion, total card weight, near-origin radiance factor |
| `float4 normal` | World-space hit normal, dominant card weight |
| `float4 radiance` | Scene-linear outgoing RGB, unused |
| `uint4 state` | GDF classification, hit clipmap, sample count, cache status |
| `uint4 candidates` | Ranked scene object IDs, `0xffffffff` for empty entries |
| `uint4 card` | Dominant card index, scene object ID, atlas U/V as float bit patterns |

The GDF classifications are outside (0), covered miss (1), hit (2), unavailable (3), partial (4), step limit (7) and empty interval (8). The cache statuses are no geometry hit (0), near-origin suppression (1), no covered card sample (2), sampled (3) and mismatched producer scene identities (4). A sampled hit can have zero RGB. Unset clipmap, card and object indices are `0xffffffff`.

The reference is UE 5.7.4's full-coverage, opaque diffuse Global Tracing path. This graph supplies fixed rays in place of the screen probe ray producer.

- The trace uses Metallic's meter-space clipmaps, packed coarse mip, signed constant pages and explicit page-overflow classification. Fine-atlas normal taps use the current clipmap's clamped bounds. The debug step count includes the successful hit sample.
- `buildObjectGrid` adds 64 four-candidate cells per physical page. It applies UE tile/cell culling and packed distance ranking to Metallic's per-frame instance list. The clipmap's positive endpoint belongs to its last page. Ranked instance indices resolve to full Metallic scene object IDs before publishing.
- Surface sampling uses one resident tile per cardinal card, world-space card axes for orientation and bounds, and tile-order iteration for each matching object. The atlas provides scene-linear final lighting with unit exposure scale.

### Lumen Single Bounce

Open **Lighting / Lumen / Single Bounce** after baking the receiver, offscreen emitter and blocker. Run the existing Embree-enabled tools from the repository root. `$bakerDir` is the directory containing the two executables. The scene starts with the blocker visible and the sky and celestial lights disabled.

```powershell
$scene = 'Asset/Lumen/SingleBounce.gltf'
$cache = '.cache/mesh-distance-field'
$meshes = @('SingleBounceReceiver', 'SingleBounceEmitter', 'SingleBounceBlocker')
for ($meshIndex = 0; $meshIndex -lt $meshes.Count; $meshIndex++) {
    $stem = Join-Path $cache $meshes[$meshIndex]
    & (Join-Path $bakerDir 'MetallicMeshDistanceFieldBake.exe') $scene $meshIndex "$stem.meshdf"
    if ($LASTEXITCODE -ne 0) { break }
    & (Join-Path $bakerDir 'MetallicMeshCardsBake.exe') $scene $meshIndex "$stem.meshcards" "--distance-field=$stem.meshdf"
    if ($LASTEXITCODE -ne 0) { break }
}
```

Toggle the blocker or edit the emitter's emissive factor to compare the next rendered frame. `Gather.indirectDiffuse` is scene-linear outgoing indirect diffuse, including the receiver response. `Gather.color` adds this contribution to the RGBA32F Deferred color. `Gather.probeRadiance` holds four floats per direction in probe-tile order, with RGB radiance and the GDF classification in the last component.

The sample uses the UE 5.7.4 uniform placement, equal-area directions, plane-weighted interpolation and numerical diffuse integral. Its fixed phase samples a 16-pixel grid with 8×8 centered directions per probe. Equal trace and gather resolutions specialize the trace scatter to one writer per direction. Probe depth and normals come from the current Deferred material evaluation, and Meter-space GDF hits reuse the fixed Surface Cache. The graph evaluates only card transport, with sky disabled in the sample world. Screen traces, adaptive placement, importance sampling, history, spatial filtering and World Radiance Cache belong to later stages.

`exportLumenReceivers` substitutes Lumen diffuse for the usual environment diffuse on supported opaque materials. The exported receiver response uses the existing Metallic BRDF and material occlusion, including the camera-path aerial transmittance and UE's distant illumination rescale with short-range AO disabled. `shadowedLighting` evaluates direct sunlight and world point/spot lights in a separate card lighting stage. It adapts UE's opaque GDF shadow trace to Metallic's resident tiles and uses the existing photometric light sampling. The sample uses a single fixed clipmap and per-covered-texel lighting over the world light list.

### Cornell Box GI Playground

In the **Render Graph Editor**, open **Built-in Sample** and select a Cornell Box variant.

| Sample | ID | Rendering |
| --- | --- | --- |
| Cornell Box / Path Tracing | `cornell-box-path-tracing` | Progressive OpenPBR path tracing at four samples per pixel per frame and a maximum depth of 12. |
| Cornell Box / Realtime GI | `lumen-cornell-box` | Raster visibility and shading with Lumen Final Gather, temporal accumulation and multi-bounce card lighting. |
| Cornell Box / GI Comparison | `cornell-box-compare` | Path tracing on the left and realtime GI on the right. Drag the divider in the viewport to compare the same pixels. |

All three variants open the same 2 m room, with red and green side walls, two neutral blocks and a fixed initial camera and exposure. The only emitter is a downward-facing 0.66 m × 0.64 m ceiling quad named **Ceiling Emitter**. The quad is the underside of a closed panel for Mesh Distance Field baking. The sample disables sky, Sun, Moon and auto exposure, and its punctual light list is empty. The realtime and comparison variants load their pre-baked Mesh SDFs and Mesh Cards from `Asset/Lumen/CornellBoxBakes/`.

The Cornell Box graph enables Final Gather and multi-bounce card lighting for static opaque diffuse GI. `ScreenTrace` builds a closest-depth HZB from current reversed-Z depth and reads the previous frame's composited color and depth. `Gather` places adaptive probes, tries screen tracing with previous-depth validation, falls back to the GDF and Surface Cache, and runs three spatial filtering passes. `Temporal` accumulates the integrated diffuse with reprojected depth and normal rejection. Camera cuts, extent changes and scene or lighting edits invalidate both screen-space histories.

Final Gather reads `Asset/Lumen/FastBlueNoiseVec2.rg8` for screen-probe directions. Temporal Gather reads `Asset/Lumen/FastBlueNoiseScalar.r8` when stochastic FP16 history quantization is enabled. Both 128 × 8192 textures are stored as raw bytes in the project asset directory. The assets were imported from UE 5.7.4 and verified by SHA-256 at import time.

The Cornell Box realtime and comparison graphs default to 64 rays (8 × 8) per active probe. For a higher-sample reference, select the `Gather` node and set **Runtime Settings → Rays per Probe → 256 (Reference)**, or set `Gather.tracingOctahedronResolution` to `16` in the graph. Select **64 (Standard)** to switch back. Both modes scatter into 8 × 8 gather directions before spatial filtering.

The SH3 format stores a packed R11G11B10 ambient coefficient in `Gather.probeSHAmbient` and the eight signed directional coefficients per channel in the RGBA16F `Gather.probeSHDirectional` atlas. Integration evaluates the interpolated coefficients at the receiver normal. Both graphs use a relative HZB thickness of 0.003 and a 252³ Global Distance Field over 3.6 m with an 8,000-page budget. The sample cycles through 64 ray-direction phases and sets the history limit to 32 frames. Stochastic rounding uses the scalar texture's 64 phases.

The gather pass defaults to 8 × 8 trace directions, 8 direction phases, a relative screen-trace thickness of 0.02 and a bordered 6 × 6 OCT irradiance map. The temporal pass defaults to a history limit of 10. The classic fixed-ray diagnostics select their original grid and history settings.

Inspect `Temporal.indirectDiffuse` for accumulated, unexposed GI, `Gather.indirectDiffuse` for the current frame and `Temporal.historyWeight` for accepted history. In `Gather.probeRadiance`, the fourth component stores trace class 2 for a GDF hit and 5 for an accepted screen hit. Set the **Ceiling Emitter** material's emissive color to zero for a dark control.

#### Repeatable GI image baseline

From the repository root, capture the built-in Comparison graph with its fixed camera at 384 × 384. Use a new output directory for each capture.

```powershell
cmake --build build-relwithdebinfo --target MetallicCornellBoxGIBaseline
.\build-relwithdebinfo\Source\MetallicCornellBoxGIBaseline.exe build/cornell-gi-baseline/run-001 Tools/CornellBoxGIRegions.json
python -B Tools/MeasureCornellBoxGI.py --manifest build/cornell-gi-baseline/run-001/capture.json --out build/cornell-gi-baseline/run-001/metrics.json
```

The tool uses a Vulkan preview device with ray queries and synchronization validation. At frames 128 and 256 it captures `PathTrace.color`, `Temporal.color` and `Temporal.indirectDiffuse` in the same graph execution before presentation. It zeros the scene's emissive material and captures the first and eighth dark frames. The manifest records execution stamps and the five top-left pixel rectangles. The regions all lie on the floor and cover its center, the floor side of each block contact, and mirrored areas near the red and green walls.

The report compares scene-linear ACEScg luminance, contact-to-floor luminance ratios and the red or green channel fraction relative to the center floor. It checks the reference's change between the two lit captures at the same probe phase and verifies exact RGB zero in the settled dark capture.

#### Full-resolution probe interpolation jitter

Set `Gather.fullResolutionJitterWidth` to the effective UE shader width and connect `VBuffer.depth` to `Gather.jitterDepth`. Use width `1` for the UE 5.7.4 Epic setting at standard Final Gather quality. The property defaults to zero. The depth test samples mip 0 at the proposed coordinate and reconstructs the position at the fractional UV. An accepted offset shifts the uniform probe corners and bilinear weights. Adaptive probe scoring uses the original receiver pixel.

Use the standard Cornell settings to capture frames 193–256 and both dark frames. The same-input comparison runs a second integration using the baseline's screen traces, filtered probe data and irradiance. It captures the jittered Gather images, proposed and accepted offsets, and both sets of interpolation indices and weights. The live variant feeds jittered integration into Temporal and subsequent Screen Trace feedback.

```powershell
$capture = '.\build-relwithdebinfo\Source\MetallicCornellBoxGIBaseline.exe'
& $capture build/cornell-jitter/paired Tools/CornellBoxGIRegions.json --jitter-phases --compare-interpolation-jitter
& $capture build/cornell-jitter/live Tools/CornellBoxGIRegions.json --jitter-phases --interpolation-jitter
```

#### Short Range AO and Bent Normal

The software opaque path follows UE 5.7.4 Epic's full-resolution depth horizon search. `LumenShortRangeAOPass` writes a half-resolution 11/11/10 UNORM packed world-space bent normal. `LumenTemporalGatherPass` stochastically reconstructs it at full resolution, reprojects the packed AO history with the diffuse history taps, and applies diffuse multibounce occlusion during composition. The length of the decoded vector is ambient visibility.

The capture tool enables the producer with `VBuffer.depth` and `Deferred.lumenNormal`. It enables `Deferred.exportLumenDiffuseAlbedo`, `Gather.exportIndirectIrradiance`, and the consumer's `shortRangeAO` property. With AO enabled, `Temporal.currentIndirectDiffuse` takes `Gather.indirectIrradiance`, `Temporal.diffuseAlbedo` takes `Deferred.lumenDiffuseAlbedo`, and `Temporal.shortRangeAORaw` takes `ShortRangeAO.shortRangeAORaw`. This permutation composes the opaque diffuse lobe without physical aerial perspective.

```powershell
$capture = '.\build-relwithdebinfo\Source\MetallicCornellBoxGIBaseline.exe'
& $capture build/cornell-shortao/paired Tools/CornellBoxGIRegions.json --compare-short-range-ao
& $capture build/cornell-shortao/live Tools/CornellBoxGIRegions.json --short-range-ao
```

Both modes save frames 193–256 and the first and eighth dark frames. The paired mode evaluates AO on the same probes as the baseline and keeps its feedback isolated. The live mode publishes the AO-composited color to Screen Trace. Both record the raw half-resolution and accumulated full-resolution bent normals in `capture.json`.

For a Detail Tracing comparison, use `--trace` in both groups and `--detail` in group B. The fixed-ray triangle fixtures use centered directions, the original 112³ GDF with 4,096 physical pages and a screen-trace relative depth thickness of 0.02. The capture tool saves the effective graph and reads probe placement, receiver position and normal, radiance, distance, and the per-ray handoff at the same execution as the lit images.

```powershell
$capture = '.\build-relwithdebinfo\Source\MetallicCornellBoxGIBaseline.exe'
$ab = 'build/cornell-detail-ab/run-001'
& $capture "$ab/a" Tools/CornellBoxGIRegions.json --trace
& $capture "$ab/b" Tools/CornellBoxGIRegions.json --trace --detail
foreach ($group in 'a', 'b') {
    python -B Tools/MeasureCornellBoxGI.py --manifest "$ab/$group/capture.json" --out "$ab/$group/metrics.json"
}
python -B Tools/MeasureCornellBoxGI.py --baseline "$ab/a/metrics.json" --candidate "$ab/b/metrics.json" --screen-traces on --out "$ab/ab.json"
python -B Tools/CheckCornellBoxDetailRays.py --a "$ab/a/capture.json" --b "$ab/b/capture.json" --pin Tools/CornellBoxDetailRays.json --out "$ab/triangle-oracle.json"
```

The fixed floor probe at pixel `[286, 302]` tests both outcomes. Lane `[2, 5]` hits the tall block within the Mesh SDF interval. Lane `[3, 7]` first intersects the ceiling beyond that interval and continues in the GDF. The triangle oracle starts from the same biased origin as Gather. It subtracts the receiver-to-origin offset from `detailEnd` before comparing hit distances, and checks GDF continuation using the recorded endpoints.

Group C isolates Screen-to-GDF continuation. The capture flag `--screen-gdf-continuation` sets `Gather.screenGDFContinuation` and uses the same GDF start as Detail Tracing after a rejected screen trace, `max(MinTraceDistance, ScreenEnd - 2 * SurfaceBias)`. Group C uses the existing GDF and Surface Cache without the Mesh SDF stage. Compare A to C for the effect of the Screen endpoint and C to B for the added Detail stage and its handoff.

The Sponza realtime graph opts into Screen-to-GDF continuation. The Cornell Box control remains a comparison from the minimum trace distance. Its floor oracle includes an emitter edge where the screen endpoint can advance the biased software ray past the first triangle. Add `--split-trace` to a ray capture to test the active probe and split-ray dispatches at the same fixed phase.

```powershell
& $capture "$ab/c" Tools/CornellBoxGIRegions.json --trace --screen-gdf-continuation
python -B Tools/MeasureCornellBoxGI.py --manifest "$ab/c/capture.json" --out "$ab/c/metrics.json"
python -B Tools/MeasureCornellBoxGI.py --baseline "$ab/a/metrics.json" --candidate "$ab/c/metrics.json" --screen-traces on --out "$ab/ac.json"
python -B Tools/MeasureCornellBoxGI.py --baseline "$ab/c/metrics.json" --candidate "$ab/b/metrics.json" --screen-traces on --out "$ab/cb.json"
python -B Tools/CheckCornellBoxDetailRays.py --a "$ab/a/capture.json" --c "$ab/c/capture.json" --b "$ab/b/capture.json" --pin Tools/CornellBoxDetailRays.json --out "$ab/triangle-oracle-abc.json"
```

Use `--floor-rays` in place of `--trace` for the floor witness set. It expands the probe rectangle and captures `ScreenTrace.closestHZB` in the same execution. Run `CheckCornellBoxDetailRays.py --floor` with the A and C manifests to compare the captured start with the first triangle hit from each biased origin. Pass `--c-fixed` to include a second continuation capture.

Group D is an opt-in deviation from the UE handoff. `Gather.gdfPrefixValidation` uses the same biased world ray to trace from `MinTraceDistance` to the earlier of the Screen handoff and the trace cap. A GDF hit uses the regular Surface Cache lighting for that first hit. All other prefix results proceed from the original continuation point. The feature requires `screenGDFContinuation` and the Mesh SDF stage disabled. It uses the existing quantized GDF, including its surface expansion and residency semantics.

```powershell
& $capture "$ab/d" Tools/CornellBoxGIRegions.json --floor-rays --gdf-prefix-validation
python -B Tools/MeasureCornellBoxGI.py --manifest "$ab/d/capture.json" --out "$ab/d/metrics.json"
```

Use the same `--floor-rays` capture for the C comparison, with `--screen-gdf-continuation`. The floor oracle accepts `--d` alongside `--c-fixed` to verify the emissive hit.

Group E adds `Gather.gdfPrefixRejectReceding` to group D. At an expanded prefix candidate with positive signed distance, the same quantized GDF supplies the gradient at the march sample. If its dot product with the ray direction is positive, the ray is moving away from the surface and continues with the existing step. An unavailable or degenerate gradient retains the candidate. The rule changes the UE expansion test only in the opt-in prefix stage. It uses the quantized field's local gradient as evidence, so it can still accept false hits where that evidence is inconclusive.

```powershell
& $capture "$ab/e" Tools/CornellBoxGIRegions.json --floor-rays --gdf-prefix-reject-receding
python -B Tools/MeasureCornellBoxGI.py --manifest "$ab/e/capture.json" --out "$ab/e/metrics.json"
python -B Tools/MeasureCornellBoxGI.py --baseline "$ab/d/metrics.json" --candidate "$ab/e/metrics.json" --screen-traces on --out "$ab/de.json"
```

Pass `--e` with the E manifest to the same floor oracle invocation. The fixed room-opening ray checks the receding hit from the short block's side, and the emissive ray checks that the forward-facing prefix hit survives.

Group F adds `Gather.gdfPrefixMeshValidation` to group E. It only verifies accepted GDF prefix candidates. It uses the complete list of visible Mesh SDF instances from `detailDistanceFieldPaths`, without the view-dependent Detail grid or the 1.8 m Detail limit. Each instance clips the prefix to its original local mesh bounds, then tests the finest baked Mesh SDF for a nonpositive sample with zero expansion. A confirmed nearest hit uses that mesh's distance and Surface Cache radiance. When all original bounds miss, tracing continues from the Screen handoff. If an earlier bound intersects but its Mesh SDF cannot confirm a hit, F preserves the GDF result. This preserves the Cornell emitter edge where the baked field stays positive.

```powershell
& $capture "$ab/f" Tools/CornellBoxGIRegions.json --floor-rays --gdf-prefix-mesh-validation
python -B Tools/MeasureCornellBoxGI.py --manifest "$ab/f/capture.json" --out "$ab/f/metrics.json"
python -B Tools/MeasureCornellBoxGI.py --baseline "$ab/e/metrics.json" --candidate "$ab/f/metrics.json" --screen-traces on --out "$ab/ef.json"
```

Add `--f` with the F manifest to the floor oracle invocation. The fixed rays cover two short-block corner misses, two near-plane hits returned by the GDF tail, the emitter edge and a farther ceiling hit that takes precedence over an earlier GDF candidate.

#### Tall-block contact visibility replay

`--contact-visibility` enables the F configuration and captures the probe taps used by Gather integration. The lit and settled dark captures include scene-linear Gather and Temporal outputs, the Deferred receiver data, and the probe radiance before and after all three spatial filter passes. Use fresh directories for the plain F control and the instrumented capture.

```powershell
$capture = '.\build-relwithdebinfo\Source\MetallicCornellBoxGIBaseline.exe'
$ab = 'build/cornell-contact-visibility/run-001'
& $capture "$ab/f" Tools/CornellBoxGIRegions.json --floor-rays --gdf-prefix-mesh-validation
& $capture "$ab/g" Tools/CornellBoxGIRegions.json --contact-visibility
python -B Tools/MeasureCornellBoxContactVisibility.py --baseline "$ab/f/capture.json" --manifest "$ab/g/capture.json" --out "$ab/contact-visibility.json"
```

The replay verifies Gather integration against the GPU output and follows lane `[3, 7]` at probe pixel `[286, 334]` through the original spatial filter weights. It also checks emitter-colored Screen and GDF samples within three filter steps of the ROI taps against their triangle first hits. Group G removes the verified emitter contributions from the current Gather result where the receiver-to-emitter segment hits the tall block first. The report separates the pinned sample at receiver pixel `[286, 322]` from the combined result, and includes the measured Temporal result and settled dark check.

`Gather.probeInterpolationReadback` takes an in-bounds `[x, y, width, height]` pixel rectangle with at most 128 MiB of records. Each row-major, 32-byte record contains four `uint32` probe indices followed by four FP32 weights actually used by integration. A pixel without a receiver has four `0xffffffff` indices and zero weights. Read the raw buffer through RenderDebug as eight `u32` words per pixel.

#### Temporal comparison across all probe phases

Use `--contact-temporal-phases` to capture the F configuration at eight consecutive graph executions from 248 through 255, with execution 247 as the Temporal seed. Each phase includes the Gather and Temporal images, history weights, probe data for the visibility replay, and the same-frame Path Trace reference. The capture also records the first and eighth frames after the emitter is extinguished.

```powershell
$run = 'build/cornell-contact-temporal/run-001'
& $capture "$run" Tools/CornellBoxGIRegions.json --contact-temporal-phases
python -B Tools/MeasureCornellBoxContactTemporal.py --manifest "$run/capture.json" --out "$run/temporal-replay.json"
```

The replay verifies that the fixed camera reprojects each ROI pixel to its original texel and that the measured weights reproduce the F blend and FP16 history. It then applies the triangle-verified subtraction to each phase's Gather input and advances the corrected history from the F seed. The report distinguishes the net shift relative to current Gather from the weighted history term relative to Path Trace. The replay uses the captured F Screen Trace inputs for all eight phases.

Use `--steady` with the same manifest to repeat the corrected Gather inputs and measured history weights. The replay stops when one full period leaves the FP16 history unchanged and the linear weight of the initial history is at most `1e-6`. A second replay starts from zero history to measure any remaining sensitivity to the initial FP16 values. The report compares the first cycle with the converged cycle, and applies the captured first dark frame to the converged history.

```powershell
python -B Tools/MeasureCornellBoxContactTemporal.py --manifest "$run/capture.json" --out "$run/temporal-steady.json" --steady
```

Use the same capture and steady report to attribute the remaining Gather error.

```powershell
python -B Tools/MeasureCornellBoxGatherResidual.py --manifest "$run/capture.json" --steady "$run/temporal-steady.json" --out "$run/gather-residual.json"
```

This diagnostic follows the raw ray contributions through all three spatial passes and applies G's verified emitter subtraction at each stage. It compares the captured interpolation weights with an estimate using only the highest-weight tap. The report traces emitter-colored candidates to their triangle first hits and reconstructed Screen history texels. Separate estimates set candidate radiance to zero to bound its influence. The measured G metrics retain the original F trace inputs and G subtraction.

This port uses UE 5.7.4's opaque permutations with these adaptations.

- Adaptive placement uses eight-phase Hammersley and a fixed allocation of one uniform plus up to eight adaptive probes per tile. This replaces UE's global compaction, with candidate marking recomputed from the current frame. Probe interpolation uses Deferred world position, linear depth and normal. The full-resolution blue-noise offset is zero.
- Final Gather uses UE's uniform tracing branch at matching 8 × 8 trace and gather resolutions. All rays in a screen tile share the vector blue-noise offset at `frameIndex % 8`. Before spatial filtering, the composite pass uses the same offset and a second lookup 31 slices later to scatter each ray into an octahedrally wrapped gather bin. Colliding samples sum their quantized radiance and retain the minimum valid hit distance. Metallic's unresolved rays have a negative distance and contribute no sample. Empty bins have zero radiance and distance −1. The raw ray outputs retain tracing-lane order. The Cornell ray-witness tool sets `Gather.fixedProbeDirections` for pinned geometry and contact replays.
- The spatial pass uses the four axial neighbors from UE's static filter and the jittered cone direction of the receiving probe. The final numerical diffuse integral evaluates the centers of the gather bins. Metallic keeps FP32 radiance and hit distances in meters, rejects coincident hit endpoints, and clamps the angular cosine to its valid domain. Moving-probe taps and short-range AO are outside this variant.
- Screen tracing uses a packed closest HZB addressed in Metallic's viewport and reversed-Z convention. Motion comes from the prior camera for static surfaces. Accepted screen hits read the previous composited scene-linear color before postprocessing, corresponding to UE's default screen-tracing source. GDF hits sample `Radiosity.lighting`. Metallic publishes the composited color and its device depth as graph-owned textures and uses unexposed radiance, with a pre-exposure ratio of one. UE's distance continuation, hair and multi-layer paths are outside this variant.
- Temporal accumulation reprojects with Metallic's previous camera and jitter and uses the optional 45-degree normal rejection. History contains material-modulated diffuse with UE's encoded accumulation count. The rejection threshold uses the scalar value from the UE rendering commandlet's black dummy input. Large Scene enables rigid receiver reprojection. With skylight off, the manual Sun can change while the geometry-valid history continues to accumulate.

The `Radiosity` pass traces cached outgoing card radiance with 4 × 4 hemisphere directions per probe and a four-texel probe spacing. It filters with four axial neighbors, projects to two-band spherical harmonics and integrates by card plane. It accumulates up to four updates per texel, then applies the card response to the sum of direct and indirect irradiance before adding emission. Inspect `Radiosity.indirectIrradiance` for irradiance in RGB and the accumulation count in alpha. Compare outgoing `Radiosity.lighting` with direct-and-emissive `SurfaceCache.lighting`. The full-atlas path retains indirect feedback across isolated Sun edits and combines that feedback with the current direct light before tracing. When another change invalidates the old feedback on a running atlas, it traces the current direct-and-emissive Surface Cache and starts a fresh accumulation. The initial frame initializes direct lighting and emission. Camera motion and cuts preserve feedback while the fixed atlas and scene revisions match.

Both Cornell Box and Large Scene set `ScreenTrace.historySource` to `composited` and `historyProducer` to `Temporal`. The named pass enables `publishScreenTraceHistory` and takes its `sceneDepth` from `VBuffer.depth`. Inspect `ScreenTrace.previousColor` and `ScreenTrace.previousDepth` to see the same-frame pair read on the next frame. For a Deferred comparison, set `historySource` to `deferred`, connect `Deferred.color` to `ScreenTrace.color`, disable publication and remove the `Temporal.sceneDepth` edge. Older graphs without a source setting use this Deferred path.

The radiosity path adapts UE 5.7.4's software GDF branch for a sky-disabled scene.

- Every card owns one fixed tile. The default full-atlas path updates every tile each frame. A dense dispatch replaces UE's mapped page allocator, prioritized update budget and per-page update phases. The initial frame exports direct lighting and emission and clears indirect history. Large Scene opts into per-card scheduling.
- Ray and filter noise use the UE rendering commandlet's black dummy inputs. Trace, SH, irradiance and outgoing lighting use the reference PF_FloatRGBA precision with scene-linear exposure. SH coefficients use packed FP16 buffers. The history count occupies the irradiance texture's alpha. Metallic exports direct irradiance in RGBA32F and uses its card normal, depth and shared GDF trace. Ray distance and bias use meters.
- Software probe occlusion is disabled, matching the reference default. The fixed atlas consumes one resolution for each card and has no cone-driven mip selection or page feedback.

To regenerate the shipped bakes with the baking loop above, set `$scene = 'Asset/Lumen/CornellBox.gltf'`, `$cache = 'Asset/Lumen/CornellBoxBakes'` and use these stems in source mesh order. Large Scene and Detail Tracing use the shipped bakes. The standalone World Radiance Cache sample loads copies from `.cache/mesh-distance-field/`.

```powershell
$meshes = @('CornellBoxFloor', 'CornellBoxCeiling', 'CornellBoxBackWall', 'CornellBoxLeftWall',
    'CornellBoxRightWall', 'CornellBoxShortBlock', 'CornellBoxTallBlock', 'CornellBoxCeilingEmitter')
```

### Sponza with GI

Select **Sponza with GI** in the Render Graph Editor's **Built-in Sample** selector or use the matching **Environment** action. The three entries share `Asset/Sponza/SponzaGI.metallic_scene.json`, the Sponza VSM camera, a 1280 × 720 render extent and fixed exposure. The scene uses a 24 lx sun with the sky and HDRI disabled.

| Sample | ID | Rendering |
| --- | --- | --- |
| Sponza with GI / Realtime GI + VSM | `sponza-gi-realtime` | VBuffer, cached SMRT Virtual Shadow Maps for direct sunlight, Lumen Final Gather for indirect diffuse lighting, and software Lumen Reflections. |
| Sponza with GI / Path Tracing | `sponza-gi-path-tracing` | Progressive OpenPBR path tracing with four samples per pixel per frame and a maximum path depth of 12. |
| Sponza with GI / GI Comparison | `sponza-gi-compare` | Path tracing on the left and realtime GI, reflections and VSM on the right. Drag the divider to reveal the same pixels under the shared display transform. |

`SponzaGI.gltf` reuses the original Sponza geometry buffers and materials. It places the opaque single-sided sections and the alpha-masked double-sided sections in separate meshes. The bundled Mesh SDF and Mesh Cards in `Asset/Sponza/SponzaGIBakes/` cover the opaque mesh for world tracing. Raster visibility, VSM and path tracing use both meshes. Inspect `Shadows.shadows` for direct-light visibility and `Temporal.indirectDiffuse` for accumulated scene-linear GI.

The Sponza realtime graph enables `persistentPages` on its single fixed GDF volume. Stationary frames and Sun-only edits reuse the page table, atlas, and coarse mip. Geometry, visibility, and Object Grid material changes rebuild the volume. Inspect `GlobalDistanceFieldPages.updateStatistics` for the full-refresh flag, updated and preserved virtual page counts, and updated level count.

The Sponza realtime graph enables `Gather.compactProbeAtlas` for UE 5.7.4 adaptive placement. Uniform probes occupy the first row-major range. The Mark pass records candidates, and the Spawn pass reserves the adaptive suffix from one global counter. `adaptiveProbeAllocationFraction` defaults to `0.5`. At 1280 × 720 this reserves 3,600 uniform slots and up to 1,800 adaptive slots. Integration uses each tile's list of admitted probes. Trace and filter dispatches process the active list without clearing inactive atlas slots. The option requires `activeProbeDispatch`. Gather owns placement in graphs without World Radiance Cache.

With the compact atlas, `Gather.probeAdaptiveCount` reports attempted reservations before the cap. Read `Gather.probeTileAdaptiveCounts` and `Gather.probeTileAdaptiveIndices` together for the accepted entries. Add the uniform slot count to each listed adaptive index to address `probePositions` and the raw ray buffers. `Gather.probeOrigins` maps allocated slots back to their original tile and candidate slot. `probeTraceReadback` and `probeInterpolationReadback` continue to report the original tile and slot identifiers for both layouts.

The realtime and comparison graphs composite reflections after diffuse temporal accumulation. Reflections first trace the screen depth hierarchy and validate hits against the history from `Temporal`. Unresolved rays can trace the finer `SponzaOpaqueReflection.meshdf` within 1.8 m, then continue through the shared GDF up to 40 m. World-space hits sample `Radiosity.lighting`. `Reflections.reflections` contains raw radiance and roughness fade. In `Reflections.trace`, classification 5 identifies a screen hit and 9 identifies a Mesh SDF hit. The other classes describe GDF results.

With HDRI lighting off, manual Sun-only edits preserve geometry-valid diffuse and screen trace history. Temporal compares the maximum channel of effective linear Sun irradiance with the previous rendered frame. The UE 5.7.4 global change rule uses a `1e-5` floor and invalidates diffuse history when the old-to-new ratio falls outside `[0.25, 4]`. Metallic uses working-space RGB irradiance. UE uses directional light proxy color, whose `FLinearColor::GetMax()` includes alpha. Metallic also rejects previous screen trace color on a global change to avoid feeding old composited illumination into the new gather. Other light, environment and geometry changes keep their existing screen-history invalidation rules.

Both realtime graphs enable persistent cards. Sun edits relight all 12 baked cards at the current direction and intensity, then combine direct light with existing indirect irradiance before tracing. Radiosity traces four cards per frame. A Sun edit, including a large intensity jump, keeps each resident card's indirect accumulation and phase. This follows UE's direct-before-indirect ordering. Metallic relights the full resident set on a Sun edit and budgets indirect work by fixed cards, while UE independently budgets both passes by physical page area. Metallic still resets affected card history for recaptures, roster changes and other captured-light changes.

To regenerate the pair, bake mesh index 0 of `Asset/Sponza/glTF/SponzaGI.gltf` with the default Mesh SDF and Mesh Card options. Pass `--distance-field=Asset/Sponza/SponzaGIBakes/SponzaOpaque.meshdf` to the card baker. The opaque mesh has open edges, so its signed-distance field uses the baker's normal sign estimation on the original triangles.

Regenerate the reflection Mesh SDF with `MetallicMeshDistanceFieldBake Asset/Sponza/glTF/SponzaGI.gltf 0 Asset/Sponza/SponzaGIBakes/SponzaOpaqueReflection.meshdf 1.01 154`. The final argument caps the mesh distance-field resolution in voxels.

### Lumen Large Scene

**Lighting / Lumen / Large Scene** places four emissive rooms at X = 0, 40, 82 and 156 m. Their geometry instances reuse the eight Cornell Box mesh bakes. Each room has a suspended, uniformly scaled ceiling panel for coarse GDF coverage. The Mesh SDF instance path requires orthogonal, equal-length transform axes. The sample uses the same world lighting settings as Cornell Box.

The four camera-centered GDF levels have half-extents of 25, 50, 100 and 200 m. They follow UE 5.7.4's 25 m first level and doubling rule. This fixture uses 112 cells per axis and a shared pool of 4096 R8 pages. Its fixed 512 × 512 Surface Cache uses 32 × 32 tiles.

The sample enables `persistentPages` on the quantized GDF pass. Stationary or sub-page camera movement reuses all GDF data. When a clipmap scrolls, positive and negative constant pages and resident atlas slots in the overlapping volume keep their state. Newly exposed slabs compose new distances, and the affected levels run the five coarse mip passes. Geometry, visibility and object-grid material changes trigger a full refresh, as does a camera jump that leaves any clipmap without overlap. `GlobalDistanceFieldPages.updateStatistics` contains the full-refresh flag, updated virtual page count, preserved virtual page count and updated mip level count in that order. The setting is opt-in and requires tracked render-frame submission.

Set `persistentCards` on Surface Cache and Radiosity and connect `SurfaceCache.cardUpdates` to `Radiosity.cardUpdates` to use per-card updates. Surface Cache recaptures tiles for changed transforms and captured materials. Changes to the card roster clear and recapture the fixed atlas. Isolated edits of the manual Sun with HDRI lighting and Moon disabled relight resident cards while retaining their indirect history. Other captured-light edits relight cards and reset their indirect history. Shadow occluder changes or camera motion with a multi-level GDF can also relight resident cards while retaining their indirect irradiance. `SurfaceCache.updateStatistics` reports the full-capture flag, captured card count and relit card count.

Radiosity initializes reset cards with direct lighting and emission, clears retired tiles, and dispatches its trace, filter, SH and integration stages for selected cards. Each selected card advances its own temporal phase. The sample budgets 32 cards per frame through `maxCardsPerFrame`. Mandatory resets are processed first and may exceed the budget. Any remaining budget schedules traces. `Radiosity.updateStatistics` reports updated resident cards (including resets), reset or retired slots, resident cards and the configured budget. The trace and SH outputs describe the slots traced by the current update. Both persistent passes require tracked render-frame submission.

For rigid receiver reprojection, enable `Deferred.exportLumenPreviousPosition` and `Temporal.dynamicReprojection`, and connect `Deferred.lumenPreviousPosition` to `Temporal.previousPosition`. Deferred reconstructs the previous world position from the raster hit and the previous instance transform. Temporal projects that point with the previous camera and checks each history tap against the stored depth and normal. Geometry motion can retain history for both moved and unchanged visible surfaces. Material and visibility changes, camera cuts and global illumination jumps reset diffuse history. Inspect `Temporal.historyWeight` to check rejection in newly exposed regions.

Sponza realtime, Sponza GI Comparison and Detail Tracing enable diffuse fast lighting updates. Set `fastLightingUpdate` on Gather and Temporal, enable `ScreenTrace.dynamicReprojection`, and connect `Deferred.lumenPreviousPosition` to both Gather and Temporal. Screen tracing reprojects an accepted HZB hit using its rigid displacement before testing previous depth. Mesh SDF hits reconstruct their previous world position from an instance transform in the same frame's GPU scene. Relative motion between the receiver and the hit uses UE 5.7.4's FP16 receiver speed and `0.005` threshold. Geometry history across a rigid transform is available to motion-aware diffuse Screen tracing. Camera cuts, topology changes, changed materials and changed visibility continue to invalidate it.

Gather scatters the ray motion into R8-equivalent gather bins, reduces the moving fraction, and widens the spatial filter above `0.01`. It interpolates the moving fraction with the same probe taps as indirect light. The diffuse alpha stores motion on valid receivers. Temporal uses the maximum of the current amount and the weighted reprojected amount to cap the frame count. `Temporal.fastUpdateAmount` records the current amount on pixels with eligible geometry history. `Gather.probeTraceMoving` and `Gather.probeMoving` expose the motion flags per ray and reduced fractions per active probe. Metallic obtains velocity from Deferred world positions and a Mesh SDF instance companion. Motion bins are quantized to the UE format in linear buffers, and the packed temporal state occupies the existing FP16 history alpha. This opaque software path takes its motion from accepted Screen and Mesh SDF hits.

The graph enables `stochasticInterpolation` on Gather. This uses UE's single-probe selection and unit sample weight with the rendering commandlet's black dummy noise input.

At the opening camera, the first four rays in `Trace.results` target the tall blocks in near-to-far order. The next four target the emitters. The second component of each ray's `state` identifies the GDF hit level. `GlobalDistanceFieldPages.statistics` reports page residency and overflow. To inspect recentering after a camera cut, move to eye `[156, 12.36, 20]` and look at `[156, 11.76, 0]` with a 60-degree vertical field of view. The far room is then covered by the finest clipmap. Inspect `Temporal.indirectDiffuse` for the scene output and `Temporal.historyWeight` for camera history.

### Lumen Reflections

**Lighting / Lumen / Reflections** (`lumen-reflections`) reuses the Large Scene graph and bakes. Its scene document gives the neutral plaster a mirror lobe and the blocks perceptual roughness 0.22. `Deferred.exportLumenSpecular` exports the integrated receiver response in RGB and roughness in alpha. `Reflections` samples the GDF and `Radiosity.lighting`, then composites the reflection over `Temporal.color`. The diffuse screen history is published before this composite.

The pass runs the UE 5.7.4 software GDF permutation for opaque isotropic single-lobe materials with the sky disabled. It generates one full-resolution ray per eligible pixel. Mirror rays use the reflected direction and rough rays use bounded GGX visible-normal sampling. `maxRoughness` defaults to 0.4 with a 0.1-wide fade. The sample takes radiance from the fixed Surface Cache and applies Metallic's integrated BRDF response. The permutation disables screen and Mesh SDF traces, the world radiance cache, and temporal and spatial denoisers. `Reflections.reflections` contains raw scene-linear RGB and the fade. `Reflections.trace` contains hit distance in meters, GDF classification, clipmap index and card-cache classification.

The UE port has the following adaptations.

- Diffuse reprojection transfers the previous world-space raster hit through an RGBA32F texture. UE transfers encoded screen velocity. Metallic uses camera and scene revisions, the Sun irradiance ratio and other lighting revisions to invalidate history. It uses its existing depth and normal tests for moved receivers. UE's hit-motion-driven fast lighting update is outside this path.
- Reflections read Metallic's sparse R8 GDF and fixed Surface Cache in meter-space world coordinates. They use the scene-linear ACEScg card radiance at unit pre-exposure. Ray direction and cone retain the reference FP16 rounding, and the maximum distance is rounded in FP16 centimeters before conversion to meters.
- The compositor multiplies by Metallic's integrated OpenPBR specular response, including receiver occlusion and view transmittance. UE evaluates its receiver specular response from its material and BRDF lookups. Metallic uses RGBA16F for raw radiance and fade, and a full-precision debug trace with meter distances and separate classifications.

### Lumen World Radiance Cache

**Lighting / Lumen / World Radiance Cache** (`lumen-radiance-cache`) uses the Large Scene geometry and Cornell Box mesh bakes. The sample tracks four world-space probe clipmaps with 48 cells per axis. They double in size from a 25 m half-extent. Each physical slot stores 32 × 32 equal-area directional samples. The sample uses a 32 × 32 slot atlas, a refresh budget equivalent to 100 full-resolution probe traces per frame, and a 100 m directional trace limit. New slots are initialized even when this requires tracing them at a reduced directional resolution.

The pass marks world probes from the screen probes placed for Final Gather. It remaps resident slots as clipmaps scroll, traces new slots before exposing them to Gather, and refreshes older slots in priority order. Directional traces read the same GDF and Surface Cache lighting as Gather. The cache filters adjacent probes using hit distances and fixes up the octahedral borders for bilinear sampling. `RadianceCache.updateState` words 24–29 count marked, reused, allocated, evicted and overflowed probes, followed by directional trace tiles. `RadianceCache.cacheRadiance` holds the bordered scene-linear atlas.

Under a manual Sun with HDRI and Moon disabled, direction edits and intensity ratios within `[0.25, 4]` keep resident probes eligible for scheduled refresh. The comparison uses the previous submitted lighting sample and starts from zero for a new scene. A global Sun change performs a cold cache update with four times the priority selection budget for that frame. The new-probe downsampling test continues to use the base `traceBudget`, as in UE 5.7.4. `RadianceCache.updateState` word 30 reports the effective priority budget in full-resolution probe equivalents, and word 31 reports the global Sun change flag. Geometry, visibility, lighting outside this manual Sun path, and environment revisions continue to invalidate history through the existing revision checks.

The Detail Tracing graph uses `LumenScreenProbePlacementPass` to publish one compact atlas before both World Radiance Cache and Gather execute. Enable `compactProbeAtlas` on both consumers. Wire the placement outputs `probePositions` and `probeAdaptiveCount` to both passes. Gather also reads `probeOrigins`, `probeTileAdaptiveCounts`, and `probeTileAdaptiveIndices`. A custom `adaptiveProbeAllocationFraction` must allocate the same capacity on Placement and Gather. The World Radiance Cache uses the producer's allocated capacity and skips invalid receivers. Its attempted adaptive count can exceed capacity, so the Mark dispatch clamps the adaptive suffix before reading probe positions.

Set `LumenProbeVisualizePass.compactProbeAtlas` and connect the positions, origins, tile counts and tile indices from Placement to visualize that layout. Its **Probe Slot** control continues to select the original candidate ordinal, and reports a missing slot when that candidate did not spawn. If `compactProbeAtlas` is omitted, the WRC sample can still supply the dense `RadianceCache.placedProbes` output to Gather and the probe visualizer.

Gather follows the UE 5.7.4 software `TraceVoxels` sampling condition. After any accepted Screen or Detail hit exits, valid WRC coverage caps the GDF trace and enables the cache sample with sphere parallax correction and trilinear interpolation. Remaining transparency weights the sample. A GDF miss, unavailable page with no later hit, or empty interval leaves transparency at 1. These rays publish `Gather.probeRadiance` alpha 6, including samples that return black. An opaque GDF hit has transparency 0 and keeps alpha 2. The readback still reports its cache sample.

The port follows the UE 5.7.4 software, opaque, skylight-off `FORCE_UNIFORM_TRACES` permutation with one radiance mip. It uses Metallic's perspective camera and meter-space world coordinates, current Deferred positions and normals, and the existing GDF and fixed Surface Cache. The X-packed indirection uses persistent graph buffers. Packed FP16 buffers hold directional radiance and hit distance, and the final atlas uses RGBA16F. Neighbor occlusion clamps atlas lookups to the current slot and treats a zero-length segment as visible. The angular filter skips a coincident hit that has no direction. A successfully submitted frame with matching scene, visibility, transforms and compatible lighting can supply history. The Sun test uses Metallic's maximum effective working-space RGB irradiance. UE's directional-light proxy color also includes alpha in its maximum and has a separate Sky Light test. Metallic uses the configured base budget directly, with the global Sun change multiplier. UE can additionally apply per-view `LightingUpdateSpeed` and an editing-view multiplier. UE's WRC coverage test uses clipmap membership at marked positions. Metallic also requires all eight neighboring probes to have resident atlas slots. UE marches using the coarse mip when a fine page is missing. Metallic treats an overflow page with a saturated coarse mip as unavailable and continues at the next clipmap. Metallic reports empty intervals explicitly and takes zero steps. UE's BRDF-PDF trace tiles, hardware tracing, skylight and alternate irradiance or mip configurations are outside this path.

### Lumen Detail Tracing

**Lighting / Lumen / Detail Tracing** (`lumen-detail-tracing`) uses the Large Scene and the same Cornell Box mesh bakes. Set `Gather.detailTracing` to `true`, supply `detailDistanceFieldPaths`, and connect `VBuffer.depth` to `Gather.detailDepth`. This option requires Final Gather and can run with screen traces or WRC disabled. `meshSDFTraceDistance` defaults to 1.8 m and is capped by `maxTraceDistance`.

The sample enables `activeProbeDispatch` and `splitProbeTracing`. Gather writes confirmed HZB hits, compacts eligible near-field rays into an indirect Mesh SDF dispatch, then compacts all remaining rays into an indirect GDF dispatch. Both software stages resume at `max(MinTraceDistance, SavedEnd - 2 * SurfaceBias)`. An unresolved Detail ray supplies its updated endpoint after the UE centimeter-space distance encode/decode. Rays that skip Detail use the screen endpoint. The per-view grid uses 64-pixel tiles, logarithmic depth slices, closest and furthest depth bounds, and the UE froxel and SDF intersection tests. `Gather.probeRadiance` alpha 9 identifies an accepted Mesh SDF hit. The 40 m view cutoff and distance fade follow UE 5.7.4. Mesh hits sample scene-linear Surface Cache lighting. WRC updates continue to trace the GDF.

This port covers opaque, uniformly scaled Mesh SDFs in perspective views with skylight disabled. Metallic builds the candidate grid with compute dispatches and reduces the current depth directly to the HZB interval inspected by the culler. The emissive exception to the object size test uses the material's emissive factor. The candidate pool reserves the full instance-cell upper bound and rejects configurations beyond its buffer limit. The stages use Metallic's baked finest SDF mip and fixed resident card tiles. Per-primitive draw-distance overrides, heightfields, hair, card feedback, and the UE moving-hit path are outside this permutation.

### Lumen Debug Views

Open **Lumen** in the viewport toolbar for **Cornell Box / Realtime GI**, **Cornell Box / GI Comparison**, **Lumen / Large Scene**, **Lumen / Detail Tracing**, or **Lumen / World Radiance Cache**. **Final Output** returns to the graph's presentation view.

| View | Reading the result |
| --- | --- |
| Lumen Scene | Scene-linear Surface Cache lighting at the software GDF hit. |
| Geometry Normals | World-space GDF normals encoded as `N * 0.5 + 0.5`. |
| Surface Cache | Scene lighting with magenta where the trace attempted card sampling and found zero coverage. |
| Probe Placement | Yellow uniform probes and magenta adaptive probes at their actual receiver pixels. |
| Probe Trace Source | Red for Screen, green for Mesh SDF, blue for GDF, and cyan for World Radiance Cache. Orange identifies unavailable GDF data, violet partial coverage, and magenta the step limit. Gray is a miss, dark gray an empty interval, and the checker an absent probe slot. |
| Raw Probe Radiance / Filtered Probe Radiance | Scene-linear directional radiance before and after the spatial filter. |
| Probe Hit Distance | Original Gather trace distances displayed over 0–4 m by default. Negative unresolved distances use the trace source color. |
| Final Gather / Temporal GI / History Weight | Current indirect diffuse, accumulated indirect diffuse, and the accepted history contribution as grayscale. |
| World Radiance Cache Atlas | The resident directional radiance atlas on graphs with a World Radiance Cache. |

The trace and radiance views use one directional atlas cell per 16 × 16 screen tile. **Probe Slot** selects the uniform probe or an adaptive slot in every tile. Raw traces use Gather's 8 × 8 or 16 × 16 lanes, and the filtered view uses its 8 × 8 directions. The optional `LumenProbeDebug.distanceRange` runtime setting changes the distance display range in meters.

The world views use the UE 5.7.4 geometry-normal remap and uncovered-card color from `LumenSurfaceCacheSampling.ush`, with Metallic camera rays, GDF tracing and fixed card tiles. The probe views adapt the `LumenScreenProbeDebug.usf` point colors and `LumenScreenProbeTracing.usf` trace colors to a screen overlay and directional atlases. Their additional colors expose Metallic trace classifications. The debug passes are preview sinks. The graph's presentation outputs drive the regular render, and a preview recompile restarts Temporal history.

### Lumen Probe Trace Readback

Set `Gather.probeTraceReadback` to `[tileX, tileY, tileWidth, tileHeight]` in a Final Gather graph. Each tile covers 16 × 16 render pixels. Choose a nonempty rectangle inside the current view grid whose records occupy at most 128 MiB. Removing the property disables the output and selects the regular tracing shader permutation.

Capture `Gather.probeTraceReadback` at `AfterPass` with layout `LumenProbeTraceReadback`. Each 48-byte record describes one direction before probe filtering. Records are ordered by selected tile in row-major order, then by probe slot (0–8), then by direction (0–63, X first). The index is `((localTileY * tileWidth + localTileX) * 9 + slot) * 64 + laneY * 8 + laneX`. For shared placement, match the logical slot through `ScreenProbePlacement.probeOrigins` to the packed receiver in `ScreenProbePlacement.probePositions`. In other graphs, inspect `RadianceCache.placedProbes` when WRC owns placement or `Gather.probePositions` when Gather owns placement.

| Bytes | Fields |
| --- | --- |
| 0–15 | `screenEnd`, `detailEnd`, `gdfMin`, `gdfMax` in meters |
| 16–31 | Raw scene-linear WRC RGB and `gdfHitDistance` in meters |
| 32–47 | Original GDF classification, hit clipmap, successful sample count, and flags |

An unused distance is `-1`. `screenEnd` records the decoded UE-style handoff distance. Screen tracing derives its normal bias from the probe pixel center and corner. It measures from the unshifted receiver to the reconstructed screen endpoint, adds 2 cm for a definite HZB miss, and encodes and decodes the centimeter distance with mask `0x7FFFFFFC` before converting it back to meters. The screen trace can return `0` when its history or projected start is invalid. Metallic keeps the raw distance for the WRC reach test and for the output of an accepted Screen hit. `detailEnd` includes the receiver-to-biased-origin offset. The GDF range is relative to the biased ray origin and includes empty intervals. The original GDF classification is preserved even when `Gather.probeRadiance` publishes cache class 6. A GDF invocation skipped by an early hit has classification `0xffffffff`, and any result without a GDF hit has clipmap `0xffffffff`.

Flag bits 0–7 indicate a valid receiver, screen trace invoked, screen hit accepted, Detail trace invoked, Detail hit accepted, GDF invoked, WRC coverage valid, and WRC sampled. `wrcSampled` records interpolation even if the sample returns black or an opaque GDF hit gives it zero weight. Bits 8 and 9 indicate a GDF prefix invocation and a selected prefix hit. Bits 10–13 contain the raw prefix classification when invoked. Bit 14 records any rejected receding expansion sample in the prefix, even if a later hit is accepted. Bits 15–17 indicate Mesh SDF verification invoked, confirmed, and inconclusive. The interval and step fields describe the selected GDF segment. When Mesh SDF confirms the prefix, the hit distance is the selected Mesh SDF distance. An unused probe slot produces only sentinels and zero flags.

## Optional RHI diagnostics

`METALLIC_RHI_DIAGNOSTICS` defaults to `OFF`. Enable it in a compatible existing
build tree to compile the same Vulkan barrier/submit observation hooks into the
shared render runtime, editor, and tests. It does not change public RHI object
layouts or enable capture by itself. `MetallicRHITests --tb-trace` installs the
bounded recorder; requesting it in an OFF build fails explicitly. See
[RHI Testbench](RhiTestbench.md#m4-诊断与合法序列) for trace A/B, sequence replay,
and shrinking commands. The property and shrinker tests do not require this flag.
