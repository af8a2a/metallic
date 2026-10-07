# Shader cache warmup

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
at 4 to limit simultaneous Slang compiler memory use. --jobs N sets an explicit
positive worker count; --jobs 1 restores serial compilation. The actual worker
count is capped by the number of selected requests.

~~~powershell
.\build-release\Source\MetallicShaderCompiler.exe --jobs 8
cmake -S . -B build-release "-DMETALLIC_SHADER_WARMUP_ARGS=--jobs;8"
cmake --build build-release --target MetallicShaderWarmup
~~~

CMake's --parallel option controls the build, not shader compiler workers.
Requests compile with independent Slang sessions. Progress reports completed
requests/total, percentage, cache hits, failures, elapsed time, and the completed
shader name. Completion order may differ from request order. Errors are counted
per request and remaining requests are still processed before the final summary.
The progress percentage measures request count, not estimated remaining time.

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
Optional NTC and NRD SDK permutations, local-view deferred variants, and other
unlisted combinations continue to compile on demand. No cache is required to run.

When adding or changing requests, copy the owning runtime pass's module, entry,
capabilities, additional search paths, and macro definitions **in the same order**.
The runtime cache key includes all of them, even explicit zero-valued defines.
For native requests it also includes both descriptor strides.
Compile include-file entry points through the same root module used by the pass.
The tool deliberately reuses runtime hashing, dependency validation, SPIR-V
processing, and cache serialization instead of duplicating that logic.
