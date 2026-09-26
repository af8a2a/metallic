# Optional shader cache warmup

After configuring the project, run this target manually:

~~~powershell
cmake --build build-release --target MetallicShaderWarmup --config Release
~~~

Use your own build directory/configuration in place of the example. The target
builds the small MetallicShaderCompiler executable and warms the project's
.cache/shaders/spirv directory using the same SlangCompiler.cpp as the runtime.
It does not start Metallic, create a GPU device, load scenes, or build the editor.

Neither the tool nor the warmup target belongs to the default build. Metallic and
sample targets do not depend on them. Skipping warmup, deleting the cache, changing
shader sources, or encountering an unlisted variant leaves the original runtime
compile-on-cache-miss behavior intact.

Repeated invocations validate dependencies and reuse current entries. Compilation
errors or failure to read back a newly written cache entry produce a nonzero exit
code; other requests are still attempted. This warms SPIR-V, not driver PSO caches.

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
rendering, ReSTIR, conventional-texture path tracing and guides, SHARC/NRC path
tracing variants, and the default global-view reference/realtime deferred paths.
The realtime deferred variants include upscaler guides and opaque/transmission
StreamAsset paths. RTXCR sample requests are added only when its SDK is configured.

This is a bounded warmup set, not every combination of runtime properties.
Optional NTC and NRD SDK permutations, local-view deferred variants, and other
unlisted combinations continue to compile on demand. No cache is required to run.

When adding or changing requests, copy the owning runtime pass's module, entry,
capabilities, additional search paths, and macro definitions **in the same order**.
The runtime cache key includes all of them, even explicit zero-valued defines.
Compile include-file entry points through the same root module used by the pass.
The tool deliberately reuses runtime hashing, dependency validation, SPIR-V
processing, and cache serialization instead of duplicating that logic.
