
# Metallic Repository Guidelines

## Find the Relevant Context

Metallic is a C++23 / Slang Vulkan renderer with an SDL3 / ImGui editor. Use these entry points as needed for the task; no full-repository reading pass is required.

- Architecture and subsystem boundaries: `Documentation/ProjectArchitecture.md`. Runtime code lives in `Source/Runtime/{Scene,Render,Task,Debug}/`; editor code in `Source/Editor/`.
- Build variants, SDKs, and dependency reuse: `Documentation/Build.md` and `CMakePresets.json`.
- Shader modules and interop conventions: `Shaders/README.md`. Programs live in `Shaders/Features/`, shared modules in `Shaders/Modules/`, and GPU test probes in `tests/rhi/shaders/`.
- Render graph assets: `Pipelines/`; sample content: `Asset/`; test targets and registration: `tests/CMakeLists.txt`.
- Codex Only: For complex tasks that can be independently split, please use sub agents for parallel analysis, implementation, or validation as needed, without requiring Ultra thinking intensity. The main agent is responsible for integrating the results and final verification; Simple tasks do not require splitting.


## Build and Test

Run commands from the repository root. Windows x64 / MSVC is the main validated platform; Ninja presets need an x64 Visual Studio developer shell with `cl` available. Reuse a compatible configured build directory; do not change its compiler, generator, or SDK options to make an unrelated task build. Initialize missing dependencies with `git submodule update --init --recursive -- <submodule-path>`. Use `-DSLANG_ROOT=<path>` for an alternate Slang installation.

Common starting points (configure only when needed):

```powershell
# Editor, Ninja Debug; this preset disables tests.
cmake --preset metallic-dev
cmake --build --preset metallic-dev
.\build-dev\Source\Metallic.exe --smoke-test

# Scene/task/debug tests without the optional SDK configuration.
cmake --preset metallic-ci
cmake --build --preset metallic-ci
ctest --test-dir build-ci -R '^Metallic(Scene|Task|Debug)Tests$' --output-on-failure
```

For other configured trees, use `cmake --build <build-dir> --target <target>` and `ctest --test-dir <build-dir> -R '<test-name-regex>' --output-on-failure`. Multi-config generators also need the matching `--config Debug` / `-C Debug`, and place executables under a configuration subdirectory. CTest does not build tests; build every selected executable first. The `metallic-full` build preset does not build every registered test target.

Tests use GoogleTest through CTest. Main targets include `MetallicSceneTests`, `MetallicTaskTests`, `MetallicDebugTests`, `MetallicGPUPageTests`, and `MetallicRHITests`. Use `--gtest_list_tests` and `--gtest_filter=<pattern>` on test executables for focused runs. RHI tests also accept legacy `--list` / `--filter`; unsupported GPU capabilities can produce skipped tests, which are not runtime validation of that path. Follow the local test conventions and register new sources in `tests/CMakeLists.txt`.

`MetallicShaderWarmup` is an optional manual target; keep it outside default builds and editor/sample dependencies.

## Coding Style & Naming Conventions

The project uses C++23 through CMake. Match the existing style: 4-space indentation, no tabs, Allman braces for function definitions, same-line braces for control statements, and namespace end comments such as `} // namespace metallic::render`. Use `PascalCase` for types, `lowerCamelCase` for functions and locals, `kPascalCase` for constants, and trailing underscores for private members. Use `PascalCase` for C++ source, header, and shader file names, such as `TaskGraph.cpp`, `TaskGraph.h`, and `ScenePathTrace.slang`; existing legacy C++ file names do not need opportunistic renaming.

When applying PascalCase to identifiers and file names, use these exceptions:

- Preserve the established capitalization of technical acronyms and domain terms with specific meanings, such as `CPU`, `GPU`, `BSDF`, `BRDF`, and `BTDF`; use names such as `CPUBuffer`, `GPUPage`, `BSDFSample`, and `BRDFEvaluation`, not `CpuBuffer`, `GpuPage`, `BsdfSample`, or `BrdfEvaluation`. When renaming identifiers or files to PascalCase, restore acronym segments such as `Bsdf` to their original spelling (`BSDF`), including within compound names.
- Preserve the official capitalization of explicitly integrated external components, such as `RTXDI` and `RTXCR`; use names such as `RTXDIIntegration` and `RTXCRPass`, not `RtxdiIntegration` or `RtxcrPass`.

Keep official mixed-case spellings such as `glTF`, `sRGB`, and `scRGB`, including at the beginning of a name; examples include `glTFInstance`, `RGBA8sRGB`, and `HDRscRGB`.

## Change and Validation Boundaries

- Preserve unrelated working-tree changes. Avoid broad rewrites in `External/` or vendor shader snapshots; retain their licenses. Treat `build/`, `build-*`, `cmake-build-*`, `Testing/`, `.cache/`, and IDE folders as local output, not source.
- For shared render/RHI interface changes, inspect affected runtime callers and tests, including render graph and Vulkan implementations. For shader resource/layout changes, keep CPU bindings, Slang declarations, and affected pipeline assets consistent.
- Match validation to the changed behavior. For a bug fix, cover the failing case where practical; for documentation-only edits, check facts, links, and the diff without rebuilding the renderer. Broaden testing when affected contracts cross subsystem boundaries.
- For rendering changes, exercise the affected scene/pass and inspect its output when the GPU environment permits. A build, shader compile, or one-frame smoke test alone does not establish visual correctness, temporal stability, or full-scene memory behavior. State which runtime checks were unavailable.
- For performance work, compare the same workload and settings, record cache/warmup conditions and the measured scope, and retain correctness checks. Report pass timings as pass timings, not end-to-end speedups. Keep generated captures and images out of source control unless intentionally included as evidence.

## Commit & Pull Request Guidelines

Use short imperative commit subjects and keep commits scoped to one subsystem. PRs should explain the resulting behavior, list validation commands and outcomes (including skips or unavailable checks), link relevant issues, and provide screenshots or output images for visual changes when available.

## Slang Module Ownership

Every repository-owned `.slang` file, including feature entrypoints, SDK wrappers, generated Slang sources, and test shaders, must declare exactly one `module Name;` or `implementing Name;`. Shared code uses `import`. A split implementation uses `__include` only from its owning module and declares the same `implementing` owner. Do not create anonymous include-only Slang fragments or use `#include` on a `.slang` file. Give independent entrypoints their own modules and keep registered entrypoint names stable.

Third-party HLSL headers and shared C++/Slang interop headers may retain textual `#include` inside their canonical module owner. Do not convert authored Slang snippets to `.hlsli` to evade module ownership. Local preprocessor definitions do not cross `import` or `__include`; expose dependencies through explicit imports, resource parameters, generics, or interfaces. Existing global shader permutations remain in `SlangShaderDesc::macroDefines` and shader cache identity.

Run `python -B Tools/CheckShaderModules.py` (or CTest `MetallicShaderModuleOwnershipAudit` when Python is available) after changing shader structure, followed by compilation and behavior-appropriate validation. The audit covers `Shaders/` and all test Slang sources; external vendor source trees are excluded.
