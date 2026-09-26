
# Metallic Repository Guidelines

## Find the Relevant Context

Metallic is a C++23 / Slang Vulkan renderer with an SDL3 / ImGui editor. Use these entry points as needed for the task; no full-repository reading pass is required.

- Architecture and subsystem boundaries: `Documentation/ProjectArchitecture.md`. Runtime code lives in `Source/Runtime/{Scene,Render,Task,Debug}/`; editor code in `Source/Editor/`.
- Build variants, SDKs, and dependency reuse: `Documentation/Build.md` and `CMakePresets.json`.
- Shader modules and interop conventions: `Shaders/README.md`. Programs live in `Shaders/Features/`, shared modules in `Shaders/Modules/`, and GPU test probes in `tests/rhi/shaders/`.
- Render graph assets: `Pipelines/`; sample content: `Asset/`; test targets and registration: `tests/CMakeLists.txt`.

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

Tests use GoogleTest through CTest. Main targets include `MetallicSceneTests`, `MetallicTaskTests`, `MetallicDebugTests`, `MetallicGpuPageTests`, and `MetallicRhiTests`. Use `--gtest_list_tests` and `--gtest_filter=<pattern>` on test executables for focused runs. RHI tests also accept legacy `--list` / `--filter`; unsupported GPU capabilities can produce skipped tests, which are not runtime validation of that path. Follow the local test conventions and register new sources in `tests/CMakeLists.txt`.

`MetallicShaderWarmup` is an optional manual target; keep it outside default builds and editor/sample dependencies.

## Coding Style & Naming Conventions

The project uses C++23 through CMake. Match the existing style: 4-space indentation, no tabs, Allman braces for function definitions, same-line braces for control statements, and namespace end comments such as `} // namespace metallic::render`. Use `PascalCase` for types, `lowerCamelCase` for functions and locals, `kPascalCase` for constants, and trailing underscores for private members. Use `PascalCase` for C++ source, header, and shader file names, such as `TaskGraph.cpp`, `TaskGraph.h`, and `ScenePathTrace.slang`; existing legacy C++ file names do not need opportunistic renaming.

## Change and Validation Boundaries

- Preserve unrelated working-tree changes. Avoid broad rewrites in `External/` or vendor shader snapshots; retain their licenses. Treat `build/`, `build-*`, `cmake-build-*`, `Testing/`, `.cache/`, and IDE folders as local output, not source.
- For shared render/RHI interface changes, inspect affected runtime callers and tests, including render graph and Vulkan implementations. For shader resource/layout changes, keep CPU bindings, Slang declarations, and affected pipeline assets consistent.
- Match validation to the changed behavior. For a bug fix, cover the failing case where practical; for documentation-only edits, check facts, links, and the diff without rebuilding the renderer. Broaden testing when affected contracts cross subsystem boundaries.
- For rendering changes, exercise the affected scene/pass and inspect its output when the GPU environment permits. A build, shader compile, or one-frame smoke test alone does not establish visual correctness, temporal stability, or full-scene memory behavior. State which runtime checks were unavailable.
- For performance work, compare the same workload and settings, record cache/warmup conditions and the measured scope, and retain correctness checks. Report pass timings as pass timings, not end-to-end speedups. Keep generated captures and images out of source control unless intentionally included as evidence.

## Commit & Pull Request Guidelines

Use short imperative commit subjects and keep commits scoped to one subsystem. PRs should explain the resulting behavior, list validation commands and outcomes (including skips or unavailable checks), link relevant issues, and provide screenshots or output images for visual changes when available.

## Shader Normal/TBN Pitfall

For ray-query scene shaders, keep `HitInfo.normal` and `HitInfo.geometryNormal` as stable authored/world-space data when building tangent space and applying normal maps. Do not flip either field toward the current ray inside `traceClosest()` before tangent/bitangent construction; doing so makes the TBN basis view/path dependent and can reintroduce hard color bands or seam-like artifacts in normal visualization and path tracing. If a BSDF needs a same-hemisphere normal, apply a separate face-forward step only to the final shading normal after normal map evaluation, and keep `geometryNormal` available for front-face tests and ray offsets.
