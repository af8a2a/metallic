# Repository Guidelines

## Project Structure & Module Organization
- `Source/`: primary runtime code. `main.cpp` drives the Apple Metal renderer; `main_vulkan.cpp` is the Windows Vulkan bootstrap.
- `Source/Rendering/`: frame graph, pass registration, shader management, render passes, and shared frame/scene context.
- `Source/RHI/`: backend-agnostic interfaces plus `Metal/` and `Vulkan/` implementations.
- `Source/PipelineEditor/`: shared pipeline asset, registry, and builder code used by both the renderer and tools.
- `Tools/PipelineEditor/`: standalone ImGui/OpenGL pipeline editor target.
- `Shaders/`: Slang and MSL shader sources copied next to built binaries.
- `Pipelines/`: JSON pipeline assets such as `visibility_buffer.json`, `meshlet_debug.json`, and `vulkan_preview.json`.
- `Asset/`: runtime models and textures; copied post-build on Apple targets.
- `External/`: vendored dependencies (GLFW, ImGui, Tracy, Slang, VMA, MathLib, etc.). Avoid local edits unless a dependency update is intentional.
- `cmake/`: helper scripts, including Slang SDK setup and download logic.
- Build outputs usually live in `build/` or IDE-specific `cmake-build-*` directories.

## Build, Test, and Development Commands
- Configure: `cmake -S . -B build`  
  If `External/slang` is unavailable, pass `-DSLANG_ROOT=<sdk-root>` or set `SLANG_ROOT` / `SLANG_DIR`. Windows also requires a Vulkan SDK.
- Build renderer: `cmake --build build --target Metallic --config Debug -j4`
- Build standalone editor: `cmake --build build --target PipelineEditorTool --config Debug -j4`
- Clean rebuild: `cmake --build build --target Metallic --config Debug --clean-first -j4`
- Run renderer: `build/Source/Metallic` on single-config generators, or `build/Source/Debug/Metallic(.exe)` on multi-config generators.
- Run editor: `build/Tools/PipelineEditor/PipelineEditorTool` on single-config generators, or `build/Tools/PipelineEditor/Debug/PipelineEditorTool(.exe)` on multi-config generators.

Notes: macOS builds the full Metal renderer and copies `Shaders/`, `Asset/`, and `Pipelines/` beside the app. Windows currently targets the Vulkan bootstrap renderer plus the standalone pipeline editor.

## Coding Style & Naming Conventions
- Language: C++20; use Objective-C++ only for Apple bridge/runtime files.
- Indentation: 4 spaces, no tabs.
- Naming:
  - Types/structs/classes: `PascalCase`
  - Functions/variables: `lowerCamelCase`
  - Constants: follow existing `kCamelCase` style where present
  - Files: lowercase with underscores (for example `frame_graph.cpp`, `pass_registrations_vulkan.cpp`)
- Keep includes minimal and local; prefer small helper functions over long monolithic blocks.
- Keep shared logic in `Rendering/`, `RHI/`, or `PipelineEditor/`; isolate backend-specific code under `RHI/Metal`, `RHI/Vulkan`, or Apple-only `Platform/*.mm`.

## Testing Guidelines
- No dedicated unit-test suite exists yet; validate changes with builds and runtime smoke tests.
- Minimum validation for renderer changes: successful clean build, app launch, and verification of the touched backend path.
- Metal path: sanity-check vertex, mesh shader, visibility buffer, and meshlet debug modes; verify `F5` shader reload, `F6` pipeline reload, and `G` frame-graph DOT export when relevant.
- Vulkan path: verify the preview pipeline JSON loads, the diagnostic triangle/post stack renders, and the ImGui overlay remains responsive.
- Pipeline asset/editor changes: open `PipelineEditorTool`, load and save the affected JSON under `Pipelines/`, and confirm the renderer still accepts it.
- If you add automated tests, place them under a new `tests/` directory and wire them through CMake.

## Commit & Pull Request Guidelines
- Use imperative, feature-focused subjects such as `Add ...`, `Fix ...`, `Refactor ...`, `Switch ...`, or `Replace ...`.
- Keep commits scoped to one subsystem or behavior.
- PRs should describe what changed, why it changed, and which platform/backend was validated.
- Include build/run validation steps and screenshots or clips for UI/rendering changes.
- Link relevant issues and call out dependency, shader, pipeline, or asset updates explicitly.
