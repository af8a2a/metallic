# Repository Guidelines

## Project Structure & Module Organization
`Source/` holds the runtime: `Rendering/` contains the frame graph and passes, `RHI/` contains backend-agnostic interfaces plus `Metal/` and `Vulkan/`, `PipelineEditor/` contains shared pipeline asset code, and `Asset/`, `Scene/`, and `Platform/` cover loading, scene state, and platform bridges. `Tools/PipelineEditor/` builds the standalone editor. `Shaders/` stores Slang/MSL sources, `Pipelines/` stores JSON graphs, `Asset/` stores runtime content, and `External/` vendors third-party libraries. Treat `build/`, `build-*`, `cmake-build-*`, `.idea/`, and `.cache/` as local outputs.

## Build, Test, and Development Commands
- `cmake -S . -B build` configures the project. Pass `-DSLANG_ROOT=<sdk>` if `External/slang` is absent; Windows also requires a Vulkan SDK.
- `cmake --build build --target Metallic --config Debug` builds the renderer.
- `cmake --build build --target PipelineEditorTool --config Debug` builds the standalone pipeline editor.
- `cmake --build build --target Metallic --config Debug --clean-first` forces a clean renderer rebuild.
- Run single-config builds from `build/Source/Metallic`; MSVC builds from `build/Source/Debug/Metallic.exe`. The editor lives under `build/Tools/PipelineEditor/...`.

## Coding Style & Naming Conventions
C++20 is the default; use Objective-C++ only in `Source/Platform/*.mm`. Use 4-space indentation and no tabs. Match existing naming: `PascalCase` for types, `lowerCamelCase` for functions and locals, `kCamelCase` for constants, and lowercase underscore file names such as `frame_graph.cpp` and `vulkan_backend.cpp`. Keep shared logic in `Rendering/`, `RHI/`, or `PipelineEditor/`; isolate backend-specific code under `RHI/Metal`, `RHI/Vulkan`, or `Platform/`.

## Testing Guidelines
There is no dedicated in-tree unit test suite yet, so every change needs a build plus a smoke test on the touched path. Renderer changes should launch `Metallic`, load the relevant pipeline, and verify shader or pipeline reload (`F5` / `F6`) when affected. Pipeline asset changes should round-trip through `PipelineEditorTool` and reopen cleanly. If you add automated tests, place them under a top-level `tests/` directory and wire them into CMake.

## Commit & Pull Request Guidelines
Recent history uses imperative, feature-focused subjects such as `Refactor Vulkan ...`, `Fix ...`, and `Replace ...`. Keep commits scoped to one subsystem or behavior. Pull requests should explain what changed, why it changed, and which platform/backend you validated. Include screenshots for UI or rendering work, link related issues when relevant, and call out shader, pipeline JSON, asset, or dependency updates explicitly.
