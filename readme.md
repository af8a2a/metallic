# Metallic

Metallic is a rendering playground for modern GPU-driven rendering experiments.

- macOS uses the original Metal renderer with scene loading, mesh shaders, visibility-buffer rendering, ray-traced shadows, atmosphere, and an ImGui debug UI.
- Windows uses the newer backend-agnostic RHI path with a Vulkan 1.4 preview renderer and shared pipeline asset system.
- A standalone `PipelineEditorTool` lets you inspect and edit pipeline JSON graphs outside the renderer.

Most of the codebase is authored with heavy assistance from `Claude Opus 4.6` and `OpenAI GPT-5.4`.

## Backend Status

| Platform | Backend | Status |
|---|---|---|
| macOS (Apple Silicon) | Metal | Main renderer path |
| Windows | Vulkan 1.4 | Bootstrap / preview path |

Linux is not a supported target right now.

## Features

### Metal renderer
- Scene loading and rendering for Sponza
- Vertex forward, mesh shader forward, visibility buffer, and meshlet debug modes
- Meshlet frustum culling, backface cone culling, and GPU-driven culling
- Ray-traced shadows on the Metal path
- Atmospheric scattering sky rendering
- TAA, auto exposure, tonemapping, and per-pass debug UI
- FrameGraph execution with optional Graphviz DOT export

### Shared infrastructure
- Backend-agnostic RHI layer under `Source/RHI/`
- Data-driven pipeline assets in `Pipelines/*.json`
- Slang shader compilation for Metal and Vulkan targets
- Runtime shader reload (`F5`) and pipeline asset reload (`F6`) on the Metal renderer
- Tracy profiling and spdlog logging

### Windows Vulkan preview
- GLFW + Vulkan bootstrap through the shared RHI
- Slang-to-SPIR-V compilation at startup
- Preview post stack driven by `Pipelines/vulkan_preview.json`
- Sky preview path when atmosphere textures and the sky shader are available
- Triangle fallback path when preview resources are unavailable

### Pipeline editor tool
- Standalone ImGui + imnodes graph editor
- Loads and saves pipeline assets from `Pipelines/`
- Visualizes pass/resource dependencies and validates the DAG

## Repository Layout

```text
Source/
  main.cpp                     Metal renderer entry point
  main_vulkan.cpp              Windows Vulkan preview entry point
  Rendering/                   Frame graph, passes, shader management, frame context
  RHI/                         Shared interfaces and backend implementations
  PipelineEditor/              Shared pipeline asset, registry, and builder code
  Platform/                    Apple-specific runtime and bridge code
  Scene/                       Scene graph and inspector UI
  Asset/                       Mesh, meshlet, and material loading

Tools/
  PipelineEditor/              Standalone pipeline editor executable

Shaders/                       Slang and native MSL shader sources
Pipelines/                     JSON pipeline assets
Asset/                         Runtime textures, models, and atmosphere data
External/                      Vendored third-party dependencies
cmake/                         Build helpers, including Slang setup
```

## Building

Clone with submodules:

```bash
git clone --recursive <repo-url>
cd metallic
```

If you already cloned without submodules, sync the profiler dependency with:

```bash
git submodule update --init --recursive External/microprofile
```

If `External/slang` is missing, CMake will try to use:

1. `-DSLANG_ROOT=<sdk-root>`
2. `SLANG_ROOT`
3. `SLANG_DIR`
4. `External/slang`

### macOS / Metal

Requirements:
- Apple Silicon Mac
- Xcode command line tools

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --target Metallic -j$(sysctl -n hw.ncpu)
./build/Source/Metallic
```

Notes:
- The macOS build copies `Shaders/`, `Asset/`, and `Pipelines/` next to the executable.
- The standalone pipeline editor is also available on macOS:

```bash
cmake --build build --target PipelineEditorTool -j$(sysctl -n hw.ncpu)
./build/Tools/PipelineEditor/PipelineEditorTool
```

### Windows / Vulkan

Requirements:
- Visual Studio 2022 or another CMake-capable MSVC toolchain
- Vulkan SDK
- A Slang SDK if `External/slang` is not populated

```powershell
cmake -S . -B build -DSLANG_ROOT=D:/path/to/slang
cmake --build build --config Debug --target Metallic
build/Source/Debug/Metallic.exe
```

Standalone editor on Windows:

```powershell
cmake --build build --config Debug --target PipelineEditorTool
build/Tools/PipelineEditor/Debug/PipelineEditorTool.exe
```

Notes:
- The Windows renderer is currently a preview path, not feature parity with the Metal renderer.
- It exercises the shared RHI, shader compiler, frame graph, and pipeline asset system.

## Runtime Controls

Metal renderer:
- `F5`: rebuild shaders
- `F6`: reload pipeline JSON assets
- `G`: export `framegraph.dot`
- ImGui renderer panel: switch render modes and toggle culling, TAA, RT shadows, and sky settings

## Pipeline Assets

Pipeline graphs live in `Pipelines/` and are described as JSON:

- `resources`: declared textures/buffers
- `passes`: pass type, inputs, outputs, enabled state, side-effect flag, and pass config

Current in-tree examples include:
- `Pipelines/forward.json`
- `Pipelines/meshlet_debug.json`
- `Pipelines/visibility_buffer.json`
- `Pipelines/vulkan_preview.json`

The shared loader validates missing resources, duplicate producers, and graph cycles before execution.

## Dependencies

| Library | Purpose |
|---|---|
| `metal-cpp` | Metal C++ wrapper on macOS |
| Vulkan SDK + VMA | Windows Vulkan backend |
| GLFW | Window creation and input |
| Dear ImGui | Debug UI |
| imnodes | Pipeline editor node graph UI |
| Slang | Shader authoring and code generation |
| microprofile | Lightweight frame scope profiling |
| Tracy | CPU/GPU profiling |
| spdlog | Logging |
| nlohmann/json | Pipeline asset serialization |
| meshoptimizer | Meshlet generation and culling support |
| cgltf / stb_image | Asset loading |
| MathLib | HLSL-style math helpers |
