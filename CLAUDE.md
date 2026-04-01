# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Configure** (pass `-DSLANG_ROOT=<path>` if `External/slang` is missing; Windows needs Vulkan SDK):
```bash
cmake -S . -B build
```

**Build renderer:**
```bash
# macOS (single-config)
cmake --build build --target Metallic -j$(sysctl -n hw.ncpu)
./build/Source/Metallic

# Windows / MSVC (multi-config)
cmake --build build --config Debug --target Metallic
build/Source/Debug/Metallic.exe
```

**Build pipeline editor tool:**
```bash
cmake --build build --config Debug --target PipelineEditorTool
```

**Clean rebuild:**
```bash
cmake --build build --target Metallic --config Debug --clean-first
```

No automated test suite exists. Validate with: clean build → app launch → verify touched rendering path works. Runtime controls: `F5` shader reload, `F6` pipeline reload, `G` export framegraph DOT.

## Architecture

### Platform Split
- **macOS**: Metal renderer via `main.cpp`. Platform bridge code in `Source/Platform/*.mm` (Objective-C++).
- **Windows**: Vulkan 1.4 renderer via `main_vulkan.cpp`. Backend in `Source/RHI/Vulkan/`.
- Platform ifdefs: `#if APPLE` / `#ifdef _WIN32`. Resource utils and shader utils use `#ifdef __APPLE__` / `#ifdef _WIN32` dual paths.

### RHI Layer (`Source/RHI/`)
Abstract device/pipeline/buffer/texture/sampler interfaces in `rhi_backend.h`. Backend implementations:
- `RHI/Metal/metal_frame_graph.cpp` — Metal command encoding
- `RHI/Vulkan/vulkan_backend.cpp` — Vulkan device, swapchain
- `RHI/Vulkan/vulkan_frame_graph.cpp` — VMA-backed resources, command buffers, encoders
- `RHI/Vulkan/vulkan_descriptor_manager.cpp` — Bindless descriptors (set 0=storage buffers, set 1=sampled images, set 2=samplers)
- `RHI/Vulkan/vulkan_resource_state_tracker.h` — Hazard-aware barrier system

### Frame Graph (`Source/Rendering/frame_graph.h/cpp`)
Templated pass-based DAG. Passes declare resource reads/writes via `FGBuilder`; the graph handles lifetime, aliasing, and history slots for temporal effects. Pass types: Render, Compute, Blit.

### Data-Driven Pipeline System
Pipeline graphs are JSON files in `Pipelines/` (schema v2). `PipelineEditor/pipeline_asset.cpp` loads them; `pass_registry.cpp` maps pass type strings to factory functions. Passes register via `REGISTER_PASS` / `REGISTER_RENDER_PASS` / `REGISTER_COMPUTE_PASS` macros.

### Render Passes (`Source/Rendering/Passes/`)
Base class in `render_pass.h` with `setup()`, `executeRender/Compute/Blit()`, and `configure()`. Passes are platform-agnostic; the RHI layer handles backend differences.

### Shader Compilation (`Source/Rendering/slang_compiler.cpp`)
Slang SDK compiles shaders to MSL (Metal) or SPIR-V (Vulkan). Reflection extracts binding layouts. Mesh shader and visibility buffer sources get backend-specific patches. Exception: Metal ray-traced shadows use native MSL (`Shaders/Raytracing/raytraced_shadow.metal`).

### Asset Pipeline
glTF loading via cgltf → meshlet generation via meshoptimizer → cluster LOD hierarchy → GPU buffer upload. Key files: `Source/Asset/mesh_loader.cpp`, `meshlet_builder.cpp`, `cluster_lod_builder.cpp`.

## Coding Conventions
- C++20. Objective-C++ only for `Platform/*.mm` files.
- 4 spaces, no tabs.
- Types: `PascalCase`. Functions/variables: `lowerCamelCase`. Constants: `kCamelCase`. Files: `lowercase_underscores`.
- Keep shared logic in `Rendering/`, `RHI/`, or `PipelineEditor/`. Backend-specific code goes under `RHI/Metal/`, `RHI/Vulkan/`, or `Platform/`.

## Key Implementation Details
- VMA requires `VK_API_VERSION_1_3` (doesn't support 1.4 yet). `VMA_IMPLEMENTATION` defined only in `vulkan_frame_graph.cpp`.
- Per-frame command pools (not shared) to avoid reset conflicts.
- Mesh shader extension functions loaded dynamically via `vkGetDeviceProcAddr`.
- Push constants: 256 bytes, all stages. Shared pipeline layout from descriptor manager used for all pipelines.
- `VulkanOwnedTexture` is outside anonymous namespace to enable `dynamic_cast` in `beginRenderPass`.
- Global setup functions `vulkanSetResourceContext()` and `vulkanSetShaderContext()` must be called at startup.
- Slang SDK auto-downloaded by CMake if missing (`cmake/SetupSlang.cmake`).
- meshoptimizer v1.0 built via isolated `External/meshoptimizer_build/CMakeLists.txt` to separate `/RTC1` flag removal.

## Commit Style
Imperative, feature-focused: `Add ...`, `Fix ...`, `Refactor ...`, `Replace ...`. Scope commits to one subsystem.
