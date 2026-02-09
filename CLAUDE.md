# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

cmake and ninja are not on PATH. Use CLion-bundled tools:

```bash
# Configure
/Users/af8a2a/Applications/CLion.app/Contents/bin/cmake/mac/aarch64/bin/cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Build
/Users/af8a2a/Applications/CLion.app/Contents/bin/ninja/mac/aarch64/ninja -C build

# Run
./build/Source/rendergraph
```

The first configure will download Slang compiler (v2026.1.2) automatically to `External/slang/`.

## Architecture

Metal rendering project using C++20 on Apple Silicon (M4 Pro, AppleClang 17).

**Rendering pipeline:** GLFW window → Objective-C++ bridge attaches CAMetalLayer → metal-cpp for GPU commands → Slang shaders compiled to Metal source at runtime.

### Key source files

- `Source/main.cpp` — Entry point, Metal device/queue setup, Slang shader compilation, render loop
- `Source/metal_impl.cpp` — Sole translation unit for metal-cpp `*_PRIVATE_IMPLEMENTATION` macros (NS, MTL, CA). **These must only be defined once across all TUs** to avoid duplicate symbols. Do not add these defines to any other .cpp file.
- `Source/frame_graph.h/cpp` — FrameGraph system for declarative render pass management with Graphviz DOT export
- `Source/mesh_loader.h/cpp` — glTF mesh loading via cgltf (positions, normals, UVs, primitive groups)
- `Source/meshlet_builder.h/cpp` — Meshlet building with per-meshlet material IDs (uses meshoptimizer)
- `Source/material_loader.h/cpp` — PBR material + texture loading from glTF (stb_image)
- `Source/camera.h` — Header-only orbit camera (simd/simd.h)
- `Source/input.h/cpp` — GLFW mouse/scroll input callbacks
- `Source/tracy_metal.h/mm` — Tracy Metal GPU profiling bridge (ObjC++ with ARC)
- `Source/imgui_metal_bridge.h/mm` — ImGui Metal/GLFW integration bridge
- `Source/glfw_metal_bridge.mm/h` — Objective-C++ bridge: attaches CAMetalLayer to GLFW's NSWindow

### Dependencies

| Dependency | Location | Type |
|---|---|---|
| metal-cpp | `External/metal-cpp/` | Header-only, CMake INTERFACE lib |
| GLFW | `External/glfw/` | Git submodule |
| Slang | `External/slang/` | Auto-downloaded binary, .gitignored |
| Dear ImGui | `External/imgui/` | Static lib, Metal+GLFW backends |
| Tracy | `External/tracy/` | Git submodule, on-demand profiler |
| meshoptimizer | `External/meshoptimizer/` | Static lib, meshlet clustering |
| cgltf | `External/cgltf/cgltf.h` | Single-header glTF parser |
| stb_image | `External/stb/` | Single-header image loader |
| MathLib | `External/MathLib/` | HLSL-style math library |

After cloning, run `git submodule update --init` to fetch GLFW and Tracy.

### CMake structure

- Root `CMakeLists.txt` — Project config, includes `cmake/DownloadSlang.cmake`, finds Slang package
- `External/CMakeLists.txt` — metal-cpp INTERFACE lib (links Metal/Foundation/QuartzCore frameworks), GLFW config
- `Source/CMakeLists.txt` — `rendergraph` executable, links metal-cpp + glfw + slang + AppKit. Post-build copies Slang dylibs and Shaders/ to build dir.

### Shaders

All shaders are Slang, compiled to Metal source at runtime:
- `Shaders/triangle.slang` — Basic vertex/fragment triangle shader
- `Shaders/bunny.slang` — MVP + Blinn-Phong lit shader (vertex pipeline)
- `Shaders/meshlet.slang` — Mesh shader + fragment shader for meshlet rendering
- `Shaders/visibility.slang` — Visibility buffer mesh + fragment shader (R32Uint output)
- `Shaders/deferred_lighting.slang` — Compute shader for deferred lighting from visibility buffer

## Conventions

- Shaders are written in Slang (not MSL directly) and compiled to Metal source code at runtime via the Slang API
- Platform bridging (Cocoa/Metal layer) lives in `.mm` files with C-linkage headers
- Metal objects use manual reference counting (retain/release)
- Single-header libs use `#define *_IMPLEMENTATION` in exactly one TU: `CGLTF_IMPLEMENTATION` in `mesh_loader.cpp`, `STB_IMAGE_IMPLEMENTATION` in `material_loader.cpp`
- ObjC++ files (`.mm`) that need ARC get `-fobjc-arc` via `set_source_files_properties` in CMake
- Rendering has 3 modes: Vertex pipeline, Mesh shader, Visibility buffer (deferred lighting)



## Slang → Metal Gotchas

- Slang `mul(M, v)` generates `v * M` in Metal. Since `simd_float4x4` is column-major but Slang expects row-major, `simd_transpose()` matrices before passing as uniforms.
- Slang doesn't emit `[[user(...)]]` on mesh output struct members — must patch generated Metal source to add them.
- Slang doesn't emit `[[texture(N)]]` on `array<texture2d<...>, N>` parameters — must patch generated source.
- Slang wraps globals into a `KernelContext` struct passed to ALL entry points. Fragment shader expects all mesh-stage buffers bound even if unused.
- Global `ConstantBuffer<T>` gets `[[buffer(0)]]` — vertex buffers must start at index 1+ to avoid conflicts.



## Development
- when I start debug, I will add the issue content(log/screenshot) in `Issue` directory.You can get bug information from it.
- 