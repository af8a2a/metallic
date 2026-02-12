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
./build/Source/Metallic
```

The first configure will download Slang compiler (v2026.1.2) automatically to `External/slang/`.

## Architecture

Metal rendering project using C++20 on Apple Silicon (M4 Pro, AppleClang 17).

**Rendering pipeline:** GLFW window → Objective-C++ bridge attaches CAMetalLayer → metal-cpp for GPU commands → Slang shaders compiled to Metal source at runtime.

### Key source files

- `Source/main.cpp` — Entry point, Metal device/queue setup, Slang shader compilation, render loop
- `Source/Platform/metal_impl.cpp` — Sole translation unit for metal-cpp `*_PRIVATE_IMPLEMENTATION` macros (NS, MTL, CA). **These must only be defined once across all TUs** to avoid duplicate symbols. Do not add these defines to any other .cpp file.
- `Source/Platform/glfw_metal_bridge.mm/h` — Objective-C++ bridge: attaches CAMetalLayer to GLFW's NSWindow
- `Source/Platform/imgui_metal_bridge.h/mm` — ImGui Metal/GLFW integration bridge
- `Source/Platform/tracy_metal.h/mm` — Tracy Metal GPU profiling bridge (ObjC++ with ARC)
- `Source/Asset/mesh_loader.h/cpp` — glTF mesh loading via cgltf (positions, normals, UVs, primitive groups)
- `Source/Asset/meshlet_builder.h/cpp` — Meshlet building with per-meshlet material IDs (uses meshoptimizer)
- `Source/Asset/material_loader.h/cpp` — PBR material + texture loading from glTF (stb_image)
- `Source/Scene/scene_graph.h/cpp` — Node hierarchy, transforms, components
- `Source/Scene/scene_graph_ui.h/cpp` — ImGui scene inspector
- `Source/Rendering/camera.h` — Header-only orbit camera (simd/simd.h)
- `Source/Rendering/input.h/cpp` — GLFW mouse/scroll input callbacks
- `Source/Rendering/frame_graph.h/cpp` — FrameGraph system for declarative render pass management with Graphviz DOT export
- `Source/Rendering/render_pass.h/cpp` — Base `RenderPass` class with `RenderContext`, Tracy profiling zones, and per-pass ImGui UI
- `Source/Rendering/render_uniforms.h` — Shared uniform structs (Uniforms, ShadowUniforms, LightingUniforms, AtmosphereUniforms, TonemapUniforms)
- `Source/Rendering/visibility_constants.h` — Shared visibility buffer bit-packing constants (CPU + Slang)
- `Source/Rendering/raytraced_shadows.h/cpp` — BLAS/TLAS building, per-frame TLAS update, shadow ray compute pipeline creation
- `Source/Rendering/Passes/` — Modular render pass implementations:
  - `forward_pass.h` — Vertex/mesh shader forward rendering
  - `visibility_pass.h` — Visibility buffer mesh shader pass (R32Uint output)
  - `shadow_ray_pass.h` — Raytraced shadow compute pass
  - `sky_pass.h` — Atmospheric scattering sky rendering
  - `deferred_lighting_pass.h` — Compute deferred lighting from visibility buffer
  - `tonemap_pass.h` — Tonemapping post-process (Filmic, Uncharted2, ACES, AgX, Khronos PBR, Clip)
  - `blit_pass.h` — Blit compute output to drawable
  - `imgui_overlay_pass.h` — ImGui overlay rendering
- `Source/Rendering/pass_registrations.cpp` — Pass metadata registration for pipeline editor
- `Source/PipelineEditor/` — Standalone library for data-driven pipeline editing:
  - `pass_registry.h/cpp` — Factory pattern with `REGISTER_PASS_INFO` macro for pass metadata
  - `pipeline_asset.h/cpp` — JSON pipeline schema, load/save, DAG validation
  - `pipeline_builder.h/cpp` — Constructs FrameGraph from PipelineAsset
  - `pipeline_editor.h/cpp` — ImGui node graph editor (using imnodes)

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
| spdlog | `External/spdlog/` | Git submodule, header-only logging |
| nlohmann/json | `External/nlohmann/` | Single-header JSON library |
| imnodes | `External/imnodes/` | Node graph editor for ImGui |

After cloning, run `git submodule update --init` to fetch GLFW, Tracy, and spdlog.

### CMake structure

- Root `CMakeLists.txt` — Project config, includes `cmake/DownloadSlang.cmake`, finds Slang package
- `External/CMakeLists.txt` — metal-cpp INTERFACE lib (links Metal/Foundation/QuartzCore frameworks), GLFW, imnodes, nlohmann_json
- `Source/CMakeLists.txt` — `Metallic` executable, links PipelineEditor + rendering libs. Post-build copies Slang dylibs, Shaders/, Pipelines/, and Asset/ to build dir.
- `Source/PipelineEditor/CMakeLists.txt` — `PipelineEditor` static library for data-driven pipeline editing

### Shaders

Slang shaders are compiled to Metal source at runtime. Raytracing shaders are native Metal (Slang doesn't support Metal raytracing):
- `Shaders/Vertex/triangle.slang` — Basic vertex/fragment triangle shader
- `Shaders/Vertex/bunny.slang` — MVP + Blinn-Phong lit shader (vertex pipeline)
- `Shaders/Mesh/meshlet.slang` — Mesh shader + fragment shader for meshlet rendering
- `Shaders/Visibility/visibility.slang` — Visibility buffer mesh + fragment shader (R32Uint output)
- `Shaders/Visibility/deferred_lighting.slang` — Compute shader for deferred lighting from visibility buffer (samples shadow map at texture 99)
- `Shaders/Atmosphere/sky.slang` — Atmospheric scattering sky rendering (precomputed transmittance/scattering/irradiance textures)
- `Shaders/Post/tonemap.slang` — Fullscreen tonemapping post-process (6 methods)
- `Shaders/Raytracing/raytraced_shadow.metal` — **Native Metal** compute shader: traces shadow rays against TLAS, writes R8Unorm shadow map

## Conventions

- Shaders are written in Slang (not MSL directly) and compiled to Metal source code at runtime via the Slang API
- **Exception:** Raytracing shaders (`Shaders/Raytracing/raytraced_shadow.metal`) are native Metal Shading Language, compiled at runtime with `MTL::LanguageVersion3_1`. Slang does not support Metal raytracing.
- Platform bridging (Cocoa/Metal layer) lives in `Source/Platform/*.mm` files with C-linkage headers
- Metal objects use manual reference counting (retain/release)
- Single-header libs use `#define *_IMPLEMENTATION` in exactly one TU: `CGLTF_IMPLEMENTATION` in `Asset/mesh_loader.cpp`, `STB_IMAGE_IMPLEMENTATION` in `Asset/material_loader.cpp`
- ObjC++ files (`.mm`) that need ARC get `-fobjc-arc` via `set_source_files_properties` in CMake
- Rendering has 3 modes: Vertex pipeline, Mesh shader, Visibility buffer (deferred lighting)
- Visibility buffer pipeline: Visibility Pass → Shadow Ray Pass → Sky Pass → Deferred Lighting → Tonemap → ImGui
- Forward pipeline: Sky Pass → Forward Pass → Tonemap → ImGui
- Render passes are modular classes inheriting from `RenderPass` (see `Source/Rendering/Passes/`)
- Shader hot-reload via F5 key
- Pipeline hot-reload via F6 key (reloads JSON pipeline definition)
- Use `spdlog` for all logging (`spdlog::info`, `spdlog::warn`, `spdlog::error`). Do not use `std::cout`/`std::cerr`.

### Data-Driven Pipeline System

The rendering pipeline can be defined in JSON files (inspired by EA's Gigi):

- `Pipelines/visibility_buffer.json` — Visibility buffer deferred rendering pipeline
- `Pipelines/forward.json` — Forward rendering pipeline

**Key components (in `Source/PipelineEditor/`):**
- `pass_registry.h/cpp` — Factory pattern with registration macros
- `pipeline_asset.h/cpp` — JSON schema, load/save, DAG validation
- `pipeline_builder.h/cpp` — Constructs FrameGraph from PipelineAsset
- `pipeline_editor.h/cpp` — ImGui node graph editor (using imnodes)

**Pass Registration Macros:**
Since C++ lacks static reflection, passes must be registered with metadata using macros in `pass_registrations.cpp`:

```cpp
// Register pass metadata for the pipeline editor
REGISTER_PASS_INFO(VisibilityPass, "Visibility Pass", "Geometry",
    (std::vector<std::string>{}),                        // default inputs
    (std::vector<std::string>{"visibility", "depth"}),   // default outputs
    PassTypeInfo::Type::Render);

REGISTER_PASS_INFO(ShadowRayPass, "Shadow Ray Pass", "Lighting",
    (std::vector<std::string>{"depth"}),
    (std::vector<std::string>{"shadowMap"}),
    PassTypeInfo::Type::Compute);
```

Categories: `Geometry`, `Lighting`, `Environment`, `Post-Process`, `Utility`, `UI`

**Pipeline JSON schema:**
```json
{
  "name": "PipelineName",
  "resources": [
    { "name": "visibility", "type": "texture", "format": "R32Uint", "size": "screen" }
  ],
  "passes": [
    {
      "name": "Pass Name",
      "type": "VisibilityPass",
      "inputs": ["depth"],
      "outputs": ["visibility", "depth"],
      "enabled": true,
      "sideEffect": false,
      "config": {}
    }
  ]
}
```

**Special resources:** `$backbuffer` refers to the drawable texture.

**View menu:** Pipeline Editor opens the visual node graph editor for modifying pipelines at runtime. Features include:
- Blue nodes for resources, orange nodes for passes
- Drag to pan, scroll to zoom, minimap in corner
- Click nodes to edit properties in the side panel
- Delete key removes selected nodes
- Add menu shows passes grouped by category



## Slang → Metal Gotchas

- Slang `mul(M, v)` generates `v * M` in Metal. Since `simd_float4x4` is column-major but Slang expects row-major, `simd_transpose()` matrices before passing as uniforms.
- Slang doesn't emit `[[user(...)]]` on mesh output struct members — must patch generated Metal source to add them.
- Slang doesn't emit `[[texture(N)]]` on `array<texture2d<...>, N>` parameters — must patch generated source.
- Slang wraps globals into a `KernelContext` struct passed to ALL entry points. Fragment shader expects all mesh-stage buffers bound even if unused.
- Global `ConstantBuffer<T>` gets `[[buffer(0)]]` — vertex buffers must start at index 1+ to avoid conflicts.

## Raytracing Shadows

- **BLAS** (one per glTF mesh): built once at startup from `LoadedMesh::meshRanges` / `primitiveGroups`. Each primitive group becomes a triangle geometry descriptor.
- **TLAS** (one instance per visible scene node with a mesh): rebuilt every frame via `updateTLAS()` to track scene graph transform changes. Instance descriptor buffer is `StorageModeShared` for CPU writes.
- **Shadow ray shader** (`raytraced_shadow.metal`): native Metal compute, reads depth buffer, reconstructs world position via `invViewProj`, traces toward directional light using `intersector<triangle_data, instancing>` with `accept_any_intersection(true)`.
- **Matrix convention:** Native Metal shader uses standard column-major multiplication — do NOT `transpose()` the `invViewProj` (unlike Slang shaders which need transposed matrices).
- **Reversed-Z:** Shadow shader receives a `reversedZ` uniform to correctly detect sky pixels (depth == 0.0 for reversed-Z, depth == 1.0 for normal-Z).
- **Texture binding:** Shadow map is `R8Unorm` at texture index 99 in `deferred_lighting.slang`. Slang correctly emits `[[texture(99)]]` for scalar `Texture2D<float>` — no patching needed (unlike texture arrays).
- **CAMetalLayer:** `setFramebufferOnly(false)` is required because the visibility buffer path blits compute output to the drawable.

## Development
- when I start debug, I will add the issue content(log/screenshot) in `Issue` directory.You can get bug information from it.
-