# Metallic

A Metal rendering playground exploring modern GPU-driven techniques on Apple Silicon.
Almost all code is **generated** by `Claude Opus 4.6` and `OpenAI GPT-5.3 Codex` — pushing the limits of vibe-coding.

Developed on Apple M4 Pro.

## Features

### Rendering Modes
- **Visibility Buffer** — Mesh shader writes packed meshlet/triangle/instance IDs to R32Uint, followed by compute deferred lighting with PBR materials
- **Mesh Shader Forward** — Meshlet-based forward rendering with per-meshlet frustum and backface cone culling
- **Legacy Vertex Forward** — Traditional vertex shader forward pass

### Techniques
- **Raytraced Shadows** — Metal raytracing API with per-frame TLAS rebuild, shadow rays traced from depth buffer
- **Atmospheric Scattering** — Precomputed transmittance, scattering, and irradiance sky rendering
- **Tonemapping** — 6 methods: Filmic, Uncharted2, ACES, AgX, Khronos PBR Neutral, Clip
- **Meshlet Culling** — Frustum culling and backface cone culling per meshlet
- **Scene Graph** — Node hierarchy with transforms, directional light, and ImGui inspector
- **FrameGraph** — Declarative render pass graph with Graphviz DOT export

### Infrastructure
- **Slang Shaders** — Written in Slang, compiled to Metal source at runtime (except raytracing: native MSL)
- **Shader Hot-Reload** — Press F5 to recompile all shaders without restarting
- **metal-cpp** — Header-only C++ wrapper for Metal API
- **Tracy Profiler** — GPU and CPU profiling zones (on-demand, zero overhead when disconnected)
- **Dear ImGui** — Docking-enabled UI with per-pass controls
- **spdlog** — Structured logging

## Building

Requires macOS with Apple Silicon and Xcode command line tools.

```bash
git clone --recursive <repo-url>
cd rendergraph

# Configure (downloads Slang compiler automatically)
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build -j$(sysctl -n hw.ncpu)

# Run
./build/Source/Metallic
```

## Project Structure

```
Source/
├── main.cpp                    # Entry point, render loop
├── Platform/                   # Metal/GLFW/ImGui/Tracy bridges
├── Asset/                      # glTF mesh, meshlet, material loading
├── Scene/                      # Scene graph + ImGui inspector
└── Rendering/
    ├── frame_graph.h/cpp       # Declarative render pass graph
    ├── render_pass.h/cpp       # Base RenderPass class
    ├── raytraced_shadows.h/cpp # BLAS/TLAS management
    └── Passes/                 # Modular render passes

Shaders/
├── Vertex/                     # Vertex pipeline shaders (Slang)
├── Mesh/                       # Mesh shader pipeline (Slang)
├── Visibility/                 # Visibility buffer + deferred lighting (Slang)
├── Atmosphere/                 # Sky rendering (Slang)
├── Post/                       # Tonemapping (Slang)
└── Raytracing/                 # Shadow rays (native Metal)

External/                       # Dependencies (metal-cpp, GLFW, Slang, ImGui, Tracy, etc.)
```

## Dependencies

| Library | Purpose |
|---|---|
| [metal-cpp](https://developer.apple.com/metal/cpp/) | C++ Metal API wrapper |
| [GLFW](https://www.glfw.org/) | Window management |
| [Slang](https://shader-slang.com/) | Shader language → Metal codegen |
| [Dear ImGui](https://github.com/ocornut/imgui) | Debug UI |
| [Tracy](https://github.com/wolfpld/tracy) | GPU/CPU profiler |
| [meshoptimizer](https://github.com/zeux/meshoptimizer) | Meshlet building |
| [cgltf](https://github.com/jkuhlmann/cgltf) | glTF parsing |
| [stb_image](https://github.com/nothings/stb) | Image loading |
| [spdlog](https://github.com/gabime/spdlog) | Logging |
| [MathLib](External/MathLib/) | HLSL-style math |
