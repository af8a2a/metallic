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
- `Source/glfw_metal_bridge.mm` — Objective-C++ bridge: attaches CAMetalLayer to GLFW's NSWindow
- `Source/glfw_metal_bridge.h` — C-linkage header for the bridge function
- `Shaders/triangle.slang` — Vertex/fragment shaders in Slang (compiled to Metal at runtime)

### Dependencies

| Dependency | Location | Type |
|---|---|---|
| metal-cpp | `External/metal-cpp/` | Header-only, CMake INTERFACE lib |
| GLFW | `External/glfw/` | Git submodule |
| Slang | `External/slang/` | Auto-downloaded binary, .gitignored |

After cloning, run `git submodule update --init` to fetch GLFW.

### CMake structure

- Root `CMakeLists.txt` — Project config, includes `cmake/DownloadSlang.cmake`, finds Slang package
- `External/CMakeLists.txt` — metal-cpp INTERFACE lib (links Metal/Foundation/QuartzCore frameworks), GLFW config
- `Source/CMakeLists.txt` — `rendergraph` executable, links metal-cpp + glfw + slang + AppKit. Post-build copies Slang dylibs and Shaders/ to build dir.

## Conventions

- Shaders are written in Slang (not MSL directly) and compiled to Metal source code at runtime via the Slang API
- Platform bridging (Cocoa/Metal layer) lives in `.mm` files with C-linkage headers
- Metal objects use manual reference counting (retain/release)
