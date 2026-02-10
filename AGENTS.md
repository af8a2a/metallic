# Repository Guidelines

## Project Structure & Module Organization
- `Source/`: main application code (`main.cpp`) and rendering modules (`frame_graph.*`, mesh/material loaders, input, Metal bridges).
- `Shaders/`: Slang shader sources copied next to the executable at build time.
- `Asset/`: runtime assets (models/textures), also copied during post-build.
- `External/`: third-party dependencies (GLFW, ImGui, Tracy, meshoptimizer, Slang, MathLib). Treat as vendored code; avoid local edits unless dependency updates are intentional.
- `cmake/`: build helper scripts (including Slang download logic).
- Build outputs are typically in `build/` or IDE-specific `cmake-build-*` directories.

## Build, Test, and Development Commands
- Configure:
  - `cmake -S . -B build`
- Build:
  - `cmake --build build -j4`
- Clean rebuild (useful after shader/dependency changes):
  - `cmake --build build --clean-first -j4`
- Run app:
  - `./build/Source/Metallic`

Notes: this project targets macOS + Metal (`AppKit`, `.mm` files). `Source/CMakeLists.txt` handles shader/asset and Slang dylib copying automatically.

## Coding Style & Naming Conventions
- Language: C++20 (plus Objective-C++ for Metal/bridge files).
- Indentation: 4 spaces, no tabs.
- Naming:
  - Types/structs: `PascalCase` (e.g., `FrameGraph`, `FGResourceNode`)
  - Functions/variables: `lowerCamelCase` (e.g., `exportGraphviz`, `showGraphDebug`)
  - Files: lowercase with underscores when needed (e.g., `frame_graph.cpp`).
- Keep includes minimal and local; prefer small focused helper functions over long inline logic.

## Testing Guidelines
- No formal unit-test suite is present yet.
- Validate changes by:
  1. Successful clean build.
  2. Runtime smoke test of key paths (vertex, mesh shader, visibility buffer modes).
  3. For FrameGraph changes, verify ImGui graph view and optional DOT export (`G` key).
- If you add tests, place them under a new `tests/` folder and wire them through CMake.

## Commit & Pull Request Guidelines
- Commit style follows imperative, feature-focused subjects:
  - `Add ...`, `Fix ...`, `Switch ...`, `Replace ...`.
- Keep commits scoped (one behavior/theme per commit).
- PRs should include:
  - What changed and why.
  - Build/run validation steps performed.
  - Screenshots or short clips for UI/rendering changes.
  - Linked issue(s) when relevant.
