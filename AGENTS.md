
# Repository Guidelines

## Project Structure & Module Organization
`Source/` contains the application and runtime code. `Source/Editor/` hosts the editor shell, `Source/Runtime/Scene/` handles scene loading, and `Source/Runtime/Render/` contains Slang compilation, render graph code, render passes, and the Vulkan RHI backend under `GAPI/Vulkan/`. `Shaders/` stores Slang sources, `Pipelines/` stores JSON render graph assets, and `Asset/` contains sample content. `tests/scene/` and `tests/rhi/` build CTest executables. `External/` vendors dependencies; avoid changes there unless intentional. Treat `build/`, `build-*`, `cmake-build-*`, `Testing/`, `.cache/`, and IDE folders as local output.

## Build, Test, and Development Commands
- `git submodule update --init --recursive` initializes vendored dependencies such as microprofile.
- `cmake -S . -B build -DMETALLIC_BUILD_TESTS=ON` configures the project with tests enabled. Provide `-DSLANG_ROOT=<path>` if Slang is not available under `External/slang`.
- `cmake --build build --target Metallic --config Debug` builds the main executable.
- `cmake --build build --target MetallicSceneTests --config Debug` builds a focused test target.
- `ctest --test-dir build -C Debug --output-on-failure` runs all tests. Add `-L scene` or `-L rhi` for subsets.
- `build\Source\Debug\Metallic.exe --smoke-test` runs a quick Windows/MSVC smoke check after building.

## Coding Style & Naming Conventions
The project uses C++23 through CMake. Match the existing style: 4-space indentation, no tabs, Allman braces for function definitions, same-line braces for control statements, and namespace end comments such as `} // namespace metallic::render`. Use `PascalCase` for types, `lowerCamelCase` for functions and locals, `kPascalCase` for constants, trailing underscores for private members, and snake_case file names such as `render_graph.cpp`.

## Testing Guidelines
Tests are custom CTest executables, not a third-party unit framework. Scene tests use simple `expect` helpers and return nonzero on failure. RHI tests are registry-based, support `--list` and `--filter <text>`, and use exit code `77` for unsupported environments. Add tests under `tests/scene/` or `tests/rhi/`, wire them in `tests/CMakeLists.txt`, and keep generated images out of source control.

## Commit & Pull Request Guidelines
Recent commits use short imperative subjects such as `Add runtime scene browser and tests`, `Refactor frame graph resource handling`, and `Integrate volk for Vulkan loading`. Keep commits scoped to one subsystem. Pull requests should describe the change, list validation commands, link issues, and include screenshots or output images for editor, shader, or rendering changes.

## Agent-Specific Instructions
Avoid broad rewrites in `External/` and generated build directories. Before changing shared render or RHI interfaces, inspect runtime callers and tests so behavior stays consistent across scene, render graph, and Vulkan paths.
