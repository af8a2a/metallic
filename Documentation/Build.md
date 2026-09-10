# Faster Metallic builds

The normal CMake entry point keeps the existing feature defaults and source
dependencies. The presets require CMake 3.27+, Ninja, and (on Windows) an x64
Visual Studio Developer PowerShell/Command Prompt. Install Slang in
`External/slang`, or pass `-DSLANG_ROOT=<path>` when configuring the application.

## Daily development

```powershell
cmake --preset metallic-dev
cmake --build --preset metallic-dev --parallel 8
```

This builds the editor with glTF support. OpenUSD/oneTBB, NRD shader compilation,
NRC, NTC and Streamline are disabled; tests are off. USD files report an explicit
unsupported-build error. To run the scene, task and debug tests in this profile:

```powershell
cmake --preset metallic-ci
cmake --build --preset metallic-ci --parallel 8
ctest --preset metallic-ci
```

USD tests skip when USD is disabled; a separate test checks the disabled importer.
Closing optional integrations changes which render passes can run. Use the full
profile when working on those integrations.

## Full development with reusable dependencies

Build and install SDL3, spdlog, oneTBB and monolithic OpenUSD once:

```powershell
cmake --preset metallic-deps-debug
cmake --build --preset metallic-deps-debug
cmake --preset metallic-full
cmake --build --preset metallic-full --parallel 8
```

The dependency build installs automatically; no separate `cmake --install` is
needed. It uses up to eight compiler jobs, with one dependency project active at
a time to bound peak memory. Override `METALLIC_DEPENDENCY_JOBS` during configure
if needed. The first build still compiles all four libraries.

Installed packages live in `.cache/dependencies/<sha256>`, outside `build-full`.
Deleting or cleaning the application build therefore keeps the dependency
binaries. A clean full build imports those four libraries with `find_package`;
their C/C++ sources do not enter its build graph. Smaller libraries, GoogleTest,
and NRD/NTC when enabled still build from source. NRC/Streamline retain their
existing SDK binary integration.

The hash includes compiler identity/version/target, platform, Windows SDK,
CRT, configuration, compile/link flags, toolchain file, dependency Git revisions,
tracked and nonignored untracked changes, and the build recipes. A completion
manifest is written only after every dependency installs successfully. Re-running
the dependency configure/build with a matching completed package skips compilation.
Changing application code alone does not change this hash. Reconfigure both builds
after modifying a dependency checkout or toolchain. Builds for the same package
should run from one dependency build directory at a time.

`METALLIC_DEPENDENCY_CACHE=<absolute directory>` selects a shared cache, including
one outside this checkout. `METALLIC_DEPENDENCY_ROOT=<absolute package directory>`
selects a specific installed package; the consumer rejects mismatched manifests.
The application writes its expected inputs to
`<application-build>/metallic-dependencies-expected.txt` for comparison with the
package's `metallic-dependencies.txt` when diagnosing a mismatch. Equivalent
CMake booleans such as `ON` and `TRUE` produce the same key.
Use matching toolchain and configuration arguments for producer and consumer.
Prebuilt mode currently requires a single-config generator; Visual Studio's
multi-config generator continues to work in source mode.

Other configurations can use the standalone dependency project:

```powershell
cmake -S cmake/dependencies -B build-dependencies/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-dependencies/release
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DMETALLIC_DEPENDENCY_MODE=PREBUILT
cmake --build build-release --target Metallic --parallel 8
```

Both sides must use `-DMETALLIC_ENABLE_OPENUSD=OFF` for a package containing only
SDL3 and spdlog. The heavy submodules need initialization only in the dependency
producer/source profile; prebuilt consumers can identify uninitialized submodules
from the repository's Git locks. Small source/header dependencies still need their
usual checkouts (including SDL's vendored Vulkan headers used by volk).

To keep every dependency in the application's source build:

```powershell
cmake -S . -B build-source -DMETALLIC_DEPENDENCY_MODE=SOURCE
cmake --build build-source --target Metallic --config Debug --parallel 8
```

## Compiler cache

Install `sccache` on PATH, then add `-DMETALLIC_USE_SCCACHE=ON` to both application
and dependency configure commands. This is opt-in and supported with Ninja or
Makefiles. With MSVC, the launcher remains
`RunMsvcCompiler.cmake -> sccache -> cl.exe`, preserving localized `/showIncludes`
normalization. Debug information uses `/Z7` to avoid shared compile PDB contention.
An existing custom CMake launcher is preserved when sccache is off; conflicting
custom launchers are rejected when sccache is explicitly enabled.

Use `sccache --show-stats` to check actual hits. MSVC module compilation may bypass
the cache; dependency packages are the primary way to avoid those libraries being
rebuilt. See the [sccache documentation](https://github.com/mozilla/sccache) and
[CMake debug information settings](https://cmake.org/cmake/help/latest/variable/CMAKE_MSVC_DEBUG_INFORMATION_FORMAT.html).

Unity builds and PCH are not enabled globally: several vendors and module targets
require separate validation. CI artifact upload/download is also left to the
repository's eventual CI provider; the local package and manifest provide the
reusable output without introducing a remote service.

## Validation

Validated on Windows x64 with MSVC 19.51.36256 and CMake 4.2.1, Debug:

- Building the four dependency libraries into a fresh package took about 418 s
  with eight compiler jobs. Reconfiguring and building the completed package
  took about 1.8 s with no compilation. These timings cover the dependency stage.
- The full application compile database contains no source compilations for
  SDL3, spdlog, oneTBB or OpenUSD.
- `metallic-full` built the editor and scene/task/debug tests. Those tests passed,
  including USD, USDC and embedded-texture USDZ imports. The optional Super Sponza
  fixture remained environment-gated.
- The lightweight scene/task/debug tests, nine cache identity/rejection tests,
  and editor Vulkan smoke checks in both profiles passed.
- An MSVC build with a custom launcher preserved the launcher and recorded header
  dependencies in Ninja; changing the header scheduled recompilation. Actual
  sccache cache hits were not measured because sccache was not installed.
