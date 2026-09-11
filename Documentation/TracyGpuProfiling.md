# Tracy GPU profiling

## Ready-to-use local setup

`build-tracy-viewer/tracy-profiler.exe` is the matching Tracy Viewer. From the
repository root, use:

```powershell
# Connect to an already running Metallic instance (127.0.0.1:8086).
./scripts/StartTracy.ps1

# Start the development build of Metallic and connect the viewer.
./scripts/StartTracy.ps1 -WithMetallic

# Open the verified GPU capture.
./scripts/StartTracy.ps1 -Capture ./Captures/tracy-gpu-smoke.tracy
```

You can also double-click `scripts/Tracy.cmd` to connect, or drop a `.tracy`
file onto it to open that capture. Viewer preferences use Tracy's normal Windows
user configuration directory. The existing `build-tracy-check/tracy-capture.exe`
and `tracy-csvexport.exe` provide command-line capture and export.

## Build and integration

Metallic exports the editor's graphics RenderGraph GPU timings to Tracy. Build
with `METALLIC_ENABLE_TRACY=ON` (the default); `TRACY_ON_DEMAND=ON` records Tracy
metadata only while a viewer is connected. The existing editor CPU/GPU tables
continue to work independently.

```powershell
git submodule update --init --recursive -- External/tracy
cmake -S . -B build-dev -DMETALLIC_ENABLE_TRACY=ON
cmake --build build-dev --target Metallic
```

Run the editor and connect Tracy to `localhost:8086`. Use a viewer built from
**this checkout's `External/tracy` commit**, since capture protocol versions must
match (this checkout reports Tracy 0.13.2). To build it from a VS developer shell:

```powershell
cmake -S External/tracy/profiler -B build-tracy-viewer -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /EHsc /utf-8"
cmake --build build-tracy-viewer --config Release
```

The MSVC UTF-8 option is needed for Tracy's Unicode UI strings on Chinese Windows.
If using Ninja, make sure `ninja.exe` is also on the developer shell's `PATH`,
since Tracy builds its font embedding helper as a separate CMake project.
The viewer build fetches its own dependencies. For command-line capture, build
`External/tracy/capture` in the same way and run:

```powershell
tracy-capture -a localhost -o Captures/metallic.tracy
```

In the viewer, inspect `Metallic Graphics / RenderGraph`. Each `RenderGraph Frame`
contains the executed passes, named using their graph instance and pass type.
CPU `Editor Frame`, `RenderGraph Record`, `Queue Submit`, and wait zones plus editor frame marks help
relate command recording/submission to GPU work. GPU durations measure timestamp
intervals in milliseconds; CPU recording duration is a separate measurement.

`RenderGraph Frame` brackets the pass loop, including resource transitions
between passes. It does not include editor UI, presentation, or subsystem GPU
prologue/epilogue work. This first integration covers
`RenderGraphExecutor::execute(CommandBuffer&)`, used by the editor and preview
renderer. `execute(RenderGraphSubmitDesc)` and its async compute/copy submissions
are not yet GPU-profiled; their queue submits still appear on the CPU timeline.

The Vulkan RHI optionally enables `VK_EXT_calibrated_timestamps` and samples the
device clock with Windows QPC or Linux monotonic raw time. Tracy uses these samples
to correct CPU/GPU correlation and drift. Without the extension/host time domain,
the context is labeled `(uncalibrated)`: GPU durations remain valid but the
placement relative to CPU events is approximate. Calibration errors never prevent
rendering. The RHI sample also retains the driver's maximum deviation in nanoseconds.

Metallic owns the timestamp query pool and its three reusable slots. There are
two frame queries and two queries per pass, shared with the editor statistics.
Both boundaries use bottom-of-pipe timestamps, matching Tracy's Vulkan timing
convention and keeping serial pass timestamps ordered. Completed submissions are
resolved later with availability checks, without a profiling wait or queue-idle
call. If all slots are busy, timing for that execution is skipped.

Tracy GPU events are published as complete frames in execution order with their
original CPU recording times and thread IDs. Cancelled/incomplete recordings and
frames spanning viewer connections are discarded. The adapter is isolated in
`Source/Runtime/Render/Profiling/TracyProfiler.cpp`; because delayed events require
Tracy's internal queue format, verify captures when updating the Tracy submodule.
Contexts persist across graph recompiles. Tracy's process-wide context limit is
respected; new profiler contexts are skipped if its ID space is exhausted.

Disable the integration with `-DMETALLIC_ENABLE_TRACY=OFF` (or the existing
`-DTRACY_ENABLE=OFF`). This removes the Tracy client linkage/instrumentation and
leaves native GPU timing available.

Focused validation:

```powershell
cmake -S . -B build-dev -DMETALLIC_BUILD_TESTS=ON
cmake --build build-dev --target MetallicRhiTests
build-dev/tests/MetallicRhiTests.exe --gtest_filter=*gpu_clock_calibration*:*gpu_profiling*:*timestamp_query*
```
