# Tracy GPU profiling

## Ready-to-use local setup

On Windows, reload CMake in CLion and select the **MetallicTracy** executable
target to build and run the launcher. With no arguments it connects to
`127.0.0.1:8086`. Set the run configuration's working directory to the repository
root when using relative capture paths.

```powershell
cmake --build build-dev --target MetallicTracy
./build-dev/Tools/MetallicTracy.exe
./build-dev/Tools/MetallicTracy.exe --capture ./Captures/tracy-gpu-smoke.tracy
./build-dev/Tools/MetallicTracy.exe --with-metallic
```

Program arguments also accept `--address HOST --port PORT`, or a capture filename
as the sole argument. `--with-metallic` starts a new editor from the launcher's
**current CMake build configuration**, so build the `Metallic` target there first.
The launcher waits for Viewer to exit and returns its exit code; closing Viewer
does not close the editor.

The launcher builds independently of the editor and Viewer. It uses the existing
Release Viewer at `build-tracy-viewer/tracy-profiler.exe` (or its `Release/`
subdirectory). To use another matching Viewer, configure
`-DMETALLIC_TRACY_VIEWER=C:/path/to/tracy-profiler.exe` and rebuild the launcher.
Viewer is still built separately as described below. This keeps its dependency
graph and build configuration separate from Metallic's.

The PowerShell launcher remains available:

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

## Timing coverage and limitations

Native RenderGraph statistics support both self-submission through
`execute(RenderGraphSubmitDesc)` and frame-tracked external
`execute(CommandBuffer&)`. Graphics, Compute and Copy have separate query pools;
completed executions expose frame, node and instrumented-section
`gpuMilliseconds` / `gpuTimingAvailable`. Raw external recordings without a valid
`RenderFrameContext` completion keep CPU scopes only. These are implemented
capabilities, not a claim that every queue/device has been validated.

Tracy publication is narrower than native statistics. The adapter publishes a
GPU frame only when its frame timer and **all node timers** are graphics timers.
An execution with a compute/copy node timer therefore has no Tracy GPU frame,
even if native statistics contain valid GPU milliseconds for those nodes.
Nested section timers are neither tested by that predicate nor exported as
Tracy GPU zones; internal async work is not thereby a calibrated multi-queue
Tracy timeline. Queue submits remain visible as CPU zones.

The frame interval uses graphics timestamps. Self-submission starts it in the
first graphics segment and ends it in the graphics epilogue after joining graph
segments, including subsystem prologue/epilogue work when present. The external
path starts after the subsystem prologue and includes the epilogue. Neither path
includes editor UI or presentation. Do not add per-queue node durations to infer
frame duration, critical-path time or a synchronized multi-queue timeline.
See [completion and timing lifetime](RenderFrameLifetime.md#gpu-timing-and-completion)
and [the publication predicate](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L1828-L1949).

The Vulkan RHI optionally enables `VK_EXT_calibrated_timestamps` and samples the
device clock with Windows QPC or Linux monotonic raw time. Tracy uses these samples
to correct CPU/GPU correlation and drift. Without the extension/host time domain,
the context is labeled `(uncalibrated)`: GPU durations remain valid but the
placement relative to CPU events is approximate. Calibration errors never prevent
rendering. The RHI sample also retains the driver's maximum deviation in nanoseconds.

Metallic owns three reusable timing slots, with separate query pools per queue
and slot. The bounded query budget includes two frame queries, two per pass and
additional space for nested scopes / parallel recordings, shared with native
statistics. Both interval boundaries use bottom-of-pipe timestamps. Completed
submissions are resolved later with completion and availability checks, without
a profiling wait or queue-idle call.

Collection requires device `timestampQueries`, a graphics timestamp pool and a
valid frame completion; each timed queue also requires nonzero
`timestampValidBits` and a successfully created pool. If all slots are pending or
still in flight, that execution is not sampled. Exhausting the interval budget
or section capacity sets `profilingOverflow`; it does not fail rendering or
turn an unavailable timing into a measured zero. See
[query allocation and resolution](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp#L1770-L1949).

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
cmake --build build-dev --target MetallicRHITests
build-dev/tests/MetallicRHITests.exe --gtest_filter=*gpu_clock_calibration*:*gpu_profiling*:*timestamp_query*
```
