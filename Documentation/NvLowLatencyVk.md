# NvLowLatencyVk / NVIDIA Reflex

Metallic uses Streamline's Reflex and PCL plugins to drive `NvLowLatencyVk.dll`
on Windows/Vulkan. `sl.common` already owns the SDK device and sleep semaphore;
Metallic does not initialize a second low-latency device. This works with ordinary
render graphs as well as DLSS-SR/RR.

`METALLIC_ENABLE_NV_LOW_LATENCY` defaults to `ON`. It requires a usable Streamline
SDK, `sl_reflex.h`, `sl_pcl.h`, `sl.reflex.dll`, `sl.pcl.dll`, and
`NvLowLatencyVk.dll`. CMake deploys those plugins beside the executable only when
available and enabled. Missing optional Reflex files leave DLSS available.
Disabling Streamline also disables this integration.
`NvLowLatencyVk.dll` itself is required by `sl.common`, including when Reflex is
disabled, so an SDK missing that shared dependency is treated as unusable.

`VK_NV_low_latency2` is an alternative backend, not an additional switch required
by this integration. Streamline prefers it when its device entry points are
enabled, but Metallic currently uses the SDK backend. A native extension path
would also need present-ID features, low-latency swapchain creation and frame
attribution; it must not introduce a second sleep/marker stream alongside the SDK.

The editor's **NVIDIA Reflex** menu selects **Off**, **On** (default), or
**On + Boost**. Set `METALLIC_REFLEX_MODE=off`, `on`, or `boost` before launch to
override the initial mode. The menu disables unavailable controls and shows the
latest complete driver report after warm-up. Render latency measures simulation
start to GPU render completion; it does not include display/scanout latency.

Each application frame owns one Streamline token. Frame-slot reuse waits complete
before sleep so they do not age the sampled input. Sleep runs before SDL input
polling; simulation (starting immediately before input sampling), command
recording/submission, and main-window presentation use that same token. DLSS
evaluations within the frame reuse it. The scope also advances the frame ID when
no DLSS pass runs. Offscreen evaluations without an
application scope continue allocating their own tokens. Frame IDs do not reset
when accumulation history resets or the swapchain is recreated. Abandoned frames
close their open marker phases without inventing a presentation.

Off mode still calls sleep and markers, as required by the SDK for driver frame
limits and timing. Settings changes apply before the next frame's sleep. Runtime
API failures disable low-latency control while leaving rendering available.
Minimized windows and detached ImGui windows suspend the low-latency mode and
timing; returning to a single visible window restores the selected mode. Secondary
swapchain attribution and PC-latency ping/flash instrumentation are not implemented.

## Validation

```powershell
cmake -S . -B build-dev -DMETALLIC_BUILD_TESTS=ON
cmake --build build-dev --target Metallic MetallicRhiTests
ctest --test-dir build-dev -L reflex --output-on-failure
```

The three Reflex smoke tests render 128 frames in Off/On/Boost with Vulkan
validation. On supported hardware they require actual driver timestamps covering
simulation, render submission, presentation and GPU work. Unsupported
hardware returns a CTest skip. Logs are saved under
`build-dev/tests/editor-reflex-{off,on,boost}/editor.log`.
The general `METALLIC_SMOKE_TEST_FRAMES` limit is 256 to allow the SDK's report
history to warm up. Existing DLSS camera and multi-viewport smoke tests cover the
other editor paths. Run GPU tests with access to the normal NVIDIA driver services:
a restrictive process sandbox can leave NGX shutdown waiting on its telemetry
service's named pipe after rendering has already completed.

Integration references: NVIDIA's
[Streamline Reflex guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideReflex.md),
[PCL guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuidePCL.md),
and the bundled `External/streamline/external/reflex-sdk-vk/inc/NvLowLatencyVk.h`.
