# MiniZorah capture-start crash: stale Nsight command buffer

Investigation: 2026-09-24. Nsight Graphics 2026.3.1, build 38722833,
RTX 5060, Windows driver 616.92. The original capture,
`MetallicGPUDrivenSample_2026_09_24_11_12_06.ngfx-capture`, is zero bytes.

## Evidence

Capturing MiniZorah reproduces the screenshot's access violation at
`ngfx-capture-interception.dll + 0xec137`, during Capture Begin. The instruction
is `movq 0x1a8(%rcx), %rax` with `rcx == 0`, reading address 0x1a8. CLI and SDK
reproductions reach it through `Swapchain::present`. The user's supplied
`0ff9532f-b773-4cdd-8240-57e46dad5c97.dmp` initially contained the SDK smoke
regression at 11:24:10. A replacement dump at 11:39:13 is from the ordinary
editor with the reset-only workaround already compiled in. It faults at the
same RVA, with a recycled-data value `rcx = 0x0d0b0d0c0b0a0905`.

The routine starts at RVA `0xec0e0`, iterates an event list at object offsets
`0x228`/`0x230`, and obtains its device pointer from offset `0xa8`. Inspecting a
live, valid object at this routine's entry gives the MSVC RTTI name:

```text
.?AVCommandBuffer@Vulkan@Capture@Pylon@NV@@
```

In the failing live run, the object's memory has been reused for editor
property data (`resetSerial`, `camera.eye`, `camera.center`) instead. This is
evidence of a stale Nsight command-buffer reference, not simply an absent
optional Vulkan feature. The private NVIDIA source is unavailable, so the
precise internal ownership defect remains unverified.

## Workaround

Resetting before free was insufficient. Destruction breakpoints on Nsight's
`CommandBuffer` virtual destructor traced the reused address to
`RenderGraphExecutor::execute()` clearing its completed submission-slot buffers.
The same address was later polled in `DeviceImpl::~DeviceImpl()`'s
`vkDeviceWaitIdle`, producing another access violation at `+0xec0e7` while
reading the event list. This connects the stale pointer to a concrete engine
retirement path even when a capture itself happened to succeed.

During capture injection, completed native command buffers are now reset and
returned to their command pool's available list instead of being freed.
Creating another RHI wrapper reuses an available native buffer. Retired pools
are reset and cached per device and queue family, so transient texture uploads,
resize, and graph recreation also reuse native objects. The device cache is
mutex-protected; individual pools retain Vulkan's existing external
synchronization requirement. Native pools and their buffers are destroyed at
device teardown, after GPU idle and external-library shutdown.

This preserves the capture wrappers' native objects for the device lifetime
without allocating another native buffer every frame. Reuse limits the normal
cache to the maximum simultaneous demand per queue family/pool. Reset discards
recordings and descriptor heap reservations; command allocator memory can remain
at its high-water mark until device shutdown during capture sessions.
Callers must still complete work
before retiring buffers/pools; no GPU wait is added. Ordinary execution retains
its original immediate-free behavior.

Both SDK and external injection are detected by the existing
`NsightGraphicsCapture::vulkanInjectionActive()` helper. No Nsight capture
settings, Streamline annotations, DLSS, CLAS, OMM, or async compute features are
disabled by the final change. Installed NVIDIA binaries are unchanged.

**TODO:** Remove capture-only recycling once an updated Nsight runtime passes
full-size MiniZorah capture, resize, repeated captures and teardown without it.
Record Nsight and driver versions.

## Rejected hypotheses

`--no-streamline-capture` initially produced a successful capture, and an SDK
capture with that setting replayed three times. However, the same SDK setting
subsequently crashed at the identical address (the supplied user dump). It is
not a reliable fix and has been removed. Disabling lazy resource collection
also reproduced the crash. Disabling private-data lookups passed Capture Begin
but did not complete the captured frame; it is not used by the workaround.

The `VK_NV_low_latency` revision warning remains during successful runs. The
earlier OMM/Aftermath workarounds are independent of this failure.

## Regression

The opt-in SDK regression uses the normal viewport resolution (the initial
regression incorrectly retained the 256-pixel smoke preview), warms up 90
MiniZorah frames, then captures three times in one process using the editor
button's request/Present path. Between captures it resizes the window to
1280x720 and back to 1600x900, allowing resize debounce and DLSS recreation.
It rejects reduced-size previews and requires nonempty completed artifacts
before shutdown. From a configured RelWithDebInfo build:

```powershell
cmake --build build-relwithdebinfo --target MetallicGPUDrivenSample Metallic --parallel 8
$env:METALLIC_NSIGHT_GRAPHICS_CAPTURE = '1'
$env:METALLIC_SMOKE_TEST_NSIGHT_CAPTURE = '1'
$env:METALLIC_SMOKE_TEST_HIDDEN = '1'
./build-relwithdebinfo/Source/MetallicGPUDrivenSample.exe --smoke-test
```

Use the reported path with the installed replayer:

```powershell
& 'C:/Program Files/NVIDIA Corporation/Nsight Graphics 2026.3.1/host/windows-desktop-nomad-x64/ngfx-replay.exe' --present-hidden --loop-count 3 --verbose --no-block-on-incompatibility --no-crash-reporting '<capture-path>'
```

Sandboxed test processes initially waited in `NvTelemetryAPI64.dll` / `_nvngx.dll`
at teardown. The SDK stack was in `shutdownStreamline()`, after capture completed.
Final process-exit and replay checks run outside the sandbox; a forcibly
terminated sandbox process is not counted as a passing exit.

Historical reset-only results (superseded by the 11:39 user crash):

| Check | Result |
| --- | --- |
| RelWithDebInfo `Metallic` and `MetallicGPUDrivenSample` | Build succeeds |
| Three independent SDK MiniZorah captures | All export and exit 0 |
| First final capture, three replay loops | `Replay completed successfully`, exit 0 |
| Ordinary MiniZorah without injection, 16 frames | Exit 0 |
| Final capture function metadata | NGX DLSS evaluation and CLAS commands retained |

The retained historical artifact is
`Captures/NsightGraphics/MetallicGPUDrivenSample_2026_09_24_11_33_24.ngfx-capture`
(2,667,005,784 bytes). The subsequent passing captures were at 11:34:08 and
11:34:24. `reset-replay.log`, `reset-functions.txt`, `reset.png`, and
`no-injection.log` contain the final replay, metadata, embedded screenshot and
ordinary-run evidence. These reduced-preview checks did not establish that
reset-before-free fixed the interactive failure.

## Native recycling follow-up

Evidence for this follow-up is in `.tmp/nsight-command-lifetime/`. `user.dmp`
and `user.txt` preserve the replacement 11:39 dump; `track.txt` and `retire.txt`
connect the invalid pointer to its earlier destructor calls. `reuse-full.log`
records a full-size capture and clean exit. `reuse-resize.log` records three
captures in one process at viewport sizes 1198x438, 878x258, and 1198x438, with
exit code 0. Streamline annotations, DLSS and independent compute remain enabled.

Final recycling verification:

| Check | Result |
| --- | --- |
| Full-size SDK capture and teardown | Exit 0 |
| Three SDK captures with two resizes in one process | All export; exit 0 |
| Ordinary interactive executable, external injection at frame 600 | Capture exported; CLI exit 0 |
| Last resized SDK artifact, three replay loops | Exit 0, replay completed successfully |
| External frame-600 artifact, three replay loops | Exit 0, replay completed successfully |
| Ordinary execution without injection, 16 smoke frames | Exit 0 |

The retained SDK artifact is
`Captures/NsightGraphics/MetallicGPUDrivenSample_2026_09_24_11_47_38.ngfx-capture`.
The external artifact is `.tmp/nsight-command-lifetime/external-frame600.ngfx-capture`
(890,737,032 bytes). Its metadata retains NGX evaluation and CLAS commands.
`replay-sdk.log`, `replay-external.log`, `no-injection.log`, `functions.txt`,
and `frame600.png` contain the final verification evidence. The external launch
uses no `--smoke-test` and no experimental Nsight troubleshooting workarounds:

```powershell
$env:METALLIC_NSIGHT_GRAPHICS_CAPTURE = '0'
$env:METALLIC_SHADER_CAPTURE_SYMBOLS = '1'
& 'C:/Program Files/NVIDIA Corporation/Nsight Graphics 2026.3.1/host/windows-desktop-nomad-x64/ngfx-capture.exe' --exe 'E:/metallic/build-relwithdebinfo/Source/MetallicGPUDrivenSample.exe' --working-dir 'E:/metallic' --capture-frame 600 --frame-count 1 --output-file 'E:/metallic/.tmp/nsight-command-lifetime/external-frame600.ngfx-capture' --terminate-after-capture --no-hud --no-block-on-first-incompatibility --no-block-on-interfering-application
```

After narrowing the cache mutex scope so replacing an existing output pool cannot
recursively acquire it, both executables were rebuilt (`build-final.log`). The
three-capture/two-resize regression passed again with exit 0 (`reuse-final.log`,
11:55:10, 11:55:16, and 11:55:20 captures). These duplicate capture files were
removed after verification; the logs and earlier replay-verified artifacts remain.

Ignored local evidence is under `.tmp/nsight-capture-20260924/`: `user.dmp`,
`user-dump.txt`, `live-crash2.txt` (fault and reused memory), `object.txt` (RTTI),
`reset-before-free*.log`, and build/replay diagnostics. Interactive Graphics
Debugger replay and ZorahFull were not tested.
