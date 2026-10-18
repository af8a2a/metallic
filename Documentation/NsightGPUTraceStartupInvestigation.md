# GPU Trace startup and exit failures in RelWithDebInfo

Investigation: 2026-10-17, Windows, RTX 5070 Ti / driver 617.42,
Nsight Graphics 2026.3.1 / SDK 0.9.2,
`build-relwithdebinfo/Source/LookDev.exe --nsight-gputrace`.

## Host target-ready timeout

The reported run completed shader compilation and then spent 583.9 seconds in
native pipeline warmup (110 created pipelines). The GPU Trace host had already
started, so its target-ready timeout elapsed before `createDevice` returned.
The subsequent activation handshake reported `GPU Trace host exited`, and the
application returned 1. That log does not show a shader compiler crash.

SDK injection and initialization still happen before Vulkan initialization.
Host launch now happens in `prepareBeforeSubmission`, after device creation and
native PSO warmup, immediately before SDK activation. The 60-second activation
watchdog, capture deadlines, and shader warmup remain enabled. The regression
script checks this ordering and permits an explicit whole-process budget via
`STARTUP_TIMEOUT_SECONDS` for cold-cache runs.

## Exit-time D3D12 dependency lifetime

After fixing startup, the real smoke test produced three GPU Trace files but
then raised an access violation while exiting. LLDB identified:

```text
CRT exit -> WarpViz.Injection.dll -> WarpVizTarget.dll -> D3D12Core.dll
D3D12Core + 0x53b3e: mov rax, [rax + 0x10]
RAX = 0x7ffde4f36078
```

An unload breakpoint enabled only after entering `EditorApplication::shutdown`
showed `nvoglv64.dll` calling `FreeLibrary(D3D12.dll)` inside `vkDestroyDevice`.
The DLL was mapped at `0x7ffde4f20000`; the stale virtual table was at its RVA
`0x16078`. The preceding instructions load the object from `[rcx + 0x98]`, then
its virtual table, before attempting the indirect call. Nsight's process-exit
callback consequently references an object whose virtual table has been
unmapped. Addresses describe this diagnostic run, not portable signatures.

Skipping `closeTraceHost` in the debugger did not change the fault. Host
termination is therefore not required to trigger it. The SDK exposes no
GPU Trace shutdown/uninject API that could move this cleanup before Vulkan
device destruction.

`prepareForGraphicsShutdown` now pins the already-loaded `D3D12.dll` before
editor graphics teardown, only for GPU Trace sessions. It does not load the
DLL when absent. Windows' documented
[`GetModuleHandleExW` PIN flag](https://learn.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-getmodulehandleexw)
keeps the module mapped through the injected process-exit callback. Normal
Vulkan cleanup and host cleanup still run. This retains a runtime dependency
until process termination; it does not skip destructors or suppress exceptions.

This compatibility workaround covers editor shutdown after device creation.
It does not establish safety for a device-creation failure that destroys an
internal partial device before control returns to the editor. Reevaluate the
pin when upgrading Nsight/driver versions, and remove it when the same capture
and normal-exit regression passes without it.

## Local evidence

The rebuilt RelWithDebInfo LookDev passed all three maintained startup cases:

| Case | Result |
| --- | --- |
| `export` | Three completed GPU Trace exports, including viewport resize and restore; application exit 0. |
| `invalid-metrics` | Specific host diagnostic and expected application exit 1, without hanging. |
| `disabled` | Full shader/native PSO warmup, rendered frames, exit 0; no Nsight handshake or DLL pin. |

The 12 CPU-only script fixtures also passed. These exercise ordering and timeout
validation; they are separate from the real GPU runs above. Executable SHA-256:
`7cf54016d00cf0f78d7306e1f27abff99d0e3b9d5c5d271f9dfcea71ec37c491`.

The ignored directory `build-relwithdebinfo/nsight-startup-fix-20261017/` holds
the supplied logs, build logs, debugger stacks, unload trace, minidumps, and
per-case commands/results with executable hashes. Captures remain under
`Captures/NsightGraphics/`; these generated files are not source assets.

- `ReportedStartup.log` / `ReportedHost.log`: original target-ready timeout.
- `export/`: host-launch fix alone; three exports followed by access violation.
- `debug-keep-host/`: same exit failure with host cleanup skipped.
- `debug-shutdown-unload/`: driver unload stack and stale virtual table evidence.
- `export-fixed/`: full regression for the combined fix.
- `invalid-metrics/` and `disabled/`: error-path and ordinary-startup regressions.

The application PSO cache is isolated to this evidence directory. Shader and
driver caches are not cleared. Differences from the reported cold startup
are not a performance A/B result.


## Live export operation timeout (2026-10-18)

A later interactive LookDev session completed startup but exited with code 1
about six seconds after requesting its first GPU Trace. The owned host log
`GpuTraceHost-37588.log` reported `Operation timeout, consider retrying with
timeouts disabled.` The editor detected the dead host and deliberately stopped
its render loop; this evidence does not show a native access violation or
Vulkan device loss.

The installed Nsight 2026.3.1 CLI distinguishes general operation timeouts from
`--trace-timeout` (trace start through data transfer). The live host now uses
`--no-timeout` as advised by its diagnostic, while retaining `--trace-timeout
120`, the independent 60-second startup watchdog, and the application's
120-second export polling deadline. The polling deadline cannot interrupt an
internally blocked Vulkan call; integration runs also use an external process
budget. Host operation-timeout messages are now surfaced in the editor's own
error message instead of being reduced to "host exited".

RelWithDebInfo LookDev exported three fresh GPU Trace files and exited 0,
covering 1797x694, 1957x784, and restored 1797x694 viewports. The smoke workflow
uses the same capture request and Present boundaries as the Profiler button.
The invalid-metrics startup regression also passed with a specific diagnostic
and exit 1. Its first attempt was rejected earlier by source-epoch validation;
the isolated rerun reached and verified the intended error path.
This validates the default LookDev workload and resize sequence, not every
possible interactive session. The repeated missing `kFeatureDLSS_G` diagnostics
are a separate issue: the application does not load Frame Generation, and the
reported session continued rendering after those messages.

Evidence: `.cache/lookdev-nsight-timeout-20261018/` contains the supplied logs,
build and integration logs, and `result.json` with executable/capture hashes.
Fresh traces are under `Captures/NsightGraphics/LookDev_2026_10_18_14_06_52`,
`...14_06_58`, and `...14_07_03` (all `.ngfx-gputrace`). This is capture failure
validation, not a performance measurement; no GPU counter analysis or timing
speedup is claimed.
