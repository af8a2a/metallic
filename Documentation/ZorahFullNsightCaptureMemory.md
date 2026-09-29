# ZorahFull Nsight capture memory-pressure failure

Investigation: 2026-09-29, Windows, RTX 5070 Ti (16 GB), driver 616.92,
Nsight Graphics 2026.3.1 / build 38722833, MSVC RelWithDebInfo. The sample uses
optimized capture-symbol shaders, Streamline DLSS-SR, and the normal ZorahFull
pipeline and streaming budgets.

## Failure and correction

The supplied log ends in `VBuffer / CLAS build: OutOfMemory`. A 48,128-byte
allocation in the CLAS scratch domain is denied while the driver reports
19,053,507,184 bytes used against a 14,351,957,960-byte budget. VMA's backing
blocks account for 9,239,179,872 bytes. The original executable reproduced the
same failure during capture with a 1,280-byte publication upload. This is a
budget admission failure, not evidence of a native access violation.

`MeshletStreamCompactClasPool::flushPublications` incorrectly inherited the
generic CLAS buffer usage, including acceleration-structure storage and shader
input. The buffer is actually only CPU-written data copied into the address
and page tables. That usage selected device-local host-visible memory and
made mandatory publication fail under Nsight's transient memory pressure.
The upload now declares only `TransferSource` and uses the Upload accounting
domain. The existing Vulkan pure-staging policy consequently prefers host
memory on discrete GPUs. GPU completion retention, publication ordering and
cancellation replay are unchanged. Allocation errors also identify the
publication upload instead of producing an empty CLAS diagnostic.

After correcting that allocation, capture completed, but the following frame
could still exit because DLSS returned `eWarnOutOfVRAM`. The bundled NVIDIA
implementation, `External/Streamline/source/plugins/sl.common/commonInterface.cpp`,
checks the budget **after successful evaluation**, and only replaces `eOk`
with this warning. SR and RR evaluation now retain the valid output and
warning diagnostics. Other SDK calls and actual evaluation errors still fail.
CLAS growth admission and texture migration backpressure remain enabled;
capture does not bypass the device-local budget.

## Regression coverage

`RhiResource.clas_compact_lifecycle` now revives and GPU-publishes an existing
CLAS after setting the device-local heap limit to one byte. It reads back the
published page-table state and checks that no device-local admission was
denied. This case runs on devices with a separate host heap. Normal validation
and Nsight-injected runs passed; `RhiResource.unified_memory_budget` also passed.

The editor capture regression now waits for streamed root geometry and CLAS
readiness (up to 1,200 frames), rather than assuming 90 frames is enough for
ZorahFull. Resize coverage grows then restores the settled window size so
saved dock panels do not collapse the viewport to an unsupported 30-pixel
DLSS output. These changes affect the smoke test only.

Run from an x64 MSVC developer shell:

```powershell
cmake --build build-relwithdebinfo --target MetallicGPUDrivenSample Metallic -j 8
cmake --build build-scheduling-release --target MetallicRhiTests -j 8
& build-scheduling-release/tests/MetallicRhiTests.exe '--gtest_filter=*clas_compact_lifecycle:*unified_memory_budget' --output-dir build-relwithdebinfo/zorah-nsight-regression
& build-scheduling-release/tests/MetallicRhiTests.exe --rhi-nsight-capture --rhi-no-validation '--gtest_filter=*clas_compact_lifecycle' --output-dir build-relwithdebinfo/zorah-nsight-injected
$env:METALLIC_SMOKE_TEST_NSIGHT_CAPTURE = '1'
& build-relwithdebinfo/Source/MetallicGPUDrivenSample.exe --zorah-full --smoke-test
Remove-Item Env:METALLIC_SMOKE_TEST_NSIGHT_CAPTURE
```

Local raw evidence is preserved under
`build-relwithdebinfo/zorah-nsight-fix-20260929/`, including the unchanged
user log, failing baseline, intermediate runs, GPU regression logs and replay
logs. Captures remain under `Captures/NsightGraphics/`; generated artifacts
are not source-controlled.

## Full-scene capture results

The final `ready.stdout.log` run reached scene readiness at frame 399, with
103,528 / 103,528 geometry-and-CLAS readiness entries complete. It exported
three captures and exited with code 0 (`ready.result.json`). Inspection of
the first capture's embedded screenshot confirmed the rendered ZorahFull
courtyard rather than the loading overlay.

| Capture filename | Viewport | Bytes |
| --- | --- | ---: |
| `MetallicGPUDrivenSample_2026_09_29_22_25_08.ngfx-capture` | 1797 x 660 | 11,944,257,720 |
| `MetallicGPUDrivenSample_2026_09_29_22_26_27.ngfx-capture` | 1957 x 750 | 12,614,939,016 |
| `MetallicGPUDrivenSample_2026_09_29_22_26_54.ngfx-capture` | 1797 x 660 | 12,754,759,560 |

The last capture passed metadata/log extraction and an actual three-iteration
replay with exit code 0 and `Replay completed successfully` in
`ready-replay.stdout.log`:

```powershell
& 'C:/Program Files/NVIDIA Corporation/Nsight Graphics 2026.3.1/host/windows-desktop-nomad-x64/ngfx-replay.exe' --present-hidden --loop-count 3 --no-block-on-incompatibility --no-crash-reporting --verbose 'E:/metallic/Captures/NsightGraphics/MetallicGPUDrivenSample_2026_09_29_22_26_54.ngfx-capture'
```

The capture contains no embedded error-severity log messages. Nsight still
reports its capture/replayer version-string warning despite matching build
38722833, and warns that NGX replay can produce artifacts. NGX remained enabled.
The embedded screenshot is capture-side visual evidence; no pixel comparison
of newly rendered replay output was performed.

Nsight still temporarily exceeds device-local headroom; optional CLAS growth
and texture migrations may be deferred. The test deliberately retains the
normal scene, resource budgets, ray tracing and DLSS instead of weakening the
workload to avoid that pressure. This is a capture correctness check, not a
performance result or a guarantee of smooth capture-time frame pacing.
