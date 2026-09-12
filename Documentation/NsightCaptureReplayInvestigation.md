# Nsight capture replay: Reflex warning and EXT OMM index restoration

Investigation date: 2026-09-12. This concerns opening
`Captures/NsightGraphics/MetallicGPUDrivenSample_2026_09_12_19_11_05.ngfx-capture`,
after the earlier KHR OMM interception and Aftermath workarounds. See
[the preceding investigation](NsightKhrOpacityMicromapInvestigation.md).

## Findings

The capture uses `VK_NV_low_latency2` calls, while also declaring the legacy
`VK_NV_low_latency` extension. The displayed legacy-extension revision warning
does not identify the cause of this replay crash. The observed failure is a null
dereference in Nsight's restoration of an EXT OMM attachment with implicit
triangle indices. A debugger-only guard for that restoration branch allows the
same capture to replay successfully without changing its extension declarations
or Reflex calls.

The follow-up now provides a runtime workaround: Nsight EXT attachments use a
real UINT32 identity index buffer. Ordinary KHR attachments retain implicit
indices. Newly exported captures pass unmodified CLI replay; old captures still
contain their original implicit-index build data and require recapture with the
rebuilt application. Installed NVIDIA binaries remain unchanged on disk.

## What the low-latency warning means

The capture reports:

```text
Vulkan extension version: VK_NV_low_latency (using: 2 > supported: 1)
```

Here `2` is the revision of the extension named `VK_NV_low_latency`, rather than
the `2` suffix in the separate extension name `VK_NV_low_latency2`. Khronos
documents revision 2 of the legacy extension as adding `LegacyNV` entry points
for compatibility layers. See [the legacy extension specification](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_low_latency.html).

The captured function stream contains `vkSetLatencyMarkerNV` at event 1 and
`vkLatencySleepNV` at event 3, followed by further marker calls. These are
[low_latency2 commands](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_low_latency2.html),
providing evidence from the actual running binaries, independently of the
Streamline source checkout.

In `External/streamline/source/plugins/sl.reflex/reflexEntry.cpp`, Reflex adds
the legacy extension whenever low latency is available, and also adds LL2 for
the NVIDIA Turing-or-newer branch. The Vulkan compute implementation in
`source/platforms/sl.chi/vulkan.cpp` prefers `CreateVkNvLowLatency2`, with
`CreateNvLowLatencyVk` as a fallback. Metallic's device setup in `VulkanRhi.cpp`
supplies the present-id dependency; the Streamline interposer appends the
Reflex extensions. Removing an entry only from Metallic's own extension vector
would not remove a later interposer addition.

The existing `[Reflex] NvLowLatencyVk via Streamline` message in
`VulkanStreamline.cpp` is a fixed log label, not a query of the active backend.
It must not be used as evidence that LL2 failed or that the legacy backend ran.

**TODO(Streamline extension selection): Avoid requiring the legacy extension
when a verified LL2-only configuration is selected.** Check both the plugin's
extension declaration and backend fallback behavior, including the packaged
SDK binaries actually deployed by CMake. This would address the warning; it
is separate from the reproduced OMM replay failure.

## Version labels

Only one Nsight installation was discovered:
`C:/Program Files/NVIDIA Corporation/Nsight Graphics 2026.3.1`.
The capture and installed replayer both report build ID `38722833`. The replayer
prints `2026.3` in its compatibility warning but `2026.3.1` in its own Replay
Information section. The UI file version also says `2026.3`. These observations
do not establish that the user opened a different, older installation. The
version warning remains present in the successful guarded experiment.

## Reproduced crash

The local RTX 5070 Ti uses driver 616.64. The installed `ngfx-replay.exe` SHA-256
is `19794FA39C6F592F293A34B891AB4FEE611D1355099DDB8858B48A38585136D6`.
The capture includes 104 acceleration structures and 10 EXT micromaps.

Run the official CLI without displaying or activating a desktop window:

```powershell
& 'C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.3.1\host\windows-desktop-nomad-x64\ngfx-replay.exe' --present-hidden --loop-count 1 --no-block-on-incompatibility --no-crash-reporting --verbose 'E:\metallic\Captures\NsightGraphics\MetallicGPUDrivenSample_2026_09_12_19_11_05.ngfx-capture'
```

The unmodified replay exits with `0xc0000005` at
`Initializing GPU Memory Objects ... 0%`. LLDB stops inside `ngfx-replay.exe`
at RVA `0x2bb3a5`, before captured-frame execution. The faulting instruction is
`movq (%rdi,%rax), %r14`, with both address registers zero. Export-only symbols
mislabel the surrounding code as an offset from
`GetPylonReplayerRecaptureInterface`; this is not a recovered private function
name.

The preceding instructions traverse a geometry pNext chain looking for sType
`1000396009`, which identifies
`VkAccelerationStructureTrianglesOpacityMicromapEXT`. They restore its
micromap handle, then enter the index-buffer restoration path. The latter
eventually writes the restored device address at attachment offset `0x18`,
the `indexBuffer` field. The attachment at the failure contains:

```text
indexType        = VK_INDEX_TYPE_NONE_KHR
indexBuffer      = 0
indexStride      = 0
baseTriangle     = 0
usageCountsCount = 2
pUsageCounts     = valid array
ppUsageCounts    = null
micromap         = non-null
```

These are Metallic's implicit-index parameters from
`makeExtMicromapAttachment()`. Vulkan specifies that `indexBuffer` must be null
when `indexType` is `VK_INDEX_TYPE_NONE_KHR`; the geometry's triangle index is
then used directly. See [the EXT attachment specification](https://docs.vulkan.org/refpages/latest/refpages/source/VkAccelerationStructureTrianglesOpacityMicromapEXT.html).
Putting a dummy non-null address into the NONE configuration is not a valid fix.

## Causal experiment

| Replay configuration | Observed result |
| --- | --- |
| Default initialization | Access violation at RVA `0x2bb3a5` |
| `--no-multithreaded-init` | Same failure |
| Serial initialization plus `--no-capture-replay-memory` | Same instruction and null address |
| Serial initialization plus a debugger guard for null implicit OMM indices | 10 guards taken; full replay succeeds; process exits 0 |

The guard checks the attachment sType, NONE index type, and zero index address
before moving the program counter from RVA `0x2bb3a5` to `0x2bb41f`, immediately
after the index-buffer restoration branch. It leaves micromap restoration,
resource initialization, extension declarations, NGX, and Reflex calls active.
The original compatibility warnings remain in the successful run. This is a
diagnostic experiment tied to this binary, not a supported replay workaround.

The inspected `E:/vk_mini_samples/samples/mm_opacity/mm_opacity.cpp` uses
`VK_INDEX_TYPE_UINT32`, a real identity index buffer, and a four-byte stride.
`mm_process.cpp` fills that buffer with `0, 1, ..., triangleCount - 1`. This
avoids the absent-index-buffer case used by Metallic.

**TODO(Nsight OMM replay): Remove the explicit identity index workaround once
Nsight restores implicit OMM indices correctly.** Verify a fresh Sponza capture
with NONE indices in both CLI replay and the Graphics Debugger, including BLAS
compaction and repeated replay/reset. Record the Nsight/driver versions, restore
NONE in the EXT size/build attachment helper, and remove its identity-buffer
allocation/cache. Retain the capture regression. Keep ordinary KHR behavior
independent of this EXT-only issue. Successful injection/build tests alone do
not establish successful replay.

Local ignored evidence is in `.tmp/nsight-low-latency/`: `capture-191105/`
contains exported metadata and the function stream; `replay-default.log`,
`replay-serial-init.log`, `replay-resource-stack.log`, and `replay-omm-stack.log`
contain the reproductions; `guard_omm.py` and `replay-omm-guard.log` contain the
process-local guard and successful replay.

## Follow-up: ngfx-rpc crash and engine workaround

The next user capture, `MetallicGPUDrivenSample_2026_09_12_19_38_28.ngfx-capture`,
still reproduces the original failure with the unmodified engine. The supplied
Graphics Debugger crash offset, `ngfx-rpc.exe + 0x326a15`, is the same restoration
branch in the installed RPC executable: it scans for OMM sType `1000396009`,
restores the micromap handle, and faults on `movq (%rdi,%rax), %r14`. Its later
instruction at RVA `0x326a8b` writes the recovered index address at attachment
offset `0x18`. This matches the CLI replayer's RVA `0x2bb3a5` failure. Merely
recapturing before applying an engine change preserves the problematic data.

`ensureMicromapIdentityIndices()` now creates a host-upload buffer containing
`0, 1, ..., triangleCount - 1`, flushes it before queue submission, and uses its
device address with UINT32 and a four-byte stride. It enables AS/OMM build-input
and device-address buffer usages, sharing across graphics/compute families when
needed. The OMM owns immutable index allocations, reuses sufficient capacities,
and retains earlier allocations until destruction so subsequent recordings
cannot invalidate buffers referenced by in-flight BLAS builds. The scene already
keeps its OMMs alive through BLAS compaction and traversal. Size queries use the
same type/stride without allocating or reading an index buffer.

The capture helper additionally supports explicit NGFX frame boundaries, and
`--rhi-nsight-export` uses them in `gpu_driven_sponza_realtime_pipeline` to export
one warmed-up frame without a window. The normal editor retains Present frame
boundaries. The test waits for artifact completion before resizing or releasing
scene resources. Reproduce from a configured RelWithDebInfo build:

```powershell
./build-relwithdebinfo/tests/MetallicRhiTests.exe --rhi-no-validation --rhi-realtime --rhi-async-compute --rhi-aftermath --rhi-nsight-export --gtest_filter='*gpu_driven_sponza_realtime_pipeline' --output-dir .tmp/nsight-low-latency/explicit-sponza
```

Use the capture path reported by that test with the unmodified replayer's
`--present-hidden --loop-count 3 --no-block-on-incompatibility --no-crash-reporting`
options. The compatibility switch suppresses the known warning dialog; it does
not bypass any resource-restoration code.

Verification on 2026-09-12, Nsight 2026.3.1 and driver 616.64:

| Check | Result | Local evidence under `.tmp/nsight-low-latency/` |
| --- | --- | --- |
| User's 19:38 capture before the workaround | Same null dereference | `replay-193828-before.log` |
| Rebuilt Sponza, independent compute, Aftermath | 36 frames pass; 10 OMMs / 34,908 alpha triangles; capture exported | `explicit-sponza.log` / `.json` |
| New capture with unmodified CLI replayer | 3 loops complete; exit 0 | `replay-explicit-sponza.log` |
| EXT OMM bake/visibility and scene AS | 3/3 pass; exit 0 | `explicit-omm-ext.log` / `.json` |
| Ordinary KHR OMM control | 3/3 pass; exit 0 | `explicit-omm-khr.log` / `.json` |
| EXT OMM with validation | 2/2 pass; exit 0; no new OMM VUIDs | `explicit-omm-validation.log` / `.json` |

The fresh artifact is
`explicit-sponza/nsight/MetallicRhiTests_2026_09_12_19_51_02.ngfx-capture`.
Its legacy low-latency revision warning remains present during successful replay.
It contains the Sponza realtime workload and 104 acceleration structures; it
does not contain the editor UI or swapchain. Interactive Graphics Debugger
launch was not exercised. The five EXT/KHR visibility PNGs have identical
SHA-256 hashes; alpha candidates still decrease from 20,480 to 4,224.

The validation run still reports the six previously documented surrounding
RHI/descriptor VUIDs (`06332`, `06329`, `02806`, `08740`, `00328`, `00331`);
this is not a validation-clean engine run. The Sponza test still logs the
previously recorded independent Streamline teardown mini-dump but exits zero.
That plugin issue is not addressed by this OMM replay workaround.

`Metallic.exe`, `MetallicGPUDrivenSample.exe`, and `MetallicRhiTests.exe` were
rebuilt successfully (`build-explicit-indices.log`). The GPUDrivenSample binary
was updated at 19:49:28 local time. Restart the rebuilt application and capture
again; replacing the executable cannot change an existing capture's build data.
