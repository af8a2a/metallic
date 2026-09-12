# Nsight capture crash while building KHR opacity micromaps

Investigation date: 2026-09-12. The original failure is documented below. A
subsequent workaround selects an EXT OMM backend for Nsight Graphics, while
ordinary execution retains KHR OMM.

## Nsight EXT backend workaround

**TODO(Nsight KHR OMM): Remove this temporary EXT workaround when Nsight Graphics
supports `VK_KHR_opacity_micromap`.** Before removal, verify KHR OMM builds under
SDK/external injection, including the required null build-range semantics, and
successful frame capture and replay. Re-run the OMM visibility and alpha-test
candidate regressions with KHR enabled in the injected process. Record the
verified Nsight version, restore KHR selection for capture sessions, and remove
the workaround-only EXT object/build/attachment and SPIR-V branches, associated
metadata that is no longer used, and EXT-specific tests/documentation. Retain
the injection regression coverage and this historical crash investigation.

`NsightGraphicsCapture::vulkanInjectionActive()` recognizes successful SDK
injection and, on Windows, an already loaded `ngfx-capture-interception.dll`
from external injection. Detection occurs during device selection. Injection
success remains recorded even if initializing the capture activity later fails.
Merely installing Nsight or compiling SDK support does not select EXT.

The capture path enables `VK_EXT_opacity_micromap` and its own feature structure;
it does not enable KHR OMM or its device-address-commands dependency for OMM.
It uses native `VkMicromapEXT` creation/destruction, EXT size queries and builds,
and EXT BLAS attachments. Resource buffers receive the required EXT usage bits,
and micromap builds use EXT stage/access masks before sharing scratch memory or
feeding a BLAS build. The old-validation-layer guard applies only to KHR OMM.

`RayTracingTriangleGeometryDesc` carries the geometry's OMM usage histogram so
the EXT BLAS size query can run before recording the micromap build. The scene
builder passes the baked per-primitive histogram, and the RHI checks that its
counts cover exactly the geometry's triangles. Histogram pointers are consumed
only during size queries and command recording, not retained for GPU execution.

The EXT shader patch declares `SPV_EXT_opacity_micromap` and the legacy
`RayTracingOpacityMicromapEXT` capability (5381). The local driver did not reduce
ray-query alpha candidates without this declaration, despite successful OMM
builds. The EXT path does not emit `OpacityMicromapIdKHR`, whose capability
requires enabling the KHR feature. Ordinary KHR devices retain their existing
execution-mode patch. The two patches run after compiler/cache lookup, so cache
keys and driver registration use the actual device-specific binary. See the
[SPIR-V capability requirements](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html).

BLAS compaction remains enabled. Compaction of the separate EXT micromap object
is not exposed through the RHI's AS-only query pool; requesting that optional
flag returns `Unsupported`. Scene micromaps do not request it. EXT handles have
no AS device address, so `valid()` recognizes their native handle while
`deviceAddress()` returns zero for these objects.

The RHI OMM regression now builds two micromaps to exercise shared scratch and
per-BLAS usage data, then compares ray visibility against shader alpha traversal
through compaction, cutoff, UV transform, alpha-zero, and BLEND material edits.
Run it in separate processes with and without `--rhi-nsight-capture`; use
`--rhi-no-validation` for the KHR control when installed validation layers are
older than 1.4.357.

### Workaround verification

Final RelWithDebInfo builds of `Metallic`, `MetallicGPUDrivenSample`, and
`MetallicRhiTests` succeeded on 2026-09-12. The headless regression processes
produced these results with the installed Nsight Graphics 2026.3.1:

| Configuration | Tests passed | Local report |
| --- | --- | --- |
| Ordinary execution, KHR OMM, validation off | 3/3 | `.tmp/rtas-relwithdebinfo/khr-final.json` |
| Nsight SDK injection, EXT OMM, validation off | 3/3 | `.tmp/rtas-relwithdebinfo/ext-final.json` |
| Nsight SDK injection, EXT OMM, validation on | 2/2 | `.tmp/rtas-relwithdebinfo/ext-final-validation.json` |

The first two runs cover OMM baking, ray-query visibility, and scene acceleration
structure construction. The validation run covers the two OMM tests. All tests
ran without skips. Both backends reduce alpha-test candidates from 20,480 to
4,224, and SHA-256 comparisons of all five material-variant visibility images
match between KHR and EXT. This verifies actual OMM use as well as build success.

Validation still reports buffer access capability, allocation, and descriptor
warnings from surrounding RHI/scene paths; this is not a validation-clean run.
The recorded VUIDs are `uniformAndStorageBuffer16BitAccess-06332`,
`uniformAndStorageBuffer8BitAccess-06329`, `VkMemoryAllocateInfo-pNext-02806`,
`VkShaderModuleCreateInfo-pCode-08740`, and
`VkWriteDescriptorSet-descriptorType-00328` / `00331`. No OMM-specific validation
diagnostics were observed. Full logs use the corresponding report names with a
`.log` suffix. These checks exercise real SDK injection and GPU builds/traversal;
they do not include an editor frame capture or capture replay.

## Finding

The installed Nsight Graphics 2026.3.1 capture interceptor dereferences a null
build-range pointer while intercepting `vkCmdBuildAccelerationStructuresKHR` for
a KHR opacity micromap. Metallic supplies a null range as required by the Vulkan
specification. The fault occurs on the CPU in `ngfx-capture-interception.dll`
during command recording, before this OMM build can be submitted to the GPU.

This identifies the failure in this specific capture path and installed binary;
it does not establish the behavior of other Nsight versions or every KHR OMM API.

## Reproduction

The existing `RhiRendering.opacity_micromap_ray_query` test reproduces the editor's
scene-resource preparation path using two alpha-masked triangles. It first runs
without OMM, then builds real KHR OMM and checks ray visibility across five
material variants. Its test devices do not enable Streamline or Aftermath.

`tests/rhi/main.cpp` now accepts `--rhi-nsight-capture` to inject the same Nsight
SDK before test-device creation. The diagnostic does not require an editor
window, HUD, or an explicit frame-capture request.

| Configuration | Nsight injection | Result |
| --- | --- | --- |
| Debug | Off | OMM ray-query test passes, including actual micromap builds |
| Debug | On | Non-OMM phase completes; first OMM build raises `0xc0000005` |
| RelWithDebInfo | Off | Same OMM ray-query test passes |
| RelWithDebInfo | On | LLDB stops in the capture DLL at the same OMM build path |

Build the diagnostic from an x64 Visual Studio developer shell:

```powershell
cmake --preset metallic-relwithdebinfo -DMETALLIC_BUILD_TESTS=ON
cmake --build build-relwithdebinfo --target MetallicRhiTests --parallel 6
```

Run the control and injected cases in separate processes:

```powershell
./build-relwithdebinfo/tests/MetallicRhiTests.exe --rhi-no-validation --gtest_filter='*opacity_micromap_ray_query*' --output-dir .tmp/rtas-relwithdebinfo/rel-baseline
./build-relwithdebinfo/tests/MetallicRhiTests.exe --rhi-no-validation --rhi-nsight-capture --gtest_filter='*opacity_micromap_ray_query*' --output-dir .tmp/rtas-relwithdebinfo/rel-injected
```

For debugger inspection, also pass `--gtest_catch_exceptions=0`. The tests use
`--rhi-no-validation` because Metallic's existing guard disables KHR OMM when
validation layers older than 1.4.357 are active, which can hide this failure.

The RelWithDebInfo preset explicitly sets `METALLIC_DEFAULT_NSIGHT_CAPTURE=true`.
This explains its different default behavior from Release; the injected Debug
reproduction shows that optimization is not necessary to trigger the fault.

## Debugger evidence

Loaded interceptor:

```text
C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.3.1\target\windows-desktop-nomad-x64\ngfx-capture-interception.dll
Size:   15023792 bytes
SHA256: 0DB570F1E24EE8501745B369C7F56EF7B2E289934F0F79E8D78B46737425E1B8
Fault:  RVA 0x901ac0
```

The fault is `0xc0000005`, reading address zero. LLDB reports `r9 = 0` at the
following four-field copy routine:

```asm
movl (%r9), %eax       ; faults here
movl %eax, (%r8)
movl 0x4(%r9), %eax
movl %eax, 0x4(%r8)
movl 0x8(%r9), %eax
movl %eax, 0x8(%r8)
movl 0xc(%r9), %eax
movl %eax, 0xc(%r8)
retq
```

The immediate caller walks build infos with an 80-byte stride, reads
`geometryCount` at offset `0x30`, and loops over range records with a 16-byte
stride. Before calling the copy routine, it loads `ppBuildRangeInfos[i]` and adds
`j * 16`, without excluding OMM or checking the null range. Reconstructed behavior
of this loop, rather than Nsight source code:

```cpp
for (uint32_t i = 0; i < infoCount; ++i) {
    for (uint32_t j = 0; j < pInfos[i].geometryCount; ++j) {
        copyFourUint32(destination[i] + j, ppBuildRangeInfos[i] + j);
    }
}
```

At Metallic's caller frame, LLDB independently confirms:

```text
buildInfo.type          = VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR
buildInfo.mode          = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
buildInfo.geometryCount = 1
noRanges                = nullptr
micromapData.sType      = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR
micromapData.usageCountsCount = 1
```

The stack leads directly through `VulkanRhi.cpp:6267`,
`SceneAccelerationStructure.cpp:915`, and
`ScenePathTraceResources::pumpPrepareAsync`, matching the reported editor stack.
The scene builder is recording the micromap build at this point; its ordinary
BLAS build loop follows afterward.

Local evidence is retained under `.tmp/rtas-relwithdebinfo/` (ignored output):

- `baseline.log`: Debug without injection, passing actual OMM builds.
- `injected-before.log`: Debug injection, successful non-OMM phase followed by AV.
- `rel-baseline.log`: RelWithDebInfo without injection, passing OMM test.
- `lldb.log`: exception, full registers, copy routine, and loaded DLL mapping.
- `lldb-parameters.log`: caller loop and Metallic's build parameters.
- `inspect.lldb` and `inspect-parameters.lldb`: debugger command scripts.

## Why EXT support does not cover this call

KHR OMM changes the API, rather than simply renaming the EXT extension. EXT uses
`VkMicromapEXT` and `vkCmdBuildMicromapsEXT`; KHR represents the micromap as a
`VkAccelerationStructureKHR` with a new type and geometry kind, and reuses
`vkCmdBuildAccelerationStructuresKHR`. Khronos explicitly documents these API
changes in issues 5 and 6 of the [KHR extension specification](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_opacity_micromap.html).

For `VK_GEOMETRY_TYPE_MICROMAP_KHR`, `ppBuildRangeInfos[i]` **must be null** under
VUID `VUID-vkCmdBuildAccelerationStructuresKHR-ppBuildRangeInfos-11544`.
Metallic's `isMicromap ? &noRanges : rangePointers.data()` follows that rule:
the outer array is valid, and its first element is null. Replacing it with a
dummy range would violate this requirement. See the [build command specification](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdBuildAccelerationStructuresKHR.html).

The currently published [Nsight supported Vulkan extensions](https://docs.nvidia.com/nsight-graphics/UserGuide/appendix.html#supported-vulkan-functions)
list includes `VK_EXT_opacity_micromap`, but not `VK_KHR_opacity_micromap`.
The installed interception DLL also contains the EXT type/command strings and
not the new KHR OMM type names. This is supporting evidence; the debugger's
actual null dereference is the decisive evidence for this failure.

Device extension enumeration, feature queries, and device creation succeed
through the injected process. Those observations establish that the driver
provides the feature and the capture chain allows device creation; they do not
establish that the capture layer can copy and replay the new command semantics.
The captured caller loop demonstrates the missing handling directly.

## Scope of the original investigation

At the end of the original investigation, runtime behavior was unchanged and the
only code addition was the opt-in RHI test injection flag. The EXT workaround
above was implemented afterward. It does not substitute dummy KHR ranges or
modify NVIDIA binaries.

The observed interceptor needs to recognize the new OMM geometry and its null
range semantics. Complete KHR capture support would also need to handle the new
OMM structures and object relationships; this investigation stops at the first
proven failure and does not claim those later paths have been tested.

## Comparison with the local vk_mini_samples checkout

The local `E:/vk_mini_samples` checkout at commit `994ac9f` was inspected on
2026-09-12, together with its existing `mm_opacity.exe` and runtime log. The
`samples/mm_opacity` implementation uses **VK_EXT_opacity_micromap**. It does not
enable or implement the KHR OMM build path used by Metallic.

The existing `_bin/Debug/log_mm_opacity.txt`, last written at
2026-09-12 17:59:06 +08:00, provides the distinction explicitly:

```text
Available Device Extensions :
...
[ ] VK_KHR_opacity_micromap (v. 1)     // line 109
...
[x] VK_EXT_opacity_micromap (v. 2)     // line 234
```

The annotations above identify the original log lines. The logger in
`E:/nvpro_core2/nvvk/context.cpp:634` enumerates all available extensions and marks
`x` only when an extension belongs to the application's selected extension list.
Consequently, the KHR entry demonstrates driver availability, not application
enablement. The log also records successful micromap and BLAS builds at lines
365 and 368. Listing Nsight layers among available instance layers does not, by
itself, establish which capture activity was active for that log.

### API comparison

| Operation | Local mm_opacity sample | Metallic |
| --- | --- | --- |
| Selected extension | `VK_EXT_opacity_micromap` | `VK_KHR_opacity_micromap` |
| Feature structure | `VkPhysicalDeviceOpacityMicromapFeaturesEXT` | `VkPhysicalDeviceOpacityMicromapFeaturesKHR` |
| OMM object | `VkMicromapEXT` | `VkAccelerationStructureKHR`, type `OPACITY_MICROMAP_KHR` |
| Object creation | `vkCreateMicromapEXT`, backed by a buffer | `vkCreateAccelerationStructure2KHR`, backed by an address range |
| Size query | `vkGetMicromapBuildSizesEXT` | `vkGetAccelerationStructureBuildSizesKHR`, null primitive-count pointer for OMM |
| OMM build | `vkCmdBuildMicromapsEXT` | `vkCmdBuildAccelerationStructuresKHR`, null range element for OMM |
| BLAS attachment | `VkAccelerationStructureTrianglesOpacityMicromapEXT` | `VkAccelerationStructureTrianglesOpacityMicromapKHR` |

Source locations establishing the sample's actual choices:

- `E:/vk_mini_samples/samples/mm_opacity/mm_opacity.cpp:868`: EXT feature structure.
- `E:/vk_mini_samples/samples/mm_opacity/mm_opacity.cpp:885`: requests the EXT extension.
- `E:/vk_mini_samples/samples/mm_opacity/mm_process.cpp:220`: EXT size/build structures.
- `E:/vk_mini_samples/samples/mm_opacity/mm_process.cpp:246`: creates `VkMicromapEXT`.
- `E:/vk_mini_samples/samples/mm_opacity/mm_process.cpp:255`: records `vkCmdBuildMicromapsEXT`.
- `E:/vk_mini_samples/samples/mm_opacity/mm_opacity.cpp:467`: attaches EXT OMM to triangle geometry.

Corresponding Metallic code is in `VulkanRhi.cpp`: extension selection at line
2286, KHR micromap geometry construction at line 543, the OMM build at line 6267,
and address-based OMM creation at line 7372.

The sample does call `vkCmdBuildAccelerationStructuresKHR` later, but only for
ordinary triangle BLAS and instance TLAS. Its helper at
`E:/nvpro_core2/nvvk/acceleration_structures.cpp:116` supplies an actual range array
from `asBuildRangeInfo.data()`. It never sends the OMM geometry plus null range
combination that faults in Metallic's Nsight reproduction. This is the decisive
implementation difference: a shared KHR BLAS command name does not mean the
sample is constructing KHR micromap objects.

The inspected `_bin/Debug/mm_opacity.exe` also contains the EXT extension and
`vkCmdBuildMicromapsEXT` strings, and no KHR OMM extension or micromap geometry
type strings. This is consistent with the source and log, rather than evidence
of a separate KHR implementation in the existing binary.

### Capture and scheduling differences

The sample entry point creates the Vulkan context directly; it contains no
`NGFX_GraphicsCapture_Inject_Vulkan` or activity-initialization calls. Metallic's
capture-enabled path initializes the Nsight capture SDK before graphics-device
creation. Thus the sample's reported external Nsight connection is not itself
an identical SDK-injection experiment.

The sample also submits and waits for its micromap build before constructing
BLAS, while Metallic prepares scene resources asynchronously and inserts AS
build barriers. This scheduling difference does not explain the proven CPU
null dereference: Metallic faults inside the interception call before the OMM
command can be submitted. Neither changing a queue wait nor adding a GPU barrier
would change the range pointer the interceptor reads.

This comparison used source, binary inspection, and an existing runtime log; it
did not launch a new sample capture or modify the sample. It establishes that
the supplied sample exercises EXT OMM, so its successful Nsight connection does
not test the failing KHR OMM command parameters. The comparison does not establish
that every external Nsight activity fails for KHR OMM. Reusing the sample's OMM
backend would require implementing the EXT object/build/attachment lifecycle,
not simply changing an extension name or supplying a dummy KHR range.
