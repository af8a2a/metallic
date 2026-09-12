# Nsight capture crash while building KHR opacity micromaps

Investigation date: 2026-09-12. Status: cause reproduced and localized; no capture
compatibility fallback applied.

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

## Scope of this investigation

The runtime RHI and capture policy retain the original KHR OMM behavior. The only
source addition for this investigation is the opt-in RHI test injection flag.
No automatic OMM disablement, EXT backend conversion, dummy-range substitution,
or modification of NVIDIA binaries has been applied.

The observed interceptor needs to recognize the new OMM geometry and its null
range semantics. Complete KHR capture support would also need to handle the new
OMM structures and object relationships; this investigation stops at the first
proven failure and does not claim those later paths have been tested.
