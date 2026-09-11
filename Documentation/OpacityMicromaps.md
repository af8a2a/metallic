# KHR opacity micromaps

Scene BLAS builds optionally use `VK_KHR_opacity_micromap` and its
`VK_KHR_device_address_commands` dependency. `DeviceDesc::enableOpacityMicromap`
defaults to true; `DeviceCapabilities::opacityMicromap` reports actual availability.
Unsupported devices retain ordinary shader alpha traversal.

## Scene bake and traversal

`OpacityMicromapBake` computes conservative coverage from the same base-color
alpha, factor, cutoff, UV transform, nearest mip-0 and wrap sampling used by the
scene ray queries. Each mixed triangle is subdivided to level 4 (256
microtriangles), limited by the device. Fully uniform triangles collapse to level
0. Four-state data uses Vulkan's space-filling curve order and least-significant
bit packing. UV bounds include every potentially sampled texel; mixed bounds
remain unknown rather than classifying only a few point samples.

- MASK: below cutoff is transparent, above/equal cutoff is opaque; mixed bounds
  invoke the existing shader alpha test.
- BLEND: zero alpha is transparent, one is opaque, intermediate alpha remains
  unknown. The renderer still treats accepted partial-alpha hits as opaque;
  this change does not add continuous alpha compositing or transmittance.
- Neural textures, unavailable CPU image data, divergent instance materials,
  and bakes exceeding the memory budget keep the shader fallback. A single
  material's coverage tables are bounded to 128 MiB and scene bake data to 256 MiB.

`SceneAccelerationStructureBuilder` uploads packed states and triangle records
with 128-byte alignment, creates an opacity-micromap AS using
`vkCreateAccelerationStructure2KHR`, builds it with
`vkCmdBuildAccelerationStructuresKHR`, then attaches it to the triangle geometry
of the BLAS. Build barriers order OMM, BLAS, compaction and TLAS work. Upload
buffers retire after GPU completion; OMM AS storage remains alive with the BLAS,
including compacted BLAS. Scene AS statistics include resident OMM memory.

The shared scene resources expose the resulting TLAS to real-time shadows,
SIGMA, path tracing, ReSTIR and material visualization. Alpha factor/cutoff/mode
and texture-transform changes invalidate the baked scene resources. Rigid
instance transforms continue to use TLAS refits.

## Shader and SDK compatibility

KHR ray queries require `SPV_KHR_opacity_micromap` and the
`OpacityMicromapIdKHR` execution mode enabled with a true Boolean constant.
The Vulkan backend applies this to RayQuery SPIR-V at shader module/shader object
creation, after compiler cache lookup. Driver binaries, hashes and Aftermath
registration use the resulting device-specific code. Non-RayQuery modules are
unchanged. An explicit device opt-out leaves shader binaries untouched.

`VulkanOpacityMicromap.h` supplies the small set of KHR declarations missing
from the current 1.4.350 SDK. It uses KHR AS objects, not the legacy EXT micromap
API. No vendored dependencies are modified.

Use validation layers **1.4.357 or newer** for OMM development. The 1.4.350 layer
rejects the new structures/enumerants before calls reach the driver. With that
older layer enabled, Metallic explicitly logs the limitation and keeps shader
alpha traversal. A newer layer can be selected for a process with `VK_LAYER_PATH`;
there is no need to replace the SDK to compile the engine.

## Validation

```powershell
cmake --build build --target MetallicRhiTests --parallel 6
build/tests/MetallicRhiTests.exe --filter opacity_micromap --rhi-validation
```

`opacity_micromap_bake` checks conservative states, subdivision packing,
constant triangles, cutoff equality and partial-alpha semantics.
`opacity_micromap_ray_query` compares 4096 GPU rays with OMM disabled/enabled,
checks candidate reduction, and repeats after BLAS compaction and cutoff, imported
UV transform, alpha-factor and BLEND edits. It saves visibility PNGs and skips
KHR execution when the effective capability is unavailable.

Official references: [Vulkan KHR OMM](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_opacity_micromap.html),
[SPIR-V KHR OMM](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_opacity_micromap.asciidoc).

Validated on RTX 5070 Ti / driver 616.64 with a separately built Khronos
1.4.357 validation layer: 20,480 probe rays preserved visibility, while shader
alpha candidates fell from 20,480 to 4,224 (79.4%). No Vulkan validation messages
were emitted by the OMM tests. Position-fetch/BLAS/realtime-shadow regressions
(4 tests) and the SIGMA suite (9 tests) also passed.
