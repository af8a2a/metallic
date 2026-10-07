# Static ray-tracing coverage acceleration

Applications provide geometry and static coverage semantics to the RHI. The
Vulkan implementation optionally accelerates coverage with opacity micromaps
(OMM); micromap formats, subdivisions, usage histograms, storage, build commands
and attachments are private to the backend. Unsupported devices keep ordinary
shader alpha traversal.

## Public RHI contract

`RayTracingCoverageDesc` describes MASK/BLEND mode, alpha factor/cutoff, CPU RGBA8
pixels and transformed UVs in original BLAS triangle order. Its sampling contract
matches `AlphaCoverage.hlsli`: four-tap bilinear, repeat wrapping and the supplied
finest-resident image. Empty pixels mean constant texture alpha one. Missing,
dynamic or otherwise unavailable coverage is represented by a null geometry
`coverage` pointer.

`Device::prepareRayTracingBottomLevelBuild()` consumes the CPU spans
synchronously and returns an opaque `RayTracingBottomLevelBuildPlan`. The plan
freezes geometry, flags, sizes and internal coverage resources without submitting
GPU work. Applications create the BLAS with
`createRayTracingAccelerationStructure(plan)` and record its build with
`RayTracingAccelerationStructureBuildDesc::plan` using the plan's scratch size.
CPU coverage inputs may be released after preparation. The plan may be released
after recording: commands retain build inputs through completion, and the BLAS
owns resident coverage dependencies. The device must outlive its plan and BLAS.
A plan is consumed by its first successful Build recording, including when that
recording is later abandoned. Reusing it or recording an Update with it is
rejected; a fresh build needs a fresh plan. `AllowUpdate` preparation keeps shader
coverage traversal so mutable BLAS data does not reuse immutable coverage.

Ordinary BLAS builds remain available through geometry descriptors. Static
coverage must use the prepared-plan path so size queries and recording cannot
choose different acceleration data. Builds and compaction stay in caller-owned
command buffers; preparation does not submit hidden GPU work.
Compaction of a coverage BLAS requires a fresh destination created from the raw
AS descriptor. Once that destination owns coverage dependencies, another
compaction into it is rejected so recorded or existing traversal cannot lose
its resident coverage resources.

The scene layer resolves materials, transformed UVs, CPU image data and whether
instances can share the same static coverage. It does not bake, allocate, upload
or attach micromaps. Shared resources expose the resulting TLAS to real-time
shadows, SIGMA, path tracing, ReSTIR and material visualization. Coverage changes
invalidate scene resources; rigid instance transforms continue to use TLAS
refits. Public statistics use `coverageAccelerationCount`,
`coverageAcceleratedTriangleCount` and `coverageAccelerationBytes`.

## Vulkan implementation

The private `GAPI/Vulkan/OpacityMicromapBake` computes conservative bounds over
all potentially sampled texels. Mixed triangles normally use subdivision level
4, limited by the device; uniform triangles collapse to level 0. Four-state data
uses Vulkan's space-filling curve order and least-significant-bit packing.

- MASK: below cutoff is transparent, above/equal cutoff is opaque; mixed bounds
  invoke the existing shader alpha test.
- BLEND: zero alpha is transparent, one is opaque, intermediate alpha remains
  unknown and invokes the existing shader Coverage evaluation.
- Neural textures, unavailable CPU image data, divergent instance materials and
  bakes exceeding memory limits keep shader traversal. CPU prefix tables are
  bounded to 128 MiB per coverage source; packed bake data retained by live plans
  and recordings shares a 256 MiB per-device budget. Scene CPU image/UV snapshots
  additionally share a 256 MiB budget per scene preparation.

The backend uploads states and triangle records, builds the OMM, orders it before
the BLAS and retains storage with the BLAS. Compaction inherits those resident
dependencies without retaining the retired source BLAS allocation. Upload
buffers retire after recorded GPU work completes.

Backend diagnostics can opt out through
`vulkan::deviceExtensions(desc).enableOpacityMicromap` (default true) and inspect
`vulkan::deviceCapabilities(device).opacityMicromap`. Applications do not need
either to provide coverage. Vulkan selects KHR OMM and its device-address-command
dependency, or the EXT compatibility route when capture/injection requires it.

KHR ray queries require `SPV_KHR_opacity_micromap` and the `OpacityMicromapIdKHR`
execution mode; EXT uses its corresponding capability. The backend patches
RayQuery SPIR-V at shader module/object creation after compiler cache lookup.
Driver binaries, hashes and Aftermath registration use device-specific code.
Non-RayQuery modules and an opted-out device keep the original code.

`VulkanOpacityMicromap.h` supplies KHR declarations missing from SDK 1.4.350.
No vendored dependencies are modified. For the KHR route, use validation layers
1.4.357 or newer: the older 1.4.350 layer rejects new structures before they reach
the driver, and Metallic keeps shader traversal when that layer is active. A
newer layer can be selected per process with `VK_LAYER_PATH`.

## Validation

Build the selected executable before running it, using a compatible configured
build directory. For example:

```powershell
cmake --build build --target MetallicRHITests --parallel 6
build/tests/MetallicRHITests.exe --filter opacity_micromap --rhi-validation
```

`opacity_micromap_bake` checks conservative states, packing, constant triangles,
cutoff equality and partial-alpha semantics through the private backend baker.
`opacity_micromap_scene_coverage_snapshot` checks indexed transformed UV order and
input independence after editable source geometry, material, images and the scene
are changed or destroyed.
`opacity_micromap_ray_query` compares 4096 GPU rays with backend OMM disabled and
enabled, checks ordinary-plan fallback, candidate reduction, compaction, cutoff,
imported UV transform, alpha-factor, BLEND edits and TLAS refits.
`opacity_micromap_ray_query_partitioned` additionally covers top-level backend
switches.

The 2026-10-07 RTX 5070 Ti / driver 617.42 retest passes both variants in mapped
mode. Native stops at the OMM-disabled baseline's bilinear alpha assertion,
before testing the OMM-enabled branch; this is not evidence of an OMM-specific
execution failure. See the [native driver retest](NativeDescriptorHeapStrideWorkaround.md#driver-61742-follow-up)
for the matched-toolchain comparisons and remaining validation limits.

`opacity_micromap_build_plan_lifetime` releases CPU inputs before recording and
the plan/vertex/scratch wrappers before submission, compacts the BLAS, destroys
the source allocation and then checks six analytic hit/miss rays. Geometry is
nonopaque and coverage is fully opaque; zero shader candidates verify that the
surviving BLAS still uses its coverage acceleration. This GPU case requires
ray queries, bindless descriptors and effective OMM support; a skip does not
validate the lifecycle.

The prepared-plan refactor was validated on 2026-10-07 with RTX 5070 Ti / driver
616.92 using `build-scheduling-release`:

- `Metallic`, `MetallicRHITests` and `MetallicVulkanDeviceFeaturesTests` built;
  the Vulkan feature/property unit tests passed.
- All five `opacity_micromap` cases passed with `--rhi-no-validation`; KHR OMM
  was enabled and none skipped. Visibility matched shader fallback through the
  standard and partitioned paths. Aggregate candidate counts were 24,576 to
  6,912 and 32,768 to 10,688 respectively; these are traversal counters, not
  frame-time measurements. The compaction lifetime and overwrite rejection
  checks also passed.
- Nineteen related RHI/scene/material cases passed, including both raster/RT/
  shadow Coverage comparisons and Sponza asynchronous RTAS upload. The final
  compaction guard was additionally verified by rerunning the seventeen RHI/
  scene cases in the final binary.
- With the installed 1.4.350 validation layer, ten selected cases passed without
  VUID errors; the three OMM GPU cases skipped because that layer disables KHR
  OMM. Actual OMM execution above was checked without the validation layer.

The pre-refactor integration baseline on RTX 5070 Ti / driver 616.64 with a
separate Khronos 1.4.357 validation layer preserved 20,480 probe rays and reduced
alpha candidates from 20,480 to 4,224 (79.4%). That historical result does not
validate the prepared-plan implementation or its new lifecycle test.

Official references: [Vulkan KHR OMM](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_opacity_micromap.html),
[SPIR-V KHR OMM](https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_opacity_micromap.asciidoc).
