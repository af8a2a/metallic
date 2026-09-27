# Bindless compute resources

All application compute shaders that previously declared `vk::binding` now import `Core`. Its ABI is declared in
`Shaders/Modules/Core/ComputeResources.slang`. This covers RTXDI/ReGIR,
Standard and OpenPBR path tracing and guides, SHaRC/NRC, neural textures,
visibility-buffer deferred shading/material binning, environment precomputation,
post processing and debug passes. NRD already uses its own native bindless ABI.

## Shader and runtime contract

`ComputeKernel` owns executable code and its `ParameterAbi`. Direct dispatch,
indirect dispatch and immutable prepared batches all record through
`PreparedComputeDispatch`. `ComputeProgram` is a resource-table input encoder for
existing Core shaders; it has no private heap, descriptor-table pool or separate
command-recording implementation.

- The application push data is one 64-bit `ParameterRoot` address for both typed
  kernels and Core resource tables. The RHI adds its heap header where required.
- Core loads `ComputeResourceParameters {resources, constants}` from that root.
  Existing `getResource<T>()`, `getResourceArray<T>()`, `getData<T>()` and
  `getConstants<T>()` accessors continue to work. CPU parameter layouts remain
  explicit; all affected shaders must be recompiled for the new root ABI.
- `ComputeProgramBindingDesc::binding` selects an application slot in `[0, 255]`,
  independent of Vulkan descriptor binding or layout declaration order.
  A slot is 16 bytes: a canonical handle and a payload. Image-array payloads point
  to immutable lists of handles, without requiring contiguous descriptor indices.
  DataBuffer slots hold a device address plus element count and stride.
- Acceleration structures retain the full 64-bit device address. Ordinary and
  partitioned top-level structures use the same resource type and resolver.
- `ParameterWriter` owns resource leases and immutable uploads. Frame writers use
  a submission arena and packets reject other frame generations. Standalone
  writers own their storage; command buffers retain recorded packets, pipelines
  and indirect allocations. Commands/pools must not be reset before GPU completion.
- Every indirect item has its own encoded constants and argument slice. The
  entire batch's device, parameter ABI and frame scope are checked before its
  first dispatch. Resource barriers remain a RenderGraph/caller responsibility.
- `usesResourceTable`, `resourceTableCount` and `resourceTableIndex` are removed.
  Legacy descriptor/push mapping experiments use a test-local raw RHI fixture.
  Mapped/native descriptor lowering is selected by `SlangDescriptorHeapMode`,
  independently of the common compute parameter ABI.

The editor's ImGui backend and closed Streamline SDK retain their external
descriptor-set interoperability. Those are separate from the application shader
resource interface.

The Slang conventions are described in
[DescriptorHandle and acceleration structures](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/03-convenience-features.html);
Vulkan's common-array mapping is described in
[Descriptor Heaps](https://docs.vulkan.org/spec/latest/chapters/descriptorheaps.html).

## Validation

`compute_kernel_prepared_standalone_batch` checks typed direct/indirect packets,
standalone storage, ABI rejection, a stale tail before recording any batch prefix,
and wrapper destruction. `prepared_dispatch_parallel_snapshot_lifetime` also
checks equivalent layouts declared in different orders. Both exercise mapped
and native lowering.

The results below are historical validation records, not the current run.

The RTXDI, OpenPBR and position-fetch shader tests inspect SPIR-V to reject
scalar/fixed-size descriptor bindings and require native heap arrays and
address-based compute data. Frame descriptor snapshots cover repeated dispatches, concurrent frames and
clearing a program with GPU work pending. The
material-binning GPU test covers indirect batches with per-dispatch constants
and compatible shader permutations.

Use `MetallicRhiTests --rhi-validation --gtest_filter=<filter>` with filters such
as `*rtxdi*`, `*frame_descriptor_snapshots*`,
`*material_binning_indirect_coverage*`, `*pathtracing_guides_shader_compile*`,
`*scene_ray_tracing_position_fetch*` and `*visibility_buffer_deferred_openpbr*`.
The complete RTXDI preview also requires `METALLIC_ENABLE_NRD=ON`.

On 2026-09-11, the Debug `Metallic` and `MetallicRhiTests` targets built with NRD
enabled. Thirty-one distinct focused RHI tests passed, including the full
RTXDI/confidence/RELAX/composite preview, temporal RTXDI, textured Standard/OpenPBR
path tracing, position fetch, visibility-buffer deferred shading, indirect
batches, frame lifetimes, SH, exposure and SHaRC/NRC lighting. NTC with and without
cooperative vectors and all four SHaRC/NRC shader permutations compiled. NTC
execution was not exercised by these tests.

The broad GPU run used `METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled` for the
previously documented driver-cache diagnostic. RTXDI/RELAX and descriptor
snapshots also passed with the default driver cache setting, as did the editor
`--smoke-test`. Reports and images are under `.cache/native-compute-validation/`.

`radiance_cache_lighting` passed its rendering assertions, but emitted
`VUID-vkDestroyDevice-device-05137` warnings for undestroyed `VkBuffer` objects
during teardown. The source of those warnings has not been established; this
run is not a validation-clean result for NRC teardown. The other focused
regressions did not emit Vulkan validation warnings.
