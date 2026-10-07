# Bindless compute resources

All application compute shaders that previously declared `vk::binding` now import `Core`. Its ABI is declared in
`Shaders/Modules/Core/ComputeResources.slang`. This covers RTXDI/ReGIR,
Standard and OpenPBR path tracing and guides, SHaRC/NRC, neural textures,
visibility-buffer deferred shading/material binning, environment precomputation,
post processing and debug passes. NRD already uses its own native bindless ABI.

## Shader and runtime contract

The current resource representation and compute migration are specified in
[ResourceAccessABI](ResourceAccessABI.md#compute-execution-layers). The root is a
12-byte descriptor-relative span; named-resource shaders load a 24-byte pair of
resource and constant spans. Numeric slot tables and ordinary-data BDA access
are retired.

`ComputeKernel` is the sole application compute executable. Typed parameters and
the non-executable `ComputeResourceEncoder` migration adapter both produce
`EncodedParameters`, then `PreparedComputeDispatch`. New passes use typed
parameters directly. Production `ComputeProgram` no longer exists.

`ParameterWriter` retains resource leases and immutable uploads. Frame packets
reject other frame generations; standalone packets own their storage. Prepared
packets retain the kernel and indirect allocation slices. The full batch's device,
ABI and frame scope are checked before recording its first dispatch. Resource
barriers remain a RenderGraph/caller responsibility.

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

Use `MetallicRHITests --rhi-validation --gtest_filter=<filter>` with filters such
as `*rtxdi*`, `*frame_descriptor_snapshots*`,
`*material_binning_indirect_coverage*`, `*pathtracing_guides_shader_compile*`,
`*scene_ray_tracing_position_fetch*` and `*visibility_buffer_deferred_openpbr*`.
The complete RTXDI preview also requires `METALLIC_ENABLE_NRD=ON`.

On 2026-09-11, the Debug `Metallic` and `MetallicRHITests` targets built with NRD
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
