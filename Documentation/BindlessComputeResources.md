# Bindless compute resources

All application compute shaders that previously declared `vk::binding` now use
`Shaders/Libraries/Resources/ComputeResources.slang`. This covers RTXDI/ReGIR,
Standard and OpenPBR path tracing and guides, SHaRC/NRC, neural textures,
visibility-buffer deferred shading/material binning, environment precomputation,
post processing and debug passes. NRD already uses its own native bindless ABI.

## Shader and runtime contract

- `ComputeProgramBindingDesc::binding` is an application slot in `[0, 255]`.
  It indexes a table of `uint2` handles, not a Vulkan descriptor binding.
- Image, buffer and sampler slots contain an absolute `BindlessHandle::shaderIndex`
  in `x` and zero in `y`. Texture arrays use a contiguous range starting at that
  index. Their handles are marked `nonuniform` before dereferencing.
- Acceleration-structure slots contain the full 64-bit device address. The
  bundled Slang 2026.1.2 interprets `DescriptorHandle<RaytracingAccelerationStructure>`
  as an address, not a descriptor index. Both ordinary and partitioned structures
  expose their addresses through the RHI. Do not truncate these to a heap index.
- The native RHI prepends an 8-byte heap header. ComputeProgram then pushes the
  resource-table and constants addresses (8 bytes each), for 24 bytes total.
  Shader entry points load their existing parameter types with
  `METALLIC_CONSTANTS`. Parameter field
  order and CPU/shader layouts are unchanged.
- `resourceTableCount` and `resourceTableIndex` select application resource tables.
  Each active frame retains its heap, table, constants and pipeline until GPU
  completion. Repeated use of a table in a frame gets another snapshot. Indirect
  batches share a table but give every dispatch its own constants address.
- Callers using a command buffer without a `RenderFrameContext` must keep the
  program/resources alive and complete previous uses before rewriting the table,
  with distinct table indices for overlapping dispatches, as before.

ComputeProgram defaults to this ABI and creates pipelines without per-resource
`ShaderBindingMappingDesc` entries. The bundled Slang lowers `DescriptorHandle`
to unbounded typed arrays; the RHI maps only those common sampler/resource arrays
to `VK_EXT_descriptor_heap`. This compiler does not support
`spvDescriptorHeapEXT` yet. The explicit `usesResourceTable = false` path remains
for the legacy descriptor-heap code-pattern diagnostic only.

The editor's ImGui backend and closed Streamline SDK retain their external
descriptor-set interoperability. Those are separate from the application shader
resource interface.

The Slang conventions are described in
[DescriptorHandle and acceleration structures](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/03-convenience-features.html);
Vulkan's common-array mapping is described in
[Descriptor Heaps](https://docs.vulkan.org/spec/latest/chapters/descriptorheaps.html).

## Validation

The RTXDI, OpenPBR and position-fetch shader tests inspect SPIR-V to reject
scalar/fixed-size descriptor bindings and require native heap arrays and
address-based compute data. Frame descriptor snapshots cover different table
indices, concurrent frames and clearing a program with GPU work pending. The
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
