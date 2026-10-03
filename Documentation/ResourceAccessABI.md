# DR resource access ABI

The current migration follows the DR-first design: shader resource references
are 32-bit indices, ordinary buffer positions are descriptor-relative spans,
and physical pointers remain an explicit capability. This document supersedes
the historical ordinary-data BDA direction in `SharedResourceRegistry.md`.

## Image/buffer migration

- `ShaderResourceABI.h` defines 4-byte `GPUResourceHandle` and `GPUSamplerHandle`,
  12-byte `GPUBufferSpan`, and an explicitly separate `PhysicalPtr<T>`.
- `ShaderCore` owns `ResourceHandle<T>`, `SamplerHandle`, `BufferSpan<T>` and
  `RWBufferSpan<T>`. `resolveUniform` and `resolveNonUniform` isolate compiler
  descriptor handles and heap access. Business shaders do not store Slang's
  `DescriptorHandle<T>` representation in the new ABI.
- Vulkan image and buffer regions share the maximum of their aligned descriptor
  sizes as their index stride. Individual descriptor writes
  retain their native byte sizes. Capacity, offsets, shader indices and mappings
  for compute/graphics/shader objects use the same stride. Native compilation
  enables `SPIRVUnifiedDescriptorHeapStride`; shader cache request version is 24.
  Samplers retain their separate heap and stride.
- Before creating a Vulkan shader module, `DescriptorHeapSPIRV.h` resolves
  `OpConstantSizeOfEXT` for image/buffer/sampler types to the device's aligned
  descriptor sizes. The generic disk cache remains device-independent; the
  existing pipeline content hash and shader-object code use the specialized
  device binary. This avoids the observed native task/mesh payload validation
  failure caused by specialization expressions referencing opaque sizes.
- All renderer image/buffer accesses use DR: postprocessing, lighting, scene and
  material data, GPU-driven culling/raster/streaming, path tracing, RTXDI, SHaRC,
  and the maintained NRD adapter. Production shaders construct engine handles
  and explicitly resolve resources; Slang descriptor representation stays in Core.
- Lighting and material binning use raw bounded spans, including atomic updates
  to bin counts. Typed StructuredBuffer DR objects remain where appropriate,
  including SDK interfaces; DR does not require changing their data layout.
- `ParameterTransport::DescriptorBuffer` replaces `DeviceAddress`. The root push
  payload is a 12-byte `GPUBufferSpan` (index, byte offset, word count).
  `ParameterRoot.getParameters<T>()` performs a raw descriptor load; neither large
  parameter blocks nor resource/constant/texture-index tables use BDA.
- `ComputeProgram` remains a compatibility adapter for numeric logical slots.
  Its 24-byte root contains two DR spans; each 24-byte slot carries a resource
  value (only AS needs both words), a DR payload span and a checked data stride.
  Image arrays contain 32-bit indices, and indirect-batch constants use bounded
  offsets within one immutable upload. `getData<T>` rejects stride mismatches.
- `ShaderDataSpan`, `DataSpan`, `dataBuffer`, and address-returning `data` /
  `EncodedParameters::address` APIs are removed. `dataSpan` and `sampledImages`
  return descriptor spans. BufferSlice registration retains the allocation after
  its movable Buffer wrapper disappears and shares identity with Buffer registration.
- Mapped device selection and creation now require and enable storage-buffer
  nonuniform indexing, alongside the existing image indexing features.

```cpp
ParameterWriter writer(device, registry, commands.frameContext());
params.source = writer.sampledImageHandle(source.view());
params.output = writer.storageImageHandle(output.view());
params.histogram = writer.bufferSpan<uint32_t>(histogram.buffer());
auto packet = writer.encode(params, kAutoExposureABI, ParameterTransport::InlinePush);
```

```slang
Texture2D<float4> source = resolveUniform(params.source);
uint value = params.histogram.load(index);
params.histogram.store(index, value + 1);
// A wave-divergent resource identity must use resolveNonUniform,
// loadNonUniform or storeNonUniform explicitly.
```

`bufferSpan` accepts a `BufferSlice` or a Buffer with an optional `BufferRange`. All spans of one allocation
share its full raw buffer descriptor; the byte offset and element count stay in
the packet. Validation rejects empty, misaligned, nonintegral or out-of-allocation
ranges and ranges exceeding the 32-bit byte-addressing domain. It requires Storage
usage, not a caller-provided GPU address or an explicit ShaderDeviceAddress flag.
Out-of-span loads return zero and stores do nothing. Bounds checks use the count
authored by the CPU; arbitrary GPU-authored span arithmetic must preserve these
invariants itself. This is not a claim that device-wide robust buffer access is enabled.

Registration and encoding keep existing registry provenance and submission
leases: destroying the CPU wrapper after recording does not release the GPU
allocation early. RenderGraph dependencies, access scopes and queue ownership
remain explicit and independent of descriptor resolution.

## Boundaries and remaining architecture work

- ImGui's public Vulkan texture/descriptor-set ABI remains an external boundary
  in `EditorDisplayRenderer` / `EditorDisplay.slang`, including detached windows
  and HDR UI composition. This renderer migration does not replace ImGui's
  Vulkan backend. Attachments, transfers, vertex/index and indirect API bindings
  are not shader resource descriptors.
- AS retains its full device-address resolver because of the native heap load
  issue documented in `AsHandleInvestigation.md`. Do not truncate an AS address
  to `ResourceHandle`. Genuine GPU-VA API operations remain physical-address
  capabilities; ordinary shader data and parameter roots no longer use them.
- Range-specific SRV/UAV/CBV views, universal null slot, separate transient
  descriptor arenas, and removal of logical ComputeProgram slots remain future
  architecture work. They are not required to make existing image/buffer accesses
  use DR. Invalid indices remain `UINT32_MAX`; zero is still allocatable.
- Descriptors retain stable registry identity and submission leases. Parameter
  arena backing buffers now each retain one descriptor; growing an arena never
  relocates live indices. Completed arenas keep that descriptor for reuse.
- No D3D12 backend or physical-pointer emulation is introduced. Both mapped and
  native Vulkan modes implement this same shader-facing ABI.

## Validation entry points

Build `MetallicRHITests` and `Metallic` in a compatible configured MSVC tree.
Run the following filter with `--rhi-validation --rhi-bindless` in both `mapped` and `native`
`METALLIC_SLANG_DESCRIPTOR_MODE` environments:

```text
*parameter_spirv_layout*:*resource_abi_spans*:*registry_*:*buffer_slice*:*material_binning*:*cluster_light_grid*:*environment*:*regir*:*binding_nonuniform_images*:*scene_ray_tracing_position_fetch*:*render_graph_gpu_driven_mixed_producer_render:*prepared*:*bindless*:*image_sample*:*material_shader_object*:*bunny_wireframe*:*hzb*:*color_grading*:*auto_exposure*:*slider_debug*:*final_descriptor_indices_heap_switch*:*native_descriptor_heap*
```

The new probe checks distinct descriptors within a wave, nonzero subrange offsets,
nested typed raw loads, bounds/guard values, shared descriptor identity and GPU
allocation lifetime after CPU wrapper destruction. The parameter layout test compares storage SPIR-V member offsets to C++ in both
modes. The raw-loaded LUT type uses an explicit shared-type layout probe plus
the existing LUT pixel test, because value types need no SPIR-V Offset decoration.
Registry root probes also reject PhysicalStorageBufferAddresses capability.

Compiler option reference: [Slang compilation options](https://docs.shader-slang.org/en/stable/external/slang/docs/user-guide/08-compiling.html).
Descriptor size semantics: [Vulkan shader descriptor sizes](https://docs.vulkan.org/spec/latest/chapters/interfaces.html).

## Full image/buffer and parameter-root verification on 2026-10-03

- Reused `build-pass-stages-nrd` (MSVC Release, NRD/tests enabled); built
  `Metallic`, `MetallicRHITests`, `MetallicNRDTests` and `MetallicShaderCompiler`.
- The filter above passed 65/65 in each of mapped and native modes, with no
  skips or VUIDs. This includes GPU readback/rendering and static layout checks;
  the two runs cover the same 65 tests, not 130 distinct cases.
- `MetallicNRDTests` passed 11/11 in each mode, including supported shader
  permutations, Reference accumulation/reset, REBLUR/RELAX radiance, SIGMA
  shadow and ray-traced occlusion/history readback.
- Manual shader warmup completed 210 requests with zero failures in each mode
  (42 existing cache hits mapped, 3 native). Both subsequent editor smoke runs
  reported 210/210 cache hits, submitted and presented a frame, and exited 0.
- Coverage also includes bounded/nonuniform spans, retained BufferSlice copies,
  indirect consumption, parameter arena reuse and parallel submission, material
  binning, HZB, GPU-driven mixed producers, ReGIR with Standard/OpenPBR/RTXDI,
  exposure history, LUT/slider pixels, environment prefiltering and scene ray
  queries. The exported environment capture and material preview were inspected.
- Evidence is under `.cache/dr-resource-abi/`: `migration-build.log`,
  `migration-final-{mapped,native}.{xml,log}`,
  `migration-nrd-final-{mapped,native}.{xml,log}`,
  `migration-warmup-{mapped,native}.log` and `migration-smoke-{mapped,native}.log`.
  RHI image reports are under `rhi-test-output/reports/17909944514177921`
  (mapped) and `17909947225450556` (native).
- Existing broken Vulkan loader registrations emit GENERAL messages. These stay
  in logs; the NRD and prepared-view test sinks count API VALIDATION messages
  separately. No machine-wide loader configuration was changed.

These checks do not establish extended editor interaction, full-scene memory
stability or optional Streamline/DLSS/NRC runtime correctness. The external ImGui
descriptor-set ABI and AS address boundary described above remain explicit.

## Earlier first-slice verification on 2026-10-03

- Reused `build-pass-stages-nrd` (MSVC Release, tests/NRD enabled,
  Streamline disabled). `MetallicRHITests` and `Metallic` build successfully.
- Final validation-enabled runs: mapped 36/36, native 36/36, no skips and no
  VUIDs. Each run contains 26 RHI checks and 10 SPIR-V static checks; these are
  the same checks in two environments, not 72 distinct tests.
- Both editor `--smoke-test` runs exited 0 and submitted/presented a frame.
  Their shader warmup each reported 210 requests, 65 existing cache hits and
  zero failures. Logs are `smoke-mapped.log` and `smoke-native.log` in the same
  evidence directory. This covers startup/presentation, not extended interaction.
- Coverage includes the new nonuniform span probe, CPU/SPIR-V field offsets,
  exposure adaptation/history/cancellation, LUT and HDR slider pixels, FinalBlit,
  registry retention/multi-queue submission, mixed-width atomics, material binning
  and indirect consumption, mixed-producer raster, BLAS compaction/TLAS refit
  and position-fetch normals/tangents. FinalBlit's exported solid-color image
  was also visually inspected.
- Unified-stride validation regression was isolated against HEAD's original
  `SlangCompiler.cpp` and `VulkanRHI.cpp`. That backend/compiler baseline produced
  zero payload VUIDs; the initial unified implementation produced 8 task-payload
  and 4 mesh-payload VUIDs. Device size specialization removed them while retaining
  unified indices and passing the same GPU workload. Temporary baseline source
  replacement was restored byte-for-byte and the current implementation rebuilt.
- Local evidence: `.cache/dr-resource-abi/verified-{mapped,native}.{xml,log}`,
  `baseline-native.log`, `cross-native.log`, `specialized-native.log` and
  `delivery-build.log`. Initial failures remain available for audit. Broken local
  Vulkan loader manifest registrations still emit GENERAL messages; they are
  separate from API validation and were not changed by this work. Slider's
  invalid-input test deliberately logs a rejected pass.

No performance improvement, full-scene memory stability, D3D12 support,
partitioned-AS migration or interactive DLSS/NRC verification is claimed.
