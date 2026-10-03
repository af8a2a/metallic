# DR resource access ABI

The current migration follows the DR-first design: shader resource references
are 32-bit indices, ordinary buffer positions are descriptor-relative spans,
and physical pointers remain an explicit capability. This document supersedes
the ordinary-data BDA direction in `SharedResourceRegistry.md` for migrated paths.

## Implemented first slice

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
- FinalBlit, SliderDebug (including the DLSS overlay), ColorGradingLUT and
  AutoExposure use the new handles. AutoExposure's histogram, history and
  exposure records use raw descriptor buffer loads/stores, with no ordinary-data
  physical pointer. Their parameter ABI IDs are version 2.
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

`bufferSpan` accepts an optional `BufferRange`. All spans of one allocation
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

## Remaining migration

This is a working vertical slice, not completion of the repository-wide design.

1. Migrate lighting, scene/material data, GPU-driven/streaming and SDK adapters
   from the legacy `ShaderResourceHandle`/`DataSpan`/slot APIs. Remove the legacy
   representations only after all callers and GPU regressions are converted.
2. Integrate and validate indexed AS resolution, including partitioned AS. The
   legacy AS resolver still carries a full device address because of the native
   heap load issue documented in `AsHandleInvestigation.md`. Do not pass an AS
   address as a `ResourceHandle` or use the new generic resolver for AS yet.
3. Add explicit SRV/UAV/CBV view registration and descriptor range identity where
   needed. The first span implementation shares a full raw storage-buffer view.
4. Reserve and initialize a universally valid null slot, then change defaults.
   For now `UINT32_MAX` means invalid; slot zero remains allocatable. An invalid
   handle must never be resolved. Default-constructed spans have count zero.
5. Add transient descriptor arenas retired by actual submission completion, while
   preserving persistent stable indices. Current descriptors use registry leases
   and deferred collection; frame parameter arenas already follow completion.
6. Complete named pass parameters and remove legacy slots. ParameterRoot and
   true GPU-VA/indirect API operations remain explicit physical-address boundaries.
   ColorGradingLUT currently retains the existing immutable parameter-root transport.

Attachment views remain separate from shader views. No D3D12 backend or physical
pointer emulation is introduced by this slice. The default mapped compiler mode
is unchanged; it implements the same DR-facing ABI through Vulkan heap mappings.

## Validation entry points

Build `MetallicRHITests` and `Metallic` in a compatible configured MSVC tree.
Run the following filter with `--rhi-validation` in both `mapped` and `native`
`METALLIC_SLANG_DESCRIPTOR_MODE` environments:

```text
*resource_abi_spans*:*post_process_parameter_spirv_layout*:*final_descriptor_indices_heap_switch*:*native_descriptor_heap*:*auto_exposure*:*render_graph_final_blit*:*slider_debug_hdr_pixels*:*color_grading_unreal_lut_aces2*:*registry_*
```

The new probe checks distinct descriptors within a wave, nonzero subrange offsets,
nested typed raw loads, bounds/guard values, shared descriptor identity and GPU
allocation lifetime after CPU wrapper destruction. The existing parameter layout
test compares actual emitted SPIR-V member offsets to C++ offsets in both modes.

Compiler option reference: [Slang compilation options](https://docs.shader-slang.org/en/stable/external/slang/docs/user-guide/08-compiling.html).
Descriptor size semantics: [Vulkan shader descriptor sizes](https://docs.vulkan.org/spec/latest/chapters/interfaces.html).

## Verified on 2026-10-03

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
