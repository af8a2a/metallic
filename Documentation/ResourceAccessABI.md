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
- Scene renderer image/buffer accesses use DR: postprocessing, lighting, scene and
  material data, GPU-driven culling/raster/streaming, path tracing, RTXDI, SHaRC,
  and the maintained NRD adapter. EditorDisplay remains the ImGui boundary
  described below. Production scene shaders construct engine handles
  and explicitly resolve resources; Slang descriptor representation stays in Core.
- Lighting and material binning use raw bounded spans, including atomic updates
  to bin counts. Typed StructuredBuffer DR objects remain where appropriate,
  including SDK interfaces; DR does not require changing their data layout.
- `ParameterTransport::DescriptorBuffer` replaces `DeviceAddress`. The root push
  payload is a 12-byte `GPUBufferSpan` (index, byte offset, word count).
  `ParameterRoot.getParameters<T>()` performs a raw descriptor load; neither large
  parameter blocks nor resource/constant/texture-index tables use BDA.
- Production `ComputeProgram` shaders read named resource structs through
  `getResourceParameters<T>()`, then resolve their explicit handles/spans. The
  shared `NamedResourceParameters.h` declares the CPU/Slang wire fields.
  `NamedResourceLayouts.h` maps CPU input IDs to field offsets; those IDs never
  reach the production GPU packet. The scene block is 440 bytes, replacing the
  sparse 24-byte-per-slot table. Scalar images no longer allocate index arrays.
- `ComputeProgramDesc.resourceParameters` validates field type, bounds, alignment,
  overlap and array representation before pipeline creation. The encoder copies
  layout metadata and preserves immutable prepared dispatch/submission leases.
  Its 24-byte root still carries resource and constant DR spans. Image arrays
  contain 32-bit indices; named data spans count words and `typedBufferSpan<T>`
  validates divisibility before exposing typed elements. AS fields remain 64-bit.
- The slot adapter remains available for RHI test shaders, including rendering
  fixtures that have not yet adopted named parameters. There
  are no `getResource`, `getResourceArray` or `getData` calls in production
  Features/Interop or generated material source. CPU input IDs remain compatible
  with existing pass setup; they are not shader descriptor indices.
- `ShaderDataSpan`, `DataSpan`, `dataBuffer`, and address-returning `data` /
  `EncodedParameters::address` APIs are removed. `dataSpan` and `sampledImages`
  return descriptor spans. BufferSlice registration retains the allocation after
  its movable Buffer wrapper disappears and shares identity with Buffer registration.
- Mapped device selection and creation now require and enable storage-buffer
  nonuniform indexing, alongside the existing image indexing features.

```cpp
ParameterWriter writer(device, registry, RenderFrameContext::from(commands));
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
  descriptor arenas, and removal of the remaining CPU input/test adapter remain
  future architecture work. They are not required to make existing image/buffer accesses
  use DR. Invalid indices remain `UINT32_MAX`; zero is still allocatable.
- Descriptors retain stable registry identity and submission leases. Parameter
  arena backing buffers now each retain one descriptor; growing an arena never
  relocates live indices. Completed arenas keep that descriptor for reuse.
- No D3D12 backend or physical-pointer emulation is introduced. Both mapped and
  native Vulkan modes implement this same shader-facing ABI.

## Remaining shader audit on 2026-10-03

Static inspection covered 231 tracked shader/include files under `Shaders/`,
45 under `tests/rhi/shaders/`, and resource expressions in `Source/`, `Tools/`
and `scripts/`. Vendor submodule implementations are outside this inventory.
Comments were excluded from the legacy-call, descriptor-handle and fixed-binding
counts. These categories describe different migration layers; they are not all
evidence of non-DR resource access.

- `Features/PostProcess/EditorDisplay.slang` is the only fixed image/sampler
  binding found under `Shaders/`. Its two sets match `EditorDisplayRenderer`'s
  Vulkan layouts, ImGui draw callbacks and HDR10 output descriptor-set binding.
  Migrating it requires a coordinated editor/ImGui binding change, including
  detached windows; changing the shader declarations alone is insufficient.
- Production Features/Interop and generated material source have no legacy
  `getResource<T>`, `getResourceArray<T>` or `getData<T>` calls. Core retains
  their definitions. NRD's generated bindings resolve engine handles and its
  resource/constants tables use DR spans.
- The only physical pointer declaration found in Metallic's shader modules is
  the explicit `PhysicalPtr<T>` capability. It has no shader call sites. The AS
  resolver retains its separate 64-bit address path; 64-bit raster atomics and
  counters are buffer contents, not physical addresses.

The **24 test shaders still using the numeric-slot adapter** are listed below.
All names are relative to `tests/rhi/shaders/` and have the `.slang` suffix.
Their image/buffer accesses already resolve through DR in `ComputeResources`;
the remaining migration is to named parameter fields with matching CPU layouts.

| Area | Shaders |
| --- | --- |
| Postprocess/debug | `AutoExposureFixture`, `HZBSPDFixture`, `SliderDebugFixture` |
| Lighting/environment | `ClusterLightGridLookupProbe`, `EnvironmentPrefilterFieldProbe`, `FrameEnvironmentProbe`, `PhotometricProbe`, `ReGIRVirtualLightProbe`, `SphericalHarmonicsProbe` |
| Materials/guides | `DLSSMotionVectorProbe`, `MaterialBinningProbe`, `MaterialRuntimeProbe`, `RealtimeGuideProbe` |
| Geometry/view | `GPUDrivenConeProbe`, `TwoPassOcclusionProbe`, `UnifiedTopLevelProbe`, `ViewConstantsProbe` |
| Textures/uploads | `SceneUploadProbe`, `TextureFootprintProbe`, `TextureStreamingProbe` |
| Resource/dispatch fixtures | `BatchBarrierProbe`, `DataSliceProbe`, `FrameResourceProbe`, `NativeDescriptorHandles` |

Another category contains **12 test shaders exposing Slang `DescriptorHandle`**:

- Algorithm/render fixtures that can adopt engine handles are `HybridClusterProbe`,
  `HybridRasterProbe`, `PreparedRasterProbe`, `StreamClusterClassificationProbe`,
  `StreamMeshProbe`, `TessellationSplitProbe` and `WaveWorkProbe`.
- Backend/debug probes are `FinalDescriptorIndices`, `NativeDescriptorAtomics`,
  `NativeDescriptorHandles`, `ShaderPrintfEcho` and `ShaderTraceFixture`.
  Preserve their compiler/heap/debug coverage when changing their spelling.
  In particular, `NativeDescriptorHandles.unsafeNativeAsMain` deliberately bypasses
  the AS resolver and must be rejected by compilation; it is not a production leak.

`NativeDescriptorHandles` appears in both categories. Two additional tests use
fixed bindings: `DescriptorHeapCodePattern` and `GeneratedCommandsProbe`.
Their C++ fixtures explicitly create conventional Vulkan descriptor layouts;
these are separate from the production named-resource migration.

This follow-up audit changed documentation only. It did not rerun compilation or
GPU tests; runtime results below belong to the preceding implementation validation.

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

## Named production resource parameters: verification on 2026-10-03

The remaining numeric shader lookups are migrated to shared named fields, for example:

```slang
let resources = getResourceParameters<SceneResourceParameters>();
StructuredBuffer<uint> indices = resolveBuffer<StructuredBuffer<uint>>(resources.indices);
Texture2D<float4> materialTexture = resolveNonUniform(
    ResourceHandle<Texture2D<float4>>(resources.materialTextures.load(textureIndex)));
```

CPU program initialization supplies the matching `resourceParameters` layout.
Path tracing, deferred lighting, RTXDI, scene visualization, stream RTAS
visualization, shadows, NTC, material-value generation, visibility materials,
upscaler guides and GPU probes use this path. New standalone passes can continue
to use `ComputeKernel` with a directly authored typed parameter packet.

- Reused the same MSVC Release/NRD-enabled tree. Final renderer, RHI and NRD
  executables build successfully; `git diff --check` passes.
- In each of mapped and native modes, the related RHI suite ran 53 cases:
  50 passed and 3 skipped. An additional 3/3 run covers stream RTAS visualization,
  visibility-material serial/parallel pixels, and the strengthened named-resource
  lifetime test. The latter repeats one case, for **52 distinct passing cases
  and 3 skips per mode** across the two runs. No VUIDs were observed.
- The new encoder test uses sparse CPU input IDs 7/213 with a 16-byte GPU block,
  checks overlap/out-of-range/type/representation rejection, mutates borrowed
  layout metadata after initialization, and releases every caller-owned source
  wrapper/slice before recording. GPU readback verifies a nonzero slice offset,
  constants and bounded access after packet retention.
- The skipped Zorah stream-material probe requires an unavailable local asset.
  Both opacity-micromap probes pass their fallback path but skip OMM because the
  installed validation layer cannot validate the required extension. These skips
  do not validate the unavailable paths.
- NRD passes 11/11 in each mode. Both final editor smoke runs exit 0 after
  submitting/presenting a frame; each reports 210 shader warmup requests,
  210 existing cache hits and zero failures. Earlier manual warmup also completed
  all 210 requests with zero failures in each mode (73 existing cache hits).
- Stream transmission/cutout output was visually inspected. The initial RHI run
  rendered error material during an overlapping shared-module edit; its standalone
  rerun and both fixed-source suites pass. Initial evidence remains available.
- An expanded run additionally exposed a failure in the unchanged, copy-only
  `DebugControl checkpoint capture and lifetime` fixture at its submitted-prefix
  evidence assertion. Its root cause was not investigated as part of this migration;
  it is excluded from the related-suite totals above and retained in `final-native.log`.
  The migrated GPUProbe's separate numeric/watch/binding-restoration test passes.
  Its validation assertion now distinguishes API VALIDATION errors from the local
  loader GENERAL registration messages, which remain in captured evidence.

Evidence is under `.cache/dr-named-resources/`: `verified-{mapped,native}.{xml,log}`,
`supplement-{mapped,native}.{xml,log}`, `nrd-{mapped,native}.{xml,log}`,
`smoke-final-{mapped,native}.log`, and the `build-*.log` files. The original expanded
run and transmission diagnostic are preserved separately. No long-session editor,
full-scene memory-stability or optional NTC/Streamline/DLSS/NRC runtime claim is made.

## Earlier full image/buffer and parameter-root verification on 2026-10-03

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
