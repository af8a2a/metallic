# Reserved Neural Texture Compression support

NTC is reserved for future virtual-texture (VT) work. Ordinary scene loading and
rendering no longer perform inference on material samples. Path tracing, RTXDI,
material visualization, shadows, OpenPBR and material-value programs use only
conventional textures. Their shader requests have no NTC or cooperative-vector
variants, includes, resource bindings or capability requirements.

## Asset behavior

The glTF importer ignores optional `NV_texture_swizzle` metadata and uses the
ordinary texture `source`. A required `NV_texture_swizzle` extension is unsupported
and rejected by the existing required-extension check. Assets intended for normal
rendering must supply conventional textures; NTC-only assets are not decoded by
the ordinary loader. There is no automatic NTC preparation, upload, replacement,
residency accounting or inference during normal scene rendering.

## Retained boundary

- `Source/Runtime/Render/NeuralTextureResources.*` retains explicit experimental
  metadata, latent-texture and inference-weight preparation for future work.
- `Shaders/Interop/NeuralTextures.slang` retains the Generic INT8 and cooperative
  vector adapters, using an isolated `NeuralTextureResourceParameters` packet.
  Ordinary scene shaders do not import this module.
- The Vulkan format and cooperative-vector primitives remain available to explicit
  consumers. They are not selected by ordinary material shaders.

`METALLIC_ENABLE_NTC` defaults to `OFF`, including the full preset. An experimental
build can opt in with `-DMETALLIC_ENABLE_NTC=ON -DRTXNTC_ROOT=<SDK checkout>` and
explicitly build `MetallicReservedNTC`. This target is excluded from the default
build and is not linked into the editor or normal renderer. The SDK lookup also
accepts `External/RTXNTC-Library` and `../RTXNTC/libraries/RTXNTC-Library`.
The retained SDK backend requires Vulkan headers; CUDA, DX12 and prebuilt
SDK decompression shaders remain disabled.

Enabling this target does not restore glTF parsing or material inference.
A future VT implementation must explicitly define tile feedback, residency and
budgeting, decoding/transcoding, synchronization and conventional fallback before
connecting NTC to its tile cache. No VT feature is implemented by this change.

## Layout and validation

The material texture record is 48 bytes with two alignment-padding words and no
neural placeholder fields; the complete legacy material record is 720 bytes.
The padding is required by StructuredBuffer vector alignment: the two `float4`
transforms start at offsets 16 and 32. Without it, Slang `sizeof` reports a compact
40-byte record while SPIR-V still uses 48-byte records and a 720-byte material stride.
The former texture-set count word is removed from pass constants. PathTrace and
RTXDI constants are 252 bytes, material visualization constants are 100 bytes,
and shadow geometry constants are 4 bytes. The ordinary scene resource packet
is 384 bytes and needs no tail padding. CPU/Slang declarations, array strides,
resource-field offsets and byte probes are updated together.

Regression coverage checks conventional glTF fallback, required-extension
rejection, ordinary material classification, CPU/GPU material layout and shader
requests without neural permutations. Rendering and optional reserved-SDK
validation must be reported separately; compilation alone does not establish
visual correctness or VT readiness.

## Initial NTC removal validation on 2026-10-11

Reused `build-pass-stages-nrd` without changing its SDK/compiler configuration.
The editor, shader compiler, scene tests, shader-request tests and RHI tests built.

- Scene import: 2 focused tests passed for optional neural metadata fallback and
  unsupported required extensions.
- Shader requests: 19 tests passed, including ordinary requests without neural
  permutations, raster-only Deferred SPIR-V and custom-material warmup/cache reuse.
  `PrewarmedOpenPBRHitsTheRuntimeRequestCache` was excluded from the final run:
  the initial run exposed an obsolete optimizer-dependent function-call assertion,
  which was removed; its lengthy all-debug-mode compilation was stopped. The
  complete debug-mode cache test was not rerun.
- RTXDI, material visualization and screen-space shadow requests compiled with
  zero failures through `MetallicShaderCompiler --filter`.
- RHI: 8 focused tests passed with Vulkan validation: ordinary/typed material
  binning, material GPU ABI, feature-policy rendering, 256-frame HDR captures,
  ValueIR texture footprints, material generations and mapped/native resource
  field bounds. The initial scene packet used an explicit tail word to make CPU/Slang
  sizes agree at 392 bytes (subsequently compacted to 384 bytes). Auto/Dynamic relative RMSE was `1.66213e-9`.
- Inspected tone-mapped previews of the raw Reference, Deferred and RTXCR
  PathTrace outputs. The test checks finite HDR values; this does not establish
  long-session or full-scene temporal stability. No Vulkan VUID errors appeared;
  the machine reported two pre-existing missing layer-manifest warnings.
- Reserved backend: SDK-enabled `NeuralTextureResources.cpp` passed MSVC `/Zs`.
  The opt-in target was not linked or exercised on GPU, and VT is not implemented.

Local evidence is under `build-pass-stages-nrd/ntc-final-evidence` and the
`ntc-*.log` files in that build tree. No performance improvement is claimed.

## Layout cleanup and native Slang follow-up on 2026-10-11

Removed the remaining NTC placeholder members and unnecessary tail words. The
material texture record keeps only the eight bytes required to align its vector
transforms; a GPU byte probe verifies the complete 720-byte material and array
stride. Shared alpha coverage, texture decoding and RTXCR material adaptation are
Slang modules. OpenPBR path vertices use a typed Slang function with explicit
transport state and control results. Program-local surface composition remains
an include because its types and feature configuration belong to the caller.
Vendor shader snapshots are unchanged.

Reused `build-pass-stages-nrd` and built `Metallic`, `MetallicRHITests`,
`MetallicShaderRequestsTests` and `MetallicShaderCompiler` successfully.
Eleven shader-request tests passed, including custom-material warmup. Four focused
GPU tests passed with Vulkan validation for material ABI, both material-binning
routes and resource-field bounds.
Eight additional GPU tests passed for HDR capture, ValueIR texture footprints,
OpenPBR/Slab coverage and shadows, material visualization, RTXDI, OpenPBR debug
views and path-trace stage parameter layout. The debug-view test initially compared
coarse fallback geometry with later streamed detail; it now waits for stable
geometry before its 24 cases and passed a focused rerun with the original
comparison thresholds unchanged. No Vulkan VUID errors appeared.

The 256-frame Reference, Deferred and RTXCR PathTrace HDR outputs match the
initial NTC-removal baseline byte for byte: all three have zero changed components,
zero maximum absolute error and zero relative RMSE. Evidence is in
`build-pass-stages-nrd/slang-native-evidence/migration-comparison.json`.
The debug-view rerun is recorded in `slang-debug-stability-rhi.log` and
`slang-debug-stability-evidence` under the same build tree.
These checks establish equivalence for those workloads, not a performance gain
or long-session temporal stability.
