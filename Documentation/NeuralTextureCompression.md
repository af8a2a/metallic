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

The material texture record retains its 48-byte layout with two reserved words;
the former texture-set count word in existing pass constants is also reserved.
The ordinary scene resource packet no longer contains neural resource handles.
CPU/Slang declarations and resource-field offsets are changed together.

Regression coverage checks conventional glTF fallback, required-extension
rejection, ordinary material classification, CPU/GPU material layout and shader
requests without neural permutations. Rendering and optional reserved-SDK
validation must be reported separately; compilation alone does not establish
visual correctness or VT readiness.

## Validation on 2026-10-11

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
  field bounds. The scene packet uses an explicit tail word to make CPU/Slang
  sizes agree at 392 bytes. Auto/Dynamic relative RMSE was `1.66213e-9`.
- Inspected tone-mapped previews of the raw Reference, Deferred and RTXCR
  PathTrace outputs. The test checks finite HDR values; this does not establish
  long-session or full-scene temporal stability. No Vulkan VUID errors appeared;
  the machine reported two pre-existing missing layer-manifest warnings.
- Reserved backend: SDK-enabled `NeuralTextureResources.cpp` passed MSVC `/Zs`.
  The opt-in target was not linked or exercised on GPU, and VT is not implemented.

Local evidence is under `build-pass-stages-nrd/ntc-final-evidence` and the
`ntc-*.log` files in that build tree. No performance improvement is claimed.
