# Native Slang OpenPBR

`import OpenPBR; using Metallic.OpenPBR;` provides Metallic's native Slang
implementation of Adobe's OpenPBR 1.1 BSDF. The module contains the complete
fixed lobe stack, energy compensation, thin film, dispersion, homogeneous volume
helpers and staged preparation. It does not include the vendor language shim or
embed the large lookup arrays in SPIR-V.

The port is derived from `External/openpbr-bsdf`. `Upstream.json` records the
original file hashes. Adobe copyright notices and the Apache 2.0 license are
retained. The original directory is unchanged and remains the independent GPU
reference and the source of the CPU-uploaded energy/LTC tables.

## Interface

Implement `IOpenPBRContext` with statically resolved feature functions and
`sample2D` / `sample3D`. The provider determines resource access and must follow
`DataConstants.slang` table IDs, orientation, normalized coordinates and clamp
semantics. Static feature choices must agree between preparation and evaluation.
There are no resource bindings or push constants in this module.

```slang
let inputs = openPBRMakeDefaultResolvedInputs();
// Supply authored geometry bases and resolved parameters before preparation.
let prepared = openPBRPrepare(luts, inputs, throughput, wavelengthsNm, exteriorIOR, wo);
let projected = openPBREval(luts, prepared, wi);
let density = openPBRPdf(luts, prepared, wi);
```

`openPBRSample` returns projected BSDF / PDF, density, direction and event flags.
Other outputs are undefined when density is zero, as in the Adobe API. Eval
already includes the projected cosine. Emission and the interior volume are
available on `OpenPBRPreparedBSDF`. This port preserves Adobe's camera-path
transport convention; the material adapter continues to reject Importance mode.

Native function/type names replace the cross-language macros. Input and lobe
member spellings retain the upstream mathematical vocabulary for auditability.
The two minimal microfacet lobes and thin-wall wrappers are concrete Slang
types; context-dependent operations use Slang generics rather than macro
instantiation or dynamic interface dispatch.

## Material integration

Production LUTs use embedded precomputed R16_UNORM energy textures and a
RGBA32_FLOAT LTC texture. The shared texture provider performs hardware linear
sampling; 3D lookup combines two XY slices in float to bound filtering error.
There is no production array/manual-load option. See
[texture LUTs](../../../Documentation/OpenPBRTextureLuts.md) for formats,
sampler bindings, numerical tests and the dedicated `--texture-luts` benchmark.

`import OpenPBRClosure; using Metallic.Material;` exposes
`OpenPBRClosure<TContext>` and `OpenPBRPreparedClosure<TContext>` through the
Surface interfaces. Assign the closure's `context`, `inputs` and `occlusion`;
`prepare` copies the immutable BSDF LUT context into its prepared result, so
prepare/eval/pdf/sample share the same static feature specialization and lookup
resources. This context must not contain material-instance or material texture
evaluation state. Production uses an empty, statically bound context.
`SurfaceSamplingContext` supplies throughput, wavelengths and exterior IOR.
Importance transport still fails closed; projected evaluation, sample weight,
event flags and transmission eta retain the existing conventions.

`import OpenPBRTextureLUT;` exposes texture-only LUT sampling functions.
`OpenPBRSurface.slang` owns production resource access and material parameter /
texture evaluation. It calls native Slang types and functions directly, without
legacy OpenPBR HLSL name wrappers or Adobe feature / LUT callback macros.
PT, Deferred and the legacy visibility preview specialize the same closure
module. Only the reference branch of the independent Adobe comparison probe
retains the upstream API and macros. Material payload ABI, normal/TBN and BSDF
math are unchanged.

## Exact-work optimizations

- Fixed RGB and six-lobe loops explicitly unroll, avoiding dynamic local array
  indexing. Loop arithmetic and summation order are retained.
- Thin-film dielectric Fresnel coefficients are consumed per channel instead of
  retaining an array of three complex structs across two loops. The first
  channel remains shared when dispersion is disabled; complex TIR phase remains.
- Exactly zero diffuse/MMS multipliers skip their value/LUT work. These checks
  use physical color multipliers, **not** throughput-dependent proposal weights.
- Zero fuzz coverage skips reflection/LTC work. PDF and sampling bypass require
  a zero mixture probability; the all-black `0/0 -> 0.5` fallback is preserved.
  The reference sample weight/PDF round trip is retained.

No LUT hardware-filter approximation, roughness approximation, altered sampling
threshold, removed lobe or lossy prepared-state packing is introduced. Remaining
opportunities include joint Eval/PDF microfacet evaluation and feature-specific
prepared storage; both require separate contract and correctness work.

## Validation and performance scope

The closure-module migration is covered by `material_closure_openpbr_stages`
(two concrete contexts, including stateful LUT access),
`material_surface_lighting_framework`, `material_value_closure_scene`,
`ray_material_execution_queue`, and `render_graph_pathtracing_guides_shader_compile`.
`lookdev_render_paths` also saves raw `.hdr.bin` evidence. On 2026-10-06,
the shaderball and Studio M05 Fuzz each produced byte-identical RGBA32_FLOAT
PT/Deferred images before and after migration (256x256, 32 frames).
The local comparison record is `build/openpbr-closure-image-comparison.json`.
This is correctness evidence, not a GPU performance measurement.

`RHIRendering.material_openpbr_native_equivalence` compiles the unchanged Adobe
headers and native module into separate kernels. It checks 32,768 cases in each
of two layouts: alternating material types within waves and contiguous material
regions. Sixteen families cover dielectric, metal, transmission, thin wall,
coat, fuzz, thin film, dispersion, anisotropy, SSS, volume and all-black cases,
including grazing directions and both IOR sides.

The test compares diffuse/specular Eval, directional PDF, sample weights,
direction, event flags, emission, extinction, albedo and volume anisotropy.
Finite outputs are required; flags compare exactly, and numeric error is bounded
by `abs(native - reference) / max(1, abs(reference)) <= 2e-4`.

For timing, use `--rhi-no-validation`, independent processes and fresh output
directories. The measured scope is eight GPU dispatches plus their WAW barriers,
divided by eight; upload, readback, compilation and PSO creation are excluded.
LUTs/output are device-local. The probe uses exact buffer LUT interpolation, so
its speedup is **not** a claim about production texture LUTs or whole-frame time.
Production scene tests separately check integration and rendered output.

The reproducible runner saves and verifies the actual SPIR-V, raw readbacks,
source/executable hashes, timings and process logs:

```powershell
python Tools/Perf/OpenPBRBenchmark.py run --exe build-scheduling-release/tests/MetallicRHITests.exe --output build/openpbr-new --runs 5
python Tools/Perf/OpenPBRBenchmark.py verify build/openpbr-new
```

Build `MetallicRHITests` first. Run ordinary timing serially with no injected
validation/profiler layers. The runner does not build, edit shaders or change GPU
clocks. Live runs require a new directory and use bounded child-process timeouts.
The 2026-10-05 acceptance record is in
[`Documentation/OpenPBRNativeSlang.md`](../../../Documentation/OpenPBRNativeSlang.md).
