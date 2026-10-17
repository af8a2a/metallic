# vk_gltf_renderer full BSDF port

`VkGltfBSDF` is an experimental, resource-free module. It has no imports, and
does not depend on OpenPBR, its closure code, or its LUTs. The owner is
[`VkGltfBSDF.slang`](../VkGltfBSDF.slang); implementation units declare that
owner explicitly.

The source is the **full** BSDF invoked by `vk_gltf_renderer`'s path tracer,
provided by `nvpro_core2/nvshaders/bsdf_functions.h.slang`. The snapshot includes
NVIDIA's diffuse reflection/transmission, anisotropic GGX dielectric/metal
reflection, thin/volume specular transmission, clearcoat, Charlie sheen,
iridescence, dispersion, and retroreflection. Volume extinction and
Henyey–Greenstein evaluation/sampling helpers are retained. Porting these
helpers does not by itself implement participating-medium transport in the
renderer.

The original copyright and Apache-2.0 notices remain in each implementation
unit. The complete license is in
[`Shaders/Licenses/VkGltfBSDF-Apache-2.0.txt`](../../Licenses/VkGltfBSDF-Apache-2.0.txt).
[`SourceManifest.json`](SourceManifest.json) records the exact source-file
hashes and the local repository revisions used during porting. Hashes identify
the actual snapshot, including any local upstream changes; revision IDs alone
do not assert a clean upstream tree.

Changes from the source are module/namespace ownership, explicit visibility,
native Slang qualifiers/constants/array types, and public API wrappers. The
full BSDF's mathematical expressions and stochastic lobe selection are kept.
The separate `Simple` BSDF and unused raster-only Khronos/LUT helpers were not
copied. This module does **not** copy the glTF texture evaluator: scene texture
sampling and conversion to `PreparedMaterial` belong to the caller.

Use `import VkGltfBSDF;` and `using Metallic.VkGltf;`. Construct a fully initialized
material with `defaultPreparedMaterial()`, then assign scene parameters. The
default function intentionally retains upstream defaults, including metallic
equal to one. The overload with base color, metallic, perceptual roughness,
shading normal, and geometry normal constructs an orthonormal frame.

The detailed fields are declared in [`Material.slang`](Material.slang). In
particular, `roughness.xy` holds microfacet **alpha**, normally perceptual
roughness squared; clearcoat and sheen roughness remain perceptual.
`thickness == 0` selects thin-walled transmission. The caller swaps `ior1` and
`ior2` on a volume's inside surface; thin-walled materials do not swap. Frames
must be normalized and consistently oriented. Iridescence thickness is in nm.

`evaluateBSDF(material, k1, k2, xi)` returns `projectedDiffuse`,
`projectedGlossy`, and `pdf`. Both directions point away from the surface, and
the returned contributions **already include the cosine**. Evaluation is
stochastic: `xi` must contain three uniform random values in `[0,1)`, and its
PDF is the selected lobe's PDF as used by the source renderer's estimator.
Do not substitute a deterministic full-mixture interpretation of that PDF.

`sampleBSDF(material, k1, xi)` returns direction, weight, PDF and event bits.
Multiply throughput by `weight` directly; it is already `bsdf_over_pdf`.
Discard `eventType == kEventAbsorb`. A valid impulse has `pdf == kDiracPDF`
and receives MIS weight one against a continuous light sampler. Reflection,
transmission, diffuse, glossy and impulse bits are public constants.

The spectral approximations for iridescence and dispersion are authored in
linear sRGB/Rec.709. A renderer with another working color space must establish
an explicit color-space boundary; passing ACEScg color components to this
module is not equivalent to the reference renderer. The BSDF port deliberately
does not import Metallic's color pipeline or silently alter these formulas.

## Validation

[`VkGltfBSDFProbe.slang`](../../../tests/rhi/shaders/VkGltfBSDFProbe.slang) compiles
the actual public API without production resources. The probe writes five
`float4` records per input material: projected diffuse/PDF, projected
glossy/event, sample direction/PDF, sample weight, and extinction/HG value.

`Tools/ValidateVkGltfBSDF.py` independently compiles that probe and the original
upstream source through Slang's CPU C++ target. It verifies the upstream
snapshot hashes before compiling and compares 19 material cases, including
extension combinations, with 4,096 randomized samples each. The comparison
gate is `abs(a-b) <= 2e-6 + 2e-6*abs(a)` and all values must be finite.
Separate analytic checks cover Lambert reflection, Lambert transmission,
Beer extinction and the HG formula. The double-precision HG reference uses
an explicit float32 conditioning bound near its forward/backward peak.

From the repository root, in an x64 Visual Studio developer shell:

```powershell
python -B Tools/ValidateVkGltfBSDF.py `
  --upstream E:/nvpro_core2/nvshaders `
  --output build-release/vk-gltf-bsdf-validation-new
```

Outside the developer shell, append
`--vcvars64 E:/VisualStudio/VC/Auxiliary/Build/vcvars64.bat`.
The output directory must be new. Raw float results, generated CPU shaders,
build logs, hashes and the summary are retained. This validates the isolated
BSDF port on CPU; it does not establish Vulkan execution, scene-material
mapping, integrator correctness, or GPU performance.
