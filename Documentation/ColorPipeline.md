# Metallic color pipeline

Metallic renders in scene-linear **ACEScg (AP1 / ACES white, approximately D60)**.
Input/authoring spaces and display encodings are independent of that working space.
[ACEScg specification](https://docs.acescentral.com/specifications/acescg/) defines
AP1 and its white point. ACES 2.0 remains the output transform.

## Compatibility and lifetime

`METALLIC_WORKING_COLOR_SPACE=acescg` selects ACEScg (the default).
`METALLIC_WORKING_COLOR_SPACE=rec709` selects the historical linear Rec.709/D65
renderer. Set it before launching the editor, samples, tests or shader warmup.
The selection is immutable for the process lifetime; restart to change it.
CPU upload conversion and every Slang session use the same selection. The shader
cache request hash includes it, including generated material programs.

```powershell
$env:METALLIC_WORKING_COLOR_SPACE = 'rec709'
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='ColorSpace.*:RHIRendering.working_color_cpu_gpu_texture_contract'
$env:METALLIC_WORKING_COLOR_SPACE = 'acescg'
.\build-scheduling-release\tests\MetallicRHITests.exe --gtest_filter='ColorSpace.*:RHIRendering.working_color_cpu_gpu_texture_contract'
```

`ColorSpaceMatrices.h` is the single CPU/Slang source for Rec.709, AP0, AP1,
Rec.2020 RGB/XYZ matrices and Bradford D65/ACES-white adaptation. `WorkingColor`
provides input/output conversions, relative-white XYZ and working-space Y.
Finite negative values survive these transforms. Reflection/transmission bounds
and invalid-radiance handling remain separate material/lighting decisions.

## Assets and authoring

glTF color factors, historical scene JSON, editor material fields and legacy
Material Value IR v1 retain **linear Rec.709/D65 authoring semantics**. Saving
keeps those authored values. `resolveWorkingMaterial` converts color parameters
at upload: base color, emission, specular tint, attenuation, diffuse transmission,
hair color and hair diffuse tint. Scalar weights, alpha, IOR and geometry data
are unchanged. GPUScene, streamed materials and resident ray-query materials use
this common boundary. Material Value IR performs its old authoring-space math
and converts color-valued outputs back to working RGB before applying reflectance
or emission bounds. This preserves native AP1 colors whose Rec.709 coordinates
can be negative or exceed one. Untagged Rec.709 texture samples retain IR v1
semantics; explicitly tagged wide-gamut colors enter the Rec.709 authoring basis
before IR arithmetic.

Texture semantics belong to the **material usage**, allowing one image to be
referenced by both color and data slots. Base color, emission, specular color and
diffuse transmission color are Color; normal, ORM, displacement, weights and
other scalar slots are Data. Transfer decoding and gamut conversion are separate.
PNG color samples are decoded manually; compressed KTX2 transfer semantics keep
the existing vkFormat contract. Hardware sRGB decoding is never repeated.
NTC samples retain their source-space meaning and use the same color adapter.

`TextureInfo.transform0.w` is a numeric mask shared with the shader:

| Bits | Meaning |
| --- | --- |
| 0 | Sample already linear, via explicit source tag or hardware decode |
| 1 | BC5 normal XY reconstruction |
| 2 | Data usage: no transfer/gamut conversion |
| 3-4 | Source primaries: Rec.709=0, AP1=1, AP0=2, Rec.2020=3 |

For Rec.709 source textures the authored `factor * texture` is evaluated in the
source basis before changing basis. This preserves glTF modulation and avoids
multiplying two separately transformed vectors. Tagged AP1/AP0/Rec.2020 textures likewise modulate in their declared source basis,
including when rendering in Rec.709 compatibility mode. Runtime conversion is an interim cost;
offline texture cooking, compression changes and a full OCIO graph are deferred.

Material asset texture resources accept `colorSpace` (`srgb`, `lin_rec709`,
`acescg` / `lin_ap1_scene`, `lin_ap0`, `lin_rec2020`). Data slots reject this tag.
Export/save retains color texture tags. USD Preview Surface honors texture
`sourceColorSpace` (`auto`, `sRGB`, `raw` and supported explicit names); raw means
linear Rec.709. USD constant colors continue the existing linear Rec.709 policy.
Full USD/MaterialX color-management graphs are outside this migration.

Imported and editor/JSON punctual light RGB remains authored linear Rec.709;
the common GPULight upload converts it. Existing editor float color controls are
labelled accordingly. Candela, lumen, lux, exposure scalars and geometry are
unchanged. Procedural sky constants convert once before scene lighting.

Environment settings store `sourceColorSpace`; world JSON accepts `colorSpace`
with the same names. Untagged HDR is linear Rec.709/D65. Explicitly tagged native
ACEScg HDR images are accepted. Color conversion occurs **before** spherical
mips, SH projection, specular convolution and importance PDF generation. Changing
only the source tag invalidates/reloads the environment. Tagged LDR images load
raw normalized 8/16-bit samples before the declared transfer function is decoded;
HDR images retain floating-point samples. Untagged LDR environments preserve the
historical stb gamma 2.2 decode for compatibility. A missing JSON `colorSpace`
remains missing when saved. C++ callers declaring linear Rec.709 explicitly set
`sourceColorSpaceExplicit=true`; non-default source spaces imply an explicit tag.
The existing stb decoder supports HDR and its existing image formats; EXR
ingestion is not added here.

## Lighting and post processing

AutoExposure, environment/light PDF, punctual sampling and RTXDI target weights
use `WorkingColor::luminance`. ACEScg Y uses AP1 matrix row 1. Rec.709 compatibility
keeps the original rounded Y coefficients. Chromatic adaptation preserves the
neutral axis but can change Y for saturated colors: adapted-white color matches
and unadapted absolute XYZ stimuli must not be treated as the same test.

OpenPBR, PathTrace and VBuffer consume working RGB. VBuffer keeps energy color
vectors and metalness/transmission inputs in FP32 in ACEScg mode; its other
bounded scalar weights retain the existing FP16 option. Component maxima used for
Russian roulette, zero-throughput termination and emission factorization remain
component maxima, rather than being mislabeled photometric Y. Linear convolution,
accumulation and interpolation operate in the selected basis. These operations
and per-channel BSDF evaluation can change saturated rendered appearances when
the rendering basis changes, even with correct input/output transforms.

## External boundaries

NRD RELAX frontend signals are converted working RGB -> linear Rec.709 before
vendor packing. RTXDIComposite converts denoised signals back before multiplying
working albedos; RTXDIConfidence reconstructs those same working signals before
metering. Normals, roughness, hit distances and motion are Data. Vendor NRD math
and YCoCg helpers are unchanged. NRD's sanitization can clip negative Rec.709
components of AP1 colors outside the Rec.709 gamut; full native AP1 denoiser
specialization needs separate validation. See [NRD contract](NRDColorContract.md).

DLSS-SR/RR receive linear HDR working color (`colorBuffersHDR=true`). RR albedos
use the same linear working basis; normals, roughness and motion remain Data.
The integrated Streamline guides require linear/HDR inputs and do not prescribe
RGB primaries, so retaining ACEScg is an integration choice, not a claim about
model training. Display/data diagnostics bypass SR/RR evaluation, resize in their
own encoding and retain it through post processing. Returning to scene rendering
resets SDK history. Same-size diagnostic copies also work without bindless;
resizing diagnostics requires the bindless resize kernel. Existing RR beauty
and RR/SR Off compilation keep their previous heap requirements. DLSS-NR retains
its display sRGB RGBA8 input.
[SR guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS.md)
and [RR guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md).

NRC/SHARC scene values retain the selected working basis consistently in training,
query and accumulation. NRC's vendored LogLuv packing has Rec.709-based numeric
matrices; these are paired compression/decompression transforms, not a scene
luminance decision. This path needs color/precision testing independently of
SR/RR and remains subject to the SDK's nonnegative signal packing limits.

## Display and UI

`SceneLinear` and `ExposedLinear` mean the selected working RGB, independent of
format. ColorGrading converts Rec.709 compatibility input to AP1, while ACEScg
input is already AP1. ACES internal grading/output math is unchanged. LUT coordinates
represent the current scene basis. FinalBlit without a LUT also converts working
input to display Rec.709 before its legacy curve or scRGB mapping.

| Encoding | Fixed meaning |
| --- | --- |
| sRGB | Display Rec.709/D65, sRGB transfer |
| scRGB | Display linear Rec.709/D65, 1 = 80 nits |
| HDR10 | Display BT.2020/D65, absolute ST 2084 PQ |

UI, calibration ramps and normal/ID debug images remain display/data colors.
BaseColor and shadow-transmittance diagnostics use `DisplayLinearRec709`, fixed
linear Rec.709/D65; normal/ID/front-face diagnostics retain authored sRGB display
code values. Diagnostic misses are black so a single output never mixes scene
radiance and display/data encodings. Both diagnostic encodings bypass physical
exposure and ACES grading; SDR encodes display-linear values to sRGB and HDR maps
them to scRGB paper white. Per-frame metadata is published in graph order before
parallel recording snapshots are frozen, allowing debug/final hot switching.
AutoExposure, CopyColor, SliderDebug and SR/RR propagate those encodings.
SliderDebug requires equal input encodings. The
standalone RTXCR procedural material demo retains its independent Rec.709-to-sRGB
display fixture; it does not supply a scene-linear graph signal.

## Validation and references

`ColorSpace.*` tests known primaries, white adaptation, round trips, neutrals,
compatibility identity and authoring preservation. The `working_color_cpu_gpu_texture_contract`
GPU test checks matrices, Y, sRGB decoding, Color/Data bypass, native AP1 identity,
source-basis factor modulation, negative and HDR values. Run both process modes.
`MetallicWorkingColorAudit` rejects new engine-owned hardcoded Rec.709 scene Y
coefficients and direct Rec.709->AP1 entry transforms outside WorkingColor.

Historical LookDev `Reference.json` and source captures retain their Rec.709
meaning. `ACEScgRuntimeReference.json` records the new runtime capture contract.
Historical captures are not relabelled as ACEScg. New runtime captures belong in
the build output alongside their mode and settings. A shader compile or smoke test
alone does not validate HDR monitor presentation, SDK temporal stability, or
full-scene performance. Record those results separately. See [migration validation](ACEScgMigrationValidation.md)
for the tested workloads, artifacts and remaining limits.
