# OpenPBR native Slang implementation

The material system now uses `import OpenPBR` in place of textual inclusion of
the Adobe cross-language implementation. The module lives in
[`Shaders/Modules/OpenPBR/`](../Shaders/Modules/OpenPBR/README.md); its statically
typed LUT/feature context has no renderer bindings. The current OpenPBR material
program and Closure/PreparedClosure interfaces continue to own parameter
resolution, transport conventions and lighting integration.

The original `External/openpbr-bsdf` files and license are unchanged. The native
files retain attribution and have a hash-indexed upstream provenance record.
Both the original and native implementations are compiled independently in the
GPU differential test. Production uses the native implementation; reference
headers remain in the test and CPU LUT upload paths.

## Rendering patterns addressed

| Pattern | Native implementation | Correctness boundary |
|---|---|---|
| Cross-language macros and implicit LUT callbacks | Slang module, native types/functions, static generic provider | Existing static feature choices and LUT interpolation preserved |
| Fixed RGB/lobe loops indexing local arrays | Explicit unrolling | Same arithmetic and summation order |
| Three-channel complex Fresnel temporary array | Consume one channel at a time; share first channel when dispersion is off | Preserve complex TIR phase and exact Fresnel functions |
| Zero diffuse/MMS multipliers still evaluate tables/BRDF | Exact physical-zero fast paths | Never infer zero BSDF from throughput-dependent sampling weights |
| Absent fuzz still performs LTC and mixture work | Skip zero coverage work; bypass the mixture only when its probability is zero | Preserve the all-black 0/0 fallback and sample weight/PDF round trip |

Hardware-filtered LUT replacement, approximate square roots, altered lobe
thresholds and feature removal were deliberately not used. Joint Eval/PDF
evaluation and smaller feature-specific prepared states remain separate work.

## Acceptance — 2026-10-05

Windows x64, Release/MSVC, Slang from the configured repository SDK, Vulkan on
RTX 5070 Ti (driver 616.92). Built `MetallicRHITests` in
`build-scheduling-release`; no build preset/compiler/SDK changes were made.

- `material_openpbr_native_equivalence`: 32,768 cases per layout, two layouts,
  16 material families. Unoptimized port matched the reference exactly. Final
  exact-work paths have maximum scaled error **1.57784e-7**; all outputs finite,
  sample flags match exactly. Validation-enabled run passed.
- `material_closure_openpbr_stages`: passed, including stage/texture counters,
  normal mapping, both IOR sides and unsupported Importance transport.
- `material_surface_lighting_framework`: passed for OpenPBR/Lambert/Mirror.
- `material_program_shader_contract` and `material_program_guide_compile`:
  both passed, including OpenPBR path-tracing guide variants.
- `material_graph_scene`: passed. Graph/SDK and binned/unbinned error **0**;
  texture/normal relative errors **9.37e-8** in PT and Deferred.
- `material_value_closure_scene`: passed. Layer boundary, binning and secondary
  emission linearity errors **0**.
- `openpbr_lookdev_reference_capture`: passed; inspected the 768x768 shaderball
  after 256 frames / 1024 spp. This is renderer integration validation, not a
  new cross-renderer Painter comparison.

The numerical tolerance is `abs(native-reference)/max(1,abs(reference)) <= 2e-4`.
The test checks diffuse/specular Eval, PDF, sample weight/direction/event,
emission and volume outputs against the unchanged Adobe implementation. This
establishes consistency with that snapshot, not independent physical accuracy.

## Performance result: inconclusive

The final evidence contains five discovery and three confirmation processes,
each with two warmup and eight measured alternating AB/BA rounds. The timed
scope is eight dispatches plus WAW barriers divided by eight, with 32,768 cases
per dispatch. LUT/output buffers are GPU-local; compilation, upload and readback
are outside timestamps. Validation/profiling are off; clocks are unchanged.

| Layout | Discovery reduction, mean [95% CI] | Confirmation reduction, mean [95% CI] |
|---|---|---|
| Mixed material types within waves | 6.16% [3.06, 9.25]% | 6.31% [-5.14, 17.76]% |
| Contiguous material regions | 12.48% [9.18, 15.79]% | 6.04% [-9.89, 21.97]% |

The confirmation intervals cross zero. **No stable acceleration is certified.**
Correctness-validated reductions in redundant work are retained, with this
performance limitation recorded. Earlier exploratory results do not override
the final uncertainty, and no further significance-seeking runs were performed.
This buffer-LUT probe does not measure production texture LUT sampling, a
production pass, or whole-frame speedup. Register/occupancy metrics were not
collected. Initial host-visible-buffer timings were rejected as performance
evidence because their scope was dominated by PCIe access.

Local evidence (generated, not source-controlled):

- `build/openpbr-native-final-validation/`: validation, actual SPIR-V and readbacks.
- `build/openpbr-native-production/`: five production tests, PNGs and acceptance JSON.
- `build/openpbr-final-discovery/` and `build/openpbr-final-confirmation/`: hashed
  manifests, raw SPIR-V/readbacks, process logs and timing CSVs.
- `build/openpbr-final-report/`: reviewed chart, process means, 95% intervals,
  decision, dependency versions and reproducible `plot_report.py`.

Use [`Tools/Perf/OpenPBRBenchmark.py`](../Tools/Perf/OpenPBRBenchmark.py) to collect
or independently verify the raw evidence. Report generation runs the current
repository verifier before plotting; it does not execute code from evidence
packages.
