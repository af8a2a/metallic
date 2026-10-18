# Native Slang Coverage

Coverage remains a visibility decision, independent of Surface evaluation and
Closure preparation. Substrate-style layer coverage belongs to Closure
composition; this change does not turn layer weights or transmission into
geometry rejection.

## Compilation and instance data

`MaterialValueIR::slice(true)` selects the Coverage root. The backend emits one
straight-line Slang function per canonical slice and a program-ID dispatcher in
the named `MaterialCoverageGenerated` module. GPU evaluation no longer reads
opcodes or maintains a dynamically indexed `float4[64]` intermediate table.
The existing 64-live-node and authoring validation limits remain in effect.

The 80-byte instance layout is preserved. The historical `coverageOffset` field
now contains a Coverage program ID; `coverageCount` is the live node count and
zero selects ordinary alpha behavior. Four parameter slots retain their offsets.
Production uploads contain only instance records, without an instruction tail.
The CPU instruction lowering is retained for the independent test oracle.

The emitter preserves the old per-node `clamp(-1e6, 1e6)`, including UV and alpha
inputs, normalized-vector fallback, saturated lerp factor, swizzles, and select
predicate. Final opacity saturation, MASK cutoff comparison, OPAQUE acceptance,
and ordinary BLEND alpha behavior are unchanged. The existing dependency flag
continues to skip base-alpha texture reads when the Coverage expression does not
need them. Texture footprint/mip policy is unchanged.

## Cache and live edits

Surface retains its own source and key. Coverage has independent
`coverageSource()` and `coverageKey()` values. Generated modules live under
`.cache/materials/coverage/<key>/MaterialCoverageGenerated.slang`; existing bytes
must match the generated source, otherwise materialization fails. The generated
directory and `METALLIC_CUSTOM_COVERAGE=1` enter the Slang request/cache identity.

Changing only instance parameters updates the immutable binding generation
without changing either executable key. Changing Coverage code invalidates
Coverage consumers even if there are zero Surface programs. Adding or removing
Coverage likewise updates the shader variant. Local material shader warmup also
recognizes Coverage-only materials and materializes the independent module.

The runtime consumers include VisibilityBuffer raster/tessellation/stream
variants, ScenePathTrace/Deferred, ray pipeline hit stages, RTXDI and material
visualization. VisibilityBuffer updates its pipelines without reopening the
stream asset or discarding resident geometry. RayQuery evaluates Coverage for
each relevant candidate before committing the intersection; Surface material
binning cannot replace this step.

## Validation and measurement scope

`MetallicShaderRequestsTests` exercises independent generated-code/cache identity,
parameter-only cache reuse and Coverage-only warmup. `MaterialCoverageTests`
checks production visibility winners against ray and shadow distances through
material edits, including adding/removing Coverage and ordinary BLEND behavior.
`stream_material_shading` covers streamed/decoded geometry and hardware/hybrid
raster agreement. Run the Slang ownership audit after shader structure changes.

The opt-in `material_coverage_codegen_benchmark` compares the production native
evaluator with the frozen interpreter in `tests/rhi/shaders/LegacyCoverageOracle.slang`.
The oracle is not a production fallback. Both arms use the same executable,
expression fixtures and input distributions; each arm has its own valid input
encoding. Readbacks establish expression/acceptance equivalence. GPU timestamps
measure the evaluator dispatch, not a scene frame or a complete ray traversal.
Ordinary timing runs disable validation and exclude shader/PSO creation and
warmup. Independent processes, rather than dispatch samples, are the statistical
replicates. No register spilling or occupancy improvement is inferred from the
source-level removal of the array.

## Acceptance run, 2026-10-18

Release editor and the selected test targets built successfully in the existing
configured trees. Validation evidence is under
`build/coverage-native-20261017/` (local output, not source-controlled):

- Scene material tests: 37/37 passed (`scene-material-tests-local-temp.log`).
- Shader request/cache/warmup tests: 57/57 passed (`shader-requests.log`).
- Module audit: 365 files, including 253 modules and 112 implementations;
  module-audit Python tests: 5/5 passed.
- OpenPBR and Slab production Coverage regressions each passed 11 editing states,
  checking 42,284 MASK raster/ray/shadow pixels and the BLEND alpha contract;
  the Coverage compiler regression passed (`correctness.log`).
- Stream/decoded and hardware/hybrid regression passed (`stream-retry.log`).
  The eight recorded debug/final views have zero interior RGB error over
  3,131 compared pixels per view. Dynamic Coverage parameter updates also passed.
  The initial stream attempt was invalidated by a source save during compilation;
  the frozen-source rerun passed. Coverage mask and stream images were inspected.

The verified A/B evidence is `ab-fixed/`, with plots, raw-data mapping and the
reproduction script in `ab-fixed/Report/Report.md`. Eight independent processes
(two ABBA blocks, four pairs) ran on RTX 5070 Ti, driver 617.42. Each workload used
512 x 512 threads, four evaluations/thread, 16 warmup and 32 measured dispatches.
The following times are means of the four process medians for each arm;
reductions and intervals are computed from paired process ratios.

| Workload | Legacy ms | Native ms | Mean reduction | 95% paired t interval |
|---|---:|---:|---:|---:|
| Default alpha control | 0.007600 | 0.006608 | 13.05% | 12.57–13.53% |
| Simple expression | 0.055500 | 0.008720 | 84.29% | 84.05–84.52% |
| Complex expression | 0.713480 | 0.012044 | 98.31% | 98.29–98.33% |
| Eight mixed programs | 0.720300 | 0.043404 | 93.97% | 93.93–94.02% |
| Numerical/operation edges | 0.066688 | 0.019392 | 70.92% | 70.73–71.11% |

All five full-image readbacks matched exactly, including opacity sums, per-repeat
acceptance masks and alpha-dependency flags. Same-arm outputs/SPIR-V were stable;
custom legacy/native SPIR-V differed as required. The evidence manifest passed
the independent verifier. The initial `ab/` attempt failed before the complex
case could execute because the test's unary JSON constructor produced an object
instead of an argument array; it is retained but excluded from the measured run.

These results measure only the evaluator compute dispatch. They exclude texture
sampling, rasterization, traversal, lighting and presentation. The control is a
default-alpha path through each backend, not an identical-shader A/A control.
Frame trajectories retain occasional spikes; the statistic is the process
median, without outlier removal. Clocks were unmodified and system-wide GPU
exclusivity was not established. No full-scene speedup, register-spill reduction
or occupancy improvement was measured.
