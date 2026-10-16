# Path-tracing link-time specialization experiment (2026-10-12)

Pinned Slang link-time constants and types produced the same final code as
equivalent static configurations in the real `OpenPBRRayQueryPathTraceGuides`
program. At O1, many resolved constant branches still retain their bodies.
Dead-branch cleanup and O2 target generation reduce this program's measured
Driver creation cost; the link-time mechanism itself does not supply stronger
dead-code elimination. This follows the
[resource narrow-read experiment](PathTracingResourceNarrowReads.md). Production
shader defaults, scene eligibility and compiler/session policy are unchanged.

## Current owned-module recheck (2026-10-13)

The experiment was rerun against the current module-owned shader layout,
including `Modules/OpenPBRSurface.slang` and the owned
`Modules/OpenPBRPathTracing/OpenPBRPathVertex.slang` implementation. The
repaired overlay preserves public context methods, gates
`materialDomain(transport)`, and declares the capability interface and all
extern/export providers in `Metallic.SceneOpenPBR`. Each generated provider
declares its own module. The preparation snapshots and redirects additional
shader search roots as well as the project shader tree.

Thirteen current-source O1 arms passed compilation and Vulkan 1.4 scalar-layout
validation. Direct source compilation and serialized-module linking of the
original both matched the fresh Bench reference byte-for-byte: 6,947,064 bytes,
FNV `2d09e5ccab53bd72`, SHA-256
`645602e7448196a055ccc8a5e7f33a42d22a8d1031a303d9e54e856cb299d270`.
This is a new baseline; the 2026-10-12 input identities and measurements below
remain historical evidence.

| Current O1 arm | Bytes | Instructions | Basic blocks | Functions / calls |
| --- | ---: | ---: | ---: | ---: |
| Current production | 6,947,064 | 449,813 | 60,501 | 1 / 0 |
| Static / constant / type, general | 6,947,188 | 449,824 | 60,504 | 1 / 0 |
| Static / constant / type, no-fiber | 6,947,188 | 449,824 | 60,504 | 1 / 0 |
| Static / constant / type, opaque-dielectric | 6,947,188 | 449,824 | 60,504 | 1 / 0 |

For each of the three masks, static constants and link-time constants are
identical raw bytes; concrete policy types and link-time types are also
identical raw bytes. Constant and type outputs differ in their ID assignments,
but become byte-identical under `spirv-opt --compact-ids`, which only renumbers
IDs. All thirteen outputs contain 495 loop merges, zero Phi instructions and
zero Vulkan specialization-constant instructions. The additional gate
scaffold contributes eleven instructions and three blocks without eliminating
the disabled bodies at O1.

An independent instruction/operand comparison of the current compact-ID
static outputs confirms successful provider resolution and retained control
flow. General versus no-fiber differs in exactly one instruction: at zero-based
instruction ordinal 82,491, `OpBranchConditional` changes its condition from
`%318` (`OpConstantTrue`) to `%353` (`OpConstantFalse`), while its targets
`%65375` and `%65376` remain identical. General versus opaque-dielectric
differs in exactly 777 instructions: 699 `OpBranchConditional` condition
operands and 78 `OpLogicalNot` input operands change from the true constant to
the false constant. No opcode, instruction word count or branch target changes.
Thus current O1 link-time constants/types do not remove more code than their
matched static controls; choosing a narrower feature mask still leaves those
resolved dead branches in the emitted program.

Evidence: [current frozen preparation](../build-release/link-time-current-20261013/preparation/Preparation.json),
[current compilation and baseline identity](../build-release/link-time-current-20261013/current-O1/summary.json),
[validated byte/structure comparisons](../build-release/link-time-current-20261013/current-O1-analysis/Analysis.json)
and [sealed operand comparison and input identities](../build-release/link-time-current-20261013/operand-analysis/OperandDifferences.json).
The [read-only parser](../build-release/link-time-current-20261013/operand-analysis/InspectOperands.py)
can reproduce the operand evidence with `--output-dir` pointing to a fresh
directory, without altering either sealed input tree.
These are single structural verification runs, not repeated compiler timing
samples or rendering qualification. The Driver timings of 68.480, 58.157,
53.203 and 36.927 seconds reported below use the older emitted inputs and
must not be presented as current-module measurements. Current O2 elimination,
compile-plus-create costs and GPU behavior need separate current-input evidence.
The subsequent [current dead-branch qualification](ShaderDeadBranchCleanup.md)
records separate uncached-compiler, Driver and GPU measurements, and an additive
Slang optimizer probe emitting exactly the qualified cleanup bytes. Its Driver
benefit is attributable to the late cleanup, not the link-time mechanism.

Two effects must be distinguished: specializing a general shader for a smaller
scene feature set, and supplying those same choices later through the linker.
Only the second comparison can establish an advantage of link-time
specialization itself. Replacing existing `false` values for dispersion or
sheen/coat cannot be counted as additional code removal in this workload.

## Pinned capability and proposed integration

The pinned compiler is Slang 2026.18.2. Its headers expose module loading,
component composition, module serialization and binary freshness checks. The
[official link-time specialization guide](https://docs.shader-slang.org/en/latest/external/slang/docs/user-guide/10-link-time-specialization.html)
documents separate `extern` declarations and `export` providers. Constants and
types become known before target code generation. A linked type provider must
match the declared name and interface conformance. The experiment must verify
the syntax against this SDK, rather than infer compatibility from the latest
documentation alone.

This is distinct from Vulkan pipeline specialization constants. The intended
final SPIR-V contains ordinary resolved values; no `VkSpecializationInfo` or
`[[vk::constant_id]]` is required. No driver-time specialization boundary or
native pipeline reuse follows from using a Slang link-time declaration.

The existing compiler path composes `{module, entryPoint}` in
`Source/Runtime/Render/Core/SlangCompiler.cpp`. A future API would load the
chosen provider in the same session, append it to that component list, link,
and call `getEntryPointCode`. Adding a provider search path does not itself
select an exported definition; compose the selected provider explicitly.
Missing exports should fail visibly in the experiment; a silent default would
obscure whether specialization actually occurred. The pinned constant probe
accepts duplicate providers and selects the first, so the host must reject
conflicting exports/provider selection instead of relying on linker errors.

The smallest constant experiment changes three gates, while retaining the
existing material resource access and LUT callbacks:

```slang
// Consumer; definitions arrive from a separate provider module.
extern static const bool kSceneEnableFiber;
extern static const bool kSceneEnableTranslucency;
extern static const bool kSceneEnableMetallic;

// Existing concrete SceneOpenPBRContext methods use these values.
// The vertex's fiber branch tests kSceneEnableFiber before materialDomain().
```

For a type experiment, use a small feature-policy interface with the three
static methods instead of replacing the whole renderer resource context.
An extern policy type can resolve to either a general or opaque implementation
through an exported alias. The concrete `SceneOpenPBRContext` continues to own
`sample2D` and `sample3D`; its feature methods delegate to the policy type.
This avoids exporting scene descriptors or resource helper functions merely
to test link-time types. Interface names and fully qualified symbol names
must be identical in the consumer and provider.

## Real feature gates and controls

| Gate | Existing reachable site | General values | Opaque fixture values |
| --- | --- | ---: | ---: |
| Fiber | `OpenPBRPathVertex.slang`: material-domain branch calling `shadeSceneFiber` | true | false |
| Translucency | `OpenPBRSurface.slang`: `SceneOpenPBRContext.enableTranslucency` | true | false |
| Metallic | `OpenPBRSurface.slang`: `SceneOpenPBRContext.enableMetallic` | true | false |

The OpenPBR module already uses static context methods in lobe construction,
selection, evaluation, PDFs and reflection/transmission coefficients. These
are actual code-removal opportunities when their previously enabled values
become false. The fiber integration reaches RTXCR scattering and its lighting
and sampling paths. An SDK capability flag such as `METALLIC_HAS_RTXCR` is not
a proof that a particular scene contains a fiber material.

The measured entry also includes the legacy integrator source, but does not
call its legacy `pathTrace` entry. Removing a branch in an unreachable function
does not establish a final-code reduction. The overlay must gate the called
OpenPBR vertex branch. A later production implementation covering legacy
entries would also need consistent gates there.

Compare the following arms with identical O1, floating-point behavior,
descriptor ABI, entry point, capabilities, debug settings and output bindings:

1. Current production source as the baseline.
2. Ordinary literal/macro values at the three same gates, general and opaque.
3. Link-time constants at those gates, general and opaque.
4. An ordinary concrete feature-policy type with the same interface/scaffold,
   general and opaque.
5. A link-time feature-policy type at those gates, general and opaque.

All general arms retain the current features. The literal/macro opaque arm is
the control for code removed because the scene is narrower. The concrete-type
arm separates the policy/interface scaffold from link-time replacement. A macro that
physically excludes whole source blocks is a separate front-end experiment;
it is not equivalent to changing a static boolean in otherwise parsed source.
Constants/types can permit reuse of unspecialized module IR across variants,
but do not guarantee stronger dead-code elimination than equivalent static
arguments. Measure reuse separately from uncached source compilation.

## Opaque fixture eligibility

The fixed fixture graph is `build/compute-create-fixtures/OpaquePerf.json`.
It runs the OpenPBR pass with SHARC off, two samples, depth 12, guides enabled
and the immutable `Opaque/Scene.gltf` asset. That glTF has one material and
eleven primitives, all referencing material zero. The material explicitly
sets metallic factor zero and roughness 0.5, has no textures or material
extensions, and uses the default opaque alpha mode. Scene defaults provide
zero transmission, zero diffuse transmission and no RTXCR hair identity.
It therefore provides a concrete test for disabling all three gates.

| Input | SHA-256 |
| --- | --- |
| `Opaque/Scene.gltf` | `6ad3de75deea6f9d95763bae571c329751baae79e7602481986255b7be3e52e2` |
| `Opaque/Scene.bin` | `6735376d96c08983456957a8f6c5be62dd740607e2e9fa87712b543d53d2a718` |
| `OpaquePerf.json` | `2d3fa99666a624d55f5bc4a55b2828384b23f41f34c7482d4e167dd3ac4f969f` |

Opaque alpha mode by itself is insufficient: an opaque glTF material can
still carry transmission extensions. The feature union must cover every
material reachable by primary, secondary and shadow rays, including streamed
assets and custom material programs. A general-purpose renderer cannot use
these fixture choices as defaults. If a new material or edit violates the
feature set, the renderer must select or create a compatible broader pipeline
before dispatching the affected scene. GPU comparison remains necessary even
for eligible scenes because different static paths can change floating-point
rounding or random-variable remapping.

## Measurement and qualification

Freeze the current post-narrow-read source, provider source, dependencies and
compiler options for each arm. Record source and final-device SPIR-V identities
separately. Validate each final SPIR-V with Vulkan 1.4 and scalar block layout.
Count instructions, functions, calls, CFG blocks/edges, loops and variable
storage classes. ID compaction or byte-count reduction alone does not locate
the driver cost or establish better execution.

Use sequential independent processes with alternating paired order for
timings. Record CLI wall time and nested SDK phases without adding overlapping
counters. For driver comparison, reuse exact raw SPIR-V through real compute
recipes, bypass native-object reuse, use empty application pipeline caches,
and record the controllable driver-cache setting, device, driver and validation
state. A disabled controllable driver cache does not prove that every driver
compiler state is cold.

GPU qualification would use the same OpaquePerf graph, environment, resolution,
frames and fixed tolerances as the prior report. Compare HDR and all seven
guide outputs; preserve RNG, authored normal/TBN handling, feedback writes
and resource ABI. A CPU compile or successful pipeline creation alone cannot
qualify a new specialized shader. Report pass dispatch times as pass times,
and do not infer an end-to-end renderer speedup from them.

### Resolved settings do not imply dead-code removal at O1

The frozen production baseline was recompiled and matched the current Bench
byte-for-byte: 6,945,092 bytes, FNV `f0282250bfa994e8`, SHA-256
`efbf280f5f1a0025d21a7e64414a40ae0927aa44387a09cee0bd3a70af4c0b2b`.
Both direct source compilation and serialized-module linking reproduced it.

Thirteen real Guides arms passed compilation and Vulkan 1.4 scalar-layout
validation. For each of general, no-fiber and opaque-dielectric settings,
link-time constants were **byte-identical to static constants**, and link-time
types were **byte-identical to their concrete-type control**. Constant and
type outputs became byte-identical after ID renumbering only. Thus the tested
link-time mechanisms did not remove more code than equivalent static arguments.

| O1 arm | Bytes | Instructions | Basic blocks | Functions / calls |
| --- | ---: | ---: | ---: | ---: |
| Current production | 6,945,092 | 449,692 | 60,520 | 1 / 0 |
| Static / constant / type, general | 6,945,216 | 449,703 | 60,523 | 1 / 0 |
| Static / constant / type, no-fiber | 6,945,216 | 449,703 | 60,523 | 1 / 0 |
| Static / constant / type, opaque-dielectric | 6,945,216 | 449,703 | 60,523 | 1 / 0 |

All contain zero Vulkan specialization-constant instructions. The added gate
scaffold itself adds eleven instructions and three blocks to the original.
Direct source compilation of static no-fiber also matched its serialized-IR
result, excluding serialization as the cause of the retained code.

The compact-ID general/no-fiber outputs differ in **exactly one instruction**:
`OpBranchConditional` changes from `OpConstantTrue` to `OpConstantFalse`.
The entire body remains in the output. Opaque-dielectric differs in 777
instructions: 699 conditional branches and 78 `OpLogicalNot` operations.
This establishes that provider resolution succeeded; the missing step is
removal of unreachable control flow after inlining at this optimization level.

Evidence: [frozen inputs](../build-release/link-time-guides/preparation-v1/Preparation.json),
[compiler arms](../build-release/link-time-guides/smoke-v2/summary.json),
[validated comparisons](../build-release/link-time-guides/analysis-v2/Analysis.json),
and [operand differences](../build-release/link-time-guides/dce-diagnostic-v1/OperandDifferences.json).
The first failed run, `smoke-v1`, is retained separately: its IR serialization
command supplied a profile without a target. It is excluded from measurements.
The corrected stage uses `-target spirv -no-codegen -emit-ir`; linking enables
target code generation separately.

### Common dead-branch cleanup control

An untimed diagnostic ran only `spirv-opt --eliminate-dead-branches
--simplify-instructions`, with Vulkan 1.4 and scalar block layout. All outputs
passed the same `spirv-val` settings. This is a distinct optimization experiment,
not a benefit supplied by a link-time constant/type.

| Input / two-pass output | Bytes | Instructions | Basic blocks |
| --- | ---: | ---: | ---: |
| Current production, unchanged | 6,945,092 | 449,692 | 60,520 |
| Current production, cleaned | 3,312,976 | 215,192 | 28,236 |
| General capability scaffold, cleaned | 3,313,052 | 215,199 | 28,238 |
| No-fiber scaffold, cleaned | 3,312,268 | 215,135 | 28,222 |
| Opaque-dielectric scaffold, cleaned | 3,181,824 | 206,669 | 27,026 |

The production input loses 234,500 instructions (52.15%) and 32,284 blocks
(53.34%). It initially contained 845 directly true and 882 directly false
conditional branches; both counts become zero. Most reduction comes from
common dead code already present in the general shader. After cleanup, the
no-fiber setting removes only another 64 instructions; the opaque setting
removes another 8,530 instructions (3.96%) relative to the general scaffold.
Do not attribute the common reduction to scene feature specialization.

Evidence: [original cleanup control](../build-release/link-time-guides/dce-original-baseline-v1/DeadBranchSummary.json)
and [feature cleanup controls](../build-release/link-time-guides/dce-diagnostic-v1/DeadBranchSummary.json).
Structural validation establishes SPIR-V validity, not image or temporal
equivalence. Driver timings and GPU qualification are reported separately.

A separate textual `#if` control physically excludes the fiber branch before
parsing. Its general output exactly matches production; its no-fiber O1 output
is 6,443,264 bytes (-7.23%). This is stronger early exclusion than the tested
O1 boolean gates, but repeats front-end compilation and is a scene restriction.
It must not be reported as a benefit of link-time types.

Changing **final target generation** to O2 while reusing the frozen O1 consumer
and provider modules produced 1,943,960 bytes for general capabilities and
1,571,372 bytes for no-fiber (-19.17% relative to that O2 general control).
All four extra outputs passed Vulkan 1.4 scalar-layout validation. This mixed
source-IR/target experiment is distinct from compiling the whole runtime shader
with O2. Matched static-vs-link-time O2 controls are checked separately before
attributing elimination to the mechanism.

Evidence: [textual controls](../build-release/link-time-guides/preprocessor-smoke/summary.json),
[O2 target-generation commands](../build-release/link-time-guides/final-link-O2/Summary.json)
and [extra validation](../build-release/link-time-guides/extra-control-validation/Validation.json).

All eight matched O2 arms passed validation. For each configuration, static
constants, link-time constants, static concrete types and link-time types now
produce **identical raw bytes**, not just identical counts or compact IDs.
General produces 111,753 instructions / 16,453 blocks; no-fiber produces
90,277 instructions / 13,110 blocks. A fresh fully frozen current production
source compilation with O2 also exactly matches the O2 general result,
SHA-256 `167a1af7ede5510a58439bfa5f5636ab1d88884413f29e859bd8b643013dcb77`.
This removes the mixed-stage confound for the emitted general bytecode; the
Driver table still identifies the actual source of its input.

Evidence: [O2 matched comparisons](../build-release/link-time-guides/matched-O2-analysis/Analysis.json)
and [whole-source O2 identity](../build-release/link-time-guides/direct-original-O2/Summary.json).

**O2 has a substantial compiler cost.** The exploratory whole-source O2
diagnostic took 74.088 seconds, whereas the separate O1 baseline identity
probe took 11.030 seconds. Four equivalent general O2 relink diagnostics ranged
59.184–72.951 seconds (median 64.566); no-fiber ranged 61.395–71.792 seconds
(median 70.891). These are exploratory, not paired frontend timing samples,
and different mechanisms emitting the same bytes are not statistical repeats
of one compiler arm. They do show that reusing consumer/provider IR does not
remove expensive target optimization. Driver-only reductions cannot establish
an end-to-end cold-compilation improvement. Retain O1 defaults; next compare
full uncached compile-plus-create costs and a prefilled SPIR-V-cache scenario,
alongside rendering/GPU qualification.

### Independent Driver creation comparison

Twelve fresh Bench processes replayed exact SPIR-V recipes: three repetitions
of four arms, sequentially in the recorded order. The device was NVIDIA
GeForce RTX 5060, driver 616.92, cache UUID
`e686062a10e5dfb6adcf90d3a502d334`. All arms used the same compute state,
mapped descriptors, 12-byte user push data and normal driver optimization.
Application caches were empty; `METALLIC_VK_INTERNAL_PIPELINE_CACHE=disabled`.
Debug, validation and pipeline-statistics capture were off. No Slang compilation
or dispatch was included. Every creation reported valid feedback, a native
cache miss, no metadata hit, no native-object reuse, matching recipe state hash,
and unchanged source/device SPIR-V bytes.

| Arm | Native create median [min, max] | Driver median | Paired ratio vs O1 |
| --- | ---: | ---: | ---: |
| Current O1 | 68.480 s [66.981, 70.894] | 68.480 s | 1.000000 |
| Current O1 + two cleanup passes | 58.157 s [57.976, 60.935] | 58.157 s | 0.859519 |
| Frozen O1 IR, O2 target, general | 53.203 s [47.818, 53.357] | 53.203 s | 0.750462 |
| Frozen O1 IR, O2 target, no-fiber | 36.927 s [33.080, 37.450] | 36.927 s | 0.520873 |

The paired-median reductions versus O1 are 14.05%, 24.95% and 47.91%
respectively. Ratios of independent arm medians differ (15.07%, 22.31% and
46.08% reductions); do not mix these estimators. Within the O2 controls,
no-fiber/general paired ratios are 0.694070, 0.691789 and 0.701870, with median
0.694070 (30.59% reduction). Median lock wait remained 0.0008–0.0009 ms.
Driver compilation dominates these calls; shrinking input instruction counts
is helpful here but does not predict a proportional timing reduction.

These are three-process samples per arm, with a recorded, not fully balanced,
four-arm schedule. They establish the observed direction for this workload,
not a universal speedup or a guarantee that all OS/compiler state was cold.
The O2 no-fiber result applies to an explicitly restricted scene; it does not
justify replacing the renderer's general shader by default.

Evidence: [raw commands, reports and times](../build-release/link-time-guides/native-paired-v1/summary.json),
[verified ratios and identities](../build-release/link-time-guides/NativeQualification.json)
and [exact recipes](../build-release/link-time-guides/native-recipes-v1/Inputs.json).
After all processes, all 346 captured repository input hashes still matched,
including dependencies reached through the initial additional include path.
The initial preparation did not freeze the complete additional include
directory, so it cannot rule out an uncaptured name appearing there. Driver
measurements depend only on the sealed emitted bytes. The
[legacy import qualification](../build-release/link-time-guides/legacy-import-qualification/Qualification.json)
records that limit; future preparations snapshot and redirect every additional
include root as well.

The transformed boolean-policy shaders in this batch have **not** been
qualified by rendering HDR/guides or timing GPU dispatch. At the time of that
batch the graph harness had no independent precompiled-SPIR-V substitution
path. The separate material-backend follow-up below now adds that path and
qualifies its five coarse-Type artifacts in fixed scenes. That qualification
does not cover the boolean-policy artifacts here. Production defaults remain
unchanged.

### Pinned support probes

Forty small SPIR-V outputs passed validation; 26 comparisons were byte-identical,
including twelve matched static/link-time cases across O0/O1/O2 and eight CLI/API
source/serialized-IR cases. These prove SDK support independently of Guides.
O1 constants remove the optional arithmetic exactly as their static control;
the O1 Type probe retains a trivial constant branch, also present in its static
generic control. O2 removes that branch. Neither proves universal stronger DCE.

Evidence: [support summary](../build-release/link-time-support/Summary.json),
[API probe](../build-release/link-time-support/APIProbe.cpp) and
[reproduction commands](../build-release/link-time-support/README.md).

## Cache and lifecycle requirements before production adoption

The present `ShaderRequest` and `SlangShaderDesc` have no provider list. An
extension must own provider identities/content alongside the existing request
strings. Include the selected feature values/type identity and provider order
in request equality and disk-cache keys. In-memory provider source requires a
content hash; a synthetic filename alone is insufficient. Collect and validate
dependencies from both the program and provider modules, including transitively
imported files, and register those sources for hot reload even when linking
fails. Native PSO keys should continue to use finalized device SPIR-V and the
complete PSO/RHI ABI; object reuse does not replace the driver binary cache.

If module IR is persisted, treat compiler version, session options, target,
macro set, capabilities, descriptor ABI and all dependencies as cache inputs.
Serialize imported modules separately. The pinned SDK provides
`ISession::isBinaryModuleUpToDate` and
`CompilerOptionName::UseUpToDateBinaryModule`; enable freshness checks before
allowing `.slang-module` search-path precedence. There is a pinned-SDK exception:
when the primary source cannot be found on the search paths, the freshness API
treats the binary as standalone and skips compiler-version and option-set hash
comparison (`slang.h`, `isBinaryModuleUpToDate` comment). Source-less deployments
therefore need an independently validated manifest for those identities.
Test source and option changes explicitly, including changes in imported
modules and missing primary/secondary sources. A stale binary must be
recompiled or rejected rather than silently reused.

The current fresh-session policy avoids retaining stale source IR. Any session
or module reuse must also honor the pinned global-session thread-safety
contract. Use worker-owned sessions or a deliberate serialized access policy;
the pipeline single-flight cache does not make compiler sessions thread-safe.
Production adoption remains contingent on measured benefit and scene-wide
eligibility, invalidation and rendering qualification.

## Reproduce and next step

The isolated tool is
[`Tools/Perf/ValidateSlangLinkTimeSpecialization.py`](../Tools/Perf/ValidateSlangLinkTimeSpecialization.py).
Use a fresh output directory for each stage; its manifests reject changed,
missing or unexpected files, altered shader hashes, incomplete schedules and
different feature masks. The baseline report must contain one successful
mapped O1 Guides request with debug disabled and custom materials disabled.
For example, with the pinned Python runtime or a working Python installation:

```powershell
# First generate a fresh compile-only reference for the custom-materials-off
# Guides request N selected from --list --filter OpenPBRRayQueryPathTraceGuides.
& .\build-release\Source\MetallicShaderPipelineBench.exe --request-index N `
  --iterations 1 --compile-only --slang-optimization O1 `
  --descriptor-mode mapped --debug-mode disabled --shader-cache reuse `
  --cache-dir build-release/<fresh-cache> --output-dir build-release/<fresh-reference>
python -B Tools/Perf/ValidateSlangLinkTimeSpecialization.py prepare `
  --report build-release/<fresh-reference>/report.json `
  --request-index N --output-dir build-release/<fresh-inputs>
python -B Tools/Perf/ValidateSlangLinkTimeSpecialization.py run `
  --preparation build-release/<fresh-inputs> --output-dir build-release/<fresh-compile> `
  --variants original-general static-general constant-general type-static-general type-general `
    static-no-fiber constant-no-fiber type-static-no-fiber type-no-fiber
python -B Tools/Perf/ValidateSlangLinkTimeSpecialization.py relink `
  --input-dir build-release/<fresh-compile> --output-dir build-release/<fresh-O2> `
  --target-optimization O2 --variants static-general constant-general `
    type-static-general type-general static-no-fiber constant-no-fiber `
    type-static-no-fiber type-no-fiber
python -B Tools/Perf/ValidateSlangLinkTimeSpecialization.py analyze `
  --input-dir build-release/<fresh-O2> --output-dir build-release/<fresh-analysis>
```

IR reuse currently covers the consumer/provider modules, not all imported
modules: imports may still parse frozen source. The tool records initial
precompile cost separately from relink cost. No paired frontend speedup or
complete module-cache benefit is claimed by the exploratory runs.

The next production experiment should qualify the O2 target path and common
dead-branch cleanup on actual HDR/guides and GPU timestamps, then compare the
general and scene-wide no-fiber variants. Include the costly O2 target
optimization in uncached startup measurements and measure offline/prewarmed
SPIR-V reuse separately. Keep link-time constants/types as
typed configuration and front-end reuse mechanisms; the demonstrated code
removal depends on the target optimization pipeline. A policy Type holding
only boolean methods does not itself form a native compilation boundary or
solve the large inlined program. A later coarse implementation-selection Type
should have its own matched concrete-type control and rendering contract.

Validation of the tool uses the registered
`MetallicSlangLinkTimeSpecializationTests` together with the existing frontend
and SPIR-V analysis tests: **3/3 CTest suites passed**, including **20/20**
focused link-time evidence tests. The final check verified seven evidence
manifests, all twelve recipe/device matches, and local report links.
No C++ runtime or production shader changed in this experiment.

## Implementation-selection Type: proposal and measured follow-up

The [material-backend follow-up](PathTracingMaterialBackendExperiment.md)
implements this coarse boundary in frozen harness overlays and compiles all
five matched arms. It records actual O1 structure, compiler phases and uniform
cleanup results and three paired native-creation measurements. Those results
supersede this section's initial unmeasured status. Exact artifact rendering
now qualifies all five coarse-Type outputs in three fixed SHARC-off,
CUSTOM_MATERIALS=0 scenes, including complete loaded-material No-Fiber checks;
HDR, temporal frames and guides are numerically identical. Dynamic material
changes, other permutations and production scene eligibility are not covered.
The direct original-source control observes 16.31% less native creation time
by paired median, but its Mixed dispatch result is +13.70% with large
chronological drift. Smaller SPIR-V and output equivalence therefore do not
qualify the current narrower Type as a general runtime default. The follow-up
keeps original-source and linked-General denominators separate.

The subsequent [material-record read comparison](PathTracingResourceNarrowReads.md#material-record-follow-up-2026-10-13)
preserves the General implementation in raw and typed-member overlays. Both
pass exact-artifact fixed-scene output checks, but neither establishes a stable
compiler or native-creation gain. This separates resource transport from Type
selection. The next compilation-complexity experiment should test explicit
function boundaries on the current General shader, keeping the same reachable
features and output contract; linked Types alone do not prevent target inlining.

The current module-based 2026-10-13 repeat still shows the same limitation.
The [current O1 comparison](../build-release/link-time-current-20261013/current-O1-analysis/Analysis.json)
reports 6,947,188 bytes, 449,824 instructions and 60,504 blocks for every
general/no-fiber/opaque policy-field arm. Matched static/link-time constants and
matched concrete/link-time Types are byte-identical. The current production
baseline is 6,947,064 bytes / 449,813 instructions / 60,501 blocks. These
observations do not establish an advantage for supplying boolean methods through
a linked Type.

The original coarse-boundary proposal was to select the **whole material-domain dispatch
implementation**, with one Type supplying both found-vertex shading and guide
evaluation. A `GeneralScenePathMaterialBackend` would preserve the current
Surface/Fiber/Slab dispatch. A `NoFiberOpenPBRBackend`, tested first with custom
materials disabled, would call only the existing OpenPBR implementation. Its
methods would contain no `shadeSceneFiber` call or Fiber guide extraction,
rather than guard those calls with another static boolean. This changes the
reachable call graph before target inlining. The follow-up above now measures
this code-elimination boundary and its native/GPU effects. It does not form a
native compilation boundary or establish a benefit from linking alone. The
remaining design notes describe the original proposal and deployment contract.

The concrete sites are:

- [`OpenPBRPathVertex.slang`](../Shaders/Modules/OpenPBRPathTracing/OpenPBRPathVertex.slang),
  `shadeOpenPBRPathVertex`, from its `loadSceneMaterial`/material-domain tail
  (line 434). Extract that complete tail, including the existing alpha retry,
  domain-specific shading and termination bookkeeping. Keep common miss,
  volume-attenuation and ray-cone work unchanged. The general implementation
  retains the current order: Fiber handling precedes BLEND coverage and Surface
  family selection. The narrow implementation retains coverage and the call to
  `shadeOpenPBRSurfacePathVertex`, including its transmitting OpenPBR behavior.
- [`OpenPBRSurface.slang`](../Shaders/Modules/OpenPBRSurface.slang),
  `evaluateSceneSurfaceGuides` (line 602), is a second independent domain dispatch.
  The same selected Type must supply this method too; changing only vertex
  shading would leave the guide side reachable. The narrow method should retain
  the current concrete OpenPBR material evaluation and its returned summary,
  evaluated hit and seed. It must not reuse the path's prepared closure.
- [`SceneSurface.slang`](../Shaders/Modules/SceneSurface.slang),
  `loadSceneMaterial`/`loadSceneOpenPBRMaterial` (lines 213/218), still creates the
  720-byte upload record before extracting model data. A separate resource
  experiment can read a compact classification or coverage view at these domain
  boundaries, then load only the selected model's fields. Keep the CPU upload
  stride and descriptors unchanged. Field offsets must follow
  [`SceneMaterialData.h`](../Source/Runtime/Render/Material/SceneMaterialData.h)
  and its std430 texture padding; do not reinterpret the noncontiguous OpenPBR
  fields as a packed model record. Preserve complete-record bounds and missing
  resource behavior, and verify the mapped readback before comparing timings.

Place the public backend contract alongside the existing path state types in
the `OpenPBRPathTracing` owner; provider modules can import that owner and export
the selected implementation Type. Keep every new module/implementation owned,
use explicit imports and keep the registered entrypoint names stable. Start in
frozen harness overlays: the production compiler still needs explicit provider
composition and provider-aware cache/hot-reload identity before deployment.
The existing `ISceneSlabModel` associated-Closure selection in
`OpenPBRSurface.slang` (line 430) is an actual implementation-selection precedent;
it does not choose a model by storing an existential interface at runtime.

No-Fiber eligibility must cover all materials reachable by primary, secondary
and shadow rays, including streamed assets and custom programs. Disabling the
RTXCR SDK flag or observing an opaque primary hit is insufficient. Begin with
the fixed Opaque/Mixed/Sky fixtures and custom-materials-off configuration;
retain metallic, transmission, volume, MASK/BLEND and existing debug behavior.
When a streamed material, edit or program requires Fiber/Slab, acquire a broader
compatible pipeline before publishing that material to an affected dispatch.
Keep bindings, feedback writes, texture-load/RNG order, authored TBN mapping and
ray offsets unchanged. Guide evaluation keeps its independent seed and
face-forward policy, including camera-motion and specular-endpoint handling.

Use these matched arms with the same O1, mapped ABI, postprocess policy and
normal Driver optimization:

| Arm | Purpose |
| --- | --- |
| Existing general shader | Current behavior and cost baseline |
| Concrete general backend Type | Isolate extraction/interface scaffold |
| Link-time general backend Type | Compare linking against the identical implementation |
| Concrete no-Fiber OpenPBR backend Type | Measure the narrower reachable implementation |
| Link-time no-Fiber OpenPBR backend Type | Compare linking against the identical narrow implementation |

Run the resource-view change as a subsequent separate pair so its effect is not
attributed to Type selection. Count reachable functions/calls, retained Fiber
operations, CFG size, storage-class variables and finalized SPIR-V after the
same cleanup. Compile/module-reuse cost, cold Driver creation and GPU dispatch
remain separate scopes. Requalify HDR, seven guide outputs, moving-camera
captures and material/RNG/feedback behavior. Only a scene-wide eligibility proof
plus these checks could justify using the narrower backend. The linked follow-up
records the compiled arms and keeps these remaining deployment gates explicit.
