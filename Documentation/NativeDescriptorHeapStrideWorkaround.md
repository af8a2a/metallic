# Native descriptor heap literal strides

`TO-REMOVE(VVL payload-size)` marks a temporary compiler policy. Native Vulkan
shaders use Slang's `SPIRVResourceHeapStride` and `SPIRVSamplerHeapStride` options
to emit literal `OpDecorate ArrayStride`. Metallic no longer replaces
`OpConstantSizeOfEXT` instructions before creating a shader module.

The shared resource stride is the maximum of the image and buffer descriptor
sizes after their respective descriptor alignments. The sampler stride is the
sampler descriptor size used by the CPU heap writer. These values are queried
from the GPU and must not be hard-coded. Explicit compilation requests can supply
the values for offline compilation; otherwise runtime compilation uses the
selected device's configuration. Both values are part of the native compiler
cache identity. Mapped shaders do not depend on them.

Startup warmup performs an instance-only property query before its compilation
workers start. It does not create a logical device or change the runtime's
global Vulkan dispatch table. The single-GPU renderer publishes the selected
device's strides before runtime shader compilation. Native SPIR-V whose heap
stride does not match that device, or which still uses opaque-size expressions,
is rejected with an instruction to recompile; module creation never repairs it.

## Why the policy exists

On 2026-10-07, actual graphics-pipeline creation with native production task and
masked mesh shaders reproduced false payload-limit errors in official Validation
Layers 1.4.350 and 1.4.363. Each shader's payload is a fixed `uint[32]` (128 bytes),
independent of descriptor sizes. Retaining the unified heap stride expression
`OpConstantSizeOfEXT -> OpSpecConstantOp UGreaterThan -> OpSpecConstantOp Select`
causes `08758` and `08755`. Replacing the sizes with literal values removes both
errors. All these pipelines still return `VK_SUCCESS`, so the validation
callbacks, not just pipeline creation results, are the regression criterion.

Validation Layers marks an entire module as containing specialization constants
when any `OpSpecConstantOp` is present. Its payload constructor then records
`kSpecConstant` (4294967293) even when the payload's own type is entirely literal.
The task/mesh limit checks interpret this unresolved marker as a byte count.
The heap expression evaluator's support for `Select`/`UGreaterThan` does not
remove the module-wide flag. The same constructor remained in upstream main
`10de01c3da417a12bb28204525bccdff97104f9e`, checked from source only.

The independent verification also checked a valid literal 128-byte payload and
genuinely oversized 20,000/32,768-byte payloads, confirming that the validation
checks remained active. The test machine was RTX 5070 Ti / NVIDIA 616.92.
Verification covered real shader-module and pipeline creation, with no queue
submission. It did not establish rendering or performance of that old binary.

Sources:

- [Slang heap stride compiler options](https://docs.shader-slang.org/en/stable/external/slang/docs/command-line-slangc-reference.html)
- [VVL payload constructor at the audited main commit](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/10de01c3da417a12bb28204525bccdff97104f9e/layers/state_tracker/shader_module.cpp#L2862-L2871)
- [VVL task payload limit check](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/10de01c3da417a12bb28204525bccdff97104f9e/layers/core_checks/cc_spirv.cpp#L1895-L1910)

## Verification of the literal policy

The 2026-10-07 implementation was built with MSVC, Slang 2026.18.2 and the
existing Release configuration. Compiler/cache regressions cover both stride
inputs, mapped independence, missing/invalid inputs, debug output and shared
warmup/runtime requests. Startup tests pass, and real native ColorResize warmup
queries the test GPU's 32/32-byte strides, compiles once and subsequently hits
the cache.

With isolated official VVL 1.4.363, GPU nested-layout, final descriptor index and
heap switching, mixed 32/64-bit atomics, native texture sampling, MixedProducer
task/mesh rendering and native DebugPrintf readbacks pass. A shader deliberately
compiled with an incompatible resource stride is rejected before driver module
creation, with an explicit instruction to recompile for 32/32. Three production
native compile-only tests also pass with an explicit offline test ABI. No VUID,
including `08758` or `08755`, occurs in these runs.

The separate native GPU-driven alpha-mask pixel test still fails on its front
coverage assertion: 12100 pixels, against an upper limit of 7500; unexpected
colors are zero and dark pixels are 4284. The preserved pre-change RHI executable
(2026-10-07 16:07:32, not rebuilt for the comparison) reproduces exactly these
counts with the same Slang DLL, GPU and VVL. The new mapped path passes its front,
back and single-sided assertions. Native alpha-mask visual correctness remains
an existing limitation; do not report the complete rendering suite as passing.
Some supplementary comparison processes also return a nonzero exit because the
sandbox denies HTML report export; their GoogleTest XML, GPU assertions, logs
and native texture output image remain available.

Local evidence is under `.tmp/spirv-removal-research/`: compiler/startup results,
`StrideImplementationWarmup/Results.json`, and the original GPU run and focused
baseline comparison in `StrideImplementationGPU/`. These are local validation
artifacts, not source-controlled captures.

### Driver 617.42 follow-up

On 2026-10-07, the RTX 5070 Ti driver was updated from 616.92 to 617.42.
The initial alpha-mask comparison and complete tessellation pixel matrices used
the exact same SHA-256 RHI executable as the earlier 17:49 run and the same Slang
2026.18.2 DLL. VVL-enabled checks used the same isolated official VVL 1.4.363 DLL.
Existing SPIR-V caches were retained; incompatible driver pipeline caches rebuilt
normally.
The current source was also rebuilt in the compatible `build-pass-stages-nrd`
Release tree, and its core GPU checks reproduced the baseline results.

With VVL enabled, native nested layouts, final descriptor indices/heap switching,
mixed 32/64-bit atomics, texture sampling and MixedProducer rendering pass.
Standard TLAS, partitioned TLAS and position-fetch native ray-query tests also
pass, including their readbacks. Native DebugPrintf completes with matching GPU
readback and 1/1 matching echo records. The wrong-stride input is still rejected
with the explicit 32/32 recompilation instruction. The two
`DescriptorHeapShaderABI` cases are CPU SPIR-V checks, not GPU executions. No
VUID occurs in the completed core or ray-query runs.

Native alpha-mask coverage remains 12100, with 4284 dark pixels and zero
unexpected colors, identical to the 616.92 result; mapped passes. Complete
native displacement and recursive pixel matrices still differ in all 32 and 96
resident combinations respectively, while their 32 and 96 stream combinations
have no image mismatch. Mapped passes all 64/192 combinations and the subsequent
edit checks. These full pixel matrices disable VVL on both paths; the initial
VVL-enabled recursive attempt timed out at 300 seconds and is not a completed
result. See [resident image investigation](NativeResidentImageInvestigation.md)
for the image evidence and scope.

The additional native `opacity_micromap_ray_query` and partitioned variant both
stop at the OMM-disabled fallback: step 0 ray 0 has CPU bilinear alpha coverage
0.260272, below cutoff 0.5, but the GPU reports a hit. Neither test reaches its
OMM-enabled branch. Both mapped variants pass fallback/OMM visibility, edits,
compaction and candidate reduction checks. The preserved 17:49 executable
reproduces the native failure on 617.42 too. Earlier October 3 logs suggest a
native pass, but used different compiler/cache policies; there is no controlled
pre-upgrade run of this exact OMM input. Do not attribute this newly recorded
fallback failure specifically to the driver update or to Slang issue #13438.

All runs retain production pointer normalization, the typed uint64 exception,
AS address resolver and OMM SPIR-V finalization. They do not establish that any
of those policies can be removed. There is no performance comparison or full
scene/temporal acceptance in this retest.

Local evidence is under `.tmp/spirv-removal-research/Driver61742/`: initial
unchanged-binary checks in `20261007-184136-876`, complete pixel matrices in
`20261007-184901-959`, rebuilt core/readback/printf results in
`20261007-190046-998`, mapped OMM controls in `20261007-190301-879`, and the
unchanged-binary native OMM check in `20261007-190308-067`. Summaries record
driver/compiler/layer identities and executable hashes. Some processes still
exit nonzero after passing GPU assertions because sandbox access denies HTML
report path canonicalization; use the retained GoogleTest XML and readbacks to
distinguish assertion failures from report export errors.

### Installed SDK 1.4.363 retest

On 2026-10-07, rebuilt current Release binaries were retested with installed SDK
`D:/Scoop/apps/vulkan/1.4.363.0`; Slang 2026.18.2 and driver 617.42 were unchanged.
The layer DLL's SHA-256 matches the earlier isolated 1.4.363 DLL:
`2ACC317EF880F73A9862F23A964694C03F531B866FAB6D70F1412FCBAE60CAAC`.
Runners explicitly selected the new SDK tools/layer; existing session/system SDK
350 settings and the registry were unchanged. C++ Vulkan headers still come from
the SDL bundle (header version 358); this was not a header upgrade.

The core run passed 13 GPU checks and two CPU ABI checks with zero VUIDs. Native
DebugPrintf also passed GPU readback and 1/1 echo records with the new layer.
Native alpha coverage remained 12100 with 4284 dark pixels; both native OMM tests still
failed in the OMM-disabled fallback at step 0 ray 0. All three mapped controls
passed. With VVL disabled, native displacement/recursive matrices still differed
in all 32/96 resident cases out of 64/192 total combinations, with zero stream
image differences; mapped passed all combinations and edit checks.

The installed layer still reports `08758` and `08755` for raw native
`OpConstantSizeOfEXT` heap expressions. The literal 128-byte control has zero
VUIDs and genuinely oversized controls report the expected errors. Both raw
task/mesh shaders pass the new `spirv-val` (exit 0); that does not replace VVL.

Native default DLSS-RR PathTracingSample first warmed 370 requests (36 hits,
334 compiled, zero failures), created PSOs and presented a frame with exit 0,
with VVL disabled. A second run used `METALLIC_DEBUG_CONTROL=1` and
`METALLIC_DEBUG_VALIDATION=1`, actually loaded the installed 1.4.363 layer, hit
all 370 cache entries and presented a frame with exit 0 and zero VUIDs.
These are startup smoke checks, not full-scene or temporal acceptance.

All production workarounds remain enabled; the literal-stride policy cannot be
removed. Evidence is in `.tmp/spirv-removal-research/Sdk363Retest/`: core
`20261007-204055-079-core`, alpha/OMM `20261007-204221-156-alpha-omm`, pixels
`20261007-204316-043-pixels`, `20261007-205145-094-printf`, `Payload`, `Sample`
and `SampleValidated`; `Results.json` and `Report.md` aggregate the results.

### Unified SDK paths retest, 2026-10-17

The user and machine `VULKAN_SDK` / `VK_SDK_PATH` values and registered explicit
layers now select `D:/Scoop/apps/vulkan/current`, which resolves to SDK 1.4.363.0.
An independent discovery check left `VK_LAYER_PATH` unset and confirmed actual
instance and device insertion of that SDK's validation DLL. Its SHA-256 remains
`2ACC317EF880F73A9862F23A964694C03F531B866FAB6D70F1412FCBAE60CAAC`, identical to
the previously tested 1.4.363 layer. This corrects default discovery; it does not
introduce a newer validation implementation. Driver 617.42 and Slang 2026.18.2
are unchanged. The renderer builds still use SDL's Vulkan header version 358.

The existing Release targets were rebuilt. Direct test processes explicitly
selected the new SDK tools and layer, and validation-enabled runs record actual
instance/device layer insertion. The RHI tree's cached
`METALLIC_TESTBENCH_LAYER_PATH` still pointed to SDK 1.4.350 during these direct
runs; after GPU testing it was reconfigured to
`D:/Scoop/apps/vulkan/current/Bin` without changing the compiler or build variant.
This configuration repair is not an additional CTest or GPU run.
CMake also discovered the installed SPIRV-Tools headers/libraries that had been
cached as not found. The tested executables were not rebuilt after this repair;
its complete cache delta is recorded in `ConfigurationRepair.json`.

Across the 27 planned cases, 25 pass and two alpha-mask assertions fail:

- Core: 13 GPU execution/readback cases and two CPU ABI cases pass, with zero
  VUIDs. Native DebugPrintf also passes its GPU readback and 1/1 echo records.
- Material coverage: native and mapped, including their Slab variants, all pass
  with zero VUIDs. Each case checks 30,752 MASK pixels across raster, ray and
  shadow results; the four BLEND opacity cases check only the ray probe.
- Tessellation: both modes pass all 32 displacement and 96 recursive Stream
  image comparisons and subsequent live-edit checks. Both modes disable VVL for
  these matrices. Three initial matrices stopped on source-epoch notifications;
  fresh-process retries completed with no image differences. The removed
  Resident path is not covered by these current matrices.
- Native default DLSS-RR PathTracingSample completes normal CPU shader warmup
  and native PSO prewarm, submits and presents a frame, and exits 0 with zero
  VUIDs. The previous PSO-loading crash was not reproduced. CPU warmup reports
  132 source requests, zero existing cache hits and zero failures; native
  prewarm creates 109 compute pipelines, with all 109 recorded creation results
  successful. Its 35 graphics source configurations are skipped by the prewarm
  recipe, so 132 sources must not be described as 132 created pipelines.

The alpha-mask test fails identically in both modes: coverage 11668, dark 4716,
unexpected colors zero, and no VUIDs. Its scene helper and test are unchanged
from the earlier comparison, but shared renderer behavior has changed. A static
candidate explanation is that its rear, untextured, alpha=1 BLEND surface is now
classified opaque and fills holes in the front MASK; the counts fit that
geometry. No winner/material-ID capture or isolated control has confirmed this
explanation. These failures establish neither a native-only regression nor a
general MASK failure; the separate material coverage checks pass.

The independent create-only payload probe still blocks policy removal. A
historical raw 128-byte task paired with a synthetic mesh reports false `08758`
and `08755`; a synthetic raw unified-stride pair also reports false `08760`.
The callbacks interpret the 128-byte payload as 4294967293 bytes. Pipeline
creation returns success, and the historical raw task passes the new
`spirv-val`, but neither result cancels those validation failures. Literal and
specialization-controlled 128-byte controls are clean; genuine 20,000/32,768-byte
controls still report the appropriate limits. These probes submit no GPU work.
The current production `TessPatchPayload` is 12,372 bytes; 128 bytes describes
only the historical task and synthetic controls.

The literal-stride policy remains required. Native pointer normalization, typed
uint64 handling and AS resolution remain enabled. Current coverage/CLAS tests
do not exercise the removed OMM tests or establish that historical OMM problems
are fixed. This is current-worktree evidence at base HEAD
`1b8d320c99ea620c0f5dfebbcda3843737f44be6`, with shader and other concurrent
working-tree changes recorded per attempt; it is not a controlled SDK A/B,
full-scene/temporal acceptance or a performance comparison.

Local evidence is under `.tmp/native-compatibility/20261017-UnifiedPaths/`.
`Results.json` and `Report.md` merge the latest result per case while retaining
the original timeout and source-epoch attempts. `Environment.json` and the
default-discovery logs identify the selected layer; `Payload/Runs/20261017-144322-106/`
contains the final eight independent controls. Source/executable hashes and
the pre-repair CMake cache are retained alongside the logs.

## Conditions for removal

Do not retire this policy solely because a newer SDK recognizes descriptor heap
extensions or can evaluate heap stride expressions. Verify all of the following:

1. Unmodified native task and masked mesh compiler output retains
   `OpConstantSizeOfEXT` and unified stride expressions, while graphics-pipeline
   creation reports no false payload-limit VUID (`08758`, `08755` or `08760`)
   for fixed literal payloads, including the 128-byte control and current
   production payload.
2. Genuine oversized payloads and payload array lengths controlled by
   specialization constants still produce the appropriate errors; do not suppress
   VUIDs or simply remove the module-wide specialization check.
3. Native descriptor layout/readback, sampling, atomics, and the affected
   GPU-driven rendering regressions pass with the unchanged device-independent
   output and the same CPU heap layout.

Then restore `SPIRVUnifiedDescriptorHeapStride` and remove the literal compiler
options, stride request/default state, physical-device warmup query, paired
offline CLI flags, device stride compatibility check, and the associated tests.
Bump the compiler request version again so cached literal-ABI binaries cannot be
reused as device-independent output. The removed SPIR-V size-rewriting pass must
not return. The independent native pointer normalization and OMM finalization
are separate policies and are outside this removal.
