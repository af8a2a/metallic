# Shader compilation complexity (2026-10-12)

This document distinguishes measured compiler evidence from candidate shader
experiments. Production shader source and external snapshots were not modified
by this research. The fixed baseline is the measured Guides request with
`METALLIC_CUSTOM_MATERIALS=0`,
`METALLIC_HAS_RTXCR=1`, SHARC off and the general multi-sample path. Its O1
SPIR-V contains one large function. Attribute experiments should retain O1,
floating-point mode, descriptor ABI, source capabilities and rendering settings.

## Existing cost evidence and identity

The `compute-cold-optimization-n3` O1 prefill report records front-end/cache wall
19.816 s, code-generation API 18.775 s, and SDK-reported downstream time 17.999 s.
The downstream duration is contained in code generation, not an extra additive
phase. Module loading took 0.783 s and component `link_ms` only 0.035 ms; that
small component-link call does not include all IR linking/optimization.

The independent `compute-frontend-cli-O1` diagnostic produced byte-for-byte
identical SPIR-V to that Benchmark module and entry point. Its raw SDK report
has `compileInner=18.762 s`, downstream compile 17.40 s, specialization 196.83 ms,
resource-usage specialization 66.77 ms, and force-inlining passes 7.20 ms.
Nested timers overlap. These observations locate the high front-end cost in the
downstream SPIR-V processing as a whole; they do not assign 17.40 s to one pass
or reveal the driver's internal compiler stages.

The relevant program is module
`Features/PathTracing/OpenPBRRayQueryPathTraceGuides`, Slang entry
`openPbrRayQueryPathTraceGuidesMain`, finalized SPIR-V entry `main`, mapped ABI,
Slang 2026.18.2 / O1 / debug disabled. The identical diagnostic bytecode is
6,940,612 bytes, FNV-1a64 `ef751a428c7a9c48`, SHA-256
`414c4001b496f6e44cdd2e86faa57339d891acc4473e77f34da0b9d196f63b38`.
Retain both raw compiler and final device-rewritten identities; a shared source
filename or entry label is insufficient for an ISA/cold-creation comparison.
Compiler binary, dependent modules/includes, macro set, device/driver and PSO
state identities belong with each result.

The prior O0 arm exceeded the declared HDR tolerance at two pixels. This analysis
does not change the production O1 default or treat O0 as an accepted equivalent.
See [Compute experiments](ComputePipelineExperiments.md) for complete paired
creation/execution conditions and the existing qualification failure.

## Static SPIR-V findings from exact real PSO recipes

`Tools/Perf/AnalyzeSPIRVComplexity.py` now reads raw SPIR-V without changing it,
checks recipe byte count/FNV binding, and exports instruction categories,
per-function CFG/SCCs, static call graph, callsite expansion contributors and
storage classes. The SDK Khronos opcode grammar and GLSL.std.450 grammar are
hashed in its report. This structural reader complements `spirv-val`; it is not
a complete validator. Thirteen CPU tests cover diamonds, loops, 64-bit switches,
recursion, deep graphs, source attribution, memory storage classes and malformed
inputs/recipe identity. The completed 36-module analysis is in
[`spirv-complexity-20261012-final`](../build-release/spirv-complexity-20261012-final/Complexity.json).

| Guides | O0 | O1 | O2 |
| --- | ---: | ---: | ---: |
| SPIR-V bytes | 709,664 | 6,940,612 | 1,992,720 |
| Module instructions | 42,673 | 447,484 | 115,049 |
| Functions | 574 | 1 | 1 |
| Largest function instructions | 4,192 | 446,028 | 113,727 |
| Basic blocks | 4,594 | 58,220 | 17,277 |
| CFG edges | 5,020 | 74,427 | 22,503 |
| Loop merges | 55 | 495 | 242 |
| Phi nodes | 0 | 0 | 7,369 |
| Ordinary cold native median, 3 processes, seconds | 56.160 | 63.109 | 50.303 |

Cold medians come from the prior controlled-cache, capture-disabled experiment;
the static analysis reuses its corresponding finalized bytecode. The O1 Guides
instruction count is 27.65 times the next largest shader in this particular
12-Compute opaque-scene cohort (`gpuDrivenStreamAssetCooperativeLodMain`,
16,181 instructions). It is already the dominant startup PSO in the prior scene
measurement. This cohort does not contain a SceneRTXDI recipe, so these findings
do not establish a SceneRTXDI ranking or identify the driver's internal passes.

The O0 entry reaches 574 distinct functions and its full static-callsite
expansion proxy is **1,029,336**, with maximum static call depth **20**. Thus
smaller serialized O0 does not establish a smaller program after inlining.
The proxy visits all static callsites, including both branch arms, counts a
callsite inside a loop once, repeats function headers/parameters, and performs
no constant folding/dead-code removal. It is neither an actual optimized code
size nor dynamic work nor a fraction of Driver compile time. An independent
topological recomputation agrees with the analyzer's SCC-based calculation.

| O0 static expansion contributor | Occurrences × own instructions | Contribution to proxy |
| --- | ---: | ---: |
| `Metallic.getResourceParameters` | 603 × 497 = 299,691 | 29.115% |
| `openPBRComplexDivide` | 2,223 × 30 = 66,690 | 6.479% |
| `openPBRSquare` | 5,252 × 6 = 31,512 | 3.061% |
| `Metallic.getParameters` | 604 × 52 = 31,408 | 3.051% |
| `openPBRFresnel` | 795 × 36 = 28,620 | 2.780% |

Function IDs disambiguate identically named generic specializations; retained
`OpName` does not encode their complete signatures. All three Guides modules
lack `OpLine`/`OpNoLine`, so the tool does not invent source-line attribution.
The O0 names support matching source functions and selecting experiments;
they do not prove which source line consumed compile time.

O1 has **26,347 Function-storage variables, 40,669 Function-storage loads and
49,761 Function-storage stores**; O2 has **66 / 319 / 147** respectively. These
are IR temporaries, not evidence of local GPU memory or spilling. O1/O2 have
279/36 surviving `LoopControl Unroll` requests; counts neither specify trip
counts nor establish which requests the Driver honored. A large CFG SCC can
contain many nested loops and should not be described as a single loop.

Reproduce the structural analysis without compiling or creating a Pipeline:

```powershell
& 'C:/Users/jiangwentao/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B `
  Tools/Perf/AnalyzeSPIRVComplexity.py `
  --recipe-dir build-release/compute-gpu-optimization-512/block-00-O0-pipeline-normal/recipes `
  --recipe-dir build-release/compute-gpu-optimization-512/block-00-O1-pipeline-normal/recipes `
  --recipe-dir build-release/compute-gpu-optimization-512/block-00-O2-pipeline-normal/recipes `
  --output-dir build-release/spirv-complexity-new
```

Metallic shader disk-cache `.spv` files contain a container header/dependencies;
use raw recipe modules or validate and extract the container first.

## Highest-priority source experiment: narrow resource reads

The whole-packet getter is a stronger measured structural candidate than a
blanket optimization-level change. `Shaders/Modules/Core/NamedResources.slang:8`
loads the 24-byte two-span compute root, then the full 384-byte
`SceneResourceParameters` packet for each leaf access. The existing
`getResourceParameterField<TPacket,TField>` at line 18 reads only the resources
span and the requested field while preserving full-root/full-packet size,
alignment and bounds checks. Keep the existing wire ABI and use this helper at
the leaf adapters; introduce separately declared views with `offsetof` checks
rather than rearranging the packet.

Candidate views, in bytes, derived from current declarations:

| Field/view | Offset | Bytes | Contract to preserve |
| --- | ---: | ---: | --- |
| `scene` / acceleration-structure address | 0 | 8 | `resolveAccelerationStructure` |
| `lights` + `lightsPdf` | 88 | 8 | Span/count and PDF behavior |
| material sampler | 108 | 4 | Sampler identity |
| material textures span | 112 | 12 | Bounds and `resolveNonUniform` |
| materials handle | 140 | 4 | Material identity |
| OpenPBR 2D/3D LUT spans + sampler | 172 | 28 | Both bounds and nonuniform resolution |
| ray stream header/instances/page table/pages | 244 | 16 | Resource identities |
| ReGIR grid | 260 | 4 | Separate from nonadjacent lights fields |
| environment view | 356 | 28 | Distribution/LUT and sampling semantics |

These offsets are research candidates, not a new API or an implementation.
Confirm CPU/shader offsets by static assertions before use. The existing
`named_resource_field_views_and_bounds` RHI test covers truncated packet,
out-of-bounds and unaligned accesses. Extend coverage for new views and compare
the same final GPU outputs. Guides reads camera constants once at its entry;
most of the 604 root-getter occurrences above accompany the 603 resource-getter
occurrences. Begin with resource leaf getters, then inspect O1 bytecode before
attributing a benefit to this change. If O1/Driver already eliminates the
additional fields, the experiment may show no native or GPU improvement.

## Driver executable evidence and ISA availability

Added opt-in Compute Pipeline capture using
`METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY=<fresh directory>`. It enables
`VK_KHR_pipeline_executable_properties` when supported and creates pipelines
with both capture-statistics and capture-internal-representations flags. It
retains the driver's statistic names/descriptions/formats and exact representation
bytes in a per-capture manifest with device/driver/cache UUID, PSO and both
input/final SPIR-V fingerprints. Properties/statistics/representation enumeration
is capped at 256 entries, representations at 256 MiB total, with finite
`VK_INCOMPLETE` retries and exclusive capture-directory creation across
processes. Unsupported queries and empty representations are recorded. Files
use numeric names, not driver-provided path text. The opt-in does not change
ordinary pipelines. Shader Object has no `VkPipeline` executable to query.

Queries/export happen outside the native creation timer; they remain inside
the caller's pipeline wall time. Capture flags may change compilation. Use
fresh explicit application caches and independent processes, and keep these
diagnostic timings separate from the ordinary creation/GPU experiments.
[Vulkan describes internal representations as implementation-dependent](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPipelineExecutableInternalRepresentationsKHR.html):
they may be intermediate IR, binary or assembly. Their label/content must be
examined before calling them ISA.

Three independent captured recipe replays completed on the same RTX 5060 /
driver 616.92, with debug/validation disabled, controlled internal cache disabled
and empty application caches. All input/final SPIR-V byte fingerprints match
the static table. All three query the `CS` executable successfully, subgroup 32,
and return **zero internal representations** (`VK_SUCCESS`). The driver
executable statistics are:

| Captured Guides | O0 | O1 | O2 |
| --- | ---: | ---: | ---: |
| Register Count | 255 | 255 | 255 |
| Binary Size (bytes) | 1,637,248 | 1,776,640 | 1,777,024 |
| Stack Size (bytes) | 0 | 0 | 0 |
| Shared Memory Size (bytes) | 12,288 | 12,288 | 12,288 |
| Local Memory Size, raw UINT64 | 68,719,479,456 | 68,719,489,520 | 68,719,489,600 |

The local-memory statistic is implausibly large for per-thread allocation and
is retained verbatim, unqualified. Do not truncate it to 32 bits, reinterpret
its upper bits, or claim measured spill traffic/occupancy from this value.
The equal register count alone does not establish occupancy or local-memory
behavior. Captured O1→O2 cuts serialized instructions by 74.3%, but executable
byte size changes by only **384 bytes (+0.022%)** and register count is unchanged.
The Driver therefore largely converges these two front-end forms to similar
code size; easier incoming IR may still reduce intermediate compilation work.
Neither the code-size statistic nor the existing ~20% lower ordinary O2 native
median identifies a particular Driver pass. O2's much higher front-end cost also
prevents treating that native reduction as an overall source-cold speedup.

The [capture summary](../build-release/driver-complexity-20261012-v2/summary.json)
links the exact recipes, commands, Bench executable hash and raw manifests.
It records n=1 diagnostic durations, not a new performance qualification.
To reproduce a single baseline capture in a fresh output directory:

```powershell
$env:METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY = "$PWD/build-release/driver-capture-new/representations"
$env:METALLIC_VK_INTERNAL_PIPELINE_CACHE = 'disabled'
try {
  & ./build-release/Source/MetallicShaderPipelineBench.exe `
    --recipe-file build-release/compute-gpu-optimization-512/block-00-O1-pipeline-normal/recipes/compute-5aa0734d26f3cc25.recipe.json `
    --iterations 1 --pipeline-cache empty --shader-cache reuse `
    --descriptor-mode mapped --debug-mode disabled `
    --cache-dir build-release/driver-capture-new/cache `
    --output-dir build-release/driver-capture-new/bench
} finally {
  Remove-Item Env:METALLIC_VK_PIPELINE_EXECUTABLE_DIRECTORY
  Remove-Item Env:METALLIC_VK_INTERNAL_PIPELINE_CACHE
}
```

### Native binary and Nsight attempts

The existing O1 Shader Object and Pipeline Binary containers were parsed and
verified against their magic/version/lengths/FNV checksum and successful
independent-process import records. Original caches were not modified.
Shader Object payload is 4,430,994 bytes with no ELF. Pipeline Binary payload
is 5,385,894 bytes; its standard Zstd frame at byte 4 decompresses exactly to
12,090,716 bytes beginning `NVDANVVMNvuc`, still with no ELF. Installed CUDA
13.3.73 `nvdisasm` rejects all these payloads as non-ELF. Header text alone
does not prove an NVVM IR schema. Raw disassembly with guessed instruction
offsets/architecture was not used. These cache payload sizes do not measure
machine instructions or executable code bytes. See
[NativeBinaryInspection.json](../build-release/shader-complexity-research/native-binaries/NativeBinaryInspection.json)
and its sibling `Inspection.md` for exact commands, source/import bindings,
tool identities, hashes, decompression checks and rejection logs.

The [Nsight Graphics skill](C:/Users/jiangwentao/.codex/skills/cli-anything-nsight-graphics/SKILL.md)
and its CLI doctor confirm installed Nsight Graphics 2026.3.1 capture/replay.
That supported CLI exposes trace/capture metadata, screenshots and timing,
but no arbitrary shader/SASS export. Native desktop UI control is unavailable
in this session. [NVIDIA's Shader Profiler documentation](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html)
says full SASS disassembly requires Pro builds; this installation's Pro
eligibility was not established. A next diagnostic session can use Real-Time
Shader Profiling to examine actual instruction mix/stalls and, where eligible,
full SASS. Source correlation requires a separately identified debug-bearing
module: current release SPIR-V has no `OpLine`. Retain ordinary no-capture
timings/output checks as the performance reference.

No trustworthy Driver ISA was obtained by these three routes. The executable
statistics and static SPIR-V findings above are the available evidence, and
they support experiments without fabricating assembly-level attribution.

Validation completed for this change:

- Release `MetallicShaderPipelineBench` and `MetallicRHITests` build succeeded.
- Focused perf CTest passed 5/5 targets, including the 13-case complexity suite.
- Bench CLI tests passed six cases; four opt-in GPU cases were skipped.
- `typed_named_kernel_parameters_and_lifetime` passed with diagnostics absent,
  and twice in separate processes with capture and Vulkan validation enabled;
  no validation errors were emitted. Each process created two distinct capture
  directories for the same two PSO hashes, with earlier manifests retained.
- All three real Guides diagnostic replays succeeded and preserve their exact
  raw/final bytecode identities. The real shadow-only candidate passed SPIR-V
  validation. No candidate shader was run in the renderer or promoted, and
  production shader source/optimization defaults remain unchanged.

## Source boundaries and feature state

- `OpenPBRSurface.slang:33-46` specializes sheen/coat off when custom materials
  are off; dispersion is always off; translucency and metallic are on. The
  fixed request's growth cannot be assigned to disabled coat/dispersion or Slab
  variants. This conclusion must be revisited for requests with other macros.
- `OpenPBRPathVertex.slang:59-77` retains the fiber branch at runtime. The
  `METALLIC_HAS_RTXCR=1` request contains Chiang hair preparation/evaluation even
  when the measured room contains only opaque surface materials.
- `SceneFiberPath.slang:8-106` evaluates the prepared fiber closure in celestial,
  punctual and environment direct-light paths, then samples it. The same hair
  evaluator has multiple static call sites before exhaustive inlining.
- `OpenPBRPathVertex.slang:170-240` keeps prepared BSDF state alive over direct
  lighting, PDF/evaluation and nested visibility; `:275` later samples it. The
  SHARC-query single-sample special case already moves remaining BSDF work before
  visibility, but the general SHARC-off multi-sample baseline does not use that
  branch. Preserve RNG and lighting/termination order in any lifetime experiment.
- `OpenPBRRadianceCache.slang:10-24` directly evaluates the full mixture in a
  light loop; `OpenPBRPathVertex.slang:209-211` separately calls PDF and projected
  evaluation for the environment. `OpenPBR/AggregateLobe.slang:145-213` traverses
  the six base lobes independently for value and PDF. `:217-307` samples a
  selected lobe, then evaluates/PDFs the other matched lobes to produce the full
  mixture weight. Dropping those mixture terms changes the estimator.
- `OpenPBRClosure.slang:11-128` stores and passes a nested prepared BSDF value;
  `OpenPBR/BSDF.slang:43-52` nests fuzz → coating → aggregate and its six lobes.
  Whole-struct constants/loads/inserts/extracts may grow intermediate IR, but
  source by-value syntax alone does not prove actual memory copies or spills.
- `OpenPBRRayQueryPathTraceGuides.slang:65-72` evaluates its own material. Its
  guide stream, footprint and face-forward policy deliberately differ from the
  path stream (`:112-123`). Reusing the path closure for guides would violate
  that existing contract. Keep authored normal/TBN data stable.

## Lowest-risk experiment: selective noinline at O1

[Slang's documented noinline attribute](https://docs.shader-slang.org/en/latest/external/core-module-reference/attributes/noinline.html)
requests that a downstream compiler retain a function boundary. SPIR-V
`DontInline` is a performance hint, not a guarantee that the driver preserves a
machine-code call. [Function control specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_function_control)
and [SPIRV-Tools inline pass](https://github.com/KhronosGroup/SPIRV-Tools/blob/main/source/opt/inline_pass.cpp)
show the representation and the optimizer's check.

A local integer probe tested the installed Slang 2026.18.2 compiler. O1 and O2
without `[noinline]` each produced one function, zero calls and two loop bodies.
With `[noinline]` each produced two functions, two calls and one loop body;
the helper retained `DontInline`. O1 byte size changed 1,676 → 1,428; O2
1,272 → 1,168. All four binaries passed `spirv-val --target-env vulkan1.3`.
This proves the pinned SDK can preserve this boundary in simple SPIR-V; it does
not measure Guides creation, numerical equivalence or GPU performance.
The [probe summary](../build-release/shader-complexity-research/InlineProbeResults.json)
and [source](../build-release/shader-complexity-research/InlineProbe.slang) are
local diagnostics outside source control. They record hashes and compiler/tool
identities. Reproduce a probe arm with:

```powershell
$root = 'build-release/shader-complexity-research'
& External/slang/bin/slangc.exe "$root/InlineProbe.slang" -entry main `
  -stage compute -target spirv -profile spirv_1_6 -O1 `
  -D TEST_NOINLINE=1 -o "$root/O1-1.spv"
& 'C:/VulkanSDK/1.4.363.0/Bin/spirv-val.exe' --target-env vulkan1.3 `
  "$root/O1-1.spv"
& 'C:/VulkanSDK/1.4.363.0/Bin/spirv-dis.exe' "$root/O1-1.spv" `
  -o "$root/O1-1.spvasm"
```

A real Guides **shadow-only** isolated O1 compile was also completed. All 339
shader files were copied into an ignored overlay along with the required
relative CPU interop/includes; the only shader edit adds `[noinline]` to
`openPbrShadowTransmittance`. The depfile confirms that shader dependencies came
from the overlay, and original source hashes remained unchanged. It uses the
same fixed request's CLI flags/macros/capabilities as the byte-identical O1
reference, with depfile/include diagnostics added. `spirv-val --target-env
vulkan1.4` passed. The full binding, compiler/source hashes, command and results
are in [CandidateShadowOnly.json](../build-release/shader-complexity-research/CandidateShadowOnly.json).

| O1 Guides | Baseline | Shadow-only noinline |
| --- | ---: | ---: |
| Bytes | 6,940,612 | 6,946,528 (+0.085%) |
| Instructions | 447,484 | 447,751 (+0.060%) |
| Functions / calls | 1 / 0 | 3 / 2 |
| Largest function instructions | 446,028 | 427,449 (-4.17%) |
| Basic blocks / CFG edges | 58,220 / 74,427 | 58,222 / 74,427 |

The two helpers each contain 9,364 instructions and 1,453 basic blocks, retain
`DontInline`, and each have one static caller. Their common source name does not
mean they were deduplicated. This proves preservation of the boundary in the
large real shader but shows **no total-size reduction** for this candidate.
Compile timing overlapped the C++ build and is not a performance measurement.
Driver creation, dispatch and output qualification were not run for the
candidate. Keep it as a bounded maximum-function-size experiment, below narrow
resource access in priority; do not claim a cold-creation improvement.

Run isolated variants in this order; inspect final device SPIR-V before timing:

| Variant | Boundary | Reason / risk |
| --- | --- | --- |
| Shadow-only | `OpenPBRRayQueryPathTrace.slang:134`, `openPbrShadowTransmittance` | Shared transparent traversal under direct and environment lighting. Keeps deeply nested ray-query/texture/control flow outside the large path caller. Pass-by-value interior stack may create an ABI/temporary cost. |
| Fiber-only | `SceneFiberPath.slang:106`, `shadeSceneFiber` | Isolates the entire uncommon material domain while keeping runtime semantics. Check actual ray-query legality/driver call behavior and fiber scenes. |
| Fiber evaluator | `RTXCRFiber.slang:23`, `RTXCRChiangPreparedClosure.evalProjected` | Narrows repeated Chiang evaluator expansion across three light paths; a smaller alternative to the whole integrator boundary. |
| BSDF eval/PDF | `OpenPBRClosure.slang:30-35`, `evalProjectedComponents` and `pdf` | Reuses costly pure BSDF subgraphs at the integrator facade. Large prepared-state arguments could trade inlining growth for stack/register overhead; capture ISA/local-memory evidence. |
| BSDF sample | `OpenPBRClosure.slang:60`, `sampleWeightedComponents` | Separates full mixture sampling after visibility. Keep failure-output handling and full mixture PDF unchanged. |

First run each boundary independently, then combine the winners. Do not begin
with every function noinline: resource/legalization functions may require
inlining and many small calls can cost more than they save. Inspect `OpFunction`
controls, maximum function size, branch/Phi counts and maximum call depth. If
Slang/legalization already removed a boundary, classify it as an ineffective
hint before claiming the driver was tested with a smaller caller.

## Custom SPIR-V passes: explicit second experiment

The installed SDK documentation (`External/slang/share/doc/slang/user-guide/08-compiling.md:378-384`)
says `-Xspirv-opt` appends passes to the selected optimization preset; it does
not replace O1's preset. The public API exposes
`CompilerOptionName::DownstreamArgs` with tool name plus one argument per line
(`slang.h:1109`). There is no proven public `UseDefaultPasses=false` switch in
this SDK. Adding `--ssa-rewrite` at O1 cannot undo exhaustive inlining.
[The current Slang downstream implementation](https://github.com/shader-slang/slang/blob/master/source/slang-glslang/slang-glslang.cpp)
corroborates preset-plus-extra-pass behavior and includes exhaustive inline in
the default sequence; it is an upstream reference, not a claim that current
master equals this installed binary.

The reproducible alternate experiment starts with the already preserved O0
SPIR-V function graph and runs an explicit external optimizer sequence. Do not
promote the result because it is smaller: the previous O0 rendering experiment
failed the existing tolerance for two pixels. It must be newly qualified against
the current O1 reference. Keep an unmodified O0 arm to distinguish SSA cleanup
from function preservation, plus a pure `--compact-ids` control to distinguish
ID density from actual program transformations.

Installed tools are in `C:/VulkanSDK/1.4.363.0/Bin`, reported
`SPIRV-Tools v2026.4 v2026.4.rc2-0-gef96ed76`. One conservative candidate is:

```powershell
$opt = 'C:/VulkanSDK/1.4.363.0/Bin/spirv-opt.exe'
& $opt --target-env=vulkan1.3 --preserve-bindings --preserve-interface `
  --eliminate-dead-branches --eliminate-dead-functions `
  --eliminate-local-single-block --eliminate-local-single-store `
  --ssa-rewrite --simplify-instructions --eliminate-dead-code-aggressive `
  --cfg-cleanup --compact-ids input-O0.spv -o output-ssa-no-inline.spv
& 'C:/VulkanSDK/1.4.363.0/Bin/spirv-val.exe' --target-env vulkan1.3 `
  --scalar-block-layout output-ssa-no-inline.spv
```

This is a prepared candidate, not a completed Guides A/B. Avoid `-O`, `-Os`,
`--inline-entry-points-exhaustive` and unrestricted `--scalar-replacement=0` in
the first experiment. Add bounded scalar replacement later only if aggregate
opcodes remain a measured problem. Run validation after each pass during
diagnostics and keep that overhead separate from timed compilation. Descriptor
heap extension requests need tools/environment validation matching their exact
capabilities rather than assuming a mapped-descriptor command applies.

## Medium-risk changes if boundary hints are effective

1. Fuse BSDF projected value and PDF for the same direction at the adapter API,
   sharing GGX half-vector/Fresnel/distribution intermediates. Keep the same
   six-lobe mixture, normalization, event flags and numerical operation order
   where possible. Compare environment eval/PDF and all sample weights; avoid
   changing vendor snapshots for a first prototype.
2. Shorten prepared-closure live ranges around visibility by computing the
   values needed after traversal earlier. The query/single-sample branch in
   `OpenPBRPathVertex.slang:189-206` provides an existing narrow example. Extending
   it requires proving identical random-number consumption and debug return
   order. Measure actual registers, local-memory spills and ISA bytes; smaller
   SPIR-V alone is insufficient.
3. Derive a small validated scene capability mask (fiber, nested dielectric,
   custom families) and compile a general fallback plus only the needed feature
   variant. The guarantee must include secondary rays, stream arrivals, hot
   updates and shadows. A material weight equal to zero in one fixture is not
   sufficient reason to remove the feature from the general renderer.
4. If selective boundaries still leave a large machine-code body, separate
   primary guides/material evaluation and the secondary continuation kernel.
   Preserve authored primary hits and independent guide RNG, carry compact
   transport state, and budget queue/bandwidth/barrier costs. Existing
   `OpenPBRPathContinuation.slang` and `RayMaterialQueue.slang` provide relevant
   boundaries, but do not establish that full wavefront continuation is wired.
   [pbrt's primary implementation discussion](https://pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU)
   explains the tradeoff: smaller kernels can allocate registers separately and
   improve coherence, but queued state adds memory traffic and launches. This
   supports a bounded experiment, not an assumed speedup on Metallic.

Each candidate should record final bytecode hash, unchanged workload/settings,
front-end/downstream/native creation times, pipeline executable code/register/
local-memory statistics when available, and pass timestamps from ordinary runs.
Qualify final HDR and all seven guides at current tolerances on opaque, mixed
MASK/BLEND/nested-dielectric, sky/miss, debug-normal and fiber workloads; include
secondary hits and scene changes for capability variants. Do not label O1→O0
as equivalent merely because almost every pixel agrees. A verified improvement
must reduce cold creation and retain acceptable GPU time and correctness.
