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

## Conditions for removal

Do not retire this policy solely because a newer SDK recognizes descriptor heap
extensions or can evaluate heap stride expressions. Verify all of the following:

1. Unmodified native task and masked mesh compiler output retains
   `OpConstantSizeOfEXT` and unified stride expressions, while graphics-pipeline
   creation reports neither payload-limit VUID for a fixed 128-byte payload.
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
