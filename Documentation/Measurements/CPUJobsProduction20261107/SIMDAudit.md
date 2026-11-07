# SIMD/backend adoption audit

This is a read-only audit of the M2 implementation at `1069270cc`, followed by
the production-scene investigation for issue #73. No compiler, correctness test
or performance workload was run for this audit. The earlier immutable evidence
is [the kernel report](../CPUJobs20261106/Kernels/Report.md), including its source
and executable hashes. Production frame evidence must be reported separately.

## Existing M2 acceptance

| Requirement | Owning source / evidence | Assessment |
| --- | --- | --- |
| Independent CPU kernel target | `cmake/CPUKernels.cmake`, `Source/Runtime/CPUKernels/` | `MetallicCPUKernels` exposes no TaskGraph, Vulkan, RHI or Slang types. |
| C++ baseline and optional ISPC | `AABBTransformScalar.cpp`, `AABBTransformAutoVectorized.cpp`, `CPUKernels.cmake` | Scalar and compiler-vectorized C++ remain buildable when ISPC is absent/disabled. The three implementations use the same interval algorithm. |
| Cross-core ownership | `AABBTransform.ispc`, `AABBTransform.h` | Exported kernels process only the supplied range. There is no ISPC `launch/sync` or additional worker runtime. |
| ISA policy | `AABBTransform.cpp::supportsISPC`, `CPUKernels.cmake` | SSE4.2/AVX2 variants, generated dispatch, minimum-ISA fallback and AVX2-only CPUID/XCR0 gating exist. Host C++ does not receive an AVX2 requirement. AVX-512 is optional and was not evaluated. |
| Numerical contract | `AABBTransform.h`, `tests/cpu/AABBTransformTests.cpp` | Finite ordered bounds, finite affine transforms/intermediates, non-overlapping output and a scale-sensitive tolerance are explicit. A double-precision eight-corner oracle, random/edge values, masked tails and genuine offsets above 4 GiB are covered by the recorded tests. NaN/Inf are outside this internal kernel contract. |
| Async range composition | `tests/cpu/AABBParallelForTests.cpp` | Producer -> range -> consumer publication covers 1/2/4/8 workers, all partition strategies and every backend. Existing evidence records six integrated tests passing; this audit is not a rerun. |
| Separate SIMD/scheduling attribution | `Kernels/Raw.csv`, `CPUJobs20261106/Range/Raw.jsonl` | Kernel runs have one caller thread; scheduler runs hold the synthetic kernel fixed. They are separate experiments and must not be multiplied into a claimed scene speedup. |
| Compiler/assembly comparison | Prior kernel report and source flags | Recorded MSVC vectorization diagnostics and packed XMM/YMM assembly exist. The ISPC coherent branch has full-gang/unmasked and tail/masked paths. No hardware SIMD utilization counters were collected. |

The code search at this audit finds `transformAABBs()` in the CPU-kernel library
and its test/benchmark callers. It is not silently substituted into scene
loading, scene bounds, picking or visibility code. `Automatic` selects C++ auto
vectorization; explicit ISPC requests report the actual backend or C++ fallback.

## Why production adoption is conditional

The archived one-thread result is workload-dependent: runtime-dispatched AVX2
was 1.72x faster than C++ auto at 65,536 AABBs, but 27% slower at 1,048,576.
The separately measured simple AoS-to-SoA conversion was material and did not
include an output conversion. These results support an optional candidate,
not an unconditional renderer switch.

The current scene transform helper in
[`scene.cpp`](../../../Source/Runtime/Scene/scene.cpp) consumes an AoS `Bounds`
and matrix, preserves `Bounds::valid`, and includes eight transformed corners.
Its callers include initial scene bounds and later bounds recomputation.
[`ScenePicker.cpp`](../../../Source/Runtime/Scene/ScenePicker.cpp) has a similar
helper and consumes the result for its BVH. The independent SoA kernel instead
requires valid finite batches and promises tolerance equivalence rather than
bitwise identity or conservative outward rounding. An adoption must therefore
establish the consumer's error/margin policy, preserve invalid/empty bounds,
and include filtering, packing, output publication and relevant lifetime costs.
This audit does not infer that import or scene mutation already guarantees the
kernel's stronger finite-input contract.

The next decision uses the fixed `SuperSponza` production workload: identify
actual CPU scope/critical-path contributions, then compare a complete owning
path if it is material. A static scene's bounds update must not be promoted to a
frame hotspot merely because an isolated transform kernel has a speedup.
Geometry cooking and meshlet simplification also cannot be relabeled as this
affine-AABB workload without an equivalent kernel and correctness comparison.

No additional backend/source change is justified merely to mark an adoption
checkbox. Retain C++ default and explicit ISPC until a matched real workload
shows a repeatable complete-path benefit with correctness preserved. A negative
or immaterial result is an adoption decision, not evidence that the isolated
kernel implementation is missing.

## Remaining limits

Physical unsupported-ISA hardware, hardware SIMD utilization and production
AoS/SoA layout changes remain untested by the archived M2 runs. Current
production telemetry/plots must keep these limits explicit. Inclusive CPU
profiler scope durations may overlap and do not by themselves establish a DAG
critical path. Process/frame statistics and resource peaks must name their
measurement windows, units, sample populations and legacy/shared executable
identities. Missing metrics stay unavailable; no artificial zeros or additive
scope breakdown may be used to complete acceptance.
