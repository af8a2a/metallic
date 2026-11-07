# Issue 73 acceptance checklist

This records the evidence and final scope disposition on 2026-11-07. The
[production report](Report.md) separates completed correctness/resource work
from the ordinary frame matrix that the user stopped to reduce the test
budget. No further test or benchmark is required by this checklist under that
revised budget. Unexecuted matrix cells and unavailable checks are not passes.

Full original production records are retained in the [versioned evidence
release](Storage.md); source control contains compact per-run summaries,
metadata, original hashes and as-run plots. Storage migration adds no samples
and does not change any verification or performance result below.

## Evidence already established

| Milestone | Established result | Evidence / boundary |
| --- | --- | --- |
| M0 backend ownership/audit | Locked stdexec and oneTBB revisions, existing BWOS stealing, queue and wake semantics audited | [CPUJobSystem](../../CPUJobSystem.md); source inspection is not measured fairness |
| M0 scheduler A/B | 14,952 raw samples, workers 1/2/4/8, ordinary/directed scheduling, TaskGraph, legacy, oneTBB, three range strategies and grain comparisons | [Scheduling report](../CPUJobs20261106/Report.md), raw records and source/binary identities |
| M0 observability | Wall quantiles/throughput, available queue delay and callback occupancy, range CAS/steals, coarse process CPU accounting | Actual OS idle/wakeup, upstream deque steals and isolated atomic instruction cost remain unavailable |
| M0 reporting conditions | First-call versus warmed conditions, interleaved candidates, validation separate, raw data and reproducible scripts | No flushed-cache claim or renderer FPS inference |
| M1 range API | Static/Dynamic/HalfSteal, packed endpoint CAS, wide arithmetic, aligned slots, bounded lanes | No ISA/Vulkan dependency; caller chooses grain |
| M1 lifecycle/composition | Async entry/completion, joined outer calls, nested inline stop inheritance, shared ownership, admission/shutdown serialization | Existing TaskGraph cancellation/failure semantics retained |
| M1 production migration | Meshlet adapter owns no threads; benchmark-only legacy comparison retained | Dynamic/grain 1 preserves ordered publication; scratch slot is not thread affinity |
| M1 correctness | Final Task33/33; all19 range tests repeated100 times passed | Exact element counts, tails, huge intervals, exceptions/stop/nesting/concurrent submission/teardown and diagnostic identity/timestamps; no speed threshold assertions |
| M1 real builder equivalence | 288 generated-grid full-builder steady samples with all logical geometry hashes equal | [Meshlet report](../CPUJobs20261106/Meshlet/Report.md); generated grids are not SuperSponza loading or whole-frame performance |
| M2 optional independent target | C++ fallback and optional SSE4.2/AVX2/multi-target ISPC build/test variants | No launch/sync, no global AVX2 requirement |
| M2 numerical contract | Scalar/auto/ISPC interval algorithm checked against double corner oracle, random/finite edge cases, masked tails and real large offsets | NaN/Inf and conservative outward rounding are outside this experimental contract |
| M2 codegen and attribution | 3,456 one-thread samples, compiler diagnostics/assembly, contiguous masked-tail versus full-gang code, packing costs | [Kernel report](../CPUJobs20261106/Kernels/Report.md); cannot multiply by scheduler scaling into a frame speedup |
| M2 TaskGraph composition | 1/2/4/8 workers and every strategy/backend preserve publication | Existing integrated kernel tests; no production adoption implied |
| M3 final regression | Task33, Scene160 passed/9 explicit skips, CPU kernels6 passed; core RHI26 and separate sync RHI18 passed | The focused synchronization profile is separate from the full-DLSS scene's core validation; unavailable paths are not implied by these passes |
| M3 range correlation | Owned operation/backend metadata, graph/node/thread identities, same-clock endpoints and nested parent identity | [Range example](RangeDiagnostics/Example.json), repeatable manual example source; diagnostics are optional |

## Production evidence and scope disposition

| Acceptance item | Disposition and evidence |
| --- | --- |
| Workload identity | Established: requested SuperSponza scene, authored fixed camera, prebuilt stream, full texture identities, actual graph/SDK quality and render/output extent retained in [Assets](Assets.json), [configuration](SuperSponzaConfig.json) and the report. This is a fixed view, not a camera route or edit workload. |
| Matched implementation/settings | Established: same Release renderer, compiler, SDKs, TaskSystem, graph, quality and cache policy; only the linked Meshlet adapter differs. It is not a comparison against the entire issue-start renderer. |
| Runtime conditions | Established: optimized shader mode, capture-injection state, VSync request/log evidence, Reflex requested/effective settings and sensor cadence retained. Validation/diagnostic timing is excluded from ordinary performance comparisons. |
| Frame timing contract | Established for retained complete runs: normal frames-in-flight and Present, start-to-start loop timing, labelled final end-boundary interval, no per-frame drain/readback. Coordinator segments and inclusive CPU scopes remain separate. |
| CPU attribution | Established within instrumentation scope: [Notes](Notes.md) and [task graph audit](SoakAudit/TaskGraphAudit.json). Non-validation diagnostic soak runs each contain6000 `Prepare VBuffer` graphs inside their6000-frame measured envelope; core-validation runs contain none because the debug observer forces inline preparation. Actual pass command recording remains Serial. Queue delay is not isolated wakeup time, and these scopes do not reconstruct an arbitrary DAG critical path. |
| Changed-work attribution | Established: fixed frames reuse the prebuilt streamasset; LOD construction and experimental SIMD are not executing frame paths. The separate real SuperSponza cooker exercises the changed adapter. Absence of range diagnostics alone is not used to prove this distinction. |
| Real cook correctness | Established:48 fresh-output cook processes plus48 separate validation processes, workers1/2/4/8 and three independent ABBA/BAAB/ABBA blocks. All complete-file hashes/geometry manifests match across candidates and match the original streamasset; all payload validations pass. Cook timing excludes the separate validation process. Source/OS caches were retained, not flushed cold. |
| Real cook result | Recorded without a universal speedup claim: at workers8 sampled peak thread count drops19→12. Workers4/8 are approximately2.2% slower in aggregate process wall time; results depend on worker count/block. The ownership reduction is demonstrated; unconditional throughput improvement is not. |
| CPU/GPU resources | Established within stated ownership/windows: OS resource records and high-water counters, simultaneous tracked RHI allocation/device-local peaks, per-domain peaks and streaming/texture samples retained. Independent domain peaks are not summed. RHI peaks exclude external SDK/driver allocations; NVML is whole-device sampled/cached data, not exact renderer VRAM. |
| Core scene validation | Established: corrected final four-process ABBA, each120 captured frames, full-DLSS scene, core validation active, zeroVUID. Failed SDK-sync startup and the earlier harness-rejected600-frame pilot remain separate records. |
| Bounded temporal/residency soak | Established: four fresh-process ABBA runs ×6000 captured frames passed the [soak gates](SoakAudit/Audit.json). [Audit definitions/scripts](SoakAudit/) retain identity, memory-window and image-comparison boundaries. This is a static-view residency/correctness soak, not ordinary FPS evidence or camera-return/edit validation. |
| Final regressions | Established: Task33/33 and19 range tests ×100; Scene160 passed/9 explicit asset/USD skips; kernels6/6; core RHI26/26 and independent sync RHI18/18. Skips and the Streamline sync limitation remain explicit. |
| Ordinary frame matrix | **User-budget limited:** original56-process plan was stopped. Prewarm runs0..7 are excluded. Only runs8..11 form a complete W1 ABBA block, two processes per arm ×600 frames. Completed one-sided W2 run12 and interrupted run13 are excluded. No W2/4/8 paired result or three-block complete matrix is claimed. |
| Reduced frame statistics | Established as a descriptive single block only: `frame-perf-reduced-super-sponza` retains the complete W1 block and its selection rationale; original records retain `UserBudgetStop`. Run11 has large CPU/GPU tails. Keep those observations; do not infer stable FPS improvement or characterize rare tails from two processes per arm. No extra process was launched to form the reduced population. |
| Reporting/adoption decision | [Report](Report.md), raw records, audits and plots preserve these populations. Retain stdexec and C++ Automatic, and the shared-worker ownership design with its measured throughput limits. Neither inactive-frame adapter comparisons nor optional-kernel microbenchmarks establish a production FPS/SIMD gain. |

## Conditional work that is not mandatory replacement work

M4 currently has no trigger. The raw ordinary/directed comparison has no
consistent winner, and TaskGraph's tiny-node gap is not isolated to the
upstream deque. Retaining stdexec therefore satisfies the decision gate.
Chase-Lev implementation, new sender/backend adaptation, buffer-reclamation
proofs and replacement-specific mixed-priority/long-running tests become
required only if the production evidence changes that gate and a mature
backend/configuration alternative cannot address the measured bottleneck.

AVX-512 is optional. ISPC production adoption is also conditional on a complete
owning-path benefit and the consumer's correctness contract; the optional
kernel target is already implemented. The current SoA finite/tolerance
contract is not a blanket replacement for scene bounds used by culling.
See [SIMD adoption audit](SIMDAudit.md).

## Limits that must not be presented as passing tests

- ThreadSanitizer was unavailable on this Windows toolchain. Repeated stress
  and exact-coverage tests passed, but are not a data-race-detector pass.
- Physical unsupported-ISA hardware and AVX-512 were not tested. Runtime gates
  and fallback builds provide narrower evidence.
- Hardware SIMD utilization, true upstream steal counters and measured
  worker sleep/wakeup transitions are unavailable.
- Tracked RHI allocation peaks exclude external SDK/driver allocations.
  Sampled NVML device usage and heap budgets do not prove an exact total
  process VRAM lifetime peak.
- Full-DLSS scene synchronization validation is unavailable through the current
  Streamline interposer's instance-extension handling. Core scene validation
  and18 independent non-Streamline synchronization tests passed; they do not
  turn that unavailable combination into a passing run.
- The user ended the ordinary performance experiment after one complete W1
  block. The remaining worker/block matrix was not executed and is not claimed
  complete. No more testing or benchmarking follows from this document.
