# CPU jobs implementation and validation

Historical scope note, added 2026-11-07: this page records the initial
pre-production delivery and its validation state. The production acceptance
described as unmeasured below was investigated afterward; see the
[SuperSponza production report](../CPUJobsProduction20261107/Report.md) and
[final scope checklist](../CPUJobsProduction20261107/AcceptanceChecklist.md).
The ordinary frame matrix was curtailed by the user's test budget; it is not
claimed fully completed. The original measurement data/provenance are retained.

Work ran in the isolated `codex/issue-73-cpu-jobs` worktree on November 6–7,
2026. The issue's M0/M1 first-delivery loop is implemented, together with the
independent M2 kernel backend and scoped M3 integration regressions. The full
roadmap's production-scene critical-path, frame-time and resource-peak acceptance
remains unmeasured; this delivery does not claim that all roadmap checkboxes are
complete or that renderer FPS improved.

## Implementation and decisions

- `ParallelFor.h/.cpp` provides Static, Dynamic and AdaptiveHalfSteal ranges,
  joined calls and asynchronous TaskGraph entry/completion nodes. Owner/front
  and thief/back claims share one packed 64-bit CAS state. Wide intermediate
  arithmetic supports empty ranges, short tails and `UINT32_MAX` endpoints.
- TaskSystem remains the only production worker-pool owner. Nested calls run
  inline and inherit the containing callback's stop token. Admission and the
  shutdown snapshot share a lock; graph start and event callbacks run outside
  that lock, and shutdown drains accepted work before destroying the pool.
- Meshlet adapters own no threads. Their dense slots are exclusive within an
  invocation, without OS-thread affinity. Per-invocation range state and the
  builder's local scratch/results replace the old adapter submission mutex.
  Dynamic/grain 1 retains monotonic claims for the builder's ordered publication;
  HalfSteal is not applied to that blocking callback. Offline tools explicitly
  initialize and drain the shared runtime; other callers can use serial fallback.
- `MetallicCPUKernels` isolates scalar, compiler-vectorized and optional ISPC
  SSE4.2/AVX2 kernels from TaskGraph/RHI. Its input, alias and numerical contracts
  are explicit. ISPC owns no threads. `Automatic` retains C++ auto vectorization;
  measured ISPC performance varies with batch size and packing is material.
- M4's replacement gate is not met. Retain stdexec's mature stealing/wakeup
  backend: ordinary versus directed scheduling had no consistent winner, and
  tiny TaskGraph costs cannot be attributed to the upstream deque.

The audit/design is [CPUJobSystem.md](../../CPUJobSystem.md). Evidence separates
[scheduling](Report.md), [complete generated meshlet builds](Meshlet/Report.md)
and [single-thread kernels](Kernels/Report.md). These retain 14,952 scheduling,
288 steady meshlet and 3,456 kernel samples, plus raw data and provenance.

## Builds and checks

Windows 11 build 26300, Ryzen 7 7800X3D, MSVC 19.44.35229 x64, Ninja,
Release `/O2`; optional NVIDIA SDK integrations and Tracy were disabled.
Slang 2026.18.2 and locked dependencies were used. A separate `build-issue73`
tree was configured; existing parent-worktree build configurations were not changed.

```powershell
cmake -S . -B build-issue73 -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TRY_COMPILE_CONFIGURATION=Release `
  -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded `
  '-DCMAKE_POLICY_VERSION_MINIMUM=3.5' `
  -DMETALLIC_BUILD_TESTS=ON -DMETALLIC_ENABLE_OPENUSD=OFF `
  -DMETALLIC_ENABLE_NVIDIA_SDKS=OFF -DMETALLIC_ENABLE_TRACY=OFF
cmake --build build-issue73 --target MetallicTaskTests MetallicSceneTests `
  MetallicMeshletCook MetallicResidentMeshletCook MetallicCPUKernelTests `
  MetallicRHITests MetallicMeshletSceneBench MetallicMeshletSceneBenchLegacy -j 8
```

The local configuration also selected the existing Slang installation with
`SLANG_ROOT` and reused pinned GDeflate/Zstd source downloads through their
`FETCHCONTENT_SOURCE_DIR_*` overrides. `Embedded` avoids the partially installed
local PDB helper; benchmark builds use the same compiler/optimization policy.

| Check | Result |
| --- | --- |
| Task + range tests | 30/30 passed |
| Range concurrency/teardown stress | 16 tests, 100 consecutive iterations passed |
| Scene tests | 160 passed, 9 explicit USD/local-asset skips |
| Meshlet correctness | Deep LOD equality, scratch exclusivity, nested/failure/reuse passed; all A/B geometry hashes identical |
| Kernel standalone configurations | Missing-compiler fallback, explicit OFF, multi-target, SSE4.2-only, AVX2-only each passed 5 tests |
| Task-integrated kernels | 6/6 passed, including producer/range/consumer publication for 1/2/4/8 workers and every strategy/backend |
| Vulkan integration selection | 18/18 passed, no skips, core validation enabled |
| Diff and embedded HTML | Diff checks passed; generated HTML fragments reconstruct the original content and remain below the compiler's literal limit |

Scene filesystem tests initially hit Windows sandbox temporary-directory atomic
replacement error 5. They passed with normal file permissions and temporary
data under the worktree's `.cache` directory. This was not treated as a passing
sandbox run. Windows ThreadSanitizer and physical unsupported-ISA hardware were
unavailable.

## Vulkan regression scope

GPU: NVIDIA GeForce RTX 5070 Ti, driver 621.01. The capabilities report confirms
core validation enabled with an active messenger and native descriptor pointers.
Bindless and independent compute queues were requested. The following selection
passed in 46.3 seconds, including cold on-demand shader/pipeline compilation:

```powershell
build-issue73/tests/MetallicRHITests.exe --rhi-validation --rhi-bindless `
  --rhi-async-compute --gtest_color=no `
  '--gtest_filter=*parallel_recording*:*parallel_preparation*:*pipelined_batch*:*pipelined_graph*:*joined_submission*:*frame_parallel*:*parallel_registry*:*registry_pipelined*:*prepared_dispatch_parallel*:*stream_rt_representation_parallel_admission*:*visibility_preparation_serial_parallel_pixels*:*display_encoding_runtime_parallel_propagation*' `
  --output-dir build-issue73/rhi-issue73
```

Coverage includes recording contexts/slots, immutable dispatch snapshots,
parallel preparation failure/join, GPU progress, frame completion, joined
cross-queue and implicit waits, fork/join, display propagation, serial/parallel
VisibilityBuffer pixels, and streaming RT admission. Intentional failure-injection
cases emit expected application error logs; no VUID/validation error messages
were observed. The generated material bunny output was visually inspected.
Reports and images remain local build output. This correctness run is not a
whole-frame performance A/B or a long-running scene-memory/temporal-stability test.

Three isolated build-compatibility fixes enabled these regressions on the local
MSVC toolchain: complete `DebugResourceBinding` before its default span argument,
complete authored `RenderMaterial` before its default span argument, and split
the embedded report into adjacent raw literals below 16 KiB. These change header
dependencies/generated source spelling, not the recording or submission protocol.
