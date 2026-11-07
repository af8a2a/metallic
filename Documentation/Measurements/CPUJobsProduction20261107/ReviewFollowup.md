# PR 74 review follow-up

This records code-review fixes after the original acceptance captures. No
performance benchmark is rerun. The archived cooking times, frame distributions
and resource peaks describe their recorded source/binary revisions; they are
**not measurements of the revised SceneLoader or single-lane implementation**.

## P1: preserve parallel LOD work through SceneLoader

The reviewed path called the complete meshlet builder from a TaskSystem
callback. Its nested inline fallback prevented deadlock but serialized the
groups of a large single primitive. That fallback remains safe for synchronous
callers; SceneLoader now uses explicit asynchronous stages instead.

- [DeferredMeshletBuild](../../../Source/Runtime/Scene/DeferredMeshletBuild.h)
  owns one primitive's staged operation. Bounds use a flat range graph; each
  CLOD level has independent group-prepare nodes and ordered publish nodes.
- Dependencies enforce `prepare[i] -> publish[i]` and
  `publish[i-1] -> publish[i]`. A scratch slot is reused only after
  `publish[i-slotCount] -> prepare[i]`. Thus preparation can overlap while
  output publication stays deterministic, without a worker waiting for peers.
- A completion-dependent continuation advances the builder and submits the
  next stage. It does not synchronously wait on a child graph in the same pool.
- Within one SceneLoader load, a single worker-derived admission/scratch budget
  is divided among admitted primitives. The sum of their lanes stays within
  that budget, rather than multiplying a full scratch window per primitive.
  Temporary builder buffers are released before the slot admits its next
  primitive. This does not claim a separate global budget across all concurrent
  scene-load requests.
- Stages capture shared ownership of their pipeline/candidate and register
  successors before returning. Cancellation stops outstanding runs and breaks
  the handle's state/run ownership cycle; active callbacks may drain with the
  candidate still alive. A cancelled or failed candidate is not published as a
  successful scene result.

The implementation is in [SceneLoader.cpp](../../../Source/Runtime/Scene/SceneLoader.cpp)
and [scene.cpp](../../../Source/Runtime/Scene/scene.cpp). Focused validation is
recorded below; the old external cooker A/B is not substituted for this real
SceneLoader path.

## P2: effective single-lane synchronous ranges

[parallelFor](../../../Source/Runtime/Task/ParallelFor.cpp) computes the effective
lane count before choosing execution. One batch, a one-worker runtime with
`maxWorkers=0`, and other effective-one-lane calls run inline, avoiding the
entry/range/completion graph. Existing nested inline behavior and the same
RangeState stop, exception and diagnostic paths remain in use. This is a code
and functional-coverage fix; no new microbenchmark speedup is claimed.

## P2: evidence storage and history

[Storage.md](Storage.md) describes the byte-preserving versioned release asset,
per-run compact summaries, hashes, retention and offline restoration. The
measurement-only Git commit is rebuilt without the large raw blobs; original
measurement provenance and measured code commits remain unchanged. No original
Metadata, raw record, plotting snapshot or validated manifest is rewritten.

## Validation status for these revisions

| Check | Result |
| --- | --- |
| Effective-one-lane ParallelFor focused regressions | 21/21 `ParallelForTest.*` passed; covers caller thread, zero graph submissions, stop, exceptions and diagnostics for all strategies |
| SceneLoader and meshlet focused regressions | 17 selected checks passed. The new cold-load cases compare one and two primitives on one and four workers against the synchronous reference, including exact float bits. A rendezvous verifies two different workers enter prepare nodes in the same primitive's LOD stage when its lane budget permits. A cancellation check drains the tasks and verifies neither scene nor cache was published. |
| Performance matrix / cooker throughput / new resource peaks | Not rerun, per the confirmed user budget |

The Windows/MSVC builds used the existing `build-issue73` and
`build-issue73-production` configurations:

```powershell
cmake --build build-issue73 --target MetallicTaskTests
./build-issue73/tests/MetallicTaskTests.exe --gtest_filter=ParallelForTest.*
cmake --build build-issue73-production --target MetallicSceneTests
./build-issue73-production/tests/MetallicSceneTests.exe '--gtest_filter=SceneLoading.*:MeshletBuildParallelTest.*:MeshletReferenceSceneBuild.*:SceneImport.MeshletLODPartition:SceneImport.MeshletPersistence:SceneImport.MeshletStreamAsset-SceneLoading.SuperSponzaImagesSmoke:SceneLoading.ZorahFullOpenReportsMissingCook'
```

The first Scene run passed 16 checks; `ZeroNormalRepairUsesReferencePositionsConsistently`
hit Win32 error 5 while atomically publishing a cache under the sandbox's
temporary directory. Only that check was rerun with `TEMP` and `TMP` set to
a writable worktree directory and it passed. The fixture tests that require
the external SuperSponza and Zorah assets were outside this focused run.

The range policy still declines to steal a short tail when the victim has only
one complete grain plus that tail. This is a conservative small-work policy,
not evidence of a universal scheduling advantage. The existing long-tail
benchmark results retain their declared workloads and do not imply otherwise.
ISPC remains experimental with a finite-input/tolerance contract, not bitwise
equivalence or conservative outward bounds; it is not adopted for production
culling by these fixes.
