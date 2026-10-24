# Persistent GPU BLAS allocation

Dynamic RT representations retain their BLAS across unchanged frames. A changed
representation is built into temporary implicit storage, allocated using its
GPU-reported encoded size, moved into persistent storage, and published before
TLAS construction. Clean representations can move into free holes without being
rebuilt. The GPU backend does not perform the legacy whole-arena Repack.

This storage policy operates on the existing representation owners, including
Sharing, Merged and discrete-cache representations. It does not change the
authored cut or select a different LOD to make a temporary allocation fit. See
[RT representations](RTRepresentations.md) for selection and residency rules,
and [GPU CLAS allocation](GPUCLASAllocation.md) for the referenced CLAS lifetime.

## Budgets

| Setting | Meaning |
| --- | --- |
| `METALLIC_BLAS_STORAGE_BYTES` | Explicit persistent capacity, nonzero, device-aligned, and no larger than `maxBlasBytes` or `UINT32_MAX` |
| `METALLIC_BLAS_RELOCATE_BYTES` | Maximum clean-relocation bytes per allocator submission; default 4 MiB; zero disables clean relocation |
| `maxBlasBytes` | Persistent arena hard cap; default 512 MiB |
| `maxBlasStagingBytes` | Separate implicit build-output hard cap; default 512 MiB |

The two size properties are part of `MeshletStreamRuntimeDesc` and scene streaming
configuration. The persistent arena defaults to the driver upper bound selected
by the existing reference/build admission limits, which is no larger than
`maxBlasBytes`. A smaller `METALLIC_BLAS_STORAGE_BYTES` value changes persistent
capacity without reducing the temporary build-output requirement. This makes it
useful for a controlled capacity-pressure comparison.

GPU staging is sized for the queried worst case of the admitted build batch.
If that requirement exceeds `maxBlasStagingBytes`, initialization fails with an
error. It does not silently lower RT quality, truncate the cut or fall back to a
different allocator. GPU actual-size allocation is the sole streamed RT implementation.

Both arenas are ordinary physically allocated buffers, not sparse reservations.
The staging allocation remains present on stable frames, even when the indirect
build count is zero. Persistent storage can therefore fit actual objects tightly
while total device memory increases relative to the legacy backend. Report
persistent capacity, actual live allocations, dirty-build staging, relocation
staging, build scratch, Move scratch and allocator metadata separately. Clean
relocation has a fixed temporary buffer equal to its byte budget (4 MiB by
default); setting the budget to zero omits this buffer. Its physical capacity is
reported as `blasRelocationStagingBytes`. Neither size cap is a total renderer
VRAM limit; other resources and the RHI device allocation policy still apply.

Malformed capacity overrides and out-of-range relocation budgets are errors.
The retired allocator A/B selector is no longer read.

## Ordered GPU work

The implementation is in
[`MeshletStreamRuntime.cpp`](../Source/Runtime/Render/Streamer/MeshletStreamRuntime.cpp),
[`StreamBLASAllocator.slang`](../Shaders/Features/GPUDriven/StreamBLASAllocator.slang)
and its [CPU ABI](../Source/Runtime/Render/Streamer/MeshletStreamBLASAllocator.h).
The parameter ABI is 184 bytes, including relocation-staging, owner-priority
and admission-cache addresses. Retired selector words remain padding.
Allocator configuration is immutable for one runtime. GPU commands own its
changing allocation state; CPU feedback is observational.
Free bytes are updated by the serialized allocation/release operations. Finding
the largest hole only scans the highest nonempty size bin; publication sums live
allocations independently so tests can verify live plus free equals capacity.

1. **Compare and prepare.** Existing cut and CLAS generation checks identify
   dirty owners. Parallel owner passes gather obsolete allocation releases and
   compact dirty build records. A serial pass applies those releases after prior
   arena consumers have completed. Valid unchanged owners remain resident.
   Reference and build-count admission still applies.
2. **Implicit build.** Only the dirty list is passed to the native BLAS build.
   The driver writes packed staging addresses and actual encoded sizes into GPU
   buffers. No CPU size or address readback gates the next operation.
3. **Allocate.** Encoded size is rounded to the device's bottom-level storage
   alignment. A GPU free-gap allocator searches size bins and splits a fitting
   gap. Returned adjacent gaps coalesce. The first 64 alignment units use exact
   size bins; larger gaps use logarithmic bins (96 bins in total). These bins
   index free lists and do not round object allocations to cluster-count buckets.
4. **Move.** GPU-produced address arrays and counts copy successful dirty builds
   from build staging to persistent slots. Clean relocation first copies selected
   objects from the persistent arena into separate relocation staging. After that
   copy completes, the allocator releases their old arena slots, merges adjacent
   holes and allocates return destinations. A second Move copies the staged clean
   objects back into the persistent arena. All copies use the RHI `BottomLevel`
   object type and Vulkan `noMoveOverlap = true`.
5. **Publish.** After the final Move completes, a parallel owner pass publishes
   offsets and capacities and updates cut metadata. Representation-to-instance
   resolution and TLAS generation consume this published state. A rejected dirty
   owner follows the existing complete fallback policy; it never exposes a
   partially allocated BLAS.

Owner preparation and publication run in parallel. Serial
work is restricted to applying dirty releases, actual-size allocation, planning
clean moves and modifying their free-gap topology. GPU allocation is not a
parallel per-cluster allocator, and its cost is not constant in the number of
dirty owners or examined holes. There is no CPU readback between Build, Allocate,
Move and Publish.

## Incremental relocation and convergence

Clean relocation starts only after an actual-size request fails because free
space is fragmented. A persistent requested-size target keeps the bounded scan
advancing even when negative caching suppresses that request on the next frame.
It stops when a sufficiently large gap exists, or all currently free bytes form
one gap. It does not proactively move clean objects merely because holes exist.

Clean relocation targets lower-address free space. It scans at most
`min(ownerCount, 256)` owners per submission and advances a persistent cursor.
The sum of selected aligned allocation sizes cannot exceed the relocation-byte
budget. Dirty-build copies have a separate list; they are bounded by admitted
builds and persistent capacity, not by the clean-relocation setting.

Each clean source remains allocated until its first copy into relocation staging
completes. The original slot can then be released safely, allowing a smaller
adjacent hole to merge with it before return allocation. This permits progress
when a lower hole alone is too small to receive the object, which a direct
non-overlapping same-arena copy cannot handle. The return Move reads exclusively
from relocation staging, so the new arena destination may overlap the now-retired
original range without violating native Move's non-overlap requirement.

Production relocation therefore does not use RHI `allowAliasedStorage`; that
contract remains available and covered by the focused RHI test. Staging, scratch
and address/count buffers are retained through submission. The two clean copies
move at most twice the logical relocation-byte budget. `blasRelocatedBytes`
counts each selected object once, rather than counting both physical copies.

Incremental relocation may need several frames, and a large object can remain
rejected while total free bytes exceed its requirement but the largest hole
does not. The scan and byte limits bound attempted clean relocation; they do not
guarantee a fixed number of frames until every fragmentation pattern resolves.
An individual BLAS larger than the relocation byte budget cannot be moved.

A failed actual-size allocation records a negative cache entry for the examined
cut and allocator free-space epoch. An unchanged rejected cut is not rebuilt
when the epoch is unchanged or its recorded encoded size still exceeds the
current largest free gap. Returning/coalescing space, including completion of
a clean move, permits retry only once a large enough contiguous gap exists.
A changed cut also permits retry. This preserves stable zero-build behavior
under persistent capacity pressure while allowing recovery when space changes.
Clean moves can still occur during convergence without increasing BLAS builds.

## Queue ordering, cancellation and old versions

The runtime relies on the submission owner's ordering contract, rather than
treating a number of elapsed CPU frames as evidence that an old TLAS is unused.
In the queued RenderGraph path:

- `RenderGraphExecutor::execute(RenderGraphSubmitDesc)` adds the prior frame's
  aggregate completion to the first submission on every participating queue.
- `GPUCompletionPoint::appendWaits` uses `AllCommands`. The submission tracker
  signals `AllCommands` and aggregates the latest accepted signal per queue.
  The next frame's arena writes therefore follow all prior graph ray and TLAS
  consumers, even when two CPU recording slots are active.
- Async TLAS construction depends on the traversal/BLAS producer. Its compute
  branch joins the graphics branch before dependent ray consumers. A pass's
  completion is its join, not the earlier fork producer.
- Registered external output-reader completions are retained and applied to
  every submitted batch before shared resources are reused.

The relevant contracts are implemented in
[`RenderGraphExecutor.cpp`](../Source/Runtime/Render/RenderGraph/RenderGraphExecutor.cpp)
and [`RenderFrameContext.cpp`](../Source/Runtime/Render/Core/RenderFrameContext.cpp).
Raw externally submitted command buffers and direct runtime callers must supply
equivalent ordering for all consumers; the allocator does not discover arbitrary
external TLAS references. Buffers shared with async compute must declare both
graphics and compute queue access.

Initialization becomes committed only when its submission transaction is
accepted. Cancelling an unsubmitted recording executes no allocator mutations,
and a retry initializes or continues the last submitted state. If submission
accepts a prefix before a later failure, `RenderFrameContext::cancel` preserves
that prefix's completion and resources. The executor carries that completion
forward and invalidates the recorded graph state before retry. Accepted GPU work
is not rolled back by CPU cancellation.

Within one allocator submission, dirty objects replace cuts only after Move and
publication. Clean old locations remain allocated through the copy to relocation
staging, and publication waits for the return copy; prior frame consumers have
already completed under the contract above. CLAS generation
checks and the existing CLAS residency retirement rules remain necessary: moving
a BLAS does not extend or replace the lifetime of the CLAS objects it references.

## Validation and measurement boundaries

Focused regressions are registered in `MetallicRHITests`:

- [`stream_blas_gpu_allocator`](../tests/rhi/StreamBLASAllocatorTests.cpp)
  executes the allocator shader with synthetic size/address inputs. It exercises
  bucket-boundary scenarios, stable and grow/shrink sequences, cancellation,
  negative-cache rejection/recovery, disjoint pending move ranges, accounting,
  byte-bounded fragmentation recovery, 32 untouched frames with unpressured holes,
  pressure surviving a bounded scan that initially finds no movable owner, and
  suppressed retries across multiple relocation epochs until a fitting gap exists.
  It does not execute native BLAS builds
  or native Moves and cannot establish driver performance or rendered quality.
- [`cluster_blas_indirect_move`](../tests/rhi/ClusterBLASMoveTests.cpp) builds a
  real implicit BLAS, performs indirect Move and traces analytical rays in one
  submission before reading source addresses back. It also relocates within one
  arena. The scratch-assisted case executes `persistent -> relocation staging ->
  lower persistent` in one submission: the return range overlaps its retired
  source, while each native Move uses separate source/destination buffers without
  aliased-storage permission. TLAS analytical rays check the returned BLAS and
  surviving independent copies over 32 frames, including hit identity, distance,
  barycentrics and masks. This validates the native Move contract, not full-scene
  allocator convergence or shadow/reflection quality.
- [`stream_rt_representation_lifecycle`](../tests/rhi/StreamRTRepresentationTests.cpp)
  and [`stream_rt_representation_parallel_admission`](../tests/rhi/StreamRTRepresentationAdmissionTests.cpp)
  exercise the integration and existing representation behavior. Their results,
  validation settings and any skips must be recorded for the actual binary tested.

A performance acceptance run must use the same scene, camera pressure trajectory,
render settings and cache/warmup conditions for the implementations under
comparison. Retired legacy A/B backends are not available in production. Record live encoded bytes, allocated bytes, free bytes, largest hole,
build counts, Repack events, clean Move counts/logical bytes, negative-cache
suppressions, allocation failures and physical arena/staging/scratch allocations.
Distinguish logical relocation bytes from the two native copies. Report pass timings
and frame P95/P99 separately, and include process/device peak-memory measurement
scope and sampling limitations. Compare stationary convergence after motion as
well as the moving capacity-pressure interval.

Inspect shadow and reflection output throughout the trajectory, including
capacity rejection and recovery. Compilation, shader validation, synthetic
allocator tests and one-frame smoke tests are not full-scene temporal or quality
acceptance. This document describes the implementation and required evidence;
it makes no measured performance or peak-memory improvement claim.

`captureQuality` with `captureReflections` in the full-roam harness adds an
untimed OpenPBR path-trace diagnostic. It consumes the VBuffer raster metadata,
published geometry and TLAS through graph inputs, rather than creating another
geometry streamer. It uses fixed transport samples, no history or radiance cache,
and includes material reflections in the full path-traced output. Exported final
images validate that sampled endpoint; they are not a temporal image sequence or
isolated specular radiance. Keep these captures separate from performance runs.
