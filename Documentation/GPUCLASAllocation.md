# GPU CLAS allocation and publication

The compact streamed CLAS path can construct, allocate, relocate and publish a
new page in one command submission. Exact per-cluster sizes remain on the GPU;
the CPU does not read those sizes or upload the resulting CLAS addresses before
BLAS construction. This removes a submission boundary from this dependency
chain. It does not, by itself, establish a scene-loading or frame-time speedup.

## Backend selection

Compact CLAS automatically uses the sparse GPU allocator on devices exposing
the RHI `sparseBuffers` capability. Otherwise it retains the CPU compact backend
as a compatibility fallback. Disabling `compactClas` keeps the fixed-stride
path. The environment backend A/B selector has been removed. Direct allocator
tests may still construct ordinary or sparse pools to exercise their contracts.

## Ordered GPU work

`MeshletStreamGPUCLASPool` owns the allocator state, page allocations, CLAS
address table, page publication table and arena. The existing fixed CLAS pool
provides bounded staging storage for each outstanding batch.

1. **Build:** construct CLAS in staging storage and write actual cluster sizes
   to a device buffer.
2. **Allocate:** reduce cluster sizes into aligned page requirements. A GPU
   allocator searches size bins of free gaps, splits a fitting gap, and
   reserves the entire page. Retiring allocations can return and coalesce
   adjacent gaps. Allocation failure rejects the complete page.
3. **Move:** generate source/destination address arrays and the object count on
   the GPU. `moveClusterAccelerationStructuresIndirect` consumes these arrays
   without CPU address materialization.
4. **Publish:** after Move, publish the page state and generation. Consumers
   must check this state before using its address-table range. A rejected page
   remains empty; partially moved or partially allocated pages are not exposed.
5. **BLAS:** the terminal fallback pass reads the GPU page and address tables,
   builds references for complete ready primitives, constructs their BLASes,
   and publishes fallback addresses. Dynamic RT representations consume the
   same ordered CLAS publication and generations in subsequent traversal/build
   work. See [RT representations](RTRepresentations.md).

Explicit dependencies separate these stages. Size reduction and Move-address
generation are parallel; one allocator invocation mutates the free-gap topology.
“GPU allocation” therefore does not imply that all allocation operations run
concurrently or that allocator cost is independent of page count.

The CPU still schedules page requests and reserves address-table **indices**.
The CLAS addresses stored at those indices are authoritative only on the GPU.
The GPU backend's `clusterAddress()` deliberately supplies no CPU address mirror;
clients use the GPU table rather than substituting delayed host observations.

## Asynchronous CPU observations

There is still a 16-byte outcome record per page operation: success, allocated
bytes, encoded bytes and operation kind. The CPU reads completed outcomes to
update statistics, retire staging batches, recognize rejected requests and
regulate further admission/growth. These records do not contain a per-cluster
size array or destination addresses, and their collection does not gate the
already recorded GPU publication or BLAS consumers.

CPU `pageHasClas()` and scene startup readiness remain delayed mirrors. The
fallback GPU pass does not use that mirror to decide whether to consume a newly
published page in the same submission. Bounded outstanding batches and memory
pressure can still postpone new work; the one-submission chain is a capability,
not a guarantee that every request is admitted immediately.

## Budgets and sparse storage

Virtual capacity, physically committed bytes, live allocated page bytes and
temporary build/scratch storage are different quantities. An ordinary GPU arena
allocates its full capacity physically. Sparse reservation preserves one device
address while binding physical blocks on demand; only physical commitment is
charged as sparse backing in the device memory budget. Staging, allocator
metadata and scratch allocations also consume memory and must be included when
reporting total peak usage.

Sparse virtual capacity must be a multiple of the device's sparse block
alignment. An unaligned reservation is rejected with `InvalidArgument`; the RHI
does not enlarge its virtual budget silently. A commitment request within a
valid reservation may be rounded upward to physical block alignment. Growth is
bounded by the reservation and constrained by the device allocation policy.
The implementation uses the configured growth quantum, not a universal
hard-coded 128 MiB requirement.

Growth considers delayed live allocation sizes, conservative requirements for
pending builds and observed allocation pressure. On physical allocation failure,
the pool keeps existing backing and can continue within that capacity, delaying
growth retries. Whole-page allocation rejection preserves the existing fallback
behavior; it does not publish a truncated CLAS set.

Sparse commitment is currently **append-only**. Freeing pages makes their gaps
reusable but does not shrink physical commitment or move live CLAS allocations.
There is no global defragmentation. A page can be rejected despite sufficient
total free bytes if no contiguous gap is large enough. The GPU backend does not
currently export complete gap/fragmentation telemetry: a zero
`fragmentedFreeBytes` value is not proof of zero fragmentation. Likewise,
`emptyStorageBytes` only identifies a completely unused committed arena; it is
not a measurement of all free holes.

## Cancellation and old versions

Build/allocation/publication commands use tracked submission transactions. A
cancelled unsubmitted batch never executes its GPU allocation or publication;
its page and staging bookkeeping becomes eligible for retry. Previously
submitted pages remain valid. Initialization is committed on submission, and a
second recording cannot initialize the same arena over unresolved initial work.

CPU `wanted` state is scheduling demand rather than a GPU transaction result.
Cancelling a revive recording preserves the requested demand and marks GPU
publication for retry. Explicit `retirePages()` revokes that demand. Cancellation
does not make an unexecuted publication visible.

Retirement captures the GPU uses that may still reference the old addresses.
Free is recorded only after the queued-frame delay and those uses complete or
are cancelled. Address-table slots stay reserved until the free submission
completes. Cancelling that free leaves it retryable. A revive before free can
reuse the existing allocation; a request after an already recorded free waits
for that version to retire before rebuilding.

Sparse binding has a separate lifetime: an accepted bind survives cancellation
of a later command recording. Its semaphore dependency is retained and reused by
subsequent consumers. Physical backing and the virtual buffer remain alive while
submitted commands reference them. Destroying the last CPU buffer wrapper does
not release in-flight memory.

## Regression tests

Use an already configured RHI test build with CLAS support; build the selected
executable before running it. For the existing Windows Release test tree:

```powershell
cmake --build build-pass-stages-nrd --target MetallicRHITests

# Direct allocator fixture always uses an ordinary arena, including exact-fit
# and fragmented budgets smaller than sparse allocation granularity.
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --gtest_filter='*clas_gpu_allocator_lifecycle*:*sparse_buffer_append_cancel_lifetime*' --rhi-validation --rhi-bindless --output-dir build/clas-direct-regression

# Exercise the production capability-selected backend.
.\build-pass-stages-nrd\tests\MetallicRHITests.exe --gtest_filter='*stream_rt_representation_lifecycle*' --rhi-validation --rhi-bindless --output-dir build/clas-runtime
```

`clas_gpu_allocator_lifecycle` uses real CLAS, BLAS, TLAS and analytic ray queries
in the first successful submission, followed by 32 stable frames. It covers
cancel/retry, retire/revive, an old BLAS reader, exact-fit admission, full-arena
and fragmented whole-page rejection, cancelled free and adjacent-gap coalescing.
Its completed readbacks are assertions, not inputs to initial CLAS publication.
It derives later adversarial test capacities from completed allocation sizes.

`sparse_buffer_append_cancel_lifetime` covers virtual reservation accounting,
unaligned capacity rejection, rounded commitment, growth without address changes,
cross-binding reads, cancellation after bind acceptance and a deterministically
blocked GPU reader retaining physical backing. Unsupported capabilities produce
a skip, which is not validation of that path.

The publication revision accessor is O(1); dispatch parameter construction never
requests a full page-statistics scan. CPU maintenance visits only dirty or retiring
pages. Storage reservations and resident statistics are updated at lifecycle
transitions, including cancellation and completion, rather than rescanning the
resident set for every command or profiling snapshot.

## Performance measurement boundaries

Run the same scene, trajectory, executable, shaders, resolution, light settings
and warmup policy for explicit `cpu`, `gpu` and `sparse` modes. Use independent
runs in a balanced order, retain per-run results, and separate validation/image
captures from timing runs. Test both loading/moving workloads and a stable camera.

The profiler separates `CLAS scratch build`, `CLAS allocate`, `CLAS move` and
`CLAS publish`. Do not add these child intervals to their enclosing `CLAS build`
interval. `clasAllocatorPages` counts allocator page operations, not clusters.
Report stage timings, readiness latency, rejection/build counts and total/device
memory peaks separately; a smaller CLAS allocation is not the same as a smaller
whole-renderer memory peak.

Readiness latency counters describe pool-frame distance from its request to the
submission recording that publishes the page; CPU observation occurs later.
They exclude earlier I/O/decompression and time before the pool receives the
request, and are not GPU elapsed-time measurements. Histogram bins are frames
0 through 31, then **32 or more**; a percentile in the final bin is a lower
bound. Compare sample counts and request/admission conditions alongside the
percentiles, especially under rejection or budget pressure. The GPU backend
preserves the first request frame across admission/replacement waits; the CPU
compact backend starts its page record when the build is admitted. Their
percentiles therefore do not cover identical queuing intervals under pressure.

Scene A/B evidence and final performance conclusions belong in the accompanying
measurement report. Compilation, the analytic fixture or a short smoke run alone
does not establish scene quality, temporal stability or performance gains.
