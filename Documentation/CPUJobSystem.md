# CPU Job System: baseline audit and range scheduling

This document records the source audit for [issue #73](https://github.com/af8a2a/metallic/issues/73)
against repository commit `8e0d94587`. It separates existing functionality,
the range-scheduling design, and measured evidence. Source inspection is not a
performance measurement. Benchmark results and validation must identify the
revision and executable actually used.

## Existing worker runtime

`MetallicRuntimeTask` is already a separate library, defined in
[`Source/CMakeLists.txt`](../Source/CMakeLists.txt). Its backend is
`exec::static_thread_pool`, with operation lifetime managed by
`exec::async_scope`; no new worker pool is needed for CPU ranges.

The locked Git submodules are:

| Dependency | Revision | Relevant implementation |
| --- | --- | --- |
| NVIDIA stdexec | `6d7ad689f4d4831c5136e4abe1c601f9a3b64e43` | `include/exec/static_thread_pool.hpp`, `include/exec/detail/bwos_lifo_queue.hpp` |
| oneTBB | `a00cc3b8b5fb4d8115e9de56bf713157073ed68c` | `include/oneapi/tbb/version.h`: 2021.9.0, interface 12090 |

The inspected `E:/metallic/External/` checkouts have these exact HEADs. The
independent worktree initially has empty submodule directories; that absence is
not evidence that the renderer has no stdexec or oneTBB dependency.

### stdexec queues, stealing and wakeup

At the locked revision, `_static_thread_pool` has per-worker BWOS LIFO queues.
The owner alone pushes/pops the back; other workers steal from the front.
Remote submissions use per-submitting-thread intrusive atomic queues with one
destination queue per worker. `thread_state::try_pop()` takes local work first,
then drains the destination's remote queues through `try_remote()`.

An idle worker samples randomized near-NUMA victims and then all victims, up to
`thread_count + 1` attempts in each group. A steal moves one task, not half of a
numeric range. `pop()` yields before entering its mutex/condition-variable
sleep path. `notify()` exchanges the worker state to `notified`; if the worker
was sleeping it synchronizes through the same mutex before notifying. Active
worker/thief counters preserve an active thief while a potential victim is
awake. `request_stop()` sets the stop flag under the worker mutex and wakes it.

These mechanisms already provide work stealing and coordinated wakeup. They do
not establish FIFO fairness, bounded task latency, OS affinity, or a stable
logical-job-index to native-thread mapping. Local LIFO preference and remote
drain timing are workload-dependent; fairness and wake latency require
measurement. No backend steal-attempt or sleep-duration counters are exported
to Metallic by this implementation.

`get_scheduler_on_thread(i)` chooses the remote destination modulo pool size.
The resulting task can subsequently be stolen: this is a queue-selection hint,
not an affinity contract. Its enqueue overload always uses a remote queue and
notifies the target, including submissions made from another pool worker.

`get_scheduler()` instead recognizes an in-pool submitting thread and pushes
locally when its NUMA constraints permit. An external submitter chooses a
random permitted destination and notifies it. Thus ordinary versus directed
scheduling changes locality, remote atomics and wake traffic, as well as task
distribution. An A/B must hold the DAG and worker count fixed. It must not
attribute an observed difference solely to wakeup.

### TaskGraph lifecycle and observation

The owning code is
[`TaskSystem.cpp`](../Source/Runtime/Task/TaskSystem.cpp), particularly
`TaskSystemBackend::schedule`, `detail_TaskExecutionState::scheduleNode`,
`runNode`, `finishNode`, `finishGraph`, and `TaskSystem::submit`.

The original backend round-robins ready nodes through directed schedulers.
`submit()` validates graph names, edges and cycles before constructing runtime
nodes. Atomic dependency counts release successors; a failed prerequisite
skips its dependents, while independent branches retain the existing execution
semantics. Callback exceptions become failed task results. Cooperative stop
tokens are owned by each graph; stop requests cancel unstarted work and running
callbacks may observe `TaskContext::stopRequested()`.

Execution state is retained in the backend's active-execution map and by
scheduled operations. Shutdown closes admission, requests stop for active
graphs, waits for graph completion, drains `async_scope`, and then destroys the
pool. A range adapter must use this ownership path rather than retaining an
unowned pool pointer or running untracked callbacks during teardown.

`TaskGraphRun::wait()` explicitly returns `WaitFromTask` from a task callback.
`shutdownTaskSystem()` also rejects callback re-entry. Synchronous range work
called from a callback must execute inline; scheduling a child graph and
blocking on it would violate this contract and can exhaust every worker slot.
An asynchronous child graph is valid only if its caller does not wait while
holding such a slot and its captured data outlives the child graph.

The existing snapshots expose graph/execution IDs, stable node handles,
scheduled/start/finish timestamps, native thread IDs, task metadata and errors.
`ITaskEventSink` reports submitted/state-changed/completed events. The render
debug task sink in
[`RenderDebug.cpp`](../Source/Runtime/Render/Debug/RenderDebug.cpp) already
associates these IDs with debug events. These facilities can identify queue
delay (`start - scheduled`) and callback wall occupancy (`finish - start`),
but callback wall occupancy includes preemption and blocking; it is not CPU
execution time. Sink callbacks are synchronous and may perturb benchmarks.

The observed scheduling path also includes graph validation, per-node state
allocation, a graph snapshot mutex, timestamp reads and event-sink snapshots
under a sink mutex, even with no subscriber. A TaskGraph-per-element baseline
therefore measures this entire service rather than a bare deque operation.
Keeping runner count bounded amortizes these existing costs over ranges;
profiling must establish their magnitude before changing observation policy.

The existing
[`TaskGraphTests.cpp`](../tests/task/TaskGraphTests.cpp) covers empty/invalid
graphs, chains and diamond joins, move-only captures, callback exceptions,
cooperative and pre-run cancellation, event metadata, shutdown/restart/drain,
and worker-wait rejection. These are regressions to preserve, not features to
reimplement in a range scheduler.

## Meshlet production contract

The original
[`MeshletBuildParallel.cpp`](../Source/Runtime/Scene/MeshletBuildParallel.cpp)
owns `workerCount - 1` persistent native threads, with the submitting thread
running slot zero. A submission mutex serializes callers of each shared pool.
An epoch/condition variable starts all workers; `nextTask.fetch_add(1)` claims
individual indices. The first exception stops further claims, the caller joins
every participating worker, and then rethrows it. Same-pool nested calls run
inline with the current scratch slot. `shared(workerCount)` retains separate
pools for every requested worker limit, so these pools can coexist with the
TaskSystem pool and each other.

`buildClusterLodParallel()` in
[`scene.cpp`](../Source/Runtime/Scene/scene.cpp) allocates one
`MeshletLODWorkerScratch` per dense slot. Cluster-bound callbacks write distinct
clusters; group callbacks read stable inputs, simplify using their exclusive
scratch slot, and produce per-group results. Publication is explicitly ordered
by `nextOutput` and `outputReady` inside the group callback. This last point
matters: the current meshlet callback may block waiting for an earlier index;
it is not an arbitrary nonblocking SIMD kernel.

The migration therefore uses monotonic Dynamic claims for meshlet work and
retains the ordered publication algorithm. Giving all available workers later
intervals while an earlier interval's runner is still queued can deadlock this
particular callback. Half-stealing is suitable for independent range callbacks;
changing the meshlet default to it requires first moving ordered publication
out of worker callbacks and measuring the resulting memory and runtime costs.

Scratch-slot exclusivity during one joined submission is required. Native
thread affinity across submissions is not needed by the production caller.
The original test's stable slot-to-thread expectation is stronger than stdexec
provides and must not be preserved as an undocumented promise. Tests should
prove dense, exclusive scratch use, nested slot preservation, join/exception
lifetime and byte-equivalent meshlet outputs instead.

The legacy implementation is useful as a benchmark-only A/B baseline. Keeping
its independent native pool active in the production adapter would defeat the
shared-runtime objective.

## RenderGraph features to reuse

Sections 14–18 of
[`ParallelRecordingAndSubmission.md`](ParallelRecordingAndSubmission.md)
describe functionality already present in the audited revision:

| Existing feature | Boundary to preserve |
| --- | --- |
| `CPURecordingPolicy::ParallelJoined` | Only audited passes record concurrently, with exclusive recording contexts. It is independent of async GPU queue support. |
| `prepareJoined()` | Immutable captures and independent CPU outputs; join all work before coordinator publication, including on failure. |
| `RecordedBatch::seal()` | Recorded command ownership and frame generation, without implying queue acceptance or GPU completion. |
| `SubmissionReceipt` | Only successful native acceptance publishes a completion point. |
| Pipelined submission | Coordinator accepts an ordered ready prefix while independent batches record. |
| Joined submission combining | Plan/seal compatible same-queue batches after all workers join; preserve cross-queue producers and wait boundaries. |

`RenderGraphExecutor::execute` acquires the shared TaskSystem, bounds worker
participation, and inlines execution when nested, debugging or single-worker
conditions require it. Recording batches are TaskGraph nodes. The coordinator
owns queue submission and cancellation order, and always joins started work
before releasing inputs. `prepareJoined()` follows the same joined ownership
model. The new CPU range layer must not introduce another recording pool,
change pass policy based on `supportsAsyncQueue()`, or move GPUScene, history,
streamer, registry or SDK state publication into range callbacks.

[`ParallelSchedulingMeasurements20260926.md`](ParallelSchedulingMeasurements20260926.md)
is historical evidence for a different measured machine/workload. MiniZorah's
native recording was light and its frame wait dominated the recorded CPU
execute interval; increasing recording workers did not establish a whole-frame
gain. This motivates testing geometry preparation first. Those numbers are
not a baseline for today's CPU range implementation and must not be presented
as its speedup.

Existing RenderGraph scheduling diagnostics and
[`AnalyzeSchedulingDiagnostics.py`](../Tools/AnalyzeSchedulingDiagnostics.py)
already separate coordinator prepare, worker recording, waits, seal,
tracker/native submit and ready delay. Diagnostics are opt-in; worker-local
metrics merge only after joining. Tracy CPU scopes are configured through
[`TracyProfiler.h`](../Source/Runtime/Render/Profiling/TracyProfiler.h).
[`CPUThreadProfile.h`](../Source/Runtime/Render/Profiling/CPUThreadProfile.h)
offers Windows thread-cycle and user/kernel-time endpoints; its documented
accounting granularity and same-thread requirements remain relevant.

## Range-layer design

The worker runtime has one owner: TaskSystem. A range operation creates bounded
runner nodes using the existing TaskGraph completion, exception, cancellation
and snapshot machinery. The scheduler assigns contiguous subranges to those
runners; a callback's dense runner slot owns its scratch storage for the
duration of that invocation. A slot is not a native CPU/thread identity.
Different concurrent range operations own different slot arrays and state.
`WorkerContext::executionId` and `node` identify asynchronous runner tasks.
Nested inline calls borrow their parent task identity; external inline calls
have no graph identity. Optional diagnostics must be exclusive to one in-flight
operation and may be read or reused after joining.

Static splitting partitions a range evenly. Dynamic splitting claims monotonic
grain-sized batches from one atomic counter. Adaptive half-stealing keeps one
remaining interval per runner: the owner claims from the front, a thief removes
roughly half the remaining interval from the back. Both modify the same atomic
packed endpoint state with CAS, so the owner and thieves cannot receive
overlapping elements. The 32-bit endpoint domain must be explicit; endpoint
arithmetic and grain rounding use a wide intermediate and support an exclusive
end of `UINT32_MAX` without wrapping a claim. Empty and short tails are valid.
Padding state avoids sharing a cache line between independent owners.

One runner claims work without scheduling one TaskGraph node per element or
per SIMD lane. SIMD kernels receive only their assigned `[begin,end)` and
never create threads or use ISPC `launch/sync`. Grain size and optional width
alignment are caller policy, not a hardwired ISA requirement.

A synchronous outer call acquires the shared runtime, submits runner work and
joins every started callback before returning or propagating an error. A
callback-side synchronous call executes inline, preserving any outer scratch
slot owned by a compatibility adapter. An asynchronous form adds range runners
and a completion dependency node to a caller-owned graph. Consumers depend on
that completion node; they do not wait inside a worker. Captures and range
storage stay alive through graph completion. Cancellation is cooperative:
already claimed callbacks may finish, but no unclaimed work is promised to run.
The implementation exposes these forms as `parallelFor()` and
`addParallelFor()` in
[`ParallelFor.h`](../Source/Runtime/Task/ParallelFor.h). Async dependencies enter
through the returned `entry` node and leave through `completion`. External stop
tokens fail the completion and skip dependent publication; graph stop keeps
the existing `Cancelled` status. Nested inline calls inherit the parent
callback's graph stop token. Callbacks must not wait for another range;
Static and HalfSteal do not promise monotonic index execution.

`addParallelFor()` fixes its runner count when the graph is assembled using
the currently initialized runtime. Without one, it creates one runner; a later
runtime initialization does not repartition that already assembled operation.

When the global TaskSystem is unavailable, standalone geometry cooking needs a
documented serial path or explicit caller initialization. It must not silently
construct a second persistent pool. Worker limits constrain participation in
the shared runtime; they do not resize the global pool or reserve physical
cores. The existing default TaskSystem policy leaves one hardware thread free;
applications may choose a lower limit for editor/IO responsiveness.

## Measurement interpretation and decision gate

Use Release builds, fixed worker counts 1/2/4/8 and deterministic seeded
inputs. Separate first-use/pool construction, first submission and warmed
samples. Interleave candidate order across repeated rounds; preserve raw
samples and exact CLI/environment metadata. CPU-only runs do not involve
Vulkan validation. Rendering regression runs enable validation separately.

The minimum matrix covers uniform compute, randomized cost, a localized long
tail, bandwidth-bound contiguous access and a locality-sensitive access
pattern. DAG tests compare ordinary/directed stdexec scheduling on chain and
fan-out/fan-in shapes, uniform/heavy-tail callbacks, nested requests,
cancellation and shutdown. Compare the same callback and worker limit against
the original meshlet atomic-per-index pool, the unified strategies, and oneTBB
`parallel_for` when that library can be linked without changing the renderer's
optional-SDK profile.

| Metric | Interpretation / availability |
| --- | --- |
| Wall time, P50/P95/P99, throughput | Measurable per submission; report sample count and quantile convention. |
| Task queue delay | Snapshot `startTime - scheduledTime`; includes OS scheduling and instrumentation. |
| Per-runner callback wall occupancy | Available with opt-in range diagnostics; includes preemption, not CPU utilization. |
| Unoccupied interval | Wall window minus callback occupancy is only a derived occupancy bound; do not label it OS idle time. |
| Range claim/CAS/steal counts | Exposed by range diagnostics if enabled; distinct from stdexec task-queue steals. |
| stdexec steal attempts/success | Unavailable without upstream instrumentation. |
| Wakeup delay | Unavailable from public pool API; queue delay is not an isolated wakeup metric. |
| CPU utilization, atomic instruction time | Require OS sampling/hardware counters; neither follows from callback wall timestamps. |
| Critical path / whole-frame time | Require a separate fixed scene/rendering experiment; microbenchmark scaling is insufficient. |

Do not add a general-purpose self-owned deque based on source inspection or a
single noisy result. M4 requires a repeatable bottleneck, configuration and
oneTBB alternatives, and real workload evidence. Until those exist, retain
stdexec and report performance limitations explicitly. This design does not
claim a TaskSystem speedup, SIMD speedup or FPS improvement.

## Generated meshlet workload A/B

[MeshletSceneBench.cpp](../Tools/CPUJobs/MeshletSceneBench.cpp) calls the
production meshlet builder on fresh 64/128-cell grids. Plain, attributed and
protected-seam inputs exercise bounds, simplification, reclustering and ordered
publication. Each build owns its output. The timer covers the complete
buildStreamMeshletsForPrimitive call, excluding source generation, input
copying, output hashing, runtime initialization and output destruction.

The legacy executable links
[LegacyMeshletSceneAdapter.cpp](../Tools/CPUJobs/LegacyMeshletSceneAdapter.cpp)
directly instead of extracting the new adapter from the scene static library.
This benchmark-only copy of the original 8e0d94587 implementation retains its
old persistent pools and per-index atomic claim. Both executables initialize
the same-size TaskSystem explicitly; the legacy adapter additionally owns its
old worker pools. They run in separate processes.

[MeasureMeshletScene.py](../Tools/CPUJobs/MeasureMeshletScene.py) alternates
legacy/shared process order and rotates configurations and worker limits. It
retains serial first-build and warmup records separately. Every result hashes
all logical RenderPrimitive fields, including every meshlet/LOD field, using
exact float bits and canonical integer bytes; padding and pointers are
excluded. Serial, parallel, repeated and cross-executable hashes must agree
before the script writes a successful summary.

Run the script with --new and --legacy pointing to the two configured target
executables, --rounds 6, and --output naming a fresh result directory. Its
output contains executable/source SHA256 provenance, invocation order,
original process JSONL and logs, all raw samples, and a summary. Nearest-rank
P95/P99 with six observations are usually the maximum observation; these
samples cannot characterize rare latency tails. This is a generated workload
using the real meshlet algorithm, not a production-scene frame-time or FPS
benchmark.

## Correlating range diagnostics with graph execution

Opt-in range diagnostics retain an operation ID, an owned operation name and
backend label, the input interval, grain, strategy and creation time. The
backend label is generic caller metadata; callers with runtime kernel dispatch
should report the backend that actually ran. Labels are copied at the API
boundary, so temporary label storage need not survive graph execution.

Each started lane records its dense slot, execution/node identity, native
thread hash, and steady-clock start/finish endpoints, including exception
paths. Join these records to TaskGraph snapshots by execution ID and node
handle. Both use the same clock and thread-hash convention. Task queue delay
comes from the node's start minus scheduled timestamp, not range creation
time. Creation can precede graph submission by an arbitrary caller interval.
Lanes cancelled before starting retain empty identity and timestamp fields.

Nested inline lanes are explicitly marked and borrow the containing task's
identity; they create neither a new node nor a separate queue delay. Their
interval lies inside that task's interval and must not be added to the parent
when computing a critical path. Callback-active nanoseconds count completed
batches; lane start/finish also encloses claim/steal overhead and exceptional
callbacks. Neither quantity measures OS thread running or sleep time.

When diagnostics is null, operation-ID allocation, label copying and additional
range timestamp reads are disabled. The existing TaskGraph snapshot timestamps
and callback-scope identity remain available. The task layer has no renderer,
Tracy, Vulkan or ISPC dependency; presentation/export belongs to its observer.

## Production evidence and retained decisions

The [SuperSponza production report](Measurements/CPUJobsProduction20261107/Report.md)
and [acceptance checklist](Measurements/CPUJobsProduction20261107/AcceptanceChecklist.md)
extend the original synthetic/generated evidence with real fresh-output cooking,
full-DLSS core validation, independent RHI synchronization validation and a
bounded 6000-frame residency soak. The [source/window audit](Measurements/CPUJobsProduction20261107/Notes.md)
distinguishes serial pass recording, joined VBuffer preparation and prewarm
stream loading. Fixed-camera frames reuse a prebuilt streamasset; they do not
execute the changed LOD builder or the experimental SIMD kernel.

The real cooker preserves complete stream hashes and geometry manifests across
48 cooks and 48 separate validations, including equality with the original
asset. At workers 8 the sampled thread peak drops 19→12, while workers 4/8 are
approximately 2.2% slower in aggregate process wall time. Shared ownership is
the demonstrated benefit; there is no unconditional throughput gain. Retain
stdexec and C++ `Automatic`; no evidence triggers M4 or production ISPC adoption.

The user stopped further testing to reduce the budget. Ordinary frame evidence
is limited to one complete W1 ABBA block, two processes per arm ×600 frames,
after excluding eight prewarm processes. One-sided/interrupted W2 runs remain
excluded and recorded. A large CPU/GPU tail in one legacy process prevents a
stable FPS-improvement claim; the original three-block workers1/2/4/8 matrix is
not complete. The report preserves this limitation rather than extrapolating
from the cook, soak or microbenchmarks.
