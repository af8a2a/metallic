# WorkControl isolated production replay contract

Design baseline: `515fbc11c4e3a57860ce6df514de930faf041240`.
Status: contract; implementation and machine acceptance are reported separately.
The current scratch implementation and supported limits are described in
[WorkControlReplay.md](../Tools/Perf/WorkControlReplay.md).

## Scope and identity

One WorkControl indirect dispatch, early OR late, selected from one drained,
primed WorkloadCase target frame. Reuse the complete WorkloadCase/P3 case,
camera, graph/generation, asset identity, frozen residency, four-frame history
priming and production shader identity. Never substitute a later checkpoint or
another frame's inputs for the selected dispatch. Async raster, tessellation,
force-hardware, Printf, validation and other profilers are outside the first
counter-collection contract. Validation-only runs are separate evidence.

Capture the actual pipeline lease, module/entry, input AND device SPIR-V bytes
and hashes, specialization, subgroup requirements, bindless ABI, descriptor
heap generation, full push bytes and every reachable descriptor's kind, slot,
shader index, allocation generation, offset, range and alias group. A shader
cache key or equal marker name is not binding identity. Keep leases alive until
all submitted work completes. Reject rebinding/reallocation between capture
and replay. A scratch implementation must declare and verify a one-to-one
relocation map, including indices embedded inside buffers; it must not claim
physical binding equality. The implementation selects private allocations at
identical typed indices and retains the production executable. It reports
physical relocation explicitly; physical binding equality is not claimed.

## Resource closure

Audit roots: `streamClusterRasterWorkControlMain`,
`rasterStreamClusterWorkBins`, `StreamRasterCooperative.slang`, the low-wave
`rasterStreamCluster(..., false, false)` fallback and
`HybridRasterTriangle.slang`. Changes to this closure invalidate qualification.

| Resource / addressing | Access by selected dispatch | Required capture |
| --- | --- | --- |
| `push.hybridClusterBuffer` | Read: header, stable software list | Full allocation, including bins[4..11], capacity and list at `16 + 4 * bins[5]` |
| cluster indirect buffer | Indirect read | All bytes; command at offset 48, 12 bytes `(x,y,z)`; preserve 65535-wide flattening |
| `push.paramsBuffer` | Read | Full struct, both cameras, jitter, extents, page bounds, task count and frame data |
| `push.activeHeaderBuffer` | Read | Full header, active count/capacity/max clusters |
| `push.activeGroupBuffer` | Read | Full allocation, masks, flags, page/instance IDs, transforms |
| `push.pageTableBuffer` | Read in this dispatch | Full entries including `lastRequestFrame`, not WorkloadCase's mappings-only hash |
| `push.pageBuffer` | Read | Resident payload bytes and exact descriptor range; headers, positions, triangle bytes |
| `push.rasterBindingsBuffer` | Read | Entire struct; record base/capacity and nested descriptor indices |
| `rasterBindings.gpuSceneInstanceBuffer` | Read | Exact range/dimensions and instance identity flags |
| `bins[11]` pixel buffer | Atomic read/modify/write | Full allocation before and immediately after control; 64-bit packed depth/visibility, including inactive tail; bins[9] is reversed-Z, bins[10] is subpixel precision |
| request/visible-record resources in fallback | Conditional declarations; publication disabled by `false,false` | Verify reachable compiled path or retain private copies; unknown access rejects replay |

The first implementation rejects minimum subgroup sizes below 32. The fallback's
`testVisibility=false` and `publishRecord=false` suppress page
requests and visible-record publication; do not infer this from descriptor RW
types alone. Inspect the full transitive call graph. No traversal, cull,
classification, bin finish, resolve, residency update or history update may be
included in a replay submission. Group shared memory is dispatch-local.

HZB, instance visibility, previous/target camera history, classification policy,
cut, page residency/publication generation and upstream counters determine the
captured lists. Preserve their frozen identity and pre/post production sentinels
even where the chosen dispatch does not directly read them. No replay result
may reach a production resolve, page request consumer or history publisher.

## Capture, execution and gates

1. Freeze existing case; drain every producer/consumer queue and suspend frame
   submission, streaming/publication and host descriptor/parameter updates.
2. At the selected dispatch boundary, copy the complete read/write closure and
   indirect args into immutable snapshots with explicit transfer barriers.
   Record the actual production binding identity. Execute the ordinary control
   dispatch once; immediately capture its entire pixel allocation before any
   later software dispatch, hardware work or resolve changes the observation.
3. Finish/drain the control frame. Preserve production continuation bytes and
   tracked resource states separately from dispatch-input snapshots. Those are
   different states and cannot share a checkpoint. Record protected production
   state including page request ages, request queues, residency and history.
4. With renderer submission/publication suspended, restore the dispatch input
   bytes and resource states (or initialize an independently owned scratch
   closure). Read back and byte-compare restored inputs. Then submit exactly
   one indirect dispatch using the retained pipeline, heap and push bytes.
   Restoration/copy/verification commands are outside the measured submission.
5. Wait with a finite deadline; byte-compare full direct output with the
   same-frame control. Restore production continuation state and byte-compare
   all protected allocations. Only then release the renderer lease. Full direct
   pixel equality is sufficient; downstream equality must execute resolve on
   independent state and must identify which output regions are checked.
6. Only after successful unprofiled equality, binding and no-leak gates may a
   new NvPerf session begin. Before EVERY SDK pass, restore and verify the same
   dispatch snapshot, submit the same isolated dispatch, drain, compare output
   and verify restoration. SDK pass count and completion drive the loop, with a
   declared maximum; never satisfy multiple passes by advancing production
   frames. Reject dropped ranges/bytes, duplicate or absent ranges and partial
   decode. Record reset-induced cache conditions; diagnostic times are not
   ordinary performance samples.

Same queue alone does not prove exclusivity. Capture the engine submission
lease/queue ledger and overlapping work. External GPU activity remains a
separate interference limitation; no whole-device counter becomes shader- or
source-line-exclusive merely because the application submitted one dispatch.

## Failure and evidence

State progression: Capturing -> Captured -> RestoringInputs -> Ready ->
Submitted -> Completed -> Compared -> RestoringProduction -> Verified.
Counter eligibility requires Verified. Cancellation is checked before every
submission and pass; an accepted submission must retire before resources can
be reset/freed. Submit rejection, wait timeout, device error and restore failure
are distinct terminal causes. Preserve both the initiating and cleanup errors.
Unknown completion or failed restoration poisons the process: no next frame,
no publication, no success evidence, no reuse of the session/resources. A
bounded outer runner terminates only its owned process if driver cleanup hangs.

Archive versioned manifest, frozen identity, binding table and SPIR-V bytes,
input/control/replay/continuation/restored raw bytes, pass ledger, submit/wait
results, cancellation/deadline, state checks, process lifecycle, SDK config and
raw counter images. Hashes protect integrity, not provenance. Current verifier
recomputes byte comparisons and lifecycle gates without executing archived
code. A saved `equal=true` or `restored=true` is not proof.

Keep existing `metallic-nvperf-v1` evidence valid ONLY as
`in-frame-command-ranges-on-graphics-queue`. New isolated evidence requires its
own schema and gates; never upgrade old archives by filling missing fields.
Report `counterImageReevaluated=false` unless the current verifier actually
decodes the raw SDK image again.

Required negative cases: changed SPIR-V/heap/push/indirect offset; missing
resource or alias; stale frame/history; nonzero initial pixels; one-byte output
mismatch; page age/request/history leak with unchanged depth; second pass
without reset; cancellation before/after accepted submit; timeout; rejected
submit; device loss; failed input or continuation restoration; old-evidence
scope promotion; truncated/tampered bytes; partial decode. CPU fault injection
does not establish GPU restoration, queue isolation or visual correctness.
