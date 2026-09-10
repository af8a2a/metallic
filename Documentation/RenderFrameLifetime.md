# Render frame lifetime foundation

The editor uses two frame slots, each with its own `RenderFrameContext`, command
pool/buffer and acquire semaphore. It selects `submittedFrameIndex % 2` and waits
only when reusing that slot. The upload subsystem and GPU scene receive the same
two-slot capacity and slot index. Offscreen preview remains synchronous with one
slot. Presentation mode is unchanged.

## Submission contract

`QueueSubmissionTracker` owns a timeline semaphore for one queue. A
`GpuCompletionPoint` identifies a recording/submission batch, independently of
the CPU frame number or reusable slot index:

- Recording: not complete and not waitable.
- Submitting: one or more segments succeeded; the open batch is not reusable,
  complete or waitable. Record all command buffers before submitting any segment.
- Submitted: the batch is sealed; every contributing queue must reach its last
  successful timeline value. Values are assigned only after `Queue::submit` succeeds.
- Cancelled: no GPU submission exists; recorded commands must already have been
  reset/discarded. Copies of the point observe cancellation.

Use one tracker per submitting queue, with externally serialized calls. Each
tracked command buffer must begin with the same recording frame. The tracker
rejects command buffers from a different frame or an earlier recording of a
reused frame context. `submit` closes a single-submission frame. For a batch,
call each queue tracker's `submitSegment` and finally `frame.finishSubmission()`.
Each returned segment point covers only that submission; the frame point covers
all successful segments. Repeated signals on one queue coalesce to its last value.
`value()` returns zero for a composite point; use `wait`, `isComplete` or
`appendWaits` rather than comparing timeline values belonging to different queues.
Finite waits share one timeout budget across all contributing timelines.

On partial failure, `frame.cancel()` seals the successful prefix and preserves its
resources until completion; it cannot cancel work already submitted to the GPU.
Direct `Queue::submit` remains available to legacy callers, which continue to own
command-buffer/resource lifetimes and wait before resetting or destroying them.

The normal sequence is `frame.begin(frameNumber)`, reset the command pool,
`commandBuffer.begin(&frame)`, record/end, then `tracker.submit(desc, frame)`.
`frame.begin` waits for its previous submission before releasing retained
resources. On an abandoned recording, reset the command pool before
`frame.cancel()`. Device and queue ownership must outlive their completion
points and retained resources.

`GpuCompletionPoint::appendWaits` exports/coalesces GPU waits. The point must stay
alive until the consumer completes. `RenderFrameContext::addDependency` retains
it and adds waits to tracked submissions; `CommandBuffer::addDependency` retains
it until the command buffer's next recording and adds waits even to direct
`Queue::submit`. Explicit waits and command-buffer waits are coalesced by timeline.

## Resource rules

- `RenderFrameContext::retain` keeps shared resource generations alive from
  recording until completion and slot reuse. `DeferredReleaseQueue` retires
  resources using their supplied completion points, without counting CPU frames.
  Subsystem retirement retains old generations against every outstanding frame,
  including when a newer replacement recording is subsequently cancelled.
- Tracked uploads use the supplied slot, preserve allocation cursors across
  graph executions, and reject reuse while that slot is incomplete. Old staging
  buffers survive growth and uploader destruction. Constant arenas have aligned
  slot strides and report overflow instead of wrapping over earlier allocations.
  End the active upload frame before switching slots.
- `ComputeProgram` uses immutable descriptor-table snapshots per dispatch.
  A table may be rewritten only after its completion point finishes. Recorded
  work retains the pipeline generation through clear/reinitialization. Callers
  must still retain the actual buffers, textures and samplers bound to that table.
- History transitions retain the resource generation through resize/reset.
  General-to-General transitions emit barriers for successive read/write access;
  these barriers establish GPU dependencies independently of a CPU wait.
- Path-tracing cache parameter uploads use completion-protected allocations.
  Timestamp results are consumed only after the recording's submission completes;
  cancelled recordings discard their pending results.

## Overlap and compatibility

`RenderGraphPass::supportsFrameOverlap()` explicitly opts a pass into recording
while its previous submission is pending. The enabled paths are standard/OpenPBR
path tracing with radiance cache off, material visualization, ray-query
visualization, triangle rasterization, clear, copy and image sample passes.

Passes that still use singleton host-written buffers, mutable custom heaps,
readbacks or SDK contexts retain a graph-level completion wait. This includes
the legacy wireframe/shader-object paths, GPU-driven streaming, DLSS/NRD/RTXDI,
and SHaRC/NRC modes. New/custom passes default to this compatibility behavior
until their resource lifetimes are audited. The editor still has two slots;
those graphs serialize GPU-dependent recording rather than risking data races.

The executor tracks completion points for externally submitted command buffers.
Compilation, resizing, shader reload and destruction wait before replacing
graph resources or query pools. Runtime scene identity/content/transform changes
also drain previous graph work before updating shared geometry or acceleration
structures. Steady-state rendering and camera push-constant updates can overlap.
Shared render targets use explicit GPU barriers, including same-state color/depth
attachment and transfer writes; a second copy of each graph target is unnecessary
for the current single graphics queue.

Legacy upload callers retain their existing synchronization contract. A cancelled
graph recording also needs its CPU-side resource-state/history bookkeeping
invalidated before retrying; the preview renderer marks the graph dirty on
cancellation. Self-submission failure similarly requires recompilation.

A graphics completion point does not establish presentation completion. Acquire
semaphores belong to frame slots; render-finished semaphores belong to swapchain
images and are reused only after reacquiring that image. Suboptimal acquisition
still submits the acquired image so its semaphore is consumed. Swapchain resize
and shutdown drain the device; viewport descriptor replacement drains submitted
editor work before freeing the old ImGui descriptor.
History ordering across independent queues also requires explicit GPU waits;
one queue's timeline value must not be treated as global device completion.

## Self-submitting graphs and multiple queues

`execute(RenderGraphSubmitDesc)` now owns two slots, with a frame context and
command pools/buffers per slot and queue. An externally initialized host with
only one slot keeps that limit. The default host/upload capacity is two. The
executor waits only for the slot being reused; `slotWaitTimeoutNanoseconds = 0`
provides nonblocking backpressure without beginning or mutating a new frame.
Legacy passes or scene changes still require the compatibility completion wait.

Each actual `Queue*` has a persistent submission tracker. One segment is recorded
per pass. Declared resource uses generate producer/consumer and successive-access
dependencies, including read/read layout transitions. Cross-queue predecessors
become timeline waits; same-queue predecessors use submission order and resource
barriers. Disjoint audited branches can execute independently. There is no shared
multi-producer timeline and no unconditional signal/wait chain between every pass.

Graph buffers/textures and graph upload buffers allow graphics, compute and copy
queue-family access. Vulkan uses concurrent sharing when these resolve to multiple
families. A resource moving to another queue uses an acquire barrier after a GPU
wait: source stage/access are NONE, while layout and destination access remain
explicit. Shader-read stages are filtered to the command pool's queue capabilities.
This follows the [Khronos semaphore/layout-transition examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html).
Imported/private exclusive resources are not automatically ownership-transferred.

`supportsAsyncQueue()` is separate from `supportsFrameOverlap()`. It requires all
GPU hazards to be reflected graph resources and private resources to be compatible
with the chosen queue. Audited triangle, clear, image sample, color copy and buffer
write/copy passes opt in. Color copy selects the copy queue. A missing optional
compute/copy queue falls back to the supplied graphics queue. Other passes run on
graphics and form an ordering boundary; explicit subsystem hooks run in graphics
prologue/epilogue segments with dependencies on the graph. The always-present
scene resource registry and upload subsystem have no GPU hooks of their own.

The next frame's first submission on each queue waits on the previous graph's
aggregate completion. This allows CPU recording overlap and independent queues
within a frame while preserving shared render targets and history across frames.
It does not yet overlap GPU work between frames that use these shared resources.
`historyResources`, when supplied, is advanced using the graph's frame index.
GPU timing collection for self-submission remains unavailable; existing external
command-buffer timestamp collection is unchanged.

`lastSubmittedCompletion()` returns the whole graph's completion, including a
partially submitted batch after failure. `waitCompletions` accepts caller-supplied
GPU dependencies. External `execute(CommandBuffer&)` and `transitionOutput` attach
the last self-submission wait to the command buffer automatically; tracked external
consumers are also included in subsequent graph waits/rebuild protection. Legacy
untracked consumers must still be completed before graph reuse/destruction.
Queue changes do not recycle a slot's command pools until that slot completes.
Compilation, shader reload and destruction wait for every outstanding slot.

On recording/submission failure, unsubmitted recordings are discarded, successful
segments retain their resources, and the graph is marked uncompiled. History is
reset/reinitialized because its recorded states may not match executed commands.
Recompile before retrying. Passes/subsystems that publish private state while
recording must also reset their own state after an abandoned recording; graph
recompilation cannot roll back those private publications. Preflight and busy-slot
failures leave the graph usable.

This mirrors the separation in Unreal's Vulkan submission code: payload completion
belongs to a queue, and a device-wide completion waits for every queue endpoint
(`Engine/Source/Runtime/VulkanRHI/Private/VulkanSubmission.cpp`,
`RHIBlockUntilGPUIdle`), rather than treating graphics completion as global.

## Validation

`FrameContextTests.cpp` blocks submitted GPU work on a host-signalled timeline
semaphore to exercise pending resource retention, slot reuse rejection, upload
growth/destruction, descriptor snapshots across dispatches/submissions, and
history resize/reset while three dependent submissions are queued. The history
readback must contain `41, 42, 43`. The two-slot graph test records a second frame
while the first is blocked, cycles both slots across six submissions, checks
upload/image readbacks, and verifies external completion tracking during rebuild.
Additional tests cover partial multi-queue completion, timeline coalescing,
independent copy progress while graphics is blocked, self-submitted two-slot
backpressure, six-frame upload reuse, pending external consumer handoff,
copy/compute/graphics dependency chains and fan-out, graphics-to-copy image
transitions, temporal history and failure/recompile recovery. Run with Vulkan
synchronization validation enabled:

```powershell
$env:VK_KHRONOS_VALIDATION_VALIDATE_SYNC = 'true'
build\tests\MetallicRhiTests.exe --gtest_filter=RhiCommand.frame_*:RhiResource.frame_*:RhiRendering.frame_* --rhi-validation
$env:METALLIC_SMOKE_TEST_FRAMES = '6'
build\Source\Metallic.exe --smoke-test
```

The synchronization-validation environment setting is documented in the
[LunarG layer settings](https://vulkan.lunarg.com/doc/view/latest/windows/khronos_validation_layer.html).
Multi-frame smoke mode is bounded to 256 frames and defaults to one frame.

Latest validation for the multi-queue extension: 36 focused RHI regressions passed
with synchronization validation enabled, plus six-frame editor smoke runs for
material visualization and path tracing. No VUID or synchronization hazards were
reported after the output-buffer usage fix. The GPU-driven smoke tests emitted
two mesh/fragment shader interface warnings (unused mesh output Location 1).
