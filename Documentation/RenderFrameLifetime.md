# Render frame lifetime foundation

The editor uses two frame slots, each with its own `RenderFrameContext`, command
pool/buffer and acquire semaphore. It selects `submittedFrameIndex % 2` and waits
only when reusing that slot. The upload subsystem and GPU scene receive the same
two-slot capacity and slot index. Offscreen preview remains synchronous with one
slot. Presentation mode is unchanged.

## Submission contract

`QueueSubmissionTracker` owns a timeline semaphore for one queue. A
`GpuCompletionPoint` identifies a single recording/submission, independently of
the CPU frame number or reusable slot index:

- Recording: not complete and not waitable.
- Submitted: receives its timeline value only after `Queue::submit` succeeds.
- Cancelled: no GPU submission exists; recorded commands must already have been
  reset/discarded. Copies of the point observe cancellation.

Use one tracker per submitting queue, with externally serialized calls. Each
tracked command buffer must begin with the same recording frame. The tracker
rejects command buffers from a different frame or an earlier recording of a
reused frame context. Submit all command buffers for that frame in one tracked
submission. Direct `Queue::submit` remains available to legacy callers, which
continue to own their synchronization and lifetimes.

The normal sequence is `frame.begin(frameNumber)`, reset the command pool,
`commandBuffer.begin(&frame)`, record/end, then `tracker.submit(desc, frame)`.
`frame.begin` waits for its previous submission before releasing retained
resources. On an abandoned recording, reset the command pool before
`frame.cancel()`. Device and queue ownership must outlive their completion
points and retained resources.

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

Existing synchronous render-graph self-submission and legacy upload callers
retain their current contracts. A cancelled graph recording also needs its
CPU-side resource-state/history bookkeeping invalidated before retrying; the
preview renderer marks the graph dirty on cancellation.

A graphics completion point does not establish presentation completion. Acquire
semaphores belong to frame slots; render-finished semaphores belong to swapchain
images and are reused only after reacquiring that image. Suboptimal acquisition
still submits the acquired image so its semaphore is consumed. Swapchain resize
and shutdown drain the device; viewport descriptor replacement drains submitted
editor work before freeing the old ImGui descriptor.
History ordering across independent queues also requires explicit GPU waits;
one queue's timeline value must not be treated as global device completion.

## Validation

`FrameContextTests.cpp` blocks submitted GPU work on a host-signalled timeline
semaphore to exercise pending resource retention, slot reuse rejection, upload
growth/destruction, descriptor snapshots across dispatches/submissions, and
history resize/reset while three dependent submissions are queued. The history
readback must contain `41, 42, 43`. The two-slot graph test records a second frame
while the first is blocked, cycles both slots across six submissions, checks
upload/image readbacks, and verifies external completion tracking during rebuild.
Run with Vulkan synchronization validation enabled:

```powershell
$env:VK_KHRONOS_VALIDATION_VALIDATE_SYNC = 'true'
build\tests\MetallicRhiTests.exe --gtest_filter=RhiCommand.frame_*:RhiResource.frame_*:RhiRendering.frame_* --rhi-validation
$env:METALLIC_SMOKE_TEST_FRAMES = '6'
build\Source\Metallic.exe --smoke-test
```

The synchronization-validation environment setting is documented in the
[LunarG layer settings](https://vulkan.lunarg.com/doc/view/latest/windows/khronos_validation_layer.html).
Multi-frame smoke mode is bounded to 64 frames and defaults to one frame.
