# Streaming Roadmap — Milestone G Detailed Breakdown

Sub-roadmap for **roadmap.md Section 2.7 — Optional residency / streaming layer**.

Reference implementation: `vk_lod_clusters` (`E:\vk_lod_clusters`).

---

## Current State Summary

Metallic already has a working **CPU-managed node-level residency system** in `ClusterStreamingService` (cluster_streaming_service.h). What exists:

- **Always-resident coarsest LOD**: root/coarsest LOD nodes are pinned resident on rebuild
- **GPU residency request generation**: meshlet_cull.slang emits `ClusterResidencyRequest` via atomic worklist when a node is not resident
- **CPU readback + promote/evict loop**: `runUpdateStage()` reads requests from host-visible buffer, promotes pending nodes within budget, FIFO eviction of oldest dynamic nodes
- **Group page table**: per-group `uint32_t` mapping group index to resident heap offset (UINT32_MAX = invalid)
- **Resident heap allocator**: free-range list with first-fit allocation and coalescing merge on release
- **Memory budget**: configurable `streamingBudgetNodes` (default 64)
- **Debug stats**: resident/pending/evicted/promoted counts, heap usage

What the reference (`vk_lod_clusters`) has that Metallic does **not**:

| Feature | vk_lod_clusters | Metallic |
|---------|----------------|----------|
| Group-level (not node-level) residency | Yes — `StreamingGroup` per group | No — node granularity only |
| Age-based GPU-side unload requests | `stream_agefilter_groups.comp.glsl` | No — CPU FIFO eviction only |
| GPU scene patch shader | `stream_update_scene.comp.glsl` | No — CPU memcpy only |
| Async transfer queue integration | `StreamingStorage` + async VkBufferCopy | No |
| Per-frame load/unload caps | `maxLoads` / `maxUnloads` counters | Uncapped (all pending promoted per frame) |
| Frame-index deduplication of requests | atomicMax with frame tag in 64-bit VA | atomicOr flag (no frame tag) |
| GPU persistent CLAS allocator | 5-shader pipeline (gap scan + bin + alloc) | N/A (no RT CLAS streaming) |
| Streaming statistics readback | `StreamingStats` + GPU memory accounting | Debug counters only |
| Multi-task pipelining | 3 task slots with timeline semaphores | Single-frame synchronous |

---

## Phase G.1 — Group-Level Residency Granularity

**Goal:** Switch from node-level to group-level residency tracking, matching the vk_lod_clusters model where each `GPUClusterGroup` is independently loadable/unloadable.

### G.1.1 — Group residency state buffer

Replace the per-node `lodNodeResidencyBuffer` with a per-group residency state buffer.

- New buffer: `groupResidencyBuffer` — one `uint32_t` per group
- Bit flags: `kGroupResident` (bit 0), `kGroupRequested` (bit 1), `kGroupAlwaysResident` (bit 2)
- Always-resident groups: the coarsest-LOD groups (leaf groups of the coarsest node in each primitive group)
- Remove `m_residencyNodeLeafGroups` indirection — residency decisions are per-group

**Files:** `cluster_streaming_service.h`, `gpu_cull_resources.h`

### G.1.2 — GPU request generation at group level

Modify `meshlet_cull.slang` to emit residency requests per-group instead of per-node.

- During LOD traversal, when a group is reached but not resident, emit a `ClusterResidencyRequest` with the group index
- Update `ClusterResidencyRequest` struct: replace `targetNodeIndex` with `targetGroupIndex`, add `lodLevel`
- Atomic deduplication: `InterlockedOr` on `groupResidencyBuffer[groupIndex]` with `kGroupRequested`

**Files:** `meshlet_cull.slang`, `gpu_cull_resources.h`

### G.1.3 — CPU-side group promote/evict

Rework `ClusterStreamingService` internals to operate per-group:

- `m_dynamicResidentGroups` replaces `m_dynamicResidentNodes` (FIFO/LRU list of group indices)
- Heap allocator unchanged (already operates at meshlet-index granularity)
- `ensureResidentHeapSliceForGroup()` allocates heap range and memcpys cluster indices for one group
- `invalidateResidentGroupsForGroup()` releases heap range and resets page table entry
- Budget expressed in group count or total meshlet count (configurable)

**Files:** `cluster_streaming_service.h`

### G.1.4 — Per-frame load/unload caps

Add configurable caps to prevent frame stalls:

- `m_maxLoadsPerFrame` (default 128): max groups promoted per update
- `m_maxUnloadsPerFrame` (default 256): max groups evicted per update
- Excess requests deferred to next frame via `m_pendingResidencyGroups` queue

**Files:** `cluster_streaming_service.h`

**Done when:** streaming operates per-group, ImGui stats show group-level resident/pending/evicted counts, and the resident raster path is visually correct.

---

## Phase G.2 — GPU-Side Age-Based Unload Requests

**Goal:** Move eviction decisions from CPU FIFO to GPU age-tracking, so the GPU tells the CPU which groups to unload based on actual usage.

### G.2.1 — Per-group age counter

Add a `uint16_t age` field per resident group, stored in a GPU-accessible buffer.

- New buffer: `groupAgeBuffer` — one `uint16_t` per group (or pack into groupResidencyBuffer)
- Age incremented each frame for all resident groups
- Age reset to 0 when a group's meshlets are accessed during traversal

**Files:** `cluster_streaming_service.h`, `gpu_cull_resources.h`, new shader

### G.2.2 — Age filter compute shader

New shader: `stream_agefilter_groups.slang`

- One thread per active resident group
- Increments age (saturates at 0xFFFF)
- If age > `ageThreshold` (default 16): appends group to unload request worklist
- Dispatched after meshlet cull pass (traversal resets ages of accessed groups)
- Uses the existing worklist pattern (`gpuDrivenAppendWorkItemSlot`)

**Files:** new `Shaders/Streaming/stream_agefilter_groups.slang`, `gpu_driven_constants.h`

### G.2.3 — Unload request readback

Extend readback to include unload requests:

- New unload request buffer + worklist state buffer
- CPU reads unload requests alongside load requests in `runRequestReadbackStage()`
- Groups in unload list are evicted (heap released, page table invalidated, age reset)
- Remove CPU-side FIFO eviction — GPU age tracking supersedes it

**Files:** `cluster_streaming_service.h`

### G.2.4 — Traversal age reset

In `meshlet_cull.slang`, when a group's meshlets are emitted as visible, reset that group's age to 0 via atomic write/store to `groupAgeBuffer`.

**Files:** `meshlet_cull.slang`

**Done when:** eviction is driven by GPU age, groups unseen for N frames are automatically unloaded, and memory stays within budget without CPU-side FIFO.

---

## Phase G.3 — GPU Scene Patch Shader

**Goal:** Replace CPU memcpy for page table + resident heap updates with a GPU compute shader that patches the scene, eliminating per-frame host writes on the critical path.

### G.3.1 — StreamingPatch data structure

Define a patch descriptor uploaded from CPU to GPU:

```cpp
struct StreamingPatch {
    uint32_t groupIndex;
    uint32_t residentHeapOffset;   // kInvalidAddress for unload
    uint32_t clusterStart;         // source offset in groupMeshletIndices
    uint32_t clusterCount;
};
```

- CPU prepares patch list each frame (load patches + unload patches)
- Upload via staging buffer or upload ring

**Files:** `gpu_cull_resources.h`, `cluster_streaming_service.h`

### G.3.2 — Scene update compute shader

New shader: `stream_update_scene.slang`

- One thread per patch operation
- For load patches: copies meshlet indices from source buffer to resident heap, writes page table entry
- For unload patches: writes `kInvalidAddress` to page table entry
- Dispatched before meshlet cull pass in the frame graph

**Files:** new `Shaders/Streaming/stream_update_scene.slang`

### G.3.3 — Upload service integration

Use `VulkanUploadService` or `VulkanUploadRing` to stage patch data:

- `stageBuffer()` for patch list upload
- `recordPendingUploads()` in the streaming update pass command buffer
- Source meshlet indices buffer remains device-local (read by the update shader)

**Files:** `cluster_streaming_update_pass.h`, `cluster_streaming_service.h`

### G.3.4 — Remove host-visible resident heap

Once the GPU update shader handles all page table and heap writes:

- Change `residentGroupMeshletIndicesBuffer` from host-visible to device-local
- Change `lodGroupPageTableBuffer` from host-visible to device-local
- Keep `groupResidencyBuffer` host-visible for readback (or use dedicated readback copy)

**Files:** `cluster_streaming_service.h`

**Done when:** page table and resident heap are updated entirely on GPU, no per-frame host-visible memcpy on the render path, and the result is visually identical.

---

## Phase G.4 — Streaming Storage & Async Transfer

**Goal:** Decouple geometry data upload from the graphics queue using the async transfer queue (Phase 0.3 infrastructure).

### G.4.1 — StreamingStorage allocator

Create a device-local storage pool for cluster group data:

- `StreamingStorage` class: owns a large device-local buffer (configurable, default 512 MB)
- Sub-allocator: offset-based allocation (similar to existing resident heap free-range allocator)
- Each loaded group gets a contiguous region in the storage buffer
- On unload, region returned to free list with coalescing

**Files:** new `Source/Rendering/streaming_storage.h`

### G.4.2 — Staging buffer and copy regions

CPU-side upload preparation:

- Per-frame staging buffer (host-visible, sized to `maxTransferBytes`, default 32 MB)
- For each load request: memcpy cluster data into staging buffer, record `VkBufferCopy` region
- Batch all copies into a single `vkCmdCopyBuffer` on the transfer command buffer

**Files:** `streaming_storage.h`, `cluster_streaming_service.h`

### G.4.3 — Async transfer queue submission

Use the existing async transfer queue infrastructure (Phase 0.3):

- Record copy commands on the transfer command buffer
- Submit with timeline semaphore signal
- Graphics queue waits on transfer semaphore before streaming update pass
- Transfer and graphics work can overlap across frames

**Files:** `cluster_streaming_service.h`, `cluster_streaming_update_pass.h`, `main_vulkan.cpp`

### G.4.4 — Multi-task pipelining

Allow N tasks in flight (default 2-3):

- Each task has: staging allocation, copy regions, patch list, completion semaphore value
- Tasks cycle through: prepare → transfer → update → recycle
- CPU can prepare task N+1 while GPU executes task N

**Files:** `cluster_streaming_service.h`

**Done when:** geometry uploads happen on the transfer queue, graphics queue is not stalled by uploads, and pipeline telemetry shows overlap between transfer and render.

---

## Phase G.5 — Frame-Index Request Deduplication

**Goal:** Prevent redundant load requests for the same group within a single frame by encoding the frame index into the group address.

### G.5.1 — 64-bit virtual address scheme

Adopt the vk_lod_clusters pattern:

- Group addresses stored as `uint64_t` instead of `uint32_t` page table offsets
- Valid address: actual device address or heap offset (bit 63 = 0)
- Invalid address: `STREAMING_INVALID_ADDRESS_START | frameIndex` (bit 63 = 1)
- When a group is requested, `atomicMax` writes the current frame index into the invalid address
- If `atomicMax` returns the current frame index, the request was already made this frame — skip

### G.5.2 — Shader-side deduplication

In `meshlet_cull.slang`:

```slang
uint64_t prevAddr = atomicMax(groupAddresses[groupIndex], currentFrameTag);
bool alreadyRequested = (prevAddr == currentFrameTag);
if (!alreadyRequested) {
    // append to load request worklist
}
```

This replaces the current `InterlockedOr` flag-based approach and eliminates duplicate requests without CPU intervention.

### G.5.3 — CPU request handler update

- Readback only the load/unload counters and geometry group arrays (not per-group state)
- Frame index validated on CPU to ensure readback corresponds to correct frame

**Done when:** each group is requested at most once per frame regardless of how many instances reference it.

---

## Phase G.6 — Comprehensive Statistics & Monitoring

**Goal:** Provide detailed streaming telemetry for performance analysis and budget tuning.

### G.6.1 — StreamingStats structure

```cpp
struct StreamingStats {
    uint32_t residentGroupCount;
    uint32_t residentClusterCount;
    uint32_t alwaysResidentGroupCount;
    uint32_t dynamicResidentGroupCount;

    uint64_t storagePoolCapacityBytes;
    uint64_t storagePoolUsedBytes;
    uint32_t residentHeapCapacity;
    uint32_t residentHeapUsed;

    uint32_t loadRequestsThisFrame;
    uint32_t unloadRequestsThisFrame;
    uint32_t loadsExecutedThisFrame;
    uint32_t unloadsExecutedThisFrame;
    uint32_t loadsDeferredThisFrame;

    uint64_t transferBytesThisFrame;
    float    transferUtilization;       // actual / budget

    uint32_t failedAllocations;
};
```

### G.6.2 — ImGui streaming dashboard

Extend the existing "Vulkan Sponza" debug window:

- "Streaming" collapsible section with:
  - Resident groups / total groups (bar chart)
  - Storage pool usage (bar chart)
  - Load/unload requests per frame (rolling graph)
  - Transfer bandwidth utilization
  - Age histogram of resident groups
  - Per-LOD level residency breakdown

### G.6.3 — GPU-side statistics readback

Add a small statistics buffer written by streaming shaders:

- Age filter shader writes: unload count, average age
- Update shader writes: patches applied, copy bytes
- Readback via host-visible copy (statistics only, not control path)

**Done when:** all streaming metrics visible in ImGui, no per-frame control-path readbacks, statistics readbacks are optional.

---

## Phase G.7 — Memory Budget Automation

**Goal:** Automatically tune streaming budget based on available VRAM and scene complexity.

### G.7.1 — VRAM budget query

Query available VRAM via `VK_EXT_memory_budget`:

- `vkGetPhysicalDeviceMemoryProperties2` with `VkPhysicalDeviceMemoryBudgetPropertiesEXT`
- Compute available headroom: `budget - usage` per heap
- Set streaming pool size as fraction of available headroom (e.g., 50%)

**Files:** `vulkan_backend.cpp`, `cluster_streaming_service.h`

### G.7.2 — Adaptive budget

- If failed allocations > 0 per frame: increase eviction aggressiveness (lower age threshold)
- If storage pool utilization < 50%: decrease eviction aggressiveness
- Smoothed over N frames to avoid oscillation

### G.7.3 — Budget presets

- "Low" (256 MB storage, 64 groups budget)
- "Medium" (512 MB storage, 256 groups budget)
- "High" (1 GB storage, 1024 groups budget)
- "Auto" (VRAM-based)

Expose in ImGui streaming settings.

**Done when:** Metallic can run on GPUs with varying VRAM without manual budget tuning.

---

## Phase G.8 — Robustness & Edge Cases

**Goal:** Handle error conditions, device-lost recovery, and streaming correctness under stress.

### G.8.1 — Error tracking

Add GPU-side error counters (inspired by vk_lod_clusters' 24 error fields):

- `errorUpdate`: scene patch failure
- `errorAgeFilter`: age filter overflow
- `errorAllocation`: heap allocation failed on GPU
- `errorPageTable`: page table write out of bounds
- Log errors via spdlog on CPU readback

### G.8.2 — Streaming state reset on resize/reload

- Pipeline reload (`F6`): flush all in-flight streaming tasks, rebuild state
- Shader reload (`F5`): preserve streaming state (shaders are stateless)
- Window resize: no streaming impact (screen-space LOD recalculated automatically)

### G.8.3 — Graceful degradation

- If streaming storage exhausted: fall back to always-resident coarsest LOD (already the case)
- If transfer queue unavailable: fall back to graphics queue copies
- If age filter dispatch fails: fall back to CPU-side FIFO eviction

**Done when:** streaming recovers cleanly from all error conditions without visual corruption or crashes.

---

## Dependency Graph

```
G.1 (Group-Level Residency)
 └─> G.2 (Age-Based Unload)
      └─> G.3 (GPU Scene Patch)
           └─> G.4 (Async Transfer)
                └─> G.5 (Frame-Index Dedup)

G.6 (Statistics) — can proceed in parallel after G.1
G.7 (Budget Automation) — requires G.4 + G.6
G.8 (Robustness) — requires G.3, can proceed in parallel with G.4+
```

## Implementation Rules

- **No control-path readbacks**: GPU requests and unload decisions flow through worklist buffers; only statistics are read back.
- **Reuse existing infrastructure**: upload ring, transient allocator, worklist pattern, timeline semaphores.
- **Keep streaming optional**: `enableResidencyStreaming` toggle must remain functional; non-streaming path always works.
- **Validate after each phase**: build → launch → enable streaming → verify visual correctness and counters.
- **CLAS/RT streaming out of scope**: RT acceleration structure streaming is a separate feature beyond Milestone G. The GPU persistent allocator from vk_lod_clusters is not needed until RT streaming is tackled.
