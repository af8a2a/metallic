# Streaming Roadmap — Remaining Non-RT Work

Sub-roadmap for **roadmap.md Section 2.7 — Optional residency / streaming layer**.

Reference implementation: `vk_lod_clusters` (`E:\vk_lod_clusters`).

This revision reflects the current Metallic codebase state as of the latest
comparison pass. It intentionally tracks only the **non-RT / non-CLAS** work
that is still worth doing next.

---

## Scope

Included here:

- Cluster/group residency for raster cluster LOD rendering
- Load/touch/unload request generation and handling
- GPU scene patching
- Streaming storage and transfer queue integration
- Telemetry, budgeting, fallback behavior, and reload robustness

Explicitly out of scope for this document:

- RT / CLAS streaming
- BLAS caching
- CLAS compaction
- Persistent CLAS allocator
- Any ray tracing acceleration structure residency work

---

## Current State Summary

Metallic already implements the majority of the original Milestone G plan:

- **Group-level residency** is in place
  - `groupResidencyBuffer` uses `Resident`, `Requested`, `AlwaysResident`, and `Touched` bits
  - coarsest groups are pinned resident
- **64-bit page-table based request deduplication** is in place
  - invalid addresses encode the request frame index
  - shader-side `InterlockedMax` prevents duplicate missing-group requests within a frame
- **GPU request generation** is in place
  - `meshlet_cull.slang` emits load requests for missing groups
  - resident touches are emitted separately from missing-load requests
- **GPU age-filter driven unload requests** are in place
  - `stream_agefilter_groups.slang` increments ages and appends unload requests
  - CPU FIFO eviction remains as a fallback if the age-filter dispatch is missing
- **GPU scene patching** is in place
  - `stream_update_scene.slang` writes the page table
  - dynamic group meshlet indices are copied via transfer/upload path
- **Device-local streaming storage + async transfer** are in place
  - `StreamingStorage` owns the device-local storage pool
  - staging buffers and copy regions are prepared per task
  - dedicated transfer queue + timeline semaphore wait path is integrated
  - graphics-queue copy fallback exists when transfer is unavailable
- **Multi-task pipelining** is in place
  - three task slots cycle through prepared / transfer submitted / update queued
- **Per-frame caps** are in place
  - load and unload caps exist and are configurable
- **Telemetry and budgeting** are in place
  - ImGui exposes storage usage, request counts, transfer utilization, age histogram, and per-LOD residency
  - VRAM budget query via `VK_EXT_memory_budget` is wired in
  - budget presets and adaptive age-threshold tuning are implemented
- **Reload and fallback robustness** are largely in place
  - pipeline reload resets streaming task state
  - streaming remains optional
  - coarsest-LOD fallback remains valid when storage is exhausted

---

## Status Matrix

| Original Phase | Status in Metallic | Notes |
|---|---|---|
| G.1 Group-level residency granularity | Complete | Group bits, always-resident groups, group request flow all exist |
| G.2 GPU-side age-based unload requests | Complete | GPU unload requests exist; CPU FIFO remains as fallback only |
| G.3 GPU scene patch shader | Complete | Patch struct + shader + device-local page table are in use |
| G.4 Streaming storage & async transfer | Complete | Storage pool, staging, async transfer queue, and task ring exist |
| G.5 Frame-index request deduplication | Complete | 64-bit invalid-address tagging is in shader path |
| G.6 Statistics & monitoring | Complete enough | Main telemetry exists; remaining work is mostly overhead reduction |
| G.7 Memory budget automation | Complete | VRAM query, presets, auto sizing, adaptive age policy exist |
| G.8 Robustness & edge cases | Partial | Core reload/fallback/error paths exist; stress hardening remains |

---

## Remaining Differences vs `vk_lod_clusters` (Non-RT)

The most important remaining differences are no longer functional gaps; they are
mostly about **state ownership, scaling, and steady-state overhead**.

### 1. No compact GPU resident-group list yet

`vk_lod_clusters` keeps a compact `activeGroups` style resident list and runs
its age filter over resident groups only.

Metallic currently:

- tracks resident order primarily in CPU-side vectors such as `m_dynamicResidentGroups`
- dispatches the age filter over `groupCount`, not over `residentGroupCount`
- rebuilds debug/telemetry views by scanning broad group ranges

Impact:

- age-filter cost scales with total scene group count instead of active resident count
- future GPU-only residency bookkeeping is harder to layer in cleanly

### 2. CPU still owns the canonical residency / age mirror

Metallic currently keeps CPU-side arrays such as:

- `m_groupResidencyState`
- `m_groupAgeState`
- `m_groupResidentSinceFrame`

These are mirrored into per-frame buffers and also used for CPU-side request
handling and telemetry reconstruction.

Impact:

- steady-state behavior still depends on full CPU-side canonical state
- age progression is mirrored on CPU even though GPU age-filter logic exists
- the render path is not yet fully GPU-owned for residency metadata

### 3. Full-buffer host writes still happen in the steady state

`uploadCanonicalStateToFrame()` still performs broad uploads / clears of:

- group residency state
- age state
- request buffers
- unload request buffers

Impact:

- steady-state frames still pay broad host memcpy cost
- the streaming control path is functionally correct, but not yet as lean as it can be

### 4. Request handling still relies on large host-visible request resources

The current request and unload worklists are read back through host-visible
buffers. This works and is simple, but it is not the cleanest long-term scaling
model for large scenes.

Impact:

- residency request buffers remain broad CPU-visible resources
- the system is correct, but the readback model is more heavyweight than necessary

### 5. Some debug and telemetry work scales with scene size

The current dashboard is useful, but parts of the bookkeeping still scan all
groups to build:

- age histogram
- resident-group totals
- per-LOD residency breakdown

Impact:

- debug overhead can become noticeable on larger content
- telemetry cost is less bounded than it could be

---

## Next-Step Plan

The next plan should focus on **scaling down CPU ownership and full-array work**
without changing the existing feature set.

## Phase N.1 — Compact Active Resident Group Table

**Goal:** Introduce a compact GPU-visible active resident group list so streaming
passes scale with resident groups instead of total groups.

### N.1.1 — Resident group list resources

- Add a compact `activeResidentGroupsBuffer`
- Add a matching count buffer or counter field
- Keep the current page table and group residency bits as-is

### N.1.2 — Update path integration

- Extend the streaming update path so load/unload patches also maintain the active list
- Keep always-resident groups represented in the compact list
- Preserve current CPU-side vectors during transition for validation

### N.1.3 — Age-filter dispatch over resident groups

- Change the age filter to iterate the compact active list
- Dispatch against `activeResidentGroupCount`, not `totalGroupCount`
- Keep the current full-group path as a debug fallback until validated

Current status: compact dispatch is the default path; a debug toggle can still
switch the GPU age filter back to full-group dispatch for validation.

**Done when:** age-filter cost scales with active resident groups and the result
is visually identical to the current implementation.

---

## Phase N.2 — Remove Per-Frame Full Residency / Age Uploads

**Goal:** Stop treating CPU-side full arrays as the render-path source of truth.

### N.2.1 — Move steady-state residency ownership GPU-side

- Transition `groupResidencyBuffer` toward GPU-owned steady-state data
- Transition `groupAgeBuffer` toward GPU-owned steady-state data
- Keep CPU mirrors only for validation, fallback handling, and UI as needed

Current status: steady-state frames now keep GPU-written residency/age state in
place and reuse it on the next buffered frame; full CPU uploads remain only for
rebuild/reset paths and for frames that missed the GPU age-filter pass.

### N.2.2 — Replace broad uploads with small deltas

- Upload only the per-frame changes:
  - loads
  - unloads
  - touched-bit clears / resets
  - request-queue seeds
- Avoid full memcpy of residency / age arrays once initialization is complete

Current status: steady-state frames now reseed only the request/unload worklist
headers instead of clearing the full payload buffers, and no-update steady-state
frames keep reusing the previous active resident-group table contents. A full
active-list upload still remains on rebuild/reset paths and before applying
queued active-list patch batches.

### N.2.3 — Rebuild-only path

- Keep the existing full canonical upload path only for:
  - scene rebuild
  - pipeline reload reset
  - source-buffer signature changes

**Done when:** steady-state frames no longer memcpy full residency and age arrays.

---

## Phase N.3 — Tighten Request / Readback Resources

**Goal:** Keep the current host-handled load/unload model, but reduce reliance on
large permanently host-visible control buffers.

### N.3.1 — Separate steady-state GPU buffers from readback staging

- Move request payload buffers to device-local where practical
- Use explicit copy-to-readback resources for request download
- Keep frame-tag validation exactly as it is today

### N.3.2 — Preserve fallback behavior

- Keep the current host-visible fallback path available when required by platform or backend limits
- Preserve the CPU FIFO unload fallback if GPU age-filter execution is unavailable

**Done when:** request handling no longer depends on broad host-visible SSBOs in
the common Vulkan path.

---

## Phase N.4 — Reduce CPU-Side Full-Scene Scans

**Goal:** Make telemetry and debug accounting scale with active work instead of scene size.

### N.4.1 — Incremental streaming telemetry

- Derive resident counts from the compact resident table
- Update per-LOD residency incrementally on load/unload
- Cache histogram inputs where possible

### N.4.2 — UI throttling / optional expensive views

- Only rebuild heavier graphs when the debug section is visible
- Allow throttled refresh for age histogram and per-LOD bars on very large scenes

### N.4.3 — Validation mode

- Keep an optional developer-only slow validation pass that cross-checks CPU mirrors against GPU-derived counts

**Done when:** streaming debug views do not require broad per-frame group scans in the common case.

---

## Phase N.5 — Stress Validation & Hardening

**Goal:** Validate the optimized path under pressure before further feature work.

### N.5.1 — Functional validation

- Build and run with streaming on/off
- Verify coarsest-LOD fallback remains correct
- Verify F5 shader reload preserves streaming correctness
- Verify F6 pipeline reload resets tasks and recovers cleanly

### N.5.2 — Pressure validation

- Low storage budget
- High camera motion
- Streaming bursts with many pending groups
- Transfer-queue unavailable fallback
- GPU age-filter missing fallback

### N.5.3 — Telemetry validation

- Ensure resident counts, unload counts, transfer bytes, and age histogram remain sane during stress
- Verify no visual corruption during repeated load/unload churn

**Done when:** optimized non-RT streaming remains stable under low-budget and high-motion conditions.

---

## Implementation Rules

- Keep `enableResidencyStreaming` optional and regression-free
- Do not regress the non-streaming path
- Prefer incremental refactors around `ClusterStreamingService`
- Keep the current fallback behavior until the replacement path is proven
- Validate visually after each phase
- RT / CLAS streaming remains out of scope until this roadmap is exhausted
