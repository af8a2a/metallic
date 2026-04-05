# Metallic Framework Alignment Roadmap

This roadmap compares `E:/metallic` against `E:/NRI` and `E:/nvpro_core2`, and defines the framework work that should be aligned before aggressively implementing modern graphics techniques such as new Vulkan extensions, full ray tracing pipelines, GI, and large-scale GPU-driven rendering.

## Goal

Before stacking advanced rendering features, bring Metallic to a stable "modern graphics platform" baseline:

- capability-first device and feature model
- precise synchronization and resource state tracking
- multi-queue scheduling
- scalable descriptor/bindless strategy
- mature pipeline cache and shader artifact flow
- full Vulkan RT platform layer
- robust transient/upload/readback memory systems
- stronger profiling/debug tooling

---

## Current baseline summary

Metallic already has strong starting points:

- Vulkan + Metal RHI
- VMA-backed Vulkan resources
- Slang-based shader compilation with reflection-driven layout generation
- FrameGraph/pass system with transient and history resources
- partial bindless support
- BLAS/TLAS support
- mesh shader and GPU-driven visibility groundwork
- HZB and meshlet culling groundwork
- shader hot reload and profiling hooks

Main gaps versus NRI / nvpro_core2 quality:

- limited device capability / limit exposure
- coarse barrier and hazard tracking
- weak multi-queue support
- no full RT pipeline + SBT layer yet
- no mature PSO / persistent pipeline cache system
- transient memory aliasing is not yet a first-class allocator
- upload/readback/streaming systems are still utility-like
- debug/profiling/crash tooling can be much stronger

---

## Phase 0 — Platform hardening (must complete first)

These are the highest-priority platform items. They should be completed before serious work on new Vulkan extensions, full RT, or GI.

### 0.1 Capability / feature / limit system

**Target:** make device capability reporting complete, centralized, and queryable by all higher-level systems.

Add a unified capability layer that exposes:

- Vulkan API version
- enabled extensions
- Vulkan 1.1 / 1.2 / 1.3 / future feature structs
- descriptor indexing capabilities and limits
- bindless capacities and policies
- mesh/task shader capabilities and limits
- ray tracing properties and limits
- queue family capabilities
- subgroup / wave properties
- optional modern features such as descriptor buffer, shader object, maintenance extensions, timeline semaphore, etc.

**Why:** NRI and nvpro_core2 both treat feature negotiation as a first-class subsystem. Metallic should stop scattering feature assumptions across renderer code.

**Outcome:** all advanced systems can branch on one canonical capability model.

### 0.2 Resource state tracking and barrier system

**Target:** replace coarse transitions with a modern hazard-aware synchronization layer.

Expand the current model into:

- precise image + buffer access state tracking
- read/write/access/stage/layout modeling
- subresource range support where needed
- barrier synthesis from pass/resource usage
- queue ownership transfer support
- validation/debug views for hazards and transitions

**Why:** this is the core prerequisite for async compute, copy queues, transient aliasing, RT build/trace overlap, and scalable GPU-driven pipelines.

**Outcome:** Metallic moves from “mostly works” synchronization to “framework-grade” correctness and scalability.

### 0.3 Multi-queue execution model

**Target:** support graphics, compute, and copy queues as real scheduling resources.

Add:

- queue family selection strategy
- command pools per queue type
- timeline semaphore-based submission model
- queue-aware pass/task scheduling hooks
- ownership transfer rules and helpers

**Why:** modern rendering features expect async upload, async compute, and queue overlap.

**Outcome:** enables future async HZB, async culling, texture streaming, RTAS build overlap, denoisers, and GI pipelines.

### 0.4 PSO / pipeline cache / shader cache system

**Target:** make pipeline compilation scalable and persistent.

Add:

- Vulkan `VkPipelineCache` persistence
- engine-level PSO key/hash system
- central pipeline cache manager
- shader variant/permutation management
- optional background compilation / warmup support
- compile telemetry and diagnostics

**Why:** advanced rendering work multiplies pipeline count quickly.

**Outcome:** reduced hitching, better stability for feature growth, easier debugging of shader/pipeline state.

### 0.5 Full Vulkan RT platform layer

**Target:** elevate current BLAS/TLAS support into a full ray tracing platform.

Add:

- ray tracing pipeline abstraction
- shader group management
- SBT generation and allocation
- `vkCmdTraceRaysKHR` dispatch support
- RT pipeline cache integration
- extension points for future RT features

**Why:** AS support alone is not enough for modern RT GI / reflections / path tracing work.

**Outcome:** Metallic becomes RT-platform-ready rather than only RTAS-ready.

### 0.6 Transient / upload / readback memory systems

**Target:** make short-lived and streaming resource management first-class.

Add:

- transient texture pool
- transient buffer arena
- upload ring / upload heap
- readback heap
- batched staging uploader
- timeline-based recycle and retirement
- aliasing of non-overlapping transient resources

**Why:** GI, denoising, GPU-driven rendering, and RT all create heavy transient memory pressure.

**Outcome:** better memory efficiency and cleaner frame orchestration.

---

## Phase 1 — Core systems alignment

These items should follow immediately after Phase 0, or be developed in parallel when dependencies allow.

### 1.1 Descriptor / bindless strategy hardening

**Target:** move from partial bindless support to a long-term resource binding model.

Decide and implement a stable path:

- continue descriptor indexing with stronger global bindless tables, or
- incrementally adopt `VK_EXT_descriptor_buffer`

In either case, add:

- stable global resource index allocation
- lifetime tracking for bindless entries
- stronger capacity management
- support for partially bound and variable-size ranges where useful
- clear rules for sampled/storage/AS descriptors

**Why:** GPU-driven rendering, modern material systems, and RT all depend on stable large-scale resource binding.

### 1.2 Upload / streaming / readback service layer

**Target:** turn utility uploads into a formal service layer.

Add:

- one-shot upload helpers for asset creation
- persistent per-frame upload system
- async transfer path when available
- explicit readback APIs for stats, debug, capture, and GPU-driven feedback
- fence/timeline tracked CPU visibility

**Why:** this area is much more mature in NRI and nvpro_core2 than in Metallic.

### 1.3 Debug / profiling / diagnostics layer

**Target:** build stronger operational tooling into the graphics platform.

Add:

- Vulkan debug names and markers throughout resources/passes/pipelines
- GPU timestamp query abstraction
- pass/pipeline timing support
- optional pipeline statistics
- better shader compile diagnostics
- device lost handling path
- optional integration hooks for RenderDoc / Nsight / Aftermath-class tooling

**Why:** advanced rendering work becomes expensive to debug without this layer.

### 1.4 Shader artifact / reflection / variant flow

**Target:** strengthen the Slang-based pipeline into a scalable artifact system.

Add:

- offline shader cache where appropriate
- persisted reflection artifacts or generated metadata
- specialization/variant management
- debug vs optimized compile modes
- dependency tracking for reloads and recompilation

**Why:** Slang is already a strong choice; the missing part is cache/permutation/tooling maturity.

### 1.5 Clear architectural layering

**Target:** separate platform core from helpers and renderer policy.

Recommended split:

- **Core:** device, queues, sync, resources, descriptors, pipelines, RT primitives
- **Helpers:** upload, readback, transient allocators, shader cache, profiling helpers
- **Renderer:** FrameGraph, passes, HZB, culling, GI, denoisers, scene systems

**Why:** this matches the strongest lessons from NRI and nvpro_core2 and keeps future extension work manageable.

---

## Phase 2 — GPU-driven rendering scale-up

After Phase 0 is stable, Phase 2 should stop being framed as “add more GPU-driven techniques” and instead become a concrete migration path from Metallic’s current flat meshlet-cull pipeline toward a `vk_lod_clusters`-style GPU scene traversal model.

### Current delta vs `vk_lod_clusters`

Metallic already has useful groundwork:

- GPU meshlet cull -> indirect visibility draw path
- HZB build + previous-frame occlusion sampling
- offline meshlet generation and offline `ClusterLODData` build/cache
- FrameGraph integration for visibility/depth/HZB passes

But compared with `vk_lod_clusters`, Metallic is still missing the runtime pieces that make the path scale:

- no explicit GPU scene tables for instances / geometries / LOD nodes / LOD groups
- no GPU-side instance classification stage before meshlet work expansion
- no traversal pass that consumes `ClusterLODData` and emits group/cluster worklists
- no resident-address / scene patching model for future streaming
- visibility still begins from a CPU-provided `visibleMeshletNodes` list rather than a GPU-owned scene traversal result

This means Phase 2 should be executed in the following milestones and not skipped ahead.

### 2.1 Milestone A — Freeze a GPU scene ABI

**Target:** establish the runtime data model that all later culling, traversal, shading, and RT-facing systems consume.

Add:

- GPU instance table with stable instance IDs, transforms, geometry/material indirection, and visibility flags
- GPU geometry table with meshlet ranges, primitive-group ranges, bounds, and `lodRootNode` handles
- GPU LOD hierarchy tables backed by the existing `ClusterLODData` payload (`nodes`, `groups`, `groupMeshletIndices`, `levels`)
- stable material/texture/resource indices shared across culling and shading
- explicit CPU->GPU upload/update ownership rules

**Why:** `vk_lod_clusters` works because traversal, culling, rasterization, and streaming all speak the same GPU scene schema. Metallic already has much of the data, but not the runtime ABI.

**Done when:** visibility and rendering passes can consume shared scene tables without pass-local CPU packing.

### 2.2 Milestone B — Generalize indirect and worklist building

**Target:** promote the current counter/compaction/indirect-args path into a reusable subsystem.

Generalize:

- append/consume worklists
- counter + prefix-sum patterns
- draw/dispatch argument writers
- typed indirect buffers for mesh/task/compute work
- reusable clear/reset/seed helpers
- stats buffers for produced and consumed work items

**Why:** Metallic already has the seed of this in meshlet culling and indirect argument generation, but it is still pass-specific. `vk_lod_clusters` shows that traversal, LOD selection, raster setup, and streaming all need the same primitives.

**Done when:** new compute passes can produce draw/dispatch work without adding one-off counter layouts or CPU-side fixups.

### 2.3 Milestone C — Add an instance/object classification front-end

**Target:** stop expanding meshlet work from a CPU-assembled visible-node list.

Add:

- GPU instance classification pass sourced from scene tables
- instance frustum culling
- optional coarse HZB/occlusion test at instance bounds level
- optional distance/error classification inputs for later LOD selection
- visible-instance worklist as the only input to downstream meshlet/cluster stages

**Why:** `vk_lod_clusters` performs instance classification and traversal seeding before cluster traversal. Metallic currently jumps from CPU `visibleMeshletNodes` to per-meshlet GPU culling, which limits scale and keeps the CPU in the visibility loop.

**Done when:** `visibleMeshletNodes` is no longer the primary visibility source for the GPU-driven path.

### 2.4 Milestone D — Make `ClusterLODData` executable at runtime

**Target:** convert the existing offline cluster LOD builder from “data exists” into “renderer traverses it”.

Add:

- GPU traversal over `lodRootNode -> nodes -> groups -> meshlets`
- a stable screen-space or error-over-distance metric for LOD selection
- output worklists for selected groups or selected meshlets
- fallback path for nodes without LOD data
- debug views for chosen LOD level, selected group counts, and meshlet expansion

**Why:** Metallic already builds and caches cluster LOD data, but it is not yet used by runtime visibility. `vk_lod_clusters` demonstrates the missing piece: hierarchical traversal that emits renderable cluster/group work entirely on the GPU.

**Done when:** far and near objects select different cluster LOD levels without CPU-side per-object LOD decisions.

### 2.5 Milestone E — Integrate HZB with traversal, not just flat meshlet culling

**Target:** move HZB from an add-on to a first-class contract in the GPU-driven pipeline.

Add:

- explicit HZB input contract for instance and group traversal stages
- previous-frame reprojection rules shared by traversal/culling shaders
- optional two-stage culling where instance/group coarse reject runs before meshlet fine reject
- debug/stat outputs for instance -> group -> meshlet reduction ratios
- async compute evaluation only after correctness and diagnostics are stable

**Why:** current Metallic HZB usage is already useful, but it is attached to the flat meshlet pass. `vk_lod_clusters` uses culling to steer traversal itself, which is the scalable direction.

**Done when:** HZB reduces traversal/render work before the final meshlet expansion stage, and the reduction is visible in pass stats.

### 2.6 Milestone F — Finish the fully resident raster path before streaming

**Target:** reach a clean “fully resident GPU-driven cluster LOD rasterizer” before introducing residency complexity.

Finish:

- draw-indirect path sourced only from GPU-produced traversal results
- no CPU per-node dispatch fallback on the main path
- stable debug/readback counters for selected instances, groups, meshlets, and rendered work
- performance baselines on representative scenes

**Why:** `vk_lod_clusters` includes streaming and RT-specific complexity, but Metallic should first lock the resident-data raster path. Otherwise debugging LOD traversal, HZB, and residency at the same time will be too expensive.

**Done when:** a fully resident scene can render through GPU traversal + indirect draw with predictable counters and without CPU visibility orchestration.

### 2.7 Milestone G — Optional residency / streaming layer

**Target:** only after Milestone F, add group-level residency inspired by `vk_lod_clusters`.

Add:

- lowest-detail-always-resident policy
- invalid-address / request-state encoding for non-resident groups
- host/device request queue for load/unload decisions
- scene patch/update pass that makes loaded groups visible without re-uploading the whole scene
- memory budget and eviction rules
- optional async transfer queue integration

**Why:** the reference sample proves this is the correct scaling model for very large scenes, but it should be a late Phase 2 milestone for Metallic, not the entry point.

**Done when:** Metallic can cap VRAM usage and progressively refine cluster groups without breaking traversal correctness.

### 2.8 Future-facing optional features

Only after the above is stable, evaluate:

- more advanced descriptor models
- device-generated command paths
- work-graph-style execution models when relevant extensions/toolchains mature

### Implementation rules for Phase 2

- do not skip directly from flat meshlet culling to streaming; land resident traversal first
- do not keep adding pass-local CPU packing once the GPU scene ABI exists; migrate passes onto shared scene tables
- do not require per-frame host readbacks to determine the next render workload; statistics readbacks are acceptable, control-path readbacks are not
- reuse the existing offline `ClusterLODData` build/cache pipeline before inventing new geometry preprocessing formats
- keep Vulkan-first GPU-driven execution portable enough that a later Metal fallback can either use the same scene ABI or cleanly opt out at the pass level

---

## Phase 3 — Ray tracing platform maturation

Once Phase 0 RT work is done, expand from “RT capable” to “RT ready for multiple features”.

### 3.1 RT pipeline library

Create a reusable RT system supporting:

- raygen / miss / hit group composition
- reusable shader group definitions
- shared SBT infrastructure
- pipeline specialization and cache reuse

### 3.2 AS lifecycle improvements

Strengthen:

- scratch buffer allocation/reuse
- compaction support
- update vs rebuild policies
- scene-to-AS translation layer
- profiling for AS build/update costs

### 3.3 RT feature landing zone

After the RT platform is mature, implement features in this order:

- improved RT shadows
- RT reflections
- hybrid lighting experiments
- probe tracing / DDGI-like systems
- more advanced GI or path tracing work

**Important:** do not jump to large GI systems before the underlying RT platform and transient memory systems are stable.

---

## Phase 4 — GI and modern extension adoption

This phase starts only after the platform work above is complete enough that new features do not require local hacks in the backend.

### 4.1 New Vulkan extension adoption model

Adopt new extensions through the capability system rather than feature-specific patches.

All new extension work should go through:

- capability reporting
- backend enablement
- abstraction exposure
- debug/profiling integration
- fallback or capability gating rules

### 4.2 GI prerequisites checklist

Before implementing GI systems, confirm the following are already solid:

- RT pipeline + SBT or equivalent compute tracing foundation
- robust transient resource allocator
- stable descriptor/bindless scene model
- async compute support
- precise barrier tracking
- pipeline/shader cache maturity
- strong readback/debug/profiling support

### 4.3 Candidate GI work after prerequisites

Only after the checklist is satisfied, consider:

- DDGI / probe GI
- temporal GI
- hybrid RT GI
- reservoir-based lighting experiments
- denoiser-heavy pipelines

---

## Prioritized short list

If only the most important items are tackled first, the order should be:

1. Capability / feature / limit system
2. Resource state tracking and barrier system
3. Multi-queue scheduling model
4. PSO / pipeline cache / shader cache system
5. Full RT pipeline + SBT platform layer
6. Transient / upload / readback memory systems

This is the minimum alignment set before scaling into modern graphics features aggressively.

---

## Readiness gates by feature area

### Before new Vulkan extensions

Must have:

- centralized capability model
- extension enablement policy
- precise synchronization model
- pipeline/shader cache
- profiling/debug support

### Before full ray tracing

Must have:

- RT pipeline + SBT
- RT capability/properties exposure
- AS lifecycle policy
- stable descriptor/bindless model
- upload/readback/debug support
- multi-queue synchronization

### Before GI

Must have:

- stable RT or compute tracing platform
- transient allocator maturity
- async compute
- stable scene resource model
- strong debug/profiling/readback support
- pipeline/shader cache maturity

### Before large-scale GPU-driven rendering

Must have:

- stable bindless model
- GPU scene ABI with stable instance / geometry / material / LOD tables
- generalized indirect workflow
- runtime traversal of `ClusterLODData` or equivalent hierarchy
- strong barrier/state tracking
- transient buffer allocator
- async compute support
- integrated HZB/visibility pipeline
- fully resident traversal + indirect raster path validated before streaming

---

## Final guidance

Metallic should not treat NRI or nvpro_core2 as code to imitate one-for-one. The useful alignment is architectural:

- learn **capability-first abstraction** from NRI
- learn **modern Vulkan runtime discipline** from nvpro_core2
- keep Metallic’s own strengths: Slang, FrameGraph, multi-backend RHI, visibility/HZB path, and research-friendly renderer structure

The key strategic rule is:

> first align the platform layer, then the scheduling/resource layer, and only then scale up advanced rendering features.

That order will keep future Vulkan extension work, RT, GI, and GPU-driven rendering additive instead of destabilizing.
