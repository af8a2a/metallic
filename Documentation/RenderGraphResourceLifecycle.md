# RenderGraph Resource Lifecycle, Providers, and Cross-Frame Scheduling

This document describes the architecture, contracts, conservative boundaries, and validation results delivered for [Issue #77](https://github.com/af8a2a/metallic/issues/77): **Subsystem Resource Provider, Transparent Resource Versioning, and Cross-Frame Lifecycle Management**.

The design extends Metallic's existing `ResourceRegistry`, `GraphAccessPlan`, `RenderFrameContext`, and `QueueSubmissionTracker` without introducing a second registry, altering Slang shader ABI, or forcing higher VRAM / deeper frame-in-flight defaults.

## 1. Identity, Version, and Ownership Contracts (P0)

Defined in [`RenderGraphResourceLifecycle.h`](../Source/Runtime/Render/RenderGraph/RenderGraphResourceLifecycle.h) and [`RenderGraphResourceLifecycle.cpp`](../Source/Runtime/Render/RenderGraph/RenderGraphResourceLifecycle.cpp):

- **`LogicalResourceId`**: Deterministic 64-bit domain-qualified logical resource identity (`makeLogicalResourceId(domain, key)`), separating logical resource identity from transient native handle addresses.
- **`ResourceVersionId`**: Separates two distinct axes:
  - `contentRevision`: logical content invalidation/edit revision (e.g., environment sun/cloud edit or history invalidation).
  - `allocationGeneration`: physical allocation lifetime generation (increments only when native backing allocations are recreated/evicted/reallocated).
  - `version`: monotonic logical publication/frame version.
- **`RenderGraphResourceOwnership`**: Explicit ownership taxonomy visible in `RenderGraphExecutionSnapshot` and `EditorRenderGraphViewer`:
  - `GraphOwned`
  - `TemporalOwned`
  - `ProviderOwned`
  - `ExternallyOwned`
  - `OpaqueSDK`
- **`FrameExecutionResourceSnapshot`**: Per-execution frozen binding map (`bindLogicalResource`, `bindResourceSet<T>`, `retainOwner`, `freeze()`). Once frozen before pass recording, subsequent `beginFrame(N+1)` calls or cache evictions cannot mutate or release Frame N's bound handles or owners.
- **`RenderGraphExternalDomainContract`**: Explicit contract allowing subsystems and passes with undeclared implicit GPU accesses (bindless/BDA tables, persistent page pools, or third-party SDK state) to declare their domain, ownership, and exact `conservativeReason` surfaced in `RenderGraphExecutionSnapshot::conservativeFallbackReasons`.

## 2. Subsystem Resource Provider & Declarative Environment Resources (P1)

### Provider & Hidden Producer Architecture
- Subsystems can implement **`IRenderGraphResourceProvider`** (`providerSubsystemId()`, optional `externalDomainContract()`) alongside `IRenderSubsystem`.
- Passes declare resource requests in **`RenderGraphPass::declareResourceRequests(const RenderGraphResourceRequestContext&, RenderGraphResourceRequestBuilder&, std::string&)`**, executed in topological pass order before pass recording.
- Upstream passes may publish CPU-side observer metadata (`publishOutputObserver` / `inputObserver`; for example, `VisibilityBufferPass` publishes exact `renderEye` for `"rasterInfo"`).
- Providers register conditional GPU producer work via **`RenderGraphProviderProducerWork`** (deduplicated by `dedupKey`, ordered via `prerequisiteDedupKeys` and `waitCompletions`).
- `RenderGraphExecutor` injects hidden **`RenderGraphSegmentRole::ProviderProducer`** segments on demand without forcing graph template recompilation. Provider cache publications commit **only** in `onAccepted(const GPUCompletionPoint&)` after `Queue::submit` succeeds; failed or cancelled executions invoke `onCancelled()` and discard uncommitted entries cleanly.

### EnvironmentLightingSubsystem Migration
- [`EnvironmentLightingSubsystem`](../Source/Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h) implements `IRenderGraphResourceProvider` and provides declarative **`EnvironmentRequest`** / **`EnvironmentResourceSet`** resolution (`declareEnvironmentRequest`).
- Pure CPU selection/allocation (`prepareEnvironmentResourceSet`, `prepareCelestialResources`) is separated from conditional GPU precompute (`RenderGraphProviderProducerWork`).
- Producer and consumer known accesses (including sampled PDF, filtered specular/prefilter cube, SH/atmosphere parameter buffers, and cloud/weather textures) are registered into the unified **`GraphAccessPlan`** via `trackPublishedResource` / `registerProducerWork`.
- Logical resources are keyed by stable semantic + view identity (`makeLogicalResourceId(domain, name, viewId, subresource)`), so sun/weather/cloud edits advance `contentRevision` while keeping logical identity stable.
- Static atmosphere medium LUTs (`Transmittance`, `MultiScattering`) are owned by an independent **`SharedMediumHandle`** cached in `mediumPublications_` separately from view/sun physical publications (`physicalPublications_`), so evicting an old sky entry releases its full sky textures without keeping evicted sky resources alive solely for the medium LUTs, while preserving the static medium's own `allocationGeneration` and producer completion point across views.
- Both self-submitted (`RenderGraphExecutor::execute(RenderGraphSubmitDesc)`) and standalone external (`RenderGraphExecutor::execute(CommandBuffer&)`) paths unconditionally retain the frozen `FrameExecutionResourceSnapshot` via `CommandBuffer::retainResource()` so referenced backings survive cache eviction until GPU completion even when no `RenderFrameContext` is attached.
- [`ScenePathTracePass`](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp) and [`SceneRTXDIPass`](../Source/Runtime/Render/RenderPass/BuiltinPass/SceneRTXDIPass.cpp) request `EnvironmentRequest` during `declareResourceRequests()` and consume frozen `context.resources<EnvironmentResourceSet>()` during `execute()` with zero cache mutation or `resolveRadiance()`/`updateCelestial()` calls during command recording.

## 3. Declarative Temporal Resources & Selective Cross-Frame Scheduling (P2)

### Declarative Temporal Fields & Configurable History Ring
- `HistoryResourceManager` ([`HistoryResources.h`](../Source/Runtime/Render/Core/HistoryResources.h)) supports configurable ring depth (`initialize(Device&, uint32_t slotCount = 2)`, clamped `2..3`), validates `frameOffset` bounds against the actual slot count before modulo ring mapping (rejecting out-of-range offsets such as `-2` on 2-slot rings and `INT32_MIN`), exposes `backingAllocationId`, `allocationGeneration`, `owner`, `versionId`, and `state` via `textureAtOffset` / `bufferAtOffset`, and preserves `wasValid` on read-only cancellation.
- Passes declare temporal reads/writes directly in `reflect()` via `RenderGraphField::temporalRead(key, frameOffset, graphOwnedHistory, perPassScope)` and `temporalWrite(key, graphOwnedHistory, perPassScope)`. `RenderGraph::validate()` verifies temporal metadata consistency.
- `RenderGraphExecutor` aggregates read/write usage flags per `(manager, historyKey)` before invoking a single `ensureTexture`/`ensureBuffer` per logical key (preventing allocation churn between `Sampled` reads and `ColorAttachment`/`Storage` writes and rejecting incompatible extent/format/depth/sample declarations with an explicit error).
- Temporal publication mutations are recorded per segment as `PendingTemporalPublication` records during CPU recording (never mutating shared history slot state from parallel worker threads) and applied deterministically in submission order by the coordinator inside `submitReady()`. Valid temporal reads trace their `versionId` and backing back to the exact frame that wrote them.
- [`ScenePathTracePass`](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp) accumulation history (`historyPrevious` / `historyCurrent`) is conditionally declared via `needsAccumulationHistory()` only when `accumulate == true` and the active `debugView` accumulates; disabling accumulation allocates zero `RGBA32F` history textures, and re-enabling starts cleanly from invalid history.
- [`LumenRadianceCachePass`](../Source/Runtime/Render/RenderPass/BuiltinPass/LumenRadianceCachePass.cpp) migrated its cross-frame probe cell indirection buffer to declarative graph-owned `.temporalRead("indirection", -1, /*graphOwned=*/true)` (`previousIndirection`) and `.temporalWrite("indirection", /*graphOwned=*/true)` (`savedIndirection`).

### Selective Per-Resource Cross-Frame Hazard Scheduling & Synchronization Boundaries
- **`CrossFrameResourceTracker`** tracks per-`backingAllocationId` accepted cross-frame frontiers (last writer + all active remote-queue readers + layout/queue metadata + per-queue write-visibility tracking), committed **only** inside `submitReady()` after `Queue::submit` returns an accepted receipt.
- **Cross-queue write visibility after host-observed completion**: Resource retirement (backing idle check) and cross-queue memory visibility are evaluated separately. Even after host queries observe that a writer's timeline value has completed (`isComplete() == true`), the first consuming submission on another queue still attaches the writer's semaphore wait (or an equivalent transitive device-side chain on that queue) until the destination queue has observed device-side visibility of the write/layout transition. Both ordinary graph/temporal resources and Provider cache-hit waits follow this rule.
- **Pooled texture external output layout & transactional rollback**: `RenderGraphExecutor::transitionOutput()` updates the tracked layout on the actual pooled backing under a `SubmissionTransaction` so external consumers (such as the Editor viewport sampling `FinalBlit.color` as `ShaderReadOnlyOptimal` under Vulkan optimal layouts) advance pool state on accepted submit and roll back cleanly to the last accepted layout if the external command buffer is cancelled or fails to submit.
- Selective per-resource cross-frame scheduling is **strictly opt-in**: it activates only when every condition is proven for the frame:
  1. `drainReasonMask == 0` (every pass supports frame overlap and no scene revision drain is active).
  2. No opaque subsystem pre/post-graph commands and no attached debug capture observer.
  3. Every pass overrides `supportsCrossFrameResourceScheduling() == true`. Currently audited built-in passes and unlocked production access chains:
     - `ClearColorPass` (Graphics `ColorAttachment` write / clear)
     - `CopyColorPass` (Graphics/Compute `TransferSource` -> `TransferDestination` copy)
     - `RenderGraphBufferWritePass` & `RenderGraphBufferCopyPass` (Compute storage write / transfer copy)
     - `TriangleRasterPass` (Graphics `ColorAttachment` raster write)
     - `ColorGradingLUTPass` (Async Compute 64³ 3D LUT `Storage` write -> `ShaderRead`)
     - `SliderDebugPass` (Async Compute dual-source `ShaderRead` -> `Storage` write)
     - `FinalBlitPass` (Async Compute HDR `ShaderRead` + 3D LUT `ShaderRead` -> `Storage` display write; external `transitionOutput` to `ShaderRead` for viewport presentation)
  4. No pass depends on prepared scene/streaming state, unmigrated subsystems, or any `RenderGraphExternalDomainContract` with `hasUndeclaredImplicitAccesses == true`.
  5. Every accessed resource has a known `backingAllocationId` and does not require `#60` exclusive queue-family ownership transfer (`requiresExclusiveQueueOwnershipTransfer()` checks single-family `QueueAccessBits` / `VK_SHARING_MODE_EXCLUSIVE` and unknown-ownership external imports at executor entry).
- When proven, the executor omits the prior-frame aggregate `lastSubmittedCompletion` wait and attaches exact per-segment cross-frame RAW/WAR/WAW timeline waits (`passCrossFrameWaits` + `waitReasons`), splitting coalesced batches at cross-frame wait boundaries so waits never hoist onto independent branches.
- Within a frame, cross-frame planning handles concurrent multi-queue read fan-out, read-only sharing, and read-then-write / layout-change transitions on the same backing (`writtenInCurrentFrame` + per-queue read tracking). Multi-reader retirement (`isBackingIdle`) requires **every** overlapping reader across all queues to complete before a backing may be retired or overwritten.
- When any condition is not met, the executor falls back conservatively to the prior-frame aggregate completion wait and records every reason in `RenderGraphExecutionStats::conservativeFallbackReasons` and `RenderGraphExecutionSnapshot::conservativeFallbackReasons`.

## 4. Budgeted Physical Version Pool, Viewer Diagnostics & A/B Policies (P3)

### Budgeted Physical Version Pool
- `RenderGraphCompileOptions` exposes:
  - `memoryPolicy` (default: **`RenderGraphMemoryPolicy::MinimizeMemory`**)
  - `maxFramesInFlight` (`1..3`, default: **`2`**)
  - `versionPoolBudgetBytes` (default: `0`)
  - `maxPhysicalVersionsPerResource` (`1..3`, default: `2`)
- **`RenderGraphResourceVersionPool`** manages non-aliased, non-bindless Device graph textures and buffers whose field declares explicit initialization (`Clear` or `FullOverwrite`, not `Unknown`):
  - **`MinimizeMemory` (default)**: Always reuses the single primary backing (`0` extra versions, `0` extra VRAM).
  - **`MaximizeOverlap`**: When the current physical backing is still busy on the GPU according to `CrossFrameResourceTracker`, allocates an extra physical version within `versionPoolBudgetBytes` and `maxPhysicalVersionsPerResource`, retiring replaced backings against all active frontier completions via `DeferredReleaseQueue`.

### Execution Viewer & Separated VRAM Accounting
- `RenderGraphExecutionSnapshot` and [`EditorRenderGraphViewer`](../Source/Editor/EditorRenderGraphViewer.cpp) expose:
  - Named **`ProviderProducer`** segments (`segment.debugName`, e.g., `EnvironmentProvider.PhysicalAtmosphere`) on the Queues tab, tooltips, and JSON export.
  - Per-resource `logicalResourceId`, `versionId`, `contentRevision`, `allocationGeneration`, `physicalVersionCount`, `extraPhysicalVersion`, `ownership`, real Provider `producerName`, and default-expanded `consumerNames` in the Resources tab tooltip and Inspector.
  - Per-batch cross-frame `waitReasons`, frame `memoryPolicy`, `maxFramesInFlight`, `extraVersionCount`, `extraVersionBytes`, `crossFrameResourceScheduled`, and `conservativeFallbackReasons` on the Memory/Queues tabs.
  - **Explicitly separated VRAM accounting scopes** (displayed in the Memory tab and exported in `captureJson`):
    - **Graph-managed VRAM** (`graphManagedVramBytes` / `peakManagedVramBytes`): graph-owned render targets/buffers plus version-pool backings.
    - **Captured Provider scope** (`providerManagedVramBytes`): unique backings bound as `ProviderOwned` in the execution snapshot (e.g., `EnvironmentLightingSubsystem`, `VirtualShadowMapPass` cache, `StreamerSubsystem`).
    - **Captured History scope** (`historyManagedVramBytes`): unique backings bound as `TemporalOwned` in the execution snapshot.
    - **Device-wide RHI VRAM** (`deviceTrackedVramBytes` / `peakDeviceTrackedVramBytes`): all RHI-tracked device allocations (`Device::memoryBudget()`).
    - **Device-wide Driver heap** (`driverDeviceVramBytes` / `peakDriverDeviceVramBytes`, via `VK_EXT_memory_budget`): primary device-local heap usage including opaque third-party SDK allocations (NRD, Streamline/DLSS) and driver internals.

### Release GPU A/B Benchmark Evidence (P3 & Review Batch 3)
Measured on Windows x64 MSVC Release (`build/rg-roadmap`, base revision `e00f252cc` + working tree), independent GPU queues (`graphics family 0`, `compute family 2`), Vulkan validation disabled during timing, warm shader/PSO cache, 16 warmup frames + 80 measured frames per **independent process launch**, strictly following the interleaved A-B-A-B protocol in [`AGENTS.md`](../AGENTS.md).

#### A. Production Built-in Post-Process Graph (`ProdPostProcess`, 1920×1080, Optimal Layouts)
Graph: `TriangleRasterPass` ×2 (`RasterA`, `RasterB` on Graphics) + `ColorGradingLUTPass` (`ACES 2.0` 64³ 3D LUT on Async Compute) + `SliderDebugPass` (1080p Async Compute) + `FinalBlitPass` (1080p Async Compute), invoked via `frame_production_postprocess_policy_benchmark` (`METALLIC_RG_PROD_BENCH_CONFIG`). Raw JSONs and aggregated summary: `build/rg-roadmap/review-final/perf-ab/{expA,expB,summary.txt}`.

##### Experiment A: 2 vs 3 Frames in Flight (`MinimizeMemory`)
Protocol: interleaved `A-B-A-B` extended to **8 independent launches** (4 per variant; runs 1–4 showed clear `avgFrameLatencyMs` improvement `1.007 vs 0.931 ms` while `gpuCrit` was mixed `0.921/0.988 vs 0.930/0.855 ms`, so extended to the 8-run cap and stopped).

| Config (4 independent launches each) | Wall Avg ms (runs) | Wall Mean / p50 / p95 (ms) | CPU Frame Wait Mean (ms) | CPU Drain Frames | GPU Crit Path Mean (ms) | GPU Pass Sum (Gfx / Cmp ms) | GPU Overlap (ms) | Extra Ver | Graph / Device RHI / Driver VRAM (MiB) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `2_minmem` (A) | 0.998, 1.016, 1.112, 0.827 | 0.988 / 0.330 / 4.008 | 0.663 | 0 | 0.931 | 0.106 (0.009 / 0.096) | 0.000 | 0 | 61.06 / 61.80 / 106.08 |
| `3_minmem` (B) | 0.962, 0.899, 0.705, 0.568 | 0.784 / 0.296 / 3.144 | 0.466 | 0 | 0.733 | 0.091 (0.009 / 0.082) | 0.000 | 0 | 61.06 / 62.11 / 106.08 |

- **Outcome**: Every adjacent B beat its paired A on `avgFrameLatencyMs` (`-20.7%` mean, `0.988 -> 0.784 ms`), `avgCpuFrameWaitMs` (`-29.7%`, `0.663 -> 0.466 ms`), and `p95` (`4.008 -> 3.144 ms`), with **0 extra Graph-managed VRAM** (`61.06 MiB` identical; `+0.31 MiB` device RHI frame-slot metadata) and `0` CPU drain frames.

##### Experiment B: `MinimizeMemory` vs `MaximizeOverlap` (3 Frames in Flight, 128 MiB Pool Budget)
Protocol: interleaved `A-B-A-B` extended to **8 independent launches** (4 per variant; runs 1–4 were mixed, stopped at 8-run cap with **no measurable timing difference**).

| Config (4 independent launches each) | Wall Avg ms (runs) | Wall Mean / p50 / p95 (ms) | CPU Frame Wait Mean (ms) | CPU Drain Frames | GPU Crit Path Mean (ms) | GPU Pass Sum (Gfx / Cmp ms) | GPU Overlap (ms) | Extra Ver | Graph / Device RHI / Driver VRAM (MiB) |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `3_minmem` (A) | 0.642, 0.781, 0.719, 0.814 | 0.739 / 0.307 / 3.220 | 0.408 | 0 | 0.684 | 0.090 (0.010 / 0.080) | 0.000 | 0 | 61.06 / 62.11 / 106.08 |
| `3_maxoverlap` (B) | 0.695, 0.687, 0.861, 0.750 | 0.748 / 0.319 / 3.542 | 0.410 | 0 | 0.668 | 0.110 (0.010 / 0.100) | 0.000 | 8 | 179.19 / 180.23 / 224.20 |

- **Outcome**: **No measurable timing difference** (`0.739 vs 0.748 ms` wallAvg, `0.684 vs 0.668 ms` gpuCrit; distributions fully overlap), because total GPU pass time (`~0.10 ms`) is shorter than cross-queue submission/sync latency (`~0.67 ms`), while `MaximizeOverlap` deterministically costs **`+118.13 MiB` Graph-managed VRAM** (`61.06 -> 179.19 MiB`, `extraVer = 8`).

#### B. Heavy Multi-Queue Synthetic Stress (`CrossFramePolicyBench`, P3)
Workload: `ComputeStage` (524,288-element ALU compute pass writing 2 MiB `stage`) -> `GraphicsStage` (copying `stage` + 8× `2048×2048 RGBA16Sfloat` color clears on 32 MiB `color`; 16 warmup + 80 measured frames per launch).

##### Experiment 1: `MinimizeMemory` vs `MaximizeOverlap` (3 Frames in Flight, 64 MiB Pool Budget)
Protocol: interleaved A-B-A-B (4 independent runs; stopped at 4 runs because every B was strictly below every A and the mean wall-time gap `0.1430 ms` exceeded within-variant spread `0.0885 ms`).

| Run | Config | Wall ms/frame | GPU Envelope ms/frame | CPU ms/frame | Extra Versions | Extra VRAM | Peak Managed VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `3_crossframe_minmem` (A) | 1.1198 | 1.0370 | 0.3869 | 0 | 0 B | 36.00 MiB |
| 1 | `3_crossframe_maxoverlap` (B) | 0.9515 | 0.8172 | 0.3923 | 2 | 34.00 MiB | 70.00 MiB |
| 2 | `3_crossframe_minmem` (A) | 1.1576 | 1.0298 | 0.3774 | 0 | 0 B | 36.00 MiB |
| 3 | `3_crossframe_maxoverlap` (B) | 1.0400 | 0.9815 | 0.3996 | 2 | 34.00 MiB | 70.00 MiB |
| **Mean** | **A vs B** | **1.1387 vs 0.9958 (-12.6%)** | **1.0334 vs 0.8994 (-13.0%)** | **0.3822 vs 0.3960** | **0 vs 2** | **0 vs +34.00 MiB** | **36.00 vs 70.00 MiB** |

##### Experiment 2: 2 vs 3 Frames in Flight (`MinimizeMemory`, Heavy Workload)
Protocol: interleaved A-B-A-B extended to 8 independent runs because the first 4 runs were mixed; stopped at 8 runs with **no measurable difference** (A `1.1285–1.2685 ms` vs B `0.9888–1.2203 ms`, mean gap `0.0795 ms` < within-variant spread `0.2315 ms`), because single-backing cross-frame WAR waits cap GPU overlap on heavy backings.

**Default Policy Decision**: Keep **`MinimizeMemory`** and **`maxFramesInFlight = 2`** as the conservative default. Both Experiment B (real light post-process graph: `+118.13 MiB` VRAM for 0 gain) and Experiment 1 (heavy synthetic multi-queue graph: `-12.6%` wall / `-13.0%` GPU envelope for `+34.00 MiB` VRAM) confirm `MaximizeOverlap` should remain an explicit opt-in with a hard byte budget (`versionPoolBudgetBytes`).

## 5. Incremental Domain Migrations & Conservative Boundaries (P4 + Review Batch 3)

Declarative resource migrations and explicit conservative boundaries (`RenderGraphExternalDomainContract`):

| Component | Domain / Resources | Owner | Producer | Consumer(s) | Retirement Condition | Conservative Boundary & Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| [`EnvironmentLightingSubsystem`](../Source/Runtime/Render/Subsystem/EnvironmentLightingSubsystem.cpp) | `render.environment.*` (atmosphere LUTs, radiance, PDF, SH, prefilter, clouds, celestial) | `ProviderOwned` | Hidden `ProviderProducer` (`EnvironmentLightingSubsystem`) | `ScenePathTracePass`, `SceneRTXDIPass`, Cloud consumers | `FrameExecutionResourceSnapshot` retained via `CommandBuffer::retainResource()` & `RenderFrameContext` until all reader completions retire; static medium LUTs independently held by `SharedMediumHandle` | Fully declarative for migrated consumers; verified by `physical_environment_provider_full_lifecycle_and_clouds`, `physical_environment_external_cmdbuf_retention_on_eviction`, `render_graph_viewer_real_provider_and_memory_scopes` |
| [`LumenRadianceCachePass`](../Source/Runtime/Render/RenderPass/BuiltinPass/LumenRadianceCachePass.cpp) | `previousIndirection` / `savedIndirection` (`indirection` temporal ring) | `TemporalOwned` (graph-owned ring) | `LumenRadianceCachePass` (`lumenSaveRadianceCacheIndirectionMain`) | Next frame `LumenRadianceCachePass` (`lumenRemapRadianceCacheProbesMain`) | `HistoryResourceManager` ring retention + `DeferredReleaseQueue` on resize/invalidation/reallocation | Remaining in-place persistent radiance atlas & indirect probe buffers keep `render.pass.lumen-radiance-cache` (`ExternallyOwned`, conservative); verified on 3-frame CornellBox in `frame_migrated_lumen_vsm_streamer_domains` (`savedF0=1 -> prevF1=1 -> savedF1=2`, 1,019 finite non-zero `cacheRadiance` texels) |
| [`VirtualShadowMapPass`](../Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapPass.cpp) & [`VirtualShadowMapLightsPass`](../Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapLightsPass.cpp) | `render.vsm.atlas`, `render.vsm.pageTable`, `render.vsm.projections` | `ProviderOwned` (persistent cache) / `ExternallyOwned` (pass domain) | `VirtualShadowMapPass` (committed via `SubmissionTransaction` on accepted submit) | Next-frame `VirtualShadowMapPass` cache reuse & downstream shadow consumers | `shared_ptr` cache owner retained by `FrameExecutionResourceSnapshot` + `SubmissionTransaction` rollback on cancel/device-change | In-place VSM page raster & HZB remain conservative (`render.pass.virtual-shadow-map` & `render.pass.virtual-shadow-map-lights`); verified in `frame_migrated_lumen_vsm_streamer_domains`, `virtual_shadow_map_scene`, `virtual_shadow_map_temporal_lifecycle` |
| [`StreamerSubsystem`](../Source/Runtime/Render/Streamer/StreamerSubsystem.cpp) | `render.streamer.materials`, `render.streamer.materialValues`, `render.streamer.pages`, `render.streamer.pageTable` | `ProviderOwned` | `StreamerSubsystem` (`IRenderGraphResourceProvider::declarePreparedSceneResources`) | All passes with `preparedScene != nullptr` (`VisibilityBufferPass`, `ScenePathTracePass`, `VirtualShadowMapPass`, etc.) | `prepared.snapshot->pathTraceResources` & `prepared.geometry` owners retained in `FrameExecutionResourceSnapshot` until GPU completion | Dynamic streaming page-feedback uploads & bindless tables remain conservative (`render.streamer`, `hasUndeclaredImplicitAccesses = true`); verified in `frame_production_graph_lifecycle_stress`, `frame_migrated_lumen_vsm_streamer_domains`, `streamer_render_graph_flush` |
| [`GPUSceneSubsystem`](../Source/Runtime/Render/Subsystem/GPUSceneSubsystem.h) | `render.gpu-scene` | `ExternallyOwned` | `GPUSceneSubsystem` | Scene/raster/RT passes | Subsystem frame-slot retirement | Bindless/BDA geometry, visibility streams, and light tables remain conservative |
| [`NRDDenoisePass`](../Source/Runtime/Render/RenderPass/BuiltinPass/NRDDenoisePass.cpp) | `render.sdk.nrd` | `OpaqueSDK` | NVIDIA NRD SDK | Post-process / compositing passes | Opaque SDK lifetime | Third-party SDK opaque temporal history and internal barriers remain conservative (`OpaqueSDK`) |
| [`StreamlineDLSSPass`](../Source/Runtime/Render/RenderPass/BuiltinPass/StreamlineDLSSRRPass.cpp) | `render.sdk.dlss-sr` / `render.sdk.dlss-rr` | `OpaqueSDK` | NVIDIA Streamline / DLSS SDK | Post-process / presentation passes | Opaque SDK lifetime | Third-party SDK opaque temporal state remains conservative (`OpaqueSDK`) |

### Explicit Synchronization Limits (`#58` and `#60`)
- **`#58` Same-queue writer visibility**: `CrossFrameResourceTracker` and `GraphAccessPlan` never assume a narrower prior reader's barrier covers a subsequent wider-stage consumer on the same layout; verified end-to-end by `frame_same_layout_read_fanout_writer_visibility_58` (Compute write -> Compute narrow read -> Fragment read on both optimal and unified layouts).
- **`#60` Exclusive queue-family ownership**: `RenderGraphExecutor` checks `requiresExclusiveQueueOwnershipTransfer()` at execution entry for single-family `QueueAccessBits` (`VK_SHARING_MODE_EXCLUSIVE`) and unknown-ownership external imports, preserving conservative fallback and reporting the exact reason in `conservativeFallbackReasons` and Viewer (`frame_exclusive_queue_ownership_boundary`).
- Zero `VK_LAYER_VALIDATE_SYNC=1` errors does **not** prove dynamic bindless/BDA coverage or `#58`/`#60` completeness; conservative contracts remain the primary safety boundary.

## 6. Validation Summary

Validated on Windows x64 MSVC 14.52.37025 Release (`build/rg-roadmap`, git base `e00f252cc` + review-fix working tree, real independent Graphics family 0 + Compute family 2 queues). All GPU RHI runs used `VK_LAYER_VALIDATE_SYNC=1 --rhi-validation` under **both** Vulkan optimal image layouts (`--rhi-optimal-layouts`, plus private-device `preferUnifiedImageLayouts = false`) **and** default unified image layouts, with **0 `VUID-` / `SYNC-HAZARD` / `Validation Error` messages**:

- **Slang Module Audit**: `python -B Tools/CheckShaderModules.py` -> passed (`480 files, 313 modules, 167 implementation units`).
- **Task Runtime**: `MetallicTaskTests.exe` -> **35/35 PASSED** (`build/rg-roadmap/review-final/task-tests.log`).
- **Batch 1–3 Complete Optimal-Layout RHI Validation**: **41/41 runnable tests PASSED**, **0 VUID / 0 SYNC-HAZARD** (`build/rg-roadmap/review-final/full-review-optimal.log` [36/36 runnable] + `build/rg-roadmap/review-final/batch1-extra-optimal.log` [5/5]).
- **Batch 1–3 Complete Unified-Layout RHI Validation**: **41/41 runnable tests PASSED**, **0 VUID / 0 SYNC-HAZARD** (`build/rg-roadmap/review-final/full-review-unified.log` [36/36] + `build/rg-roadmap/review-final/batch1-extra-unified.log` [5/5]).
- **Visual Inspection**: Inspected Viewer renders (`build/rg-roadmap/review-final/out-optimal/render_graph_viewer_real_provider_{queues,resources,memory}.png`) confirming named `ProviderProducer` (`EnvironmentProvider`), real Provider producer/consumers/version/backing, and separated Graph-managed / Provider / History / Device RHI / Driver VRAM scopes.
- **Production-Graph A-B-A-B Perf Experiments**: 16 isolated independent-process launches (8 ExpA + 8 ExpB) under `build/rg-roadmap/review-final/perf-ab/`.
- **Editor Multi-Frame Smoke Test**: `.\build\rg-roadmap\Source\Metallic.exe --smoke-test` -> exit code `0` (`build/rg-roadmap/review-final/editor-smoke-test.log`).
- **Explicitly Skipped / Unavailable Checks**:
  - `NRDWorkingColor.PhysicalAtmosphereHDRAndAerial` (`MetallicNRDTests`): **Not run** — `build/rg-roadmap` has `METALLIC_ENABLE_NRD=OFF`, and the only pre-existing NRD-enabled directory (`build-pass-stages-nrd`) targets uninstalled MSVC `14.51.36231` (host has `14.52.37025`). Per `AGENTS.md`, existing build directories were not reconfigured.
  - `render_graph_sharc_stages_history_and_cancel`: **Failed (missing external asset)** — requires local `Asset/meet_mat.glb` not present in checkout (`physical_environment_sharc_source_and_history` PASSED on available assets).
  - `streamed_realtime_pipeline`: **Skipped** — requires `--rhi-realtime` + DLSS-SR (`METALLIC_HAS_STREAMLINE=0`).
  - `lumen_single_bounce_realtime`: **Pre-existing unrelated shader compile error** in `Shaders/Features/DistanceField/LumenScreenProbeGather.slang:560` (`undefined identifier 'lumenProbeTraceAddress'`); `LumenRadianceCachePass` temporal migration was directly GPU-verified on CornellBox via `frame_migrated_lumen_vsm_streamer_domains` (pruning `Gather`).
