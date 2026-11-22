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
- Static atmosphere medium LUTs (`Transmittance`, `MultiScattering`) are cached in `mediumPublications_` independently of view/sun physical publications (`physicalPublications_`), enabling:
  - Same-frame cross-view sharing of static medium LUTs via prerequisite producer ordering (`sharesMedium()`).
  - Survival of static medium allocations across physical environment cache eviction (`contentRevision` increments independently of static medium `allocationGeneration`).
- [`ScenePathTracePass`](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp) and [`SceneRTXDIPass`](../Source/Runtime/Render/RenderPass/BuiltinPass/SceneRTXDIPass.cpp) request `EnvironmentRequest` during `declareResourceRequests()` and consume frozen `context.resources<EnvironmentResourceSet>()` during `execute()` with zero cache mutation or `resolveRadiance()`/`updateCelestial()` calls during command recording.

## 3. Declarative Temporal Resources & Selective Cross-Frame Scheduling (P2)

### Declarative Temporal Fields & Configurable History Ring
- `HistoryResourceManager` ([`HistoryResources.h`](../Source/Runtime/Render/Core/HistoryResources.h)) supports configurable ring depth (`initialize(Device&, uint32_t slotCount = 2)`, clamped `2..3`), offset queries (`textureAtOffset`, `bufferAtOffset`) exposing `backingAllocationId`, `allocationGeneration`, `owner`, and `state`, and offset-aware state publication (`publishTextureStateAtOffset`, `publishBufferStateAtOffset`) that preserves `wasValid` on read-only cancellation.
- Passes declare temporal reads/writes directly in `reflect()` via `RenderGraphField::temporalRead(key, frameOffset, graphOwnedHistory, perPassScope)` and `temporalWrite(key, graphOwnedHistory, perPassScope)`. `RenderGraph::validate()` verifies temporal metadata consistency.
- `RenderGraphExecutor` resolves temporal slots in `prepareTemporalResources()` (after `subsystemHost->beginFrame()` processes `InvalidateTemporalHistory`), registers them into the unified `GraphAccessPlan` with `RenderGraphResourceOwnership::TemporalOwned`, and publishes final state/validity via `finalizeTemporalPublications()`.
- [`ScenePathTracePass`](../Source/Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp) accumulation history (`historyPrevious` / `historyCurrent`) is migrated to declarative `.temporalRead("accumulation", -1)` and `.temporalWrite("accumulation")`.

### Selective Per-Resource Cross-Frame Hazard Scheduling
- **`CrossFrameResourceTracker`** tracks per-`backingAllocationId` accepted cross-frame frontiers (last writer + all active remote-queue readers + layout/queue metadata), committed **only** inside `submitReady()` after `Queue::submit` returns an accepted receipt.
- Selective per-resource cross-frame scheduling is **strictly opt-in**: it activates only when every condition is proven for the frame:
  1. `drainReasonMask == 0` (every pass supports frame overlap and no scene revision drain is active).
  2. No opaque subsystem pre/post-graph commands and no attached debug capture observer.
  3. Every pass overrides `supportsCrossFrameResourceScheduling() == true` (audited built-ins: `ClearColorPass`, `CopyColorPass`, `RenderGraphBufferWritePass`, `RenderGraphBufferCopyPass`).
  4. No pass depends on prepared scene/streaming state, unmigrated subsystems, or any `RenderGraphExternalDomainContract` with `hasUndeclaredImplicitAccesses == true`.
  5. Every accessed resource has a known `backingAllocationId` and does not require `#60` exclusive queue-family ownership transfer.
- When proven, the executor omits the prior-frame aggregate `lastSubmittedCompletion` wait and attaches exact per-segment cross-frame RAW/WAR/WAW timeline waits (`passCrossFrameWaits` + `waitReasons`), splitting coalesced batches at cross-frame wait boundaries so waits never hoist onto independent branches.
- Within a frame, cross-frame planning handles concurrent multi-queue read fan-out and read-then-write / layout-change transitions on the same backing (`writtenInCurrentFrame` + per-queue read tracking).
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

### Execution Viewer & Diagnostics
- `RenderGraphExecutionSnapshot` and [`EditorRenderGraphViewer`](../Source/Editor/EditorRenderGraphViewer.cpp) expose:
  - `ProviderProducer` segment role on the Queues tab.
  - Per-resource `logicalResourceId`, `versionId`, `contentRevision`, `allocationGeneration`, `physicalVersionCount`, `extraPhysicalVersion`, `ownership`, `producerName`, and `consumerNames` in the Inspector.
  - Per-batch cross-frame `waitReasons`, frame `memoryPolicy`, `maxFramesInFlight`, `extraVersionCount`, `extraVersionBytes`, `peakManagedVramBytes`, `crossFrameResourceScheduled`, and `conservativeFallbackReasons` on the Memory/Queues tabs.

### Release GPU A/B Benchmark Evidence (P3)
Measured on Windows x64 MSVC Release (`build/rg-roadmap`), independent GPU queues (`graphics family 0`, `compute family 2`), Vulkan validation disabled during timing, warm shader/PSO cache, workload `CrossFramePolicyBench` (`ComputeStage` 524,288-element ALU compute pass writing 2 MiB `stage` -> `GraphicsStage` copying `stage` + 8× `2048×2048 RGBA16Sfloat` color clear passes on 32 MiB `color`; 16 warmup frames + 80 measured frames per independent process launch). Each run is one independent process launch (`METALLIC_RG_BENCH_CONFIG`).

#### Experiment 1: `MinimizeMemory` vs `MaximizeOverlap` (3 Frames in Flight, 64 MiB Pool Budget)
Protocol: interleaved A-B-A-B (4 independent runs; stopped at 4 runs because every B was strictly below every A and the mean wall-time gap `0.1430 ms` exceeded within-variant spread `0.0885 ms`).

| Run | Config | Wall ms/frame | GPU Envelope ms/frame | CPU ms/frame | Extra Versions | Extra VRAM | Peak Managed VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `3_crossframe_minmem` (A) | 1.1198 | 1.0370 | 0.3869 | 0 | 0 B | 36.00 MiB |
| 1 | `3_crossframe_maxoverlap` (B) | 0.9515 | 0.8172 | 0.3923 | 2 | 34.00 MiB | 70.00 MiB |
| 2 | `3_crossframe_minmem` (A) | 1.1576 | 1.0298 | 0.3774 | 0 | 0 B | 36.00 MiB |
| 3 | `3_crossframe_maxoverlap` (B) | 1.0400 | 0.9815 | 0.3996 | 2 | 34.00 MiB | 70.00 MiB |
| **Mean** | **A vs B** | **1.1387 vs 0.9958 (-12.6%)** | **1.0334 vs 0.8994 (-13.0%)** | **0.3822 vs 0.3960** | **0 vs 2** | **0 vs +34.00 MiB** | **36.00 vs 70.00 MiB** |

#### Experiment 2: 2 vs 3 Frames in Flight (`MinimizeMemory`)
Protocol: interleaved A-B-A-B extended to 8 independent runs because the first 4 runs were mixed; stopped at 8 runs with **no measurable difference** (distributions overlap: A `1.1285–1.2685 ms` vs B `0.9888–1.2203 ms`, mean gap `0.0795 ms` < within-variant spread `0.2315 ms`), because single-backing cross-frame WAR waits cap GPU overlap regardless of slot count.

| Config (4 runs each, interleaved) | Wall ms/frame (runs) | Mean / Median | GPU Envelope Mean | Extra VRAM | Peak Managed VRAM |
| --- | --- | ---: | ---: | ---: | ---: |
| `2_crossframe_minmem` (A) | 1.2685, 1.1285, 1.2398, 1.1809 | 1.2044 / 1.2104 ms | 1.1075 ms | 0 B | 36.00 MiB |
| `3_crossframe_minmem` (B) | 0.9888, 1.2203, 1.2063, 1.0841 | 1.1249 / 1.1452 ms | 1.0584 ms | 0 B | 36.00 MiB |

**Default Policy Decision**: Keep **`MinimizeMemory`** and **`maxFramesInFlight = 2`** as the conservative default. `MaximizeOverlap` is an explicit opt-in with a hard byte budget (`versionPoolBudgetBytes`) when workloads trade extra VRAM for reduced cross-frame WAR stalls.

## 5. Conservative Boundaries & External Domain Rollout (P4)

The following subsystems and passes explicitly declare `RenderGraphExternalDomainContract` with `hasUndeclaredImplicitAccesses = true`, preserving conservative cross-frame ordering and reporting exact reasons in the Execution Viewer without rewriting third-party SDKs or dynamic bindless pipelines:

| Component | Domain Name | Ownership | Conservative Boundary |
| --- | --- | --- | --- |
| [`GPUSceneSubsystem`](../Source/Runtime/Render/Subsystem/GPUSceneSubsystem.h) | `render.gpu-scene` | `ExternallyOwned` | Bindless/BDA geometry, visibility streams, and light tables remain conservative |
| [`LumenRadianceCachePass`](../Source/Runtime/Render/RenderPass/BuiltinPass/LumenRadianceCachePass.cpp) | `render.pass.lumen-radiance-cache` | `ExternallyOwned` | Persistent multi-frame radiance cache probe atlas and indirection tables |
| [`VirtualShadowMapPass`](../Source/Runtime/Render/RenderPass/BuiltinPass/VirtualShadowMapPass.cpp) | `render.pass.virtual-shadow-map` | `ExternallyOwned` | Persistent virtual shadow map physical page pool and page table |
| [`NRDDenoisePass`](../Source/Runtime/Render/RenderPass/BuiltinPass/NRDDenoisePass.cpp) | `render.sdk.nrd` | `OpaqueSDK` | NVIDIA NRD third-party SDK opaque temporal history and internal barriers |
| [`StreamlineDLSSPass`](../Source/Runtime/Render/RenderPass/BuiltinPass/StreamlineDLSSRRPass.cpp) | `render.sdk.dlss-sr` / `render.sdk.dlss-rr` | `OpaqueSDK` | NVIDIA Streamline / DLSS third-party SDK opaque temporal state |

### Explicit Synchronization Limits (`#58` and `#60`)
- **`#58` Same-queue writer visibility**: `CrossFrameResourceTracker` does not assume that a prior-frame reader's visibility barrier covers a new consumer stage; cross-frame same-queue boundaries preserve conservative `AllCommands / MemoryRead|MemoryWrite` initial scopes in `GraphAccessPlan`.
- **`#60` Exclusive queue-family ownership**: Resources using `VK_SHARING_MODE_EXCLUSIVE` across distinct queue families force conservative fallback (`requiresConservativeFallback = true`) until explicit paired release/acquire ownership transfer is modeled.
- Zero `VK_LAYER_VALIDATE_SYNC=1` errors does **not** prove dynamic bindless/BDA coverage or `#58`/`#60` completeness; conservative contracts remain the primary safety boundary.

## 6. Validation Summary

All GPU RHI suites below ran on `build/rg-roadmap` (MSVC Release, real independent Graphics + Compute queues) with `VK_LAYER_VALIDATE_SYNC=1 --rhi-validation` and reported **0 `VUID-` / `SYNC-HAZARD` warnings**:

- **Slang Module Audit**: `python -B Tools/CheckShaderModules.py` -> passed (`480 files, 313 modules, 167 implementation units`).
- **Task Runtime**: `MetallicTaskTests.exe` -> **35/35 PASSED**.
- **P0–P4 Lifecycle, Provider, Temporal, Cross-Frame, Aliasing & Viewer RHI Suites**: **36/36 PASSED** (`render_graph_resource_lifecycle_contracts`, `render_graph_cross_frame_tracker_and_pool`, `physical_environment_render_graph_provider_scheduling`, `frame_cross_frame_overlap_and_hazards`, `frame_declarative_temporal_history`, `frame_cross_frame_policy_benchmark`, `render_graph_viewer_lifecycle_and_contracts`, plus access-plan, viewer, texture-aliasing, and provider regressions).
- **PathTrace End-to-End Environment + Declarative Temporal Accumulation**: `physical_environment_source_and_history` and `physical_environment_sharc_source_and_history` -> **2/2 PASSED** with 0 sync hazards.
- **Editor Multi-Frame Smoke Test**: `METALLIC_SMOKE_TEST_FRAMES=6 Metallic.exe --smoke-test` -> exit code `0` (6 frames submitted and presented, including mid-run viewport resize).
