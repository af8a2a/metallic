#pragma once

#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/Streamer/MeshletStreamThroughput.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace metallic::render {

struct RenderGraphProfileSection {
    std::string name;
    uint32_t parent = UINT32_MAX;
    QueueType queue = QueueType::Graphics;
    double cpuMilliseconds = 0.0;
    double gpuMilliseconds = 0.0;
    bool gpuTimingAvailable = false;
    bool cpuOnly = false;
};

// Per-frame work counts for aggregate CPU timings; no per-page clock reads.
struct StreamCPUWorkCounters {
    uint32_t allocationAttempts = 0;
    uint32_t budgetRetrySuppressed = 0;
    uint32_t requestDuplicatesMerged = 0;
    uint32_t admissionCandidates = 0;
    uint32_t admissionPriorityPops = 0;
    uint32_t priorityRecomputed = 0;
    uint32_t priorityReused = 0;
    uint32_t admissionCalls = 0;
    uint32_t admissionBypassed = 0;
    uint32_t demandVisited = 0; // Demand classification visits, excluding prefetch side effects.
    uint32_t demandMembershipTests = 0; // Previous unused set bit checks, without PageEntry lookups.
    uint32_t demandPrefetchVisited = 0;
    uint32_t coldCandidateTests = 0;
    uint32_t demandTransitions = 0;
    uint32_t demandEpochUpdates = 0;
    uint32_t demandStaleBatches = 0;
    uint32_t demandNewerThanFeedback = 0;
    uint32_t demandUnused = 0;
    uint32_t demandRefreshed = 0;
    uint32_t demandIncompleteProtected = 0;
    uint32_t coldCandidates = 0;
    uint32_t coldVisited = 0;
    uint32_t coldStateRejected = 0;
    uint32_t coldClasLookups = 0;
    uint32_t coldAgeRejected = 0;
    uint32_t coldScheduleFailed = 0;
    uint32_t coldPressureScheduled = 0;
    uint32_t coldRetentionScheduled = 0;
    uint32_t pendingFreePages = 0;
};

// One completed independent graphics upload; frame IDs are producer history frames.
// Completion latency includes queueing and polling, unlike the timestamp interval.
struct TextureUploadProfile {
    uint64_t sequence = 0, requestFrame = 0, submitFrame = 0, completionFrame = 0;
    uint64_t bytes = 0;
    uint32_t images = 0;
    bool gpuTimingAvailable = false;
    double gpuMilliseconds = 0, completionObservedMilliseconds = 0;
};

// CPU-visible streaming counters; never triggers a GPU readback or page scan.
struct SceneStreamingProfile {
    std::string passName;
    std::string assetPath;
    std::string softwareRasterIdentity; // Resolved module/entry/SPIR-V hash and runtime mode, JSON.
    std::string diagnosticSourceIdentity; // Same-process frozen source identity; diagnostics only.
    std::string diagnosticTextureSemanticIdentity; // Untimed capture only; source slots and published mip metadata.
    bool nodeDemandC1 = false; // Effective conservative RT node demand visibility pruning.
    bool nodeDemandC2 = false; // Effective offscreen node threshold pruning; requires C1.
    uint32_t rtDemandMode = 0;
    bool rtHistoryPacked = false;
    uint64_t rtLodHistoryBytes = 0;
    uint64_t generation = 0;
    uint64_t frameIndex = 0;
    uint64_t feedbackFrame = UINT64_MAX;
    // Geometry LOD diagnostics, from the same completed feedback frame.
    // Counts are instance-groups, not unique pages. Selected means emitted cut,
    // before raster visibility/occlusion and independent of CLAS readiness.
    bool lodTransitionTelemetryEnabled = false;
    uint64_t lodTransitionHistoryBytes = 0;
    uint32_t lodDemandedGroups = 0, lodOwnPageBlockedGroups = 0, lodDependencyBlockedGroups = 0;
    uint32_t lodCatchupActivatedGroups = 0, lodCatchupSelectedGroups = 0, lodCatchupSelectedClusters = 0;
    uint32_t lodThresholdSelectedGroups = 0, lodThresholdSelectedClusters = 0;
    uint32_t lodUnclassifiedSelectedGroups = 0, lodUnclassifiedSelectedClusters = 0;
    bool predictivePrefetchEnabled = false, prefetchForecastActive = false, prefetchMeasuredLatency = false;
    bool prefetchEnabled = false, prefetchMemoryWatermarkBlocked = false, prefetchQueueBlocked = false;
    double prefetchHorizonMilliseconds = 0, prefetchTranslationDistance = 0, prefetchRotationDegrees = 0;
    uint64_t prefetchLatencySamples = 0;
    double prefetchDemandLatencyP95Milliseconds = 0;
    uint64_t totalPrefetchAdmitted = 0, totalPrefetchUsed = 0, totalPrefetchDeferred = 0;
    uint32_t prefetchGpuRequests = 0, prefetchGpuDropped = 0;
    bool adaptivePageRetentionEnabled = false;
    uint64_t geometryReclaimReserveBytes = 0, geometryDemandReserveBytes = 0;
    uint64_t coldResidentBytes = 0, pendingFreeBytes = 0, evictedGeometryBytes = 0;
    uint32_t evictedPrefetchPages = 0;
    bool blasFeedbackAvailable = false;
    uint32_t blasDiagnosticResetEpoch = 0;
    uint64_t blasFeedbackFrame = 0;
    uint32_t blasBuildCount = 0, blasClusterReferences = 0, blasOverflowCount = 0;
    uint32_t blasRequestedClusterReferences = 0;
    uint32_t blasRequestedInstances = 0;
    uint32_t blasReferenceBudgetRejected = 0;
    uint32_t blasBuildBudgetRejected = 0;
    uint32_t blasOversizedInstances = 0;
    uint32_t blasMissingClasInstances = 0;
    uint32_t blasInvalidGroups = 0;
    uint32_t blasLiveClusterReferences = 0, blasAdmittedInstances = 0;
    bool blasInstanceReuseEnabled = true;
    uint32_t blasDirtyInstances = 0;
    uint32_t blasReusedInstances = 0;
    uint32_t blasStorageRejected = 0;
    uint32_t blasArenaUsedBytes = 0;
    uint32_t blasArenaRepack = 0;
    bool blasGPUAllocatorEnabled = false;
    bool blasPriorityEnabled = false;
    bool blasPriorityCacheEnabled = false;
    bool blasMergedSharingEnabled = false;
    uint32_t blasFallbackPolicy = 0;
    bool blasMixedTileFastPathEnabled = false;
    uint32_t blasPrioritySortReused = 0;
    uint32_t blasPriorityAdmissionReused = 0;
    uint32_t blasRequestedVisibleConsumers = 0, blasRequestedOffscreenConsumers = 0;
    uint32_t blasRejectedVisibleConsumers = 0, blasRejectedOffscreenConsumers = 0;
    uint32_t blasAdmittedOffscreenConsumers = 0, blasPriorityEvictedOwners = 0;
    uint32_t blasCompactionTargetBytes = 0, blasWaitingBuildCount = 0;
    uint32_t blasPostBuildBudgetRejected = 0, blasPhysicalAllocationFailed = 0;
    uint32_t blasWaitingClaimBytes = 0, blasMaxWaitingBytes = 0, blasUnknownBuildCount = 0;
    uint32_t blasProjectedRetryDeferredCount = 0, blasProjectedDirtyClaimBytes = 0;
    float blasRejectedMaxErrorPixels = 0, blasRejectedVisibleMaxErrorPixels = 0;
    uint32_t blasEncodedBytes = 0, blasAllocatedBytes = 0;
    uint32_t blasFreeBytes = 0, blasLargestFreeBytes = 0;
    uint32_t blasRelocatedCount = 0, blasRelocatedBytes = 0;
    uint32_t blasSuppressedBuilds = 0, blasFailedAllocations = 0;
    uint64_t blasPersistentBytes = 0, blasStagingBytes = 0, blasMoveScratchBytes = 0;
    uint64_t blasRelocationStagingBytes = 0;
    uint32_t blasPublicationInvalidated = 0;
    uint32_t blasSemanticChangeCount = 0;
    bool blasRepresentationsEnabled = false;
    uint32_t blasRepresentationCount = 0;
    uint32_t blasSharingInstances = 0;
    uint32_t blasMergedInstances = 0;
    uint32_t blasCachedInstances = 0;
    uint32_t blasRepresentationGroupCount = 0;
    uint32_t blasRepresentationGroupOverflow = 0;
    uint32_t blasCacheDependencyInvalidations = 0;


    uint64_t geometryUsedBytes = 0;
    uint64_t geometryBudgetBytes = 0;
    // Immutable group/topology allocation requests and accepted initialization;
    // these are CPU counters, not proof of physical residency after capture injection.
    bool deviceImmutableMetadata = false, immutableMetadataReady = false;
    uint64_t immutableMetadataBytes = 0, immutableMetadataAllocatedBytes = 0;
    uint64_t immutableMetadataSubmittedBytes = 0, immutableMetadataStagingBytes = 0;
    uint64_t immutableMetadataUploadBatches = 0;
    uint64_t immutableGroupBytes = 0, immutableTopologyBytes = 0;
    uint64_t clasUsedBytes = 0;
    uint64_t clasCapacityBytes = 0; // Configured budget; not physical allocation.
    uint64_t clasAllocatedBytes = 0;
    uint32_t clasStorageChunks = 0;
    uint64_t clasStartBytes = 0, clasGrowBytes = 0, clasEmptyBytes = 0;
    uint64_t clasGrowthCount = 0, clasReleasedBytes = 0;
    uint64_t clasPersistentAllocatedBytes = 0, clasPersistentUsedBytes = 0, clasPersistentGrowBytes = 0;
    uint64_t clasTransientAllocatedBytes = 0, clasTransientUsedBytes = 0, clasFragmentedFreeBytes = 0;
    uint64_t clasEncodedBytes = 0;
    uint64_t clasWorstCaseBytes = 0;
    uint64_t clasScratchBytes = 0;
    uint64_t clasVirtualBytes = 0, clasSizeReadbackBytes = 0;
    uint32_t clasAllocatorPages = 0;
    uint64_t clasReadyLatencySamples = 0, clasReadyLatencyFrameTotal = 0, clasReadyLatencyFrameMax = 0;
    uint32_t clasReadyLatencyFramesP50 = 0, clasReadyLatencyFramesP95 = 0;
    std::array<uint64_t, 33> clasReadyLatencyHistogram{};
    uint32_t clasMovedClusters = 0;
    bool clasEnabled = false;
    uint32_t clasResidentPages = 0;
    uint32_t clasResidentClusters = 0;
    uint32_t clasRetiringPages = 0;
    uint32_t clasPendingPages = 0;
    uint32_t clasBuiltPages = 0;
    uint32_t clasBuiltClusters = 0;
    uint32_t clasRejectedPages = 0;
    uint64_t clasTotalBuiltPages = 0;
    uint64_t clasTotalBuiltClusters = 0;
    uint32_t totalPages = 0;
    uint32_t residentPages = 0;
    uint32_t pendingPages = 0;
    uint32_t gpuLoadRequests = 0; // Latest completed feedback; pair with feedbackFrame, unlike this frame's consumed requests.
    uint64_t geometryPendingTasks = 0;
    uint32_t geometryPendingPatches = 0;
    uint32_t geometryPublicationRevision = 0;
    uint64_t clasPublicationRevision = 0;
    uint32_t ioQueued = 0;
    uint32_t ioActive = 0;
    uint32_t uploadQueued = 0;
    uint32_t requests = 0;
    uint32_t uploads = 0;
    uint32_t evictions = 0;
    uint32_t requestOverflows = 0;
    uint32_t allocationFailures = 0;
    uint64_t uploadBytes = 0;
    uint64_t totalUploadBytes = 0;
    // Host-staged envelope bytes (includes CPU-only sideband), not PCIe traffic.
    uint64_t storedUploadBytes = 0;
    uint64_t totalStoredUploadBytes = 0;
    uint64_t gpuDecompressedPages = 0;
    uint64_t totalGpuDecompressedPages = 0;
    uint64_t loadFailures = 0;
    TextureUploadProfile textureUpload;
    bool textureStreaming = false;
    uint64_t textureResidentBytes = 0, textureBudgetBytes = 0, texturePendingBytes = 0, textureRetiredBytes = 0;
    uint64_t textureUpgrades = 0, textureDowngrades = 0, textureBudgetDeferrals = 0;
    uint64_t textureFeedbackFrames = 0, textureUploadBytes = 0, textureMaxRequestFrames = 0;
    uint32_t textureRefinedImages = 0, textureRequestedImages = 0, texturePendingImages = 0;
    MeshletStreamThroughput throughput;
    StreamCPUWorkCounters cpuWork;
};

} // namespace metallic::render
