#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/Streamer/MeshletStreamThroughput.h"
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
struct StreamCpuWorkCounters {
    uint32_t demandVisited = 0;
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

// CPU-visible streaming counters; never triggers a GPU readback or page scan.
struct SceneStreamingProfile {
    std::string passName;
    std::string assetPath;
    uint64_t generation = 0;
    uint64_t frameIndex = 0;
    uint64_t feedbackFrame = UINT64_MAX;
    uint64_t geometryUsedBytes = 0;
    uint64_t geometryBudgetBytes = 0;
    uint64_t clasUsedBytes = 0;
    uint64_t clasCapacityBytes = 0;
    uint64_t clasEncodedBytes = 0;
    uint64_t clasWorstCaseBytes = 0;
    uint64_t clasScratchBytes = 0;
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
    MeshletStreamThroughput throughput;
    StreamCpuWorkCounters cpuWork;
};

} // namespace metallic::render
