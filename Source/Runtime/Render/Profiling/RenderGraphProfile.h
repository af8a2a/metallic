#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
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
    uint64_t loadFailures = 0;
};

} // namespace metallic::render
