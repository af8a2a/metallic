#pragma once

#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Streamer/StreamingUploads.h"

#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render {

class MeshletStreamRuntime;

struct MeshletStreamInitialLoadStats {
    uint64_t pumpCalls = 0;
    uint64_t batches = 0;
    uint64_t uploadBytes = 0;
    double pumpMilliseconds = 0.0;
    double gpuWaitMilliseconds = 0.0;
};

// A serial, completion-driven queue for a new stream's mandatory root resources.
// Call before graph recording starts using that stream. The caller serializes
// queue access, and keeps Device and the runtime alive until reset has drained
// any accepted work after an error.
class MeshletStreamInitialLoader {
public:
    MeshletStreamInitialLoader() = default;
    ~MeshletStreamInitialLoader();

    MeshletStreamInitialLoader(const MeshletStreamInitialLoader&) = delete;
    MeshletStreamInitialLoader& operator=(const MeshletStreamInitialLoader&) = delete;

    Result<> initialize(Device& device, std::string& log);
    // Always attempts one batch when incomplete. The budget limits additional
    // batches; an individual recording/GPU wait may exceed it. Successful return
    // guarantees that every submitted batch is complete before graph handoff.
    // metadataOnly initializes Device tables while preserving lazy root loading.
    Result<> pump(MeshletStreamRuntime& runtime, double budgetMilliseconds,
        bool& complete, std::string& log, bool metadataOnly = false);
    Result<> reset();
    const MeshletStreamInitialLoadStats& stats() const { return stats_; }

private:
    Device* device_ = nullptr;
    Queue* queue_ = nullptr;
    StreamingUploads uploads_;
    QueueSubmissionTracker submissions_;
    RenderFrameContext frame_;
    std::unique_ptr<CommandPool> commandPool_;
    std::unique_ptr<CommandBuffer> commands_;
    uint64_t frameIndex_ = 0;
    MeshletStreamInitialLoadStats stats_;
};

} // namespace metallic::render
