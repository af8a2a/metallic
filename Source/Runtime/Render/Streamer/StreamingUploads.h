#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <memory>
#include <string>

namespace metallic::render {

struct RenderGraphStreamingStats {
    uint64_t frameIndex = 0;
    uint32_t flushCount = 0;
    uint32_t flushesWithWork = 0;
    uint32_t transferCount = 0;
    uint32_t bufferTransferCount = 0;
    uint32_t textureTransferCount = 0;
    uint64_t transferBytes = 0;
    uint64_t bufferTransferBytes = 0;
    uint64_t textureTransferBytes = 0;
    StreamerStats streamer;
};

class StreamingUploads {
public:
    StreamingUploads() = default;
    ~StreamingUploads();

    StreamingUploads(StreamingUploads&&) noexcept = delete;
    StreamingUploads& operator=(StreamingUploads&&) noexcept = delete;

    StreamingUploads(const StreamingUploads&) = delete;
    StreamingUploads& operator=(const StreamingUploads&) = delete;

    Result initialize(Device& device, std::string& log, uint32_t frameSlotCount = 3);
    void reset();

    void beginFrame();
    Result beginFrame(RenderFrameContext& frame);
    void flush(CommandBuffer& commandBuffer, const StreamUploadPhaseCallback& phase = {});
    void endFrame();

    Streamer* streamer() const { return streamer_.get(); }
    bool initialized() const { return streamer_ != nullptr; }
    const RenderGraphStreamingStats& stats() const { return stats_; }

private:
    std::unique_ptr<Streamer> streamer_;
    RenderGraphStreamingStats stats_;
    bool frameActive_ = false;
};

class StreamingUploadFrameScope {
public:
    explicit StreamingUploadFrameScope(StreamingUploads& subsystem);
    ~StreamingUploadFrameScope();

    StreamingUploadFrameScope(const StreamingUploadFrameScope&) = delete;
    StreamingUploadFrameScope& operator=(const StreamingUploadFrameScope&) = delete;

private:
    StreamingUploads* subsystem_ = nullptr;
};

} // namespace metallic::render
