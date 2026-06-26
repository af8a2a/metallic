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

class RenderGraphStreamingSubsystem {
public:
    RenderGraphStreamingSubsystem() = default;
    ~RenderGraphStreamingSubsystem();

    RenderGraphStreamingSubsystem(RenderGraphStreamingSubsystem&&) noexcept = delete;
    RenderGraphStreamingSubsystem& operator=(RenderGraphStreamingSubsystem&&) noexcept = delete;

    RenderGraphStreamingSubsystem(const RenderGraphStreamingSubsystem&) = delete;
    RenderGraphStreamingSubsystem& operator=(const RenderGraphStreamingSubsystem&) = delete;

    Result initialize(Device& device, std::string& log);
    void reset();

    void beginFrame();
    void flush(CommandBuffer& commandBuffer);
    void endFrame();

    Streamer* streamer() const { return streamer_.get(); }
    bool initialized() const { return streamer_ != nullptr; }
    const RenderGraphStreamingStats& stats() const { return stats_; }

private:
    std::unique_ptr<Streamer> streamer_;
    RenderGraphStreamingStats stats_;
    bool frameActive_ = false;
};

class RenderGraphStreamingFrameScope {
public:
    explicit RenderGraphStreamingFrameScope(RenderGraphStreamingSubsystem& subsystem);
    ~RenderGraphStreamingFrameScope();

    RenderGraphStreamingFrameScope(const RenderGraphStreamingFrameScope&) = delete;
    RenderGraphStreamingFrameScope& operator=(const RenderGraphStreamingFrameScope&) = delete;

private:
    RenderGraphStreamingSubsystem* subsystem_ = nullptr;
};

} // namespace metallic::render
