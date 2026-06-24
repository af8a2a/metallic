#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <memory>
#include <string>

namespace metallic::render {

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

private:
    std::unique_ptr<Streamer> streamer_;
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
