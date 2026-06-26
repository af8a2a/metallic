#include "Runtime/Render/RenderGraph/RenderGraphStreamingSubsystem.h"

namespace metallic::render {
namespace {

StreamerDesc defaultRenderGraphStreamerDesc()
{
    StreamerDesc desc;
    desc.constantBufferSize = 256ull * 1024ull;
    desc.dynamicBufferSizePerFrame = 4ull * 1024ull * 1024ull;
    desc.queuedFrameCount = 3;
    desc.dynamicBufferDesc.usage = BufferUsageBits::TransferSource |
        BufferUsageBits::Storage |
        BufferUsageBits::Constant;
    return desc;
}

} // namespace

RenderGraphStreamingSubsystem::~RenderGraphStreamingSubsystem()
{
    endFrame();
}

Result RenderGraphStreamingSubsystem::initialize(Device& device, std::string& log)
{
    log.clear();
    if (streamer_ != nullptr) {
        return {};
    }

    Result result = device.createStreamer(defaultRenderGraphStreamerDesc(), streamer_);
    if (!result || streamer_ == nullptr) {
        log = "createStreamer(RenderGraphStreamingSubsystem) returned ";
        log += resultToString(result);
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

void RenderGraphStreamingSubsystem::reset()
{
    endFrame();
    streamer_.reset();
    stats_ = {};
}

void RenderGraphStreamingSubsystem::beginFrame()
{
    frameActive_ = streamer_ != nullptr;
    ++stats_.frameIndex;
    stats_.flushCount = 0;
    stats_.flushesWithWork = 0;
    stats_.transferCount = 0;
    stats_.bufferTransferCount = 0;
    stats_.textureTransferCount = 0;
    stats_.transferBytes = 0;
    stats_.bufferTransferBytes = 0;
    stats_.textureTransferBytes = 0;
    stats_.streamer = streamer_ != nullptr ? streamer_->stats() : StreamerStats{};
}

void RenderGraphStreamingSubsystem::flush(CommandBuffer& commandBuffer)
{
    if (streamer_ != nullptr) {
        const StreamerStats streamerStats = streamer_->stats();
        const StreamerPendingCopyStats pendingCopies = streamerStats.pendingCopies;
        ++stats_.flushCount;
        if (pendingCopies.copyCount() > 0) {
            ++stats_.flushesWithWork;
        }
        stats_.transferCount += pendingCopies.copyCount();
        stats_.bufferTransferCount += pendingCopies.bufferCopyCount;
        stats_.textureTransferCount += pendingCopies.textureCopyCount;
        stats_.transferBytes += pendingCopies.copyBytes();
        stats_.bufferTransferBytes += pendingCopies.bufferCopyBytes;
        stats_.textureTransferBytes += pendingCopies.textureCopyBytes;
        stats_.streamer = streamerStats;
        commandBuffer.copyStreamedData(*streamer_);
        stats_.streamer = streamer_->stats();
    }
}

void RenderGraphStreamingSubsystem::endFrame()
{
    if (frameActive_ && streamer_ != nullptr) {
        streamer_->endFrame();
        stats_.streamer = streamer_->stats();
    }
    frameActive_ = false;
}

RenderGraphStreamingFrameScope::RenderGraphStreamingFrameScope(RenderGraphStreamingSubsystem& subsystem)
    : subsystem_(&subsystem)
{
    subsystem_->beginFrame();
}

RenderGraphStreamingFrameScope::~RenderGraphStreamingFrameScope()
{
    if (subsystem_ != nullptr) {
        subsystem_->endFrame();
    }
}

} // namespace metallic::render
