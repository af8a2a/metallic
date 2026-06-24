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
}

void RenderGraphStreamingSubsystem::beginFrame()
{
    frameActive_ = streamer_ != nullptr;
}

void RenderGraphStreamingSubsystem::flush(CommandBuffer& commandBuffer)
{
    if (streamer_ != nullptr) {
        commandBuffer.copyStreamedData(*streamer_);
    }
}

void RenderGraphStreamingSubsystem::endFrame()
{
    if (frameActive_ && streamer_ != nullptr) {
        streamer_->endFrame();
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
