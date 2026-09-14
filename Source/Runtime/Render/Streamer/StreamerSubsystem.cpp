#include "Runtime/Render/Streamer/StreamerSubsystem.h"

#include <algorithm>

namespace metallic::render {

Result StreamerSubsystem::initialize(const RenderSubsystemInitContext& context, std::string& log)
{
    device_ = &context.device;
    return uploads_.initialize(context.device, log, context.host.frameSlotCount());
}

Result StreamerSubsystem::beginFrame(const RenderSubsystemFrameContext& context, RenderChangeBits&, std::string&)
{
    collectReleasedStreams();
    if (context.frameResources) { return uploads_.beginFrame(*context.frameResources); }
    uploads_.beginFrame();
    return {};
}

void StreamerSubsystem::endFrame(const RenderSubsystemFrameContext&)
{
    uploads_.endFrame();
}

void StreamerSubsystem::shutdown()
{
    // The host waits for submitted work and retires graph passes before this.
    streams_.clear();
    resources_.clear();
    uploads_.reset();
    device_ = nullptr;
}

Result StreamerSubsystem::acquireStream(const MeshletStreamRuntimeDesc& desc, bool debugReadback,
    std::shared_ptr<MeshletStreamRuntime>& outSession, std::string& log, PipelineCache* cache)
{
    if (!device_) { return makeError(Error::InvalidArgument); }
    collectReleasedStreams();
    auto session = std::make_shared<MeshletStreamRuntime>();
    session->setDebugReadbackEnabled(debugReadback);
    Result result = session->initialize(*device_, desc, log, cache);
    if (!result) { return result; }
    streams_.push_back(session);
    outSession = std::move(session);
    return {};
}

void StreamerSubsystem::collectReleasedStreams()
{
    std::erase_if(streams_, [](const auto& session) { return session.use_count() == 1; });
}

void StreamerSubsystem::flush(CommandBuffer& commands)
{
    if (streamer() && streamer()->stats().pendingCopies.copyCount() != 0) { uploads_.flush(commands); }
}

} // namespace metallic::render
