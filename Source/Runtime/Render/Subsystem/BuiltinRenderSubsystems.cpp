#include "Runtime/Render/Subsystem/BuiltinRenderSubsystems.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include <array>

namespace metallic::render {

Result RenderUploadSubsystem::initialize(
    const RenderSubsystemInitContext& context,
    std::string& log)
{
    return streaming_.initialize(context.device, log);
}

Result RenderUploadSubsystem::beginFrame(
    const RenderSubsystemFrameContext&,
    RenderChangeBits&,
    std::string&)
{
    streaming_.beginFrame();
    return {};
}

void RenderUploadSubsystem::endFrame(const RenderSubsystemFrameContext&)
{
    streaming_.endFrame();
}

void RenderUploadSubsystem::shutdown()
{
    streaming_.reset();
}

void RenderUploadSubsystem::flush(CommandBuffer& commandBuffer)
{
    if (Streamer* currentStreamer = streaming_.streamer();
        currentStreamer == nullptr || currentStreamer->stats().pendingCopies.copyCount() == 0) {
        return;
    }
    streaming_.flush(commandBuffer);
}

void SceneResourcesSubsystem::shutdown()
{
    manager_.clear();
}

bool registerBuiltInRenderSubsystems(RenderSubsystemHost& host, std::string& log)
{
    if (!host.isRegistered(RenderUploadSubsystem::kSubsystemId) &&
        !host.registerSubsystem<RenderUploadSubsystem>(log)) {
        return false;
    }
    constexpr std::array<RenderSubsystemId, 1> sceneDependencies{
        RenderUploadSubsystem::kSubsystemId,
    };
    if (!host.isRegistered(SceneResourcesSubsystem::kSubsystemId) &&
        !host.registerSubsystem<SceneResourcesSubsystem>(sceneDependencies, log)) {
        return false;
    }
    constexpr std::array<RenderSubsystemId, 1> environmentDependencies{
        RenderUploadSubsystem::kSubsystemId,
    };
    return host.isRegistered(EnvironmentLightingSubsystem::kSubsystemId) ||
        host.registerSubsystem<EnvironmentLightingSubsystem>(environmentDependencies, log);
}

} // namespace metallic::render
