#include "Runtime/Render/Subsystem/BuiltinRenderSubsystems.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"

#include <array>

namespace metallic::render {

bool registerBuiltInRenderSubsystems(RenderSubsystemHost& host, std::string& log)
{
    if (!host.isRegistered(StreamerSubsystem::kSubsystemId) &&
        !host.registerSubsystem<StreamerSubsystem>(log)) {
        return false;
    }
    constexpr std::array dependencies{StreamerSubsystem::kSubsystemId};
    if (!host.isRegistered(GPUSceneSubsystem::kSubsystemId) &&
        !host.registerSubsystem<GPUSceneSubsystem>(dependencies, log)) {
        return false;
    }
    return host.isRegistered(EnvironmentLightingSubsystem::kSubsystemId) ||
        host.registerSubsystem<EnvironmentLightingSubsystem>(dependencies, log);
}

} // namespace metallic::render
