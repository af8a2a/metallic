#pragma once

#include "Runtime/Render/Streamer/StreamerSubsystem.h"

namespace metallic::render {

bool registerBuiltInRenderSubsystems(RenderSubsystemHost& host, std::string& log);

} // namespace metallic::render
