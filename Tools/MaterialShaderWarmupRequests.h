#pragma once

#include "Runtime/Render/Core/ShaderRequests.h"
#include <filesystem>

namespace metallic::tools {

// Generate installed local samples' material programs through the runtime CPU
// compiler. This does not load scene geometry, textures or initialize a GPU.
std::vector<render::ShaderRequest> materialShaderWarmupRequests(
    const std::filesystem::path& projectRoot, const std::string& rtxcrInclude);

} // namespace metallic::tools
