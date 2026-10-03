#pragma once

#include "Runtime/Render/Core/BuiltinShaderRequests.h"

namespace metallic::tools {

using ShaderWarmupRequest = render::ShaderRequest;

inline std::vector<ShaderWarmupRequest> shaderWarmupRequests()
{
    return render::builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
}

} // namespace metallic::tools
