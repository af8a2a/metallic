#pragma once

#include "Runtime/Render/Core/BuiltinShaderRequests.h"
#include "MaterialShaderWarmupRequests.h"

namespace metallic::tools {

using ShaderWarmupRequest = render::ShaderRequest;

inline std::vector<ShaderWarmupRequest> shaderWarmupRequests()
{
    auto requests = render::builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    for (auto& request : materialShaderWarmupRequests(PROJECT_SOURCE_DIR, METALLIC_RTXCR_SHADER_INCLUDE_DIR)) {
        if (std::find(requests.begin(), requests.end(), request) == requests.end()) {
            requests.push_back(std::move(request));
        }
    }
    return requests;
}

} // namespace metallic::tools
