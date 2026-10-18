#pragma once

#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/Core/ComputeResourceParameters.h"
#include "Runtime/Render/Core/NamedResourceParameters.h"
#include "Runtime/Render/Core/ResourceRegistry.h"

#include <optional>

namespace metallic::render {

// The production unsplit compute path tracer pushes the resource root directly.
// Legacy descriptor-buffer roots remain an explicit supported policy.
struct PathTraceShaderPolicy {
    bool inlineComputeRoot = true;
    bool trustedSceneParameters = true;
    bool textureFeedbackWave = false;

    ParameterTransport rootTransport() const
    {
        return inlineComputeRoot ? ParameterTransport::InlinePush : ParameterTransport::DescriptorBuffer;
    }
    uint32_t userPushDataSize() const
    {
        return static_cast<uint32_t>(inlineComputeRoot ? sizeof(ComputeResourceParameters) : sizeof(GPUBufferSpan));
    }
};

inline bool isPathTraceShaderRequest(const ShaderRequest& request)
{
    for (const auto program : {SceneShaderProgram::PathTrace, SceneShaderProgram::PathTraceGuides,
            SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides}) {
        const auto identity = sceneShaderIdentity(program);
        if (request.module == identity.module && request.entry == identity.entry) { return true; }
    }
    return false;
}

inline void applyPathTraceShaderPolicy(ShaderRequest& request, const PathTraceShaderPolicy& policy)
{
    if (policy.trustedSceneParameters && !policy.inlineComputeRoot) {
        throw std::invalid_argument("Trusted scene parameters require an inline compute root");
    }
    if (policy.inlineComputeRoot) { request.defines.emplace_back("METALLIC_INLINE_COMPUTE_ROOT", "1"); }
    if (policy.textureFeedbackWave) {
        request.defines.emplace_back("METALLIC_TEXTURE_FEEDBACK_WAVE", "1");
        request.capabilities.emplace_back("spvMaximalReconvergenceKHR");
    }
    if (policy.inlineComputeRoot) {
        request.defines.emplace_back("METALLIC_TRUSTED_SCENE_RESOURCE_BYTES",
            policy.trustedSceneParameters ? std::to_string(sizeof(SceneResourceParameters)) : "0");
    }
}

inline ShaderRequest makePathTraceShaderRequest(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines = {}, const PathTraceShaderPolicy& policy = {})
{
    auto request = makeSceneShaderRequest(program, options, extraDefines);
    if (!isPathTraceShaderRequest(request)) { throw std::invalid_argument("Expected a compute path tracing source"); }
    applyPathTraceShaderPolicy(request, policy);
    return request;
}

// Only recognize authored transport spellings. This must not silently infer a
// native layout for duplicate, unknown, or inconsistent ABI definitions.
inline std::optional<PathTraceShaderPolicy> parsePathTraceShaderPolicy(const ShaderRequest& request)
{
    PathTraceShaderPolicy policy{false, false, false};
    bool hasInline = false, hasTrusted = false, hasWave = false;
    for (const auto& [name, value] : request.defines) {
        if (name == "METALLIC_INLINE_COMPUTE_ROOT") {
            if (hasInline || value != "1") { return std::nullopt; }
            hasInline = true;
            policy.inlineComputeRoot = true;
        } else if (name == "METALLIC_TRUSTED_SCENE_RESOURCE_BYTES") {
            if (hasTrusted || (value != "0" && value != std::to_string(sizeof(SceneResourceParameters)))) { return std::nullopt; }
            hasTrusted = true;
            policy.trustedSceneParameters = value != "0";
        } else if (name == "METALLIC_TEXTURE_FEEDBACK_WAVE") {
            if (hasWave || value != "1") { return std::nullopt; }
            hasWave = true;
            policy.textureFeedbackWave = true;
        }
    }
    if (hasInline != hasTrusted) { return std::nullopt; }
    return policy;
}

} // namespace metallic::render
