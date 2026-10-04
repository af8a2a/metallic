#pragma once

#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <string_view>
#include <utility>

namespace metallic::render {

// Own all request strings. A view is created only at the synchronous compiler
// boundary, so copying/moving a request never leaves spans pointing at old data.
struct ShaderRequest {
    std::string module;
    std::string entry;
    std::vector<std::string> capabilities;
    std::vector<std::pair<std::string, std::string>> defines;
    std::vector<std::string> searchPaths;
    std::string searchPath = PROJECT_SOURCE_DIR "/Shaders";
    std::string profile = kDefaultSlangProfileName;
    SlangDescriptorHeapMode descriptorHeapMode = SlangDescriptorHeapMode::Default;

    bool operator==(const ShaderRequest&) const = default;
};

class ShaderRequestView {
public:
    explicit ShaderRequestView(const ShaderRequest& request) : request_(request)
    {
        for (const auto& capability : request.capabilities) { capabilities_.push_back(capability.c_str()); }
        for (const auto& path : request.searchPaths) { paths_.push_back(path.c_str()); }
        for (const auto& [name, value] : request.defines) { defines_.push_back({name.c_str(), value.c_str()}); }
    }
    ShaderRequestView(ShaderRequest&&) = delete;

    SlangShaderDesc desc() const
    {
        return {.moduleName = request_.module.c_str(), .entryPointName = request_.entry.c_str(),
            .searchPath = request_.searchPath.c_str(), .additionalSearchPaths = paths_,
            .profileName = request_.profile.c_str(), .capabilities = capabilities_, .macroDefines = defines_,
            .descriptorHeapMode = request_.descriptorHeapMode};
    }

private:
    const ShaderRequest& request_;
    std::vector<const char*> capabilities_;
    std::vector<const char*> paths_;
    std::vector<SlangMacroDefine> defines_;
};

enum class SceneShaderProgram {
    PathTrace, PathTraceGuides, OpenPBRPathTrace, OpenPBRPathTraceGuides,
    RealtimeLighting, Deferred, DeferredBinned, SharcClear, SharcResolve, Tonemap,
};

struct ShaderProgramIdentity {
    const char* module;
    const char* entry;
};

inline ShaderProgramIdentity sceneShaderIdentity(SceneShaderProgram program)
{
    switch (program) {
    case SceneShaderProgram::PathTrace: return {"Features/PathTracing/ScenePathTrace", "scenePathTraceMain"};
    case SceneShaderProgram::PathTraceGuides: return {"Features/PathTracing/ScenePathTraceGuides", "scenePathTraceGuidesMain"};
    case SceneShaderProgram::OpenPBRPathTrace: return {"Features/PathTracing/OpenPBRRayQueryPathTrace", "openPbrRayQueryPathTraceMain"};
    case SceneShaderProgram::OpenPBRPathTraceGuides: return {"Features/PathTracing/OpenPBRRayQueryPathTraceGuides", "openPbrRayQueryPathTraceGuidesMain"};
    case SceneShaderProgram::RealtimeLighting: return {"Features/Lighting/SceneRealtimeLighting", "sceneRealtimeLightingMain"};
    case SceneShaderProgram::Deferred: return {"Features/VisibilityBuffer/VisibilityBufferDeferred", "visibilityBufferDeferredMain"};
    case SceneShaderProgram::DeferredBinned: return {"Features/VisibilityBuffer/VisibilityBufferDeferred", "visibilityBufferDeferredBinnedMain"};
    case SceneShaderProgram::SharcClear: return {"Features/PathTracing/SceneSharcMaintenance", "sharcClearMain"};
    case SceneShaderProgram::SharcResolve: return {"Features/PathTracing/SceneSharcMaintenance", "sharcResolveMain"};
    case SceneShaderProgram::Tonemap: return {"Features/PostProcess/ScenePathTraceTonemap", "scenePathTraceTonemapMain"};
    }
    std::unreachable();
}

struct SceneShaderOptions {
    bool customMaterials = false;
    bool streamMaterials = false;
    bool streamRayQueries = false;
    bool globalView = false;
    bool hasRTXCR = false;
    bool hasNTC = false;
    bool cooperativeVector = false;
    bool positionFetch = false;
    bool deferredFloat16 = true;
    bool upscalerGuides = false;
    std::string materialInclude;
    std::string rtxcrInclude;
    std::string ntcInclude;
};

inline ShaderRequest makeSceneShaderRequest(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines = {})
{
    const bool deferred = program == SceneShaderProgram::Deferred || program == SceneShaderProgram::DeferredBinned;
    const auto identity = sceneShaderIdentity(program);
    ShaderRequest request{.module = identity.module, .entry = identity.entry,
        .capabilities = {"spvGroupNonUniformBallot"}};
    if (!deferred) { request.capabilities.insert(request.capabilities.begin(), "spvRayQueryKHR"); }
    if (!deferred && options.positionFetch) { request.capabilities.emplace_back("spvRayQueryPositionFetchKHR"); }
    if (options.cooperativeVector) { request.capabilities.emplace_back("spvCooperativeVectorNV"); }
    if (program == SceneShaderProgram::SharcClear || program == SceneShaderProgram::SharcResolve ||
        program == SceneShaderProgram::Tonemap) { return request; }

    const auto flag = [](bool value) { return value ? "1" : "0"; };
    request.defines = {
        {"METALLIC_CUSTOM_MATERIALS", flag(options.customMaterials)},
        {"METALLIC_STREAM_MATERIALS", flag(options.streamMaterials)},
        {"METALLIC_STREAM_RAY_QUERIES", flag(!deferred && options.streamRayQueries)},
        {"METALLIC_GLOBAL_VIEW", flag(options.globalView)},
        {"METALLIC_HAS_RTXCR", flag(!deferred && options.hasRTXCR)},
        {"METALLIC_HAS_NTC", flag(options.hasNTC)},
        {"METALLIC_NTC_COOPERATIVE_VECTOR", flag(options.cooperativeVector)},
        {"SCENE_RAYQUERY_ENABLE_POSITION_FETCH", flag(!deferred && options.positionFetch)},
        {"METALLIC_DEFERRED_LIGHT_GRID", flag(deferred)},
        {"METALLIC_REALTIME_DEFERRED", flag(deferred)},
    };
    request.defines.emplace_back("METALLIC_DEFERRED_UPSCALER_GUIDES", flag(deferred && options.upscalerGuides));
    if (deferred) { request.defines.emplace_back("METALLIC_DEFERRED_FP16", flag(options.deferredFloat16)); }
    for (const auto& define : extraDefines) { request.defines.emplace_back(define.name, define.value); }
    if (options.customMaterials) { request.searchPaths.push_back(options.materialInclude); }
    if (!deferred && options.hasRTXCR) { request.searchPaths.push_back(options.rtxcrInclude); }
    if (options.hasNTC) { request.searchPaths.push_back(options.ntcInclude); }
    return request;
}

enum class SceneRayQueryProgram { RTXDI, MaterialVisualization };

struct SceneRayQueryOptions {
    bool positionFetch = false;
    bool hasNTC = false;
    bool cooperativeVector = false;
    std::string ntcInclude;
};

inline ShaderRequest makeSceneRayQueryRequest(SceneRayQueryProgram program, const SceneRayQueryOptions& options)
{
    ShaderRequest request;
    if (program == SceneRayQueryProgram::RTXDI) {
        request.module = "Features/ReSTIR/SceneRTXDI";
        request.entry = "sceneRtxdiMain";
    } else {
        request.module = "Features/Debug/SceneMaterialVisualize";
        request.entry = "sceneMaterialVisualizeMain";
    }
    request.capabilities = {"spvRayQueryKHR"};
    if (options.positionFetch) { request.capabilities.emplace_back("spvRayQueryPositionFetchKHR"); }
    if (options.cooperativeVector) { request.capabilities.emplace_back("spvCooperativeVectorNV"); }
    request.defines = {{"SCENE_RAYQUERY_ENABLE_POSITION_FETCH", options.positionFetch ? "1" : "0"},
        {"METALLIC_HAS_NTC", options.hasNTC ? "1" : "0"},
        {"METALLIC_NTC_COOPERATIVE_VECTOR", options.cooperativeVector ? "1" : "0"}};
    if (options.hasNTC) { request.searchPaths.push_back(options.ntcInclude); }
    return request;
}

inline ShaderRequest makeColorResizeShaderRequest()
{
    return {.module = "Features/PostProcess/ColorResize", .entry = "colorResizeMain"};
}

struct ShadowShaderOptions {
    bool streamed = false;
    bool streamTlas = false;
    bool hasNTC = false;
    bool cooperativeVector = false;
    std::string ntcInclude;
};

inline ShaderRequest makeShadowShaderRequest(const ShadowShaderOptions& options)
{
    ShaderRequest request{.module = "Features/Lighting/ScreenSpaceShadows", .entry = "rayTracedShadowsMain",
        .capabilities = {"spvRayQueryKHR"},
        .defines = {{"METALLIC_STREAM_SHADOWS", options.streamed ? "1" : "0"},
            {"METALLIC_STREAM_TLAS", options.streamTlas ? "1" : "0"},
            {"METALLIC_HAS_NTC", options.hasNTC ? "1" : "0"},
            {"METALLIC_NTC_COOPERATIVE_VECTOR", options.cooperativeVector ? "1" : "0"}}};
    if (options.cooperativeVector) { request.capabilities.emplace_back("spvCooperativeVectorNV"); }
    if (options.hasNTC) { request.searchPaths.push_back(options.ntcInclude); }
    return request;
}

} // namespace metallic::render
