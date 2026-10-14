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
    DescriptorHeapShaderStrides descriptorHeapStrides;
    SlangShaderOptimizationLevel optimizationLevel = SlangShaderOptimizationLevel::Default;

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
            .descriptorHeapMode = request_.descriptorHeapMode,
            .descriptorHeapStrides = request_.descriptorHeapStrides,
            .optimizationLevel = request_.optimizationLevel};
    }

private:
    const ShaderRequest& request_;
    std::vector<const char*> capabilities_;
    std::vector<const char*> paths_;
    std::vector<SlangMacroDefine> defines_;
};

enum class SceneShaderProgram {
    PathTrace, PathTraceGuides, OpenPBRPathTrace, OpenPBRPathTraceGuides,
    RealtimeLighting, Deferred, DeferredBinned, SharcClear, SharcResolve, OpenPBRSharcResolve,
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
    case SceneShaderProgram::OpenPBRSharcResolve:
    case SceneShaderProgram::SharcResolve: return {"Features/PathTracing/SceneSharcMaintenance", "sharcResolveMain"};
    }
    std::unreachable();
}

struct SceneShaderOptions {
    bool customMaterials = false;
    bool globalView = false;
    bool hasRTXCR = false;
    bool deferredFloat16 = true;
    bool upscalerGuides = false;
    std::string materialInclude;
    std::string rtxcrInclude;
};

inline ShaderRequest makeSceneShaderRequest(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines = {})
{
    if (program == SceneShaderProgram::OpenPBRPathTraceGuides) {
        const auto update = std::find_if(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
            return std::string_view(define.name) == "SHARC_UPDATE";
        });
        const auto updateCount = std::count_if(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
            return std::string_view(define.name) == "SHARC_UPDATE";
        });
        // Sparse training bypasses guide evaluation and calls the same path
        // dispatcher. Preserve duplicate definitions for Slang to interpret.
        if (updateCount == 1 && std::string_view(update->value) == "1") {
            program = SceneShaderProgram::OpenPBRPathTrace;
        }
    }
    const bool deferred = program == SceneShaderProgram::Deferred || program == SceneShaderProgram::DeferredBinned;
    const auto identity = sceneShaderIdentity(program);
    ShaderRequest request{.module = identity.module, .entry = identity.entry,
        .capabilities = {"spvGroupNonUniformBallot"}};
    if (!deferred) {
        request.capabilities.insert(request.capabilities.begin(),
            {"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV"});
    }
    if (program == SceneShaderProgram::OpenPBRSharcResolve) {
        // OpenPBR's lobe/direction keys cannot be decoded as SHaRC grid positions.
        request.defines.emplace_back("SHARC_BLEND_ADJACENT_LEVELS", "0");
        return request;
    }
    if (program == SceneShaderProgram::SharcClear || program == SceneShaderProgram::SharcResolve) { return request; }

    const auto flag = [](bool value) { return value ? "1" : "0"; };
    request.defines = {
        {"METALLIC_CUSTOM_MATERIALS", flag(options.customMaterials)},
        {"METALLIC_GLOBAL_VIEW", flag(options.globalView)},
        {"METALLIC_HAS_RTXCR", flag(!deferred && options.hasRTXCR)},
        {"METALLIC_DEFERRED_LIGHT_GRID", flag(deferred)},
        {"METALLIC_REALTIME_DEFERRED", flag(deferred)},
    };
    request.defines.emplace_back("METALLIC_DEFERRED_UPSCALER_GUIDES", flag(deferred && options.upscalerGuides));
    if (deferred) {
        bool deferredFloat16 = options.deferredFloat16;
        if (program == SceneShaderProgram::DeferredBinned) {
            const auto classCount = std::count_if(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
                return std::string_view(define.name) == "MATERIAL_CLASS";
            });
            const bool backgroundOnly = std::any_of(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
                return std::string_view(define.name) == "MATERIAL_CLASS" && std::string_view(define.value) == "0";
            });
            const bool explicitPrecision = std::any_of(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
                return std::string_view(define.name) == "METALLIC_DEFERRED_FP16";
            });
            // Class zero only shades the environment and never evaluates the
            // deferred FP16 lighting path. Preserve ambiguous/explicit overrides.
            if (classCount == 1 && backgroundOnly && !explicitPrecision) { deferredFloat16 = false; }
        }
        request.defines.emplace_back("METALLIC_DEFERRED_FP16", flag(deferredFloat16));
    }
    for (const auto& define : extraDefines) { request.defines.emplace_back(define.name, define.value); }
    if (options.customMaterials) { request.searchPaths.push_back(options.materialInclude); }
    if (!deferred && options.hasRTXCR) { request.searchPaths.push_back(options.rtxcrInclude); }
    return request;
}

enum class SceneRayQueryProgram { RTXDI, MaterialVisualization };

inline ShaderRequest makeSceneRayQueryRequest(SceneRayQueryProgram program)
{
    ShaderRequest request;
    if (program == SceneRayQueryProgram::RTXDI) {
        request.module = "Features/ReSTIR/SceneRTXDI";
        request.entry = "sceneRtxdiMain";
    } else {
        request.module = "Features/Debug/SceneMaterialVisualize";
        request.entry = "sceneMaterialVisualizeMain";
    }
    request.capabilities = {"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV"};
    return request;
}

inline ShaderRequest makeColorResizeShaderRequest()
{
    return {.module = "Features/PostProcess/ColorResize", .entry = "colorResizeMain"};
}

inline ShaderRequest makeEditorDisplayShaderRequest(const char* entry, bool hdr,
    bool scRgbImage, bool sRGBAttachment, const std::string& whiteNits)
{
    if (std::string_view(entry) == "editorOutputVertex" || std::string_view(entry) == "editorOutputPQFragment") {
        return {.module = "Features/PostProcess/EditorOutput", .entry = entry,
            .descriptorHeapMode = SlangDescriptorHeapMode::Native};
    }
    if (std::string_view(entry) == "editorDisplayVertex") {
        // Display conversion only runs in the fragment entry. Share the vertex
        // request across all display settings in runtime and startup warmup.
        return {.module = "Features/PostProcess/EditorDisplay", .entry = entry,
            .descriptorHeapMode = SlangDescriptorHeapMode::Mapped};
    }
    std::string canonicalWhiteNits = whiteNits;
    if (std::string_view(entry) == "editorDisplayFragment") {
        // At exactly 80 nits, HDR conversion of an sRGB image is the same
        // identity conversion as SDR output to an sRGB attachment.
        if (hdr && !scRgbImage && whiteNits == "80.000000") {
            hdr = false;
            sRGBAttachment = true;
        } else if (hdr) {
            sRGBAttachment = false;
        }
        // Paper white only scales SDR scRGB previews and HDR sRGB images.
        if (hdr == scRgbImage) { canonicalWhiteNits = "203.000000"; }
    }
    return {.module = "Features/PostProcess/EditorDisplay", .entry = entry,
        .defines = {{"DISPLAY_HDR", hdr ? "1" : "0"},
            {"DISPLAY_SCRGB_IMAGE", scRgbImage ? "1" : "0"},
            {"DISPLAY_SRGB_ATTACHMENT", sRGBAttachment ? "1" : "0"},
            {"DISPLAY_WHITE_NITS", canonicalWhiteNits}},
        .descriptorHeapMode = SlangDescriptorHeapMode::Mapped};
}

inline ShaderRequest makeShadowShaderRequest()
{
    ShaderRequest request{.module = "Features/Lighting/ScreenSpaceShadows", .entry = "rayTracedShadowsMain",
        .capabilities = {"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV"}};
    return request;
}

} // namespace metallic::render
