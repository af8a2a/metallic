#pragma once

#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <stdexcept>
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
    bool upscalerGuides = false; // Output policy; Deferred shader identity is shared.
    std::string materialInclude;
    std::string rtxcrInclude;
};

// -1 preserves the original static request; 0/1 share source but keep native policy.
inline int32_t openPBRCacheQuerySpecializationValue(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines)
{
    if ((program != SceneShaderProgram::OpenPBRPathTrace && program != SceneShaderProgram::OpenPBRPathTraceGuides) ||
        options.customMaterials || options.globalView || !options.hasRTXCR ||
        sceneWorkingColorSpace() != SceneWorkingColorSpace::ACEScg) { return -1; }
    if (extraDefines.empty()) { return 0; }
    return extraDefines.size() == 1 && std::string_view(extraDefines[0].name) == "SHARC_QUERY" &&
        std::string_view(extraDefines[0].value) == "1" ? 1 : -1;
}

// Standard Plain OFF stays static; Plain QUERY and Guide OFF share one source.
inline int32_t standardPathCacheQuerySpecializationValue(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines)
{
    if ((program != SceneShaderProgram::PathTrace && program != SceneShaderProgram::PathTraceGuides) ||
        options.customMaterials || options.globalView || !options.hasRTXCR ||
        sceneWorkingColorSpace() != SceneWorkingColorSpace::ACEScg) { return -1; }
    if (extraDefines.empty()) { return program == SceneShaderProgram::PathTrace ? -1 : 0; }
    return program == SceneShaderProgram::PathTrace && extraDefines.size() == 1 && std::string_view(extraDefines[0].name) == "SHARC_QUERY" &&
        std::string_view(extraDefines[0].value) == "1" ? 1 : -1;
}

// The independently qualified Single Guides pair; SINGLE remains static 1.
inline int32_t singleGuideCacheQuerySpecializationValue(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines)
{
    if (program != SceneShaderProgram::OpenPBRPathTraceGuides || options.customMaterials || options.globalView ||
        !options.hasRTXCR || sceneWorkingColorSpace() != SceneWorkingColorSpace::ACEScg) { return -1; }
    const auto single = [](const SlangMacroDefine& define) {
        return std::string_view(define.name) == "METALLIC_PT_SINGLE_SAMPLE" && std::string_view(define.value) == "1";
    };
    if (extraDefines.size() == 1 && single(extraDefines[0])) { return 0; }
    return extraDefines.size() == 2 && std::string_view(extraDefines[0].name) == "SHARC_QUERY" &&
        std::string_view(extraDefines[0].value) == "1" && single(extraDefines[1]) ? 1 : -1;
}

// Return zero for every shape outside the closed built-in class family.
inline uint32_t deferredSpecializedMaterialClass(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines)
{
    if (program != SceneShaderProgram::DeferredBinned || options.customMaterials ||
        sceneWorkingColorSpace() != SceneWorkingColorSpace::ACEScg || extraDefines.size() != 1 ||
        std::string_view(extraDefines[0].name) != "MATERIAL_CLASS") { return 0; }
    const std::string_view value(extraDefines[0].value);
    return value.size() == 1 && value[0] >= '1' && value[0] <= '4'
        ? static_cast<uint32_t>(value[0] - '0') : 0;
}

// -1 keeps the original static request; 0/1 select the closed native precision policy.
inline int32_t deferredPrecisionSpecializationValue(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines)
{
    if (options.customMaterials || sceneWorkingColorSpace() != SceneWorkingColorSpace::ACEScg) { return -1; }
    if ((program == SceneShaderProgram::Deferred && extraDefines.empty()) ||
        deferredSpecializedMaterialClass(program, options, extraDefines) != 0u) {
        return options.deferredFloat16 ? 1 : 0;
    }
    return -1;
}

// The source request deliberately excludes pipeline values. Each native program
// owns the policy tuple separately; the arrays outlive borrowed creation spans.
// IDs match the closed original-SDK auto-ID source gate; native creation validates them.
inline std::span<const ShaderSpecializationConstant> sceneShaderSpecializationConstants(
    SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines = {})
{
    // Bound to the exact four-way source module: cache Bool0, guide Bool1.
    // Other modules, including Deferred's precision/class tuples, keep their own IDs.
    static constexpr ShaderSpecializationConstant cachePolicies[4][2] = {
        {{0, ShaderSpecializationType::Bool32, 0}, {1, ShaderSpecializationType::Bool32, 0}},
        {{0, ShaderSpecializationType::Bool32, 0}, {1, ShaderSpecializationType::Bool32, 1}},
        {{0, ShaderSpecializationType::Bool32, 1}, {1, ShaderSpecializationType::Bool32, 0}},
        {{0, ShaderSpecializationType::Bool32, 1}, {1, ShaderSpecializationType::Bool32, 1}},
    };
    const int32_t cacheQuery = openPBRCacheQuerySpecializationValue(program, options, extraDefines);
    if (cacheQuery >= 0) {
        const uint32_t guide = program == SceneShaderProgram::OpenPBRPathTraceGuides ? 1u : 0u;
        return std::span<const ShaderSpecializationConstant>(cachePolicies[uint32_t(cacheQuery) * 2u + guide]);
    }
    // Single's independently closed source has only cache Bool0, not Quad's guide Bool1.
    static constexpr ShaderSpecializationConstant singleGuideCachePolicies[2] = {
        {0, ShaderSpecializationType::Bool32, 0}, {0, ShaderSpecializationType::Bool32, 1},
    };
    const int32_t singleGuideCacheQuery = singleGuideCacheQuerySpecializationValue(program, options, extraDefines);
    if (singleGuideCacheQuery >= 0) { return {&singleGuideCachePolicies[singleGuideCacheQuery], 1}; }
    // Bound to Std3's actual source: cache Bool0 and guide Bool1, independent of Quad.
    // Guide QUERY is excluded by the strict predicate; it has no fourth native tuple.
    static constexpr ShaderSpecializationConstant standardCacheGuidePolicies[3][2] = {
        {{0, ShaderSpecializationType::Bool32, 0}, {1, ShaderSpecializationType::Bool32, 0}},
        {{0, ShaderSpecializationType::Bool32, 1}, {1, ShaderSpecializationType::Bool32, 0}},
        {{0, ShaderSpecializationType::Bool32, 0}, {1, ShaderSpecializationType::Bool32, 1}},
    };
    const int32_t standardCacheQuery = standardPathCacheQuerySpecializationValue(program, options, extraDefines);
    if (standardCacheQuery >= 0) {
        const uint32_t policy = program == SceneShaderProgram::PathTraceGuides ? 2u : uint32_t(standardCacheQuery);
        return std::span<const ShaderSpecializationConstant>(standardCacheGuidePolicies[policy]);
    }
    const int32_t precision = deferredPrecisionSpecializationValue(program, options, extraDefines);
    if (precision < 0) { return {}; }
    // Observed flat source: precision Bool0. Always supply the selected value.
    static constexpr ShaderSpecializationConstant precisionPolicies[2] = {
        {0, ShaderSpecializationType::Bool32, 0}, {0, ShaderSpecializationType::Bool32, 1},
    };
    if (program == SceneShaderProgram::Deferred) { return {&precisionPolicies[precision], 1}; }
    // Observed binned source: fixed metalness Int0, opaque Bool1, precision Bool2.
    // Each of the eight complete tuples owns storage for the borrowed span.
    static constexpr ShaderSpecializationConstant policies[8][3] = {
        {{0, ShaderSpecializationType::Int32, 0}, {1, ShaderSpecializationType::Bool32, 1}, {2, ShaderSpecializationType::Bool32, 0}},
        {{0, ShaderSpecializationType::Int32, 0}, {1, ShaderSpecializationType::Bool32, 1}, {2, ShaderSpecializationType::Bool32, 1}},
        {{0, ShaderSpecializationType::Int32, 1}, {1, ShaderSpecializationType::Bool32, 1}, {2, ShaderSpecializationType::Bool32, 0}},
        {{0, ShaderSpecializationType::Int32, 1}, {1, ShaderSpecializationType::Bool32, 1}, {2, ShaderSpecializationType::Bool32, 1}},
        {{0, ShaderSpecializationType::Int32, 0xffffffffu}, {1, ShaderSpecializationType::Bool32, 1}, {2, ShaderSpecializationType::Bool32, 0}},
        {{0, ShaderSpecializationType::Int32, 0xffffffffu}, {1, ShaderSpecializationType::Bool32, 1}, {2, ShaderSpecializationType::Bool32, 1}},
        {{0, ShaderSpecializationType::Int32, 0xffffffffu}, {1, ShaderSpecializationType::Bool32, 0}, {2, ShaderSpecializationType::Bool32, 0}},
        {{0, ShaderSpecializationType::Int32, 0xffffffffu}, {1, ShaderSpecializationType::Bool32, 0}, {2, ShaderSpecializationType::Bool32, 1}},
    };
    const uint32_t materialClass = deferredSpecializedMaterialClass(program, options, extraDefines);
    return materialClass ? std::span<const ShaderSpecializationConstant>(policies[(materialClass - 1) * 2u + uint32_t(precision)])
        : std::span<const ShaderSpecializationConstant>{};
}

inline ShaderRequest makeSceneShaderRequest(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines = {})
{
    if ((program == SceneShaderProgram::OpenPBRPathTrace || program == SceneShaderProgram::OpenPBRPathTraceGuides) &&
        std::any_of(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
            return std::string_view(define.name) == "METALLIC_PT_CACHE_QUERY_SPECIALIZATION" ||
                std::string_view(define.name) == "METALLIC_PT_GUIDE_SPECIALIZATION" ||
                std::string_view(define.name) == "METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION";
        })) {
        throw std::invalid_argument("OpenPBR cache/guide specialization markers are reserved by the request factory");
    }
    if (program == SceneShaderProgram::PathTrace &&
        std::any_of(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
            return std::string_view(define.name) == "METALLIC_STANDARD_PT_CACHE_QUERY_SPECIALIZATION";
        })) {
        throw std::invalid_argument("METALLIC_STANDARD_PT_CACHE_QUERY_SPECIALIZATION is reserved by the standard PathTrace request factory");
    }
    if ((program == SceneShaderProgram::PathTrace || program == SceneShaderProgram::PathTraceGuides) &&
        std::any_of(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
            return std::string_view(define.name) == "METALLIC_STANDARD_PT_GUIDE_SPECIALIZATION";
        })) {
        throw std::invalid_argument("METALLIC_STANDARD_PT_GUIDE_SPECIALIZATION is reserved by the standard PathTrace request factory");
    }
    const int32_t standardCacheQuery = standardPathCacheQuerySpecializationValue(program, options, extraDefines);
    const int32_t cacheQuery = openPBRCacheQuerySpecializationValue(program, options, extraDefines);
    const int32_t singleGuideCacheQuery = singleGuideCacheQuerySpecializationValue(program, options, extraDefines);
    if (standardCacheQuery >= 0) { program = SceneShaderProgram::PathTrace; }
    if (cacheQuery >= 0) { program = SceneShaderProgram::OpenPBRPathTrace; }
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
    const uint32_t specializedClass = deferredSpecializedMaterialClass(program, options, extraDefines);
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
    // Deferred camera selection uses a push-buffer flag. Preserve all other ordered defines.
    if (deferred) { request.defines.erase(request.defines.begin() + 1); }
    // Preserve non-Deferred descriptors; Deferred guides are a runtime output policy.
    if (!deferred) { request.defines.emplace_back("METALLIC_DEFERRED_UPSCALER_GUIDES", "0"); }
    if (deferred) {
        const bool explicitRuntimePrecision = std::any_of(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
            return std::string_view(define.name) == "METALLIC_DEFERRED_RUNTIME_PRECISION";
        });
        // The runtime selector is factory-owned; explicit static overrides stay ordered.
        if (explicitRuntimePrecision) {
            throw std::invalid_argument("METALLIC_DEFERRED_RUNTIME_PRECISION is reserved by the Deferred request factory");
        }
        if (std::any_of(extraDefines.begin(), extraDefines.end(), [](const auto& define) {
                return std::string_view(define.name) == "METALLIC_DEFERRED_CLASS_SPECIALIZATION";
            })) {
            throw std::invalid_argument("METALLIC_DEFERRED_CLASS_SPECIALIZATION is reserved by the Deferred request factory");
        }
        // Only the closed builtin shapes are independent of the old precision macro.
        // Arbitrary values/custom sources may reference METALLIC_DEFERRED_FP16.
        if (deferredPrecisionSpecializationValue(program, options, extraDefines) >= 0) {
            request.defines.emplace_back("METALLIC_DEFERRED_RUNTIME_PRECISION", "1");
        } else {
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
    }
    if (cacheQuery >= 0) {
        request.defines.emplace_back("SHARC_QUERY", "1");
        request.defines.emplace_back("METALLIC_PT_CACHE_QUERY_SPECIALIZATION", "1");
        request.defines.emplace_back("METALLIC_PT_GUIDE_SPECIALIZATION", "1");
    } else if (singleGuideCacheQuery >= 0) {
        request.defines.emplace_back("SHARC_QUERY", "1");
        request.defines.emplace_back("METALLIC_PT_SINGLE_SAMPLE", "1");
        request.defines.emplace_back("METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION", "1");
    } else if (standardCacheQuery >= 0) {
        request.defines.emplace_back("SHARC_QUERY", "1");
        request.defines.emplace_back("METALLIC_STANDARD_PT_CACHE_QUERY_SPECIALIZATION", "1");
        request.defines.emplace_back("METALLIC_STANDARD_PT_GUIDE_SPECIALIZATION", "1");
    } else if (specializedClass) {
        request.defines.emplace_back("METALLIC_DEFERRED_CLASS_SPECIALIZATION", "1");
    } else {
        for (const auto& define : extraDefines) { request.defines.emplace_back(define.name, define.value); }
    }
    if (options.customMaterials) { request.searchPaths.push_back(options.materialInclude); }
    if (!deferred && options.hasRTXCR) { request.searchPaths.push_back(options.rtxcrInclude); }
    return request;
}

enum class OpenPBRRayPipelineStage { RayGeneration, Miss, ClosestHit, AnyHit };

// A separate material model and integrator. Do not inherit the OpenPBR cache,
// LUT, native specialization constants, or custom-material source policy.
inline ShaderRequest makeVkGltfRayPipelineShaderRequest(bool guides, bool singleSample, bool reorder,
    OpenPBRRayPipelineStage stage = OpenPBRRayPipelineStage::RayGeneration)
{
    ShaderRequest request;
    request.module = stage == OpenPBRRayPipelineStage::RayGeneration
        ? "Features/PathTracing/VkGltfRayPipelinePathTrace" : "Features/PathTracing/VkGltfRayPipelineHit";
    switch (stage) {
    case OpenPBRRayPipelineStage::RayGeneration: request.entry = "vkGltfRayPipelinePathTraceMain"; break;
    case OpenPBRRayPipelineStage::Miss: request.entry = "vkGltfRayPipelineMiss"; break;
    case OpenPBRRayPipelineStage::ClosestHit: request.entry = "vkGltfRayPipelineClosestHit"; break;
    case OpenPBRRayPipelineStage::AnyHit: request.entry = "vkGltfRayPipelineAnyHit"; break;
    default: throw std::invalid_argument("Unknown glTF ray-pipeline shader stage");
    }
    const bool rayGeneration = stage == OpenPBRRayPipelineStage::RayGeneration;
    request.defines = {{"METALLIC_CUSTOM_MATERIALS", "0"}, {"METALLIC_GLOBAL_VIEW", "0"},
        {"METALLIC_HAS_RTXCR", "0"}, {"METALLIC_VKGLTF_GUIDES", guides && rayGeneration ? "1" : "0"},
        {"METALLIC_PT_SINGLE_SAMPLE", singleSample && rayGeneration ? "1" : "0"},
        {"METALLIC_VKGLTF_SER", reorder && rayGeneration ? "1" : "0"}};
    request.capabilities.emplace_back("spvRayTracingKHR");
    if (reorder && rayGeneration) { request.capabilities.emplace_back("spvShaderInvocationReorderEXT"); }
    return request;
}

// Same glTF material and integrator, through the compile-time BSDF interface.
// Keep the direct factory and all hit-stage identities unchanged for controlled A/Bs.
inline ShaderRequest makeSimpleBSDFRayPipelineShaderRequest(bool guides, bool singleSample, bool reorder,
    OpenPBRRayPipelineStage stage = OpenPBRRayPipelineStage::RayGeneration)
{
    auto request = makeVkGltfRayPipelineShaderRequest(guides, singleSample, reorder, stage);
    if (stage == OpenPBRRayPipelineStage::RayGeneration) {
        request.defines.emplace_back("METALLIC_SIMPLE_BSDF", "1");
    }
    return request;
}

// Experimental traversal backend; preserve the canonical source policy and its
// separately owned sceneShaderSpecializationConstants tuple on ray generation.
// This first path supports cache off only. An empty source policy can still use
// SHARC_QUERY=1 internally with the original native cache Bool set to false.
inline ShaderRequest makeOpenPBRRayPipelineShaderRequest(SceneShaderProgram program, const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines = {}, OpenPBRRayPipelineStage stage = OpenPBRRayPipelineStage::RayGeneration)
{
    const bool singleGuide = program == SceneShaderProgram::OpenPBRPathTraceGuides && extraDefines.size() == 1 &&
        std::string_view(extraDefines[0].name) == "METALLIC_PT_SINGLE_SAMPLE" && std::string_view(extraDefines[0].value) == "1";
    if ((program != SceneShaderProgram::OpenPBRPathTrace && program != SceneShaderProgram::OpenPBRPathTraceGuides) ||
        (!extraDefines.empty() && !singleGuide)) {
        throw std::invalid_argument("OpenPBR ray pipeline requires cache off and a supported OpenPBR source policy");
    }
    auto request = makeSceneShaderRequest(program, options, extraDefines);
    switch (stage) {
    case OpenPBRRayPipelineStage::RayGeneration:
        // Guide policies can canonicalize to the shared plain source. Choose
        // the wrapper from that source, not from the requested output policy.
        if (request.module == "Features/PathTracing/OpenPBRRayQueryPathTrace") {
            request.module = "Features/PathTracing/OpenPBRRayPipelinePathTrace";
            request.entry = "openPbrRayPipelinePathTraceMain";
        } else {
            request.module = "Features/PathTracing/OpenPBRRayPipelinePathTraceGuides";
            request.entry = "openPbrRayPipelinePathTraceGuidesMain";
        }
        break;
    case OpenPBRRayPipelineStage::Miss:
        request.module = "Features/PathTracing/OpenPBRRayPipelineHit";
        request.entry = "openPbrRayPipelineMiss";
        break;
    case OpenPBRRayPipelineStage::ClosestHit:
        request.module = "Features/PathTracing/OpenPBRRayPipelineHit";
        request.entry = "openPbrRayPipelineClosestHit";
        break;
    case OpenPBRRayPipelineStage::AnyHit:
        request.module = "Features/PathTracing/OpenPBRRayPipelineHit";
        request.entry = "openPbrRayPipelineAnyHit";
        break;
    default:
        throw std::invalid_argument("Unknown OpenPBR ray-pipeline shader stage");
    }
    request.defines.emplace_back("METALLIC_PT_RAY_PIPELINE", "1");
    request.capabilities.emplace_back("spvRayTracingKHR");
    return request;
}

// Opt-in kernel split. Keep the original Single source policy and specialization
// ownership; only the entry point and the continuation's primary-hit source differ.
inline ShaderRequest makeOpenPBRPrimarySplitShaderRequest(const SceneShaderOptions& options,
    std::span<const SlangMacroDefine> extraDefines, bool primaryStage)
{
    if (extraDefines.size() != 1 || std::string_view(extraDefines[0].name) != "METALLIC_PT_SINGLE_SAMPLE" ||
        std::string_view(extraDefines[0].value) != "1") {
        throw std::invalid_argument("Primary guides split requires single-sample guides with cache off");
    }
    auto request = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, extraDefines);
    request.module = primaryStage ? "Features/PathTracing/OpenPBRPrimaryGuides" : "Features/PathTracing/OpenPBRPathContinuation";
    request.entry = primaryStage ? "openPbrPrimaryGuidesMain" : "openPbrPathContinuationMain";
    if (!primaryStage) { request.defines.emplace_back("METALLIC_PT_SPLIT_PRIMARY", "1"); }
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
