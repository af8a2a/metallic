#pragma once

#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/Core/ColorResizeParameters.h"
#include "Runtime/Render/Core/LightingKernelParameters.h"
#include "Runtime/Render/Core/PathTraceStageParameters.h"
#include "Runtime/Render/Core/PostProcessParameters.h"
#include "Runtime/Render/Core/RTXDIPostProcessParameters.h"
#include "Runtime/Render/Core/StreamlineDLSSParameters.h"
#include "Runtime/Render/Environment/AtmosphereKernelParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/VisibilityHybridRasterizer.h"

#include <optional>

namespace metallic::render {

struct NativeComputeWarmupVariant {
    uint32_t userPushDataSize = 0;
    bool indirectBindable = false;
    std::vector<ShaderSpecializationConstant> specializationConstants;

    bool operator==(const NativeComputeWarmupVariant&) const = default;
};

namespace native_warmup_detail {

// Only audited runtime entry points are owned here. A newly added compute entry
// must declare its actual pipeline ABI; it never inherits a guessed 12-byte root.
inline std::optional<uint32_t> userPushDataSize(const ShaderRequest& request)
{
    const auto matches = [&](std::string_view module, std::initializer_list<std::string_view> entries) {
        return request.module == module && std::find(entries.begin(), entries.end(), request.entry) != entries.end();
    };
    if (matches("Features/Debug/LightGridDebug", {"lightGridDebugMain"})) { return static_cast<uint32_t>(sizeof(LightGridDebugParams)); }
    if (matches("Features/Debug/SliderDebug", {"sliderDebugMain", "sliderDebugOverlayMain"})) { return static_cast<uint32_t>(sizeof(SliderDebugParams)); }
    if (matches("Features/Environment/EnvironmentLightingPrecompute", {"environmentLightingPrecomputeMain"})) {
        return static_cast<uint32_t>(sizeof(EnvironmentLightingPrecomputeParams));
    }
    if (matches("Features/Environment/Transmittance", {"transmittanceMain"}) ||
        matches("Features/Environment/MultiScattering", {"multiScatteringMain"}) ||
        matches("Features/Environment/SkyView", {"skyViewMain"}) ||
        matches("Features/Environment/EnvironmentCapture", {"environmentCaptureMain"}) ||
        matches("Features/Environment/AerialPerspective", {"aerialPerspectiveMain"}) ||
        matches("Features/Environment/CloudShadow", {"cloudShadowMain"})) { return static_cast<uint32_t>(sizeof(AtmospherePrecomputeParams)); }
    if (matches("Features/Lighting/BuildReGIR", {"buildReGIRMain"})) { return static_cast<uint32_t>(sizeof(BuildReGIRParams)); }
    if (matches("Features/Lighting/ClusterLightGrid", {"clusterLightGridMain"})) { return static_cast<uint32_t>(sizeof(ClusterLightGridBuildParams)); }
    if (matches("Features/Lighting/PrepareLightsPdf", {"prepareLightsPdfMain"})) { return static_cast<uint32_t>(sizeof(PrepareLightsPdfParams)); }
    if (matches("Features/PostProcess/AutoExposure", {"autoExposureHistogramMain", "autoExposureReduceMain", "autoExposureApplyMain"})) {
        return static_cast<uint32_t>(sizeof(AutoExposureParams));
    }
    if (matches("Features/PostProcess/FinalBlit", {"finalBlitMain", "finalBlitUvMain"})) { return static_cast<uint32_t>(sizeof(FinalBlitParams)); }
    if (matches("Features/PostProcess/ColorResize", {"colorResizeMain"})) { return static_cast<uint32_t>(sizeof(ColorResizeParams)); }
    if (matches("Features/ReSTIR/RTXDIComposite", {"rtxdiCompositeMain"})) { return static_cast<uint32_t>(sizeof(RTXDICompositeParams)); }
    if (matches("Features/ReSTIR/RTXDIConfidence", {"rtxdiConfidenceMain"})) { return static_cast<uint32_t>(sizeof(RTXDIConfidenceParams)); }
    if (matches("Features/PathTracing/SceneSharcMaintenance", {"sharcClearMain", "sharcResolveMain"})) {
        return static_cast<uint32_t>(sizeof(SharcMaintenanceParams));
    }
    if (matches("Features/SmokeTests/RenderGraphBuffer", {"renderGraphBufferWriteMain", "renderGraphBufferCopyMain"})) {
        return static_cast<uint32_t>(sizeof(builtin_pass::RenderGraphBufferUserPush));
    }
    if (matches("Features/PostProcess/StreamlineDLSSSupport", {"streamlineDlssAlphaMain"})) {
        return static_cast<uint32_t>(sizeof(StreamlineDLSSAlphaUserPush));
    }
    if (matches("Features/GPUDriven/GPUDrivenCulling", {"gpuDrivenPreviewResetMain", "gpuDrivenPreviewInstanceCullMain", "gpuDrivenPreviewHzbMain"}) ||
        matches("Features/GPUDriven/HZBSPD", {"hzbSpdMain"})) {
        // VisibilityBufferPass uses its common createCompute helper for SPD too.
        // Dispatching HZBSPDUserPush does not shrink that pipeline's layout.
        return static_cast<uint32_t>(sizeof(builtin_pass::GPUDrivenPreviewUserPush));
    }
    if (matches("Features/GPUDriven/GPUDrivenStreamAsset", {
            "streamClusterBinMain", "streamClusterBinP0Main", "gpuDrivenStreamAssetCullResetMain",
            "gpuDrivenStreamAssetInstanceCullMain", "gpuDrivenStreamAssetHzbMain", "gpuDrivenStreamAssetTraversalMain",
            "streamDistributedDemandMain", "streamCooperativeLodMain", "gpuDrivenStreamAssetBuildActiveMain",
            "gpuDrivenStreamAssetBuildBlasInputMain", "gpuDrivenStreamAssetBuildTlasInputMain", "streamClusterPrepareMain",
            "streamClusterCullMain", "streamClusterCullP0Main", "gpuDrivenStreamAssetDeferredMain",
            "gpuDrivenStreamAssetInitializePageTableMain", "gpuDrivenStreamAssetApplyUpdatesMain",
            "streamClusterRasterMain", "streamClusterRasterLegacyMain", "streamClusterRasterPlaneMain", "streamClusterRasterCooperativeMain"}) ||
        matches("Features/GPUDriven/GPUDrivenStreamWorkload", {"streamWorkloadResetMain", "streamWorkloadMain"}) ||
        matches("Features/GPUDriven/GPUDrivenStreamWorkRaster", {"streamClusterRasterWorkBinsMain", "streamClusterRasterWorkControlMain"}) ||
        matches("Features/GPUDriven/GPUDrivenStreamGroupRaster", {"streamClusterRasterGroup32Main"})) { return static_cast<uint32_t>(sizeof(MeshletStreamUserPush)); }
    if (matches("Features/VisibilityBuffer/VisibilityHybridRaster", {"hybridResetMain", "hybridArgumentsMain", "hybridRasterMain",
            "hybridClusterResetMain", "hybridClusterHistogramMain", "hybridClusterArgumentsMain", "hybridClusterScatterMain"})) {
        return VisibilityHybridRasterizer::pipelineUserPushDataSize();
    }
    if (matches("Features/Debug/GPUProbe", {"probe"}) ||
        matches("Features/Debug/SceneRayQueryVisualize", {"sceneRayQueryVisualizeMain"}) ||
        matches("Features/Debug/SceneMaterialVisualize", {"sceneMaterialVisualizeMain"}) ||
        matches("Features/PostProcess/ColorGradingLUT", {"composeColorGradingLUT"}) ||
        matches("Features/PostProcess/UpscalerGuideResolve", {"upscalerGuideResolveMain"}) ||
        matches("Features/Strands/NativeStrands", {"strandVisibilityMain", "strandLightingMain"}) ||
        matches("Features/PathTracing/RayMaterialQueue", {"rayMaterialResetMain", "rayMaterialCountMain", "rayMaterialPrefixMain", "rayMaterialScatterMain"}) ||
        matches("Features/ReSTIR/SceneRTXDI", {"sceneRtxdiMain"}) ||
        matches("Features/Samples/RTXCRMaterialSample", {"rtxcrMaterialSampleMain"}) ||
        matches("Features/VisibilityBuffer/VisibilityBufferMaterial", {"visibilityBufferMaterialMain"}) ||
        matches("Features/VisibilityBuffer/VisibilityMaterialBinning", {"materialBinningResetMain", "materialBinningCountMain",
            "materialBinningAllocateMain", "materialBinningClassifyMain", "materialBinningArgumentsMain"}) ||
        matches("Features/Lighting/ScreenSpaceShadows", {"rayTracedShadowsMain"})) { return static_cast<uint32_t>(sizeof(GPUBufferSpan)); }
    for (const auto program : {SceneShaderProgram::PathTrace, SceneShaderProgram::PathTraceGuides,
            SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides,
            SceneShaderProgram::RealtimeLighting, SceneShaderProgram::Deferred, SceneShaderProgram::DeferredBinned}) {
        const auto identity = sceneShaderIdentity(program);
        if (request.module == identity.module && request.entry == identity.entry) { return static_cast<uint32_t>(sizeof(GPUBufferSpan)); }
    }
    if (matches("Features/PathTracing/OpenPBRPrimaryGuides", {"openPbrPrimaryGuidesMain"}) ||
        matches("Features/PathTracing/OpenPBRPathContinuation", {"openPbrPathContinuationMain"})) { return static_cast<uint32_t>(sizeof(GPUBufferSpan)); }
    return std::nullopt;
}

inline bool sameSourcePolicy(ShaderRequest generated, const ShaderRequest& requested)
{
    // These acquisition controls do not change which runtime specialization
    // tuple belongs to the source. Keep all ordered source defines/paths intact.
    generated.descriptorHeapMode = requested.descriptorHeapMode;
    generated.descriptorHeapStrides = requested.descriptorHeapStrides;
    generated.optimizationLevel = requested.optimizationLevel;
    return generated == requested;
}

} // namespace native_warmup_detail

// An empty result is an explicit unowned ABI/policy, not a default descriptor.
// The caller separately checks the compiled execution model before creating PSOs.
// All owned entries use bindless descriptors and ordinary (not DGC) pipelines.
inline std::vector<NativeComputeWarmupVariant> nativeComputeWarmupVariants(const ShaderRequest& request)
{
    const auto pushSize = native_warmup_detail::userPushDataSize(request);
    if (!pushSize) { return {}; }
    const auto define = [&](std::string_view name) -> std::string_view {
        const auto found = std::find_if(request.defines.begin(), request.defines.end(), [&](const auto& value) { return value.first == name; });
        return found == request.defines.end() ? std::string_view{} : std::string_view(found->second);
    };
    const bool splitPrimary = request.module == "Features/PathTracing/OpenPBRPrimaryGuides";
    const bool splitContinuation = request.module == "Features/PathTracing/OpenPBRPathContinuation";
    const bool specialized = std::any_of(request.defines.begin(), request.defines.end(), [](const auto& value) {
        return value.first == "METALLIC_PT_CACHE_QUERY_SPECIALIZATION" || value.first == "METALLIC_PT_GUIDE_SPECIALIZATION" ||
            value.first == "METALLIC_PT_SINGLE_CACHE_QUERY_SPECIALIZATION" || value.first == "METALLIC_STANDARD_PT_CACHE_QUERY_SPECIALIZATION" ||
            value.first == "METALLIC_STANDARD_PT_GUIDE_SPECIALIZATION" || value.first == "METALLIC_DEFERRED_RUNTIME_PRECISION" ||
            value.first == "METALLIC_DEFERRED_CLASS_SPECIALIZATION";
    });
    if (!specialized && !splitPrimary && !splitContinuation) { return {{*pushSize, false, {}}}; }
    SceneShaderOptions options{.customMaterials = define("METALLIC_CUSTOM_MATERIALS") == "1",
        .globalView = define("METALLIC_GLOBAL_VIEW") == "1", .hasRTXCR = define("METALLIC_HAS_RTXCR") == "1",
        .customCoverage = define("METALLIC_CUSTOM_COVERAGE") == "1"};
    if (options.customMaterials && !request.searchPaths.empty()) { options.materialInclude = request.searchPaths.front(); }
    const size_t coveragePathIndex = options.customMaterials ? 1u : 0u;
    if (options.customCoverage && request.searchPaths.size() > coveragePathIndex) {
        options.coverageInclude = request.searchPaths[coveragePathIndex];
    }
    if (options.hasRTXCR && !request.searchPaths.empty()) { options.rtxcrInclude = request.searchPaths.back(); }
    std::vector<NativeComputeWarmupVariant> variants;
    const auto append = [&](std::span<const ShaderSpecializationConstant> constants) {
        NativeComputeWarmupVariant variant{*pushSize, false, {constants.begin(), constants.end()}};
        if (std::find(variants.begin(), variants.end(), variant) == variants.end()) { variants.push_back(std::move(variant)); }
    };
    if (splitPrimary || splitContinuation) {
        const SlangMacroDefine single{"METALLIC_PT_SINGLE_SAMPLE", "1"};
        if (native_warmup_detail::sameSourcePolicy(makeOpenPBRPrimarySplitShaderRequest(options, {&single, 1}, splitPrimary), request)) {
            if (splitPrimary) { append({}); }
            else { append(sceneShaderSpecializationConstants(SceneShaderProgram::OpenPBRPathTraceGuides, options, {&single, 1})); }
        }
        return variants;
    }
    const auto addPolicy = [&](SceneShaderProgram program, const SceneShaderOptions& policy,
                               std::span<const SlangMacroDefine> defines = {}) {
        if (native_warmup_detail::sameSourcePolicy(makeSceneShaderRequest(program, policy, defines), request)) {
            append(sceneShaderSpecializationConstants(program, policy, defines));
        }
    };
    const SlangMacroDefine query{"SHARC_QUERY", "1"}, single{"METALLIC_PT_SINGLE_SAMPLE", "1"};
    const SlangMacroDefine querySingle[]{query, single};
    for (const auto program : {SceneShaderProgram::PathTrace, SceneShaderProgram::PathTraceGuides,
            SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides}) {
        addPolicy(program, options);
        addPolicy(program, options, {&query, 1});
        if (program == SceneShaderProgram::OpenPBRPathTraceGuides) {
            addPolicy(program, options, {&single, 1});
            addPolicy(program, options, querySingle);
        }
    }
    for (const bool half : {false, true}) {
        auto precisionOptions = options;
        precisionOptions.deferredFloat16 = half;
        addPolicy(SceneShaderProgram::Deferred, precisionOptions);
        for (uint32_t materialClass = 0; materialClass < 5; ++materialClass) {
            const auto value = std::to_string(materialClass);
            const SlangMacroDefine material{"MATERIAL_CLASS", value.c_str()};
            addPolicy(SceneShaderProgram::DeferredBinned, precisionOptions, {&material, 1});
        }
    }
    return variants;
}

} // namespace metallic::render
