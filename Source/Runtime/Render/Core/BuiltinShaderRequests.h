#pragma once

#include "Runtime/Render/Core/ShaderRequests.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace metallic::render {

// This is an explicit list of runtime requests, not a scan of every .slang file:
// many files are modules/includes, and identical entry points with different
// ordered defines/capabilities have different runtime cache keys.
// Variant options use the same factories as the owning runtime passes.
inline std::vector<ShaderRequest> builtinShaderWarmupRequests(const std::string& rtxcrInclude)
{
    std::vector<ShaderRequest> requests;
    auto add = [&](const char* module, std::initializer_list<const char*> entries,
                   std::vector<std::string> capabilities = {},
                   std::vector<std::pair<std::string, std::string>> defines = {},
                   std::vector<std::string> searchPaths = {}) {
        for (const char* entry : entries) {
            requests.push_back({module, entry, capabilities, defines, searchPaths});
        }
    };

    add("Features/Debug/GPUProbe", {"probe"});
    add("Features/Debug/LightGridDebug", {"lightGridDebugMain"});
    add("Features/Debug/SliderDebug", {"sliderDebugMain", "sliderDebugOverlayMain"});
    add("Features/Environment/EnvironmentLightingPrecompute", {"environmentLightingPrecomputeMain"});
    add("Features/GPUDriven/GPUDrivenCulling", {"gpuDrivenPreviewResetMain", "gpuDrivenPreviewInstanceCullMain", "gpuDrivenPreviewHzbMain"});
    add("Features/GPUDriven/GPUDrivenStreamWorkload", {"streamWorkloadResetMain", "streamWorkloadMain"});
    add("Features/GPUDriven/GPUDrivenStreamWorkRaster", {"streamClusterRasterWorkBinsMain", "streamClusterRasterWorkControlMain"});
    add("Features/GPUDriven/GPUDrivenStreamGroupRaster", {"streamClusterRasterGroup32Main"});
    add("Features/GPUDriven/ResidentMeshletLOD", {"residentLodResetMain", "residentLodSelectMain", "residentLodArgumentsMain", "residentLodScatterMain"});
    add("Features/GPUDriven/GPUDrivenStreamAsset", {"streamClusterBinMain", "streamClusterBinP0Main"});
    add("Features/Lighting/BuildReGIR", {"buildReGIRMain"});
    add("Features/Lighting/ClusterLightGrid", {"clusterLightGridMain"});
    add("Features/Lighting/PrepareLightsPdf", {"prepareLightsPdfMain"});
    add("Features/PostProcess/AutoExposure", {"autoExposureHistogramMain", "autoExposureReduceMain", "autoExposureApplyMain"});
    add("Features/PostProcess/EditorDisplay", {"editorDisplayVertex", "editorDisplayFragment"});
    add("Features/PostProcess/FinalBlit", {"finalBlitUvMain", "finalBlitMain"}, {}, {{"FINAL_USE_LUT", "0"}});
    add("Features/PostProcess/FinalBlit", {"finalBlitMain"}, {}, {{"FINAL_USE_LUT", "1"}});
    add("Features/PostProcess/ColorGradingLUT", {"composeColorGradingLUT"});
    add("Features/PostProcess/StreamlineDLSSSupport", {"streamlineDlssDepthVertexMain", "streamlineDlssDepthFragmentMain", "streamlineDlssAlphaMain"});
    add("Features/PostProcess/UpscalerGuideResolve", {"upscalerGuideResolveMain"});
    requests.push_back(makeColorResizeShaderRequest());
    add("Features/ReSTIR/RTXDIComposite", {"rtxdiCompositeMain"});
    add("Features/ReSTIR/RTXDIConfidence", {"rtxdiConfidenceMain"});
    add("Features/Samples/BunnyWireframe", {"bunnyWireframeVertexMain", "bunnyWireframeFragmentMain"});
    add("Features/Samples/ImageSample", {"imageSampleVertexMain", "imageSampleFragmentMain"});
    add("Features/Samples/MaterialShaderObject", {"materialShaderObjectVertexMain", "materialShaderObjectFragmentMain", "materialShaderObjectAlternateFragmentMain"});
    add("Features/SmokeTests/RenderGraphBuffer", {"renderGraphBufferWriteMain", "renderGraphBufferCopyMain"});
    add("Features/VisibilityBuffer/VisibilityBufferComposite", {"visibilityBufferCompositeVertexMain", "visibilityBufferCompositeFragmentMain"});
    add("Features/VisibilityBuffer/VisibilityBufferMaterial", {"visibilityBufferMaterialMain"});
    add("Features/VisibilityBuffer/VisibilityHybridRaster", {"hybridResetMain", "hybridArgumentsMain", "hybridRasterMain", "hybridResolveVertexMain", "hybridResolveFragmentMain", "hybridClusterResetMain", "hybridClusterHistogramMain", "hybridClusterArgumentsMain", "hybridClusterScatterMain"});
    add("Features/Samples/Triangle", {"triangleVertexMain", "triangleFragmentMain"});
    add("Features/VisibilityBuffer/VisibilityMaterialBinning",
        {"materialBinningResetMain", "materialBinningCountMain", "materialBinningAllocateMain",
            "materialBinningClassifyMain", "materialBinningArgumentsMain"},
        {"spvGroupNonUniformBallot", "spvGroupNonUniformArithmetic"});
    add("Features/Debug/SceneRayQueryVisualize", {"sceneRayQueryVisualizeMain"}, {"spvRayQueryKHR"});
    add("Features/Debug/SceneRayQueryVisualize", {"sceneRayQueryVisualizeMain"},
        {"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV"},
        {{"SCENE_RAYQUERY_ENABLE_CLUSTER_ID", "1"}});

    const char* streamModule = "Features/GPUDriven/GPUDrivenStreamAsset";
    add(streamModule, {"gpuDrivenStreamAssetCullResetMain", "gpuDrivenStreamAssetInstanceCullMain",
        "gpuDrivenStreamAssetHzbMain", "gpuDrivenStreamAssetTraversalMain",
        "streamDistributedDemandMain", "streamCooperativeLodMain",
        "gpuDrivenStreamAssetBuildActiveMain", "gpuDrivenStreamAssetBuildBlasInputMain",
        "gpuDrivenStreamAssetBuildTlasInputMain", "streamClusterPrepareMain",
        "streamClusterCullMain", "streamClusterCullP0Main",
        "gpuDrivenStreamAssetDeferredMain", "gpuDrivenStreamAssetCompositeVertexMain",
        "gpuDrivenStreamAssetCompositeFragmentMain", "gpuDrivenStreamAssetInitializePageTableMain",
        "gpuDrivenStreamAssetApplyUpdatesMain", "gpuDrivenStreamAssetFragmentMain",
        "streamClusterRasterMain", "streamClusterRasterLegacyMain",
        "streamClusterRasterPlaneMain", "streamClusterRasterCooperativeMain"});
    add(streamModule, {"gpuDrivenStreamAssetMeshMain"}, {"spvMeshShadingEXT"});
    const std::vector<std::string> meshCapabilities{"spvMeshShadingEXT", "spvGroupNonUniformBallot"};
    add(streamModule, {"gpuDrivenStreamAssetMeshMain", "streamTessellationMesh"}, meshCapabilities);
    add(streamModule, {"streamTessellationFragment"});
    const char* visibilityModule = "Features/VisibilityBuffer/VisibilityBuffer";
    add(visibilityModule, {"visibilityBufferFragmentMain", "visibilityBufferMaskedFragmentMain",
        "visibilityClusterCountMain", "visibilityClusterBinMain", "visibilityClusterRasterMain",
        "visibilityTessellationFragment", "visibilityTessellationCullingFragment"});
    for (const char* wave : {"0", "1"}) {
        add("Features/GPUDriven/HZBSPD", {"hzbSpdMain"}, {}, {{"HZB_SPD_WAVE_OPS", wave}});
        add(visibilityModule, {"visibilityBufferAmplificationMain", "visibilityTessellationTask"},
            meshCapabilities, {{"GPU_DRIVEN_AMPLIFICATION_WAVE_OPS", wave}});
        add(streamModule, {"streamTessellationTask"}, meshCapabilities,
            {{"GPU_DRIVEN_AMPLIFICATION_WAVE_OPS", wave}});
    }
    add(visibilityModule, {"visibilityBufferMeshMain", "visibilityTessellationMesh"}, meshCapabilities);
    add(visibilityModule, {"visibilityBufferMeshMain", "visibilityTessellationMesh"}, meshCapabilities,
        {{"VISIBILITY_BUFFER_ALPHA_MASKED", "1"}});
    add(visibilityModule, {"visibilityTessellationFragment", "visibilityTessellationCullingFragment"}, {},
        {{"VISIBILITY_BUFFER_ALPHA_MASKED", "1"}});

    const bool hasRtxcr = !rtxcrInclude.empty();
    const std::vector<std::string> rtxcrPaths = hasRtxcr
        ? std::vector<std::string>{rtxcrInclude} : std::vector<std::string>{};
    if (hasRtxcr) {
        add("Features/Samples/RTXCRMaterialSample", {"rtxcrMaterialSampleMain"}, {}, {}, rtxcrPaths);
    }
    // Conventional textures, both position-fetch capability variants. Optional
    // NTC/NRD SDK and generated material programs remain runtime-compiled.
    for (bool positionFetch : {false, true}) {
        for (auto program : {SceneRayQueryProgram::RTXDI, SceneRayQueryProgram::MaterialVisualization}) {
            requests.push_back(makeSceneRayQueryRequest(program, {.positionFetch = positionFetch}));
        }
        SceneShaderOptions options{.hasRTXCR = hasRtxcr, .positionFetch = positionFetch, .rtxcrInclude = rtxcrInclude};
        for (auto program : {SceneShaderProgram::SharcClear, SceneShaderProgram::SharcResolve, SceneShaderProgram::Tonemap,
                SceneShaderProgram::RealtimeLighting, SceneShaderProgram::PathTraceGuides,
                SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides}) {
            requests.push_back(makeSceneShaderRequest(program, options));
        }
        requests.push_back(makeSceneShaderRequest(SceneShaderProgram::PathTrace, options));
        for (const char* cacheDefine : {"SHARC_UPDATE", "SHARC_QUERY", "NRC_UPDATE", "NRC_QUERY"}) {
            const SlangMacroDefine define{cacheDefine, "1"};
            requests.push_back(makeSceneShaderRequest(SceneShaderProgram::PathTrace, options, {&define, 1}));
        }
    }
    for (bool streamed : {false, true}) {
        for (bool guides : {false, true}) {
            const SceneShaderOptions options{.streamMaterials = streamed, .globalView = true, .upscalerGuides = guides};
            requests.push_back(makeSceneShaderRequest(SceneShaderProgram::Deferred, options));
            for (int materialClass = 0; materialClass < 5; ++materialClass) {
                const std::string value = std::to_string(materialClass);
                const SlangMacroDefine define{"MATERIAL_CLASS", value.c_str()};
                requests.push_back(makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, options, {&define, 1}));
            }
        }
    }
    for (bool streamed : {false, true}) {
        for (bool streamTlas : {false, true}) {
            if (!streamed && streamTlas) { continue; }
            requests.push_back(makeShadowShaderRequest({.streamed = streamed, .streamTlas = streamTlas}));
        }
    }
    // Deduplicate complete requests, preserving macro/capability/search-path
    // order and the first occurrence. No compiler cache version change is needed.
    std::vector<ShaderRequest> unique;
    for (auto& request : requests) {
        if (std::find(unique.begin(), unique.end(), request) == unique.end()) {
            unique.push_back(std::move(request));
        }
    }
    return unique;
}

} // namespace metallic::render
