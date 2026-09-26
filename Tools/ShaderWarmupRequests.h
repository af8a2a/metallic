#pragma once

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace metallic::tools {

struct ShaderWarmupRequest {
    std::string module;
    std::string entry;
    std::vector<std::string> capabilities;
    std::vector<std::pair<std::string, std::string>> defines;
    std::vector<std::string> searchPaths;
};

// This is an explicit list of runtime requests, not a scan of every .slang file:
// many files are modules/includes, and identical entry points with different
// ordered defines/capabilities have different runtime cache keys.
// Keep variants in sync with their owning passes when adding compiler options.
inline std::vector<ShaderWarmupRequest> shaderWarmupRequests()
{
    std::vector<ShaderWarmupRequest> requests;
    auto add = [&](const char* module, std::initializer_list<const char*> entries,
                   std::vector<std::string> capabilities = {},
                   std::vector<std::pair<std::string, std::string>> defines = {},
                   std::vector<std::string> searchPaths = {}) {
        for (const char* entry : entries) {
            requests.push_back({module, entry, capabilities, defines, searchPaths});
        }
    };

    add("Features/Debug/GpuProbe", {"probe"});
    add("Features/Debug/LightGridDebug", {"lightGridDebugMain"});
    add("Features/Debug/SliderDebug", {"sliderDebugMain", "sliderDebugOverlayMain"});
    add("Features/Environment/EnvironmentLightingPrecompute", {"environmentLightingPrecomputeMain"});
    add("Features/GPUDriven/GPUDrivenCulling", {"gpuDrivenPreviewResetMain", "gpuDrivenPreviewInstanceCullMain", "gpuDrivenPreviewHzbMain"});
    add("Features/GPUDriven/GPUDrivenStreamWorkload", {"streamWorkloadResetMain", "streamWorkloadMain"});
    add("Features/GPUDriven/GPUDrivenStreamWorkRaster", {"streamClusterRasterWorkBinsMain", "streamClusterRasterWorkControlMain"});
    add("Features/GPUDriven/ResidentMeshletLod", {"residentLodResetMain", "residentLodSelectMain", "residentLodArgumentsMain", "residentLodScatterMain"});
    add("Features/GPUDriven/GPUDrivenStreamAsset", {"streamClusterBinMain", "streamClusterBinP0Main"});
    add("Features/Lighting/BuildReGIR", {"buildReGIRMain"});
    add("Features/Lighting/ClusterLightGrid", {"clusterLightGridMain"});
    add("Features/Lighting/PrepareLightsPdf", {"prepareLightsPdfMain"});
    add("Features/PostProcess/AutoExposure", {"autoExposureHistogramMain", "autoExposureReduceMain", "autoExposureApplyMain"});
    add("Features/PostProcess/EditorDisplay", {"editorDisplayVertex", "editorDisplayFragment"});
    add("Features/PostProcess/FinalBlit", {"finalBlitUvMain", "finalBlitMain"});
    add("Features/PostProcess/StreamlineDlssSupport", {"streamlineDlssDepthVertexMain", "streamlineDlssDepthFragmentMain", "streamlineDlssAlphaMain"});
    add("Features/PostProcess/UpscalerGuideResolve", {"upscalerGuideResolveMain"});
    add("Features/ReSTIR/RtxdiComposite", {"rtxdiCompositeMain"});
    add("Features/ReSTIR/RtxdiConfidence", {"rtxdiConfidenceMain"});
    add("Features/Samples/BunnyWireframe", {"bunnyWireframeVertexMain", "bunnyWireframeFragmentMain"});
    add("Features/Samples/ImageSample", {"imageSampleVertexMain", "imageSampleFragmentMain"});
    add("Features/Samples/MaterialShaderObject", {"materialShaderObjectVertexMain", "materialShaderObjectFragmentMain", "materialShaderObjectAlternateFragmentMain"});
    add("Features/SmokeTests/RenderGraphBuffer", {"renderGraphBufferWriteMain", "renderGraphBufferCopyMain"});
    add("Features/VisibilityBuffer/VisibilityBufferComposite", {"visibilityBufferCompositeVertexMain", "visibilityBufferCompositeFragmentMain"});
    add("Features/VisibilityBuffer/VisibilityBufferMaterial", {"visibilityBufferMaterialMain"});
    add("Features/VisibilityBuffer/VisibilityHybridRaster", {"hybridResetMain", "hybridArgumentsMain", "hybridRasterMain", "hybridResolveVertexMain", "hybridResolveFragmentMain", "hybridClusterResetMain", "hybridClusterHistogramMain", "hybridClusterArgumentsMain", "hybridClusterScatterMain"});
    add("Features/Samples/Triangle", {"triangleVertexMain", "triangleFragmentMain"});
    add("Features/VisibilityBuffer/VisibilityMaterialBinning",
        {"materialBinningResetMain", "materialBinningClassifyMain", "materialBinningArgumentsMain"},
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
        add("Features/GPUDriven/HzbSpd", {"hzbSpdMain"}, {}, {{"HZB_SPD_WAVE_OPS", wave}});
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

    const std::string rtxcrInclude = METALLIC_RTXCR_SHADER_INCLUDE_DIR;
    const bool hasRtxcr = !rtxcrInclude.empty();
    const std::vector<std::string> rtxcrPaths = hasRtxcr
        ? std::vector<std::string>{rtxcrInclude} : std::vector<std::string>{};
    if (hasRtxcr) {
        add("Features/Samples/RtxcrMaterialSample", {"rtxcrMaterialSampleMain"}, {}, {}, rtxcrPaths);
    }
    // Conventional textures, both position-fetch capability variants. Optional
    // NTC/NRD SDK permutations remain runtime-compiled.
    for (const char* positionFetch : {"0", "1"}) {
        std::vector<std::string> rayCapabilities{"spvRayQueryKHR"};
        if (std::string(positionFetch) == "1") {
            rayCapabilities.push_back("spvRayQueryPositionFetchKHR");
        }
        const std::vector<std::pair<std::string, std::string>> rayDefines{
            {"SCENE_RAYQUERY_ENABLE_POSITION_FETCH", positionFetch},
            {"METALLIC_HAS_NTC", "0"}, {"METALLIC_NTC_COOPERATIVE_VECTOR", "0"}};
        add("Features/ReSTIR/SceneRtxdi", {"sceneRtxdiMain"}, rayCapabilities, rayDefines);
        add("Features/Debug/SceneMaterialVisualize", {"sceneMaterialVisualizeMain"}, rayCapabilities, rayDefines);

        std::vector<std::string> pathCapabilities{"spvRayQueryKHR", "spvGroupNonUniformBallot"};
        if (std::string(positionFetch) == "1") {
            pathCapabilities.push_back("spvRayQueryPositionFetchKHR");
        }
        add("Features/PathTracing/SceneSharcMaintenance", {"sharcClearMain", "sharcResolveMain"}, pathCapabilities);
        add("Features/PostProcess/ScenePathTraceTonemap", {"scenePathTraceTonemapMain"}, pathCapabilities);
        std::vector<std::pair<std::string, std::string>> pathDefines{
            {"METALLIC_STREAM_MATERIALS", "0"}, {"METALLIC_STREAM_RAY_QUERIES", "0"},
            {"METALLIC_GLOBAL_VIEW", "0"}, {"METALLIC_HAS_RTXCR", hasRtxcr ? "1" : "0"},
            {"METALLIC_HAS_NTC", "0"}, {"METALLIC_NTC_COOPERATIVE_VECTOR", "0"},
            {"SCENE_RAYQUERY_ENABLE_POSITION_FETCH", positionFetch},
            {"METALLIC_DEFERRED_LIGHT_GRID", "0"}, {"METALLIC_REALTIME_DEFERRED", "0"},
            {"METALLIC_DEFERRED_UPSCALER_GUIDES", "0"}};
        add("Features/Lighting/SceneRealtimeLighting", {"sceneRealtimeLightingMain"}, pathCapabilities, pathDefines, rtxcrPaths);
        for (const char* cacheDefine : {"", "SHARC_UPDATE", "SHARC_QUERY", "NRC_UPDATE", "NRC_QUERY"}) {
            auto defines = pathDefines;
            if (cacheDefine[0] != '\0') {
                defines.emplace_back(cacheDefine, "1");
            }
            add("Features/PathTracing/ScenePathTrace", {"scenePathTraceMain"}, pathCapabilities, defines, rtxcrPaths);
            if (cacheDefine[0] != '\0') { continue; }
            add("Features/PathTracing/ScenePathTraceGuides", {"scenePathTraceGuidesMain"}, pathCapabilities, defines, rtxcrPaths);
            add("Features/PathTracing/OpenPBRRayQueryPathTrace", {"openPbrRayQueryPathTraceMain"}, pathCapabilities, defines, rtxcrPaths);
            add("Features/PathTracing/OpenPBRRayQueryPathTraceGuides", {"openPbrRayQueryPathTraceGuidesMain"}, pathCapabilities, defines, rtxcrPaths);
        }
    }
    // Default reference and realtime deferred pipelines, including StreamAsset
    // opaque/transmission variants and all five material classes.
    for (int mode = 0; mode < 4; ++mode) {
        const bool streamed = mode >= 2;
        const bool positionFetch = mode == 1;
        std::vector<std::string> capabilities{"spvRayQueryKHR", "spvGroupNonUniformBallot"};
        if (positionFetch) {
            capabilities.push_back("spvRayQueryPositionFetchKHR");
        }
        for (const char* realtime : {"0", "1"}) {
            if (streamed && std::string(realtime) == "0") { continue; }
            std::vector<std::pair<std::string, std::string>> defines{
                {"METALLIC_STREAM_MATERIALS", streamed ? "1" : "0"},
                {"METALLIC_STREAM_RAY_QUERIES", mode == 3 ? "1" : "0"},
                {"METALLIC_GLOBAL_VIEW", "1"}, {"METALLIC_HAS_RTXCR", hasRtxcr ? "1" : "0"},
                {"METALLIC_HAS_NTC", "0"}, {"METALLIC_NTC_COOPERATIVE_VECTOR", "0"},
                {"SCENE_RAYQUERY_ENABLE_POSITION_FETCH", positionFetch ? "1" : "0"},
                {"METALLIC_DEFERRED_LIGHT_GRID", "1"}, {"METALLIC_REALTIME_DEFERRED", realtime},
                {"METALLIC_DEFERRED_UPSCALER_GUIDES", realtime}};
            add("Features/VisibilityBuffer/VisibilityBufferDeferred", {"visibilityBufferDeferredMain"},
                capabilities, defines, rtxcrPaths);
            for (int materialClass = 0; materialClass < 5; ++materialClass) {
                auto classified = defines;
                classified.emplace_back("MATERIAL_CLASS", std::to_string(materialClass));
                add("Features/VisibilityBuffer/VisibilityBufferDeferred", {"visibilityBufferDeferredBinnedMain"},
                    capabilities, classified, rtxcrPaths);
            }
        }
    }
    for (const char* streamed : {"0", "1"}) {
        for (const char* streamTlas : {"0", "1"}) {
            if (std::string(streamed) == "0" && std::string(streamTlas) == "1") { continue; }
            add("Features/Lighting/ScreenSpaceShadows", {"rayTracedShadowsMain"}, {"spvRayQueryKHR"},
                {{"METALLIC_STREAM_SHADOWS", streamed}, {"METALLIC_STREAM_TLAS", streamTlas},
                 {"METALLIC_HAS_NTC", "0"}, {"METALLIC_NTC_COOPERATIVE_VECTOR", "0"}});
        }
    }
    return requests;
}

} // namespace metallic::tools
