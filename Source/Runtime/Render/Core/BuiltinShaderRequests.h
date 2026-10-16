#pragma once

#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/HZBSPD.h"

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
    add("Features/Environment/Transmittance", {"transmittanceMain"});
    add("Features/Environment/MultiScattering", {"multiScatteringMain"});
    add("Features/Environment/SkyView", {"skyViewMain"});
    add("Features/Environment/EnvironmentCapture", {"environmentCaptureMain"});
    add("Features/Environment/AerialPerspective", {"aerialPerspectiveMain"});
    add("Features/Environment/CloudShadow", {"cloudShadowMain"});
    add("Features/GPUDriven/GPUDrivenCulling", {"gpuDrivenPreviewResetMain", "gpuDrivenPreviewInstanceCullMain", "gpuDrivenPreviewHzbMain"});
    add("Features/GPUDriven/GPUDrivenStreamWorkload", {"streamWorkloadResetMain", "streamWorkloadMain"});
    add("Features/GPUDriven/GPUDrivenStreamWorkRaster", {"streamClusterRasterWorkBinsMain", "streamClusterRasterWorkControlMain"});
    add("Features/GPUDriven/GPUDrivenStreamGroupRaster", {"streamClusterRasterGroup32Main"});
    add("Features/GPUDriven/GPUDrivenStreamAsset", {"streamClusterBinMain", "streamClusterBinP0Main"});
    add("Features/Lighting/BuildReGIR", {"buildReGIRMain"});
    add("Features/Lighting/ClusterLightGrid", {"clusterLightGridMain"});
    add("Features/Lighting/PrepareLightsPdf", {"prepareLightsPdfMain"});
    add("Features/PostProcess/AutoExposure", {"autoExposureHistogramMain", "autoExposureReduceMain", "autoExposureApplyMain"});
    for (bool hdr : {false, true}) {
        for (bool scRgb : {false, true}) {
            for (bool sRGB : {false, true}) {
                for (float white : {80.0f, 203.0f}) {
                    for (const char* entry : {"editorDisplayVertex", "editorDisplayFragment"}) {
                        requests.push_back(makeEditorDisplayShaderRequest(entry, hdr, scRgb, sRGB, std::to_string(white)));
                    }
                }
            }
        }
    }
    for (float white : {80.0f, 203.0f}) {
        for (const char* entry : {"editorOutputVertex", "editorOutputPQFragment"}) {
            requests.push_back(makeEditorDisplayShaderRequest(entry, true, false, false, std::to_string(white)));
        }
    }
    add("Features/PostProcess/FinalBlit", {"finalBlitUvMain", "finalBlitMain"}, {}, {{"FINAL_USE_LUT", "0"}});
    add("Features/PostProcess/FinalBlit", {"finalBlitMain"}, {}, {{"FINAL_USE_LUT", "1"}});
    add("Features/PostProcess/ColorGradingLUT", {"composeColorGradingLUT"});
    add("Features/Strands/NativeStrands", {"strandVisibilityMain", "strandLightingMain"});
    add("Features/PathTracing/RayMaterialQueue", {"rayMaterialResetMain", "rayMaterialCountMain", "rayMaterialPrefixMain", "rayMaterialScatterMain"});
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
    for (bool wave : {false, true}) {
        add(streamModule, {"streamTessellationTask"}, meshCapabilities,
            {{"GPU_DRIVEN_AMPLIFICATION_WAVE_OPS", wave ? "1" : "0"}});
    }
    const bool hasRtxcr = !rtxcrInclude.empty();
    const std::vector<std::string> rtxcrPaths = hasRtxcr
        ? std::vector<std::string>{rtxcrInclude} : std::vector<std::string>{};
    if (hasRtxcr) {
        add("Features/Samples/RTXCRMaterialSample", {"rtxcrMaterialSampleMain"}, {}, {}, rtxcrPaths);
    }
    // All ray-query programs consume streamed CLAS geometry. Optional SDK
    // permutations are compiled on demand; generated materials append requests.
    for (auto program : {SceneRayQueryProgram::RTXDI, SceneRayQueryProgram::MaterialVisualization}) {
        requests.push_back(makeSceneRayQueryRequest(program));
    }
    const SceneShaderOptions options{.hasRTXCR = hasRtxcr, .rtxcrInclude = rtxcrInclude};
    for (auto program : {SceneShaderProgram::SharcClear, SceneShaderProgram::SharcResolve,
            SceneShaderProgram::OpenPBRSharcResolve, SceneShaderProgram::RealtimeLighting,
            SceneShaderProgram::PathTrace, SceneShaderProgram::PathTraceGuides,
            SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides}) {
        requests.push_back(makeSceneShaderRequest(program, options));
    }
    for (const char* cacheDefine : {"SHARC_UPDATE", "SHARC_QUERY"}) {
        const SlangMacroDefine define{cacheDefine, "1"};
        for (auto program : {SceneShaderProgram::PathTrace, SceneShaderProgram::OpenPBRPathTrace,
                SceneShaderProgram::OpenPBRPathTraceGuides}) {
            requests.push_back(makeSceneShaderRequest(program, options, {&define, 1}));
        }
    }
    // Each program serves both camera sources and both upscaler-guide output policies.
    for (bool half : {false, true}) {
        const SceneShaderOptions deferredOptions{.deferredFloat16 = half};
        requests.push_back(makeSceneShaderRequest(SceneShaderProgram::Deferred, deferredOptions));
        for (int materialClass = 0; materialClass < 5; ++materialClass) {
            const std::string value = std::to_string(materialClass);
            const SlangMacroDefine define{"MATERIAL_CLASS", value.c_str()};
            requests.push_back(makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, deferredOptions, {&define, 1}));
        }
    }
    requests.push_back(makeShadowShaderRequest());
    // The runtime appends SINGLE_SAMPLE after its Base/SHARC_QUERY defines.
    // Append coverage without changing the existing ordered request prefix.
    const SlangMacroDefine singleSampleGuides[]{{"METALLIC_PT_SINGLE_SAMPLE", "1"}};
    requests.push_back(makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, singleSampleGuides));
    const SlangMacroDefine singleSampleGuideQuery[]{{"SHARC_QUERY", "1"}, {"METALLIC_PT_SINGLE_SAMPLE", "1"}};
    requests.push_back(makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, singleSampleGuideQuery));
    // Match VisibilityBufferPass LDS and optional wave SPD startup requests.
    add(kHZBSPDModule, {kHZBSPDEntryPoint}, {}, {{kHZBSPDWaveOpsDefine, "0"}});
    add(kHZBSPDModule, {kHZBSPDEntryPoint}, {}, {{kHZBSPDWaveOpsDefine, "1"}});
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
