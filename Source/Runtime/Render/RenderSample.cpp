#include "Runtime/Render/RenderSample.h"

#include <algorithm>
#include <string>
#include <utility>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

#ifndef METALLIC_RTXCR_ASSETS_ROOT
#define METALLIC_RTXCR_ASSETS_ROOT ""
#endif

namespace metallic::render {
namespace {

std::filesystem::path projectPath(std::string_view path)
{
    std::filesystem::path resolved(path);
    if (resolved.is_relative()) {
        resolved = std::filesystem::path(PROJECT_SOURCE_DIR) / resolved;
    }
    return resolved;
}

bool graphHasTextureOutput(const RenderGraph& graph, std::string_view outputName)
{
    std::string passName;
    std::string fieldName;
    if (!splitRenderGraphFieldName(outputName, passName, fieldName)) {
        return false;
    }
    const RenderGraphNode* node = graph.findNode(passName);
    std::unique_ptr<RenderGraphPass> pass = node != nullptr ? createRenderGraphPass(node->type) : nullptr;
    if (pass == nullptr) {
        return false;
    }
    pass->setProperties(node->properties);
    const RenderPassReflection reflection = pass->reflect(RenderGraphCompileContext{});
    const RenderGraphField* field = reflection.findField(fieldName, RenderGraphFieldVisibility::Output);
    return field != nullptr && field->resourceType == RenderGraphResourceType::Texture2D;
}

bool applySampleScenePath(RenderGraph& graph, const RenderSampleDesc& desc, std::string& outMessage)
{
    for (const std::string& target : desc.scenePathTargets) {
        RenderGraphNode* node = graph.findNode(target);
        if (node == nullptr) {
            outMessage = "Sample scenePathTargets node not found: " + target;
            return false;
        }
        RenderGraphProperties properties = node->properties;
        if (!properties.is_object()) {
            properties = RenderGraphProperties::object();
        }
        properties["path"] = desc.scenePath;
        if (!graph.setNodeProperties(node->id, std::move(properties))) {
            outMessage = "Sample failed to update node properties: " + target;
            return false;
        }
    }
    return true;
}

class LightGridDebugSample final : public RenderSample {
public:
    std::string_view id() const override { return "light-grid-debug"; }
    std::string_view name() const override { return "LightGrid / Coverage Heatmap"; }
    std::string_view category() const override { return "Lighting"; }
    std::string_view description() const override
    {
        return "Asset-free clustered-light test bench with deterministic point/spot lights, depth slices and overflow diagnostics.";
    }
    std::string scenePath() const override { return {}; }
    bool loadSceneInEditor() const override { return false; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/light_grid_debug.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{.enabled = false};
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class RealtimeLightingSample final : public RenderSample {
public:
    std::string_view id() const override { return "realtime-lighting"; }
    std::string_view name() const override { return "Real-time / Physical Lighting"; }
    std::string_view category() const override { return "Lighting"; }
    std::string_view description() const override
    {
        return "OpenPBR punctual lighting, ray-query shadows and GPU SH environment GI. Add lights in Physical Lighting.";
    }
    std::string scenePath() const override { return "Asset/meet_mat.glb"; }
    std::string graphPath() const override { return "Pipelines/Samples/realtime_lighting.metallic_graph.json"; }
    std::vector<std::string> scenePathTargets() const override { return {"Lighting"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{.enabled = true, .path = "Asset/ABeautifulGame/environment.hdr"};
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class OpenPbrLookDevSample final : public RenderSample {
public:
    std::string_view id() const override { return "openpbr-lookdev"; }
    std::string_view name() const override { return "LookDev / OpenPBR Default"; }
    std::string_view category() const override { return "LookDev"; }
    std::string_view description() const override
    {
        return "MaterialX reference shaderball, split HDRI and sun, fixed exposure and sRGB display. "
            "Progressive OpenPBR path tracing; see Documentation/OpenPbrLookDev.md.";
    }
    std::string scenePath() const override
    {
        return "Asset/LookDev/OpenPbrDefault/OpenPbrDefault.gltf";
    }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/openpbr_lookdev.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"PathTrace"}; }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class LookDevShadingCompareSample final : public RenderSample {
public:
    std::string_view id() const override { return "lookdev-shading-compare"; }
    std::string_view name() const override { return "LookDev / Shading Comparison"; }
    std::string_view category() const override { return "LookDev"; }
    std::string_view description() const override
    {
        return "OpenPBR / Standard BSDF comparison with a draggable divider, linked cameras and shared HDR exposure. "
            "See Documentation/SliderDebugPass.md.";
    }
    std::string scenePath() const override { return "Asset/LookDev/OpenPbrDefault/OpenPbrDefault.gltf"; }
    std::string graphPath() const override { return "Pipelines/Samples/lookdev_shading_compare.metallic_graph.json"; }
    std::vector<std::string> scenePathTargets() const override { return {"OpenPBR", "Standard"}; }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class LookDevVisibilityBufferSample final : public RenderSample {
public:
    std::string_view id() const override { return "lookdev-vbuffer"; }
    std::string_view name() const override { return "LookDev / VBuffer vs Path Tracing"; }
    std::string_view category() const override { return "LookDev"; }
    std::string_view description() const override
    {
        return "OpenPBR on both paths: GPUDriven visibility and deferred lighting versus progressive path tracing, "
            "with linked cameras and shared exposure. See Documentation/VisibilityBufferDeferred.md.";
    }
    std::string scenePath() const override { return "Asset/LookDev/OpenPbrDefault/OpenPbrDefault.gltf"; }
    std::string graphPath() const override { return "Pipelines/Samples/lookdev_vbuffer.metallic_graph.json"; }
    std::vector<std::string> scenePathTargets() const override { return {"Reference", "VBuffer", "Deferred"}; }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class PathTracingMeetMatSample final : public RenderSample {
public:
    std::string_view id() const override { return "pathtracing-meet-mat"; }
    std::string_view name() const override { return "Path Tracing / meet_mat"; }
    std::string_view category() const override { return "PathTracing"; }
    std::string_view description() const override
    {
        return "Path tracing validation sample using the meet_mat glTF scene.";
    }
    std::string scenePath() const override { return "Asset/meet_mat.glb"; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/pathtracing_meet_mat.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"PathTrace"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class PathTracingSharcMeetMatSample final : public RenderSample {
public:
    std::string_view id() const override { return "pathtracing-sharc-meet-mat"; }
    std::string_view name() const override { return "Path Tracing / meet_mat / SHaRC"; }
    std::string_view category() const override { return "PathTracing"; }
    std::string_view description() const override
    {
        return "meet_mat path tracing accelerated with the RTXGI SHaRC spatial hash radiance cache.";
    }
    std::string scenePath() const override { return "Asset/meet_mat.glb"; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/pathtracing_meet_mat_sharc.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"PathTrace"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class PathTracingNrcMeetMatSample final : public RenderSample {
public:
    std::string_view id() const override { return "pathtracing-nrc-meet-mat"; }
    std::string_view name() const override { return "Path Tracing / meet_mat / NRC"; }
    std::string_view category() const override { return "PathTracing"; }
    std::string_view description() const override
    {
        return "meet_mat path tracing accelerated with the NVIDIA Neural Radiance Cache (requires an RTX GPU).";
    }
    std::string scenePath() const override { return "Asset/meet_mat.glb"; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/pathtracing_meet_mat_nrc.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"PathTrace"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class PathTracingSample final : public RenderSample {
public:
    std::string_view id() const override { return "pathtracing-sample"; }
    std::string_view name() const override { return "PathTracingSample"; }
    std::string_view category() const override { return "PathTracing"; }
    std::string_view description() const override
    {
        return "OpenPBR RayQuery path tracing sample using the ABeautifulGame glTF scene and HDRI environment.";
    }
    std::string scenePath() const override { return "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf"; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/pathtracing_abeautiful_game_openpbr.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"PathTrace"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class PathTracingDlssRrSample final : public RenderSample {
public:
    std::string_view id() const override { return "pathtracing-sample-dlss-rr"; }
    std::string_view name() const override { return "PathTracingSample / DLSS-RR"; }
    std::string_view category() const override { return "PathTracing"; }
    std::string_view description() const override
    {
        return "OpenPBR RayQuery path tracing sample using NVIDIA DLSS Ray Reconstruction as the denoiser.";
    }
    std::string scenePath() const override { return "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf"; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/pathtracing_abeautiful_game_openpbr_dlss_rr.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"PathTrace"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
    bool requiresStreamline() const override { return true; }
};

class PathTracingDlssSrSample final : public RenderSample {
public:
    std::string_view id() const override { return "pathtracing-sample-dlss-sr"; }
    std::string_view name() const override { return "PathTracingSample / DLSS-SR"; }
    std::string_view category() const override { return "PathTracing"; }
    std::string_view description() const override
    {
        return "OpenPBR RayQuery path tracing sample upscaled with NVIDIA DLSS Super Resolution.";
    }
    std::string scenePath() const override { return "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf"; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/pathtracing_abeautiful_game_openpbr_dlss_sr.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"PathTrace"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
    bool requiresStreamline() const override { return true; }
};

class RtxdiSample final : public RenderSample {
public:
    std::string_view id() const override { return "rtxdi-sample"; }
    std::string_view name() const override { return "RTXDI / ReSTIR DI"; }
    std::string_view category() const override { return "RTXDI"; }
    std::string_view description() const override
    {
        return "Fused spatiotemporal ReSTIR DI with hierarchical local-light and environment importance sampling, denoised with NRD RELAX.";
    }
    std::string scenePath() const override { return "Asset/meet_mat.glb"; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/rtxdi_meet_mat.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"Rtxdi"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class RtxcrMaterialSample final : public RenderSample {
public:
    std::string_view id() const override { return "rtxcr-material-sample"; }
    std::string_view name() const override { return "RTXCR Claire Ponytail"; }
    std::string_view category() const override { return "RTXCR"; }
    std::string_view description() const override
    {
        return "NVIDIA Claire reference groom rendered with RTXCR DOTS geometry and the Chiang hair BSDF.";
    }
    std::string scenePath() const override
    {
        return std::string(METALLIC_RTXCR_ASSETS_ROOT) + "/Claire/ponyTail_15vtx.gltf";
    }
    bool loadSceneInEditor() const override { return true; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/rtxcr_material_showcase.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"PathTrace"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = std::string(METALLIC_RTXCR_ASSETS_ROOT) +
                "/EnvironmentMaps/studio_small_09_1k.hdr",
            .intensity = 1.5f,
            .rotationDegrees = 25.0f,
            .visible = true,
        };
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class MaterialVisualizationABeautifulGameSample final : public RenderSample {
public:
    std::string_view id() const override { return "material-visualization-abeautiful-game"; }
    std::string_view name() const override { return "Material Visualization / ABeautifulGame"; }
    std::string_view category() const override { return "Material"; }
    std::string_view description() const override
    {
        return "RayQuery material parameter visualization for the ABeautifulGame glTF scene.";
    }
    std::string scenePath() const override { return "Asset/ABeautifulGame/glTF/ABeautifulGame.gltf"; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/material_visualization_abeautiful_game.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"MaterialViz"}; }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class GPUDrivenSample final : public RenderSample {
public:
    std::string_view id() const override { return kDefaultGPUDrivenSampleId; }
    std::string_view name() const override { return "GPUDrivenSample"; }
    std::string_view category() const override { return "GPUDriven"; }
    std::string_view description() const override
    {
        return "Default GPUDriven sample producing raw visibility/depth with Wave32 AS/MS culling and optional ID/depth diagnostics.";
    }
    std::string scenePath() const override { return "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf"; }
    bool loadSceneInEditor() const override { return false; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/gpu_driven_sponza.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"GPUDriven"}; }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class GPUDrivenUsdSample final : public RenderSample {
public:
    std::string_view id() const override { return "gpu-driven-usd"; }
    std::string_view name() const override { return "GPUDrivenSample / USD"; }
    std::string_view category() const override { return "GPUDriven"; }
    std::string_view description() const override
    {
        return "Super Sponza loaded from its Y-up USDA scene, including Preview Surface materials and GeomSubset bindings.";
    }
    std::string scenePath() const override
    {
        return "Asset/SuperSponza/NewSponza_Main_USD_Yup_003.usda";
    }
    bool loadSceneInEditor() const override { return false; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/gpu_driven_sponza.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"GPUDriven"}; }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class GPUDrivenRtasVisualizationSample final : public RenderSample {
public:
    std::string_view id() const override { return "gpu-driven-rtas-visualization"; }
    std::string_view name() const override { return "GPUDrivenSample / RTAS Visualization"; }
    std::string_view category() const override { return "GPUDriven"; }
    std::string_view description() const override
    {
        return "GPUDrivenSample variant dedicated to RayQuery acceleration-structure visualization.";
    }
    std::string scenePath() const override { return "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf"; }
    bool loadSceneInEditor() const override { return false; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/gpu_driven_sponza_rtas_visualization.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"GPUDriven"}; }
    std::optional<RenderSampleEnvironmentDesc> environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class GPUDrivenStreamAssetSample final : public RenderSample {
public:
    std::string_view id() const override { return "gpu-driven-streamasset"; }
    std::string_view name() const override { return "GPUDrivenSample / StreamAsset"; }
    std::string_view category() const override { return "GPUDriven"; }
    std::string_view description() const override
    {
        return "GPUDrivenSample variant using GPUDrivenStreamAssetPass and the meshlet streamasset prototype.";
    }
    std::string scenePath() const override { return "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf"; }
    bool loadSceneInEditor() const override { return false; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/gpu_driven_sponza_streamasset.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"GPUDriven"}; }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class GPUDrivenTerrainP0Sample final : public RenderSample {
public:
    std::string_view id() const override { return "gpu-driven-terrain-p0"; }
    std::string_view name() const override { return "GPUDrivenSample / Terrain P0"; }
    std::string_view category() const override { return "GPUDriven"; }
    std::string_view description() const override
    {
        return "Houdini height-field vertical slice using the generic MeshletStreamAsset mesh-shader path.";
    }
    std::string scenePath() const override
    {
        return "Asset/MeshletCache/TerrainP0/simple_terrain_height.gltf";
    }
    bool loadSceneInEditor() const override { return false; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/gpu_driven_terrain_p0_streamasset.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"GPUDriven"}; }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

class GPUDrivenTerrainP1UnifiedSample final : public RenderSample {
public:
    std::string_view id() const override { return "gpu-driven-terrain-p1-unified"; }
    std::string_view name() const override { return "GPUDrivenSample / Terrain P1 Unified"; }
    std::string_view category() const override { return "GPUDriven"; }
    std::string_view description() const override
    {
        return "Houdini height-field StreamAsset rendered through the unified GPUScene visibility-buffer pipeline.";
    }
    std::string scenePath() const override
    {
        return "Asset/MeshletCache/TerrainP0/simple_terrain_height.gltf";
    }
    bool loadSceneInEditor() const override { return true; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/gpu_driven_terrain_p1_unified.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"GPUDriven"}; }
    std::string previewOutput() const override { return "FinalBlit.color"; }
};

const RenderSample& pathTracingMeetMatSample()
{
    static const PathTracingMeetMatSample sample;
    return sample;
}

const RenderSample& pathTracingSharcMeetMatSample()
{
    static const PathTracingSharcMeetMatSample sample;
    return sample;
}

const RenderSample& pathTracingNrcMeetMatSample()
{
    static const PathTracingNrcMeetMatSample sample;
    return sample;
}

const RenderSample& pathTracingSample()
{
    static const PathTracingSample sample;
    return sample;
}

const RenderSample& pathTracingDlssRrSample()
{
    static const PathTracingDlssRrSample sample;
    return sample;
}

const RenderSample& pathTracingDlssSrSample()
{
    static const PathTracingDlssSrSample sample;
    return sample;
}

const RenderSample& materialVisualizationABeautifulGameSample()
{
    static const MaterialVisualizationABeautifulGameSample sample;
    return sample;
}

const RenderSample& rtxdiSample()
{
    static const RtxdiSample sample;
    return sample;
}

const RenderSample& rtxcrMaterialSample()
{
    static const RtxcrMaterialSample sample;
    return sample;
}

const RenderSample& gpuDrivenSample()
{
    static const GPUDrivenSample sample;
    return sample;
}

const RenderSample& gpuDrivenUsdSample()
{
    static const GPUDrivenUsdSample sample;
    return sample;
}

const RenderSample& gpuDrivenRtasVisualizationSample()
{
    static const GPUDrivenRtasVisualizationSample sample;
    return sample;
}

const RenderSample& gpuDrivenStreamAssetSample()
{
    static const GPUDrivenStreamAssetSample sample;
    return sample;
}

const RenderSample& gpuDrivenTerrainP0Sample()
{
    static const GPUDrivenTerrainP0Sample sample;
    return sample;
}

const RenderSample& gpuDrivenTerrainP1UnifiedSample()
{
    static const GPUDrivenTerrainP1UnifiedSample sample;
    return sample;
}

std::vector<const RenderSample*> builtInRenderSamples()
{
    static const RealtimeLightingSample realtimeLighting;
    static const LightGridDebugSample lightGridDebug;
    static const OpenPbrLookDevSample openPbrLookDev;
    static const LookDevShadingCompareSample lookDevShadingCompare;
    static const LookDevVisibilityBufferSample lookDevVisibilityBuffer;
    return {
        &realtimeLighting,
        &lightGridDebug,
        &openPbrLookDev,
        &lookDevShadingCompare,
        &lookDevVisibilityBuffer,
        &pathTracingMeetMatSample(),
        &pathTracingSharcMeetMatSample(),
        &pathTracingNrcMeetMatSample(),
        &pathTracingSample(),
        &pathTracingDlssSrSample(),
        &pathTracingDlssRrSample(),
        &rtxdiSample(),
        &rtxcrMaterialSample(),
        &materialVisualizationABeautifulGameSample(),
        &gpuDrivenSample(),
        &gpuDrivenUsdSample(),
        &gpuDrivenStreamAssetSample(),
        &gpuDrivenTerrainP0Sample(),
        &gpuDrivenTerrainP1UnifiedSample(),
        &gpuDrivenRtasVisualizationSample(),
    };
}

} // namespace

RenderSampleDesc RenderSample::desc() const
{
    return RenderSampleDesc{
        .id = std::string(id()),
        .name = std::string(name()),
        .category = std::string(category()),
        .description = std::string(description()),
        .scenePath = scenePath(),
        .loadSceneInEditor = loadSceneInEditor(),
        .graphPath = graphPath(),
        .scenePathTargets = scenePathTargets(),
        .environment = environment(),
        .previewOutput = previewOutput(),
        .requiresStreamline = requiresStreamline(),
    };
}

bool loadRenderSample(
    const RenderSample& sample,
    RenderSampleLoadResult& outResult,
    std::string& outMessage)
{
    outResult = RenderSampleLoadResult{};
    outMessage.clear();

    RenderSampleDesc desc = sample.desc();
    if (desc.id.empty()) {
        outMessage = "Sample id is required";
        return false;
    }
    if (desc.graphPath.empty()) {
        outMessage = "Sample graphPath is required";
        return false;
    }
    if (desc.scenePath.empty() && (desc.loadSceneInEditor || !desc.scenePathTargets.empty())) {
        outMessage = "Sample scenePath is required";
        return false;
    }

    const std::filesystem::path graphPath = projectPath(desc.graphPath);
    RenderGraph graph;
    std::string graphMessage;
    if (!loadRenderGraphFromFile(graphPath, graph, graphMessage)) {
        outMessage = "Sample failed to load RenderGraph: " + graphMessage;
        return false;
    }

    if (!applySampleScenePath(graph, desc, outMessage)) {
        return false;
    }
    if (desc.previewOutput.empty()) {
        desc.previewOutput = graph.firstOutputName();
    }
    if (!graphHasTextureOutput(graph, desc.previewOutput)) {
        outMessage = "Sample previewOutput is not a texture output: " + desc.previewOutput;
        return false;
    }

    graph.clearDirty();
    outResult = RenderSampleLoadResult{
        .desc = std::move(desc),
        .graph = std::move(graph),
        .graphFilePath = graphPath,
    };
    outMessage = "Loaded Sample";
    return true;
}

std::vector<RenderSampleDesc> listBuiltInRenderSamples()
{
    std::vector<RenderSampleDesc> samples;
    for (const RenderSample* sample : builtInRenderSamples()) {
        samples.push_back(sample->desc());
    }
    return samples;
}

bool loadBuiltInRenderSample(
    std::string_view id,
    RenderSampleLoadResult& outResult,
    std::string& outMessage)
{
    for (const RenderSample* sample : builtInRenderSamples()) {
        if (sample->id() == id) {
            return loadRenderSample(*sample, outResult, outMessage);
        }
    }

    outResult = RenderSampleLoadResult{};
    outMessage = std::string("Unknown built-in Sample: ") + std::string(id);
    return false;
}

bool queryBuiltInRenderSampleStreamlineRequirement(
    std::string_view id,
    bool& outRequiresStreamline)
{
    outRequiresStreamline = false;
    for (const RenderSample* sample : builtInRenderSamples()) {
        if (sample->id() == id) {
            outRequiresStreamline = sample->requiresStreamline();
            return true;
        }
    }
    return false;
}

bool setRenderSampleScenePath(
    RenderSampleLoadResult& sample,
    std::string scenePath,
    std::string& outMessage)
{
    outMessage.clear();
    if (scenePath.empty()) {
        outMessage = "Sample scene path override is empty";
        return false;
    }

    sample.desc.scenePath = std::move(scenePath);
    if (!applySampleScenePath(sample.graph, sample.desc, outMessage)) {
        return false;
    }
    sample.graph.clearDirty();
    return true;
}

} // namespace metallic::render
