#include "Runtime/Render/RenderSample.h"

#include <algorithm>
#include <string>
#include <utility>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
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

bool graphHasOutput(const RenderGraph& graph, std::string_view outputName)
{
    return std::any_of(
        graph.outputs().begin(),
        graph.outputs().end(),
        [&](const RenderGraphOutput& output) {
            return makeRenderGraphFieldName(output.passName, output.fieldName) == outputName;
        });
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

RenderGraphProperties environmentProperties(const RenderSampleEnvironmentDesc& environment)
{
    return RenderGraphProperties{
        {"enabled", environment.enabled},
        {"path", environment.path},
        {"intensity", environment.intensity},
        {"rotationDegrees", environment.rotationDegrees},
        {"visible", environment.visible},
    };
}

bool applySampleEnvironment(RenderGraph& graph, const RenderSampleDesc& desc, std::string& outMessage)
{
    if (desc.environmentTargets.empty()) {
        return true;
    }

    for (const std::string& target : desc.environmentTargets) {
        RenderGraphNode* node = graph.findNode(target);
        if (node == nullptr) {
            outMessage = "Sample environmentTargets node not found: " + target;
            return false;
        }
        RenderGraphProperties properties = node->properties;
        if (!properties.is_object()) {
            properties = RenderGraphProperties::object();
        }
        properties["environment"] = environmentProperties(desc.environment);
        if (!graph.setNodeProperties(node->id, std::move(properties))) {
            outMessage = "Sample failed to update environment properties: " + target;
            return false;
        }
    }
    return true;
}

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
    RenderSampleEnvironmentDesc environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::vector<std::string> environmentTargets() const override { return {"PathTrace"}; }
    std::string previewOutput() const override { return "PathTrace.color"; }
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
    RenderSampleEnvironmentDesc environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::vector<std::string> environmentTargets() const override { return {"PathTrace"}; }
    std::string previewOutput() const override { return "PathTrace.color"; }
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
    RenderSampleEnvironmentDesc environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::vector<std::string> environmentTargets() const override { return {"PathTrace"}; }
    std::string previewOutput() const override { return "DlssRr.color"; }
};

class RtxdiSample final : public RenderSample {
public:
    std::string_view id() const override { return "rtxdi-sample"; }
    std::string_view name() const override { return "RTXDI / ReSTIR DI"; }
    std::string_view category() const override { return "RTXDI"; }
    std::string_view description() const override
    {
        return "Fused spatiotemporal ReSTIR DI over hundreds of animated lights, denoised with NRD RELAX.";
    }
    std::string scenePath() const override { return "Asset/meet_mat.glb"; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/rtxdi_meet_mat.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"Rtxdi"}; }
    std::string previewOutput() const override { return "Composite.color"; }
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
    std::string previewOutput() const override { return "MaterialViz.color"; }
};

class GPUDrivenSample final : public RenderSample {
public:
    std::string_view id() const override { return "gpu-driven-sample"; }
    std::string_view name() const override { return "GPUDrivenSample"; }
    std::string_view category() const override { return "GPUDriven"; }
    std::string_view description() const override
    {
        return "GPU-driven sample scaffold using SuperSponza and a mesh shader meshlet preview pass.";
    }
    std::string scenePath() const override { return "Asset/SuperSponza/NewSponza_Main_glTF_003.gltf"; }
    bool loadSceneInEditor() const override { return false; }
    std::string graphPath() const override
    {
        return "Pipelines/Samples/gpu_driven_sponza.metallic_graph.json";
    }
    std::vector<std::string> scenePathTargets() const override { return {"GPUDriven"}; }
    RenderSampleEnvironmentDesc environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::vector<std::string> environmentTargets() const override { return {"GPUDriven"}; }
    std::string previewOutput() const override { return "GPUDriven.color"; }
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
    RenderSampleEnvironmentDesc environment() const override
    {
        return RenderSampleEnvironmentDesc{
            .enabled = true,
            .path = "Asset/ABeautifulGame/environment.hdr",
            .intensity = 1.0f,
            .rotationDegrees = 0.0f,
            .visible = true,
        };
    }
    std::vector<std::string> environmentTargets() const override { return {"GPUDriven"}; }
    std::string previewOutput() const override { return "GPUDriven.color"; }
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
    std::string previewOutput() const override { return "GPUDriven.color"; }
};

const RenderSample& pathTracingMeetMatSample()
{
    static const PathTracingMeetMatSample sample;
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

const RenderSample& gpuDrivenSample()
{
    static const GPUDrivenSample sample;
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

std::vector<const RenderSample*> builtInRenderSamples()
{
    return {
        &pathTracingMeetMatSample(),
        &pathTracingSample(),
        &pathTracingDlssRrSample(),
        &rtxdiSample(),
        &materialVisualizationABeautifulGameSample(),
        &gpuDrivenSample(),
        &gpuDrivenStreamAssetSample(),
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
        .environmentTargets = environmentTargets(),
        .previewOutput = previewOutput(),
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
    if (desc.scenePath.empty()) {
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
    if (!applySampleEnvironment(graph, desc, outMessage)) {
        return false;
    }

    if (desc.previewOutput.empty()) {
        desc.previewOutput = graph.firstOutputName();
    }
    if (!graphHasOutput(graph, desc.previewOutput)) {
        outMessage = "Sample previewOutput is not a marked graph output: " + desc.previewOutput;
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
