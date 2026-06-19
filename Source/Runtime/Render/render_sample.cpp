#include "Runtime/Render/render_sample.h"

#include "json.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

constexpr const char* kPathTracingMeetMatSamplePath = "Samples/pathtracing_meet_mat.metallic_sample.json";

std::filesystem::path projectPath(std::string_view path)
{
    std::filesystem::path resolved(path);
    if (resolved.is_relative()) {
        resolved = std::filesystem::path(PROJECT_SOURCE_DIR) / resolved;
    }
    return resolved;
}

std::filesystem::path resolveReferencedPath(
    const std::filesystem::path& samplePath,
    std::string_view path)
{
    std::filesystem::path resolved(path);
    if (resolved.is_absolute()) {
        return resolved;
    }

    std::filesystem::path projectResolved = projectPath(path);
    if (std::filesystem::exists(projectResolved)) {
        return projectResolved;
    }
    if (samplePath.has_parent_path()) {
        return samplePath.parent_path() / resolved;
    }
    return projectResolved;
}

bool readTextFile(
    const std::filesystem::path& path,
    std::string& outText,
    std::string& outMessage)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        outMessage = "Failed to open Sample file";
        return false;
    }
    std::ostringstream stream;
    stream << file.rdbuf();
    outText = stream.str();
    return true;
}

bool readOptionalString(
    const nlohmann::json& object,
    const char* key,
    std::string& outValue,
    std::string& outMessage)
{
    auto iter = object.find(key);
    if (iter == object.end() || iter->is_null()) {
        outValue.clear();
        return true;
    }
    if (!iter->is_string()) {
        outMessage = std::string("Sample field '") + key + "' must be a string";
        return false;
    }
    outValue = iter->get<std::string>();
    return true;
}

bool readRequiredString(
    const nlohmann::json& object,
    const char* key,
    std::string& outValue,
    std::string& outMessage)
{
    if (!readOptionalString(object, key, outValue, outMessage)) {
        return false;
    }
    if (outValue.empty()) {
        outMessage = std::string("Sample field '") + key + "' is required";
        return false;
    }
    return true;
}

bool readStringArray(
    const nlohmann::json& object,
    const char* key,
    std::vector<std::string>& outValues,
    std::string& outMessage)
{
    outValues.clear();
    auto iter = object.find(key);
    if (iter == object.end() || iter->is_null()) {
        return true;
    }
    if (!iter->is_array()) {
        outMessage = std::string("Sample field '") + key + "' must be an array";
        return false;
    }
    for (const nlohmann::json& value : *iter) {
        if (!value.is_string()) {
            outMessage = std::string("Sample field '") + key + "' must contain only strings";
            return false;
        }
        outValues.push_back(value.get<std::string>());
    }
    return true;
}

bool parseRenderSampleDesc(
    const std::string& text,
    RenderSampleDesc& outDesc,
    std::string& outMessage)
{
    try {
        const nlohmann::json root = nlohmann::json::parse(text);
        if (!root.is_object()) {
            outMessage = "Sample JSON root must be an object";
            return false;
        }
        if (root.value("version", 0) != 1) {
            outMessage = "Unsupported Sample JSON version";
            return false;
        }

        RenderSampleDesc desc;
        if (!readRequiredString(root, "id", desc.id, outMessage) ||
            !readRequiredString(root, "name", desc.name, outMessage) ||
            !readRequiredString(root, "category", desc.category, outMessage) ||
            !readOptionalString(root, "description", desc.description, outMessage) ||
            !readRequiredString(root, "scene", desc.scenePath, outMessage) ||
            !readRequiredString(root, "graph", desc.graphPath, outMessage) ||
            !readStringArray(root, "scenePathTargets", desc.scenePathTargets, outMessage) ||
            !readOptionalString(root, "previewOutput", desc.previewOutput, outMessage)) {
            return false;
        }

        outDesc = std::move(desc);
        return true;
    } catch (const std::exception& exception) {
        outMessage = exception.what();
        return false;
    }
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

RenderSampleDesc makeBuiltInPathTracingMeetMatDesc()
{
    return RenderSampleDesc{
        .id = "pathtracing-meet-mat",
        .name = "Path Tracing / meet_mat",
        .category = "PathTracing",
        .description = "Path tracing validation sample using the meet_mat glTF scene.",
        .scenePath = "Asset/meet_mat.glb",
        .graphPath = "Pipelines/Samples/pathtracing_meet_mat.metallic_graph.json",
        .scenePathTargets = {"PathTrace"},
        .previewOutput = "PathTrace.color",
    };
}

} // namespace

bool loadRenderSampleFromFile(
    const std::filesystem::path& path,
    RenderSampleLoadResult& outResult,
    std::string& outMessage)
{
    outResult = RenderSampleLoadResult{};
    outMessage.clear();

    const std::filesystem::path samplePath = path.is_absolute() ? path : projectPath(path.string());
    std::string text;
    if (!readTextFile(samplePath, text, outMessage)) {
        return false;
    }

    RenderSampleDesc desc;
    if (!parseRenderSampleDesc(text, desc, outMessage)) {
        return false;
    }

    const std::filesystem::path graphPath = resolveReferencedPath(samplePath, desc.graphPath);
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
    if (!graphHasOutput(graph, desc.previewOutput)) {
        outMessage = "Sample previewOutput is not a marked graph output: " + desc.previewOutput;
        return false;
    }

    graph.clearDirty();
    outResult = RenderSampleLoadResult{
        .desc = std::move(desc),
        .graph = std::move(graph),
        .samplePath = samplePath,
        .graphFilePath = graphPath,
    };
    outMessage = "Loaded Sample";
    return true;
}

std::vector<RenderSampleDesc> listBuiltInRenderSamples()
{
    return {makeBuiltInPathTracingMeetMatDesc()};
}

bool loadBuiltInRenderSample(
    std::string_view id,
    RenderSampleLoadResult& outResult,
    std::string& outMessage)
{
    for (const RenderSampleDesc& sample : listBuiltInRenderSamples()) {
        if (sample.id == id) {
            return loadRenderSampleFromFile(projectPath(kPathTracingMeetMatSamplePath), outResult, outMessage);
        }
    }

    outResult = RenderSampleLoadResult{};
    outMessage = std::string("Unknown built-in Sample: ") + std::string(id);
    return false;
}

} // namespace metallic::render
