#include "Runtime/Render/Subsystem/LegacyEnvironmentMigration.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace metallic::render {
namespace {

struct Candidate {
    uint32_t nodeId = 0;
    std::string nodeName;
    EnvironmentSettings settings;
};

EnvironmentSettings parseEnvironment(
    const RenderGraphProperties& properties,
    const std::filesystem::path& relativePathBase)
{
    EnvironmentSettings result;
    if (!properties.is_object()) {
        return result;
    }
    result.enabled = properties.value("enabled", true);
    result.visible = properties.value("visible", true);
    result.intensity = properties.value("intensity", 1.0f);
    result.rotationDegrees = properties.value("rotationDegrees", 0.0f);
    if (!std::isfinite(result.intensity)) {
        result.intensity = 1.0f;
    }
    result.intensity = std::max(result.intensity, 0.0f);
    if (!std::isfinite(result.rotationDegrees)) {
        result.rotationDegrees = 0.0f;
    }
    if (properties.contains("path") && properties["path"].is_string()) {
        result.path = properties["path"].get<std::string>();
        if (!result.path.empty() && result.path.is_relative() && !relativePathBase.empty()) {
            result.path = relativePathBase / result.path;
        }
    }
    return result;
}

void appendUpstreamNames(
    const RenderGraph& graph,
    std::string_view passName,
    std::unordered_set<std::string>& visited,
    std::vector<std::string>& orderedNames)
{
    if (!visited.emplace(passName).second) {
        return;
    }
    orderedNames.emplace_back(passName);
    for (const RenderGraphEdge& edge : graph.edges()) {
        if (edge.dstPass == passName) {
            appendUpstreamNames(graph, edge.srcPass, visited, orderedNames);
        }
    }
}

} // namespace

LegacyEnvironmentMigrationResult migrateLegacyEnvironmentSettings(
    RenderGraph& graph,
    const std::filesystem::path& relativePathBase)
{
    LegacyEnvironmentMigrationResult result;
    std::vector<Candidate> candidates;
    for (const RenderGraphNode& node : graph.nodes()) {
        const RenderGraphProperties* environment = nullptr;
        if (node.runtimeProperties.is_object() &&
            node.runtimeProperties.contains("environment") &&
            node.runtimeProperties["environment"].is_object()) {
            environment = &node.runtimeProperties["environment"];
        } else if (node.properties.is_object() &&
            node.properties.contains("environment") &&
            node.properties["environment"].is_object()) {
            environment = &node.properties["environment"];
        }
        if (environment != nullptr) {
            candidates.push_back(Candidate{
                .nodeId = node.id,
                .nodeName = node.name,
                .settings = parseEnvironment(*environment, relativePathBase),
            });
        }
    }
    if (candidates.empty()) {
        return result;
    }

    const Candidate* selected = &candidates.front();
    if (!graph.outputs().empty()) {
        std::unordered_set<std::string> visited;
        std::vector<std::string> upstreamNames;
        appendUpstreamNames(graph, graph.outputs().front().passName, visited, upstreamNames);
        for (const std::string& name : upstreamNames) {
            const auto found = std::find_if(
                candidates.begin(),
                candidates.end(),
                [&name](const Candidate& candidate) { return candidate.nodeName == name; });
            if (found != candidates.end()) {
                selected = &(*found);
                break;
            }
        }
    }

    result.found = true;
    result.settings = selected->settings;
    result.selectedNode = selected->nodeName;
    for (const Candidate& candidate : candidates) {
        if (candidate.nodeId != selected->nodeId && candidate.settings != selected->settings) {
            result.ignoredNodes.push_back(candidate.nodeName);
        }
    }
    if (!result.ignoredNodes.empty()) {
        result.warning = "Legacy graph environment conflict: selected '" + result.selectedNode +
            "' and ignored ";
        for (size_t index = 0; index < result.ignoredNodes.size(); ++index) {
            if (index != 0) {
                result.warning += ", ";
            }
            result.warning += "'" + result.ignoredNodes[index] + "'";
        }
        result.warning += ".";
    }

    std::vector<uint32_t> nodeIds;
    nodeIds.reserve(graph.nodes().size());
    for (const RenderGraphNode& node : graph.nodes()) {
        nodeIds.push_back(node.id);
    }
    for (uint32_t nodeId : nodeIds) {
        RenderGraphNode* node = graph.findNode(nodeId);
        if (node == nullptr) {
            continue;
        }
        RenderGraphProperties properties = node->properties;
        RenderGraphProperties runtimeProperties = node->runtimeProperties;
        const bool staticRemoved = properties.is_object() && properties.erase("environment") != 0;
        const bool runtimeRemoved = runtimeProperties.is_object() && runtimeProperties.erase("environment") != 0;
        if (staticRemoved) {
            graph.setNodeProperties(nodeId, std::move(properties));
        }
        if (runtimeRemoved) {
            graph.setNodeRuntimeProperties(nodeId, std::move(runtimeProperties));
        }
    }
    return result;
}

} // namespace metallic::render
