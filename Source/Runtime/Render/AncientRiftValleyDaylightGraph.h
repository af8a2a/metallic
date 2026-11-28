#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphNode.h"

#include <algorithm>
#include <utility>

namespace metallic::render::detail {

inline bool configureAncientRiftValleyDaylightGraph(RenderGraph& graph,
    const std::filesystem::path& terrainGraphPath, std::string& message)
{
    RenderGraph terrainGraph;
    if (!loadRenderGraphFromFile(terrainGraphPath, terrainGraph, message)) { return false; }

    const auto* terrainVisibility = terrainGraph.findNode("VBuffer");
    const auto* terrainShadows = terrainGraph.findNode("Shadows");
    const auto* visibility = graph.findNode("VBuffer");
    const auto* shadows = graph.findNode("Shadows");
    const auto* deferred = graph.findNode("Deferred");
    const auto* distanceField = graph.findNode("GlobalDistanceFieldPages");
    const auto* reflections = graph.findNode("Reflections");
    if (!terrainVisibility || !terrainShadows || !visibility || !shadows || !deferred || !distanceField || !reflections) {
        message = "Ancient Rift Valley daylight requires the production atmosphere graph and streamed terrain VBuffer/VSM";
        return false;
    }
    const auto scenePath = visibility->properties.find("path");
    const auto detailDepth = std::ranges::find_if(graph.edges(), [](const RenderGraphEdge& edge) {
        return edge.dstPass == "Reflections" && edge.dstField == "detailDepth";
    });
    if (scenePath == visibility->properties.end() || detailDepth == graph.edges().end()) {
        message = "Ancient Rift Valley daylight requires a terrain scene path and the production reflection inputs";
        return false;
    }

    auto visibilityProperties = terrainVisibility->properties;
    visibilityProperties["path"] = *scenePath;
    auto shadowProperties = terrainShadows->properties;
    shadowProperties["publishWorldShadows"] = true;

    // The streamed heightfield has no mesh distance field or surface card bakes.
    // Keep the production screen traces when the scene-specific bake route would remove GI.
    auto deferredProperties = deferred->properties;
    deferredProperties.erase("sceneRealtimeGI");
    deferredProperties["retainEnvironmentDiffuse"] = true;
    auto distanceFieldProperties = distanceField->properties;
    distanceFieldProperties["extent"] = 600.0;
    auto reflectionProperties = reflections->properties;
    reflectionProperties["detailTracing"] = false;

    graph.setNodeProperties(visibility->id, std::move(visibilityProperties));
    graph.setNodeProperties(shadows->id, std::move(shadowProperties));
    graph.setNodeProperties(deferred->id, std::move(deferredProperties));
    graph.setNodeProperties(distanceField->id, std::move(distanceFieldProperties));
    graph.setNodeProperties(reflections->id, std::move(reflectionProperties));
    graph.removeEdge(detailDepth->id);
    graph.setName("Ancient Rift Valley Daylight");
    return true;
}

} // namespace metallic::render::detail
