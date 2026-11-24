#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphNode.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <array>
#include <unordered_map>

namespace metallic::render::detail {

using RealtimeGIProperties = std::unordered_map<uint32_t, RenderGraphProperties>;
inline constexpr std::array<std::string_view, 7> kRealtimeGIPasses{"GlobalDistanceFieldPages", "SurfaceCache",
    "Radiosity", "ScreenTrace", "Gather", "Temporal", "Reflections"};

inline bool isRealtimeGIOutput(std::string_view output)
{
    const auto pass = output.substr(0, output.find('.'));
    return output.starts_with("Deferred.lumen") || std::ranges::find(kRealtimeGIPasses, pass) != kRealtimeGIPasses.end();
}

inline bool hasSceneRealtimeGIRoute(const RenderGraph& graph)
{
    const auto* deferred = graph.findNode("Deferred");
    return deferred && deferred->properties.contains("sceneRealtimeGI");
}

inline bool usesSceneRealtimeGI(const RenderGraph& graph)
{
    const auto* deferred = graph.findNode("Deferred");
    if (!deferred) { return false; }
    const auto request = deferred->properties.find("sceneRealtimeGI");
    return request != deferred->properties.end() && *request == true;
}

inline uint64_t realtimeGISceneIdentity(const scene::Scene* scene, const RenderWorld& world)
{
    return scene && scene->realtimeGI() && !world.environment().enabled &&
        world.worldEnvironment().source != environment::EnvironmentSource::PhysicalAtmosphere ?
        scene->resourceIdentity() : 0;
}

inline RenderGraphProperties applyRealtimeGIProperties(const RenderGraphNode& node, const RealtimeGIProperties& overrides)
{
    const auto properties = overrides.find(node.id);
    if (properties == overrides.end()) { return node.properties; }
    auto resolved = node.properties;
    resolved.update(properties->second);
    return resolved;
}

inline bool resolveSceneRealtimeGI(RenderGraph& graph, const scene::Scene* scene, RealtimeGIProperties& overrides,
    std::string& log)
{
    for (std::string_view name : kRealtimeGIPasses) {
        if (!graph.findNode(name)) {
            log = "Scene realtime GI requires pass '" + std::string(name) + "'";
            return false;
        }
    }

    if (scene) {
        const auto& gi = *scene->realtimeGI();
        const auto paths = [](const auto& value) {
            auto result = RenderGraphProperties::array();
            for (const auto& path : value) { result.push_back(path.generic_string()); }
            return result;
        };
        auto& gdf = overrides[graph.findNode("GlobalDistanceFieldPages")->id];
        const auto bounds = scene->bounds();
        const auto center = bounds.center();
        gdf = {{"distanceFieldPaths", paths(gi.distanceFieldPaths)},
            {"center", gi.center.value_or(std::array<double, 3>{center.x, center.y, center.z})},
            {"extent", gi.extent.value_or(std::max(bounds.radius(), 1.0f))}};
        overrides[graph.findNode("SurfaceCache")->id] = {{"meshCardPaths", paths(gi.meshCardPaths)}};
        const bool detail = !gi.detailDistanceFieldPaths.empty();
        overrides[graph.findNode("Reflections")->id] = {{"detailTracing", detail},
            {"detailDistanceFieldPaths", paths(gi.detailDistanceFieldPaths)}};
        if (!detail) {
            for (const auto& edge : graph.edges()) {
                if (edge.dstPass == "Reflections" && edge.dstField == "detailDepth") {
                    graph.removeEdge(edge.id);
                    break;
                }
            }
        }
    } else {
        std::vector<std::string> consumers;
        for (const auto& edge : graph.edges()) {
            if (edge.srcPass == "Reflections" && edge.srcField == "color") {
                consumers.push_back(makeRenderGraphFieldName(edge.dstPass, edge.dstField));
            }
        }
        for (std::string_view name : kRealtimeGIPasses) { graph.removeNode(graph.findNode(name)->id); }
        for (const auto& consumer : consumers) {
            if (!graph.addEdge("Deferred.color", consumer)) {
                log = "Could not connect direct lighting to " + consumer;
                return false;
            }
        }
        overrides[graph.findNode("Deferred")->id] = {{"exportLumenReceivers", false},
            {"exportLumenPreviousPosition", false}, {"exportLumenSpecular", false}};
    }

    for (const auto& entry : overrides) {
        const auto* node = graph.findNode(entry.first);
        graph.setNodeProperties(node->id, applyRealtimeGIProperties(*node, overrides));
    }
    return true;
}

} // namespace metallic::render::detail
