#pragma once

#include "Runtime/Render/RenderGraph/RenderGraphNode.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/SceneLoad.h"

namespace metallic::editor {

inline bool isStreamSceneProducer(const render::RenderGraphNode& node)
{
    auto properties = node.properties;
    properties.merge_patch(node.runtimeProperties);
    return node.type == "VisibilityBufferPass" &&
        properties.value("enableMeshletStreaming", false) && properties.value("streamAssetOnly", false);
}

inline scene::SceneLoadOptions streamSceneLoadOptions(const render::RenderGraph& graph,
    const std::filesystem::path& sourcePath, const std::filesystem::path& projectDirectory)
{
    scene::SceneLoadOptions options;
    for (const auto& node : graph.nodes()) {
        if (!isStreamSceneProducer(node)) { continue; }
        auto properties = node.properties;
        properties.merge_patch(node.runtimeProperties);
        options.streamAssetPath = scene::meshletStreamAssetPathFor(sourcePath);
        auto previousSource = std::filesystem::path(properties.value("path", ""));
        if (previousSource.is_relative()) { previousSource = projectDirectory / previousSource; }
        std::error_code error;
        // Reloads preserve explicit cache locations; switches must never reuse
        // the previous scene's cache merely because the graph still names it.
        if (std::filesystem::equivalent(previousSource, sourcePath, error) && !error) {
            const auto configured = properties.value("streamAssetPath", "");
            if (!configured.empty()) {
                options.streamAssetPath = configured;
                if (options.streamAssetPath.is_relative()) {
                    options.streamAssetPath = projectDirectory / options.streamAssetPath;
                }
            }
        }
        break;
    }
    return options;
}

// Commit only after asynchronous metadata/cache validation succeeds.
inline bool applyStreamSceneOpen(render::RenderGraph& graph, const std::filesystem::path& sourcePath,
    const std::filesystem::path& streamAssetPath)
{
    if (streamAssetPath.empty()) { return false; }
    bool changed = false;
    for (const auto& node : graph.nodes()) {
        if (!isStreamSceneProducer(node)) { continue; }
        auto properties = node.runtimeProperties;
        properties["path"] = sourcePath.generic_string();
        properties["streamAssetPath"] = streamAssetPath.generic_string();
        properties["sceneBinding"] = "world";
        changed = graph.setNodeRuntimeProperties(node.id, std::move(properties)) || changed;
    }
    return changed;
}

} // namespace metallic::editor
