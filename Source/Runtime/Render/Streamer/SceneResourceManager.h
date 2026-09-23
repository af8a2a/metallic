#pragma once

#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Streamer/SceneStreamingTypes.h"

#include <filesystem>
#include <memory>

namespace metallic::render {

class SceneResourceManager {
public:
    Result acquire(
        Device& device,
        Queue& graphicsQueue,
        const RenderGraphProperties& properties,
        const scene::Scene* runtimeScene,
        SceneResourceFeatureBits features,
        std::shared_ptr<SceneResourceSnapshot>& outSnapshot,
        std::string& log);
    Result resolveScene(
        const RenderGraphProperties& properties,
        const scene::Scene* runtimeScene,
        const scene::Scene*& outScene,
        std::string& log);
    Result beginAcquireAsync(
        Device& device,
        Queue& graphicsQueue,
        const RenderGraphProperties& properties,
        const scene::Scene& runtimeScene,
        SceneResourceFeatureBits features,
        std::shared_ptr<SceneResourceSnapshot>& outSnapshot,
        std::string& log);
    Result pumpAsync(
        const std::shared_ptr<SceneResourceSnapshot>& snapshot,
        const scene::Scene& runtimeScene,
        double budgetMilliseconds,
        bool& complete,
        scene::SceneLoadProgress& progress,
        std::string& log);
    void discard(const std::shared_ptr<SceneResourceSnapshot>& snapshot);

    void clear();

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render
