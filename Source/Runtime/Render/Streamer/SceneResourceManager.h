#pragma once

#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Streamer/SceneStreamingTypes.h"

#include <filesystem>
#include <memory>

namespace metallic::render {

class SceneResourceManager {
public:
    [[nodiscard]] Result<std::shared_ptr<SceneResourceSnapshot>> acquire(
        Device& device,
        Queue& graphicsQueue,
        const RenderGraphProperties& properties,
        const scene::Scene* runtimeScene,
        SceneResourceFeatureBits features,
        std::string& log);
    [[nodiscard]] Result<const scene::Scene*> resolveScene(
        const RenderGraphProperties& properties,
        const scene::Scene* runtimeScene,
        std::string& log);
    [[nodiscard]] Result<std::shared_ptr<SceneResourceSnapshot>> beginAcquireAsync(
        Device& device,
        Queue& graphicsQueue,
        const RenderGraphProperties& properties,
        const scene::Scene& runtimeScene,
        SceneResourceFeatureBits features,
        std::string& log);
    [[nodiscard]] Result<bool> pumpAsync(
        const std::shared_ptr<SceneResourceSnapshot>& snapshot,
        const scene::Scene& runtimeScene,
        double budgetMilliseconds,
        scene::SceneLoadProgress& progress,
        std::string& log);
    void discard(const std::shared_ptr<SceneResourceSnapshot>& snapshot);

    void clear();

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render
