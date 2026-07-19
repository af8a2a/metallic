#pragma once

#include "Runtime/Render/RenderPass/ScenePathTraceResources.h"

#include <filesystem>
#include <memory>

namespace metallic::render {

enum class SceneResourceFeatureBits : uint32_t {
    None = 0,
    Geometry = 1u << 0,
    Materials = 1u << 1,
    MaterialTextures = 1u << 2,
    Meshlets = 1u << 3,
    StandardAccelerationStructure = 1u << 4,
    ClusterAccelerationStructure = 1u << 5,
    Environment = 1u << 6,
};

constexpr SceneResourceFeatureBits operator|(
    SceneResourceFeatureBits lhs,
    SceneResourceFeatureBits rhs)
{
    return static_cast<SceneResourceFeatureBits>(
        static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

struct SceneResourceSnapshot {
    std::filesystem::path scenePath;
    std::filesystem::path environmentPath;
    SceneResourceFeatureBits features = SceneResourceFeatureBits::None;
    std::shared_ptr<ScenePathTraceResources> pathTraceResources;
};

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
