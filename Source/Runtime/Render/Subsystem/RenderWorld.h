#pragma once

#include "Runtime/Scene/SceneEnvironment.h"

#include <cstdint>

namespace metallic::scene {
class Scene;
}

namespace metallic::render {

enum class RenderChangeBits : uint32_t {
    None = 0,
    Lighting = 1u << 0,
    Geometry = 1u << 1,
    Material = 1u << 2,
    Residency = 1u << 3,
    InvalidateTemporalHistory = 1u << 4,
};

constexpr RenderChangeBits operator|(RenderChangeBits lhs, RenderChangeBits rhs)
{
    return static_cast<RenderChangeBits>(
        static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

constexpr RenderChangeBits operator&(RenderChangeBits lhs, RenderChangeBits rhs)
{
    return static_cast<RenderChangeBits>(
        static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

constexpr RenderChangeBits& operator|=(RenderChangeBits& lhs, RenderChangeBits rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

constexpr bool hasRenderChange(RenderChangeBits value, RenderChangeBits bit)
{
    return (value & bit) != RenderChangeBits::None;
}

using EnvironmentSettings = scene::EnvironmentSettings;

class RenderWorld {
public:
    void setScene(const scene::Scene* scene);
    void notifySceneChanged();
    const scene::Scene* scene() const { return scene_; }

    void setEnvironment(EnvironmentSettings settings);
    const EnvironmentSettings& environment() const { return environment_; }

    uint64_t sceneRevision() const { return sceneRevision_; }
    uint64_t environmentRevision() const { return environmentRevision_; }
    RenderChangeBits consumeChanges();

private:
    const scene::Scene* scene_ = nullptr;
    EnvironmentSettings environment_;
    uint64_t sceneRevision_ = 1;
    uint64_t environmentRevision_ = 1;
    RenderChangeBits pendingChanges_ = RenderChangeBits::None;
};

} // namespace metallic::render
