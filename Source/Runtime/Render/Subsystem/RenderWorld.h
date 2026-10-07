#pragma once

#include "Runtime/Scene/SceneEnvironment.h"
#include "Runtime/Scene/SceneLighting.h"
#include "Runtime/Environment/WorldEnvironment.h"

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
    void notifySceneChanged(RenderChangeBits changes = RenderChangeBits::Lighting |
        RenderChangeBits::Geometry | RenderChangeBits::Material |
        RenderChangeBits::InvalidateTemporalHistory);
    const scene::Scene* scene() const { return scene_; }

    void setEnvironment(EnvironmentSettings settings);
    const EnvironmentSettings& environment() const { return environment_; }
    bool setWorldEnvironment(environment::WorldEnvironment environment);
    bool hasWorldEnvironmentOverride() const { return worldEnvironmentOverride_; }
    const environment::WorldEnvironment& worldEnvironment() const { return worldEnvironment_; }
    // Runtime time never writes back to the authored document. The host advances
    // it once per frame, before consumers acquire this immutable evaluation.
    void advanceWorldEnvironment(double deltaSeconds);
    bool setEnvironmentElapsedSeconds(double elapsedSeconds);
    double environmentElapsedSeconds() const { return environmentElapsedSeconds_; }
    double environmentJulianDateUTC() const { return environmentJulianDateUTC_; }
    environment::EnvironmentSnapshot environmentSnapshot() const
    {
        auto snapshot = evaluatedEnvironment_;
        snapshot.lightingRevision = lightingRevision_;
        return snapshot;
    }
    bool setLighting(scene::LightingSettings lighting);
    const scene::LightingSettings& lighting() const { return lighting_; }
    uint64_t lightingRevision() const { return lightingRevision_; }

    uint64_t sceneRevision() const { return sceneRevision_; }
    // Only geometry/material changes require GPUScene content fingerprints.
    uint64_t sceneContentRevision() const { return sceneContentRevision_; }
    uint64_t environmentRevision() const { return environmentRevision_; }
    RenderChangeBits consumeChanges();

private:
    void updateEvaluatedEnvironment(bool authoredChange = false);
    const scene::Scene* scene_ = nullptr;
    uint64_t sceneResourceIdentity_ = 0;
    uint64_t sceneGraphLifetimeRevision_ = 0;
    EnvironmentSettings environment_;
    scene::LightingSettings lighting_;
    environment::WorldEnvironment worldEnvironment_;
    environment::EnvironmentSnapshot evaluatedEnvironment_ = worldEnvironment_.snapshot();
    double environmentElapsedSeconds_ = 0.0;
    double environmentJulianDateUTC_ = worldEnvironment_.time.julianDateUTC;
    bool worldEnvironmentOverride_ = false;
    uint64_t celestialRevision_ = 1;
    uint64_t atmosphereRevision_ = 0;
    uint64_t weatherRevision_ = 0;
    uint64_t astronomyRevision_ = 0;
    uint64_t lightingRevision_ = 1;
    uint64_t sceneRevision_ = 1;
    uint64_t sceneContentRevision_ = 1;
    uint64_t environmentRevision_ = 1;
    RenderChangeBits pendingChanges_ = RenderChangeBits::None;
};

} // namespace metallic::render
