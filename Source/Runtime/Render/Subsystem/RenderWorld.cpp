#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <utility>

namespace metallic::render {

void RenderWorld::setScene(const scene::Scene* scene)
{
    if (scene_ == scene) {
        return;
    }
    scene_ = scene;
    (void)setWorldEnvironment(scene != nullptr ? scene->worldEnvironment() : environment::WorldEnvironment{});
    worldEnvironmentOverride_ = false;
    ++sceneRevision_;
    ++sceneContentRevision_;
    pendingChanges_ |= RenderChangeBits::Lighting |
        RenderChangeBits::Geometry |
        RenderChangeBits::Material |
        RenderChangeBits::InvalidateTemporalHistory;
}

void RenderWorld::notifySceneChanged(RenderChangeBits changes)
{
    ++sceneRevision_;
    if (hasRenderChange(changes, RenderChangeBits::Geometry) ||
        hasRenderChange(changes, RenderChangeBits::Material)) {
        ++sceneContentRevision_;
    }
    pendingChanges_ |= changes;
}

void RenderWorld::setEnvironment(EnvironmentSettings settings)
{
    if (environment_ == settings) {
        return;
    }
    environment_ = std::move(settings);
    ++environmentRevision_;
    ++lightingRevision_;
    pendingChanges_ |= RenderChangeBits::Lighting |
        RenderChangeBits::InvalidateTemporalHistory;
}

RenderChangeBits RenderWorld::consumeChanges()
{
    const RenderChangeBits changes = pendingChanges_;
    pendingChanges_ = RenderChangeBits::None;
    return changes;
}

bool RenderWorld::setLighting(scene::LightingSettings lighting)
{
    if (!scene::validLightingSettings(lighting)) {
        return false;
    }
    lighting_ = std::move(lighting);
    ++lightingRevision_;
    pendingChanges_ |= RenderChangeBits::Lighting | RenderChangeBits::InvalidateTemporalHistory;
    return true;
}

bool RenderWorld::setWorldEnvironment(environment::WorldEnvironment environment)
{
    if (!environment::validWorldEnvironment(environment)) {
        return false;
    }
    const bool firstOverride = !worldEnvironmentOverride_;
    worldEnvironmentOverride_ = true;
    if (worldEnvironment_ == environment) {
        if (firstOverride) {
            ++lightingRevision_;
            pendingChanges_ |= RenderChangeBits::Lighting | RenderChangeBits::InvalidateTemporalHistory;
        }
        return firstOverride;
    }
    worldEnvironment_ = std::move(environment);
    ++celestialRevision_;
    ++lightingRevision_;
    pendingChanges_ |= RenderChangeBits::Lighting | RenderChangeBits::InvalidateTemporalHistory;
    return true;
}

} // namespace metallic::render
