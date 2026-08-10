#include "Runtime/Render/Subsystem/RenderWorld.h"

#include <utility>

namespace metallic::render {

void RenderWorld::setScene(const scene::Scene* scene)
{
    if (scene_ == scene) {
        return;
    }
    scene_ = scene;
    ++sceneRevision_;
    pendingChanges_ |= RenderChangeBits::Lighting |
        RenderChangeBits::Geometry |
        RenderChangeBits::Material |
        RenderChangeBits::InvalidateTemporalHistory;
}

void RenderWorld::notifySceneChanged()
{
    ++sceneRevision_;
    pendingChanges_ |= RenderChangeBits::Lighting |
        RenderChangeBits::Geometry |
        RenderChangeBits::Material |
        RenderChangeBits::InvalidateTemporalHistory;
}

void RenderWorld::setEnvironment(EnvironmentSettings settings)
{
    if (environment_ == settings) {
        return;
    }
    environment_ = std::move(settings);
    ++environmentRevision_;
    pendingChanges_ |= RenderChangeBits::Lighting |
        RenderChangeBits::InvalidateTemporalHistory;
}

RenderChangeBits RenderWorld::consumeChanges()
{
    const RenderChangeBits changes = pendingChanges_;
    pendingChanges_ = RenderChangeBits::None;
    return changes;
}

} // namespace metallic::render
