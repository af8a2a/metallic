#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/Scene.h"

#include <utility>
#include <cmath>
#include <algorithm>

namespace metallic::render {

void RenderWorld::setScene(const scene::Scene* scene)
{
    const uint64_t resourceIdentity = scene != nullptr ? scene->resourceIdentity() : 0;
    const uint64_t graphLifetimeRevision = scene != nullptr ? scene->sceneGraph().lifetimeRevision() : 0;
    if (scene_ == scene && sceneResourceIdentity_ == resourceIdentity &&
        sceneGraphLifetimeRevision_ == graphLifetimeRevision) {
        return;
    }
    // Texture edits can replace GPU resource identity while the authored scene
    // remains alive. Only replacing the document/graph restarts its playback.
    if (scene_ == scene && sceneGraphLifetimeRevision_ == graphLifetimeRevision) {
        sceneResourceIdentity_ = resourceIdentity;
        return;
    }
    scene_ = scene;
    sceneResourceIdentity_ = resourceIdentity;
    sceneGraphLifetimeRevision_ = graphLifetimeRevision;
    environmentElapsedSeconds_ = 0.0;
    environmentJulianDateUTC_ = scene != nullptr ? scene->worldEnvironment().time.julianDateUTC
        : environment::EnvironmentTimeState{}.julianDateUTC;
    (void)setWorldEnvironment(scene != nullptr ? scene->worldEnvironment() : environment::WorldEnvironment{});
    updateEvaluatedEnvironment();
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
    if (scene_ != nullptr && scene_->sceneGraph().lifetimeRevision() != sceneGraphLifetimeRevision_) {
        // Revert/clear can replace a SceneDocument at the same address without
        // going through the normal editor load completion path.
        setScene(scene_);
        pendingChanges_ |= changes;
        return;
    }
    sceneResourceIdentity_ = scene_ != nullptr ? scene_->resourceIdentity() : 0;
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
            updateEvaluatedEnvironment(true);
        }
        return firstOverride;
    }
    if (worldEnvironment_.time.julianDateUTC != environment.time.julianDateUTC) {
        environmentJulianDateUTC_ = environment.time.julianDateUTC;
    }
    worldEnvironment_ = std::move(environment);
    updateEvaluatedEnvironment(true);
    return true;
}

void RenderWorld::advanceWorldEnvironment(double deltaSeconds)
{
    if (!std::isfinite(deltaSeconds) || deltaSeconds <= 0.0) { return; }
    const double elapsed = environmentElapsedSeconds_ + deltaSeconds;
    if (!std::isfinite(elapsed) || elapsed > 1e12) { return; }
    environmentElapsedSeconds_ = elapsed;
    if (!worldEnvironment_.time.paused) {
        auto runtimeTime = worldEnvironment_.time;
        runtimeTime.julianDateUTC = environmentJulianDateUTC_ + deltaSeconds * runtimeTime.timeScale / 86400.0;
        if (std::isfinite(runtimeTime.julianDateUTC)) {
            environmentJulianDateUTC_ = std::clamp(runtimeTime.julianDateUTC,
                environment::kMinimumJulianDateUTC, environment::kMaximumJulianDateUTC);
        }
    }
    updateEvaluatedEnvironment();
}

bool RenderWorld::setEnvironmentElapsedSeconds(double elapsedSeconds)
{
    if (!std::isfinite(elapsedSeconds) || elapsedSeconds < 0.0 || elapsedSeconds > 1e12 ||
        elapsedSeconds == environmentElapsedSeconds_) { return false; }
    environmentElapsedSeconds_ = elapsedSeconds;
    updateEvaluatedEnvironment();
    return true;
}

void RenderWorld::updateEvaluatedEnvironment(bool authoredChange)
{
    auto next = worldEnvironment_.snapshot(celestialRevision_, lightingRevision_, atmosphereRevision_,
        weatherRevision_, astronomyRevision_, environmentElapsedSeconds_, environmentJulianDateUTC_);
    const bool celestialChanged = next.celestial != evaluatedEnvironment_.celestial ||
        next.authoredCelestial != evaluatedEnvironment_.authoredCelestial;
    const bool atmosphereChanged = next.source != evaluatedEnvironment_.source ||
        !(next.authoredAtmosphere == evaluatedEnvironment_.authoredAtmosphere) ||
        !(next.atmosphere == evaluatedEnvironment_.atmosphere);
    const bool astronomyChanged = !(next.time == evaluatedEnvironment_.time) ||
        !(next.astronomy == evaluatedEnvironment_.astronomy) ||
        (next.astronomy.mode != environment::AstronomyMode::Manual &&
         next.evaluatedAstronomy.julianDateUTC != evaluatedEnvironment_.evaluatedAstronomy.julianDateUTC);
    const bool weatherMotion = next.source == environment::EnvironmentSource::PhysicalAtmosphere &&
        next.elapsedSeconds != evaluatedEnvironment_.elapsedSeconds &&
        next.weather.cloudEnabled && next.weather.cloudCoverage > 0.0f &&
        next.weather.cloudDensity > 0.0f && next.weather.cloudExtinctionPerKm > 0.0f &&
        next.weather.windSpeed > 0.0f;
    const bool weatherChanged = !(next.weather == evaluatedEnvironment_.weather) || weatherMotion;
    if (celestialChanged) { ++celestialRevision_; }
    if (atmosphereChanged) { ++atmosphereRevision_; }
    if (astronomyChanged) { ++astronomyRevision_; }
    if (weatherChanged) { ++weatherRevision_; }
    if (atmosphereChanged || weatherChanged) { ++environmentRevision_; }
    if (authoredChange || celestialChanged || atmosphereChanged || astronomyChanged || weatherChanged) {
        ++lightingRevision_;
        pendingChanges_ |= RenderChangeBits::Lighting | RenderChangeBits::InvalidateTemporalHistory;
    }
    next.celestialRevision = celestialRevision_;
    next.atmosphereRevision = atmosphereRevision_;
    next.astronomyRevision = astronomyRevision_;
    next.weatherRevision = weatherRevision_;
    next.lightingRevision = lightingRevision_;
    evaluatedEnvironment_ = std::move(next);
}

} // namespace metallic::render
