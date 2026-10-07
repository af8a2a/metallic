#include <fstream>

#include "RHITest.h"
#include "harness/Evidence.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/SceneDocument.h"

#include <array>
#include <cmath>
#include <limits>

namespace metallic::tests {
namespace {

#define WORLD_CHECK(expression) do { \
    if (!(expression)) { return RHITestResult::fail(#expression); } \
} while (false)

bench::Metadata worldContractMetadata(std::string coverage)
{
    return {.suite = "contract", .layer = bench::Layer::Core,
        .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
        .coverage = {std::move(coverage)}};
}

bool sameIrradiance(const render::GPUCelestialLightRecords& a, const render::GPUCelestialLightRecords& b)
{
    for (size_t light = 0; light < a.size(); ++light) {
        for (size_t channel = 0; channel < 3; ++channel) {
            if (a[light].irradiance[channel] != b[light].irradiance[channel] ||
                a[light].diskRadiance[channel] != b[light].diskRadiance[channel]) { return false; }
        }
        if (a[light].flags != b[light].flags) { return false; }
    }
    return true;
}

class DynamicWorldRuntimeStateTest final : public RHITest {
public:
    DynamicWorldRuntimeStateTest() { name = "dynamic_world_runtime_state"; type = RHITestType::Resource; }
    std::optional<bench::Metadata> metadata() const override
    {
        return worldContractMetadata("environment.runtime.clock.wind.authoring");
    }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        using namespace environment;
        using namespace render;
        scene::SceneDocument document;
        WorldEnvironment authored;
        authored.source = EnvironmentSource::PhysicalAtmosphere;
        authored.sun.enabled = authored.moon.enabled = true;
        authored.time = {.julianDateUTC = 2460483.0, .timeScale = 60.0, .paused = false};
        authored.astronomy.mode = AstronomyMode::Astronomical;
        authored.astronomy.latitudeDegrees = 35.0;
        authored.weather.cloudEnabled = true;
        authored.weather.cloudCoverage = 0.6f;
        authored.weather.windSpeed = 10.0f;
        WORLD_CHECK(document.setWorldEnvironment(authored));
        document.setDirty(false); // Establish a clean authoring baseline without asset I/O.
        const auto staticDocumentSnapshot = document.environmentSnapshot();
        RenderWorld world;
        world.setScene(&document);
        (void)world.consumeChanges();
        const auto initial = world.environmentSnapshot();
        const auto contentRevision = world.sceneContentRevision();
        world.advanceWorldEnvironment(2.0);
        const auto advanced = world.environmentSnapshot();
        WORLD_CHECK(std::abs(world.environmentJulianDateUTC() - (authored.time.julianDateUTC + 120.0 / 86400.0)) < 1e-9);
        WORLD_CHECK(advanced.elapsedSeconds == 2.0);
        WORLD_CHECK(advanced.evaluatedAstronomy.julianDateUTC == world.environmentJulianDateUTC());
        WORLD_CHECK(advanced.celestialRevision > initial.celestialRevision);
        WORLD_CHECK(advanced.astronomyRevision > initial.astronomyRevision);
        WORLD_CHECK(advanced.weatherRevision > initial.weatherRevision);
        WORLD_CHECK(advanced.atmosphereRevision == initial.atmosphereRevision);
        WORLD_CHECK(world.worldEnvironment() == authored && document.worldEnvironment() == authored);
        WORLD_CHECK(!document.dirty());
        WORLD_CHECK(document.environmentSnapshot().evaluatedAstronomy.julianDateUTC == authored.time.julianDateUTC);
        WORLD_CHECK(initial.elapsedSeconds == 0.0 && initial.celestial == staticDocumentSnapshot.celestial);
        WORLD_CHECK(world.environmentSnapshot().celestial == advanced.celestial);
        WORLD_CHECK(world.environmentSnapshot().weatherRevision == advanced.weatherRevision);
        const auto changes = world.consumeChanges();
        WORLD_CHECK(hasRenderChange(changes, RenderChangeBits::Lighting));
        WORLD_CHECK(hasRenderChange(changes, RenderChangeBits::InvalidateTemporalHistory));
        WORLD_CHECK(!hasRenderChange(changes, RenderChangeBits::Geometry));
        WORLD_CHECK(world.sceneContentRevision() == contentRevision);

        auto edited = authored;
        edited.time.paused = true;
        const double pausedJulianDate = world.environmentJulianDateUTC();
        WORLD_CHECK(world.setWorldEnvironment(edited));
        WORLD_CHECK(world.environmentJulianDateUTC() == pausedJulianDate);
        (void)world.consumeChanges();
        const auto paused = world.environmentSnapshot();
        world.advanceWorldEnvironment(5.0);
        const auto windyPause = world.environmentSnapshot();
        WORLD_CHECK(world.environmentJulianDateUTC() == pausedJulianDate);
        WORLD_CHECK(windyPause.celestial == paused.celestial);
        WORLD_CHECK(windyPause.astronomyRevision == paused.astronomyRevision);
        WORLD_CHECK(windyPause.celestialRevision == paused.celestialRevision);
        WORLD_CHECK(windyPause.weatherRevision > paused.weatherRevision);
        WORLD_CHECK(windyPause.elapsedSeconds == 7.0);
        WORLD_CHECK(!document.dirty());

        edited.time.timeScale = -120.0;
        edited.time.paused = false;
        WORLD_CHECK(world.setWorldEnvironment(edited));
        WORLD_CHECK(world.environmentJulianDateUTC() == pausedJulianDate);
        world.advanceWorldEnvironment(1.0);
        WORLD_CHECK(std::abs(world.environmentJulianDateUTC() - (pausedJulianDate - 120.0 / 86400.0)) < 1e-9);
        edited.time.timeScale = 0.0;
        WORLD_CHECK(world.setWorldEnvironment(edited));
        const auto stoppedClock = world.environmentSnapshot();
        world.advanceWorldEnvironment(7.0);
        WORLD_CHECK(world.environmentJulianDateUTC() == stoppedClock.evaluatedAstronomy.julianDateUTC);
        WORLD_CHECK(world.environmentSnapshot().astronomyRevision == stoppedClock.astronomyRevision);
        WORLD_CHECK(world.environmentSnapshot().weatherRevision > stoppedClock.weatherRevision);

        const auto beforeWindSeek = world.environmentSnapshot();
        WORLD_CHECK(world.setEnvironmentElapsedSeconds(42.5));
        WORLD_CHECK(world.environmentElapsedSeconds() == 42.5);
        WORLD_CHECK(world.environmentJulianDateUTC() == beforeWindSeek.evaluatedAstronomy.julianDateUTC);
        WORLD_CHECK(world.environmentSnapshot().weatherRevision > beforeWindSeek.weatherRevision);
        WORLD_CHECK(!world.setEnvironmentElapsedSeconds(42.5));
        WORLD_CHECK(world.setEnvironmentElapsedSeconds(0.0));
        WORLD_CHECK(!world.setEnvironmentElapsedSeconds(-1.0));
        WORLD_CHECK(!world.setEnvironmentElapsedSeconds(std::numeric_limits<double>::infinity()));
        WORLD_CHECK(!world.setEnvironmentElapsedSeconds(std::numeric_limits<double>::quiet_NaN()));
        const auto beforeInvalidAdvance = world.environmentSnapshot();
        for (double invalidDelta : {-1.0, 0.0, std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN()}) {
            world.advanceWorldEnvironment(invalidDelta);
        }
        WORLD_CHECK(world.environmentSnapshot().elapsedSeconds == beforeInvalidAdvance.elapsedSeconds);
        WORLD_CHECK(world.environmentSnapshot().lightingRevision == beforeInvalidAdvance.lightingRevision);

        edited.time.julianDateUTC += 3.0;
        WORLD_CHECK(world.setWorldEnvironment(edited));
        WORLD_CHECK(world.environmentJulianDateUTC() == edited.time.julianDateUTC);
        WORLD_CHECK(world.environmentElapsedSeconds() == 0.0);
        edited.time.paused = true;
        edited.weather.windSpeed = 0.0f;
        WORLD_CHECK(world.setWorldEnvironment(edited));
        (void)world.consumeChanges();
        const auto motionless = world.environmentSnapshot();
        world.advanceWorldEnvironment(10.0);
        const auto later = world.environmentSnapshot();
        WORLD_CHECK(later.elapsedSeconds == 10.0);
        WORLD_CHECK(later.weatherRevision == motionless.weatherRevision);
        WORLD_CHECK(later.astronomyRevision == motionless.astronomyRevision);
        WORLD_CHECK(later.lightingRevision == motionless.lightingRevision);
        WORLD_CHECK(world.consumeChanges() == RenderChangeBits::None);

        edited.weather.windSpeed = 10.0f;
        edited.source = EnvironmentSource::HDRI;
        WORLD_CHECK(world.setWorldEnvironment(edited));
        (void)world.consumeChanges();
        const auto hdriWithRetainedClouds = world.environmentSnapshot();
        world.advanceWorldEnvironment(3.0);
        WORLD_CHECK(world.environmentSnapshot().weatherRevision == hdriWithRetainedClouds.weatherRevision);
        WORLD_CHECK(world.environmentSnapshot().lightingRevision == hdriWithRetainedClouds.lightingRevision);
        WORLD_CHECK(world.sceneContentRevision() == contentRevision);
        WORLD_CHECK(world.consumeChanges() == RenderChangeBits::None);
        edited.source = EnvironmentSource::PhysicalAtmosphere;
        edited.weather.cloudExtinctionPerKm = 0.0f;
        WORLD_CHECK(world.setWorldEnvironment(edited));
        (void)world.consumeChanges();
        const auto transparentClouds = world.environmentSnapshot();
        world.advanceWorldEnvironment(3.0);
        WORLD_CHECK(world.environmentSnapshot().weatherRevision == transparentClouds.weatherRevision);
        WORLD_CHECK(world.environmentSnapshot().lightingRevision == transparentClouds.lightingRevision);
        WORLD_CHECK(world.sceneContentRevision() == contentRevision);
        WORLD_CHECK(world.consumeChanges() == RenderChangeBits::None);
        WORLD_CHECK(world.setEnvironmentElapsedSeconds(10.0));
        edited.weather.cloudExtinctionPerKm = authored.weather.cloudExtinctionPerKm;
        edited.weather.humidity = 0.5f;
        WORLD_CHECK(world.setWorldEnvironment(edited));
        WORLD_CHECK(world.environmentSnapshot().weatherRevision > later.weatherRevision);
        WORLD_CHECK(world.environmentSnapshot().atmosphereRevision > later.atmosphereRevision);
        WORLD_CHECK(world.environmentSnapshot().authoredAtmosphere == edited.atmosphere);
        WORLD_CHECK(!document.dirty() && document.worldEnvironment() == authored);
        WORLD_CHECK(resolveWorldEnvironment(&document, &world).elapsedSeconds == 10.0);
        scene::SceneDocument overrideDocument;
        auto overrideAuthored = authored;
        overrideAuthored.time.julianDateUTC += 11.0;
        WORLD_CHECK(overrideDocument.setWorldEnvironment(overrideAuthored));
        const auto overrideSnapshot = resolveWorldEnvironment(&overrideDocument, &world);
        WORLD_CHECK(overrideSnapshot.elapsedSeconds == 0.0);
        WORLD_CHECK(overrideSnapshot.evaluatedAstronomy.julianDateUTC == overrideAuthored.time.julianDateUTC);

        edited.time = {.julianDateUTC = kMaximumJulianDateUTC - 0.01, .timeScale = 1e9, .paused = false};
        WORLD_CHECK(world.setWorldEnvironment(edited));
        world.advanceWorldEnvironment(100.0);
        WORLD_CHECK(world.environmentJulianDateUTC() == kMaximumJulianDateUTC);
        edited.time.timeScale = -1e9;
        WORLD_CHECK(world.setWorldEnvironment(edited));
        WORLD_CHECK(world.environmentJulianDateUTC() == kMaximumJulianDateUTC);
        world.advanceWorldEnvironment(1e5);
        WORLD_CHECK(world.environmentJulianDateUTC() == kMinimumJulianDateUTC);
        WORLD_CHECK(document.worldEnvironment() == authored && !document.dirty());
        return RHITestResult::pass("Cached clock evaluation, reverse/zero playback, JD editing, independent wind seek/revisions and clean authoring");
    }
};
METALLIC_REGISTER_RHI_TEST(DynamicWorldRuntimeStateTest);

class DynamicWorldSceneReloadClockTest final : public RHITest {
public:
    DynamicWorldSceneReloadClockTest() { name = "dynamic_world_scene_reload_clock"; type = RHITestType::Resource; }
    std::optional<bench::Metadata> metadata() const override
    {
        return worldContractMetadata("environment.runtime.scene_reload.authoring");
    }
    RHITestResult run(RHITestContext& context) override { return check(context.outputDirectory); }
    RHITestResult runCpu(bench::Evidence& evidence) override { return check(evidence.root()); }
    RHITestResult check(const std::filesystem::path& outputDirectory)
    {
        using namespace environment;
        using namespace render;
        const auto fixtureDirectory = outputDirectory / "SceneReloadClock";
        std::filesystem::create_directories(fixtureDirectory);
        const auto source = fixtureDirectory / "Clock.gltf";
        const auto sidecar = scene::SceneDocument::sidecarPathForSource(source);
        std::error_code removeError;
        std::filesystem::remove(sidecar, removeError);
        {
            std::ofstream output(source, std::ios::binary | std::ios::trunc);
            output << R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"name":"ClockFixture"}]})";
            WORLD_CHECK(output.good());
        }
        scene::SceneDocument document;
        WORLD_CHECK(document.load(source));
        WorldEnvironment authored;
        authored.source = EnvironmentSource::PhysicalAtmosphere;
        authored.sun.enabled = authored.moon.enabled = true;
        authored.time = {.julianDateUTC = 2460483.0, .timeScale = 60.0, .paused = false};
        authored.astronomy.mode = AstronomyMode::Astronomical;
        authored.astronomy.latitudeDegrees = 35.0;
        authored.weather.cloudEnabled = true;
        authored.weather.cloudCoverage = 0.6f;
        authored.weather.windSpeed = 10.0f;
        WORLD_CHECK(document.setWorldEnvironment(authored));
        std::string message;
        WORLD_CHECK(document.save(message) && !document.dirty());
        RenderWorld world;
        world.setScene(&document);
        (void)world.consumeChanges();
        world.advanceWorldEnvironment(9.0);
        const auto running = world.environmentSnapshot();
        const auto identity = document.resourceIdentity();
        const auto graphLifetime = document.sceneGraph().lifetimeRevision();

        auto edited = authored;
        edited.weather.humidity = 0.4f;
        WORLD_CHECK(document.setWorldEnvironment(edited) && document.dirty());
        WORLD_CHECK(world.setWorldEnvironment(document.worldEnvironment()));
        world.notifySceneChanged(RenderChangeBits::Lighting | RenderChangeBits::InvalidateTemporalHistory);
        world.setScene(&document); // Per-frame replay of the same authored object.
        WORLD_CHECK(document.resourceIdentity() == identity);
        WORLD_CHECK(document.sceneGraph().lifetimeRevision() == graphLifetime);
        WORLD_CHECK(world.environmentElapsedSeconds() == running.elapsedSeconds);
        WORLD_CHECK(world.environmentJulianDateUTC() == running.evaluatedAstronomy.julianDateUTC);
        WORLD_CHECK(world.worldEnvironment() == edited);

        // Revert replaces the actual loaded document at the same address, even
        // though its persisted Julian date is identical to the current authoring.
        WORLD_CHECK(document.revert(message));
        WORLD_CHECK(document.resourceIdentity() != identity);
        WORLD_CHECK(document.sceneGraph().lifetimeRevision() != graphLifetime);
        WORLD_CHECK(document.worldEnvironment() == authored && !document.dirty());
        (void)world.setWorldEnvironment(document.worldEnvironment());
        const auto beforeReplacementRevision = world.sceneContentRevision();
        world.notifySceneChanged();
        WORLD_CHECK(world.scene() == &document);
        WORLD_CHECK(world.environmentElapsedSeconds() == 0.0);
        WORLD_CHECK(world.environmentJulianDateUTC() == authored.time.julianDateUTC);
        WORLD_CHECK(world.environmentSnapshot().celestial == document.environmentSnapshot().celestial);
        WORLD_CHECK(world.environmentSnapshot().elapsedSeconds == 0.0);
        WORLD_CHECK(world.sceneContentRevision() == beforeReplacementRevision + 1u);
        const auto replacedChanges = world.consumeChanges();
        WORLD_CHECK(hasRenderChange(replacedChanges, RenderChangeBits::Geometry));
        WORLD_CHECK(hasRenderChange(replacedChanges, RenderChangeBits::Material));
        WORLD_CHECK(hasRenderChange(replacedChanges, RenderChangeBits::InvalidateTemporalHistory));

        world.advanceWorldEnvironment(13.0);
        const auto beforeReloadIdentity = document.resourceIdentity();
        WORLD_CHECK(document.load(sidecar));
        WORLD_CHECK(document.resourceIdentity() != beforeReloadIdentity);
        world.setScene(&document); // Editor load completion uses this path.
        WORLD_CHECK(world.environmentElapsedSeconds() == 0.0);
        WORLD_CHECK(world.environmentJulianDateUTC() == authored.time.julianDateUTC);
        WORLD_CHECK(!document.dirty());
        world.advanceWorldEnvironment(3.0);
        const auto beforeFailure = world.environmentSnapshot();
        const auto beforeFailureIdentity = document.resourceIdentity();
        WORLD_CHECK(!document.load(fixtureDirectory / "Missing.gltf"));
        WORLD_CHECK(document.resourceIdentity() == beforeFailureIdentity);
        world.notifySceneChanged(RenderChangeBits::Lighting);
        world.setScene(&document);
        WORLD_CHECK(world.environmentElapsedSeconds() == beforeFailure.elapsedSeconds);
        WORLD_CHECK(world.environmentJulianDateUTC() == beforeFailure.evaluatedAstronomy.julianDateUTC);
        WORLD_CHECK(document.worldEnvironment() == authored && !document.dirty());

        document.clear();
        world.notifySceneChanged();
        WORLD_CHECK(world.environmentElapsedSeconds() == 0.0);
        WORLD_CHECK(world.environmentJulianDateUTC() == WorldEnvironment{}.time.julianDateUTC);
        WORLD_CHECK(world.worldEnvironment() == document.worldEnvironment());
        return RHITestResult::pass("Actual same-address document revert/reload/clear restart authored time and wind; edits and failed loads retain playback");
    }
};
METALLIC_REGISTER_RHI_TEST(DynamicWorldSceneReloadClockTest);

class DynamicWorldCelestialShadowBudgetTest final : public RHITest {
public:
    DynamicWorldCelestialShadowBudgetTest() { name = "dynamic_world_celestial_shadow_budget"; type = RHITestType::Resource; }
    std::optional<bench::Metadata> metadata() const override
    {
        return worldContractMetadata("environment.celestial.shadow_budget.irradiance");
    }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        using namespace environment;
        using namespace render;
        WorldEnvironment authored;
        authored.source = EnvironmentSource::PhysicalAtmosphere;
        authored.sun.enabled = authored.moon.enabled = true;
        authored.sun.illuminance = 1.0f;
        authored.moon.illuminance = 1e9f; // Physical importance follows TOA, independently of legacy lux.
        authored.moon.topOfAtmosphereIrradiance = authored.sun.topOfAtmosphereIrradiance * 0.25f;
        const auto daylight = authored.snapshot();
        const auto directBefore = buildCelestialLightRecords(daylight);
        const auto sunPlan = buildCelestialShadowPlan(daylight, {0.0, 0.0, 0.0});
        WORLD_CHECK(sunPlan.dominantIndex == 0u && sunPlan.activeMask == 3u);
        WORLD_CHECK(sunPlan.sampleCounts[0] == 48u && sunPlan.sampleCounts[1] == 12u);
        WORLD_CHECK(std::abs(sunPlan.importance[1] / sunPlan.importance[0] - 0.25f) < 1e-6f);
        WORLD_CHECK(sameIrradiance(directBefore, buildCelestialLightRecords(daylight)));
        WORLD_CHECK(daylight.celestial[0] == authored.sun && daylight.celestial[1] == authored.moon);

        authored.moon.topOfAtmosphereIrradiance = authored.sun.topOfAtmosphereIrradiance * 2.0f;
        const auto moonlight = authored.snapshot();
        const auto moonPlan = buildCelestialShadowPlan(moonlight, {0.0, 0.0, 0.0});
        WORLD_CHECK(moonPlan.dominantIndex == 1u && moonPlan.activeMask == 3u);
        WORLD_CHECK(moonPlan.sampleCounts[1] == 48u && moonPlan.sampleCounts[0] == 12u);
        authored.sun.direction = float3(0.0f, 1.0f, 0.0f);
        const auto sunBelow = authored.snapshot();
        const auto sunBelowRecords = buildCelestialLightRecords(sunBelow);
        const auto nightPlan = buildCelestialShadowPlan(sunBelow, {0.0, 0.0, 0.0});
        WORLD_CHECK(nightPlan.dominantIndex == 1u && nightPlan.activeMask == 2u);
        WORLD_CHECK(nightPlan.sampleCounts[0] == 0u && nightPlan.sampleCounts[1] == 48u);
        WORLD_CHECK(nightPlan.importance[0] == 0.0f);
        WORLD_CHECK(sunBelowRecords[0].irradiance[0] > 0.0f);
        WORLD_CHECK(sameIrradiance(sunBelowRecords, buildCelestialLightRecords(sunBelow)));
        authored.moon.direction = float3(0.0f, 1.0f, 0.0f);
        const auto bothBelow = buildCelestialShadowPlan(authored.snapshot(), {0.0, 0.0, 0.0});
        WORLD_CHECK(bothBelow.dominantIndex == UINT32_MAX && bothBelow.activeMask == 0u);
        WORLD_CHECK(bothBelow.sampleCounts[0] == 0u && bothBelow.sampleCounts[1] == 0u);

        authored.sun.direction = float3(0.0f, -1.0f, 0.0f);
        authored.moon.direction = authored.sun.direction;
        authored.moon.topOfAtmosphereIrradiance = authored.sun.topOfAtmosphereIrradiance * 1e-5f;
        const auto faint = authored.snapshot();
        const auto faintRecords = buildCelestialLightRecords(faint);
        const auto faintPlan = buildCelestialShadowPlan(faint, {0.0, 0.0, 0.0});
        WORLD_CHECK(faintPlan.activeMask == 1u && faintPlan.sampleCounts[1] == 0u);
        WORLD_CHECK(faintRecords[1].irradiance[0] > 0.0f && (faintRecords[1].flags & kGPUCelestialLightEnabled) != 0);
        WORLD_CHECK(sameIrradiance(faintRecords, buildCelestialLightRecords(faint)));
        authored.moon.topOfAtmosphereIrradiance = authored.sun.topOfAtmosphereIrradiance;
        WORLD_CHECK(buildCelestialShadowPlan(authored.snapshot(), {0.0, 0.0, 0.0}).dominantIndex == 0u);

        // The budget uses spherical local up at the observer, rather than fixed world Y.
        authored.sun.direction = float3(-1.0f, 0.0f, 0.0f);
        const auto curved = authored.snapshot();
        const auto curvedPlan = buildCelestialShadowPlan(curved, {6360000.0, -6360000.0, 0.0});
        WORLD_CHECK(curvedPlan.dominantIndex == 0u && curvedPlan.activeMask == 1u);
        WORLD_CHECK(curvedPlan.importance[1] == 0.0f);
        WORLD_CHECK(buildCelestialShadowPlan(curved, curved.atmosphere.planetCenter).activeMask == 0u);

        authored.source = EnvironmentSource::HDRI;
        authored.sun.direction = float3(0.0f, -1.0f, 0.0f);
        authored.moon.direction = authored.sun.direction;
        authored.sun.illuminance = 1000.0f;
        authored.moon.illuminance = 100.0f;
        const auto hdriPlan = buildCelestialShadowPlan(authored.snapshot(), {0.0, 0.0, 0.0});
        WORLD_CHECK(hdriPlan.dominantIndex == 0u && hdriPlan.importance[0] == 1000.0f && hdriPlan.importance[1] == 100.0f);
        return RHITestResult::pass("Sun/Moon 48/12 budgets, threshold and horizon exclusion retain both direct irradiances");
    }
};
METALLIC_REGISTER_RHI_TEST(DynamicWorldCelestialShadowBudgetTest);

class DynamicWorldScreenSpaceShadowSelectionTest final : public RHITest {
public:
    DynamicWorldScreenSpaceShadowSelectionTest() { name = "dynamic_world_screen_space_shadow_selection"; type = RHITestType::Resource; }
    std::optional<bench::Metadata> metadata() const override
    {
        return worldContractMetadata("environment.celestial.screen_space_selection");
    }
    RHITestResult run(RHITestContext&) override { return check(); }
    RHITestResult runCpu(bench::Evidence&) override { return check(); }
    RHITestResult check()
    {
        using namespace environment;
        using namespace render;
        WorldEnvironment authored;
        authored.source = EnvironmentSource::PhysicalAtmosphere;
        authored.sun.enabled = authored.moon.enabled = true;
        authored.moon.topOfAtmosphereIrradiance = authored.sun.topOfAtmosphereIrradiance * 2.0f;
        scene::LightingSettings localLighting;
        localLighting.lights.resize(3);
        localLighting.lights[0].enabled = false;
        auto lights = buildScreenSpaceShadowLightRecords(nullptr, localLighting, authored.snapshot());
        WORLD_CHECK(lights.size() == 5u && lights[0].isCelestial && lights[1].isCelestial);
        WORLD_CHECK(lights[0].sourceIndex == 0u && lights[1].sourceIndex == 1u);
        WORLD_CHECK(lights[2].sourceIndex == 0u && !lights[2].enabled && lights[3].enabled);
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, -1) == 1u);
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, 0) == 0u);
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, 1) == 1u);
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, 3) == 3u);
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, 2) == 1u);
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, 1412) == 1u);

        // A bright source near the horizon loses automatic priority to an overhead source.
        authored.sun.direction = float3(std::sqrt(1.0f - 0.01f * 0.01f), -0.01f, 0.0f);
        authored.moon.topOfAtmosphereIrradiance = authored.sun.topOfAtmosphereIrradiance * 0.1f;
        lights = buildScreenSpaceShadowLightRecords(nullptr, localLighting, authored.snapshot());
        WORLD_CHECK(lights[1].celestial.shadowImportance > lights[0].celestial.shadowImportance);
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, -1) == 1u);
        authored.moon.direction = float3(0.0f, 1.0f, 0.0f);
        lights = buildScreenSpaceShadowLightRecords(nullptr, localLighting, authored.snapshot());
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, -1) == 0u);
        authored.sun.direction = float3(0.0f, 1.0f, 0.0f);
        lights = buildScreenSpaceShadowLightRecords(nullptr, localLighting, authored.snapshot());
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, -1) == 3u);
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, 0) == 0u); // Explicit enabled requests retain their stable slot.
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, 4) == 4u);
        for (auto& light : localLighting.lights) { light.enabled = false; }
        lights = buildScreenSpaceShadowLightRecords(nullptr, localLighting, authored.snapshot());
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, -1) == UINT32_MAX);
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, 3) == UINT32_MAX);
        authored.sun.enabled = authored.moon.enabled = false;
        lights = buildScreenSpaceShadowLightRecords(nullptr, localLighting, authored.snapshot());
        WORLD_CHECK(selectScreenSpaceShadowLight(lights, -1) == UINT32_MAX);
        WORLD_CHECK(selectScreenSpaceShadowLight({}, -1) == UINT32_MAX);
        return RHITestResult::pass("Automatic elevation-weighted importance, explicit enabled slots and first enabled local fallback");
    }
};
METALLIC_REGISTER_RHI_TEST(DynamicWorldScreenSpaceShadowSelectionTest);

#undef WORLD_CHECK
} // namespace
} // namespace metallic::tests
