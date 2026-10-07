#include "RHITest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "stb/stb_image_write.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>

namespace metallic::tests {
namespace {

constexpr uint32_t kDynamicWorldSize = 128;
constexpr uint32_t kDynamicWorldFrames = 16;
constexpr uint32_t kBackgroundCornerSize = 8;

struct DynamicWorldCase {
    std::string name;
    environment::WorldEnvironment world;
    double altitudeMetres = 0.0;
};

struct DynamicWorldImage {
    std::vector<float> rgba;
    std::array<double, 3> meanRGB{};
    std::array<double, 3> backgroundMeanRGB{};
};

std::vector<float> decodeDynamicWorldHDR(const render::RenderGraphPreviewRenderer& preview)
{
    const auto& bytes = preview.readbackBytes();
    std::vector<float> values;
    if (preview.readbackFormat() == render::Format::RGBA32Sfloat && bytes.size() % sizeof(float) == 0) {
        values.resize(bytes.size() / sizeof(float));
        std::memcpy(values.data(), bytes.data(), bytes.size());
    } else if (preview.readbackFormat() == render::Format::RGBA16Sfloat && bytes.size() % sizeof(uint16_t) == 0) {
        values.reserve(bytes.size() / sizeof(uint16_t));
        for (size_t offset = 0; offset < bytes.size(); offset += sizeof(uint16_t)) {
            uint16_t half;
            std::memcpy(&half, bytes.data() + offset, sizeof(half));
            const uint32_t exponent = (half >> 10) & 31u;
            const uint32_t mantissa = half & 1023u;
            const float magnitude = exponent == 0 ? std::ldexp(float(mantissa), -24) :
                exponent == 31 ? (mantissa == 0 ? std::numeric_limits<float>::infinity() :
                    std::numeric_limits<float>::quiet_NaN()) :
                std::ldexp(float(1024u + mantissa), int(exponent) - 25);
            values.push_back((half & 0x8000u) != 0 ? -magnitude : magnitude);
        }
    }
    return values;
}

bool dynamicWorldImageStatistics(DynamicWorldImage& image, std::string& message)
{
    if (image.rgba.size() != size_t(kDynamicWorldSize) * kDynamicWorldSize * 4u) {
        message = "Expected RGBA32F/16F HDR readback at 128 x 128";
        return false;
    }
    uint32_t backgroundCount = 0;
    for (uint32_t y = 0; y < kDynamicWorldSize; ++y) {
        for (uint32_t x = 0; x < kDynamicWorldSize; ++x) {
            const size_t base = (size_t(y) * kDynamicWorldSize + x) * 4u;
            // The lower corners contain the shaderball's floor. Use the upper
            // corners as a reproducible sky ROI, away from the solar disk.
            const bool background = y < kBackgroundCornerSize &&
                (x < kBackgroundCornerSize || x >= kDynamicWorldSize - kBackgroundCornerSize);
            for (uint32_t channel = 0; channel < 4; ++channel) {
                const float value = image.rgba[base + channel];
                if (!std::isfinite(value) || (channel < 3 && value < -0.0001f)) {
                    message = "Nonfinite or negative physical HDR at pixel " + std::to_string(base / 4u) +
                        ", channel " + std::to_string(channel) + ", value " + std::to_string(value);
                    return false;
                }
                if (channel < 3) {
                    image.meanRGB[channel] += value;
                    if (background) { image.backgroundMeanRGB[channel] += value; }
                }
            }
            backgroundCount += background ? 1u : 0u;
        }
    }
    for (uint32_t channel = 0; channel < 3; ++channel) {
        image.meanRGB[channel] /= double(kDynamicWorldSize) * kDynamicWorldSize;
        image.backgroundMeanRGB[channel] /= backgroundCount;
    }
    return true;
}

double dynamicWorldRelativeRMSE(const DynamicWorldImage& reference, const DynamicWorldImage& image)
{
    double differenceSquared = 0.0;
    double referenceSquared = 0.0;
    for (size_t index = 0; index < reference.rgba.size(); ++index) {
        if ((index & 3u) == 3u) { continue; }
        const double value = reference.rgba[index];
        const double difference = value - image.rgba[index];
        referenceSquared += value * value;
        differenceSquared += difference * difference;
    }
    return std::sqrt(differenceSquared / std::max(referenceSquared, 1e-12));
}

double dynamicWorldBackgroundError(const DynamicWorldImage& reference, const DynamicWorldImage& image)
{
    double difference = 0.0;
    double magnitude = 0.0;
    for (uint32_t channel = 0; channel < 3; ++channel) {
        difference += std::abs(reference.backgroundMeanRGB[channel] - image.backgroundMeanRGB[channel]);
        magnitude += std::abs(reference.backgroundMeanRGB[channel]);
    }
    return difference / std::max(magnitude, 1e-7);
}

double dynamicWorldBackgroundBrightness(const DynamicWorldImage& image)
{
    return image.backgroundMeanRGB[0] + image.backgroundMeanRGB[1] + image.backgroundMeanRGB[2];
}

bool dynamicWorldExecutionMatches(const render::RenderGraphExecutionStats& stats, bool pathTraced)
{
    bool sawPT = false;
    bool sawDeferred = false;
    bool sawVBuffer = false;
    bool sawSlider = false;
    for (const auto& node : stats.nodes) {
        sawPT |= node.name == "Reference";
        sawDeferred |= node.name == "Deferred";
        sawVBuffer |= node.name == "VBuffer";
        sawSlider |= node.name == "Slider";
    }
    return sawPT == pathTraced && sawDeferred == !pathTraced && sawVBuffer == !pathTraced && !sawSlider;
}


std::vector<DynamicWorldCase> dynamicWorldCases(environment::WorldEnvironment base)
{
    base.source = environment::EnvironmentSource::PhysicalAtmosphere;
    base.astronomy.mode = environment::AstronomyMode::Astronomical;
    base.astronomy.latitudeDegrees = 35.0;
    base.astronomy.longitudeDegrees = 0.0;
    base.time.julianDateUTC = 2460483.0; // 2024-06-21 12:00 UTC.
    base.time.paused = true;
    base.sun.enabled = true;
    base.moon.enabled = true;
    base.weather = {};
    std::vector<DynamicWorldCase> cases{{"ClearNoon", base}};
    auto broken = base;
    broken.weather.cloudEnabled = true;
    broken.weather.cloudCoverage = 0.65f;
    broken.weather.noiseSeed = 17;
    cases.push_back({"BrokenNoon", broken});
    auto overcast = broken;
    overcast.weather.cloudCoverage = 1.0f;
    cases.push_back({"OvercastNoon", overcast});
    auto humid = broken;
    humid.weather.aerosolDensity = 3.0f;
    humid.weather.humidity = 0.8f;
    humid.weather.precipitation = 0.5f;
    cases.push_back({"HumidRainMedium", humid});
    auto twilight = base;
    twilight.time.julianDateUTC += 0.3;
    cases.push_back({"Twilight", twilight});
    auto night = base;
    night.time.julianDateUTC += 0.5;
    cases.push_back({"FullMoonNight", night});
    return cases;
}

class DynamicWorldRenderingTest final : public RHITest {
public:
    DynamicWorldRenderingTest() { name = "dynamic_world_raster_pt_cases"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        RenderSampleLoadResult sample;
        std::string log;
        if (!loadBuiltInRenderSample("dynamic-world-lookdev", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RHITestResult::fail(document.lastLoadResult().error);
        }
        const auto cases = dynamicWorldCases(document.worldEnvironment());
        std::array<std::vector<DynamicWorldImage>, 2> images;
        auto executedGraphs = RenderGraphProperties::object();
        for (uint32_t pathIndex = 0; pathIndex < 2; ++pathIndex) {
            const bool pt = pathIndex == 0;
            const std::string label = pt ? "PathTrace" : "Deferred";
            RenderGraph graph;
            if (!makeLookDevRenderGraph(sample.graph,
                    pt ? LookDevRenderPath::PathTraceOnly : LookDevRenderPath::DeferredOnly, graph, log)) {
                return RHITestResult::fail(log);
            }
            if (pt) {
                auto* node = graph.findNode("Reference");
                auto properties = node->properties;
                properties["cacheMode"] = "off";
                properties["accumulate"] = true;
                if (!graph.setNodeProperties(node->id, std::move(properties))) { return RHITestResult::fail("Set PT fixture"); }
            }
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&document);
            preview.setEnvironment(document.environment());
            if (!preview.subsystemHost()->configure<EnvironmentLightingSubsystem>(
                    {.initialDecodeTimeoutMilliseconds = 10000}, log) ||
                !preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail(log); }
            for (size_t caseIndex = 0; caseIndex < cases.size(); ++caseIndex) {
                const auto& testCase = cases[caseIndex];
                auto lighting = document.lighting();
                lighting.autoExposure.enabled = false;
                // Fixed exposure per case, shared by both rendering paths.
                lighting.exposureEV100 = caseIndex == 5 ? -3.0f : caseIndex == 4 ? 8.0f : 14.0f;
                preview.setLighting(lighting);
                preview.setWorldEnvironment(testCase.world);
                for (uint32_t frame = 0; frame < kDynamicWorldFrames; ++frame) {
                    preview.setRawReadbackEnabled(frame + 1 == kDynamicWorldFrames);
                    if (!preview.render(graph, kDynamicWorldSize, kDynamicWorldSize,
                            pt ? "Reference.color" : "Deferred.color", frame + 1 == kDynamicWorldFrames)) {
                        return RHITestResult::fail(label + "/" + testCase.name + ": " + preview.lastLog());
                    }
                    if (!dynamicWorldExecutionMatches(preview.executionStats(), pt)) {
                        return RHITestResult::fail(label + " executed the wrong branch");
                    }
                }
                DynamicWorldImage image{.rgba = decodeDynamicWorldHDR(preview)};
                if (!dynamicWorldImageStatistics(image, log)) {
                    return RHITestResult::fail(label + "/" + testCase.name + ": " + log);
                }
                const auto prefix = context.outputDirectory / (testCase.name + "_" + label);
                std::ofstream raw(prefix.string() + ".rgba32f", std::ios::binary);
                raw.write(reinterpret_cast<const char*>(image.rgba.data()), image.rgba.size() * sizeof(float));
                if (!raw) { return RHITestResult::fail("HDR evidence write failed"); }
                images[pathIndex].push_back(std::move(image));
                preview.setRawReadbackEnabled(false);
                if (!preview.render(graph, kDynamicWorldSize, kDynamicWorldSize, "FinalBlit.color") ||
                    !saveRgba8Png(prefix.string() + ".png", reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                        kDynamicWorldSize, kDynamicWorldSize, log)) {
                    return RHITestResult::fail(label + "/" + testCase.name + ": " + preview.lastLog() + log);
                }
            }
            auto executed = RenderGraphProperties::array();
            for (const auto& node : preview.executionStats().nodes) { executed.push_back({{"name", node.name}, {"type", node.type}}); }
            executedGraphs[label] = std::move(executed);
            if (!saveRenderGraphToFile(graph, context.outputDirectory / (label + ".metallic_graph.json"), log)) {
                return RHITestResult::fail(log);
            }
        }
        auto results = RenderGraphProperties::array();
        double maximumSkyError = 0.0;
        for (size_t index = 0; index < cases.size(); ++index) {
            const double skyError = dynamicWorldBackgroundError(images[1][index], images[0][index]);
            maximumSkyError = std::max(maximumSkyError, skyError);
            results.push_back({{"case", cases[index].name}, {"julianDateUTC", cases[index].world.time.julianDateUTC},
                {"exposureEV100", index == 5 ? -3.0 : index == 4 ? 8.0 : 14.0},
                {"pathTraceMeanRGB", images[0][index].meanRGB}, {"deferredMeanRGB", images[1][index].meanRGB},
                {"pathTraceSkyRGB", images[0][index].backgroundMeanRGB}, {"deferredSkyRGB", images[1][index].backgroundMeanRGB},
                {"relativeRMSE", dynamicWorldRelativeRMSE(images[1][index], images[0][index])}, {"skyRelativeError", skyError}});
        }
        RenderGraphProperties report{{"width", kDynamicWorldSize}, {"height", kDynamicWorldSize},
            {"framesPerCaseAndPath", kDynamicWorldFrames}, {"cacheMode", "off"}, {"timePaused", true}, {"windSpeed", 0},
            {"executedPasses", executedGraphs}, {"cases", results}};
        std::ofstream summary(context.outputDirectory / "DynamicWorldRendering.json");
        summary << report.dump(2) << '\n'; summary.close();
        if (!summary) { return RHITestResult::fail("Dynamic report write failed"); }
        if (maximumSkyError > 0.1) { return RHITestResult::fail("Dynamic raster/PT sky error exceeded 10%; see DynamicWorldRendering.json"); }
        for (uint32_t path = 0; path < 2; ++path) {
            if (dynamicWorldRelativeRMSE(images[path][0], images[path][1]) < 0.001 ||
                dynamicWorldRelativeRMSE(images[path][1], images[path][2]) < 0.001 ||
                dynamicWorldRelativeRMSE(images[path][1], images[path][3]) < 0.001 ||
                dynamicWorldBackgroundBrightness(images[path][5]) >= dynamicWorldBackgroundBrightness(images[path][0])) {
                return RHITestResult::fail("Cloud/weather/day-night changes did not affect the rendered scene");
            }
        }
        return RHITestResult::pass("6 dynamic states x 16 frames x actual PT/Deferred; finite HDR, shared sky, cloud/weather/night response, HDR/PNG evidence");
    }
};
METALLIC_REGISTER_RHI_TEST(DynamicWorldRenderingTest);


class DynamicWorldPlaybackRenderingTest final : public RHITest {
public:
    DynamicWorldPlaybackRenderingTest() { name = "dynamic_world_playback_history"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        RenderSampleLoadResult sample;
        std::string log;
        if (!loadBuiltInRenderSample("dynamic-world-lookdev", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RHITestResult::fail(document.lastLoadResult().error);
        }
        const auto authoredDocument = document.worldEnvironment();
        auto report = RenderGraphProperties::object();
        for (bool pt : {true, false}) {
            RenderGraph graph;
            if (!makeLookDevRenderGraph(sample.graph, pt ? LookDevRenderPath::PathTraceOnly : LookDevRenderPath::DeferredOnly,
                    graph, log)) { return RHITestResult::fail(log); }
            if (pt) {
                auto* node = graph.findNode("Reference");
                auto properties = node->properties;
                properties["cacheMode"] = "off";
                properties["accumulate"] = true;
                if (!graph.setNodeProperties(node->id, std::move(properties))) { return RHITestResult::fail("Set playback fixture"); }
            }
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&document);
            preview.setEnvironment(document.environment());
            auto lighting = document.lighting(); lighting.autoExposure.enabled = false; lighting.exposureEV100 = 14.0f;
            preview.setLighting(lighting);
            if (!preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail(preview.lastLog()); }
            auto world = dynamicWorldCases(authoredDocument)[1].world;
            world.weather.cloudExtinctionPerKm = 2.0f;
            world.weather.windSpeed = 10.0f;
            preview.setWorldEnvironment(world);
            const auto capture = [&](DynamicWorldImage& image) {
                preview.setRawReadbackEnabled(true);
                if (!preview.render(graph, kDynamicWorldSize, kDynamicWorldSize, pt ? "Reference.color" : "Deferred.color", true)) {
                    log = preview.lastLog(); return false;
                }
                image.rgba = decodeDynamicWorldHDR(preview);
                return dynamicWorldImageStatistics(image, log) && dynamicWorldExecutionMatches(preview.executionStats(), pt);
            };
            DynamicWorldImage start, wind, night;
            if (!capture(start)) { return RHITestResult::fail(log); }
            auto* runtime = preview.subsystemHost()->world();
            if (!runtime) { return RHITestResult::fail("No playback runtime owner"); }
            const auto beforeWind = runtime->environmentSnapshot();
            runtime->advanceWorldEnvironment(600.0);
            if (!capture(wind)) { return RHITestResult::fail(log); }
            const auto afterWind = runtime->environmentSnapshot();
            if (afterWind.evaluatedAstronomy.julianDateUTC != beforeWind.evaluatedAstronomy.julianDateUTC ||
                afterWind.weatherRevision <= beforeWind.weatherRevision ||
                afterWind.lightingRevision <= beforeWind.lightingRevision || runtime->worldEnvironment() != world) {
                return RHITestResult::fail("Paused astronomy/wind playback snapshot or history revision failed");
            }
            world.time.paused = false; world.time.timeScale = 3600.0;
            runtime->setWorldEnvironment(world);
            runtime->advanceWorldEnvironment(12.0); // Twelve simulated hours, independent of rendering duration.
            world.time.paused = true;
            runtime->setWorldEnvironment(world);
            if (!capture(night)) { return RHITestResult::fail(log); }
            const double windChange = dynamicWorldRelativeRMSE(start, wind);
            const double nightSky = dynamicWorldBackgroundBrightness(night) /
                std::max(dynamicWorldBackgroundBrightness(wind), 1e-12);
            const std::string label = pt ? "PathTrace" : "Deferred";
            for (auto [state, image] : {std::pair{"Start", &start}, {"Wind600Seconds", &wind}, {"Night12Hours", &night}}) {
                std::ofstream raw(context.outputDirectory / ("Playback_" + label + "_" + state + ".rgba32f"), std::ios::binary);
                raw.write(reinterpret_cast<const char*>(image->rgba.data()), image->rgba.size() * sizeof(float));
                if (!raw) { return RHITestResult::fail("Playback evidence write failed"); }
            }
            report[label] = {{"windRelativeRMSE", windChange}, {"nightToWindSkyRatio", nightSky},
                {"runtimeJulianDateUTC", runtime->environmentJulianDateUTC()}, {"weatherRevision", afterWind.weatherRevision}};
            if (windChange < 0.005 || nightSky > 0.01 || document.dirty() || document.worldEnvironment() != authoredDocument) {
                return RHITestResult::fail(label + ": live cloud/clock changes did not invalidate history or modified authoring state");
            }
        }
        std::ofstream summary(context.outputDirectory / "DynamicWorldPlayback.json"); summary << report.dump(2) << '\n';
        if (!summary) { return RHITestResult::fail("Playback report write failed"); }
        return RHITestResult::pass("Actual PT/Deferred paused-clock wind and 12-hour astronomy changes produce fresh finite HDR without modifying scene authoring");
    }
};
METALLIC_REGISTER_RHI_TEST(DynamicWorldPlaybackRenderingTest);

} // namespace
} // namespace metallic::tests
