#include "RHITest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace metallic::tests {
namespace {

constexpr uint32_t kSeamWidth = 512;
constexpr uint32_t kSeamHeight = 256;
constexpr uint32_t kSeamFrames = 4;

std::vector<float> decodeSeamHDR(const render::RenderGraphPreviewRenderer& preview)
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

bool seamExecutionMatches(const render::RenderGraphExecutionStats& stats, bool pathTraced)
{
    bool sawPT = false, sawDeferred = false, sawVBuffer = false, sawSlider = false;
    for (const auto& node : stats.nodes) {
        sawPT |= node.name == "Reference";
        sawDeferred |= node.name == "Deferred";
        sawVBuffer |= node.name == "VBuffer";
        sawSlider |= node.name == "Slider";
    }
    return sawPT == pathTraced && sawDeferred == !pathTraced && sawVBuffer == !pathTraced && !sawSlider;
}

struct SeamStatistics {
    double relativeCenterPair = 0.0;
    double relativeNeighborDifference = 0.0;
    double relativeStepExcess = 0.0;
    uint32_t rows = 0;
};

SeamStatistics measureSeam(const std::vector<float>& rgba)
{
    constexpr uint32_t middle = kSeamWidth / 2;
    double centerDifference = 0.0, neighborDifference = 0.0, magnitude = 0.0;
    SeamStatistics result;
    // The upward camera keeps these central sky bands above the LookDev mesh.
    // Omit the central third containing the finite Moon disk and its AA edge.
    for (uint32_t y = 16; y < kSeamHeight - 16; ++y) {
        if (y >= 80 && y < 176) { continue; }
        for (uint32_t channel = 0; channel < 3; ++channel) {
            const auto value = [&](uint32_t x) { return double(rgba[(size_t(y) * kSeamWidth + x) * 4u + channel]); };
            const double left = value(middle - 1), right = value(middle);
            centerDifference += std::abs(right - left);
            magnitude += (std::abs(left) + std::abs(right)) * 0.5;
            for (uint32_t offset = 2; offset <= 9; ++offset) {
                neighborDifference += (std::abs(value(middle - offset) - value(middle - offset - 1)) +
                    std::abs(value(middle + offset) - value(middle + offset - 1))) / 16.0;
            }
        }
        ++result.rows;
    }
    result.relativeCenterPair = centerDifference / std::max(magnitude, 1e-12);
    result.relativeNeighborDifference = neighborDifference / std::max(magnitude, 1e-12);
    result.relativeStepExcess = std::max(centerDifference - 4.0 * neighborDifference, 0.0) / std::max(magnitude, 1e-12);
    return result;
}

class AtmosphereSeamRenderingTest final : public RHITest {
public:
    AtmosphereSeamRenderingTest() { name = "physical_moon_longitude_seam_rendering"; type = RHITestType::Rendering; }

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
        auto world = document.worldEnvironment();
        world.source = environment::EnvironmentSource::PhysicalAtmosphere;
        world.astronomy.mode = environment::AstronomyMode::Manual;
        world.time.paused = true;
        world.sun.enabled = false;
        world.moon.enabled = true;
        world.moon.direction = float3(0.840f, -1.0f, 0.0f);
        world.moon.angularRadius = 0.266f * 0.017453292519943295f;
        world.moon.topOfAtmosphereIrradiance = environment::WorldEnvironment{}.moon.topOfAtmosphereIrradiance;
        world.weather = {};
        std::array<environment::WorldEnvironment, 2> worlds{world, world};
        worlds[1].weather = document.worldEnvironment().weather;
        worlds[1].weather.cloudEnabled = true;
        worlds[1].weather.cloudCoverage = 0.45f;
        worlds[1].weather.humidity = 0.25f;
        worlds[1].weather.noiseSeed = 17;
        worlds[1].weather.windSpeed = 0.0f;
        const RenderGraphProperties camera{{"eye", {-10.0, 2.0, 0.0}}, {"center", {-10.643, 2.766, 0.0}},
            {"up", {0.0, 1.0, 0.0}}, {"fovDegrees", 50.0}, {"projection", "perspective"},
            {"znear", 0.05}, {"zfar", 100.0}, {"reversedZ", true}};
        auto results = RenderGraphProperties::array();
        bool clearSeamFailed = false;
        for (bool pt : {true, false}) {
            const std::string label = pt ? "PathTrace" : "Deferred";
            RenderGraph graph;
            if (!makeLookDevRenderGraph(sample.graph,
                    pt ? LookDevRenderPath::PathTraceOnly : LookDevRenderPath::DeferredOnly, graph, log)) {
                return RHITestResult::fail(log);
            }
            auto* node = graph.findNode(pt ? "Reference" : "VBuffer");
            if (!node) { return RHITestResult::fail("Missing camera node in isolated rendering branch"); }
            auto properties = node->properties;
            properties["camera"] = camera;
            properties["cameraSyncGroup"] = "MoonLongitudeSeam";
            if (pt) {
                properties["cacheMode"] = "off";
                properties["accumulate"] = true;
            }
            if (!graph.setNodeProperties(node->id, std::move(properties))) { return RHITestResult::fail("Set seam camera fixture"); }
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&document);
            preview.setEnvironment(document.environment());
            auto lighting = document.lighting();
            lighting.autoExposure.enabled = false;
            lighting.exposureEV100 = -3.0f;
            preview.setLighting(lighting);
            if (!preview.subsystemHost()->configure<EnvironmentLightingSubsystem>(
                    {.initialDecodeTimeoutMilliseconds = 10000}, log) ||
                !preview.initialize(context.enableValidation, true, false)) {
                return RHITestResult::fail("Moon seam preview setup: " + log + preview.lastLog());
            }
            for (uint32_t caseIndex = 0; caseIndex < worlds.size(); ++caseIndex) {
                const std::string state = caseIndex == 0 ? "ClearMoon" : "FrozenCloudMoon";
                if (!preview.setWorldEnvironment(worlds[caseIndex])) { return RHITestResult::fail("Set Moon seam world"); }
                for (uint32_t frame = 0; frame < kSeamFrames; ++frame) {
                    preview.setRawReadbackEnabled(frame + 1 == kSeamFrames);
                    if (!preview.render(graph, kSeamWidth, kSeamHeight,
                            pt ? "Reference.color" : "Deferred.color", frame + 1 == kSeamFrames)) {
                        return RHITestResult::fail(label + "/" + state + ": " + preview.lastLog());
                    }
                    if (!seamExecutionMatches(preview.executionStats(), pt)) {
                        return RHITestResult::fail(label + " executed an unexpected branch");
                    }
                }
                const auto rgba = decodeSeamHDR(preview);
                if (rgba.size() != size_t(kSeamWidth) * kSeamHeight * 4u ||
                    !std::all_of(rgba.begin(), rgba.end(), [](float value) { return std::isfinite(value); })) {
                    return RHITestResult::fail(label + "/" + state + ": invalid HDR readback");
                }
                const auto measurement = measureSeam(rgba);
                const auto prefix = context.outputDirectory / (state + "_" + label);
                std::ofstream raw(prefix.string() + ".rgba32f", std::ios::binary);
                raw.write(reinterpret_cast<const char*>(rgba.data()), std::streamsize(rgba.size() * sizeof(float)));
                if (!raw) { return RHITestResult::fail("Moon seam HDR evidence write failed"); }
                preview.setRawReadbackEnabled(false);
                if (!preview.render(graph, kSeamWidth, kSeamHeight, "FinalBlit.color") ||
                    !saveRgba8Png(prefix.string() + ".png", reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                        kSeamWidth, kSeamHeight, log)) {
                    return RHITestResult::fail(label + "/" + state + ": " + preview.lastLog() + log);
                }
                results.push_back({{"case", state}, {"path", label}, {"relativeCenterPair", measurement.relativeCenterPair},
                    {"relativeNeighborDifference", measurement.relativeNeighborDifference},
                    {"relativeStepExcess", measurement.relativeStepExcess}, {"skyRows", measurement.rows}});
                // In this cloud-free, source-centred view the two sides are
                // mirror symmetric. Clouds may legitimately have a gradient,
                // so their image is evidence rather than this symmetry oracle.
                clearSeamFailed |= caseIndex == 0 && measurement.relativeCenterPair > 0.005;
            }
            if (!saveRenderGraphToFile(graph, context.outputDirectory / ("MoonSeam_" + label + ".metallic_graph.json"), log)) {
                return RHITestResult::fail(log);
            }
        }
        RenderGraphProperties report{{"width", kSeamWidth}, {"height", kSeamHeight}, {"frames", kSeamFrames},
            {"cacheMode", "off"}, {"exposureEV100", -3.0}, {"camera", camera}, {"clearCenterPairLimit", 0.005},
            {"skyBands", {{16, 80}, {176, 240}}}, {"cases", results}};
        std::ofstream summary(context.outputDirectory / "MoonLongitudeSeamRendering.json");
        summary << report.dump(2) << '\n'; summary.close();
        if (!summary) { return RHITestResult::fail("Moon seam report write failed"); }
        if (clearSeamFailed) {
            return RHITestResult::fail("Cloud-free Moon longitude seam exceeded 0.5% adjacent-pair difference; see MoonLongitudeSeamRendering.json");
        }
        return RHITestResult::pass("Moon-only longitude seam: actual PT/Deferred clear-sky symmetry below 0.5%, frozen-cloud HDR/PNG evidence");
    }
};
METALLIC_REGISTER_RHI_TEST(AtmosphereSeamRenderingTest);

} // namespace
} // namespace metallic::tests
