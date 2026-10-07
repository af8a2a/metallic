#include "RHITest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"
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

constexpr uint32_t kPhysicalEnvironmentSize = 128;
constexpr uint32_t kPhysicalEnvironmentFrames = 32;
constexpr uint32_t kBackgroundCornerSize = 8;

struct PhysicalEnvironmentCase {
    std::string name;
    environment::WorldEnvironment world;
    double altitudeMetres = 0.0;
};

struct PhysicalEnvironmentImage {
    std::vector<float> rgba;
    std::array<double, 3> meanRGB{};
    std::array<double, 3> backgroundMeanRGB{};
};

std::vector<float> decodePhysicalEnvironmentHDR(const render::RenderGraphPreviewRenderer& preview)
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

bool physicalEnvironmentImageStatistics(PhysicalEnvironmentImage& image, std::string& message)
{
    if (image.rgba.size() != size_t(kPhysicalEnvironmentSize) * kPhysicalEnvironmentSize * 4u) {
        message = "Expected RGBA32F/16F HDR readback at 128 x 128";
        return false;
    }
    uint32_t backgroundCount = 0;
    for (uint32_t y = 0; y < kPhysicalEnvironmentSize; ++y) {
        for (uint32_t x = 0; x < kPhysicalEnvironmentSize; ++x) {
            const size_t base = (size_t(y) * kPhysicalEnvironmentSize + x) * 4u;
            // The lower corners contain the shaderball's floor. Use the upper
            // corners as a reproducible sky ROI, away from the solar disk.
            const bool background = y < kBackgroundCornerSize &&
                (x < kBackgroundCornerSize || x >= kPhysicalEnvironmentSize - kBackgroundCornerSize);
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
        image.meanRGB[channel] /= double(kPhysicalEnvironmentSize) * kPhysicalEnvironmentSize;
        image.backgroundMeanRGB[channel] /= backgroundCount;
    }
    return true;
}

double physicalEnvironmentRelativeRMSE(const PhysicalEnvironmentImage& reference, const PhysicalEnvironmentImage& image)
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

double physicalEnvironmentBackgroundError(const PhysicalEnvironmentImage& reference, const PhysicalEnvironmentImage& image)
{
    double difference = 0.0;
    double magnitude = 0.0;
    for (uint32_t channel = 0; channel < 3; ++channel) {
        difference += std::abs(reference.backgroundMeanRGB[channel] - image.backgroundMeanRGB[channel]);
        magnitude += std::abs(reference.backgroundMeanRGB[channel]);
    }
    return difference / std::max(magnitude, 1e-7);
}

double physicalEnvironmentBackgroundBrightness(const PhysicalEnvironmentImage& image)
{
    return image.backgroundMeanRGB[0] + image.backgroundMeanRGB[1] + image.backgroundMeanRGB[2];
}

std::vector<PhysicalEnvironmentCase> physicalEnvironmentCases(environment::WorldEnvironment base)
{
    base.source = environment::EnvironmentSource::PhysicalAtmosphere;
    base.sun.enabled = true;
    base.sun.direction = float3(0.0f, -1.0f, 0.0f);
    base.moon.enabled = false;
    std::vector<PhysicalEnvironmentCase> cases{{"Noon", base}};
    auto sunset = base;
    constexpr float kDegreesToRadians = 0.017453292519943295f;
    sunset.sun.direction = float3(-std::cos(2.0f * kDegreesToRadians), -std::sin(2.0f * kDegreesToRadians), 0.0f);
    cases.push_back({"Sunset", sunset});
    auto twilight = base;
    twilight.sun.direction = float3(-std::cos(6.0f * kDegreesToRadians), std::sin(6.0f * kDegreesToRadians), 0.0f);
    cases.push_back({"Twilight", twilight});
    for (const auto& [name, altitude] : {
            std::pair{"Altitude2Km", 2000.0}, std::pair{"Altitude10Km", 10000.0},
            std::pair{"Altitude100Km", 100000.0}, std::pair{"Orbit200Km", 200000.0}}) {
        auto elevated = base;
        // Move the planet origin in double precision, keeping local camera and
        // geometry fixed. The entire LookDev scene sits at the stated altitude.
        elevated.atmosphere.planetCenter[1] -= altitude;
        cases.push_back({name, elevated, altitude});
    }
    auto rayleighOnly = base;
    rayleighOnly.atmosphere.mieScattering = float3(0.0f);
    rayleighOnly.atmosphere.mieExtinction = float3(0.0f);
    cases.push_back({"RayleighOnly", rayleighOnly});
    auto mieHeavy = base;
    mieHeavy.atmosphere.mieScattering *= 10.0f;
    mieHeavy.atmosphere.mieExtinction *= 10.0f;
    cases.push_back({"MieHeavy", mieHeavy});
    auto ozoneOff = base;
    ozoneOff.atmosphere.ozoneAbsorption = float3(0.0f);
    cases.push_back({"OzoneOff", ozoneOff});
    auto blackGround = base;
    blackGround.atmosphere.groundAlbedo = float3(0.0f);
    cases.push_back({"GroundBlack", blackGround});
    auto whiteGround = base;
    whiteGround.atmosphere.groundAlbedo = float3(1.0f);
    cases.push_back({"GroundWhite", whiteGround});
    return cases;
}

bool physicalEnvironmentExecutionMatches(const render::RenderGraphExecutionStats& stats, bool pathTraced)
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

class PhysicalEnvironmentRenderingTest final : public RHITest {
public:
    PhysicalEnvironmentRenderingTest() { name = "physical_environment_raster_pt_cases"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        RenderSampleLoadResult sample;
        std::string log;
        if (!loadBuiltInRenderSample("physical-atmosphere-lookdev", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RHITestResult::fail(document.documentWarning() + document.lastLoadResult().error);
        }
        const auto cases = physicalEnvironmentCases(document.worldEnvironment());
        auto lighting = document.lighting();
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 14.0f;
        std::array<std::vector<PhysicalEnvironmentImage>, 2> images;
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
                if (!graph.setNodeProperties(node->id, std::move(properties))) {
                    return RHITestResult::fail("Failed to disable PT radiance caches");
                }
            }
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&document);
            preview.setEnvironment(document.environment());
            preview.setLighting(lighting);
            if (!preview.subsystemHost()->configure<EnvironmentLightingSubsystem>(
                    {.initialDecodeTimeoutMilliseconds = 10000}, log) ||
                !preview.initialize(context.enableValidation, true, false)) {
                return RHITestResult::fail("Physical preview setup failed: " + log);
            }
            for (const auto& testCase : cases) {
                preview.setWorldEnvironment(testCase.world);
                for (uint32_t frame = 0; frame < kPhysicalEnvironmentFrames; ++frame) {
                    preview.setRawReadbackEnabled(frame + 1 == kPhysicalEnvironmentFrames);
                    if (!preview.render(graph, kPhysicalEnvironmentSize, kPhysicalEnvironmentSize,
                            pt ? "Reference.color" : "Deferred.color", frame + 1 == kPhysicalEnvironmentFrames)) {
                        return RHITestResult::fail(label + "/" + testCase.name + ": " + preview.lastLog());
                    }
                    if (!physicalEnvironmentExecutionMatches(preview.executionStats(), pt)) {
                        return RHITestResult::fail(label + " executed the wrong rendering branch");
                    }
                }
                PhysicalEnvironmentImage image{.rgba = decodePhysicalEnvironmentHDR(preview)};
                if (!physicalEnvironmentImageStatistics(image, log)) {
                    return RHITestResult::fail(label + "/" + testCase.name + ": " + log);
                }
                const auto prefix = context.outputDirectory / (testCase.name + "_" + label);
                std::ofstream raw(prefix.string() + ".rgba32f", std::ios::binary);
                raw.write(reinterpret_cast<const char*>(image.rgba.data()), image.rgba.size() * sizeof(float));
                if (!raw) { return RHITestResult::fail("Physical HDR evidence write failed"); }
                images[pathIndex].push_back(std::move(image));
                preview.setRawReadbackEnabled(false);
                if (!preview.render(graph, kPhysicalEnvironmentSize, kPhysicalEnvironmentSize, "FinalBlit.color") ||
                    !saveRgba8Png(prefix.string() + ".png", reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                        kPhysicalEnvironmentSize, kPhysicalEnvironmentSize, log)) {
                    return RHITestResult::fail(label + "/" + testCase.name + ": " + preview.lastLog() + log);
                }
                if (!physicalEnvironmentExecutionMatches(preview.executionStats(), pt)) {
                    return RHITestResult::fail(label + " presentation reintroduced an unused branch");
                }
            }
            auto executed = RenderGraphProperties::array();
            for (const auto& node : preview.executionStats().nodes) {
                executed.push_back({{"name", node.name}, {"type", node.type}});
            }
            executedGraphs[label] = std::move(executed);
            if (!saveRenderGraphToFile(graph, context.outputDirectory / (label + ".metallic_graph.json"), log)) {
                return RHITestResult::fail(log);
            }
        }

        auto results = RenderGraphProperties::array();
        double largestBackgroundError = 0.0;
        for (size_t index = 0; index < cases.size(); ++index) {
            const auto& pt = images[0][index];
            const auto& raster = images[1][index];
            const double backgroundError = physicalEnvironmentBackgroundError(raster, pt);
            largestBackgroundError = std::max(largestBackgroundError, backgroundError);
            results.push_back({{"case", cases[index].name}, {"altitudeMetres", cases[index].altitudeMetres},
                {"pathTraceMeanRGB", pt.meanRGB}, {"deferredMeanRGB", raster.meanRGB},
                {"pathTraceBackgroundMeanRGB", pt.backgroundMeanRGB},
                {"deferredBackgroundMeanRGB", raster.backgroundMeanRGB},
                {"fullRelativeRMSE", physicalEnvironmentRelativeRMSE(raster, pt)},
                {"backgroundRelativeError", backgroundError}});
        }
        RenderGraphProperties report{{"width", kPhysicalEnvironmentSize}, {"height", kPhysicalEnvironmentSize},
            {"framesPerCaseAndPath", kPhysicalEnvironmentFrames}, {"cacheMode", "off"},
            {"exposureEV100", lighting.exposureEV100}, {"backgroundCornerSize", kBackgroundCornerSize},
            {"backgroundCorners", "upper left and upper right"}, {"executedPasses", executedGraphs},
            {"cases", results}};
        std::ofstream summary(context.outputDirectory / "PhysicalEnvironmentRendering.json");
        summary << report.dump(2) << '\n';
        summary.close();
        if (!summary) { return RHITestResult::fail("Physical environment report write failed"); }
        if (largestBackgroundError > 0.1) {
            return RHITestResult::fail("Physical raster/PT background alignment exceeded 10%; largest=" +
                std::to_string(largestBackgroundError) + "; see PhysicalEnvironmentRendering.json");
        }
        for (uint32_t pathIndex = 0; pathIndex < 2; ++pathIndex) {
            const auto& pathImages = images[pathIndex];
            if (physicalEnvironmentRelativeRMSE(pathImages[0], pathImages[1]) < 0.001 ||
                physicalEnvironmentRelativeRMSE(pathImages[0], pathImages[2]) < 0.001 ||
                physicalEnvironmentBackgroundBrightness(pathImages[2]) >= physicalEnvironmentBackgroundBrightness(pathImages[0]) ||
                physicalEnvironmentRelativeRMSE(pathImages[cases.size() - 2], pathImages.back()) < 0.0001) {
                return RHITestResult::fail("Physical Sun/ground parameter changes did not affect " +
                    std::string(pathIndex == 0 ? "path tracing" : "raster") + "; see PhysicalEnvironmentRendering.json");
            }
        }
        return RHITestResult::pass("12 physical cases x 32 frames x actual raster/PT branches; finite HDR, "
            "sky alignment, Sun/ground response and HDR/PNG evidence; foreground differences quantified");
    }
};
METALLIC_REGISTER_RHI_TEST(PhysicalEnvironmentRenderingTest);

class PhysicalEnvironmentHistoryTest : public RHITest {
public:
    explicit PhysicalEnvironmentHistoryTest(bool nrc = false) : nrc_(nrc)
    {
        name = nrc ? "physical_environment_nrc_source_and_history" : "physical_environment_source_and_history";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        if (nrc_) {
#if METALLIC_HAS_NRC
            const char* enabled = std::getenv("METALLIC_TEST_NRC_CACHE");
            if (!enabled || std::string_view(enabled) != "1") { return RHITestResult::skip("Set METALLIC_TEST_NRC_CACHE=1 for native NRC validation"); }
#else
            return RHITestResult::skip("Built without the NRC SDK");
#endif
        }
        const auto fixture = std::filesystem::absolute(context.outputDirectory / "ConstantAP1.hdr");
        constexpr std::array<float, 3> kHDRIRadiance{32.0f, 16.0f, 8.0f};
        std::vector<float> texels(16 * 8 * 3);
        for (size_t index = 0; index < texels.size(); index += 3) {
            std::copy(kHDRIRadiance.begin(), kHDRIRadiance.end(), texels.begin() + index);
        }
        if (!stbi_write_hdr(fixture.string().c_str(), 16, 8, 3, texels.data())) {
            return RHITestResult::fail("Failed to write constant HDRI source-switch fixture");
        }
        RenderSampleLoadResult sample;
        std::string log;
        if (!loadBuiltInRenderSample("physical-atmosphere-lookdev", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RHITestResult::fail(document.lastLoadResult().error);
        }
        auto report = RenderGraphProperties::object();
        for (uint32_t pathIndex = 0; pathIndex < (nrc_ ? 1u : 2u); ++pathIndex) {
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
                properties["cacheMode"] = nrc_ ? "nrc" : "off";
                if (nrc_) { properties["bsdf"] = "standard"; }
                properties["accumulate"] = true;
                if (!graph.setNodeProperties(node->id, std::move(properties))) { return RHITestResult::fail("Set PT history fixture"); }
            }
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&document);
            auto hdri = document.environment();
            preview.setExecutionCaptureEnabled(nrc_);
            hdri.enabled = true;
            hdri.visible = true;
            hdri.path = fixture;
            hdri.intensity = 1.0f;
            hdri.rotationDegrees = 0.0f;
            hdri.sourceColorSpace = kACEScg;
            hdri.sourceColorSpaceExplicit = true;
            preview.setEnvironment(hdri);
            preview.setLighting(document.lighting());
            if (!preview.subsystemHost()->configure<EnvironmentLightingSubsystem>(
                    {.initialDecodeTimeoutMilliseconds = 10000}, log) ||
                !preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail(log); }
            auto world = document.worldEnvironment();
            world.sun.enabled = true;
            world.sun.direction = float3(0.0f, -1.0f, 0.0f);
            world.moon.enabled = false;
            auto renderState = [&](std::string_view state, uint32_t frames, PhysicalEnvironmentImage& image) {
                if (!preview.setWorldEnvironment(world)) { return false; }
                for (uint32_t frame = 0; frame < frames; ++frame) {
                    preview.setRawReadbackEnabled(frame + 1 == frames);
                    if (!preview.render(graph, kPhysicalEnvironmentSize, kPhysicalEnvironmentSize,
                            pt ? "Reference.color" : "Deferred.color", frame + 1 == frames)) { return false; }
                    if (nrc_) {
                        const auto snapshot = preview.executionSnapshot();
                        if (!snapshot || !std::any_of(snapshot->passes.begin(), snapshot->passes.end(), [](const auto& pass) {
                                return pass.name == "Reference" && std::any_of(pass.stages.begin(), pass.stages.end(),
                                    [](const auto& stage) { return stage.name == "NRC tonemap"; });
                            })) { log = "Physical NRC test did not execute native NRC resolve/tonemap"; return false; }
                    }
                }
                image.rgba = decodePhysicalEnvironmentHDR(preview);
                if (!physicalEnvironmentImageStatistics(image, log)) {
                    const auto failurePath = context.outputDirectory /
                        (label + "_" + std::string(state) + "_Failed.rgba32f");
                    std::ofstream failure(failurePath, std::ios::binary);
                    failure.write(reinterpret_cast<const char*>(image.rgba.data()),
                        std::streamsize(image.rgba.size() * sizeof(float)));
                    log = std::string(state) + ": " + log;
                    return false;
                }
                return physicalEnvironmentExecutionMatches(preview.executionStats(), pt);
            };
            PhysicalEnvironmentImage warm, dark, hdriImage, darkAgain, restored, halfSun;
            if (!renderState("warm", 16, warm)) { return RHITestResult::fail(label + ": " + preview.lastLog() + log); }
            world.sun.enabled = false;
            if (!renderState("dark", 1, dark)) { return RHITestResult::fail(label + ": " + preview.lastLog() + log); }
            world.source = environment::EnvironmentSource::HDRI;
            if (!renderState("HDRI", 1, hdriImage)) { return RHITestResult::fail(label + ": " + preview.lastLog() + log); }
            world.source = environment::EnvironmentSource::PhysicalAtmosphere;
            if (!renderState("darkAgain", 1, darkAgain)) { return RHITestResult::fail(label + ": " + preview.lastLog() + log); }
            world.sun.enabled = true;
            if (!renderState("restored", 1, restored)) { return RHITestResult::fail(label + ": " + preview.lastLog() + log); }
            world.sun.topOfAtmosphereIrradiance *= 0.5f;
            if (!renderState("halfSun", 1, halfSun)) { return RHITestResult::fail(label + ": " + preview.lastLog() + log); }
            double halfError = 0.0;
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const double expected = restored.backgroundMeanRGB[channel] * 0.5;
                halfError = std::max(halfError, std::abs(halfSun.backgroundMeanRGB[channel] - expected) / std::max(expected, 1e-7));
                if (std::abs(dark.backgroundMeanRGB[channel]) > 1e-6 ||
                    std::abs(darkAgain.backgroundMeanRGB[channel]) > 1e-6 ||
                    std::abs(hdriImage.backgroundMeanRGB[channel] - kHDRIRadiance[channel]) > 0.001) {
                    return RHITestResult::fail(label + " source/light edit retained stale sky or accumulation");
                }
            }
            if (physicalEnvironmentBackgroundBrightness(restored) < 1.0 || halfError > 0.03) {
                return RHITestResult::fail(label + " TOA edit did not reset history to half-intensity sky");
            }
            report[label] = {{"darkAfterWarmup", dark.backgroundMeanRGB}, {"HDRI", hdriImage.backgroundMeanRGB},
                {"darkAfterHDRI", darkAgain.backgroundMeanRGB}, {"restored", restored.backgroundMeanRGB},
                {"halfTOA", halfSun.backgroundMeanRGB}, {"halfTOARelativeError", halfError}};
        }
        std::ofstream summary(context.outputDirectory / (nrc_ ? "PhysicalEnvironmentNRCHistory.json" : "PhysicalEnvironmentHistory.json"));
        summary << report.dump(2) << '\n';
        if (!summary) { return RHITestResult::fail("Source/history evidence write failed"); }
        return RHITestResult::pass(nrc_ ? "Native NRC physical source switching and camera-segment history" :
            "HDRI/physical source switching, dark-frame history rejection and TOA response on both real branches");
    }

private:
    bool nrc_ = false;
};
METALLIC_REGISTER_RHI_TEST(PhysicalEnvironmentHistoryTest);

class PhysicalEnvironmentNRCHistoryTest final : public PhysicalEnvironmentHistoryTest {
public:
    PhysicalEnvironmentNRCHistoryTest() : PhysicalEnvironmentHistoryTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(PhysicalEnvironmentNRCHistoryTest);

} // namespace
} // namespace metallic::tests
