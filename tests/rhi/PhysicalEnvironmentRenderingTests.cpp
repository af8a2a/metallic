#include "RHITest.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
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
#include <numbers>

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
    bool sawCloudView = false;
    bool sawSlider = false;
    for (const auto& node : stats.nodes) {
        sawPT |= node.name == "Reference";
        sawDeferred |= node.name == "Deferred";
        sawVBuffer |= node.name == "VBuffer";
        sawCloudView |= node.name == "CloudView";
        sawSlider |= node.name == "Slider";
    }
    return sawPT == pathTraced && sawDeferred == !pathTraced && sawVBuffer == !pathTraced &&
        sawCloudView == !pathTraced && !sawSlider;
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
                            pt ? "Reference.color" : "CloudView.color", frame + 1 == kPhysicalEnvironmentFrames)) {
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
    explicit PhysicalEnvironmentHistoryTest(bool sharc = false) : sharc_(sharc)
    {
        name = sharc ? "physical_environment_sharc_source_and_history" : "physical_environment_source_and_history";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
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
        for (uint32_t pathIndex = 0; pathIndex < (sharc_ ? 1u : 2u); ++pathIndex) {
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
                properties["cacheMode"] = sharc_ ? "sharc" : "off";
                if (sharc_) {
                    properties["bsdf"] = "standard";
                    properties["sharc.entriesLog2"] = 16;
                    properties["sharc.updateStride"] = 1;
                }
                properties["accumulate"] = true;
                if (!graph.setNodeProperties(node->id, std::move(properties))) { return RHITestResult::fail("Set PT history fixture"); }
            }
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&document);
            auto hdri = document.environment();
            preview.setExecutionCaptureEnabled(sharc_);
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
                            pt ? "Reference.color" : "CloudView.color", frame + 1 == frames)) { return false; }
                    if (sharc_) {
                        const auto snapshot = preview.executionSnapshot();
                        if (!snapshot || !std::any_of(snapshot->passes.begin(), snapshot->passes.end(), [](const auto& pass) {
                                return pass.name == "Reference" && std::any_of(pass.stages.begin(), pass.stages.end(),
                                    [](const auto& stage) { return stage.name == "SHaRC query"; });
                            })) { log = "Physical SHaRC test did not execute its cache query"; return false; }
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
        std::ofstream summary(context.outputDirectory / (sharc_ ? "PhysicalEnvironmentSharcHistory.json" : "PhysicalEnvironmentHistory.json"));
        summary << report.dump(2) << '\n';
        if (!summary) { return RHITestResult::fail("Source/history evidence write failed"); }
        return RHITestResult::pass(sharc_ ? "SHaRC physical source switching and accumulation history" :
            "HDRI/physical source switching, dark-frame history rejection and TOA response on both real branches");
    }

private:
    bool sharc_ = false;
};
METALLIC_REGISTER_RHI_TEST(PhysicalEnvironmentHistoryTest);

class PhysicalEnvironmentSharcHistoryTest final : public PhysicalEnvironmentHistoryTest {
public:
    PhysicalEnvironmentSharcHistoryTest() : PhysicalEnvironmentHistoryTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(PhysicalEnvironmentSharcHistoryTest);

class AncientRiftValleyExposureReadback final : public render::IRenderDebugObserver {
public:
    explicit AncientRiftValleyExposureReadback(render::RenderGraphPreviewRenderer& preview) : preview_(preview)
    {
        preview_.setDebugObserver(this);
    }

    ~AncientRiftValleyExposureReadback() override
    {
        preview_.setDebugObserver(nullptr);
        if (readback_) { (void)device_->waitIdle(); }
    }

    bool detailedSnapshots() const override { return false; }
    void compiled(debug::DebugValue) override {}
    void endExecution(bool) override {}
    void beginExecution(render::Device& device, debug::DebugEvidenceStamp, render::RenderSubsystemHost*) override
    {
        device_ = &device;
    }

    void boundary(render::CommandBuffer& commands, std::string_view checkpoint, uint32_t, std::string_view pass,
        std::span<const render::DebugResourceBinding> resources, const debug::DebugValue&) override
    {
        using namespace render;
        if (!capture || pass != "AutoExposure" || checkpoint != "AfterPass") { return; }
        const auto source = std::ranges::find(resources, "AutoExposure.exposure", &DebugResourceBinding::id);
        if (source == resources.end() || !source->buffer) { return; }
        if (!readback_ && !device_->createBuffer({.size = sizeof(Values), .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { readback_ = std::move(value); })) {
            return;
        }
        BufferBarrierDesc barrier{.buffer = source->buffer,
            .before = resourceSyncScope(source->state, PipelineStageBits::AllCommands),
            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
            .range = {.offset = source->offset, .size = sizeof(Values)}};
        if (!commands.synchronize({.buffers = {&barrier, 1}})) { return; }
        const auto copied = source->buffer->slice({source->offset, sizeof(Values)}).and_then([&](const auto& from) {
            return readback_->slice().and_then([&](const auto& to) { return commands.copyBuffer(from, to); });
        });
        std::swap(barrier.before, barrier.after);
        captured_ = bool(commands.synchronize({.buffers = {&barrier, 1}})) && bool(copied);
    }

    using Values = std::array<float, 4>;
    std::optional<Values> readAfterFrame()
    {
        if (!std::exchange(captured_, false)) { return std::nullopt; }
        readback_->invalidate();
        const void* data = readback_->map();
        if (!data) { return std::nullopt; }
        Values values;
        std::memcpy(values.data(), data, sizeof(values));
        readback_->unmap();
        return values;
    }
    bool capture = false;

private:
    render::RenderGraphPreviewRenderer& preview_;
    render::Device* device_ = nullptr;
    std::unique_ptr<render::Buffer> readback_;
    bool captured_ = false;
};

class AncientRiftValleyDaylightTest final : public RHITest {
public:
    AncientRiftValleyDaylightTest() { name = "ancient_rift_valley_daylight"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        if (std::getenv("METALLIC_ANCIENT_RIFT_VALLEY_DAYLIGHT") == nullptr) {
            return RHITestResult::skip("Set METALLIC_ANCIENT_RIFT_VALLEY_DAYLIGHT to exercise the cooked 16-region sample");
        }
        constexpr uint32_t kWidth = 1280, kHeight = 720, kWarmupFrames = 40;
        const auto root = std::filesystem::path(PROJECT_SOURCE_DIR);
        RenderSampleLoadResult sample;
        std::string log;
        if (!loadBuiltInRenderSample("gpu-driven-ancient-rift-valley-4x4-daylight", sample, log)) {
            return RHITestResult::fail(log);
        }
        scene::SceneDocument document;
        if (!document.loadStreamMetadata(root / sample.desc.scenePath) || !document.documentWarning().empty()) {
            return RHITestResult::fail(document.lastLoadResult().error + document.documentWarning());
        }
        const auto world = document.worldEnvironment().snapshot();
        const auto& sun = world.celestial[static_cast<size_t>(environment::CelestialLightIndex::Sun)];
        const auto& exposureSettings = document.lighting().autoExposure;
        if (!document.hasStreamGeometry() || document.renderPrimitives().size() != 256 ||
            world.source != environment::EnvironmentSource::PhysicalAtmosphere || !sun.enabled || sun.direction.y >= 0.0f ||
            !world.weather.cloudEnabled || !exposureSettings.enabled) {
            return RHITestResult::fail("Expected the complete streamed terrain with physical daylight, clouds and automatic exposure");
        }

        const auto source = RenderGraphProperties::parse(std::ifstream(document.sourcePath()));
        const auto previewSettings = RenderGraphProperties::parse(std::ifstream(root /
            "Asset/Terrain/AncientRiftValley/Preview/RenderSettings.json"));
        const auto& canyon = previewSettings.at("cameras").at("CanyonVista");
        const double canyonFov = 2.0 * std::atan(18.0 / canyon.at("lensMillimeters").get<double>() * kHeight / kWidth)
            * 180.0 / std::numbers::pi;
        const std::array<std::pair<const char*, RenderGraphProperties>, 2> cameras{{
            {"CanyonVista", {{"projection", "perspective"}, {"eye", canyon.at("eye")}, {"center", canyon.at("target")},
                {"up", {0.0, 1.0, 0.0}}, {"fovDegrees", canyonFov}, {"znear", 0.5}, {"zfar", 3500.0}}},
            {"WorldOverview", source.at("extras").at("metallicTerrainRegions").at("sampleCamera")},
        }};
        auto& graph = sample.graph;
        constexpr auto kFinal = "FinalBlit.color";
        constexpr auto kExposed = "AutoExposure.color";
        constexpr auto kIndirect = "Temporal.indirectDiffuse";
        if (sample.desc.previewOutput != kFinal || !graph.markOutput(kFinal) || !graph.markOutput(kExposed) || !graph.markOutput(kIndirect)) {
            return RHITestResult::fail("Missing daylight graph capture outputs");
        }
        RenderView view;
        if (!view.setRenderResolution(kWidth, kHeight)) { return RHITestResult::fail("Invalid daylight resolution"); }
        view.setTemporalJitter(graph.viewProperties().value("temporalJitter", false));
        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        preview.setEnvironment(document.environment());
        if (!preview.setLighting(document.lighting())) { return RHITestResult::fail("Invalid daylight exposure settings"); }
        const bool enableStreamline = vulkan::streamlineSdkAvailable();
        const auto initialized = preview.initialize(context.enableValidation, false, false, false, {}, enableStreamline);
        if (!initialized) { return RHITestResult::fail("Initialize daylight renderer: " + std::string(toString(initialized))); }
        AncientRiftValleyExposureReadback exposure(preview);
        preview.setRawReadbackEnabled(true);
        auto report = RenderGraphProperties{{"sample", sample.desc.id}, {"resolution", {kWidth, kHeight}},
            {"sunDirection", {sun.direction.x, sun.direction.y, sun.direction.z}}, {"cloudCoverage", world.weather.cloudCoverage},
            {"regionCount", graph.findNode("VBuffer")->properties.at("streamRegions").size()}, {"warmupFrames", kWarmupFrames}};
        for (const auto& [name, camera] : cameras) {
            if (!view.setCameraProperties(camera)) { return RHITestResult::fail(std::string("Invalid camera: ") + name); }
            view.cameraCut();
            for (uint32_t frame = 0; frame < kWarmupFrames; ++frame) {
                if (!preview.render(graph, kWidth, kHeight, kFinal, false)) {
                    return RHITestResult::fail(std::string(name) + ": " + preview.lastLog());
                }
            }
            exposure.capture = true;
            preview.setExecutionCaptureEnabled(true);
            if (!preview.render(graph, kWidth, kHeight, kFinal)) {
                return RHITestResult::fail(std::string(name) + ": " + preview.lastLog());
            }
            exposure.capture = false;
            preview.setExecutionCaptureEnabled(false);
            const auto state = exposure.readAfterFrame();
            if (!state || !std::ranges::all_of(*state, [](float value) { return std::isfinite(value); }) ||
                (*state)[0] <= 0.0f || (*state)[3] <= 0.0f ||
                (*state)[1] < exposureSettings.minEV100 || (*state)[1] > exposureSettings.maxEV100) {
                return RHITestResult::fail(std::string(name) + ": missing or invalid GPU exposure");
            }
            if (!saveRgba8Png(context.outputDirectory / (std::string(name) + ".png"),
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), preview.width(), preview.height(), log)) {
                return RHITestResult::fail(log);
            }
            if (std::string_view(name) == "CanyonVista") {
                if (preview.width() != kWidth || preview.height() != kHeight || preview.pixels().size() != size_t(kWidth) * kHeight) {
                    return RHITestResult::fail("CanyonVista capture has an unexpected size");
                }
                const auto* rgba = reinterpret_cast<const uint8_t*>(preview.pixels().data());
                uint32_t lit = 0;
                // Terrain on the left side of this fixed camera is in the canyon wall's shadow.
                for (uint32_t y = 300; y < 580; ++y) {
                    for (uint32_t x = 120; x < 270; ++x) {
                        const auto pixel = (size_t(y) * preview.width() + x) * 4u;
                        lit += std::max({rgba[pixel], rgba[pixel + 1], rgba[pixel + 2]}) > 12;
                    }
                }
                if (lit * 100u < 95u * (580u - 300u) * (270u - 120u)) {
                    return RHITestResult::fail("CanyonVista shadowed wall lost physical sky fill");
                }
            }
            const auto snapshot = preview.executionSnapshot();
            if (!snapshot) { return RHITestResult::fail("Missing daylight execution snapshot"); }
            auto passes = RenderGraphProperties::object();
            for (const auto& pass : snapshot->passes) {
                auto stages = RenderGraphProperties::array();
                for (const auto& stage : pass.stages) { stages.push_back(stage.name); }
                passes[pass.name] = std::move(stages);
            }
            for (const auto* pass : {"VBuffer", "Shadows", "Deferred", "ScreenTrace", "Gather", "Temporal",
                    "Reflections", "CloudView", "DLSSSR", "AutoExposure", "ColorGrading", "FinalBlit"}) {
                if (!passes.contains(pass)) { return RHITestResult::fail(std::string("Daylight did not execute ") + pass); }
            }
            if (!std::ranges::any_of(passes.at("CloudView"), [](const auto& stage) { return stage == "Cloud quarter-resolution trace"; })) {
                return RHITestResult::fail("Daylight did not trace volumetric clouds");
            }
            auto& shot = report["cameras"][name];
            shot = {{"camera", camera}, {"exposureMultiplier", (*state)[0]}, {"adaptedEV100", (*state)[1]},
                {"targetEV100", (*state)[2]}, {"meteredLuminance", (*state)[3]}, {"passes", std::move(passes)}};
            for (const auto* output : {kExposed, kIndirect}) {
                if (!preview.render(graph, kWidth, kHeight, output)) {
                    return RHITestResult::fail(std::string(name) + ": " + preview.lastLog());
                }
                const auto rgba = decodePhysicalEnvironmentHDR(preview);
                if (rgba.size() != size_t(preview.width()) * preview.height() * 4u ||
                    !std::ranges::all_of(rgba, [](float value) { return std::isfinite(value); })) {
                    return RHITestResult::fail(std::string(name) + ": invalid HDR from " + output);
                }
                double positiveRGB = 0.0;
                uint32_t litPixels = 0;
                for (size_t i = 0; i < rgba.size(); i += 4u) {
                    const float rgb = std::max({rgba[i], rgba[i + 1], rgba[i + 2]});
                    if (rgb > 0.0f) { positiveRGB += rgb; ++litPixels; }
                }
                if (litPixels == 0) { return RHITestResult::fail(std::string(name) + ": empty HDR from " + output); }
                shot[output] = {{"width", preview.width()}, {"height", preview.height()},
                    {"litPixels", litPixels}, {"meanMaxRGB", positiveRGB / (rgba.size() / 4u)}};
            }
        }
        const bool dlss = vulkan::deviceCapabilities(*preview.subsystemHost()->device()).streamlineDlssSr;
        report["dlssSR"] = dlss;
        std::ofstream summary(context.outputDirectory / "AncientRiftValleyDaylight.json");
        summary << report.dump(2) << '\n';
        if (!summary) { return RHITestResult::fail("Writing daylight capture report failed"); }
        return RHITestResult::pass(dlss ? "Daylight captures with DLSS-SR" : "Daylight captures at native resolution");
    }
};
METALLIC_REGISTER_RHI_TEST(AncientRiftValleyDaylightTest);

} // namespace
} // namespace metallic::tests
