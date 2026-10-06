#include "RHITest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {

class StudioLookDevTest final : public RHITest
{
public:
    StudioLookDevTest() { name = "studio_lookdev"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        auto samples = listBuiltInRenderSamples();
        std::erase_if(samples, [](const auto& sample) { return sample.category != "Studio LookDev"; });
        if (samples.empty()) { return RHITestResult::skip("Generate White Studio 02 with BuildStudioLookDev.py"); }
        nlohmann::json report = {{"loadedScenes", 0}, {"validation", context.enableValidation}, {"captures", nlohmann::json::array()}};
        std::string log;
        // Validate every copied material/texture binding, including unchanged Painter programs.
        for (const auto& sample : samples) {
            scene::SceneDocument document;
            if (!document.load(sample.scenePath) || !document.documentWarning().empty() ||
                document.environment().path.filename() != "WhiteStudio02.hdr" ||
                !document.environment().hasExplicitSourceColorSpace() || document.lighting().autoExposure.enabled) {
                return RHITestResult::fail(sample.id + ": scene/environment contract failed " + document.documentWarning());
            }
            report["loadedScenes"] = report["loadedScenes"].get<int>() + 1;
        }
        const auto root = std::filesystem::path(PROJECT_SOURCE_DIR) / "build/MaterialValidation/WhiteStudio02";
        std::ifstream input(root / "ChartMeasurements.json");
        const auto measurements = nlohmann::json::parse(input);
        scene::SceneDocument document;
        constexpr uint32_t size = 640;
        for (const char* id : {"studio-white-overview", "studio-white-chart", "studio-white-M03_CoatedPaint-textured"}) {
            RenderSampleLoadResult sample;
            if (!loadBuiltInRenderSample(id, sample, log) || !document.load(sample.desc.scenePath)) {
                return RHITestResult::fail(log);
            }
            // Each preset owns its camera; a persistent preview's live view overrides graph defaults.
            RenderGraphPreviewRenderer preview;
            if (!preview.subsystemHost()->configure<EnvironmentLightingSubsystem>({.initialDecodeTimeoutMilliseconds = 10000}, log) ||
                !preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("Preview initialization failed: " + log); }
            preview.setRawReadbackEnabled(true);
            preview.bindRuntimeScene(&document);
            preview.setEnvironment(document.environment());
            preview.setLighting(document.lighting());
            for (const char* output : {"Reference.color", "Deferred.color", "FinalBlit.color"}) {
                for (int frame = 0; frame < 16; ++frame) {
                    if (!preview.render(sample.graph, size, size, output) ||
                        preview.lastLog().find("error material") != std::string::npos) { return RHITestResult::fail(preview.lastLog()); }
                    const auto& env = preview.subsystemHost()->get<EnvironmentLightingSubsystem>()->snapshot();
                    if (env.status != EnvironmentLightingStatus::Ready || !env.mapAvailable || env.width != 4096 || env.height != 2048) {
                        return RHITestResult::fail("4K studio HDRI was not ready: " + env.error);
                    }
                }
                const std::string stem = std::string(id) + "-" + output;
                if (!saveRgba8Png(context.outputDirectory / (stem + ".png"),
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), size, size, log)) { return RHITestResult::fail(log); }
                nlohmann::json result = {{"scene", id}, {"output", output}};
                if (std::string_view(output) != "FinalBlit.color") {
                    if (preview.readbackFormat() != Format::RGBA32Sfloat) { return RHITestResult::fail("Expected linear float32"); }
                    float maximum = 0;
                    const auto& bytes = preview.readbackBytes();
                    for (size_t offset = 0; offset < bytes.size(); offset += sizeof(float)) {
                        float value; std::memcpy(&value, bytes.data() + offset, sizeof(value));
                        if (!std::isfinite(value)) { return RHITestResult::fail(stem + ": nonfinite output"); }
                        if ((offset / 4) % 4 != 3) { maximum = std::max(maximum, value); }
                    }
                    if (maximum <= 0) { return RHITestResult::fail(stem + ": black output"); }
                    result["maximum"] = maximum;
                    std::ofstream raw(context.outputDirectory / (stem + ".rgba32f"), std::ios::binary);
                    raw.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                    if (std::string_view(id) == "studio-white-chart") {
                        float maxError = 0;
                        for (int i = 0; i < 24; ++i) {
                            const float x = .07f + (i % 6) * .155f, y = .28f - (i / 6) * .19f;
                            const float extent = 2.895f * std::tan(20.f * 3.14159265359f / 180.f);
                            const int px = static_cast<int>((.5f + x / (2 * extent)) * size);
                            const int py = static_cast<int>((.5f - y / (2 * extent)) * size);
                            color::RGB expected = measurements["patches"][i]["medianLinearRec709"].get<color::RGB>();
                            for (float& v : expected) { v *= measurements["displayGain"].get<float>(); }
                            expected = color::fromLinearRec709(expected);
                            std::array<float, 4> actual;
                            std::memcpy(actual.data(), bytes.data() + (py * size + px) * 16, 16);
                            for (int c = 0; c < 3; ++c) { maxError = std::max(maxError, std::abs(actual[c] - expected[c])); }
                        }
                        result["patchMaxAbsoluteError"] = maxError;
                        if (maxError > .002f) { return RHITestResult::fail(stem + ": chart radiance mismatch " + std::to_string(maxError)); }
                    }
                }
                report["captures"].push_back(std::move(result));
                std::ofstream(context.outputDirectory / "StudioLookDev.json") << report.dump(2);
            }
        }
        return RHITestResult::pass("All studio scenes load; 3 PT/Deferred comparisons with ready 4K HDRI; 24 photographic swatches match linear radiance");
    }
};
METALLIC_REGISTER_RHI_TEST(StudioLookDevTest);

} // namespace
} // namespace metallic::tests
