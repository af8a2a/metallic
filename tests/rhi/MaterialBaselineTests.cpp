#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Material/MaterialAsset.h"

#include <cmath>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {

class MaterialBaselineTest : public RHITest
{
public:
    explicit MaterialBaselineTest(bool assetRoundTrip = false) : assetRoundTrip_(assetRoundTrip)
    {
        name = assetRoundTrip ? "material_asset_phase0_equivalence" : "material_phase0_baseline";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        using Json = nlohmann::json;
        const auto root = std::filesystem::path(PROJECT_SOURCE_DIR);
        const auto fixtures = root / "LookDev/MaterialSystem";
        std::ifstream input(fixtures / "Cases.json");
        const auto cases = Json::parse(input);
        Json report = {{"version", 1}, {"validation", context.enableValidation}, {"cases", Json::array()}};
        for (const auto& spec : cases.at("cases")) {
            const auto id = spec.at("id").get<std::string>();
            RenderGraph graph;
            std::string log;
            if (!loadRenderGraphFromFile(fixtures / spec.at("graph").get<std::string>(), graph, log)) {
                return RHITestResult::fail(log);
            }
            scene::SceneDocument document;
            RenderGraphPreviewRenderer preview;
            if (!spec.at("scene").is_null()) {
                if (!document.load(root / spec.at("scene").get<std::string>()) ||
                    !document.documentWarning().empty()) { return RHITestResult::fail("Baseline scene load failed"); }
                if (document.cameras().empty() || document.cameras()[0].fallback || document.lighting().autoExposure.enabled) {
                    return RHITestResult::fail("Baseline requires an authored camera and manual exposure");
                }
                if (assetRoundTrip_) {
                    const auto assetRoot = context.outputDirectory / "assets" / id;
                    std::filesystem::create_directories(assetRoot);
                    const auto definition = material::defaultOpenPBRDefinition();
                    std::ofstream(assetRoot / "OpenPBR.materialdef") << material::serializeMaterialDefinition(definition);
                    material::MaterialAssetLibrary library(assetRoot);
                    for (size_t index = 0; index < document.materials().size(); ++index) {
                        material::MaterialInstance instance;
                        if (!material::createMaterialInstance(document.materials()[index], "asset://OpenPBR.materialdef", definition,
                            [](auto slot, auto) { return "imported://" + std::string(slot); }, instance, log)) { return RHITestResult::fail(log); }
                        const auto uri = "asset://Material" + std::to_string(index) + ".material";
                        if (!library.save(uri, instance, log) || !document.setMaterialAsset(static_cast<int32_t>(index), uri, assetRoot, log)) {
                            return RHITestResult::fail(log);
                        }
                    }
                }
                preview.bindRuntimeScene(&document);
                preview.setEnvironment(document.environment());
                if (!preview.setLighting(document.lighting())) { return RHITestResult::fail("Invalid baseline lighting"); }
            } else {
                const auto& env = spec.at("environment");
                preview.setEnvironment({.enabled = true, .path = (root / env.at("path").get<std::string>()).string(),
                    .intensity = env.at("intensity"), .rotationDegrees = env.at("rotationDegrees"), .visible = true});
                scene::LightingSettings lighting;
                lighting.autoExposure.enabled = false;
                if (!preview.setLighting(lighting)) { return RHITestResult::fail("Invalid fiber lighting"); }
            }
            if (!preview.initialize(context.enableValidation, true, false)) {
                return RHITestResult::fail("Baseline device initialization failed; no baseline produced");
            }
            const uint32_t width = spec.at("width"), height = spec.at("height");
            const uint32_t frames = cases.at("frames"), warmup = cases.at("warmupFrames"), count = cases.at("timingFrames");
            Json result = spec;
            result["frames"] = Json::array();
            result["resolvedGraph"] = Json::parse(serializeRenderGraphToString(graph));
            for (uint32_t frame = 0; frame < frames; ++frame) {
                const bool capture = frame + 1 == frames;
                preview.setRawReadbackEnabled(capture);
                if (!preview.render(graph, width, height, spec.at("output").get<std::string>(), capture)) {
                    return RHITestResult::fail(id + ": " + preview.lastLog());
                }
                std::vector<RenderGraphExecutionStats> completed;
                if (!preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); })) {
                    return RHITestResult::fail("Baseline timestamp readback failed");
                }
                if (frame < warmup || frame >= warmup + count) { continue; }
                if (completed.size() != 1 || !completed[0].gpuTimingAvailable || completed[0].profilingOverflow) {
                    return RHITestResult::fail("Missing, ambiguous or overflowed GPU timestamp frame");
                }
                const auto& stats = completed[0];
                Json timing = {{"frame", frame}, {"executionId", stats.executionId}, {"graphMs", stats.gpuMilliseconds},
                    {"nodes", Json::array()}};
                for (const auto& node : stats.nodes) {
                    Json entry = {{"name", node.name}, {"gpuMs", node.gpuTimingAvailable ? Json(node.gpuMilliseconds) : Json(nullptr)},
                        {"sections", Json::array()}};
                    for (const auto& section : node.sections) {
                        entry["sections"].push_back({{"name", section.name}, {"parent", section.parent},
                            {"gpuMs", section.gpuTimingAvailable ? Json(section.gpuMilliseconds) : Json(nullptr)}});
                    }
                    timing["nodes"].push_back(std::move(entry));
                }
                result["frames"].push_back(std::move(timing));
            }
            const auto format = preview.readbackFormat();
            const bool half = format == Format::RGBA16Sfloat;
            if (!half && format != Format::RGBA32Sfloat) { return RHITestResult::fail("Baseline must be linear HDR"); }
            const auto& bytes = preview.readbackBytes();
            if (bytes.size() != size_t(width) * height * (half ? 8 : 16)) { return RHITestResult::fail("Truncated HDR capture"); }
            for (size_t i = 0; i < bytes.size(); i += half ? 2 : 4) {
                if (half) {
                    uint16_t bits; std::memcpy(&bits, bytes.data() + i, 2);
                    if ((bits & 0x7c00u) == 0x7c00u) { return RHITestResult::fail("Nonfinite HDR capture"); }
                } else {
                    float value; std::memcpy(&value, bytes.data() + i, 4);
                    if (!std::isfinite(value)) { return RHITestResult::fail("Nonfinite HDR capture"); }
                }
            }
            const auto filename = id + (half ? ".rgba16f" : ".rgba32f");
            std::ofstream raw(context.outputDirectory / filename, std::ios::binary);
            raw.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            if (!raw) { return RHITestResult::fail("Baseline HDR write failed"); }
            result["image"] = filename;
            result["format"] = half ? "RGBA16F" : "RGBA32F";
            report["cases"].push_back(std::move(result));
        }
        std::ofstream output(context.outputDirectory / "MaterialBaseline.json");
        output << report.dump(2) << '\n';
        if (!output) { return RHITestResult::fail("Baseline report write failed"); }
        return RHITestResult::pass("Three fixed linear HDR references and 64 warmed GPU timestamp frames per case");
    }
private:
    bool assetRoundTrip_ = false;
};
METALLIC_REGISTER_RHI_TEST(MaterialBaselineTest);

class MaterialAssetBaselineTest final : public MaterialBaselineTest
{
public:
    MaterialAssetBaselineTest() : MaterialBaselineTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(MaterialAssetBaselineTest);

} // namespace
} // namespace metallic::tests
