#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

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
        const char* requested = std::getenv("METALLIC_MATERIAL_BASELINE_CASES");
        const auto casePath = requested ? std::filesystem::path(requested) : root / "LookDev/MaterialSystem/Cases.json";
        const auto fixtures = casePath.parent_path();
        std::ifstream input(casePath);
        const auto cases = Json::parse(input);
        const uint32_t frames = cases.at("frames"), warmup = cases.at("warmupFrames"), count = cases.at("timingFrames");
        if (!count || frames <= uint64_t(warmup) + count) {
            return RHITestResult::fail("Capture must follow the complete timing window");
        }
        Json report = {{"version", 1}, {"validation", context.enableValidation}, {"cases", Json::array()}};
        for (const auto& spec : cases.at("cases")) {
            const uint32_t caseFrames = spec.value("frameCount", frames);
            const auto captureFrames = spec.value("captureFrames", std::vector<uint32_t>{});
            if (caseFrames <= uint64_t(warmup) + count ||
                std::any_of(captureFrames.begin(), captureFrames.end(), [&](uint32_t frame) {
                    return frame >= caseFrames || (frame >= warmup && frame < uint64_t(warmup) + count);
                })) {
                return RHITestResult::fail("Differential captures must be valid frames outside the timing window");
            }
            const auto id = spec.at("id").get<std::string>();
            RenderGraph graph;
            std::string log;
            if (!loadRenderGraphFromFile(fixtures / spec.at("graph").get<std::string>(), graph, log)) {
                return RHITestResult::fail(log);
            }
            scene::SceneDocument document;
            RenderView view;
            RenderGraphPreviewRenderer preview;
            if (!preview.subsystemHost()->configure<EnvironmentLightingSubsystem>(
                {.initialDecodeTimeoutMilliseconds = 10000}, log)) { return RHITestResult::fail(log); }
            if (!spec.at("scene").is_null()) {
                if (!document.load(root / spec.at("scene").get<std::string>()) ||
                    !document.documentWarning().empty()) { return RHITestResult::fail("Baseline scene load failed"); }
                const auto graphJSON = Json::parse(serializeRenderGraphToString(graph));
                bool explicitCamera = false;
                for (const auto& node : graphJSON.at("nodes")) {
                    const auto& properties = node.at("properties");
                    if (properties.contains("camera") && properties["camera"].contains("eye") &&
                        properties["camera"].contains("center")) { explicitCamera = true; }
                }
                if ((!explicitCamera && (document.cameras().empty() || document.cameras()[0].fallback)) || document.lighting().autoExposure.enabled) {
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
                if (spec.contains("materialAsset")) {
                    for (size_t index = 0; index < document.materials().size(); ++index) {
                        if (!document.setMaterialAsset(static_cast<int32_t>(index), spec.at("materialAsset").get<std::string>(),
                            root / spec.at("materialRoot").get<std::string>(), log)) { return RHITestResult::fail(log); }
                    }
                }
                preview.bindRuntimeScene(&document);
                preview.setEnvironment(document.environment());
                if (!preview.setLighting(document.lighting())) { return RHITestResult::fail("Invalid baseline lighting"); }
            } else if (spec.contains("environment")) {
                const auto& env = spec.at("environment");
                preview.setEnvironment({.enabled = true, .path = (root / env.at("path").get<std::string>()).string(),
                    .intensity = env.at("intensity"), .rotationDegrees = env.at("rotationDegrees"), .visible = true});
                scene::LightingSettings lighting;
                lighting.autoExposure.enabled = false;
                if (!preview.setLighting(lighting)) { return RHITestResult::fail("Invalid fiber lighting"); }
            }
            Json viewCamera;
            const bool useView = spec.value("useRenderView", false);
            if (useView) {
                auto* node = graph.findNode("PathTrace");
                if (!node || !node->properties.contains("camera")) {
                    return RHITestResult::fail("Differential RenderView requires an explicit PathTrace camera");
                }
                viewCamera = node->properties.at("camera");
                view.setRenderResolution(spec.at("width"), spec.at("height"));
                if (!view.setCameraProperties(viewCamera)) { return RHITestResult::fail("Invalid RenderView camera"); }
                node->properties.erase("camera");
                preview.bindRenderView(&view);
            }
            if (!preview.initialize(context.enableValidation, true, false)) {
                return RHITestResult::fail("Baseline device initialization failed; no baseline produced");
            }
            const uint32_t width = spec.at("width"), height = spec.at("height");
            Json result = spec;
            result["frames"] = Json::array();
            result["warmup"] = Json::array();
            result["environmentTransitions"] = Json::array();
            Json previousEnvironment;
            result["resolvedGraph"] = Json::parse(serializeRenderGraphToString(graph));
            result["captures"] = Json::array();
            const auto captureOutput = [&](const std::string& outputName, uint32_t frame) {
                const auto& bytes = preview.readbackBytes();
                if (bytes.empty()) { throw std::runtime_error("Empty differential capture: " + outputName); }
                const auto filename = id + "-" + std::to_string(frame) + "-" + outputName + ".bin";
                std::ofstream file(context.outputDirectory / filename, std::ios::binary);
                file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
                if (!file) { throw std::runtime_error("Cannot write differential capture"); }
                result["captures"].push_back({{"output", outputName}, {"frame", frame}, {"file", filename},
                    {"bytes", bytes.size()}, {"format", static_cast<uint32_t>(preview.readbackFormat())}});
            };
            for (uint32_t frame = 0; frame < caseFrames; ++frame) {
                if (spec.contains("cameraFrames")) {
                    for (const auto& change : spec.at("cameraFrames")) {
                        if (change.at("frame") == frame) {
                            const auto* node = graph.findNode(change.at("node").get<std::string>());
                            if (!node) { return RHITestResult::fail("Camera sequence node missing"); }
                            if (useView) {
                                viewCamera = change.at("camera");
                                if (!view.setCameraProperties(viewCamera)) { return RHITestResult::fail("Invalid moving camera"); }
                            } else {
                                graph.setNodeRuntimeProperty(node->id, "camera", change.at("camera"));
                            }
                        }
                    }
                }
                const bool differentialCapture = std::find(captureFrames.begin(), captureFrames.end(), frame) != captureFrames.end();
                const bool capture = frame + 1 == caseFrames || differentialCapture;
                preview.setRawReadbackEnabled(capture);
                if (!preview.render(graph, width, height, spec.at("output").get<std::string>(), capture)) {
                    return RHITestResult::fail(id + ": " + preview.lastLog());
                }
                if (differentialCapture) { captureOutput(spec.at("output"), frame); }
                const auto* environment = preview.subsystemHost()->get<EnvironmentLightingSubsystem>();
                if (spec.value("requireEnvironment", true) && !environment) {
                    return RHITestResult::fail("Baseline environment subsystem missing");
                }
                if (environment) {
                    const auto& snapshot = environment->snapshot();
                    Json environmentState = {{"status", static_cast<uint32_t>(snapshot.status)},
                        {"resourceRevision", snapshot.resourceRevision}, {"mapAvailable", snapshot.mapAvailable}};
                    if (environmentState != previousEnvironment) {
                        previousEnvironment = environmentState;
                        environmentState["frame"] = frame;
                        result["environmentTransitions"].push_back(std::move(environmentState));
                    }
                    if (spec.value("requireEnvironment", true) &&
                        (snapshot.status != EnvironmentLightingStatus::Ready || !snapshot.mapAvailable)) {
                        return RHITestResult::fail(id + ": environment not ready for every baseline sample");
                    }
                }
                std::vector<RenderGraphExecutionStats> completed;
                if (!preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); })) {
                    return RHITestResult::fail("Baseline timestamp readback failed");
                }
                if (frame >= warmup + count) { continue; }
                if (completed.size() != 1 || !completed[0].gpuTimingAvailable || completed[0].profilingOverflow) {
                    return RHITestResult::fail("Missing, ambiguous or overflowed GPU timestamp frame");
                }
                const auto& stats = completed[0];
                Json timing = {{"frame", frame}, {"executionId", stats.executionId}, {"graphMs", stats.gpuMilliseconds},
                    {"warmup", frame < warmup},
                    {"asyncComputeBranches", stats.asyncComputeBranches},
                    {"nodes", Json::array()}};
                for (const auto& node : stats.nodes) {
                    Json entry = {{"name", node.name}, {"gpuMs", node.gpuTimingAvailable ? Json(node.gpuMilliseconds) : Json(nullptr)},
                        {"queue", static_cast<uint32_t>(node.queue)},
                        {"sections", Json::array()}};
                    for (const auto& section : node.sections) {
                        entry["sections"].push_back({{"name", section.name}, {"parent", section.parent},
                            {"gpuMs", section.gpuTimingAvailable ? Json(section.gpuMilliseconds) : Json(nullptr)}});
                    }
                    timing["nodes"].push_back(std::move(entry));
                }
                result[frame < warmup ? "warmup" : "frames"].push_back(std::move(timing));
            }
            const auto format = preview.readbackFormat();
            const bool half = format == Format::RGBA16Sfloat;
            if (!half && format != Format::RGBA32Sfloat) { return RHITestResult::fail("Baseline must be linear HDR"); }
            const auto& bytes = preview.readbackBytes();
            if (bytes.size() != size_t(width) * height * (half ? 8 : 16)) { return RHITestResult::fail("Truncated HDR capture"); }
            uint64_t nonfiniteComponents = 0;
            for (size_t i = 0; i < bytes.size(); i += half ? 2 : 4) {
                if (half) {
                    uint16_t bits; std::memcpy(&bits, bytes.data() + i, 2);
                    if ((bits & 0x7c00u) == 0x7c00u) { ++nonfiniteComponents; }
                } else {
                    float value; std::memcpy(&value, bytes.data() + i, 4);
                    if (!std::isfinite(value)) { ++nonfiniteComponents; }
                }
            }
            const auto filename = id + (half ? ".rgba16f" : ".rgba32f");
            std::ofstream raw(context.outputDirectory / filename, std::ios::binary);
            raw.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            if (!raw) { return RHITestResult::fail("Baseline HDR write failed"); }
            result["image"] = filename;
            result["format"] = half ? "RGBA16F" : "RGBA32F";
            result["nonfiniteComponents"] = nonfiniteComponents;
            result["validHDR"] = nonfiniteComponents == 0;
            // Additional guides are captured after the timing window. Every arm
            // executes the same frame sequence; no readback enters timed frames.
            uint32_t captureFrame = caseFrames;
            for (const auto& outputName : spec.value("captureOutputs", std::vector<std::string>{})) {
                if (spec.contains("guideCameraStep")) {
                    const auto& step = spec.at("guideCameraStep");
                    const auto* node = graph.findNode(step.at("node").get<std::string>());
                    if (!node) { return RHITestResult::fail("Guide camera node missing"); }
                    auto camera = useView ? viewCamera : node->properties.at("camera");
                    for (size_t axis = 0; axis < 3; ++axis) {
                        camera["eye"][axis] = camera["eye"][axis].get<float>() + step.at("delta")[axis].get<float>();
                    }
                    if (useView) {
                        viewCamera = camera;
                        if (!view.setCameraProperties(camera)) { return RHITestResult::fail("Invalid guide camera"); }
                    } else {
                        graph.setNodeRuntimeProperty(node->id, "camera", camera);
                    }
                }
                preview.setRawReadbackEnabled(true);
                if (!preview.render(graph, width, height, outputName, true)) {
                    return RHITestResult::fail(preview.lastLog());
                }
                captureOutput(outputName, captureFrame++);
            }
            report["cases"].push_back(std::move(result));
            // Keep completed cases and invalid raw images even if a later case fails.
            std::ofstream(context.outputDirectory / "MaterialBaseline.json") << report.dump(2) << '\n';
            if (nonfiniteComponents && cases.value("version", 1) < 2) {
                return RHITestResult::fail("Nonfinite HDR capture");
            }
        }
        std::ofstream output(context.outputDirectory / "MaterialBaseline.json");
        output << report.dump(2) << '\n';
        if (!output) { return RHITestResult::fail("Baseline report write failed"); }
        return RHITestResult::pass("Capture complete; per-case validHDR is the baseline quality gate");
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
