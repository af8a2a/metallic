#include "RHITest.h"
#include "ShaderArtifactFixture.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <gtest/gtest.h>

namespace metallic::tests {
namespace {

struct ArtifactReportGuard
{
    std::function<void()> persist;
    ~ArtifactReportGuard()
    {
        if (persist) {
            try { persist(); } catch (...) { /* Preserve the original test failure. */ }
        }
    }
};

bool artifactFailureLog(const std::string& log)
{
    std::string lower = log;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower.find("initial material compilation failed") != std::string::npos ||
        lower.find("displaying the error material") != std::string::npos || lower.find("shaderartifactoverride] rejected") != std::string::npos;
}

class ArtifactRecipeDirectoryScope
{
public:
    explicit ArtifactRecipeDirectoryScope(const std::filesystem::path& path)
        : previous_(std::getenv("METALLIC_PIPELINE_RECIPE_DIRECTORY"))
    {
        ShaderArtifactFixture::require(!std::filesystem::exists(path), "Artifact case recipe directory must be fresh");
        std::filesystem::create_directories(path);
        set(path.string());
    }
    ~ArtifactRecipeDirectoryScope() { set(previous_); }
private:
    static void set(const std::string& path)
    {
#ifdef _WIN32
        _putenv_s("METALLIC_PIPELINE_RECIPE_DIRECTORY", path.c_str());
#else
        setenv("METALLIC_PIPELINE_RECIPE_DIRECTORY", path.c_str(), 1);
#endif
    }
    std::string previous_;
};

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
        const char* artifactManifest = std::getenv("METALLIC_SHADER_ARTIFACT_MANIFEST");
        Json report = {{"version", 1}, {"validation", context.enableValidation}, {"cases", Json::array()}};
        std::shared_ptr<ShaderArtifactFixture> artifact;
        ShaderArtifactOverrideScope artifactScope;
        std::optional<size_t> artifactCaseIndex;
        uint64_t caseConsumedBefore = 0, caseRejectedBefore = 0;
        const auto syncArtifact = [&]() {
            if (!artifact) { return; }
            report["shader_artifact"] = artifact->evidence();
            if (artifactCaseIndex) {
                auto& row = report["artifact_cases"].at(*artifactCaseIndex);
                row["consumed"] = artifact->consumed() - caseConsumedBefore;
                row["rejected_target"] = artifact->rejected() - caseRejectedBefore;
                row["input_sha256"] = artifact->inputSha256();
                row["input_fnv1a64"] = artifact->inputHash();
            }
        };
        ArtifactReportGuard artifactReport;
        if (artifactManifest) {
            report["status"] = "failed";
            report["artifact_manifest"] = artifactManifest;
            report["artifact_cases"] = Json::array();
            artifactReport.persist = [&]() {
                syncArtifact();
                std::ofstream output(context.outputDirectory / "MaterialBaseline.json");
                output << report.dump(2) << '\n';
            };
        }
        const auto fail = [&](const std::string& message) {
            if (artifactManifest) {
                report["status"] = "failed";
                report["error"] = message;
                syncArtifact();
                if (artifactCaseIndex) { report["artifact_cases"].at(*artifactCaseIndex)["status"] = "failed"; }
            }
            return RHITestResult::fail(message);
        };
        try {
            if (artifactManifest) {
                artifact = std::make_shared<ShaderArtifactFixture>(artifactManifest);
                const char* capture = std::getenv("METALLIC_PIPELINE_RECIPE_DIRECTORY");
                ShaderArtifactFixture::require(capture && *capture, "Artifact correctness requires METALLIC_PIPELINE_RECIPE_DIRECTORY");
                auto installed = ShaderRegistry::instance().installArtifactOverrideForTesting(artifact);
                if (!installed) { return fail("Cannot install scoped shader artifact override"); }
                artifactScope = std::move(*installed);
                syncArtifact();
            }
            const auto root = std::filesystem::path(PROJECT_SOURCE_DIR);
            const char* requested = std::getenv("METALLIC_MATERIAL_BASELINE_CASES");
            const auto casePath = requested ? std::filesystem::path(requested) : root / "LookDev/MaterialSystem/Cases.json";
            const auto fixtures = casePath.parent_path();
            std::ifstream input(casePath);
            const auto cases = Json::parse(input);
            if (artifact) { ShaderArtifactFixture::require(!cases.at("cases").empty(), "Artifact correctness needs at least one case"); }
            const uint32_t frames = cases.at("frames"), warmup = cases.at("warmupFrames"), count = cases.at("timingFrames");
            if (!count || frames <= uint64_t(warmup) + count) {
                return fail("Capture must follow the complete timing window");
            }
            for (const auto& spec : cases.at("cases")) {
                if (artifact) {
                    caseConsumedBefore = artifact->consumed();
                    caseRejectedBefore = artifact->rejected();
                    artifactCaseIndex = report["artifact_cases"].size();
                    report["artifact_cases"].push_back({{"id", spec.at("id")}, {"status", "preparing"}});
                }
                const uint32_t caseFrames = spec.value("frameCount", frames);
                const auto captureFrames = spec.value("captureFrames", std::vector<uint32_t>{});
                if (caseFrames <= uint64_t(warmup) + count ||
                    std::any_of(captureFrames.begin(), captureFrames.end(), [&](uint32_t frame) {
                        return frame >= caseFrames || (frame >= warmup && frame < uint64_t(warmup) + count);
                    })) {
                    return fail("Differential captures must be valid frames outside the timing window");
                }
                const auto id = spec.at("id").get<std::string>();
                std::unique_ptr<ArtifactRecipeDirectoryScope> caseRecipeScope;
                std::filesystem::path caseRecipeDirectory;
                if (artifact) {
                    ShaderArtifactFixture::require(!id.empty() && id != "." && id != ".." &&
                        std::filesystem::path(id).filename() == std::filesystem::path(id), "Artifact case id must be a single path component");
                    caseRecipeDirectory = std::filesystem::path(std::getenv("METALLIC_PIPELINE_RECIPE_DIRECTORY")) / id;
                    caseRecipeScope = std::make_unique<ArtifactRecipeDirectoryScope>(caseRecipeDirectory);
                    report["artifact_cases"].at(*artifactCaseIndex)["recipe_directory"] = caseRecipeDirectory.string();
                }
                RenderGraph graph;
                std::string log;
                if (!loadRenderGraphFromFile(fixtures / spec.at("graph").get<std::string>(), graph, log)) {
                    return fail(log);
                }
                scene::SceneDocument document;
                RenderView view;
                RenderGraphPreviewRenderer preview;
                if (!preview.subsystemHost()->configure<EnvironmentLightingSubsystem>(
                    {.initialDecodeTimeoutMilliseconds = 10000}, log)) { return fail(log); }
                if (!spec.at("scene").is_null()) {
                    if (!document.load(root / spec.at("scene").get<std::string>()) ||
                        !document.documentWarning().empty()) { return fail("Baseline scene load failed"); }
                    const auto graphJSON = Json::parse(serializeRenderGraphToString(graph));
                    bool explicitCamera = false;
                    for (const auto& node : graphJSON.at("nodes")) {
                        const auto& properties = node.at("properties");
                        if (properties.contains("camera") && properties["camera"].contains("eye") &&
                            properties["camera"].contains("center")) { explicitCamera = true; }
                    }
                    if ((!explicitCamera && (document.cameras().empty() || document.cameras()[0].fallback)) ||
                        (!artifact && document.lighting().autoExposure.enabled)) {
                        return fail("Baseline requires an authored camera and manual exposure");
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
                                [](auto slot, auto) { return "imported://" + std::string(slot); }, instance, log)) { return fail(log); }
                            const auto uri = "asset://Material" + std::to_string(index) + ".material";
                            if (!library.save(uri, instance, log) || !document.setMaterialAsset(static_cast<int32_t>(index), uri, assetRoot, log)) {
                                return fail(log);
                            }
                        }
                    }
                    if (spec.contains("materialAsset")) {
                        for (size_t index = 0; index < document.materials().size(); ++index) {
                            if (!document.setMaterialAsset(static_cast<int32_t>(index), spec.at("materialAsset").get<std::string>(),
                                root / spec.at("materialRoot").get<std::string>(), log)) { return fail(log); }
                        }
                    }
                    preview.bindRuntimeScene(&document);
                    preview.setEnvironment(document.environment());
                    auto lighting = document.lighting();
                    if (artifact) { lighting.autoExposure.enabled = false; }
                    if (!preview.setLighting(lighting)) { return fail("Invalid baseline lighting"); }
                } else if (spec.contains("environment")) {
                    const auto& env = spec.at("environment");
                    preview.setEnvironment({.enabled = true, .path = (root / env.at("path").get<std::string>()).string(),
                        .intensity = env.at("intensity"), .rotationDegrees = env.at("rotationDegrees"), .visible = true});
                    scene::LightingSettings lighting;
                    lighting.autoExposure.enabled = false;
                    if (!preview.setLighting(lighting)) { return fail("Invalid fiber lighting"); }
                }
                if (artifact && spec.contains("environment")) {
                    const auto& env = spec.at("environment");
                    preview.setEnvironment({.enabled = true, .path = (root / env.at("path").get<std::string>()).string(),
                        .intensity = env.at("intensity"), .rotationDegrees = env.at("rotationDegrees"), .visible = true});
                }
                Json viewCamera;
                const bool useView = spec.value("useRenderView", false);
                if (useView) {
                    auto* node = graph.findNode("PathTrace");
                    if (!node || !node->properties.contains("camera")) {
                        return fail("Differential RenderView requires an explicit PathTrace camera");
                    }
                    viewCamera = node->properties.at("camera");
                    view.setRenderResolution(spec.at("width"), spec.at("height"));
                    if (!view.setCameraProperties(viewCamera)) { return fail("Invalid RenderView camera"); }
                    node->properties.erase("camera");
                    preview.bindRenderView(&view);
                }
                if (artifact) {
                    ShaderArtifactFixture::require(!spec.at("scene").is_null() && !useView,
                        "Artifact request 146 requires an inspected scene and pass camera (GLOBAL_VIEW=0)");
                    report["artifact_cases"].at(*artifactCaseIndex)["scene_eligibility"] =
                        artifact->inspectScene(document, Json::parse(serializeRenderGraphToString(graph)));
                    artifact->verifyInputsUnchanged();
                }
                if (!preview.initialize(context.enableValidation, true, false)) {
                    return fail("Baseline device initialization failed; no baseline produced");
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
                                if (!node) { return fail("Camera sequence node missing"); }
                                if (useView) {
                                    viewCamera = change.at("camera");
                                    if (!view.setCameraProperties(viewCamera)) { return fail("Invalid moving camera"); }
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
                        return fail(id + ": " + preview.lastLog());
                    }
                    if (artifact && (artifactFailureLog(preview.lastLog()) || artifact->rejected() != caseRejectedBefore)) {
                        return fail(id + ": rejected artifact request or error-material fallback: " + preview.lastLog());
                    }
                    if (differentialCapture) { captureOutput(spec.at("output"), frame); }
                    const auto* environment = preview.subsystemHost()->get<EnvironmentLightingSubsystem>();
                    if (spec.value("requireEnvironment", true) && !environment) {
                        return fail("Baseline environment subsystem missing");
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
                            return fail(id + ": environment not ready for every baseline sample");
                        }
                    }
                    std::vector<RenderGraphExecutionStats> completed;
                    if (!preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); })) {
                        return fail("Baseline timestamp readback failed");
                    }
                    if (frame >= warmup + count) { continue; }
                    if (completed.size() != 1 || !completed[0].gpuTimingAvailable || completed[0].profilingOverflow) {
                        return fail("Missing, ambiguous or overflowed GPU timestamp frame");
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
                if (!half && format != Format::RGBA32Sfloat) { return fail("Baseline must be linear HDR"); }
                const auto& bytes = preview.readbackBytes();
                if (bytes.size() != size_t(width) * height * (half ? 8 : 16)) { return fail("Truncated HDR capture"); }
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
                if (!raw) { return fail("Baseline HDR write failed"); }
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
                        if (!node) { return fail("Guide camera node missing"); }
                        auto camera = useView ? viewCamera : node->properties.at("camera");
                        for (size_t axis = 0; axis < 3; ++axis) {
                            camera["eye"][axis] = camera["eye"][axis].get<float>() + step.at("delta")[axis].get<float>();
                        }
                        if (useView) {
                            viewCamera = camera;
                            if (!view.setCameraProperties(camera)) { return fail("Invalid guide camera"); }
                        } else {
                            graph.setNodeRuntimeProperty(node->id, "camera", camera);
                        }
                    }
                    preview.setRawReadbackEnabled(true);
                    if (!preview.render(graph, width, height, outputName, true)) {
                        return fail(preview.lastLog());
                    }
                    if (artifact && (artifactFailureLog(preview.lastLog()) || artifact->rejected() != caseRejectedBefore)) {
                        return fail(id + ": rejected artifact request or error-material fallback: " + preview.lastLog());
                    }
                    captureOutput(outputName, captureFrame++);
                }
                if (artifact) {
                    ShaderArtifactFixture::require(artifact->consumed() > caseConsumedBefore && artifact->rejected() == caseRejectedBefore,
                        "Artifact was not consumed for this case or a target variant was rejected");
                    ShaderArtifactFixture::require(nonfiniteComponents == 0, "Artifact correctness produced nonfinite HDR");
                    artifact->verifyInputsUnchanged();
                    auto& row = report["artifact_cases"].at(*artifactCaseIndex);
                    row["actual_pipeline_recipes"] = artifact->capturedRecipes(caseRecipeDirectory);
                    row["status"] = "passed";
                    syncArtifact();
                }
                report["cases"].push_back(std::move(result));
                // Keep completed cases and invalid raw images even if a later case fails.
                std::ofstream(context.outputDirectory / "MaterialBaseline.json") << report.dump(2) << '\n';
                if (nonfiniteComponents && cases.value("version", 1) < 2) {
                    return fail("Nonfinite HDR capture");
                }
            }
            std::ofstream output(context.outputDirectory / "MaterialBaseline.json");
            if (artifact) { report["status"] = "passed"; syncArtifact(); }
            output << report.dump(2) << '\n';
            if (!output) { return fail("Baseline report write failed"); }
            return RHITestResult::pass("Capture complete; per-case validHDR is the baseline quality gate");
        } catch (const std::exception& error) {
            if (!artifactManifest) { throw; }
            return fail(std::string("Material baseline artifact/fixture failure: ") + error.what());
        }
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

TEST(ShaderArtifactFixture, NoFiberChecksEveryMaterialIncludingUnreferencedEntries)
{
    std::vector<scene::RenderMaterial> materials(3);
    EXPECT_NO_THROW(ShaderArtifactFixture::inspectMaterials(materials, true));
    materials.back().rtxcrHair = true;
    EXPECT_THROW(ShaderArtifactFixture::inspectMaterials(materials, true), std::runtime_error);
    EXPECT_NO_THROW(ShaderArtifactFixture::inspectMaterials(materials, false));
}

TEST(ShaderArtifactFixture, CustomProgramsRejectBothGeneralAndNoFiberArtifacts)
{
    std::vector<scene::RenderMaterial> materials(2);
    materials.back().valueProgram = "uninspected custom material";
    EXPECT_THROW(ShaderArtifactFixture::inspectMaterials(materials, true), std::runtime_error);
    EXPECT_THROW(ShaderArtifactFixture::inspectMaterials(materials, false), std::runtime_error);
}

TEST(ShaderArtifactFixture, LegacyCleanupPolicyRetainsItsExistingIdentity)
{
    using Json = ShaderArtifactFixture::Json;
    const Json artifact = {{"spirv_postprocess", "dead-branches"}, {"tool_version", "SPIRV-Tools fixture"}};
    EXPECT_NO_THROW(ShaderArtifactFixture::validatePostprocessProvenance(artifact));
}

TEST(ShaderArtifactFixture, PreparedSSAKeepsRuntimeNoneAndExactExternalPassHistory)
{
    using Json = ShaderArtifactFixture::Json;
    Json artifact = {{"spirv_postprocess", "none"}, {"tool_version", ""}, {"bytes", 20}, {"sha256", std::string(64, 'a')},
        {"prepared", {{"transform", "external-cleaned"}, {"input_sha256", std::string(64, 'a')}, {"input_bytes", 20},
            {"passes", Json::array()}, {"ancestry_passes", {"--eliminate-dead-branches", "--simplify-instructions"}},
            {"tool_version", "SPIRV-Tools fixture"}, {"provenance_manifest", {{"file", "Producer/Manifest.json"}, {"sha256", std::string(64, 'b')}}}}}};
    EXPECT_NO_THROW(ShaderArtifactFixture::validatePostprocessProvenance(artifact));
    artifact["prepared"]["transform"] = "external-ssa";
    artifact["prepared"]["passes"] = Json::array({"--ssa-rewrite"});
    artifact["sha256"] = std::string(64, 'c');
    EXPECT_NO_THROW(ShaderArtifactFixture::validatePostprocessProvenance(artifact));
    artifact["spirv_postprocess"] = "dead-branches";
    artifact["tool_version"] = "SPIRV-Tools fixture";
    EXPECT_THROW(ShaderArtifactFixture::validatePostprocessProvenance(artifact), std::runtime_error);
}

TEST(ShaderArtifactFixture, PreparedPolicyRejectsMissingIdentityExtraPassesAndFalseControl)
{
    using Json = ShaderArtifactFixture::Json;
    const Json original = {{"spirv_postprocess", "none"}, {"tool_version", ""}, {"bytes", 20}, {"sha256", std::string(64, 'a')},
        {"prepared", {{"transform", "external-cleaned"}, {"input_sha256", std::string(64, 'a')}, {"input_bytes", 20},
            {"passes", Json::array()}, {"ancestry_passes", {"--eliminate-dead-branches", "--simplify-instructions"}},
            {"tool_version", "SPIRV-Tools fixture"}, {"provenance_manifest", {{"file", "Producer/Manifest.json"}, {"sha256", std::string(64, 'b')}}}}}};
    for (const auto& key : {"input_sha256", "input_bytes", "tool_version", "provenance_manifest"}) {
        Json artifact = original;
        artifact["prepared"].erase(key);
        EXPECT_THROW(ShaderArtifactFixture::validatePostprocessProvenance(artifact), std::exception);
    }
    Json artifact = original;
    artifact["prepared"]["passes"] = Json::array({"--ssa-rewrite", "--compact-ids"});
    EXPECT_THROW(ShaderArtifactFixture::validatePostprocessProvenance(artifact), std::runtime_error);
    artifact = original;
    artifact["sha256"] = std::string(64, 'c');
    EXPECT_THROW(ShaderArtifactFixture::validatePostprocessProvenance(artifact), std::runtime_error);
    artifact = original;
    artifact["tool_version"] = "An optimizer was not run by this runtime";
    EXPECT_THROW(ShaderArtifactFixture::validatePostprocessProvenance(artifact), std::runtime_error);
}

} // namespace
} // namespace metallic::tests
