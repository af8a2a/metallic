#include "RHITest.h"
#include "ShaderArtifactFixture.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Environment/AtmosphereResources.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"
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

class PhysicalCloudFinalReceipt final : public render::IRenderDebugObserver
{
public:
    bool requiresResourceReadback() const override { return false; }
    render::GPUAtmosphereParameters parameters{};
    std::array<std::byte, 320> bytes{};
    bool captured = false, valid = true, completed = false;
    uint32_t executions = 0;
    std::string publication;
    void compiled(debug::DebugValue) override
    {
    }
    void beginExecution(render::Device&, debug::DebugEvidenceStamp, render::RenderSubsystemHost*) override
    {
        ++executions;
    }
    void endExecution(bool success) override
    {
        completed = success;
    }
    void boundary(render::CommandBuffer&, std::string_view, uint32_t, std::string_view,
        std::span<const render::DebugResourceBinding> resources, const debug::DebugValue&) override
    {
        for (const auto& binding : resources) {
            if (binding.metadata.value("environmentSource", std::string{}) != "PhysicalAtmosphere" ||
                !binding.id.ends_with(".parameters")) { continue; }
            const auto prefix = binding.id.substr(0, binding.id.size() - std::string_view("parameters").size());
            const bool shadow = std::any_of(resources.begin(), resources.end(), [&](const auto& other) {
                return other.id == prefix + "cloudShadow" && other.texture != nullptr &&
                    other.metadata.value("environmentSource", std::string{}) == "PhysicalAtmosphere";
            });
            if (!shadow || !binding.buffer || binding.offset != 0 || binding.buffer->desc().size != bytes.size() ||
                binding.buffer->desc().memoryLocation != render::MemoryLocation::HostUpload) { valid = false; continue; }
            const void* mapped = binding.buffer->map();
            if (!mapped) { valid = false; continue; }
            std::array<std::byte, 320> next;
            std::memcpy(next.data(), mapped, next.size());
            binding.buffer->unmap();
            if (captured && bytes != next) { valid = false; }
            bytes = next; std::memcpy(&parameters, bytes.data(), bytes.size());
            const auto& p = parameters;
            valid &= p.settings[1] > 0.5f && p.cloudLayer[2] > 0.0f && p.cloudLayer[3] > 0.0f &&
                p.cloudOptics[0] > 0.0f && p.cloudOptics[3] > 0.5f;
            captured = true; publication = prefix;
        }
    }
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
        bool nsightTraceRequested = false;
        ArtifactReportGuard shaderDebugRestore;
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
        const auto persistReport = [&]() {
            syncArtifact();
            std::ofstream output(context.outputDirectory / "MaterialBaseline.json");
            output << report.dump(2) << '\n';
        };
        ArtifactReportGuard artifactReport;
        if (artifactManifest) {
            report["status"] = "failed";
            report["artifact_manifest"] = artifactManifest;
            report["artifact_cases"] = Json::array();
            artifactReport.persist = persistReport;
        }
        const auto fail = [&](const std::string& message) {
            if (artifactManifest || nsightTraceRequested) {
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
            const bool synchronizationValidation = cases.value("synchronizationValidation", false);
            if (synchronizationValidation && !context.enableValidation) {
                return fail("Synchronization validation requires a correctness run with validation enabled");
            }
            report["synchronizationValidation"] = synchronizationValidation;
            if (artifact) { ShaderArtifactFixture::require(!cases.at("cases").empty(), "Artifact correctness needs at least one case"); }
            const uint32_t frames = cases.at("frames"), warmup = cases.at("warmupFrames"), count = cases.at("timingFrames");
            std::optional<uint32_t> nsightTraceFrame;
            if (cases.contains("nsightTraceFrame")) {
                nsightTraceRequested = true;
                artifactReport.persist = persistReport;
                report["status"] = "failed";
                report["diagnostic"] = true;
                report["performanceEligible"] = false;
                const auto& frame = cases.at("nsightTraceFrame");
                if (!frame.is_number_unsigned() || frame.get<uint64_t>() == 0 || frame.get<uint64_t>() >= warmup ||
                    cases.at("cases").size() != 1 || context.enableValidation || artifact || assetRoundTrip_) {
                    return fail("Nsight trace requires one ordinary case, validation disabled and 0 < nsightTraceFrame < warmupFrames");
                }
                nsightTraceFrame = frame.get<uint32_t>();
                const auto previous = slangShaderDebugMode();
                shaderDebugRestore.persist = [previous]() { setSlangShaderDebugMode(previous); };
                setSlangShaderDebugMode(SlangShaderDebugMode::CaptureSymbols);
                if (slangShaderOptimizationLevel(SlangShaderDesc{}) != SlangShaderOptimizationLevel::O1) {
                    return fail("Nsight trace requires optimized O1 capture symbols");
                }
                report["shaderDebugMode"] = "capture";
                report["shaderOptimization"] = "O1";
            }
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
                std::optional<uint32_t> freezeStreamingFrame;
                if (spec.contains("freezeStreamingFrame")) {
                    const auto& requestedFrame = spec.at("freezeStreamingFrame");
                    if (!requestedFrame.is_number_unsigned() || requestedFrame.get<uint64_t>() == 0 ||
                        requestedFrame.get<uint64_t>() >= warmup) {
                        return fail("Streaming freeze requires 0 < freezeStreamingFrame < warmupFrames");
                    }
                    freezeStreamingFrame = requestedFrame.get<uint32_t>();
                }
                const auto freezeStreamingNode = spec.value("freezeStreamingNode", std::string("PathTrace"));
                if (caseFrames <= uint64_t(warmup) + count ||
                    std::any_of(captureFrames.begin(), captureFrames.end(), [&](uint32_t frame) {
                        return frame >= caseFrames || (frame >= warmup && frame < uint64_t(warmup) + count);
                    })) {
                    return fail("Differential captures must be valid frames outside the timing window");
                }
                if (nsightTraceFrame && std::any_of(captureFrames.begin(), captureFrames.end(), [&](uint32_t frame) {
                        return frame == *nsightTraceFrame || frame == *nsightTraceFrame - 1;
                    })) {
                    return fail("Nsight trace frame and preceding frame must not read back images");
                }
                const auto id = spec.at("id").get<std::string>();
                std::unique_ptr<ArtifactRecipeDirectoryScope> caseRecipeScope;
                std::filesystem::path caseRecipeDirectory;
                const char* recipeDirectory = std::getenv("METALLIC_PIPELINE_RECIPE_DIRECTORY");
                if (recipeDirectory && *recipeDirectory) {
                    ShaderArtifactFixture::require(!id.empty() && id != "." && id != ".." &&
                        std::filesystem::path(id).filename() == std::filesystem::path(id), "Recipe case id must be a single path component");
                    caseRecipeDirectory = std::filesystem::path(recipeDirectory) / id;
                    caseRecipeScope = std::make_unique<ArtifactRecipeDirectoryScope>(caseRecipeDirectory);
                    if (artifact) { report["artifact_cases"].at(*artifactCaseIndex)["recipe_directory"] = caseRecipeDirectory.string(); }
                }
                RenderGraph graph;
                std::string log;
                if (!loadRenderGraphFromFile(fixtures / spec.at("graph").get<std::string>(), graph, log)) {
                    return fail(log);
                }
                if (freezeStreamingFrame) {
                    const auto* node = graph.findNode(freezeStreamingNode);
                    if (!node || node->properties.value("benchmarkFreezeStreaming", false) ||
                        node->runtimeProperties.value("benchmarkFreezeStreaming", false)) {
                        return fail("Streaming freeze needs an initially unfrozen node: " + freezeStreamingNode);
                    }
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
                    // Explicit benchmark-only instance edits preserve the authored
                    // mesh, OpenPBR surface inputs, textures, lighting and camera.
                    for (const auto& override : spec.value("materialOverrides", Json::array())) {
                        const auto index = override.at("index").get<int32_t>();
                        if (index < 0 || size_t(index) >= document.materials().size()) {
                            return fail("Baseline material override index out of range");
                        }
                        auto material = document.materials()[index];
                        if (override.contains("alphaMode")) { material.alphaMode = override.at("alphaMode").get<std::string>(); }
                        if (override.contains("alphaCutoff")) { material.alphaCutoff = override.at("alphaCutoff").get<float>(); }
                        if (override.contains("valueProgram")) {
                            material.valueProgram = override.at("valueProgram").is_null() ? std::string{} : override.at("valueProgram").dump();
                        }
                        if (override.contains("valueParameters")) {
                            material.valueParameters = override.at("valueParameters").get<std::array<float, 16>>();
                        }
                        if (!document.setMaterialProperties(index, material)) {
                            return fail("Baseline material override failed");
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
                    const auto cameraNode = spec.value("renderViewCameraNode", std::string("PathTrace"));
                    auto* node = graph.findNode(cameraNode);
                    if (!node || !node->properties.contains("camera")) {
                        return fail("Differential RenderView requires an explicit camera: " + cameraNode);
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
                if (!preview.initialize(context.enableValidation, true, false, synchronizationValidation)) {
                    return fail("Baseline device initialization failed; no baseline produced");
                }
                const uint32_t width = spec.at("width"), height = spec.at("height");
                Json result = spec;
                if (spec.contains("materialOverrides")) {
                    result["resolvedMaterials"] = Json::array();
                    for (const auto& material : document.materials()) {
                        result["resolvedMaterials"].push_back({{"name", material.name}, {"alphaMode", material.alphaMode},
                            {"alphaCutoff", material.alphaCutoff}, {"valueProgram", material.valueProgram},
                            {"valueParameters", material.valueParameters}});
                    }
                    result["frameSequence"] = "fresh preview; frame index starts at zero; static camera; capture follows timing";
                }
                ArtifactReportGuard traceCaseReport;
                if (nsightTraceFrame) {
                    result["sdkTrace"] = {{"frame", *nsightTraceFrame}, {"frames", 1},
                        {"complete", false}, {"artifactVerified", false}};
                    traceCaseReport.persist = [&]() { report["cases"].push_back(result); };
                }
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
                auto cloudReceipt = spec.value("physicalCloudReceipt", false) ? std::make_unique<PhysicalCloudFinalReceipt>() : nullptr;
                for (uint32_t frame = 0; frame < caseFrames; ++frame) {
                    if (freezeStreamingFrame && frame == *freezeStreamingFrame) {
                        const auto* node = graph.findNode(freezeStreamingNode);
                        if (!node || !graph.setNodeRuntimeProperty(node->id, "benchmarkFreezeStreaming", true)) {
                            return fail("Cannot freeze streaming after priming: " + freezeStreamingNode);
                        }
                        result["streamingFreeze"] = {{"node", freezeStreamingNode}, {"appliedFrame", frame}};
                    }
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
                    const bool proofFrame = cloudReceipt && frame + 1 == caseFrames;
                    const auto rendered = [&]() {
                        if (proofFrame) { preview.setDebugObserver(cloudReceipt.get()); }
                        struct Reset
                        {
                            RenderGraphPreviewRenderer& preview;
                            bool active;
                            ~Reset()
                            {
                                if (active) { preview.setDebugObserver(nullptr); }
                            }
                        } reset{preview, proofFrame};
                        bool traceActive = false;
                        ArtifactReportGuard traceCleanup;
                        if (nsightTraceFrame && frame == *nsightTraceFrame) {
                            traceCleanup.persist = [&]() {
                                if (!traceActive) { return; }
                                // A failed render may return before its normal submission drain.
                                const auto* host = preview.subsystemHost();
                                if (!host || !host->device() || !host->device()->waitIdle()) {
                                    result["sdkTrace"]["cleanupDrainFailed"] = true;
                                }
                                std::string error;
                                if (!profiling::endExternalNsightGpuTrace(error)) {
                                    result["sdkTrace"]["cleanupError"] = error;
                                }
                            };
                            // The preceding non-readback render returned after draining submissions.
                            std::string error;
                            if (!profiling::beginExternalNsightGpuTrace(error)) {
                                result["sdkTrace"]["error"] = error;
                                throw std::runtime_error(error);
                            }
                            traceActive = true;
                        }
                        auto rendered = preview.render(graph, width, height, spec.at("output").get<std::string>(), capture);
                        if (traceActive && rendered) {
                            // Successful preview.render also drains the captured frame before returning.
                            std::string error;
                            if (!profiling::endExternalNsightGpuTrace(error)) {
                                result["sdkTrace"]["error"] = error;
                                throw std::runtime_error(error);
                            }
                            traceActive = false;
                            result["sdkTrace"]["complete"] = true;
                        }
                        return rendered;
                    }();
                    if (!rendered) { return fail(id + ": " + preview.lastLog()); }
                    if (artifactFailureLog(preview.lastLog())) {
                        return fail(id + ": material compilation fallback: " + preview.lastLog());
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
                if (cloudReceipt) {
                    const auto* vbuffer = graph.findNode("VBuffer");
                    if (!vbuffer || !cloudReceipt->captured || !cloudReceipt->valid || !cloudReceipt->completed || cloudReceipt->executions != 1) {
                        return fail("Missing/invalid final physical cloud publication receipt");
                    }
                    const auto& camera = vbuffer->properties.at("camera");
                    const auto eye = camera.at("eye").get<std::array<float, 3>>();
                    if (camera.value("projection", std::string{}) != "orthographic" || camera.value("orthoHeight", 0.0f) != 3.0f ||
                        eye != std::array<float,3>{0,2200,4} || camera.at("center") != Json::array({0,2200,0}) ||
                        camera.at("up") != Json::array({0,1,0}) || width != 128 || height != 128 ||
                        !std::equal(eye.begin(), eye.end(), cloudReceipt->parameters.observerWorldTop.begin())) { return fail("Orthographic actual-origin receipt mismatch"); }
                    const auto& p = cloudReceipt->parameters;
                    const double altitude = std::hypot(double(p.observerPlanetBottom[0]), double(p.observerPlanetBottom[1]), double(p.observerPlanetBottom[2])) - p.observerPlanetBottom[3];
                    if (p.observerPlanetBottom[3] != 6360.0f || p.observerWorldTop[3] != 6460.0f ||
                        p.cloudLayer != std::array<float,4>{1.5f,4.0f,1.0f,1.0f} || !(altitude > 2.19 && altitude < 2.21)) { return fail("Camera must be inside active spherical cloud layer"); }
                    const auto file = id + "-physical-parameters.bin";
                    std::ofstream receipt(context.outputDirectory / file, std::ios::binary);
                    receipt.write(reinterpret_cast<const char*>(cloudReceipt->bytes.data()), cloudReceipt->bytes.size());
                    if (!receipt) { return fail("Cannot write physical publication receipt"); }
                    result["physicalCloudReceipt"] = {{"frame", caseFrames - 1}, {"observedTimedFrames", 0}, {"parameterFile", file},
                        {"bytes", 320}, {"memoryLocation", "HostUpload"}, {"environmentSource", "PhysicalAtmosphere"}, {"role", "PhysicalAtmosphere.parameters"},
                        {"publication", cloudReceipt->publication}, {"cloudLayer", cloudReceipt->parameters.cloudLayer},
                        {"cloudOptics", p.cloudOptics}, {"observerWorldTop", p.observerWorldTop}, {"observerPlanetBottom", p.observerPlanetBottom},
                        {"observerAltitudeKm", altitude}, {"resolvedCamera", camera}, {"cornerSampleOrigin", {-1.48828125,2201.48828125,4.0}},
                        {"originBounds", {{"minimum", {-1.48828125,2198.51171875,4.0}}, {"maximum", {1.48828125,2201.48828125,4.0}}}},
                        {"actualOriginCondition", true}, {"projectionProofScope", "resolved graph + actual observer packet + unchanged ScenePathMath; not GPU branch counter"}};
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
                traceCaseReport.persist = {};
                report["cases"].push_back(std::move(result));
                // Keep completed cases and invalid raw images even if a later case fails.
                std::ofstream(context.outputDirectory / "MaterialBaseline.json") << report.dump(2) << '\n';
                if (nonfiniteComponents && cases.value("version", 1) < 2) {
                    return fail("Nonfinite HDR capture");
                }
            }
            std::ofstream output(context.outputDirectory / "MaterialBaseline.json");
            if (artifact || nsightTraceRequested) { report["status"] = "passed"; syncArtifact(); }
            output << report.dump(2) << '\n';
            if (!output) { return fail("Baseline report write failed"); }
            return RHITestResult::pass("Capture complete; per-case validHDR is the baseline quality gate");
        } catch (const std::exception& error) {
            if (!artifactManifest && !nsightTraceRequested) { throw; }
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
