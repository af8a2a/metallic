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
#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "../../Shaders/Interop/StreamCLASTessellation.h"
#include "../../Shaders/Interop/ShadowRayCapture.h"

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

class TessellationFinalReceipt final : public render::IRenderDebugObserver
{
public:
    TessellationFinalReceipt(bool captureSizes, bool captureGeometry, bool captureShadowRays)
        : captureSizes_(captureSizes), captureGeometry_(captureGeometry), captureShadowRays_(captureShadowRays) {}
    bool requiresResourceReadback() const override { return false; }
    bool captured = false, completed = false;
    bool afterPassSeen = false, bindingSeen = false;
    bool sizesCaptured = false, boundsCaptured = false, boundsSeen = false;
    bool leavesCaptured = false, verticesCaptured = false;
    bool instancesCaptured = false, tlasInstancesCaptured = false, shadowRaysCaptured = false, shadowParametersCaptured = false;
    uint64_t templatePoolGeneration = 0;
    uint32_t generatedBaseClusterId = 0, instanceCount = 0;
    std::unique_ptr<render::Buffer> buffer, sizesBuffer, boundsBuffer;
    std::unique_ptr<render::Buffer> leavesBuffer, verticesBuffer, tlasInstancesBuffer, shadowRaysBuffer;
    std::vector<render::MeshletStreamGPUInstance> instances;
    render::ScreenSpaceShadowParameters shadowParameters{};
    debug::DebugValue shadowMetadata;
    void compiled(debug::DebugValue) override {}
    void beginExecution(render::Device& device, debug::DebugEvidenceStamp, render::RenderSubsystemHost*) override
    {
        captured = completed = afterPassSeen = bindingSeen = false;
        sizesCaptured = false;
        boundsCaptured = boundsSeen = false;
        leavesCaptured = verticesCaptured = false;
        instancesCaptured = tlasInstancesCaptured = shadowRaysCaptured = shadowParametersCaptured = false;
        instances.clear();
        shadowMetadata = debug::DebugValue::object();
        sizesBuffer.reset();
        boundsBuffer.reset();
        leavesBuffer.reset();
        verticesBuffer.reset();
        tlasInstancesBuffer.reset();
        shadowRaysBuffer.reset();
        device_ = &device;
        auto result = device.createBuffer({.size = 24, .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback,
            .queueAccess = render::QueueAccessBits::Graphics | render::QueueAccessBits::Compute});
        buffer = result ? std::move(*result) : nullptr;
    }
    void endExecution(bool success) override { completed = success; }
    void boundary(render::CommandBuffer& commands, std::string_view checkpoint, uint32_t,
        std::string_view pass, std::span<const render::DebugResourceBinding> resources, const debug::DebugValue& values) override
    {
        using namespace render;
        if (captureShadowRays_ && checkpoint == "ShadowRayCapture" && pass == "Shadows") {
            shadowMetadata = values;
            for (const auto& binding : resources) {
                if (binding.id == "shadowParameters") {
                    const void* mapped = binding.buffer->map();
                    if (!mapped) { continue; }
                    std::memcpy(&shadowParameters, mapped, sizeof(shadowParameters));
                    binding.buffer->unmap();
                    shadowParametersCaptured = true;
                } else if (binding.id == "shadowRays") {
                    shadowRaysCaptured = copy(commands, binding, binding.buffer->desc().size, shadowRaysBuffer);
                }
            }
            return;
        }
        if (checkpoint != "AfterPass" || pass != "VBuffer") { return; }
        afterPassSeen = true;
        if (captured || !buffer) { return; }
        for (const auto& binding : resources) {
            if (!binding.buffer) { continue; }
            const bool counts = binding.id.ends_with(".tessellationCounts");
            const bool sizes = captureSizes_ && binding.id.ends_with(".tessellationSizes");
            const bool bounds = captureSizes_ && binding.id.ends_with(".tessellationBounds");
            const bool leaves = captureGeometry_ && binding.id.ends_with(".tessellationLeaves");
            const bool vertices = captureGeometry_ && binding.id.ends_with(".tessellationVertices");
            const bool instanceData = captureGeometry_ && binding.id.ends_with(".tessellationInstanceData");
            const bool tlasInstances = captureGeometry_ && binding.id.ends_with(".tessellationTLASInstances");
            if (instanceData) {
                instanceCount = binding.metadata.at("sceneInstanceCount").get<uint32_t>();
                instances.resize(instanceCount);
                const void* mapped = binding.buffer->map();
                if (!mapped) { continue; }
                std::memcpy(instances.data(), static_cast<const std::byte*>(mapped) + binding.offset,
                    instances.size() * sizeof(instances[0]));
                binding.buffer->unmap();
                instancesCaptured = true;
                continue;
            }
            if (!counts && !sizes && !bounds && !leaves && !vertices && !tlasInstances) { continue; }
            const uint64_t bytes = counts || bounds ? 24 : binding.buffer->desc().size - binding.offset;
            auto& destination = counts ? buffer : bounds ? boundsBuffer : leaves ? leavesBuffer :
                vertices ? verticesBuffer : tlasInstances ? tlasInstancesBuffer : sizesBuffer;
            if (counts) {
                bindingSeen = true;
                templatePoolGeneration = binding.metadata.at("templatePoolGeneration").get<uint64_t>();
                generatedBaseClusterId = binding.metadata.at("generatedBaseClusterId").get<uint32_t>();
            }
            else {
                boundsSeen |= bounds;
            }
            (counts ? captured : bounds ? boundsCaptured : leaves ? leavesCaptured :
                vertices ? verticesCaptured : tlasInstances ? tlasInstancesCaptured : sizesCaptured) =
                copy(commands, binding, bytes, destination);
        }
    }
private:
    bool copy(render::CommandBuffer& commands, const render::DebugResourceBinding& binding, uint64_t bytes,
        std::unique_ptr<render::Buffer>& destination)
    {
        using namespace render;
        if (!destination) {
            auto allocation = device_->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute});
            if (!allocation) { return false; }
            destination = std::move(*allocation);
        }
        const MemoryBarrierDesc before{
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead | AccessBits::TransferWrite}};
        const MemoryBarrierDesc after{
            .before = {PipelineStageBits::Transfer, AccessBits::TransferRead | AccessBits::TransferWrite},
            .after = {PipelineStageBits::AllCommands | PipelineStageBits::Host,
                AccessBits::MemoryRead | AccessBits::MemoryWrite | AccessBits::HostRead}};
        auto result = commands.synchronize({.memory = {&before, 1}});
        if (result) { result = commands.copyBuffer(*binding.buffer->slice({binding.offset, bytes}), *destination->slice()); }
        if (result) { result = commands.synchronize({.memory = {&after, 1}}); }
        return bool(result);
    }
    bool captureSizes_;
    bool captureGeometry_;
    bool captureShadowRays_;
    render::Device* device_ = nullptr;
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
                const bool freezeResidency = spec.contains("freezeResidencyFrame");
                if (freezeResidency && spec.contains("freezeStreamingFrame")) {
                    return fail("Choose either raster snapshot freeze or source residency freeze");
                }
                const auto freezeProperty = freezeResidency ? "benchmarkFreezeResidency" : "benchmarkFreezeStreaming";
                if (freezeResidency || spec.contains("freezeStreamingFrame")) {
                    const auto& requestedFrame = spec.at(freezeResidency ? "freezeResidencyFrame" : "freezeStreamingFrame");
                    if (!requestedFrame.is_number_unsigned() || requestedFrame.get<uint64_t>() == 0 ||
                        requestedFrame.get<uint64_t>() >= warmup) {
                        return fail("Streaming freeze requires 0 < freeze frame < warmupFrames");
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
                    if (!node || node->properties.value(freezeProperty, false) ||
                        node->runtimeProperties.value(freezeProperty, false)) {
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
                        (!artifact && !spec.value("manualExposure", false) && document.lighting().autoExposure.enabled)) {
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
                if (spec.value("manualExposure", false)) {
                    auto lighting = spec.at("scene").is_null() ? scene::LightingSettings{} : document.lighting();
                    lighting.autoExposure.enabled = false;
                    lighting.exposureEV100 = spec.value("exposureEV100", 0.0f);
                    if (!preview.setLighting(lighting)) { return fail("Invalid explicit baseline exposure"); }
                }
                if (spec.contains("lighting")) {
                    const auto& sun = spec.at("lighting").at("sun");
                    const auto direction = sun.at("direction").get<std::array<float, 3>>();
                    const auto color = sun.value("color", std::array<float, 3>{1, 1, 1});
                    environment::WorldEnvironment environment;
                    environment.sun.enabled = true;
                    environment.sun.direction = {direction[0], direction[1], direction[2]};
                    environment.sun.color = {color[0], color[1], color[2]};
                    environment.sun.illuminance = sun.at("illuminance");
                    environment.sun.angularRadius = sun.at("angularRadius");
                    if (!preview.setWorldEnvironment(environment)) { return fail("Invalid explicit baseline sun"); }
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
                const bool captureShadowRays = spec.value("captureShadowRays", false);
                if (captureShadowRays) {
                    const auto* shadows = graph.findNode("Shadows");
                    const auto outputs = spec.value("captureOutputs", std::vector<std::string>{});
                    if (artifact || nsightTraceFrame || !spec.value("captureTessellationGeometry", false) || !shadows ||
                        shadows->properties.value("sigmaDenoise", true) ||
                        shadows->properties.value("diagnosticSampleFrame", UINT32_MAX) != 0u ||
                        shadows->properties.value("diagnosticCaptureRays", false) ||
                        spec.at("output") != "Deferred.color" || spec.contains("cameraFrames") || spec.contains("guideCameraStep") ||
                        std::find(outputs.begin(), outputs.end(), "Shadows.shadow") == outputs.end()) {
                        return fail("Shadow-ray capture requires final geometry, fixed sample frame 0, SIGMA off, Deferred.color and an ordinary Shadows.shadow capture");
                    }
                }
                Json shadowManifest;
                std::vector<uint8_t> capturedShadowVisibility;
                if (spec.value("captureTessellationGeometry", false) &&
                    (!spec.value("captureTessellationNativeSizes", false) || !spec.value("tessellationReceipt", false))) {
                    return fail("Geometry encoding diagnostics require final tessellation counts and native sizes");
                }
                auto tessellationReceipt = spec.value("tessellationReceipt", false) ?
                    std::make_unique<TessellationFinalReceipt>(spec.value("captureTessellationNativeSizes", false),
                        spec.value("captureTessellationGeometry", false), captureShadowRays) : nullptr;
                if (cloudReceipt && tessellationReceipt) { return fail("Choose one final publication receipt"); }
                for (uint32_t frame = 0; frame < caseFrames; ++frame) {
                    if (freezeStreamingFrame && frame == *freezeStreamingFrame) {
                        const auto* node = graph.findNode(freezeStreamingNode);
                        if (!node || !graph.setNodeRuntimeProperty(node->id, freezeProperty, true)) {
                            return fail("Cannot freeze streaming after priming: " + freezeStreamingNode);
                        }
                        result["streamingFreeze"] = {{"node", freezeStreamingNode}, {"appliedFrame", frame}, {"property", freezeProperty}};
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
                    const bool proofFrame = (cloudReceipt || tessellationReceipt) && frame + 1 == caseFrames;
                    const auto rendered = [&]() {
                        const auto* shadows = captureShadowRays && proofFrame ? graph.findNode("Shadows") : nullptr;
                        if (shadows && !graph.setNodeRuntimeProperty(shadows->id, "diagnosticCaptureRays", true)) {
                            throw std::runtime_error("Cannot enable final shadow-ray capture");
                        }
                        if (proofFrame) { preview.setDebugObserver(cloudReceipt ?
                            static_cast<IRenderDebugObserver*>(cloudReceipt.get()) : tessellationReceipt.get()); }
                        struct Reset
                        {
                            RenderGraphPreviewRenderer& preview;
                            RenderGraph& graph;
                            bool active;
                            std::optional<uint32_t> shadowPass;
                            ~Reset()
                            {
                                if (active) { preview.setDebugObserver(nullptr); }
                                if (shadowPass) { graph.setNodeRuntimeProperty(*shadowPass, "diagnosticCaptureRays", false); }
                            }
                        } reset{preview, graph, proofFrame, shadows ? std::optional<uint32_t>(shadows->id) : std::nullopt};
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
                        {"nodes", Json::array()}, {"streaming", Json::array()}};
                    for (const auto& stream : stats.streaming) {
                        timing["streaming"].push_back({{"passName", stream.passName}, {"assetPath", stream.assetPath},
                            {"diagnosticSourceIdentity", stream.diagnosticSourceIdentity},
                            {"tessellationIdentity", stream.tessellationIdentity},
                            {"originalTopologyHash", stream.originalTopologyHash},
                            {"primitiveMetadataHash", stream.primitiveMetadataHash},
                            {"residentPages", stream.residentPages}, {"textureResidentBytes", stream.textureResidentBytes}});
                    }
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
                if (tessellationReceipt) {
                    if (!tessellationReceipt->captured || !tessellationReceipt->completed) {
                        const auto& receipt = *tessellationReceipt;
                        return fail("Missing final production CLAS tessellation count receipt: " + Json{
                            {"readbackAllocated", receipt.buffer != nullptr}, {"afterPassSeen", receipt.afterPassSeen},
                            {"bindingSeen", receipt.bindingSeen}, {"copyRecorded", receipt.captured},
                            {"executionCompleted", receipt.completed}}.dump());
                    }
                    auto& buffer = *tessellationReceipt->buffer;
                    const void* mapped = buffer.map();
                    if (!mapped) { return fail("Cannot map CLAS tessellation receipt"); }
                    buffer.invalidate();
                    std::array<uint32_t, 6> statistics;
                    std::memcpy(statistics.data(), mapped, sizeof(statistics));
                    buffer.unmap();
                    const std::array<uint32_t, 4> counts{statistics[0], statistics[1], statistics[2], statistics[3]};
                    result["tessellationCounts"] = counts;
                    result["templateBoundFallbacks"] = statistics[4];
                    result["templateBoundFallbackObserved"] = statistics[5] != 0;
                    result["templatePoolGeneration"] = tessellationReceipt->templatePoolGeneration;
                    if (counts[0] == 0 || counts[1] == 0 || counts[2] != 0 || counts[3] != 0) {
                        return fail("CLAS tessellation benchmark produced empty or rejected work");
                    }
                    if (spec.value("captureTessellationNativeSizes", false)) {
                        if (!tessellationReceipt->sizesCaptured || !tessellationReceipt->sizesBuffer) {
                            return fail("Missing final native CLAS size receipt");
                        }
                        auto& sizesBuffer = *tessellationReceipt->sizesBuffer;
                        if (uint64_t(counts[0]) * sizeof(uint32_t) > sizesBuffer.desc().size) {
                            return fail("Native CLAS count exceeds size receipt capacity");
                        }
                        const void* sizesMapped = sizesBuffer.map();
                        if (!sizesMapped) { return fail("Cannot map native CLAS size receipt"); }
                        sizesBuffer.invalidate();
                        std::vector<uint32_t> sizes(counts[0]);
                        std::memcpy(sizes.data(), sizesMapped, sizes.size() * sizeof(uint32_t));
                        sizesBuffer.unmap();
                        uint64_t total = 0;
                        for (const auto size : sizes) {
                            if (!size) { return fail("Native CLAS size receipt contains an empty cluster"); }
                            total += size;
                        }
                        const auto file = id + "-CLASSizes.bin";
                        const auto bytes = sizes.size() * sizeof(uint32_t);
                        std::ofstream receipt(context.outputDirectory / file, std::ios::binary);
                        receipt.write(reinterpret_cast<const char*>(sizes.data()), bytes);
                        if (!receipt) { return fail("Cannot write native CLAS size receipt"); }
                        result["nativeCLASSizeReceipt"] = {{"file", file}, {"count", sizes.size()}, {"bytes", bytes},
                            {"totalBytes", total}, {"minBytes", *std::min_element(sizes.begin(), sizes.end())},
                            {"maxBytes", *std::max_element(sizes.begin(), sizes.end())}};
                        if (tessellationReceipt->boundsSeen) {
                            if (!tessellationReceipt->boundsCaptured) { return fail("Missing native template bound receipt"); }
                            auto& bounds = *tessellationReceipt->boundsBuffer;
                            const void* mappedBounds = bounds.map();
                            if (!mappedBounds) { return fail("Cannot map native template bound receipt"); }
                            bounds.invalidate();
                            std::array<float, 6> values;
                            std::memcpy(values.data(), mappedBounds, sizeof(values));
                            bounds.unmap();
                            const auto boundsFile = id + "-CLASBounds.bin";
                            std::ofstream output(context.outputDirectory / boundsFile, std::ios::binary);
                            output.write(reinterpret_cast<const char*>(values.data()), sizeof(values));
                            if (!output) { return fail("Cannot write native template bound receipt"); }
                            result["nativeCLASBoundsReceipt"] = {{"file", boundsFile}, {"bytes", sizeof(values)}, {"values", values}};
                        }
                    }
                    if (spec.value("captureTessellationGeometry", false)) {
                        const auto& receipt = *tessellationReceipt;
                        if (!receipt.leavesCaptured || !receipt.verticesCaptured || !receipt.instancesCaptured || !receipt.tlasInstancesCaptured) {
                            return fail("Missing actual CLAS geometry encoding receipt");
                        }
                        const auto read = [&](Buffer& source, size_t bytes, void* destination) {
                            if (bytes > source.desc().size) { throw std::runtime_error("CLAS geometry receipt exceeds buffer"); }
                            const void* data = source.map();
                            if (!data) { throw std::runtime_error("Cannot map CLAS geometry receipt"); }
                            source.invalidate();
                            std::memcpy(destination, data, bytes);
                            source.unmap();
                        };
                        std::vector<StreamCLASTessellationLeaf> leaves(counts[0]);
                        std::vector<StreamCLASTessellationVertex> vertices(size_t(counts[0]) * METALLIC_TESS_VERTICES_PER_PATCH);
                        read(*receipt.leavesBuffer, leaves.size() * sizeof(leaves[0]), leaves.data());
                        read(*receipt.verticesBuffer, vertices.size() * sizeof(vertices[0]), vertices.data());
                        for (size_t patch = 0; patch < leaves.size(); ++patch) {
                            if (leaves[patch].vertexCount > METALLIC_TESS_VERTICES_PER_PATCH) {
                                return fail("Invalid live vertex count in geometry encoding receipt");
                            }
                            // Do not export uninitialized, unused GPU allocation contents.
                            // Every live vertex word is preserved exactly, including its domain.
                            std::fill(vertices.begin() + patch * METALLIC_TESS_VERTICES_PER_PATCH + leaves[patch].vertexCount,
                                vertices.begin() + (patch + 1) * METALLIC_TESS_VERTICES_PER_PATCH, StreamCLASTessellationVertex{});
                        }
                        std::vector<uint32_t> sizes(counts[0]);
                        read(*receipt.sizesBuffer, sizes.size() * sizeof(sizes[0]), sizes.data());
                        std::vector<RayTracingGPUInstance> tlasInstances(receipt.instanceCount);
                        read(*receipt.tlasInstancesBuffer, tlasInstances.size() * sizeof(tlasInstances[0]), tlasInstances.data());
                        Json geometry = {{"schemaVersion", 1}, {"patchCount", counts[0]},
                            {"verticesPerPatch", METALLIC_TESS_VERTICES_PER_PATCH},
                            {"trianglesPerPatch", METALLIC_TESS_TRIANGLES_PER_PATCH},
                            {"leafStride", sizeof(leaves[0])}, {"vertexStride", sizeof(vertices[0])},
                            {"instanceCount", receipt.instanceCount}, {"instanceStride", sizeof(MeshletStreamGPUInstance)},
                            {"tlasInstanceStride", sizeof(RayTracingGPUInstance)}, {"generatedBaseClusterId", receipt.generatedBaseClusterId},
                            {"unusedVertexSlotsZeroed", true}, {"sha256", Json::object()},
                            {"frame", caseFrames - 1}, {"scope", "final untimed frame; actual live GPU vertices unchanged"}};
                        const auto save = [&](const char* key, const char* suffix, const void* data, size_t bytes) {
                            const auto file = id + suffix;
                            std::ofstream output(context.outputDirectory / file, std::ios::binary);
                            output.write(static_cast<const char*>(data), std::streamsize(bytes));
                            if (!output) { throw std::runtime_error("Cannot write CLAS geometry encoding receipt"); }
                            geometry[key] = file;
                            geometry["sha256"][file] = debug::debugSha256(std::span(static_cast<const uint8_t*>(data), bytes));
                        };
                        save("leavesFile", "-CLASLeaves.bin", leaves.data(), leaves.size() * sizeof(leaves[0]));
                        save("verticesFile", "-CLASVertices.bin", vertices.data(), vertices.size() * sizeof(vertices[0]));
                        save("countsFile", "-CLASCounts.bin", statistics.data(), sizeof(statistics));
                        save("sizesFile", "-CLASGeometrySizes.bin", sizes.data(), sizes.size() * sizeof(sizes[0]));
                        save("instancesFile", "-CLASInstances.bin", receipt.instances.data(), receipt.instances.size() * sizeof(receipt.instances[0]));
                        save("tlasInstancesFile", "-CLASTLASInstances.bin", tlasInstances.data(), tlasInstances.size() * sizeof(tlasInstances[0]));
                        for (const auto& stream : result["frames"].back()["streaming"]) {
                            if (stream.at("passName") == "VBuffer") {
                                geometry["identity"] = Json::parse(stream.at("tessellationIdentity").get<std::string>());
                            }
                        }
                        if (!geometry.contains("identity") || !result.contains("nativeCLASBoundsReceipt")) {
                            return fail("Encoding receipt requires bounded VBuffer identity and actual bounds");
                        }
                        geometry["templateBounds"] = result["nativeCLASBoundsReceipt"]["values"];
                        if (geometry["identity"].at("templatePoolGeneration") != receipt.templatePoolGeneration ||
                            geometry["identity"].at("templateBounds") != geometry["templateBounds"]) {
                            return fail("Final encoding receipt changed its bounded template pool");
                        }
                        const auto file = id + "-Geometry.json";
                        const auto serialized = geometry.dump(2);
                        std::ofstream output(context.outputDirectory / file, std::ios::binary);
                        output.write(serialized.data(), std::streamsize(serialized.size()));
                        if (!output) { return fail("Cannot write CLAS encoding manifest"); }
                        result["nativeCLASGeometryReceipt"] = {{"file", file}, {"patchCount", counts[0]}};
                        if (captureShadowRays) {
                            if (!receipt.shadowRaysCaptured || !receipt.shadowParametersCaptured ||
                                receipt.shadowMetadata.at("width") != width || receipt.shadowMetadata.at("height") != height ||
                                receipt.shadowParameters.shape[1] != 0.0f || receipt.shadowParameters.view.frame[0] != 0u) {
                                return fail("Missing/mismatched final production shadow-ray receipt");
                            }
                            std::vector<ShadowRayCaptureRecord> rays(size_t(width) * height);
                            read(*receipt.shadowRaysBuffer, rays.size() * sizeof(rays[0]), rays.data());
                            uint64_t activeRays = 0, hitRays = 0, proceedCalls = 0, candidates = 0, accepted = 0;
                            capturedShadowVisibility.resize(rays.size());
                            for (size_t pixel = 0; pixel < rays.size(); ++pixel) {
                                const auto& ray = rays[pixel];
                                if (ray.active > 1 || ray.hit > ray.active || !std::isfinite(ray.rawCommittedT) ||
                                    !std::isfinite(ray.shadowDistance) || !std::isfinite(ray.penumbra) ||
                                    ray.visibility != (ray.hit ? 0.0f : 1.0f) ||
                                    (ray.hit && (ray.rawCommittedT < ray.tMin || ray.rawCommittedT > ray.tMax)) ||
                                    (!ray.hit && ray.rawCommittedT != -1.0f)) {
                                    return fail("Invalid captured production shadow ray at pixel " + std::to_string(pixel));
                                }
                                activeRays += ray.active; hitRays += ray.hit; proceedCalls += ray.proceedCalls;
                                candidates += ray.nonOpaqueCandidates; accepted += ray.acceptedCandidates;
                                capturedShadowVisibility[pixel] = ray.visibility == 0.0f ? 0u : 255u;
                            }
                            if (!activeRays || !hitRays || hitRays == activeRays) { return fail("Shadow capture must contain actual hits and misses"); }
                            shadowManifest = {{"schemaVersion", 1}, {"geometryFile", file},
                                {"geometryManifestSha256", debug::debugSha256(serialized)},
                                {"recordStride", sizeof(ShadowRayCaptureRecord)}, {"rayCount", rays.size()},
                                {"width", width}, {"height", height}, {"frame", caseFrames - 1},
                                {"queryFlags", 0}, {"queryMask", 255}, {"activeCount", activeRays}, {"hitCount", hitRays},
                                {"totalProceedCalls", proceedCalls}, {"totalNonOpaqueCandidates", candidates}, {"totalAcceptedCandidates", accepted},
                                {"captureShaderSha256", receipt.shadowMetadata.at("captureShaderSha256")},
                                {"ordinaryShaderSha256", receipt.shadowMetadata.at("ordinaryShaderSha256")},
                                {"shaderMacro", "METALLIC_SHADOW_RAY_CAPTURE=1"}, {"timingInstrumented", true},
                                {"queryCounterScope", "shader-visible calls/candidates, not hardware node/triangle tests"},
                                {"dispatchMetadata", receipt.shadowMetadata}, {"sha256", Json::object()}};
                            const auto saveShadow = [&](const char* key, const char* suffix, const void* data, size_t bytes) {
                                const auto filename = id + suffix;
                                std::ofstream binary(context.outputDirectory / filename, std::ios::binary);
                                binary.write(static_cast<const char*>(data), std::streamsize(bytes));
                                if (!binary) { throw std::runtime_error("Cannot write production shadow-ray receipt"); }
                                shadowManifest[key] = filename;
                                shadowManifest["sha256"][filename] = debug::debugSha256(std::span(static_cast<const uint8_t*>(data), bytes));
                            };
                            saveShadow("recordsFile", "-ShadowRays.bin", rays.data(), rays.size() * sizeof(rays[0]));
                            saveShadow("parametersFile", "-ShadowParameters.bin", &receipt.shadowParameters, sizeof(receipt.shadowParameters));
                            saveShadow("visibilityFile", "-CapturedShadow.r8", capturedShadowVisibility.data(), capturedShadowVisibility.size());
                        }
                    }
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
                const size_t expectedBytes = size_t(width) * height * (half ? 8 : 16);
                if (bytes.size() != expectedBytes) {
                    return fail("Truncated HDR capture: " + Json{{"actualBytes", bytes.size()}, {"expectedBytes", expectedBytes},
                        {"format", half ? "RGBA16F" : "RGBA32F"}, {"declaredExtent", {width, height}},
                        {"actualExtent", {preview.width(), preview.height()}}}.dump());
                }
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
                    if (captureShadowRays && outputName == "Shadows.shadow") {
                        const auto& ordinary = preview.readbackBytes();
                        if (preview.readbackFormat() != Format::R8Unorm || ordinary.size() != capturedShadowVisibility.size() ||
                            !std::equal(ordinary.begin(), ordinary.end(), capturedShadowVisibility.begin(),
                                [](std::byte a, uint8_t b) { return std::to_integer<uint8_t>(a) == b; })) {
                            return fail("Instrumented production shadow visibility differs from the ordinary full image");
                        }
                        shadowManifest["ordinaryVisibilityFile"] = result["captures"].back().at("file");
                        shadowManifest["ordinaryVisibilitySha256"] = debug::debugSha256(std::span(
                            reinterpret_cast<const uint8_t*>(ordinary.data()), ordinary.size()));
                        shadowManifest["ordinaryVisibilityExact"] = true;
                    }
                }
                if (captureShadowRays) {
                    if (!shadowManifest.value("ordinaryVisibilityExact", false)) { return fail("Missing ordinary shadow equivalence image"); }
                    const auto filename = id + "-ShadowRays.json";
                    const auto serialized = shadowManifest.dump(2);
                    std::ofstream manifest(context.outputDirectory / filename, std::ios::binary);
                    manifest.write(serialized.data(), std::streamsize(serialized.size()));
                    if (!manifest) { return fail("Cannot write production shadow-ray manifest"); }
                    result["shadowRayReceipt"] = {{"file", filename}, {"sha256", debug::debugSha256(serialized)},
                        {"recordStride", sizeof(ShadowRayCaptureRecord)}, {"ordinaryVisibilityExact", true},
                        {"activeCount", shadowManifest.at("activeCount")}, {"hitCount", shadowManifest.at("hitCount")}};
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
