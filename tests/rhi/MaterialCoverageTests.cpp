#include "RHITest.h"
#include "Runtime/Render/Material/MaterialCoverageProgram.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/RenderPass/BuiltinPass/StreamRayTracingBindings.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"
#include <cstring>
#include <fstream>
#include <cmath>

namespace metallic::tests {
namespace {
using namespace render;

class MaterialCoverageCompilerTest final : public RHITest
{
public:
    MaterialCoverageCompilerTest() { name = "material_coverage_compiler"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext& context) override
    {
        scene::RenderMaterial material;
        material.alphaMode = "MASK";
        material.valueProgram = R"({"version":1,"baseColor":{"op":"sin","args":[{"op":"position"}]},"coverage":{"op":"mul","args":[{"op":"alpha"},{"op":"parameter","index":0}]}})";
        std::string log;
        auto first = MaterialValueProgramSet::create({&material, 1}, log);
        if (!first || first->programCount() != 1 || first->coverageProgramCount() != 1 ||
            first->instances()[0].coverageCount != 3 || first->instances()[0].coverageFlags != 1 ||
            first->inputBytes().size() != 80) { return RHITestResult::fail("Coverage slice contains bytecode or unrelated Surface nodes: " + log); }
        material.valueProgram = R"({"version":1,"baseColor":{"op":"sin","args":[{"op":"position"}]},"coverage":0.5})";
        auto second = MaterialValueProgramSet::create({&material, 1}, log);
        if (!second || first->source() != second->source() || first->key() != second->key() ||
            first->coverageSource() == second->coverageSource() || first->coverageKey() == second->coverageKey() ||
            second->instances()[0].coverageCount != 1 || second->instances()[0].coverageFlags != 0) {
            return RHITestResult::fail("Coverage changed Surface executable identity or retained an unused texture input");
        }
        material.valueProgram = R"({"version":1,"coverage":{"op":"parameter","index":0}})";
        std::vector<scene::RenderMaterial> instances(1000, material);
        for (size_t i = 0; i < instances.size(); ++i) { instances[i].valueParameters[0] = float(i) / 1000; }
        auto shared = MaterialValueProgramSet::create(instances, log);
        if (!shared || shared->programCount() != 0 || shared->coverageProgramCount() != 1 ||
            shared->inputBytes().size() != 1000 * 80 || shared->instances()[999].parameters[0] != 0.999f ||
            shared->instances()[999].coverageOffset != shared->instances()[0].coverageOffset) {
            return RHITestResult::fail("Coverage program was not shared independently of instance parameters");
        }
        instances[0].valueParameters[0] = 0.123f;
        const auto changedParameter = MaterialValueProgramSet::create(instances, log);
        if (!changedParameter || changedParameter->coverageSource() != shared->coverageSource() ||
            changedParameter->coverageKey() != shared->coverageKey()) { return RHITestResult::fail("Parameter-only edit changed Coverage executable identity"); }
        material.valueProgram = R"({"version":1,"coverage":0.25})";
        const auto coverageOnlyA = MaterialValueProgramSet::create({&material, 1}, log);
        material.valueProgram = R"({"version":1,"coverage":0.75})";
        const auto coverageOnlyB = MaterialValueProgramSet::create({&material, 1}, log);
        if (!coverageOnlyA || !coverageOnlyB || coverageOnlyA->programCount() || coverageOnlyB->programCount() ||
            coverageOnlyA->key() != coverageOnlyB->key() || coverageOnlyA->coverageKey() == coverageOnlyB->coverageKey()) {
            return RHITestResult::fail("Coverage-only edit did not have an independent executable identity");
        }
        std::filesystem::path cacheA, cacheB;
        if (!coverageOnlyA->writeCoverageInclude(context.outputDirectory / "coverage-cache", cacheA, log) ||
            !coverageOnlyB->writeCoverageInclude(context.outputDirectory / "coverage-cache", cacheB, log) || cacheA == cacheB) {
            return RHITestResult::fail("Coverage cache identities collided: " + log);
        }
        const auto cacheFile = cacheA / "MaterialCoverageGenerated.slang";
        std::ofstream(cacheFile, std::ios::trunc) << "corrupt cache";
        if (coverageOnlyA->writeCoverageInclude(context.outputDirectory / "coverage-cache", cacheA, log)) {
            return RHITestResult::fail("Corrupt Coverage cache was accepted");
        }
        std::ofstream(cacheFile, std::ios::binary | std::ios::trunc) << coverageOnlyA->coverageSource();
        for (const char* source : {R"({"version":1,"coverage":{"op":"position"}})",
                R"({"version":1,"coverage":{"op":"texture"}})", R"({"version":1,"coverage":{"op":"parameter","index":4}})"}) {
            material.valueProgram = source;
            if (MaterialValueProgramSet::create({&material, 1}, log) || log.empty()) { return RHITestResult::fail("Unsupported Coverage expression accepted"); }
        }
        material.valueProgram = R"({"version":1,"coverage":0})";
        material.alphaMode = "BLEND";
        if (MaterialValueProgramSet::create({&material, 1}, log)) { return RHITestResult::fail("Custom BLEND Coverage was not rejected"); }
        material.alphaMode = "OPAQUE";
        if (MaterialValueProgramSet::create({&material, 1}, log)) { return RHITestResult::fail("Opaque Coverage was not rejected"); }
        return RHITestResult::pass("Minimal independent slices, resource dependency elimination, shared inputs and validation");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialCoverageCompilerTest);

class CoverageRayProbePass final : public ComputePass
{
public:
    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.features = SceneResourceFeatureBits::Geometry | SceneResourceFeatureBits::Materials |
            SceneResourceFeatureBits::MaterialTextures | SceneResourceFeatureBits::AccelerationStructure, .geometry = true, .textureFeedback = true};
    }
    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{GPUSceneSubsystem::kSubsystemId};
        return required;
    }
    void describeSceneView(const RenderGraphExecutionContext&, MeshletStreamFrameDesc& view) const override
    {
        view.width = view.height = 64;
        view.camera.eye = float3(0.5f, 0.5f, 0.0f);
        view.camera.center = float3(0.5f, 0.5f, 2.0f);
        view.camera.up = float3(0, 1, 0);
        view.camera.orthographic = true;
        view.camera.orthoHeight = 1.0f;
        view.camera.znear = 0.001f;
        view.camera.zfar = 10.0f;
    }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection result;
        result.addTextureOutput("color").storageWrite().format = Format::RGBA32Sfloat;
        return result;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        const auto& geometry = *context.preparedScene->snapshot->pathTraceResources;
        const auto values = geometry.materialBinding()->values();
        std::filesystem::path coverageDirectory;
        std::vector<SlangMacroDefine> macros;
        std::vector<const char*> paths;
        std::string coveragePath;
        if (values && values->coverageProgramCount()) {
            if (!values->writeCoverageInclude(PROJECT_SOURCE_DIR "/.cache/materials", coverageDirectory, log)) { return makeError(Error::Failure); }
            coveragePath = coverageDirectory.string(); paths.push_back(coveragePath.c_str());
            macros.push_back({"METALLIC_CUSTOM_COVERAGE", "1"});
        }
        ShaderCompileResult shader;
        const char* capabilities[] = {"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV"};
        auto result = compileSlangShaderToSpirv({.moduleName = "MaterialCoverageProbe", .entryPointName = "materialCoverageProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .additionalSearchPaths = paths,
            .capabilities = capabilities, .macroDefines = macros}, log)
            .transform([&](auto value) { shader = std::move(value); });
        if (!result) { return result; }
        std::vector<ComputeResourceBindingDesc> bindings = {
            {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, scene), ComputeResourceBindingKind::AccelerationStructure}, {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, output), ComputeResourceBindingKind::StorageImage}, {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materials)},
            {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materialTextures), ComputeResourceBindingKind::SampledImage, geometry.materialTextureCount()}, {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materialValues)},
            {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, textureSamplingMetadata)},
            {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, textureDemandMip)},
            {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, textureDemandHits)}};
        builtin_pass::appendStreamRayLayout(bindings);
        return program_.initialize(*context.device, {.spirv = shader.spirv, .pushConstantSize = sizeof(uint32_t), .bindings = bindings,
            .requiresRayQuery = true, .resourceParameterSize = sizeof(render::SceneResourceParameters)}, log);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto& geometry = *context.preparedScene()->snapshot->pathTraceResources;
        const auto inputs = geometry.materialBinding();
        if (inputs->generation()->sourceRevision() != context.runtimeScene()->materialRevision()) { return makeError(Error::Failure); }
        const auto* prepared = context.preparedScene();
        if (!prepared->geometry) { return makeError(Error::InvalidArgument); }
        if (prepared->geometry->topLevelBuildPending()) {
            auto built = prepared->geometry->cmdBuildTopLevelAccelerationStructure(context.commandBuffer());
            if (!built) { return built; }
        }
        const auto stream = prepared->geometry->deferredGpuResources();
        if (!stream.valid() || !stream.accelerationStructure) { return makeError(Error::InvalidArgument); }
        if (auto* frame = RenderFrameContext::from(context.commandBuffer())) { frame->retain(inputs); }
        std::vector<ComputeDispatchBinding> bindings = {
            {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, scene), .accelerationStructure = stream.accelerationStructure},
            {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, output), .textureView = context.outputTexture("color").view()},
            {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materials), .buffer = geometry.materialBuffer()},
            {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materialTextures), .textureViews = geometry.materialTextureViews(), .sampledImages = geometry.materialTextureSnapshot()},
            {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materialValues), .buffer = inputs->valueBuffer() ? inputs->valueBuffer() : inputs->buffer()}};
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureSamplingMetadata), .buffer = prepared->textureFeedback.samplingMetadata});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandMip), .buffer = prepared->textureFeedback.demandMip});
        bindings.push_back({.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandHits), .buffer = prepared->textureFeedback.demandHits});
        builtin_pass::appendStreamRayBindings(bindings, stream, prepared->geometry->shadingChangeBuffer());
        const uint32_t push = geometry.materialTextureCount();
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .pushData = &push, .pushDataSize = sizeof(push), .groupCountX = 8, .groupCountY = 8});
    }
private:
    ComputeProgram program_;
};

class MaterialCoverageRenderingTest : public RHITest
{
public:
    explicit MaterialCoverageRenderingTest(bool slab = false, bool shadowOnly = false) : slab_(slab), shadowOnly_(shadowOnly)
    {
        name = shadowOnly ? "stream_closest_shadow_hit" : slab ? "material_coverage_slab_winner_shadow_ray" : "material_coverage_winner_shadow_ray";
        type = RHITestType::Rendering;
    }
    RHITestResult run(RHITestContext& context) override
    {
        const auto root = std::filesystem::absolute(context.outputDirectory / "coverage-scene");
        std::filesystem::create_directories(root);
        const auto path = root / "Scene.gltf";
        {
            const float vertices[] = {0,0,2, 1,0,2, 0,1,2, 1,1,2, 0,0, 1,0, 0,1, 1,1};
            const uint32_t indices[] = {2,0,1, 2,1,3};
            std::ofstream binary(root / "Mesh.bin", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(vertices), sizeof(vertices));
            binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
            std::ofstream(path) << R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1]}],
                "nodes":[{"mesh":0},{"mesh":1,"translation":[0,0,1]}],
                "meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":0}]},
                    {"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":1}]}],
                "buffers":[{"uri":"Mesh.bin","byteLength":104}],"bufferViews":[{"buffer":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":32},{"buffer":0,"byteOffset":80,"byteLength":24}],
                "accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,2],"max":[1,1,2]},
                    {"bufferView":1,"componentType":5126,"count":4,"type":"VEC2"},{"bufferView":2,"componentType":5125,"count":6,"type":"SCALAR"}],
                "images":[{"uri":"Alpha.png"}],"textures":[{"source":0}],
                "materials":[{"alphaMode":"MASK","doubleSided":true,"pbrMetallicRoughness":{"baseColorTexture":{"index":0},"baseColorFactor":[1,0,0,1]}},
                    {"doubleSided":true,"pbrMetallicRoughness":{"baseColorFactor":[0,1,0,1]}}]})";
        }
        std::string log;
        const uint8_t alphaPixel[] = {255, 255, 255, 192};
        if (!saveRgba8Png(root / "Alpha.png", alphaPixel, 1, 1, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument scene;
        if (!scene.load(path)) { return RHITestResult::fail(scene.lastLoadResult().error); }
        RenderSampleLoadResult sample;
        if (!loadBuiltInRenderSample("lookdev-vbuffer", sample, log) || !setRenderSampleScenePath(sample, path.generic_string(), log)) { return RHITestResult::fail(log); }
        for (const char* name : {"VBuffer", "Reference"}) {
            const auto id = sample.graph.findNode(name)->id;
            sample.graph.setNodeRuntimeProperty(id, "camera.eye", {0.5, 0.5, 0.0});
            sample.graph.setNodeRuntimeProperty(id, "camera.center", {0.5, 0.5, 2.0});
            sample.graph.setNodeRuntimeProperty(id, "camera.up", {0.0, 1.0, 0.0});
            sample.graph.setNodeRuntimeProperty(id, "camera.projection", "orthographic");
            sample.graph.setNodeRuntimeProperty(id, "camera.orthoHeight", 1.0);
        }
        const auto raster = sample.graph.findNode("VBuffer")->id;
        sample.graph.setNodeRuntimeProperty(raster, "autoLod", false);
        sample.graph.setNodeRuntimeProperty(raster, "lodLevel", 0);
        sample.graph.setNodeRuntimeProperty(raster, "meshletNormalConeCull", false);
        const auto deferred = sample.graph.findNode("Deferred")->id;
        sample.graph.setNodeRuntimeProperty(deferred, "debugView", "baseColor");
        // The comparison pass requires equal encodings after the color-system
        // migration: both inputs must be display-linear diagnostics.
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Reference")->id, "debugView", "baseColor");
        sample.graph.setNodeRuntimeProperty(deferred, "accumulate", false);
        registerRenderGraphPassType("CoverageRayProbe", "Coverage ray/shadow test", [] { return std::make_unique<CoverageRayProbePass>(); });
        const auto probe = sample.graph.addNode("CoverageRayProbe", "CoverageProbe");
        sample.graph.setNodeRuntimeProperty(probe->id, "path", path.generic_string());
        sample.graph.markOutput("CoverageProbe.color");
        if (shadowOnly_) {
            // Exercise published CLAS hits without unrelated raster/material hot
            // compilation. The comprehensive Coverage tests retain their graph.
            sample.graph = RenderGraph{};
            sample.graph.addNode("CoverageRayProbe", "CoverageProbe", {{"path", path.generic_string()}});
            sample.graph.markOutput("CoverageProbe.color");
        }
        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        preview.setEnvironment({.enabled = false});
        preview.setRawReadbackEnabled(true);
        if (!preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("Coverage device failed"); }
        std::ofstream report(context.outputDirectory / "Coverage.txt");
        uint64_t coverageOnlySurfaceKey = 0, coverageOnlyKey = 0;
        for (uint32_t scenario = 0; scenario < (shadowOnly_ ? 4u : 11u); ++scenario) {
            // Focused mode exercises only Coverage programs: install, change
            // executable identity, and remove. It never creates a Surface graph.
            const uint32_t step = shadowOnly_ && scenario > 0 ? scenario + 7u : scenario;
            auto edited = scene.materials()[0];
            const float shift = step == 2 ? -0.25f : step == 3 ? 1.0f : step == 4 ? -1.0f : step == 6 ? 0.25f : 0.0f;
            if (step > 0) {
                edited.valueProgram = R"({"version":1,"coverage":{"op":"add","args":[{"op":"uv"},{"op":"parameter","index":0}]}})";
                edited.valueParameters[0] = shift;
                if (step >= 5) {
                    edited.valueProgram = R"({"version":2,"nodes":{"shift":{"op":"add","args":[{"op":"swizzle","components":"xxxx","args":[{"op":"uv"}]},{"op":"parameter","index":0}]},"mask":{"op":"clamp","args":[{"ref":"shift"},-1000000,1000000]}},"outputs":{"baseColor":{"op":"parameter","index":1},"coverage":{"op":"mul","args":[{"op":"alpha"},{"ref":"mask"}]}}})";
                    edited.valueParameters[4] = step == 5 ? 1.0f : 0.0f;
                    edited.valueParameters[6] = step == 6 ? 1.0f : 0.0f;
                    if (slab_) {
                        auto graph = nlohmann::json::parse(edited.valueProgram);
                        graph["version"] = 3;
                        graph["closure"] = {{"op", "slab"}, {"reflectance", graph["outputs"]["baseColor"]}};
                        graph["outputs"].erase("baseColor");
                        edited.valueProgram = graph.dump();
                    }
                }
                if (step == 8 || step == 9) {
                    edited.valueProgram = nlohmann::json{{"version", 1}, {"coverage", {{"op", "mul"},
                        {"args", nlohmann::json::array({{{"op", "uv"}}, step == 8 ? 0.5 : 2.0})}}}}.dump();
                    const auto program = MaterialValueProgramSet::create({&edited, 1}, log);
                    if (!program || program->programCount() != 0 || (step == 9 &&
                        (program->key() != coverageOnlySurfaceKey || program->coverageKey() == coverageOnlyKey))) {
                        return RHITestResult::fail("Coverage-only edit identity failed");
                    }
                    coverageOnlySurfaceKey = program->key(); coverageOnlyKey = program->coverageKey();
                }
                if (step == 7 || step == 10) { edited.valueProgram.clear(); }
                if (!scene.setMaterialProperties(0, edited)) { return RHITestResult::fail("Coverage edit failed"); }
            }
            std::vector<float> colors;
            if (!shadowOnly_) {
                if (!preview.render(sample.graph, 64, 64, "Deferred.color")) { return RHITestResult::fail(preview.lastLog()); }
                std::ofstream(context.outputDirectory / ("Coverage-" + std::to_string(step) + ".compile.log")) << preview.lastLog();
                if (preview.lastLog().find("Initial material compilation failed") != std::string::npos ||
                    preview.lastLog().find("error[") != std::string::npos) {
                    return RHITestResult::fail("Coverage raster shader failed step=" + std::to_string(step) + ": " + preview.lastLog());
                }
                const auto rasterBytes = preview.readbackBytes();
                if (preview.readbackFormat() != Format::RGBA32Sfloat || rasterBytes.size() != 64 * 64 * 16) { return RHITestResult::fail("Raster readback format"); }
                colors.resize(rasterBytes.size() / 4);
                std::memcpy(colors.data(), rasterBytes.data(), rasterBytes.size());
                if (!saveRgba8Png(context.outputDirectory / ("Coverage-" + std::to_string(step) + ".png"),
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), 64, 64, log)) { return RHITestResult::fail(log); }
            }
            if (!preview.render(sample.graph, 64, 64, "CoverageProbe.color")) { return RHITestResult::fail(preview.lastLog()); }
            if (preview.readbackFormat() != Format::RGBA32Sfloat || preview.readbackBytes().size() != 64 * 64 * 16) {
                return RHITestResult::fail("Shadow probe readback format");
            }
            std::vector<float> rays(preview.readbackBytes().size() / 4);
            std::memcpy(rays.data(), preview.readbackBytes().data(), preview.readbackBytes().size());
            uint32_t winners = 0;
            for (uint32_t y = 1; y < 63; ++y) {
                for (uint32_t x = 1; x < 63; ++x) {
                    const size_t i = (y * 64 + x) * 4;
                    const float u = 1.0f - (x + 0.5f) / 64;
                    const bool front = step == 0 || step == 7 || step == 10 ||
                        (step == 8 || step == 9 ? u * (step == 8 ? 0.5f : 2.0f) >= 0.5f :
                            (u + shift) * (step >= 5 ? 192.0f / 255.0f : 1.0f) >= 0.5f);
                    const float t = front ? 2.0f : 3.0f;
                    const bool rasterMismatch = !shadowOnly_ &&
                        (!std::isfinite(colors[i]) || !std::isfinite(colors[i + 1]) || !std::isfinite(colors[i + 2]) ||
                            std::abs(colors[i] - (front && step != 6 ? 1.0f : 0.0f)) > 1e-5f ||
                            std::abs(colors[i + 1] - (front ? 0.0f : 1.0f)) > 1e-5f ||
                            std::abs(colors[i + 2] - (front && step == 6 ? 1.0f : 0.0f)) > 1e-5f);
                    if (!std::isfinite(rays[i]) || !std::isfinite(rays[i + 1]) ||
                        std::abs(rays[i] - t) > 1e-5f || std::abs(rays[i + 1] - t) > 1e-5f || rays[i + 3] != 0.0f || rasterMismatch) {
                        return RHITestResult::fail("Coverage winner mismatch step=" + std::to_string(step) + " x=" + std::to_string(x) +
                            " rasterMismatch=" + std::to_string(rasterMismatch) + " ray=" + std::to_string(rays[i]) +
                            " shadow=" + std::to_string(rays[i + 1]) + " shadowHitFailures=" + std::to_string(rays[i + 3]) + " expectedDistance=" + std::to_string(t));
                    }
                    winners += front;
                }
            }
            report << "step=" << step << " frontWinners=" << winners << " checked=3844 CLAS alpha traversal" << '\n';
        }
        // BLEND uses the shared Coverage evaluator through CLAS candidates. Custom Coverage on
        // BLEND remains rejected by the authoring contract (tested above).
        sample.graph = RenderGraph{};
        sample.graph.addNode("CoverageRayProbe", "CoverageProbe", {{"path",path.generic_string()}});
        sample.graph.markOutput("CoverageProbe.color");
        for (float opacity : {0.0f,0.25f,0.75f,1.0f}) {
            auto edited=scene.materials()[0]; edited.alphaMode="BLEND";
            edited.valueProgram.clear(); edited.baseColorFactor.w=opacity;
            if (!scene.setMaterialProperties(0,edited)) { return RHITestResult::fail("BLEND Coverage edit failed"); }
            if (!preview.render(sample.graph,64,64,"CoverageProbe.color")) { return RHITestResult::fail(preview.lastLog()); }
            std::vector<float> rays(preview.readbackBytes().size()/4);
            std::memcpy(rays.data(),preview.readbackBytes().data(),preview.readbackBytes().size());
            for (uint32_t y=1;y<63;++y) {
                for (uint32_t x=1;x<63;++x) {
                    size_t i=(y*64+x)*4;
                    if (std::abs(rays[i]-(opacity>0 ? 2.0f : 3.0f))>1e-5f || std::abs(rays[i+2]-opacity*192.0f/255.0f)>1e-5f || rays[i+3] != 0.0f) {
                        return RHITestResult::fail("BLEND Coverage/closest shadow mismatch, shadowHitFailures=" + std::to_string(rays[i+3]));
                    }
                }
            }
        }
        // The same uploaded geometry exercises committed opaque early-out and
        // reduced transparent decoding after a material-only lifecycle update.
        for (float transmission : {0.0f, 1.0f}) {
            for (uint32_t materialIndex = 0; materialIndex < 2; ++materialIndex) {
                auto edited = scene.materials()[materialIndex];
                edited.alphaMode = "OPAQUE";
                edited.valueProgram.clear();
                edited.baseColorFactor.w = 1.0f;
                edited.transmissionFactor = transmission;
                // This API returns false for an unchanged material as well as
                // rejected input. Assert the requested state, including no-ops.
                (void)scene.setMaterialProperties(materialIndex, edited);
                const auto& applied = scene.materials()[materialIndex];
                if (applied.alphaMode != "OPAQUE" || !applied.valueProgram.empty() ||
                    applied.baseColorFactor.w != 1.0f || applied.transmissionFactor != transmission) {
                    return RHITestResult::fail("Shadow material edit did not establish requested state");
                }
            }
            if (!preview.render(sample.graph, 64, 64, "CoverageProbe.color")) { return RHITestResult::fail(preview.lastLog()); }
            if (preview.readbackFormat() != Format::RGBA32Sfloat || preview.readbackBytes().size() != 64 * 64 * 16) {
                return RHITestResult::fail("Shadow probe readback format");
            }
            std::vector<float> rays(preview.readbackBytes().size() / 4);
            std::memcpy(rays.data(), preview.readbackBytes().data(), preview.readbackBytes().size());
            for (uint32_t y = 1; y < 63; ++y) {
                for (uint32_t x = 1; x < 63; ++x) {
                    const size_t i = (y * 64 + x) * 4;
                    if (!std::isfinite(rays[i]) || std::abs(rays[i] - 2.0f) > 1e-5f || rays[i + 3] != 0.0f) {
                        return RHITestResult::fail("Opaque/glass closest shadow mismatch transmission=" + std::to_string(transmission) +
                            " x=" + std::to_string(x) + " y=" + std::to_string(y) + " shadowHitFailures=" + std::to_string(rays[i + 3]));
                    }
                }
            }
            report << "transmission=" << transmission << " checked=3844 closest-shadow full-hit comparisons x5 ray cases; RAY_FLAG_NONE callbacks=0, forced-nonopaque callbacks>0" << '\n';
        }
        if (shadowOnly_) {
            // Revisit MASK after both instances were opaque. This must invalidate
            // previously published TLAS opacity, including Coverage-only hot edits.
            for (uint32_t step = 0; step < 3; ++step) {
                auto edited = scene.materials()[0];
                edited.alphaMode = "MASK";
                edited.transmissionFactor = 0.0f;
                edited.valueProgram = step < 2 ? nlohmann::json{{"version", 1}, {"coverage", {{"op", "mul"},
                    {"args", nlohmann::json::array({{{"op", "uv"}}, step == 0 ? 0.5 : 2.0})}}}}.dump() : std::string{};
                if (!scene.setMaterialProperties(0, edited)) { return RHITestResult::fail("Opaque-to-MASK Coverage edit failed"); }
                if (!preview.render(sample.graph, 64, 64, "CoverageProbe.color")) { return RHITestResult::fail(preview.lastLog()); }
                if (preview.readbackFormat() != Format::RGBA32Sfloat || preview.readbackBytes().size() != 64 * 64 * 16) {
                    return RHITestResult::fail("Opaque-to-MASK probe readback format");
                }
                std::vector<float> rays(preview.readbackBytes().size() / 4);
                std::memcpy(rays.data(), preview.readbackBytes().data(), preview.readbackBytes().size());
                uint32_t winners = 0;
                for (uint32_t y = 1; y < 63; ++y) {
                    for (uint32_t x = 1; x < 63; ++x) {
                        const size_t i = (y * 64 + x) * 4;
                        const float u = 1.0f - (x + 0.5f) / 64;
                        const bool front = step == 2 || u * (step == 0 ? 0.5f : 2.0f) >= 0.5f;
                        const float distance = front ? 2.0f : 3.0f;
                        if (!std::isfinite(rays[i]) || !std::isfinite(rays[i + 1]) ||
                            std::abs(rays[i] - distance) > 1e-5f || std::abs(rays[i + 1] - distance) > 1e-5f || rays[i + 3] != 0.0f) {
                            return RHITestResult::fail("Opaque-to-MASK traversal mismatch step=" + std::to_string(step) +
                                " x=" + std::to_string(x) + " failures=" + std::to_string(rays[i + 3]));
                        }
                        winners += front;
                    }
                }
                report << "opaque-to-mask-step=" << step << " frontWinners=" << winners << " checked=3844 NONE vs forced-nonopaque full-surface reference" << '\n';
            }
            return RHITestResult::pass("249860 MASK/BLEND/opaque/glass closest-shadow comparisons match full-hit classification; 49972 NONE vs forced-nonopaque full-surface traversal comparisons, opaque callbacks zero, Coverage-only install/hot edit/removal and opaque-to-MASK lifecycle verified (ray query)");
        }
        return RHITestResult::pass("42284 MASK raster/RT/shadow pixels agree; MASK/BLEND/opaque/glass closest-shadow matches full-hit classification and consumed attributes for forward/backward/miss/clipped/nonunit rays (ray query)");
    }
private:
    bool slab_;
    bool shadowOnly_;
};
METALLIC_REGISTER_RHI_TEST(MaterialCoverageRenderingTest);
class MaterialCoverageSlabRenderingTest final : public MaterialCoverageRenderingTest
{
public:
    MaterialCoverageSlabRenderingTest() : MaterialCoverageRenderingTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(MaterialCoverageSlabRenderingTest);
class StreamClosestShadowHitTest final : public MaterialCoverageRenderingTest
{
public:
    StreamClosestShadowHitTest() : MaterialCoverageRenderingTest(false, true) {}
};
METALLIC_REGISTER_RHI_TEST(StreamClosestShadowHitTest);
} // namespace
} // namespace metallic::tests
