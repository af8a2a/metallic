#include "RHITest.h"
#include "Runtime/Render/Material/MaterialCoverageProgram.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/NamedResourceLayouts.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/RenderSample.h"
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
    RHITestResult run(RHITestContext&) override
    {
        scene::RenderMaterial material;
        material.alphaMode = "MASK";
        material.valueProgram = R"({"version":1,"baseColor":{"op":"sin","args":[{"op":"position"}]},"coverage":{"op":"mul","args":[{"op":"alpha"},{"op":"parameter","index":0}]}})";
        std::string log;
        auto first = MaterialValueProgramSet::create({&material, 1}, log);
        if (!first || first->programCount() != 1 || first->coverageProgramCount() != 1 ||
            first->instances()[0].coverageCount != 3 || first->instances()[0].coverageFlags != 1 ||
            first->inputBytes().size() != 80 + 3 * 32) { return RHITestResult::fail("Coverage slice contains unrelated Surface nodes: " + log); }
        material.valueProgram = R"({"version":1,"baseColor":{"op":"sin","args":[{"op":"position"}]},"coverage":0.5})";
        auto second = MaterialValueProgramSet::create({&material, 1}, log);
        if (!second || first->source() != second->source() || first->key() != second->key() ||
            second->instances()[0].coverageCount != 1 || second->instances()[0].coverageFlags != 0) {
            return RHITestResult::fail("Coverage changed Surface executable identity or retained an unused texture input");
        }
        material.valueProgram = R"({"version":1,"coverage":{"op":"parameter","index":0}})";
        std::vector<scene::RenderMaterial> instances(1000, material);
        for (size_t i = 0; i < instances.size(); ++i) { instances[i].valueParameters[0] = float(i) / 1000; }
        auto shared = MaterialValueProgramSet::create(instances, log);
        if (!shared || shared->programCount() != 0 || shared->coverageProgramCount() != 1 ||
            shared->inputBytes().size() != 1000 * 80 + 32 || shared->instances()[999].parameters[0] != 0.999f ||
            shared->instances()[999].coverageOffset != shared->instances()[0].coverageOffset) {
            return RHITestResult::fail("Coverage program was not shared independently of instance parameters");
        }
        for (const char* source : {R"({"version":1,"coverage":{"op":"position"}})",
                R"({"version":1,"coverage":{"op":"texture"}})", R"({"version":1,"coverage":{"op":"parameter","index":4}})"}) {
            material.valueProgram = source;
            if (MaterialValueProgramSet::create({&material, 1}, log) || log.empty()) { return RHITestResult::fail("Unsupported Coverage expression accepted"); }
        }
        material.valueProgram = R"({"version":1,"coverage":0})";
        material.alphaMode = "OPAQUE";
        if (MaterialValueProgramSet::create({&material, 1}, log)) { return RHITestResult::fail("Opaque Coverage was not rejected"); }
        return RHITestResult::pass("Minimal independent slices, resource dependency elimination, shared inputs and validation");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialCoverageCompilerTest);

class CoverageRayProbePass final : public ComputePass
{
public:
    explicit CoverageRayProbePass(std::shared_ptr<uint32_t> micromapCount) : micromapCount_(std::move(micromapCount)) {}
    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.features = SceneResourceFeatureBits::Geometry | SceneResourceFeatureBits::Materials |
            SceneResourceFeatureBits::MaterialTextures | SceneResourceFeatureBits::StandardAccelerationStructure};
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
        ShaderCompileResult shader;
        const char* capabilities[] = {"spvRayQueryKHR"};
        auto result = compileSlangShaderToSpirv({.moduleName = "MaterialCoverageProbe", .entryPointName = "materialCoverageProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .capabilities = capabilities}, log)
            .transform([&](auto value) { shader = std::move(value); });
        if (!result) { return result; }
        const ComputeProgramBindingDesc bindings[] = {
            {0, ComputeResourceBindingKind::AccelerationStructure}, {1, ComputeResourceBindingKind::StorageImage}, {2}, {3}, {4}, {5}, {6},
            {9, ComputeResourceBindingKind::SampledImage, geometry.materialTextureCount()}, {97}};
        return program_.initialize(*context.device, {.spirv = shader.spirv, .pushConstantSize = 8, .bindings = bindings,
            .requiresRayQuery = true, .resourceParameters = kSceneProbeResourceLayout}, log);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto& geometry = *context.preparedScene()->snapshot->pathTraceResources;
        const auto inputs = geometry.materialBinding();
        *micromapCount_ = geometry.accelerationStructure().stats().opacityMicromapCount;
        if (inputs->generation()->sourceRevision() != context.runtimeScene()->materialRevision()) { return makeError(Error::Failure); }
        if (inputs->values()->coverageProgramCount() && geometry.accelerationStructure().stats().opacityMicromapCount) {
            return makeError(Error::Failure); // The only MASK material now has dynamic Coverage.
        }
        if (auto* frame = RenderFrameContext::from(context.commandBuffer())) { frame->retain(inputs); }
        const ComputeDispatchBinding bindings[] = {
            {.binding = 0, .accelerationStructure = geometry.accelerationStructure().accelerationStructure()},
            {.binding = 1, .textureView = context.outputTexture("color").view()},
            {.binding = 2, .buffer = geometry.shadingVertexBuffer()}, {.binding = 3, .buffer = geometry.indexBuffer()},
            {.binding = 4, .buffer = geometry.primitiveBuffer()}, {.binding = 5, .buffer = geometry.instanceBuffer()},
            {.binding = 6, .buffer = geometry.materialBuffer()},
            {.binding = 9, .textureViews = geometry.materialTextureViews(), .sampledImages = geometry.materialTextureSnapshot()},
            {.binding = 97, .buffer = inputs->valueBuffer() ? inputs->valueBuffer() : inputs->buffer()}};
        const uint32_t push[] = {geometry.materialTextureCount(), 0};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .pushData = push, .pushDataSize = sizeof(push), .groupCountX = 8, .groupCountY = 8});
    }
private:
    ComputeProgram program_;
    std::shared_ptr<uint32_t> micromapCount_;
};

class MaterialCoverageRenderingTest : public RHITest
{
public:
    explicit MaterialCoverageRenderingTest(bool slab = false) : slab_(slab)
    { name = slab ? "material_coverage_slab_winner_shadow_ray" : "material_coverage_winner_shadow_ray"; type = RHITestType::Rendering; }
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
        const auto micromapCount = std::make_shared<uint32_t>(0);
        registerRenderGraphPassType("CoverageRayProbe", "Coverage ray/shadow test", [micromapCount] { return std::make_unique<CoverageRayProbePass>(micromapCount); });
        const auto probe = sample.graph.addNode("CoverageRayProbe", "CoverageProbe");
        sample.graph.setNodeRuntimeProperty(probe->id, "path", path.generic_string());
        sample.graph.markOutput("CoverageProbe.color");
        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        preview.setEnvironment({.enabled = false});
        preview.setRawReadbackEnabled(true);
        if (!preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("Coverage device failed"); }
        std::ofstream report(context.outputDirectory / "Coverage.txt");
        for (uint32_t step = 0; step < 8; ++step) {
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
                if (step == 7) { edited.valueProgram.clear(); }
                if (!scene.setMaterialProperties(0, edited)) { return RHITestResult::fail("Coverage edit failed"); }
            }
            if (!preview.render(sample.graph, 64, 64, "Deferred.color")) { return RHITestResult::fail(preview.lastLog()); }
            const auto rasterBytes = preview.readbackBytes();
            if (preview.readbackFormat() != Format::RGBA32Sfloat || rasterBytes.size() != 64 * 64 * 16) { return RHITestResult::fail("Raster readback format"); }
            std::vector<float> colors(rasterBytes.size() / 4);
            std::memcpy(colors.data(), rasterBytes.data(), rasterBytes.size());
            if (!saveRgba8Png(context.outputDirectory / ("Coverage-" + std::to_string(step) + ".png"),
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), 64, 64, log)) { return RHITestResult::fail(log); }
            if (!preview.render(sample.graph, 64, 64, "CoverageProbe.color")) { return RHITestResult::fail(preview.lastLog()); }
            std::vector<float> rays(preview.readbackBytes().size() / 4);
            std::memcpy(rays.data(), preview.readbackBytes().data(), preview.readbackBytes().size());
            uint32_t winners = 0;
            for (uint32_t y = 1; y < 63; ++y) {
                for (uint32_t x = 1; x < 63; ++x) {
                    const size_t i = (y * 64 + x) * 4;
                    const bool front = step == 0 || step == 7 || (1.0f - (x + 0.5f) / 64 + shift) * (step >= 5 ? 192.0f / 255.0f : 1.0f) >= 0.5f;
                    const float t = front ? 2.0f : 3.0f;
                    if (!std::isfinite(rays[i]) || !std::isfinite(rays[i + 1]) ||
                        !std::isfinite(colors[i]) || !std::isfinite(colors[i + 1]) || !std::isfinite(colors[i + 2]) ||
                        std::abs(rays[i] - t) > 1e-5f || std::abs(rays[i + 1] - t) > 1e-5f ||
                        std::abs(colors[i] - (front && step != 6 ? 1.0f : 0.0f)) > 1e-5f ||
                        std::abs(colors[i + 1] - (front ? 0.0f : 1.0f)) > 1e-5f ||
                        std::abs(colors[i + 2] - (front && step == 6 ? 1.0f : 0.0f)) > 1e-5f) {
                        return RHITestResult::fail("Coverage winner mismatch step=" + std::to_string(step) + " x=" + std::to_string(x) +
                            " red=" + std::to_string(colors[i]) + " green=" + std::to_string(colors[i + 1]) +
                            " ray=" + std::to_string(rays[i]) + " shadow=" + std::to_string(rays[i + 1]));
                    }
                    winners += front;
                }
            }
            report << "step=" << step << " frontWinners=" << winners << " checked=3844 opacityMicromaps=" << *micromapCount << '\n';
        }
        return RHITestResult::pass("Layered VBuffer winner, production RT/shadow Coverage, texture alpha, shared parameters and legacy restoration agree for 30752 pixels");
    }
private:
    bool slab_;
};
METALLIC_REGISTER_RHI_TEST(MaterialCoverageRenderingTest);
class MaterialCoverageSlabRenderingTest final : public MaterialCoverageRenderingTest
{
public:
    MaterialCoverageSlabRenderingTest() : MaterialCoverageRenderingTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(MaterialCoverageSlabRenderingTest);
} // namespace
} // namespace metallic::tests
