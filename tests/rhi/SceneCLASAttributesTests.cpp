#include "RHITest.h"
#include "harness/Fixtures.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/BuiltinPass/StreamRayTracingBindings.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Scene/Scene.h"
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {
using namespace render;
#define FETCH_REQUIRE(expression) do { const Result<> checked = (expression); if (!checked) { return RHITestResult::fail(std::string(#expression) + ": " + toString(checked) + " " + log); } } while(false)

class CLASAttributesProbePass final : public ComputePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.features = SceneResourceFeatureBits::Materials | SceneResourceFeatureBits::MaterialTextures |
            SceneResourceFeatureBits::AccelerationStructure, .geometry = true, .textureFeedback = true};
    }
    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{GPUSceneSubsystem::kSubsystemId};
        return required;
    }
    RenderGraphSceneDependency sceneDependency() const override { return {RenderGraphSceneSource::World}; }
    void describeSceneView(const RenderGraphExecutionContext&, MeshletStreamFrameDesc& view) const override
    {
        view.width = view.height = 64;
        view.camera.eye = float3(5, -0.5f, 10);
        view.camera.center = float3(5, -0.5f, 3);
        view.camera.up = float3(0, 1, 0);
        view.camera.orthographic = true;
        view.camera.orthoHeight = 16;
        view.camera.znear = 0.001f;
        view.camera.zfar = 100;
    }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addBufferOutput("hits").buffer(96 * sizeof(float), 4 * sizeof(float)).storageWrite().hostReadback();
        return reflection;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        const auto& material = *context.preparedScene->snapshot->pathTraceResources;
        const char* capabilities[]{"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV"};
        auto shader = compileSlangShaderToSpirv({.moduleName = "Features/SmokeTests/SceneCLASAttributesProbe",
            .entryPointName = "sceneCLASAttributesProbeMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .capabilities = capabilities, .descriptorHeapMode = properties().value("native", false)
                ? SlangDescriptorHeapMode::Native : SlangDescriptorHeapMode::Mapped}, log);
        if (!shader) { return makeError(shader.error()); }
        std::vector<ComputeResourceBindingDesc> layout{
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, scene), ComputeResourceBindingKind::AccelerationStructure},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureSamplingMetadata)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandMip)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandHits)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures), ComputeResourceBindingKind::SampledImage, material.materialTextureCount()},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput)}};
        builtin_pass::appendStreamRayLayout(layout);
        return program_.initialize(*context.device, {.spirv = shader->spirv, .pushConstantSize = sizeof(float),
            .bindings = layout, .requiresRayQuery = true, .resourceParameterSize = sizeof(SceneResourceParameters)}, log);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto* prepared = context.preparedScene();
        if (prepared->geometry->topLevelBuildPending()) {
            auto result = prepared->geometry->cmdBuildTopLevelAccelerationStructure(context.commandBuffer());
            if (!result) { return result; }
        }
        const auto stream = prepared->geometry->deferredGpuResources();
        const auto& material = *prepared->snapshot->pathTraceResources;
        auto values = material.materialBinding();
        std::vector<ComputeDispatchBinding> bindings{
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, scene), .accelerationStructure = stream.accelerationStructure},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials), .buffer = material.materialBuffer()},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues), .buffer = values->valueBuffer() ? values->valueBuffer() : material.materialBuffer()},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureSamplingMetadata), .buffer = prepared->textureFeedback.samplingMetadata},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandMip), .buffer = prepared->textureFeedback.demandMip},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandHits), .buffer = prepared->textureFeedback.demandHits},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures), .textureViews = material.materialTextureViews(), .sampledImages = material.materialTextureSnapshot()},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput), .buffer = context.outputBuffer("hits").buffer()}};
        builtin_pass::appendStreamRayBindings(bindings, stream, prepared->geometry->shadingChangeBuffer());
        const float translation = context.runtimeScene()->nodes()[0].localMatrix.a03 - 3.0f;
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .pushData = &translation, .pushDataSize = sizeof(translation)});
    }
private:
    ComputeProgram program_;
};

class SceneCLASAttributesTest : public RHITest {
public:
    explicit SceneCLASAttributesTest(bool authoredTangents = false, bool native = false)
        : authoredTangents_(authoredTangents), native_(native)
    {
        type = RHITestType::Rendering;
        name = native ? "scene_clas_attributes_native" : authoredTangents ? "scene_clas_attributes_authored_tangents" : "scene_clas_attributes";
    }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::gpuMetadata({"clas.scene.transform.attributes.authoredTBN"}, bench::Layer::Core, "ray-query", "core", {"readback.bin"}, native_);
    }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = context.outputDirectory / name;
        std::filesystem::create_directories(directory);
        const auto path = directory / "scene.gltf";
        {
            const float attributes[] = {
                0, 0, 0, 2, 0, 0, 0, 1, 1, 2, 1, 1,
                0, -0.70710678f, 0.70710678f, 0, -0.70710678f, 0.70710678f,
                0, -0.70710678f, 0.70710678f, 0, -0.70710678f, 0.70710678f,
                0, 0, 1, 0, 0, 1, 1, 1,
            };
            const uint32_t indices[] = {2, 0, 1, 2, 1, 3};
            const float tangents[] = {1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1};
            std::ofstream binary(directory / "mesh.bin", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(attributes), sizeof(attributes));
            binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
            binary.write(reinterpret_cast<const char*>(tangents), sizeof(tangents));
            std::ofstream gltf(path);
            gltf << R"json({
                "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
                "nodes":[{"mesh":0,"translation":[3,-2,1],"scale":[2,3,4]}],
                "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2)json";
            if (authoredTangents_) { gltf << ",\"TANGENT\":4"; }
            gltf << R"json(},"indices":3}]}],
                "buffers":[{"uri":"mesh.bin","byteLength":216}],
                "bufferViews":[
                    {"buffer":0,"byteOffset":0,"byteLength":48},
                    {"buffer":0,"byteOffset":48,"byteLength":48},
                    {"buffer":0,"byteOffset":96,"byteLength":32},
                    {"buffer":0,"byteOffset":128,"byteLength":24},
                    {"buffer":0,"byteOffset":152,"byteLength":64}],
                "accessors":[
                    {"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,0],"max":[2,1,1]},
                    {"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},
                    {"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"},
                    {"bufferView":3,"componentType":5125,"count":6,"type":"SCALAR"},
                    {"bufferView":4,"componentType":5126,"count":4,"type":"VEC4"}]
            })json";
        }


        std::string log;
        auto created = bench::createTestDevice(context, {.applicationName = "CLAS scene attributes",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayQuery = true, .enableClusterAccelerationStructure = true});
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("CLAS/ray query unavailable"); }
        if (!created) { return RHITestResult::fail("Device creation failed"); }
        auto& device = **created;
        scene::Scene scene;
        if (!scene.load(path)) { return RHITestResult::fail(scene.lastLoadResult().error); }
        static const bool registered = registerRenderGraphPassType("CLASAttributesProbe", "Streamed CLAS attributes",
            [] { return std::make_unique<CLASAttributesProbePass>(); });
        (void)registered;
        RenderGraph graph;
        graph.addNode("CLASAttributesProbe", "Probe", {{"path", path.string()}, {"native", native_}});
        graph.markOutput("Probe.hits");
        RenderGraphExecutor executor;
        executor.bindRuntimeScene(&scene);
        auto compiled = executor.compile(device, graph, 64, 64, log);
        if (native_ && hasError(compiled, Error::Unsupported)) { return RHITestResult::skip("native descriptor heap unavailable"); }
        FETCH_REQUIRE(compiled);
        for (uint32_t step = 0; step < 2; ++step) {
            const float translationX = float(step);
            auto moved = scene.nodes()[0].localMatrix;
            moved.a03 = 3.0f + translationX;
            if (step && !scene.setNodeLocalMatrix(0, moved)) { return RHITestResult::fail("instance edit failed"); }
            FETCH_REQUIRE(executor.execute({.graphicsQueue = device.getQueue(QueueType::Graphics),
                .computeQueue = device.getQueue(QueueType::Compute)}));
            FETCH_REQUIRE(executor.waitForSubmittedWork(10'000'000'000ull));
            auto* output = executor.outputResource("Probe.hits")->buffer;
            std::array<float, 96> actual{};
            const void* mapped = output->map();
            if (!mapped) { return RHITestResult::fail("probe readback map failed"); }
            output->invalidate(); std::memcpy(actual.data(), mapped, sizeof(actual)); output->unmap();
            bench::readbackEvidence(context, "readback.bin", std::span<const float>(actual));
            auto near = [](float a, float b) { return std::isfinite(a) && std::abs(a - b) < 0.0001f; };
            for (uint32_t ray = 0; ray < 3; ++ray) {
                const float* hit = actual.data() + ray * 24;
                const bool right = ray == 1;
                if (!near(hit[0], (right ? 6.0f : 4.0f) + translationX) ||
                    !near(hit[1], right ? 0.25f : -1.25f) || !near(hit[2], right ? 4.0f : 2.0f) ||
                    !near(hit[3], right ? 6.0f : (ray == 2 ? 12.0f : 8.0f)) ||
                    !near(hit[4], 0.0f) || !near(hit[5], -0.8f) || !near(hit[6], 0.6f) ||
                    !near(hit[7], ray == 2 ? 0.0f : 1.0f) ||
                    !near(hit[8], 0.0f) || !near(hit[9], -0.8f) || !near(hit[10], 0.6f) ||
                    !near(hit[11], 1.0f) ||
                    !near(hit[20], right ? 0.75f : 0.25f) || !near(hit[21], right ? 0.75f : 0.25f)) {
                    return RHITestResult::fail("incorrect transformed hit/UV/normal at ray " + std::to_string(ray) + " values=" + std::to_string(hit[0]) + "," + std::to_string(hit[1]) + "," + std::to_string(hit[2]) + "," + std::to_string(hit[3]) + "; " + std::to_string(hit[4]) + "," + std::to_string(hit[5]) + "," + std::to_string(hit[6]) + "," + std::to_string(hit[7]));
                }
                if (authoredTangents_ && (!near(hit[12], 1.0f) || !near(hit[15], -1.0f) ||
                    !near(hit[17], -0.6f) || !near(hit[18], -0.8f))) {
                    return RHITestResult::fail("authored tangent handedness or back-face TBN changed");
                }
            }
            if (!near(actual[3 * 24 + 11], 0.0f)) { return RHITestResult::fail("miss ray reported a hit"); }

        }
        return RHITestResult::pass("CLAS streamed hits, transforms, UVs, authored normals and back-face TBN");
    }
private:
    bool authoredTangents_, native_;
};
METALLIC_REGISTER_RHI_TEST(SceneCLASAttributesTest);
class SceneCLASAuthoredTangentsTest final : public SceneCLASAttributesTest {
public: SceneCLASAuthoredTangentsTest() : SceneCLASAttributesTest(true) {}
};
class SceneCLASNativeAttributesTest final : public SceneCLASAttributesTest {
public: SceneCLASNativeAttributesTest() : SceneCLASAttributesTest(true, true) {}
};
METALLIC_REGISTER_RHI_TEST(SceneCLASAuthoredTangentsTest);
METALLIC_REGISTER_RHI_TEST(SceneCLASNativeAttributesTest);
#undef FETCH_REQUIRE
} // namespace
} // namespace metallic::tests
