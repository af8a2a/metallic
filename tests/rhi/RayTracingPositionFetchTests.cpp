#include "RhiTest.h"

#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/RenderPass/ScenePathTraceResources.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Scene/Scene.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

namespace metallic::tests {
namespace {

#define FETCH_REQUIRE(expression) do { \
    const render::Result result = (expression); \
    if (!result) { return RhiTestResult::fail(std::string(#expression) + ": " + toString(result) + " " + log); } \
} while (false)

class SceneRayTracingPositionFetchTest : public RhiTest {
public:
    explicit SceneRayTracingPositionFetchTest(bool authoredTangents = false)
        : authoredTangents_(authoredTangents)
    {
        type = RhiTestType::Rendering;
        name = authoredTangents ? "scene_ray_tracing_position_fetch_authored_tangents" : "scene_ray_tracing_position_fetch";
    }

    RhiTestResult run(RhiTestContext& context) override
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

        // Retain fallback results for both the original and refitted instance.
        std::array<std::array<float, 96>, 2> baseline{};
        for (bool positionFetch : {false, true}) {
            std::string log;
            std::unique_ptr<render::Device> device;
            const auto setup = render::createDevice({
                .applicationName = "Scene position fetch test",
                .enableValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true,
                .enableRayTracingAccelerationStructure = true,
                .enableRayQuery = true,
                .enableRayTracingPositionFetch = positionFetch,
            }, device);
            if (!setup && render::hasError(setup, render::Error::Unsupported)) {
                return RhiTestResult::skip("ray query/descriptor heap unavailable");
            }
            FETCH_REQUIRE(setup);
            if (!device->capabilities().rayQuery || !device->capabilities().bindlessDescriptorHeap) {
                return RhiTestResult::skip("ray query/descriptor heap unavailable");
            }
            if (positionFetch && !device->capabilities().rayTracingPositionFetch) {
                return RhiTestResult::skip("fallback passed; position fetch unavailable on this device");
            }
            if (!positionFetch) {
                render::RayTracingAccelerationStructureBuildSizes sizes;
                const auto unavailable = device->queryRayTracingAccelerationStructureBuildSizes({
                    .flags = render::RayTracingAccelerationStructureBuildFlags::AllowDataAccess,
                }, sizes);
                if (device->capabilities().rayTracingPositionFetch ||
                    !render::hasError(unavailable, render::Error::Unsupported)) {
                    return RhiTestResult::fail("disabled position fetch accepted a data-access BLAS");
                }
            }
            auto* queue = device->getQueue(render::QueueType::Graphics);
            if (queue == nullptr) { return RhiTestResult::fail("graphics queue unavailable"); }
            scene::Scene scene;
            if (!scene.load(path)) { return RhiTestResult::fail(scene.lastLoadResult().error); }
            render::ScenePathTraceResources resources;
            FETCH_REQUIRE(resources.beginPrepareAsync(*device, *queue, {{"path", path.string()}}, scene, log));
            bool complete = false;
            scene::SceneLoadProgress progress;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!complete && std::chrono::steady_clock::now() < deadline) {
                FETCH_REQUIRE(resources.pumpPrepareAsync(10.0, complete, progress, log));
                if (!complete) { std::this_thread::yield(); }
            }
            if (!complete || !resources.valid()) { return RhiTestResult::fail("scene preparation timed out: " + log); }
            if (resources.accelerationStructure().stats().compactedBlasBytes == 0) {
                return RhiTestResult::fail("fixture did not exercise BLAS compaction");
            }
            render::RayTracingAccelerationStructureProperties accelerationProperties;
            FETCH_REQUIRE(device->queryRayTracingAccelerationStructureProperties(accelerationProperties));
            if (resources.accelerationStructure().stats().geometryBytes != accelerationProperties.instanceRecordSize) {
                return RhiTestResult::fail("BLAS build-only vertex/index buffers remained resident");
            }
            if (resources.shadingVertexBuffer()->desc().structureStride != 16 ||
                resources.shadingVertexBuffer()->desc().size != 4 * 16 ||
                (resources.fallbackPositionBuffer() == nullptr) != positionFetch ||
                (!positionFetch && (resources.fallbackPositionBuffer()->desc().structureStride != 12 ||
                    resources.fallbackPositionBuffer()->desc().size != 4 * 12))) {
                return RhiTestResult::fail("compact shading/fallback position buffer layout mismatch");
            }

            const char* capabilities[] = {"spvRayQueryKHR", "spvRayQueryPositionFetchKHR"};
            const render::SlangMacroDefine defines[] = {
                {"SCENE_RAYQUERY_ENABLE_POSITION_FETCH", positionFetch ? "1" : "0"},
            };
            render::ShaderCompileResult shader;
            const auto compiled = render::compileSlangShaderToSpirv({
                .moduleName = "Features/SmokeTests/ScenePositionFetchProbe",
                .entryPointName = "scenePositionFetchProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                .capabilities = capabilities,
                .capabilityCount = positionFetch ? 2u : 1u,
                .macroDefines = defines,
                .macroDefineCount = 1,
            }, shader);
            log = shader.diagnostics;
            FETCH_REQUIRE(compiled);
            std::vector<render::ComputeProgramBindingDesc> layout = {
                {0, render::ComputeResourceBindingKind::AccelerationStructure},
                {2}, {3}, {4}, {5}, {6},
                {9, render::ComputeResourceBindingKind::SampledImage, render::kScenePathTraceMaxMaterialTextures},
                {63},
            };
            if (!positionFetch) {
                layout.push_back({render::kSceneFallbackPositionsBinding});
            }
            render::ComputeProgram program;
            FETCH_REQUIRE(program.initialize(*device, {
                .spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * sizeof(uint32_t),
                .pushConstantSize = sizeof(float), .bindings = layout.data(),
                .bindingCount = static_cast<uint32_t>(std::size(layout)),
            }, log));
            std::unique_ptr<render::Buffer> output;
            FETCH_REQUIRE(device->createBuffer({
                .size = sizeof(baseline[0]), .structureStride = 4 * sizeof(float),
                .usage = render::BufferUsageBits::Storage,
                .memoryLocation = render::MemoryLocation::HostReadback,
            }, output));
            render::QueueSubmissionTracker tracker;
            FETCH_REQUIRE(tracker.initialize(*device, *queue));
            std::unique_ptr<render::CommandPool> pool;
            std::unique_ptr<render::CommandBuffer> commands;
            FETCH_REQUIRE(device->createCommandPool(*queue, pool));
            FETCH_REQUIRE(pool->createCommandBuffer(commands));
            render::RenderFrameContext frame;
            // Drain before destroying buffers/pipelines on every return path.
            struct Drain {
                render::RenderFrameContext& frame;
                render::CommandPool& pool;
                ~Drain()
                {
                    if (frame.completion().isSubmitted()) { (void)frame.wait(); }
                    (void)pool.reset();
                    (void)frame.reset();
                }
            } drain{frame, *pool};
            for (uint32_t step = 0; step < 2; ++step) {
                const float translationX = static_cast<float>(step);
                if (step != 0) {
                    auto moved = scene.nodes()[0].localMatrix;
                    moved.a03 += translationX;
                    if (!scene.setNodeLocalMatrix(0, moved)) { return RhiTestResult::fail("instance edit failed"); }
                    FETCH_REQUIRE(resources.syncRuntimeScene(&scene, log));
                }
                FETCH_REQUIRE(frame.begin(step));
                FETCH_REQUIRE(pool->reset());
                FETCH_REQUIRE(commands->begin(&frame));
                FETCH_REQUIRE(resources.uploadMaterialTextures(*commands));
                std::vector<render::ComputeDispatchBinding> bindings = {
                    {.binding = 0, .accelerationStructure = resources.accelerationStructure().accelerationStructure()},
                    {.binding = 2, .buffer = resources.shadingVertexBuffer()},
                    {.binding = 3, .buffer = resources.indexBuffer()},
                    {.binding = 4, .buffer = resources.primitiveBuffer()},
                    {.binding = 5, .buffer = resources.instanceBuffer()},
                    {.binding = 6, .buffer = resources.materialBuffer()},
                    {.binding = 9, .textureViews = resources.materialTextureViews().data(),
                        .textureViewCount = render::kScenePathTraceMaxMaterialTextures},
                    {.binding = 63, .buffer = output.get()},
                };
                if (!positionFetch) {
                    bindings.push_back({.binding = render::kSceneFallbackPositionsBinding,
                        .buffer = resources.fallbackPositionBuffer()});
                }
                FETCH_REQUIRE(program.dispatch({
                    .commandBuffer = commands.get(), .bindings = bindings.data(),
                    .bindingCount = static_cast<uint32_t>(std::size(bindings)),
                    .pushData = &translationX, .pushDataSize = sizeof(translationX),
                }));
                FETCH_REQUIRE(commands->end());
                render::CommandBuffer* submitted[] = {commands.get()};
                FETCH_REQUIRE(tracker.submit({.commandBuffers = submitted, .commandBufferCount = 1}, frame));
                FETCH_REQUIRE(frame.wait(10'000'000'000ull));
                std::array<float, 96> actual{};
                const void* mapped = output->map();
                if (mapped == nullptr) { return RhiTestResult::fail("probe readback map failed"); }
                output->invalidate();
                std::memcpy(actual.data(), mapped, sizeof(actual));
                output->unmap();
                auto near = [](float a, float b) { return std::isfinite(a) && std::abs(a - b) < 0.0001f; };
                for (uint32_t ray = 0; ray < 3; ++ray) {
                    const float* hit = actual.data() + ray * 24;
                    const bool right = ray == 1;
                    if (!near(hit[0], (right ? 6.0f : 4.0f) + translationX) ||
                        !near(hit[1], right ? 0.25f : -1.25f) || !near(hit[2], right ? 4.0f : 2.0f) ||
                        !near(hit[3], right ? 6.0f : (ray == 2 ? 12.0f : 8.0f)) ||
                        !near(hit[7], ray == 2 ? 0.0f : 1.0f) ||
                        !near(hit[8], 0.0f) || !near(hit[9], -0.8f) || !near(hit[10], 0.6f) ||
                        !near(hit[11], 1.0f) || !near(hit[19], right ? 1.0f : 0.0f) ||
                        !near(hit[20], right ? 0.75f : 0.25f) || !near(hit[21], right ? 0.75f : 0.25f)) {
                        return RhiTestResult::fail("incorrect transformed hit/UV/normal at ray " + std::to_string(ray));
                    }
                    if (authoredTangents_ && (!near(hit[12], 1.0f) || !near(hit[15], -1.0f) ||
                        !near(hit[17], -0.6f) || !near(hit[18], -0.8f))) {
                        return RhiTestResult::fail("authored tangent handedness or back-face TBN changed");
                    }
                }
                if (!near(actual[3 * 24 + 11], 0.0f)) { return RhiTestResult::fail("miss ray reported a hit"); }
                if (!positionFetch) { baseline[step] = actual; }
                for (size_t value = 0; value < actual.size(); ++value) {
                    if (!near(actual[value], baseline[step][value])) {
                        return RhiTestResult::fail("position fetch changed a hit attribute at float " + std::to_string(value) +
                            ": fetch=" + std::to_string(actual[value]) + ", fallback=" + std::to_string(baseline[step][value]));
                    }
                }
            }
        }
        return RhiTestResult::pass("fetch/fallback agree after BLAS compaction and TLAS refit, including UVs and back-face TBN");
    }

private:
    bool authoredTangents_ = false;
};

METALLIC_REGISTER_RHI_TEST(SceneRayTracingPositionFetchTest);

class ScenePositionFetchAuthoredTangentsTest final : public SceneRayTracingPositionFetchTest {
public:
    ScenePositionFetchAuthoredTangentsTest() : SceneRayTracingPositionFetchTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(ScenePositionFetchAuthoredTangentsTest);

class SceneShadingVertexPackingTest final : public RhiTest {
public:
    SceneShadingVertexPackingTest()
    {
        type = RhiTestType::Rendering;
        name = "scene_shading_vertex_packing";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::string log;
        std::unique_ptr<render::Device> device;
        FETCH_REQUIRE(render::createDevice({.applicationName = "Compact scene vertices",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device));
        if (!device->capabilities().bindlessDescriptorHeap) { return RhiTestResult::skip("descriptor heap unavailable"); }
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        std::vector<std::array<float, 3>> directions{
            {0, 0, 0}, {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
        };
        for (uint32_t index = 0; index < 512; ++index) {
            const float z = 1.0f - 2.0f * (static_cast<float>(index) + 0.5f) / 512.0f;
            const float radius = std::sqrt(1.0f - z * z);
            const float angle = static_cast<float>(index) * 2.39996323f;
            directions.push_back({radius * std::cos(angle), radius * std::sin(angle), z});
        }
        std::vector<render::SceneShadingVertex> vertices;
        for (size_t index = 0; index < directions.size(); ++index) {
            const auto& d = directions[index];
            vertices.push_back({
                .normal = render::packSceneNormal(d[0], d[1], d[2]),
                .tangent = render::packSceneTangent(-d[2], d[0], d[1], index % 2 == 0 ? -1.0f : 1.0f),
                .texcoord = {65536.125f + static_cast<float>(index), -4096.25f - static_cast<float>(index) * 0.125f},
            });
        }
        std::vector<std::array<float, 12>> actual(vertices.size());
        std::unique_ptr<render::Buffer> input, output;
        FETCH_REQUIRE(device->createBuffer({.size = vertices.size() * sizeof(vertices[0]), .structureStride = 16,
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostUpload}, input));
        FETCH_REQUIRE(device->createBuffer({.size = actual.size() * sizeof(actual[0]), .structureStride = 16,
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}, output));
        void* mapped = input->map();
        if (mapped == nullptr) { return RhiTestResult::fail("packing input map failed"); }
        std::memcpy(mapped, vertices.data(), vertices.size() * sizeof(vertices[0]));
        input->flush(); input->unmap();
        render::ShaderCompileResult shader;
        const auto compiled = render::compileSlangShaderToSpirv({
            .moduleName = "Features/SmokeTests/SceneShadingVertexProbe", .entryPointName = "sceneShadingVertexProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
        }, shader);
        log = shader.diagnostics;
        FETCH_REQUIRE(compiled);
        const render::ComputeProgramBindingDesc layout[] = {{2}, {63}};
        render::ComputeProgram program;
        FETCH_REQUIRE(program.initialize(*device, {
            .spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * sizeof(uint32_t),
            .bindings = layout, .bindingCount = 2, .requiresRayQuery = false,
        }, log));
        render::QueueSubmissionTracker tracker;
        FETCH_REQUIRE(tracker.initialize(*device, queue));
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        FETCH_REQUIRE(device->createCommandPool(queue, pool));
        FETCH_REQUIRE(pool->createCommandBuffer(commands));
        render::RenderFrameContext frame;
        struct Drain {
            render::RenderFrameContext& frame;
            render::CommandPool& pool;
            ~Drain()
            {
                if (frame.completion().isSubmitted()) { (void)frame.wait(); }
                (void)pool.reset();
                (void)frame.reset();
            }
        } drain{frame, *pool};
        FETCH_REQUIRE(frame.begin(0));
        FETCH_REQUIRE(commands->begin(&frame));
        const render::ComputeDispatchBinding bindings[] = {{.binding = 2, .buffer = input.get()}, {.binding = 63, .buffer = output.get()}};
        FETCH_REQUIRE(program.dispatch({.commandBuffer = commands.get(), .bindings = bindings, .bindingCount = 2,
            .groupCountX = static_cast<uint32_t>(vertices.size())}));
        FETCH_REQUIRE(commands->end());
        render::CommandBuffer* submitted[] = {commands.get()};
        FETCH_REQUIRE(tracker.submit({.commandBuffers = submitted, .commandBufferCount = 1}, frame));
        FETCH_REQUIRE(frame.wait(10'000'000'000ull));
        mapped = output->map();
        if (mapped == nullptr) { return RhiTestResult::fail("packing readback map failed"); }
        output->invalidate();
        std::memcpy(actual.data(), mapped, actual.size() * sizeof(actual[0]));
        output->unmap();
        float maximumNormalError = 0.0f, maximumTangentError = 0.0f;
        for (size_t index = 0; index < vertices.size(); ++index) {
            const auto& d = directions[index];
            const float tangent[] = {-d[2], d[0], d[1]};
            for (size_t component = 0; component < 3; ++component) {
                const float normalError = std::abs(actual[index][component] - d[component]);
                const float tangentError = std::abs(actual[index][4 + component] - tangent[component]);
                if (!std::isfinite(normalError) || normalError > 0.00015f ||
                    !std::isfinite(tangentError) || tangentError > 0.0003f) {
                    return RhiTestResult::fail("CPU packing / GPU decoding direction mismatch at vertex " + std::to_string(index));
                }
                maximumNormalError = std::max(maximumNormalError, normalError);
                maximumTangentError = std::max(maximumTangentError, tangentError);
            }
            if (actual[index][7] != (index % 2 == 0 ? -1.0f : 1.0f) ||
                std::memcmp(actual[index].data() + 8, vertices[index].texcoord, sizeof(vertices[index].texcoord)) != 0) {
                return RhiTestResult::fail("tangent sign or full-precision UV changed during packing");
            }
        }
        return RhiTestResult::pass("519 directions: maximum component error normal=" + std::to_string(maximumNormalError) +
            ", tangent=" + std::to_string(maximumTangentError) + "; zero vectors, handedness and float32 UV preserved");
    }
};
METALLIC_REGISTER_RHI_TEST(SceneShadingVertexPackingTest);

#undef FETCH_REQUIRE

} // namespace
} // namespace metallic::tests
