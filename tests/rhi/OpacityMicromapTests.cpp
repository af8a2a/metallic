#include "RhiTest.h"

#include "Runtime/Render/RayTracing/OpacityMicromapBake.h"
#include "Runtime/Render/RenderPass/ScenePathTraceResources.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/GAPI/Vulkan/OpacityMicromapSpirv.h"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <numeric>
#include <thread>

namespace metallic::tests {
namespace {

#define OMM_REQUIRE(expression) do { \
    const render::Result checked = (expression); \
    if (!checked) { return RhiTestResult::fail(std::string(#expression) + ": " + toString(checked) + " " + log); } \
} while (false)
#define OMM_EXPECT(expression, message) do { if (!(expression)) { return RhiTestResult::fail(message); } } while (false)

class OpacityMicromapBakeTest final : public RhiTest {
public:
    OpacityMicromapBakeTest()
    {
        name = "opacity_micromap_bake";
        type = RhiTestType::Resource;
    }
    RhiTestResult run(RhiTestContext&) override
    {
        scene::RenderPrimitive primitive;
        primitive.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        primitive.texcoords0 = {{0, 0}, {1, 0}, {0, 1}};
        scene::RenderMaterial material;
        material.alphaMode = "MASK";
        scene::RenderImage::Mip image{.width = 8, .height = 8, .pixels = std::vector<uint8_t>(8 * 8 * 4, 255)};
        for (size_t y = 0; y < 8; ++y) {
            for (size_t x = 0; x < 4; ++x) { image.pixels[(y * 8 + x) * 4 + 3] = 0; }
        }
        render::BakedOpacityMicromap baked;
        for (uint32_t level = 0; level <= 5; ++level) {
            OMM_EXPECT(render::OpacityMicromapBaker(material, &image).bake(primitive, level, baked), "bake failed");
            OMM_EXPECT(std::accumulate(baked.stateCounts.begin(), baked.stateCounts.end(), uint64_t(0)) == (1u << (2 * level)), "subdivision coverage mismatch");
            std::array<uint64_t, 4> packedCounts{};
            for (uint32_t i = 0; i < (1u << (2 * level)); ++i) {
                ++packedCounts[(baked.data[i / 4] >> ((i % 4) * 2)) & 3];
            }
            OMM_EXPECT(packedCounts == baked.stateCounts, "bird-curve indexing is not bijective");
            if (level >= 3) {
                OMM_EXPECT(baked.stateCounts[0] && baked.stateCounts[1] && baked.stateCounts[3], "coverage lost transparent/opaque/boundary states");
            }
        }
        material.baseColorFactor.w = 0;
        OMM_EXPECT(render::OpacityMicromapBaker(material, &image).bake(primitive, 4, baked) && baked.stateCounts[0] == 1 && baked.data.size() == 1, "constant-transparent triangle was not collapsed");
        material.alphaCutoff = 0;
        OMM_EXPECT(render::OpacityMicromapBaker(material, &image).bake(primitive, 4, baked) && baked.stateCounts[1] == 1, "cutoff equality must accept alpha zero");
        material.alphaMode = "BLEND";
        material.baseColorFactor.w = 0.5f;
        OMM_EXPECT(render::OpacityMicromapBaker(material, nullptr).bake(primitive, 4, baked) && baked.stateCounts[3] == 256, "partial BLEND alpha must stay unknown");
        material.baseColorFactor.w = 1.0f;
        OMM_EXPECT(render::OpacityMicromapBaker(material, nullptr).bake(primitive, 4, baked) && baked.stateCounts[1] == 1, "constant BLEND alpha one should be opaque");
        std::vector<uint32_t> invalid = {0x07230203, 0x10600, 0, 1, 0, 0};
        std::vector<uint32_t> patched;
        OMM_EXPECT(!render::vulkan::enableOpacityMicromapSpirv(invalid, patched), "invalid SPIR-V instruction accepted");
        return RhiTestResult::pass("coverage, packed bird order, cutoff equality, constant triangles, and partial-alpha states");
    }
};
METALLIC_REGISTER_RHI_TEST(OpacityMicromapBakeTest);

class OpacityMicromapRayQueryTest final : public RhiTest {
public:
    OpacityMicromapRayQueryTest()
    {
        name = "opacity_micromap_ray_query";
        type = RhiTestType::Rendering;
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::string log;
        const auto directory = context.outputDirectory / name;
        std::filesystem::create_directories(directory);
        const auto path = directory / "scene.gltf";
        {
            const float vertices[] = {0,0,2, 1,0,2, 0,1,2, 1,1,2, 0,0, 1,0, 0,1, 1,1};
            // Deliberately nonsequential vertex order to exercise barycentrics.
            const uint32_t indices[] = {2,0,1, 2,1,3};
            std::ofstream binary(directory / "mesh.bin", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(vertices), sizeof(vertices));
            binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
            std::ofstream gltf(path);
            gltf << R"json({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
                "nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":0}]}],
                "buffers":[{"uri":"mesh.bin","byteLength":104}],
                "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":32},{"buffer":0,"byteOffset":80,"byteLength":24}],
                "accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,2],"max":[1,1,2]},
                    {"bufferView":1,"componentType":5126,"count":4,"type":"VEC2"},{"bufferView":2,"componentType":5125,"count":6,"type":"SCALAR"}],
                "images":[{"uri":"alpha.png"}],"textures":[{"source":0}],
                "materials":[{"alphaMode":"MASK","alphaCutoff":0.5,"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}]
            })json";
        }
        std::vector<uint8_t> pixels(32 * 32 * 4, 255);
        for (size_t y = 0; y < 32; ++y) {
            for (size_t x = 0; x < 32; ++x) {
                pixels[(y * 32 + x) * 4 + 3] = x < 16 ? 0 : y < 16 ? 180 : 255;
            }
        }
        OMM_EXPECT(saveRgba8Png(directory / "alpha.png", pixels.data(), 32, 32, log), "could not save alpha fixture");
        std::ifstream sourceFile(path);
        const std::string originalGltf((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
        sourceFile.close();
        using Probe = std::array<std::array<uint32_t, 2>, 64 * 64>;
        std::array<Probe, 5> baseline{};
        uint64_t fallbackCandidates = 0, ommCandidates = 0;
        for (bool enable : {false, true}) {
            { std::ofstream restore(path); restore << originalGltf; }
            std::unique_ptr<render::Device> device;
            const auto setup = render::createDevice({.applicationName = "KHR Opacity Micromap Test",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
                .enableOpacityMicromap = enable}, device);
            if (render::hasError(setup, render::Error::Unsupported)) { return RhiTestResult::skip("ray queries unavailable"); }
            OMM_REQUIRE(setup);
            if (enable && !device->capabilities().opacityMicromap) { return RhiTestResult::skip("fallback passed; KHR OMM unavailable"); }
            if (!enable) {
                render::RayTracingAccelerationStructureBuildSizes sizes;
                const auto unavailable = device->queryRayTracingAccelerationStructureBuildSizes({.type = render::RayTracingAccelerationStructureType::OpacityMicromap}, sizes);
                OMM_EXPECT(render::hasError(unavailable, render::Error::Unsupported), "disabled OMM was accepted");
            }
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            scene::Scene loaded;
            OMM_EXPECT(loaded.load(path), loaded.lastLoadResult().error);
            render::ScenePathTraceResources resources;
            OMM_REQUIRE(resources.beginPrepareAsync(*device, queue, {{"path", path.string()}}, loaded, log));
            bool complete = false;
            scene::SceneLoadProgress progress;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!complete && std::chrono::steady_clock::now() < deadline) {
                OMM_REQUIRE(resources.pumpPrepareAsync(10.0, complete, progress, log));
                if (!complete) { std::this_thread::yield(); }
            }
            OMM_EXPECT(complete && resources.valid(), "scene preparation timed out: " + log);
            OMM_EXPECT(resources.accelerationStructure().stats().opacityMicromapCount == (enable ? 1u : 0u), "scene BLAS did not bake the expected OMM");
            OMM_EXPECT(resources.accelerationStructure().stats().compactedBlasBytes != 0, "BLAS compaction was not exercised");
            const char* capabilities[] = {"spvRayQueryKHR"};
            render::ShaderCompileResult compiled;
            const auto compile = render::compileSlangShaderToSpirv({.moduleName = "Features/SmokeTests/OpacityMicromapProbe",
                .entryPointName = "opacityMicromapProbeMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                .capabilities = capabilities, .capabilityCount = 1}, compiled);
            log = compiled.diagnostics;
            OMM_REQUIRE(compile);
            std::vector<uint32_t> patched, twice;
            OMM_EXPECT(render::vulkan::enableOpacityMicromapSpirv(compiled.spirv, patched) && patched != compiled.spirv &&
                render::vulkan::enableOpacityMicromapSpirv(patched, twice) && patched == twice, "RayQuery OMM mode missing or not idempotent");
            const render::ComputeProgramBindingDesc layout[] = {
                {0, render::ComputeResourceBindingKind::AccelerationStructure}, {2}, {3}, {4}, {5}, {6},
                {9, render::ComputeResourceBindingKind::SampledImage, render::kScenePathTraceMaxMaterialTextures}, {63}};
            render::ComputeProgram program;
            OMM_REQUIRE(program.initialize(*device, {.spirv = compiled.spirv.data(), .byteSize = compiled.spirv.size() * 4,
                .pushConstantSize = 4, .bindings = layout, .bindingCount = uint32_t(std::size(layout))}, log));
            std::unique_ptr<render::Buffer> output;
            OMM_REQUIRE(device->createBuffer({.size = sizeof(Probe), .structureStride = 8,
                .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}, output));
            render::QueueSubmissionTracker tracker;
            OMM_REQUIRE(tracker.initialize(*device, queue));
            std::unique_ptr<render::CommandPool> pool;
            std::unique_ptr<render::CommandBuffer> commands;
            OMM_REQUIRE(device->createCommandPool(queue, pool));
            OMM_REQUIRE(pool->createCommandBuffer(commands));
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
            for (uint32_t step = 0; step < baseline.size(); ++step) {
                if (step != 0) {
                    auto material = loaded.materials()[0];
                    if (step == 1) { material.alphaCutoff = 0.9f; }
                    if (step == 2) {
                        // The scalar material editor intentionally preserves texture bindings.
                        // Import an edited KHR_texture_transform to exercise rotation/wrap/rebuild.
                        std::string transformed = originalGltf;
                        const std::string original = "\"baseColorTexture\":{\"index\":0}";
                        transformed.replace(transformed.find(original), original.size(),
                            R"json("baseColorTexture":{"index":0,"extensions":{"KHR_texture_transform":{"offset":[1.2,-0.35],"rotation":1.57079632679,"scale":[1.5,1]}}})json");
                        { std::ofstream changed(path); changed << transformed; }
                        OMM_EXPECT(loaded.load(path), loaded.lastLoadResult().error);
                        material = loaded.materials()[0];
                        material.alphaCutoff = 0.9f;
                    }
                    if (step == 3) { material.baseColorFactor.w = 0; }
                    if (step == 4) { material.alphaMode = "BLEND"; material.baseColorFactor.w = 1; }
                    OMM_EXPECT(loaded.setMaterialProperties(0, material), "material edit failed");
                    OMM_REQUIRE(resources.syncRuntimeScene(&loaded, log));
                }
                OMM_REQUIRE(frame.begin(step));
                OMM_REQUIRE(pool->reset());
                OMM_REQUIRE(commands->begin(&frame));
                OMM_REQUIRE(resources.uploadMaterialTextures(*commands));
                const render::ComputeDispatchBinding bindings[] = {
                    {.binding = 0, .accelerationStructure = resources.accelerationStructure().accelerationStructure()},
                    {.binding = 2, .buffer = resources.shadingVertexBuffer()}, {.binding = 3, .buffer = resources.indexBuffer()},
                    {.binding = 4, .buffer = resources.primitiveBuffer()}, {.binding = 5, .buffer = resources.instanceBuffer()},
                    {.binding = 6, .buffer = resources.materialBuffer()},
                    {.binding = 9, .textureViews = resources.materialTextureViews().data(), .textureViewCount = render::kScenePathTraceMaxMaterialTextures},
                    {.binding = 63, .buffer = output.get()}};
                const uint32_t textureCount = uint32_t(resources.materialTextureViews().size());
                OMM_REQUIRE(program.dispatch({.commandBuffer = commands.get(), .bindings = bindings,
                    .bindingCount = uint32_t(std::size(bindings)), .pushData = &textureCount, .pushDataSize = 4,
                    .groupCountX = 8, .groupCountY = 8}));
                OMM_REQUIRE(commands->end());
                render::CommandBuffer* submitted[] = {commands.get()};
                OMM_REQUIRE(tracker.submit({.commandBuffers = submitted, .commandBufferCount = 1}, frame));
                OMM_REQUIRE(frame.wait(10'000'000'000ull));
                Probe actual{};
                const void* mapped = output->map();
                OMM_EXPECT(mapped != nullptr, "readback map failed");
                output->invalidate();
                std::memcpy(actual.data(), mapped, sizeof(actual));
                output->unmap();
                uint32_t hits = 0;
                for (size_t ray = 0; ray < actual.size(); ++ray) {
                    hits += actual[ray][0];
                    if (!enable) { baseline[step][ray] = actual[ray]; fallbackCandidates += actual[ray][1]; }
                    else { ommCandidates += actual[ray][1]; }
                    OMM_EXPECT(actual[ray][0] == baseline[step][ray][0], "OMM changed visibility at step " + std::to_string(step) + " ray " + std::to_string(ray));
                }
                if (step == 0) { OMM_EXPECT(hits == 2048, "initial alpha mask incorrect"); }
                if (step == 1) { OMM_EXPECT(hits == 1024, "cutoff edit did not change visibility"); }
                if (step == 3) { OMM_EXPECT(hits == 0, "alpha edit retained stale OMM"); }
                if (enable) {
                    std::vector<uint8_t> png(64 * 64 * 4, 255);
                    for (size_t ray = 0; ray < actual.size(); ++ray) {
                        png[ray * 4] = png[ray * 4 + 1] = png[ray * 4 + 2] = actual[ray][0] ? 255 : 0;
                    }
                    OMM_EXPECT(saveRgba8Png(directory / ("visibility-" + std::to_string(step) + ".png"), png.data(), 64, 64, log), "could not save output");
                }
            }
        }
        OMM_EXPECT(ommCandidates < fallbackCandidates / 2, "OMM did not reduce shader alpha candidates");
        return RhiTestResult::pass("KHR OMM visibility equals fallback after compaction/cutoff/UV/alpha/BLEND edits; candidates " +
            std::to_string(fallbackCandidates) + " -> " + std::to_string(ommCandidates));
    }
};
METALLIC_REGISTER_RHI_TEST(OpacityMicromapRayQueryTest);

#undef OMM_REQUIRE
#undef OMM_EXPECT
} // namespace
} // namespace metallic::tests
