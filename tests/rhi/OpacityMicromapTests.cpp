#include "RhiTest.h"
#include "harness/Fixtures.h"

#include "Runtime/Render/RayTracing/OpacityMicromapBake.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GAPI/Vulkan/OpacityMicromapSpirv.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <numeric>
#include <thread>

namespace metallic::tests {
namespace {

#define OMM_REQUIRE(expression) do { \
    const render::Result<> checked = (expression); \
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
    std::optional<bench::Metadata> metadata() const override
    {
        bench::Metadata result{.suite = "extensions", .layer = bench::Layer::Core,
            .coverage = {"opacityMicromap.bake.opaque.transparent.unknown"}, .artifacts = {"bake.json"}};
        result.requirements.requiresDevice = false;
        result.requirements.validation = bench::Validation::Off;
        result.requirements.queues.clear();
        return result;
    }
    RhiTestResult runCpu(bench::Evidence& evidence) override { return check(&evidence); }
    RhiTestResult run(RhiTestContext&) override { return check(nullptr); }
private:
    RhiTestResult check(bench::Evidence* evidence)
    {
        bench::Json counts = bench::Json::array();
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
            counts.push_back({{"subdivision", level}, {"states", packedCounts}});
            if (evidence) { evidence->bytes("bake-" + std::to_string(level) + ".bin", std::as_bytes(std::span(baked.data))); }
            OMM_EXPECT(packedCounts == baked.stateCounts, "bird-curve indexing is not bijective");
            if (level >= 3) {
                OMM_EXPECT(baked.stateCounts[0] && baked.stateCounts[1] && baked.stateCounts[3], "coverage lost transparent/opaque/boundary states");
            }
        }
        scene::RenderImage::Mip edge{.width = 2, .height = 1, .pixels = {255,255,255,0, 255,255,255,255}};
        material.alphaCutoff = 0.75f;
        primitive.texcoords0 = {{0.51f,0.5f}, {0.52f,0.5f}, {0.51f,0.51f}};
        OMM_EXPECT(render::OpacityMicromapBaker(material, &edge).bake(primitive, 0, baked) && baked.stateCounts[3] == 1,
            "bilinear half-texel footprint was incorrectly classified opaque");
        primitive.texcoords0 = {{0.98f,0.5f}, {0.99f,0.5f}, {0.98f,0.51f}};
        OMM_EXPECT(render::OpacityMicromapBaker(material, &edge).bake(primitive, 0, baked) && baked.stateCounts[3] == 1,
            "bilinear repeat seam was incorrectly classified opaque");
        primitive.texcoords0 = {{0,0}, {1,0}, {0,1}};
        material.alphaCutoff = 0.5f;
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
        if (evidence) { evidence->json("bake.json", counts); }
        return RhiTestResult::pass("coverage, packed bird order, cutoff equality, constant triangles, and partial-alpha states");
    }
};
METALLIC_REGISTER_RHI_TEST(OpacityMicromapBakeTest);

class OpacityMicromapRayQueryTest : public RhiTest {
public:
    explicit OpacityMicromapRayQueryTest(bool partitioned = false) : partitioned_(partitioned)
    {
        name = partitioned ? "opacity_micromap_ray_query_partitioned" : "opacity_micromap_ray_query";
        type = RhiTestType::Rendering;
    }
    std::optional<bench::Metadata> metadata() const override
    {
        if (partitioned_) { return std::nullopt; } // Combined PTLAS/OMM remains an integration case.
        return bench::comparisonMetadata({"opacityMicromap.scene.alpha.compaction.edits.refit"}, bench::Layer::Core,
            "ray-query", {"ray-query-omm", "opacityMicromap", bench::Capability::OpacityMicromap, 0.0, 0.0, "alpha-candidates.json"});
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
            // Two distinct primitives exercise OMM scratch reuse and per-BLAS histograms.
            gltf << R"json({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1,2,3]}],
                "nodes":[{"mesh":0},{"mesh":1,"translation":[2,0,0]},{"mesh":0,"translation":[4,0,0]},{"mesh":1,"translation":[6,0,0]}],
                "meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":0}]},
                    {"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":1}]}],
                "buffers":[{"uri":"mesh.bin","byteLength":104}],
                "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":32},{"buffer":0,"byteOffset":80,"byteLength":24}],
                "accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[0,0,2],"max":[1,1,2]},
                    {"bufferView":1,"componentType":5126,"count":4,"type":"VEC2"},{"bufferView":2,"componentType":5125,"count":6,"type":"SCALAR"}],
                "images":[{"uri":"alpha.png"}],"textures":[{"source":0}],
                "materials":[{"alphaMode":"MASK","alphaCutoff":0.5,"pbrMetallicRoughness":{"baseColorTexture":{"index":0}}},
                    {"alphaMode":"MASK","pbrMetallicRoughness":{"baseColorFactor":[1,1,1,0]}}]
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
        std::array<Probe, 9> baseline{};
        uint64_t fallbackCandidates = 0, ommCandidates = 0;
        bench::Json observations = bench::Json::array();
        const auto variants = context.deviceDesc ? std::vector<bool>{context.deviceDesc->enableOpacityMicromap} : std::vector<bool>{false, true};
        for (bool enable : variants) {
            { std::ofstream restore(path); restore << originalGltf; }
            bench::TestDevice device;
            const auto setup = bench::createTestDevice(context, {.applicationName = "Opacity Micromap Test",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
                .enableOpacityMicromap = enable, .enablePartitionedAccelerationStructure = partitioned_, .enableAsyncCompute = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
            if (render::hasError(setup, render::Error::Unsupported)) { return RhiTestResult::skip("ray queries unavailable"); }
            OMM_REQUIRE(setup);
            if (partitioned_ && !device->capabilities().partitionedAccelerationStructure) {
                return RhiTestResult::skip("PTLAS unavailable");
            }
            if (enable && !device->capabilities().opacityMicromap) { return RhiTestResult::skip("fallback passed; OMM unavailable"); }
            if (!enable) {
                render::RayTracingAccelerationStructureBuildSizes sizes;
                const auto unavailable = device->queryRayTracingAccelerationStructureBuildSizes({.type = render::RayTracingAccelerationStructureType::OpacityMicromap}).transform([&](auto rhiValue) { sizes = std::move(rhiValue); });
                OMM_EXPECT(render::hasError(unavailable, render::Error::Unsupported), "disabled OMM was accepted");
            }
            auto& queue = *device->getQueue(render::QueueType::Graphics);
            scene::Scene loaded;
            OMM_EXPECT(loaded.load(path), loaded.lastLoadResult().error);
            render::ScenePathTraceResources resources;
            OMM_REQUIRE(resources.beginPrepareAsync(*device, queue, {{"path", path.string()}, {"topLevelBackend", partitioned_ ? "partitioned" : "standard"}}, loaded, log));
            bool complete = false;
            scene::SceneLoadProgress progress;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!complete && std::chrono::steady_clock::now() < deadline) {
                OMM_REQUIRE(resources.pumpPrepareAsync(10.0, progress, log).transform([&](auto value) { complete = std::move(value); }));
                if (!complete) { std::this_thread::yield(); }
            }
            OMM_EXPECT(complete && resources.valid(), "scene preparation timed out: " + log);
            OMM_EXPECT(resources.accelerationStructure().stats().opacityMicromapCount == (enable ? 4u : 0u), "scene BLAS did not bake the expected OMMs");
            OMM_EXPECT(resources.accelerationStructure().stats().compactedBlasBytes != 0, "BLAS compaction was not exercised");
            const char* capabilities[] = {"spvRayQueryKHR"};
            render::ShaderCompileResult compiled;
            const auto compile = render::compileSlangShaderToSpirv({
                .moduleName = "Features/SmokeTests/OpacityMicromapProbe",
                .entryPointName = "opacityMicromapProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders",
                .capabilities = {capabilities, 1},
            }, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); });
            log = compiled.diagnostics;
            OMM_REQUIRE(compile);
            std::vector<uint32_t> patched, twice;
            OMM_EXPECT(render::vulkan::enableOpacityMicromapSpirv(compiled.spirv, patched) && patched != compiled.spirv &&
                render::vulkan::enableOpacityMicromapSpirv(patched, twice) && patched == twice, "RayQuery OMM mode missing or not idempotent");
            OMM_EXPECT(render::vulkan::enableOpacityMicromapSpirv(compiled.spirv, patched, true) && patched != compiled.spirv &&
                render::vulkan::enableOpacityMicromapSpirv(patched, twice, true) && patched == twice, "RayQuery EXT OMM capability missing or not idempotent");
            const render::ComputeProgramBindingDesc layout[] = {
                {0, render::ComputeResourceBindingKind::AccelerationStructure}, {2}, {3}, {4}, {5}, {6},
                {9, render::ComputeResourceBindingKind::SampledImage, resources.materialTextureCount()}, {63}};
            render::ComputeProgram program;
            OMM_REQUIRE(program.initialize(*device, {
                .spirv = compiled.spirv,
                .pushConstantSize = 4,
                .bindings = {layout, uint32_t(std::size(layout))},
            }, log));
            std::unique_ptr<render::Buffer> output;
            OMM_REQUIRE(device->createBuffer({.size = sizeof(Probe), .structureStride = 8,
                .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
            render::QueueSubmissionTracker tracker;
            OMM_REQUIRE(tracker.initialize(*device, queue));
            std::unique_ptr<render::CommandPool> pool;
            std::unique_ptr<render::CommandBuffer> commands;
            OMM_REQUIRE(device->createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
            OMM_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
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
            for (uint32_t step = 0; step < (partitioned_ ? 9u : 7u); ++step) {
                if (step > 0 && step < 5) {
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
                if (step >= 5 && step < 7) {
                    auto transform = loaded.nodes()[0].localMatrix;
                    transform.a03 = step == 5 ? 3.5f : 0.0f;
                    const auto before = resources.accelerationStructure().stats();
                    const auto address = resources.accelerationStructure().accelerationStructure()->deviceAddress();
                    OMM_EXPECT(loaded.setNodeLocalMatrix(0, transform), "instance move failed");
                    OMM_REQUIRE(resources.syncRuntimeScene(&loaded, log));
                    const auto after = resources.accelerationStructure().stats();
                    OMM_EXPECT(before.compactedBlasBytes == after.compactedBlasBytes &&
                        before.opacityMicromapBytes == after.opacityMicromapBytes &&
                        address == resources.accelerationStructure().accelerationStructure()->deviceAddress(),
                        "transform update replaced BLAS/OMM/top-level storage");
                }
                if (step >= 7) {
                    // Reusing a scene with another backend must invalidate the resource cache.
                    OMM_REQUIRE(resources.beginPrepareAsync(*device, queue, {{"path", path.string()},
                        {"topLevelBackend", step == 7 ? "standard" : "partitioned"}}, loaded, log));
                    bool ready = false;
                    const auto switchDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                    while (!ready && std::chrono::steady_clock::now() < switchDeadline) {
                        OMM_REQUIRE(resources.pumpPrepareAsync(10.0, progress, log).transform([&](auto value) { ready = std::move(value); }));
                        if (!ready) { std::this_thread::yield(); }
                    }
                    OMM_EXPECT(ready && resources.valid(), "backend switch did not finish");
                }
                const bool expectPartitioned = partitioned_ && step != 7;
                OMM_EXPECT(resources.accelerationStructure().accelerationStructure()->desc().topLevelBackend ==
                    (expectPartitioned ? render::RayTracingTopLevelBackend::Partitioned : render::RayTracingTopLevelBackend::Standard),
                    "material/transform update changed the selected backend");
                OMM_EXPECT(!expectPartitioned || resources.accelerationStructure().stats().partitionCount > 1,
                    "fixture did not exercise multiple partitions");
                OMM_REQUIRE(frame.begin(step));
                OMM_REQUIRE(pool->reset());
                OMM_REQUIRE(commands->begin(&frame));
                OMM_REQUIRE(resources.uploadMaterialTextures(*commands));
                const render::ComputeDispatchBinding bindings[] = {
                    {.binding = 0, .accelerationStructure = resources.accelerationStructure().accelerationStructure()},
                    {.binding = 2, .buffer = resources.shadingVertexBuffer()}, {.binding = 3, .buffer = resources.indexBuffer()},
                    {.binding = 4, .buffer = resources.primitiveBuffer()}, {.binding = 5, .buffer = resources.instanceBuffer()},
                    {.binding = 6, .buffer = resources.materialBuffer()},
                    {.binding = 9, .textureViews = {resources.materialTextureViews().data(), resources.materialTextureCount()}},
                    {.binding = 63, .buffer = output.get()}};
                const uint32_t textureCount = uint32_t(resources.materialTextureViews().size());
                OMM_REQUIRE(program.dispatch({
                    .commandBuffer = commands.get(),
                    .bindings = {bindings, uint32_t(std::size(bindings))},
                    .pushData = &textureCount,
                    .pushDataSize = 4,
                    .groupCountX = 8,
                    .groupCountY = 8,
                }));
                OMM_REQUIRE(commands->end());
                render::CommandBuffer* submitted[] = {commands.get()};
                OMM_REQUIRE(tracker.submit({.commandBuffers = {submitted, 1}}, frame));
                OMM_REQUIRE(frame.wait(10'000'000'000ull));
                Probe actual{};
                const void* mapped = output->map();
                OMM_EXPECT(mapped != nullptr, "readback map failed");
                output->invalidate();
                std::memcpy(actual.data(), mapped, sizeof(actual));
                output->unmap();
                bench::readbackEvidence(context, "readback.bin", std::span<const std::array<uint32_t, 2>>(actual));
                uint32_t hits = 0;
                for (size_t ray = 0; ray < actual.size(); ++ray) {
                    hits += actual[ray][0];
                    // Independent CPU reference for AlphaCoverage.hlsli's four-tap
                    // mip-0 bilinear repeat sampler. Do not compare to old nearest counts.
                    const auto& material = loaded.materials()[0];
                    const float u = (float(ray % 64) + 0.31f) / 64.0f;
                    const float v = (float(ray / 64) + 0.67f) / 64.0f;
                    const auto& t = material.baseColorTexture.uvTransform;
                    const float tu = t[0] * u + t[1] * v + t[2], tv = t[3] * u + t[4] * v + t[5];
                    const float x = (tu - std::floor(tu)) * 32.0f - 0.5f;
                    const float y = (tv - std::floor(tv)) * 32.0f - 0.5f;
                    const int ix = int(std::floor(x)), iy = int(std::floor(y));
                    const float fx = x - std::floor(x), fy = y - std::floor(y);
                    const auto alpha = [&](int px, int py) {
                        return float(pixels[(((py + 32) % 32) * 32 + (px + 32) % 32) * 4 + 3]) / 255.0f;
                    };
                    const float a0 = alpha(ix, iy) * (1 - fx) + alpha(ix + 1, iy) * fx;
                    const float a1 = alpha(ix, iy + 1) * (1 - fx) + alpha(ix + 1, iy + 1) * fx;
                    const float coverage = std::clamp(material.baseColorFactor.w * (a0 * (1 - fy) + a1 * fy), 0.0f, 1.0f);
                    const bool expected = step != 5 && (material.alphaMode == "BLEND" ? coverage > 0 :
                        coverage >= std::clamp(material.alphaCutoff, 0.0f, 1.0f));
                    OMM_EXPECT(actual[ray][0] == uint32_t(expected), "bilinear alpha mismatch at step " +
                        std::to_string(step) + " ray " + std::to_string(ray) + " coverage=" + std::to_string(coverage) +
                        " actual=" + std::to_string(actual[ray][0]) + " OMM=" + std::to_string(enable));
                    if (!enable) { baseline[step][ray] = actual[ray]; fallbackCandidates += actual[ray][1]; }
                    else { ommCandidates += actual[ray][1]; }
                    observations.push_back(actual[ray][0]);
                    OMM_EXPECT(context.evidence || actual[ray][0] == baseline[step][ray][0], "OMM changed visibility at step " + std::to_string(step) + " ray " + std::to_string(ray));
                }
                if (step == 0) { OMM_EXPECT(hits > 0 && hits < 4096, "alpha fixture needs both hits and holes"); }

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
        bench::comparisonEvidence(context, {{"gltf", originalGltf}, {"alpha", pixels}, {"steps", 7}}, observations,
            context.deviceDesc && context.deviceDesc->enableOpacityMicromap);
        if (context.evidence) {
            context.evidence->json("alpha-candidates.json", {{"candidates", fallbackCandidates + ommCandidates}});
            return RhiTestResult::pass("CPU bilinear alpha oracle passed for all material/transform steps");
        }
        OMM_EXPECT(ommCandidates < fallbackCandidates / 2, "OMM did not reduce shader alpha candidates: " +
            std::to_string(fallbackCandidates) + " -> " + std::to_string(ommCandidates));
        return RhiTestResult::pass("OMM visibility equals fallback after compaction/cutoff/UV/alpha/BLEND edits, transforms and backend switches; candidates " +
            std::to_string(fallbackCandidates) + " -> " + std::to_string(ommCandidates));
    }
private:
    bool partitioned_ = false;
};
class PartitionedOpacityMicromapRayQueryTest final : public OpacityMicromapRayQueryTest {
public:
    PartitionedOpacityMicromapRayQueryTest() : OpacityMicromapRayQueryTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(OpacityMicromapRayQueryTest);
METALLIC_REGISTER_RHI_TEST(PartitionedOpacityMicromapRayQueryTest);

#undef OMM_REQUIRE
#undef OMM_EXPECT
} // namespace
} // namespace metallic::tests
