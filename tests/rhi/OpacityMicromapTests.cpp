#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "RHITest.h"
#include "harness/Fixtures.h"
#include "harness/RayQueryFixture.h"
#include "TestResourceParameters.h"
#include "TestComputeProgram.h"

#include "Runtime/Render/GAPI/Vulkan/OpacityMicromapBake.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GAPI/Vulkan/OpacityMicromapSPIRV.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "Runtime/Render/RayTracing/SceneCoverage.h"

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
    const auto& checked = (expression); \
    if (!checked) { return RHITestResult::fail(std::string(#expression) + ": " + render::resultToString(checked) + " " + log); } \
} while (false)
#define OMM_EXPECT(expression, message) do { if (!(expression)) { return RHITestResult::fail(message); } } while (false)

class OpacityMicromapBakeTest final : public RHITest {
public:
    OpacityMicromapBakeTest()
    {
        name = "opacity_micromap_bake";
        type = RHITestType::Resource;
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
    RHITestResult runCpu(bench::Evidence& evidence) override { return check(&evidence); }
    RHITestResult run(RHITestContext&) override { return check(nullptr); }
private:
    RHITestResult check(bench::Evidence* evidence)
    {
        bench::Json counts = bench::Json::array();
        render::RayTracingCoverageTriangle triangle;
        triangle.uv = {{{0, 0}, {1, 0}, {0, 1}}};
        std::vector<uint8_t> pixels(8 * 8 * 4, 255);
        for (size_t y = 0; y < 8; ++y) {
            for (size_t x = 0; x < 4; ++x) { pixels[(y * 8 + x) * 4 + 3] = 0; }
        }
        render::RayTracingCoverageDesc coverage{.width = 8, .height = 8, .pixelsRGBA8 = pixels, .triangles = {&triangle, 1}};
        render::detail::BakedOpacityMicromap baked;
        for (uint32_t level = 0; level <= 5; ++level) {
            OMM_EXPECT(render::detail::OpacityMicromapBaker(coverage).bake(level, baked), "bake failed");
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
        const std::array<uint8_t, 8> edge{255,255,255,0, 255,255,255,255};
        coverage.width = 2; coverage.height = 1; coverage.pixelsRGBA8 = edge;
        coverage.alphaCutoff = 0.75f;
        triangle.uv = {{{0.51f,0.5f}, {0.52f,0.5f}, {0.51f,0.51f}}};
        OMM_EXPECT(render::detail::OpacityMicromapBaker(coverage).bake(0, baked) && baked.stateCounts[3] == 1,
            "bilinear half-texel footprint was incorrectly classified opaque");
        triangle.uv = {{{0.98f,0.5f}, {0.99f,0.5f}, {0.98f,0.51f}}};
        OMM_EXPECT(render::detail::OpacityMicromapBaker(coverage).bake(0, baked) && baked.stateCounts[3] == 1,
            "bilinear repeat seam was incorrectly classified opaque");
        triangle.uv = {{{0,0}, {1,0}, {0,1}}};
        coverage.width = 8; coverage.height = 8; coverage.pixelsRGBA8 = pixels;
        coverage.alphaCutoff = 0.5f;
        coverage.alphaFactor = 0;
        OMM_EXPECT(render::detail::OpacityMicromapBaker(coverage).bake(4, baked) && baked.stateCounts[0] == 1 && baked.data.size() == 1, "constant-transparent triangle was not collapsed");
        coverage.alphaCutoff = 0;
        OMM_EXPECT(render::detail::OpacityMicromapBaker(coverage).bake(4, baked) && baked.stateCounts[1] == 1, "cutoff equality must accept alpha zero");
        coverage.mode = render::RayTracingCoverageMode::Blend;
        coverage.pixelsRGBA8 = {};
        coverage.alphaFactor = 0.5f;
        OMM_EXPECT(render::detail::OpacityMicromapBaker(coverage).bake(4, baked) && baked.stateCounts[3] == 256, "partial BLEND alpha must stay unknown");
        coverage.alphaFactor = 1.0f;
        OMM_EXPECT(render::detail::OpacityMicromapBaker(coverage).bake(4, baked) && baked.stateCounts[1] == 1, "constant BLEND alpha one should be opaque");
        std::vector<uint32_t> invalid = {0x07230203, 0x10600, 0, 1, 0, 0};
        std::vector<uint32_t> patched;
        OMM_EXPECT(!render::vulkan::enableOpacityMicromapSpirv(invalid, patched), "invalid SPIR-V instruction accepted");
        if (evidence) { evidence->json("bake.json", counts); }
        return RHITestResult::pass("coverage, packed bird order, cutoff equality, constant triangles, and partial-alpha states");
    }
};
METALLIC_REGISTER_RHI_TEST(OpacityMicromapBakeTest);

class SceneCoverageSnapshotTest final : public RHITest {
public:
    SceneCoverageSnapshotTest() { name = "opacity_micromap_scene_coverage_snapshot"; type = RHITestType::Resource; }
    std::optional<bench::Metadata> metadata() const override
    {
        bench::Metadata result{.suite = "extensions", .layer = bench::Layer::Core,
            .coverage = {"coverage.snapshot.canonicalUV.sourceIndependence"}};
        result.requirements.requiresDevice = false;
        result.requirements.validation = bench::Validation::Off;
        result.requirements.queues.clear();
        return result;
    }
    RHITestResult runCpu(bench::Evidence& evidence) override { return check(evidence.root()); }
    RHITestResult run(RHITestContext& context) override { return check(context.outputDirectory / name); }
private:
    RHITestResult check(const std::filesystem::path& directory)
    {
        scene::RenderPrimitive primitive;
        primitive.positions = {{0,0,2}, {1,0,2}, {0,1,2}};
        primitive.texcoords0 = {{0,0}, {1,0}, {0,1}};
        primitive.indices = {2,0,1};
        scene::RenderMaterial material;
        material.alphaMode = "MASK";
        material.baseColorTexture.uvTransform = {2,3,4, 5,6,7};
        render::SceneCoverageInput direct(material, nullptr, primitive);
        const std::array<std::array<float, 2>, 3> expected{{{7,13}, {4,7}, {6,12}}};
        OMM_EXPECT(direct.valid() && direct.desc().triangles.size() == 1 && direct.desc().triangles[0].uv == expected,
            "coverage snapshot did not apply UV transform in original indexed triangle order");
        primitive.texcoords0.assign(3, {100,100}); primitive.indices = {0,1,2};
        material.baseColorTexture.uvTransform = {1,0,0, 0,1,0}; material.baseColorFactor.w = 0;
        OMM_EXPECT(direct.desc().triangles[0].uv == expected && direct.desc().alphaFactor == 1.0f,
            "coverage snapshot aliased editable geometry or material properties");

        std::filesystem::create_directories(directory);
        const auto path = directory / "snapshot.gltf";
        {
            const float attributes[] = {0,0,2, 1,0,2, 0,1,2, 0,0, 1,0, 0,1};
            const uint32_t indices[] = {2,0,1};
            std::ofstream binary(directory / "snapshot.bin", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(attributes), sizeof(attributes));
            binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
            std::ofstream gltf(path);
            gltf << R"json({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
                "meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"indices":2,"material":0}]}],
                "buffers":[{"uri":"snapshot.bin","byteLength":72}],
                "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":24},{"buffer":0,"byteOffset":60,"byteLength":12}],
                "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,2],"max":[1,1,2]},
                    {"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"},{"bufferView":2,"componentType":5125,"count":3,"type":"SCALAR"}],
                "images":[{"uri":"snapshot.png"}],"textures":[{"source":0}],
                "materials":[{"alphaMode":"MASK","pbrMetallicRoughness":{"baseColorTexture":{"index":0}}}]})json";
        }
        scene::Scene loaded;
        OMM_EXPECT(loaded.load(path), "snapshot fixture failed to load: " + loaded.lastLoadResult().error);
        scene::RenderImage::Mip mip{.width = 2, .height = 1, .pixels = {255,255,255,0, 255,255,255,255}};
        OMM_EXPECT(loaded.setImageDecodeResult(0, {mip}, {}), "could not install decoded image fixture");
        auto snapshots = render::makeSceneCoverageInputs(loaded, render::scenePrimitiveCoverage(loaded));
        OMM_EXPECT(snapshots.size() == 1 && snapshots[0] && snapshots[0]->valid(), "scene coverage snapshot missing");
        const auto& snapshot = snapshots[0]->desc();
        OMM_EXPECT(snapshot.pixelsRGBA8.data() != loaded.images()[0].decodedMips[0].pixels.data(),
            "scene coverage snapshot shared mutable decoded image storage");
        const auto savedUV = snapshot.triangles[0].uv;
        mip.pixels.assign(8, 0);
        OMM_EXPECT(loaded.setImageDecodeResult(0, {mip}, {}), "could not replace source decoded image");
        auto changed = loaded.materials()[0]; changed.alphaCutoff = 0.9f;
        OMM_EXPECT(loaded.setMaterialProperties(0, changed), "could not edit source material");
        loaded = scene::Scene();
        OMM_EXPECT(snapshot.width == 2 && snapshot.height == 1 && snapshot.pixelsRGBA8.size() == 8 &&
            snapshot.pixelsRGBA8[3] == 0 && snapshot.pixelsRGBA8[7] == 255 &&
            snapshot.alphaCutoff == 0.5f && snapshot.triangles[0].uv == savedUV,
            "coverage snapshot changed or lost storage after source replacement/destruction");
        return RHITestResult::pass("canonical indexed UVs and alpha snapshots survive source geometry, material, image and scene changes");
    }
};
METALLIC_REGISTER_RHI_TEST(SceneCoverageSnapshotTest);

class OpacityMicromapRayQueryTest : public RHITest {
public:
    explicit OpacityMicromapRayQueryTest(bool partitioned = false) : partitioned_(partitioned)
    {
        name = partitioned ? "opacity_micromap_ray_query_partitioned" : "opacity_micromap_ray_query";
        type = RHITestType::Rendering;
    }
    std::optional<bench::Metadata> metadata() const override
    {
        if (partitioned_) { return std::nullopt; } // Combined PTLAS/OMM remains an integration case.
        return bench::comparisonMetadata({"opacityMicromap.scene.alpha.compaction.edits.refit"}, bench::Layer::Core,
            "ray-query", {"ray-query-omm", "opacityMicromap", bench::Capability::OpacityMicromap, 0.0, 0.0, "alpha-candidates.json"});
    }

    RHITestResult run(RHITestContext& context) override
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
        const auto variants = context.deviceDesc ? std::vector<bool>{render::vulkan::deviceExtensions(*context.deviceDesc).enableOpacityMicromap} : std::vector<bool>{false, true};
        for (bool enable : variants) {
            { std::ofstream restore(path); restore << originalGltf; }
            bench::TestDevice device;
            render::DeviceDesc desc{.applicationName = "Opacity Micromap Test",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
                .enablePartitionedAccelerationStructure = partitioned_, .enableAsyncCompute = true};
            render::vulkan::deviceExtensions(desc).enableOpacityMicromap = enable;
            const auto setup = bench::createTestDevice(context, std::move(desc)).transform([&](auto rhiValue) { device = std::move(rhiValue); });
            if (render::hasError(setup, render::Error::Unsupported)) { return RHITestResult::skip("ray queries unavailable"); }
            OMM_REQUIRE(setup);
            if (partitioned_ && !device->capabilities().partitionedAccelerationStructure) {
                return RHITestResult::skip("PTLAS unavailable");
            }
            if (enable && !render::vulkan::deviceCapabilities(*device).opacityMicromap) { return RHITestResult::skip("fallback passed; OMM unavailable"); }
            if (!enable) {
                auto vertexResult = device->createBuffer({.size = sizeof(bench::kRayVertices),
                    .usage = render::BufferUsageBits::AccelerationStructureBuildInput | render::BufferUsageBits::ShaderDeviceAddress,
                    .memoryLocation = render::MemoryLocation::HostUpload});
                OMM_REQUIRE(vertexResult);
                auto& vertices = *vertexResult;
                render::RayTracingCoverageTriangle triangle;
                triangle.uv = {{{0,0}, {1,0}, {0,1}}};
                const render::RayTracingCoverageDesc coverage{.triangles = {&triangle, 1}};
                auto slice = vertices->slice();
                OMM_REQUIRE(slice);
                const render::RayTracingTriangleGeometryDesc geometry{.vertexBuffer = *slice, .vertexStride = 12,
                    .vertexCount = 3, .indexType = render::RayTracingIndexType::None, .primitiveCount = 1,
                    .flags = render::RayTracingGeometryFlags::None, .coverage = &coverage};
                auto fallback = device->prepareRayTracingBottomLevelBuild({&geometry, 1});
                OMM_REQUIRE(fallback);
                OMM_EXPECT((*fallback)->valid() && (*fallback)->sizes().accelerationStructureSize != 0 &&
                    (*fallback)->coverageStats().geometryCount == 0, "static coverage did not retain ordinary BLAS fallback");
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
            OMM_EXPECT(resources.accelerationStructure().stats().coverageAccelerationCount == (enable ? 4u : 0u), "scene BLAS did not bake the expected OMMs");
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
            const render::ComputeResourceBindingDesc layout[] = {
                {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, scene), render::ComputeResourceBindingKind::AccelerationStructure}, {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, vertices)}, {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, indices)}, {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, primitives)}, {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, instances)}, {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materials)},
                {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materialTextures), render::ComputeResourceBindingKind::SampledImage, resources.materialTextureCount()}, {METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, probeOutput)}};
            render::ComputeProgram program;
            OMM_REQUIRE(program.initialize(*device, {
                .spirv = compiled.spirv,
                .pushConstantSize = 4,
                .bindings = {layout, uint32_t(std::size(layout))},
                .resourceParameterSize = sizeof(render::SceneResourceParameters),
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
                        before.coverageAccelerationBytes == after.coverageAccelerationBytes &&
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
                OMM_REQUIRE(commands->begin(frame.submissionContext()));
                OMM_REQUIRE(resources.uploadMaterialTextures(*commands));
                const render::ComputeDispatchBinding bindings[] = {
                    {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, scene), .accelerationStructure = resources.accelerationStructure().accelerationStructure()},
                    {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, vertices), .buffer = resources.shadingVertexBuffer()}, {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, indices), .buffer = resources.indexBuffer()},
                    {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, primitives), .buffer = resources.primitiveBuffer()}, {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, instances), .buffer = resources.instanceBuffer()},
                    {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materials), .buffer = resources.materialBuffer()},
                    {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, materialTextures), .textureViews = {resources.materialTextureViews().data(), resources.materialTextureCount()}},
                    {.binding = METALLIC_RESOURCE_MEMBER(render::SceneResourceParameters, probeOutput), .buffer = output.get()}};
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
            context.deviceDesc && render::vulkan::deviceExtensions(*context.deviceDesc).enableOpacityMicromap);
        if (context.evidence) {
            context.evidence->json("alpha-candidates.json", {{"candidates", fallbackCandidates + ommCandidates}});
            return RHITestResult::pass("CPU bilinear alpha oracle passed for all material/transform steps");
        }
        OMM_EXPECT(ommCandidates < fallbackCandidates / 2, "OMM did not reduce shader alpha candidates: " +
            std::to_string(fallbackCandidates) + " -> " + std::to_string(ommCandidates));
        return RHITestResult::pass("OMM visibility equals fallback after compaction/cutoff/UV/alpha/BLEND edits, transforms and backend switches; candidates " +
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

class OpacityMicromapBuildPlanLifetimeTest final : public RHITest {
public:
    OpacityMicromapBuildPlanLifetimeTest()
    {
        name = "opacity_micromap_build_plan_lifetime";
        type = RHITestType::Rendering;
    }
    std::optional<bench::Metadata> metadata() const override
    {
        auto result = bench::gpuMetadata({"coverage.plan.cpuInputRelease.commandRetention.compactionOwnership"},
            bench::Layer::RHI, "ray-query-omm", "extensions", {"readback.bin", "lifetime.json"});
        result.requirements.capabilities.insert(result.requirements.capabilities.end(),
            {bench::Capability::RayQuery, bench::Capability::OpacityMicromap});
        return result;
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::string log;
        auto created = bench::createTestDevice(context, {.applicationName = "Coverage plan lifetime",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayTracingAccelerationStructure = true, .enableRayQuery = true});
        if (render::hasError(created, render::Error::Unsupported)) { return RHITestResult::skip("ray queries unavailable"); }
        OMM_REQUIRE(created);
        auto device = std::move(*created);
        if (!device->capabilities().rayQuery || !device->capabilities().bindlessDescriptorHeap ||
            !render::vulkan::deviceCapabilities(*device).opacityMicromap) {
            return RHITestResult::skip("ray-query, bindless descriptors and OMM required");
        }
        auto& queue = *device->getQueue(render::QueueType::Graphics);
        const auto makeBuffer = [&](uint64_t size, render::MemoryLocation location = render::MemoryLocation::Device) {
            return device->createBuffer({.size = size,
                .usage = render::BufferUsageBits::Storage | render::BufferUsageBits::ShaderDeviceAddress |
                    render::BufferUsageBits::AccelerationStructureBuildInput | render::BufferUsageBits::AccelerationStructureStorage,
                .memoryLocation = location});
        };
        auto verticesResult = makeBuffer(sizeof(bench::kRayVertices), render::MemoryLocation::HostUpload);
        OMM_REQUIRE(verticesResult);
        auto vertices = std::move(*verticesResult);
        void* mapped = vertices->map();
        OMM_EXPECT(mapped != nullptr, "vertex upload map failed");
        std::memcpy(mapped, bench::kRayVertices.data(), sizeof(bench::kRayVertices));
        vertices->flush(); vertices->unmap();
        std::unique_ptr<render::RayTracingBottomLevelBuildPlan> plan;
        constexpr auto flags = render::RayTracingAccelerationStructureBuildFlags::PreferFastTrace |
            render::RayTracingAccelerationStructureBuildFlags::AllowCompaction;
        {
            // These CPU arrays and descriptors deliberately die before recording.
            std::vector<uint8_t> pixels(4 * 4 * 4, 255);
            std::vector<render::RayTracingCoverageTriangle> triangles(1);
            triangles[0].uv = {{{0,0}, {1,0}, {0,1}}};
            const render::RayTracingCoverageDesc coverage{.width = 4, .height = 4,
                .pixelsRGBA8 = pixels, .triangles = triangles};
            auto slice = vertices->slice();
            OMM_REQUIRE(slice);
            const render::RayTracingTriangleGeometryDesc geometry{.vertexBuffer = *slice, .vertexStride = 12,
                .vertexCount = 3, .indexType = render::RayTracingIndexType::None, .primitiveCount = 1,
                .flags = render::RayTracingGeometryFlags::None, .coverage = &coverage};
            auto updatePlan = device->prepareRayTracingBottomLevelBuild({&geometry, 1},
                render::RayTracingAccelerationStructureBuildFlags::PreferFastTrace |
                    render::RayTracingAccelerationStructureBuildFlags::AllowUpdate);
            OMM_REQUIRE(updatePlan);
            OMM_EXPECT((*updatePlan)->valid() && (*updatePlan)->sizes().updateScratchSize != 0 &&
                (*updatePlan)->coverageStats().geometryCount == 0, "updatable BLAS did not use mutable shader coverage fallback");
            const render::RayTracingCoverageDesc partialBlend{.mode = render::RayTracingCoverageMode::Blend,
                .alphaFactor = 0.5f, .triangles = triangles};
            auto partialGeometry = geometry;
            partialGeometry.coverage = &partialBlend;
            auto partialPlan = device->prepareRayTracingBottomLevelBuild({&partialGeometry, 1});
            OMM_REQUIRE(partialPlan);
            OMM_EXPECT((*partialPlan)->valid() && (*partialPlan)->sizes().accelerationStructureSize != 0 &&
                (*partialPlan)->coverageStats().geometryCount == 0, "constant partial BLEND allocated all-unknown coverage acceleration");
            OMM_REQUIRE(device->prepareRayTracingBottomLevelBuild({&geometry, 1}, flags)
                .transform([&](auto value) { plan = std::move(value); }));
        }
        OMM_EXPECT(plan && plan->valid() && plan->coverageStats().geometryCount == 1 &&
            plan->coverageStats().triangleCount == 1, "fixture did not prepare accelerated coverage");
        const auto coverageStats = plan->coverageStats();
        const auto sizes = plan->sizes();
        std::unique_ptr<render::RayTracingAccelerationStructure> source;
        OMM_REQUIRE(device->createRayTracingAccelerationStructure(*plan).transform([&](auto value) { source = std::move(value); }));
        auto properties = device->queryRayTracingAccelerationStructureProperties();
        OMM_REQUIRE(properties);
        auto scratchResult = makeBuffer(sizes.buildScratchSize + properties->scratchAlignment);
        OMM_REQUIRE(scratchResult);
        auto scratch = std::move(*scratchResult);
        auto queries = device->createRayTracingAccelerationStructureCompactionQueryPool({.queryCount = 1});
        OMM_REQUIRE(queries);
        {
            bench::GPUCommands build(queue);
            OMM_REQUIRE(build.initialize(*device));
            OMM_REQUIRE(build.commands->resetRayTracingAccelerationStructureCompactionQueries(**queries, 0, 1));
            auto scratchSlice = scratch->slice();
            OMM_REQUIRE(scratchSlice);
            auto shortScratchSlice = scratch->slice({.size = 1});
            OMM_REQUIRE(shortScratchSlice);
            const auto shortScratch = build.commands->buildRayTracingAccelerationStructure({.destination = source.get(),
                .scratchBuffer = *shortScratchSlice, .plan = plan.get()});
            OMM_EXPECT(render::hasError(shortScratch, render::Error::InvalidArgument),
                "undersized scratch was accepted for a prepared build plan");
            OMM_REQUIRE(build.commands->buildRayTracingAccelerationStructure({.destination = source.get(),
                .scratchBuffer = *scratchSlice, .plan = plan.get()}));
            const auto repeated = build.commands->buildRayTracingAccelerationStructure({.destination = source.get(),
                .scratchBuffer = *scratchSlice, .plan = plan.get()});
            OMM_EXPECT(render::hasError(repeated, render::Error::InvalidArgument), "a prepared build plan was recorded twice");
            plan.reset();
            vertices.reset();
            scratch.reset();
            OMM_REQUIRE(build.commands->writeRayTracingAccelerationStructureCompactedSize(**queries, 0, *source));
            OMM_REQUIRE(build.submitAndWait());
        }
        std::array<uint64_t, 1> compactedSizes{};
        OMM_REQUIRE((*queries)->readResults(0, compactedSizes));
        OMM_EXPECT(compactedSizes[0] > 0 && compactedSizes[0] <= sizes.accelerationStructureSize,
            "invalid BLAS compacted size");
        auto compactedResult = device->createRayTracingAccelerationStructure({.buildFlags = flags, .size = compactedSizes[0]});
        OMM_REQUIRE(compactedResult);
        auto compacted = std::move(*compactedResult);
        {
            bench::GPUCommands copy(queue);
            OMM_REQUIRE(copy.initialize(*device));
            OMM_REQUIRE(copy.commands->compactRayTracingAccelerationStructure(*source, *compacted));
            const auto overwrite = copy.commands->compactRayTracingAccelerationStructure(*source, *compacted);
            OMM_EXPECT(render::hasError(overwrite, render::Error::InvalidArgument),
                "compaction overwrote a destination already owning coverage dependencies");
            OMM_REQUIRE(copy.submitAndWait());
        }
        std::weak_ptr<void> sourceAllocation = source->retainAllocation();
        source.reset();
        OMM_EXPECT(sourceAllocation.expired(), "compacted BLAS retained the retired source allocation");
        OMM_EXPECT(compacted->coverageStats().geometryCount == coverageStats.geometryCount &&
            compacted->coverageStats().storageBytes == coverageStats.storageBytes,
            "compaction did not preserve resident coverage dependencies");

        render::RayTracingGPUInstance instance;
        instance.customIndexAndMask = 37 | (1u << 24);
        instance.shaderBindingTableRecordOffsetAndFlags = uint32_t(render::RayTracingInstanceFlags::TriangleFacingCullDisable) << 24;
        instance.accelerationStructureReference = compacted->deviceAddress();
        auto instancesResult = makeBuffer(sizeof(instance), render::MemoryLocation::HostUpload);
        OMM_REQUIRE(instancesResult);
        auto instances = std::move(*instancesResult);
        mapped = instances->map();
        OMM_EXPECT(mapped != nullptr, "instance upload map failed");
        std::memcpy(mapped, &instance, sizeof(instance)); instances->flush(); instances->unmap();
        auto tlasSizes = device->queryRayTracingAccelerationStructureBuildSizes({
            .type = render::RayTracingAccelerationStructureType::TopLevel, .instanceCount = 1});
        OMM_REQUIRE(tlasSizes);
        auto tlasResult = device->createRayTracingAccelerationStructure({.type = render::RayTracingAccelerationStructureType::TopLevel,
            .size = tlasSizes->accelerationStructureSize});
        OMM_REQUIRE(tlasResult);
        auto tlas = std::move(*tlasResult);
        scratchResult = makeBuffer(tlasSizes->buildScratchSize + properties->scratchAlignment);
        OMM_REQUIRE(scratchResult);
        scratch = std::move(*scratchResult);
        {
            bench::GPUCommands build(queue);
            OMM_REQUIRE(build.initialize(*device));
            auto instanceSlice = instances->slice(), scratchSlice = scratch->slice();
            OMM_REQUIRE(instanceSlice); OMM_REQUIRE(scratchSlice);
            OMM_REQUIRE(build.commands->buildRayTracingAccelerationStructure({.destination = tlas.get(),
                .instanceBuffer = *instanceSlice, .instanceCount = 1, .scratchBuffer = *scratchSlice}));
            OMM_REQUIRE(build.submitAndWait());
        }
        const char* capabilities[] = {"spvRayQueryKHR"};
        auto compiled = render::compileSlangShaderToSpirv({.moduleName = "CoverageBuildPlanProbe",
            .entryPointName = "coverageBuildPlanMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
            .capabilities = capabilities, .descriptorHeapMode = render::SlangDescriptorHeapMode::Mapped}, log);
        OMM_REQUIRE(compiled);
        const render::ComputeResourceBindingDesc layout[] = {{METALLIC_RESOURCE_MEMBER(metallic::tests::UnifiedTopLevelProbeResources, scene), render::ComputeResourceBindingKind::AccelerationStructure}, {METALLIC_RESOURCE_MEMBER(metallic::tests::UnifiedTopLevelProbeResources, output)}};
        render::ComputeProgram program;
        OMM_REQUIRE(program.initialize(*device, {.spirv = compiled->spirv, .bindings = layout,
            .resourceParameterSize = sizeof(metallic::tests::UnifiedTopLevelProbeResources)}, log));
        auto outputResult = makeBuffer(sizeof(bench::RayObservations), render::MemoryLocation::HostReadback);
        OMM_REQUIRE(outputResult);
        auto output = std::move(*outputResult);
        auto poolResult = device->createCommandPool(queue);
        OMM_REQUIRE(poolResult);
        auto pool = std::move(*poolResult);
        auto commandsResult = pool->createCommandBuffer();
        OMM_REQUIRE(commandsResult);
        auto commands = std::move(*commandsResult);
        render::QueueSubmissionTracker tracker;
        OMM_REQUIRE(tracker.initialize(*device, queue));
        render::RenderFrameContext frame;
        struct Drain {
            render::RenderFrameContext& frame; render::CommandPool& pool;
            ~Drain() { if (frame.completion().isSubmitted()) { (void)frame.wait(); } (void)pool.reset(); (void)frame.reset(); }
        } drain{frame, *pool};
        OMM_REQUIRE(frame.begin(0)); OMM_REQUIRE(commands->begin(frame.submissionContext()));
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::UnifiedTopLevelProbeResources, scene), .accelerationStructure = tlas.get()}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::UnifiedTopLevelProbeResources, output), .buffer = output.get()}};
        OMM_REQUIRE(program.dispatch({.commandBuffer = commands.get(), .bindings = bindings}));
        OMM_REQUIRE(commands->end());
        render::CommandBuffer* submitted[] = {commands.get()};
        OMM_REQUIRE(tracker.submit({.commandBuffers = submitted}, frame));
        OMM_REQUIRE(frame.wait(5'000'000'000ull));
        bench::RayObservations actual{};
        mapped = output->map();
        OMM_EXPECT(mapped != nullptr, "probe readback map failed");
        output->invalidate(); std::memcpy(actual.data(), mapped, sizeof(actual)); output->unmap();
        try { (void)bench::rayOracle(actual); }
        catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
        OMM_EXPECT(std::all_of(actual.begin(), actual.end(), [](const auto& ray) { return ray.padding == 0.0f; }),
            "released coverage reached shader candidate traversal instead of the opaque micromap");
        bench::readbackEvidence(context, "readback.bin", std::span<const bench::RayObservation>(actual));
        if (context.evidence) {
            context.evidence->json("lifetime.json", {{"cpuInputsReleasedBeforeRecording", true},
                {"planReleasedBeforeSubmission", true}, {"compactionSourceRetired", sourceAllocation.expired()},
                {"coverageStorageBytes", coverageStats.storageBytes}, {"analyticRays", actual.size()}});
        }
        return RHITestResult::pass("coverage survived CPU input/plan release and source BLAS retirement after compaction; six analytic rays passed");
    }
};
METALLIC_REGISTER_RHI_TEST(OpacityMicromapBuildPlanLifetimeTest);

#undef OMM_REQUIRE
#undef OMM_EXPECT
} // namespace
} // namespace metallic::tests
