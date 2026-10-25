#include "RHITest.h"
#include "harness/Fixtures.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/StreamRayTracingBindings.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/TessellationPatterns.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

constexpr uint32_t kRayCount = 8;
constexpr uint32_t kRayWords = 32;
constexpr uint32_t kImageSize = 128;
constexpr uint64_t kOutputBytes = sizeof(float) * (kRayCount * kRayWords + kImageSize * kImageSize * 4);
using TessellationObservations = std::array<float, kRayCount * kRayWords>;
constexpr std::array<std::array<float, 2>, kRayCount> kRayUV{{
    {.25f, .25f}, {.75f, .25f}, {.25f, .75f}, {.75f, .75f},
    {.5f, .5f}, {.125f, .375f}, {.875f, .625f}, {1.5f, .5f}}};
constexpr uint32_t kHeightWidth = 65;

float2 rayUV(uint32_t ray, bool comparisonRays)
{
    constexpr std::array<std::array<float, 2>, kRayCount> offsets{{
        {.013f, -.007f}, {-.017f, .011f}, {.011f, .013f}, {-.013f, .009f},
        {.013f, -.007f}, {.007f, -.011f}, {-.009f, .013f}, {0, 0}}};
    return float2(kRayUV[ray][0], kRayUV[ray][1]) +
        (comparisonRays ? float2(offsets[ray][0], offsets[ray][1]) : float2(0, 0));
}

// The graph's AS field carries a TLAS, not its streamed attribute tables. This
// test bridge publishes the exact production builder's runtime; a separate
// geometry request by the probe would create an unrelated streaming session.
std::weak_ptr<MeshletStreamRuntime> tessellationProducer;

class CLASTessellationBuildPass final : public ComputePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext& context) const override
    {
        return producer().sceneResourcesRequired(context);
    }
    std::span<const RenderSubsystemId> requiredSubsystems() const override { return producer().requiredSubsystems(); }
    RenderGraphSceneDependency sceneDependency() const override { return producer().sceneDependency(); }
    QueueType queueType() const override { return producer().queueType(); }
    bool supportsFrameOverlap() const override { return producer().supportsFrameOverlap(); }
    bool supportsAsyncQueue() const override { return producer().supportsAsyncQueue(); }
    bool supportsPipelinedSubmission() const override { return producer().supportsPipelinedSubmission(); }
    std::vector<RenderGraphRuntimeSetting> runtimeSettings() const override { return producer().runtimeSettings(); }
    RenderPassReflection reflect(const RenderGraphCompileContext& context) const override { return producer().reflect(context); }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        return producer().compile(context, log);
    }
    void describeSceneView(const RenderGraphExecutionContext& context, MeshletStreamFrameDesc& view) const override
    {
        producer().describeSceneView(context, view);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto result = producer().execute(context);
        if (result) { tessellationProducer = context.preparedScene()->geometry; }
        return result;
    }
private:
    RenderGraphPass& producer() const
    {
        producer_->setProperties(properties());
        return *producer_;
    }
    std::unique_ptr<RenderGraphPass> producer_ = builtin_pass::createSceneAccelerationStructurePass();
};

uint8_t heightTexel(uint32_t x)
{
    const float u = (float(x) + .5f) / float(kHeightWidth);
    return x == 0 || x + 1 == kHeightWidth ? 0 :
        uint8_t(std::lround(255.f * (1.f - std::abs(2.f * u - 1.f))));
}

struct ReferenceHit {
    float height;
    float3 normal;
};

std::array<std::vector<ReferenceHit>, kRayCount> referenceTessellationHits(
    uint32_t factor = 4, uint32_t splitDepth = 0, bool comparisonRays = false)
{
    // Independently interpolate the LUT domain, sample the UNORM texture and
    // intersect each emitted triangle on the CPU. Ring triangles can cross the
    // ridge; using the analytic height alone would incorrectly reject them.
    std::array<std::vector<ReferenceHit>, kRayCount> result;
    const auto pattern = makeTessellationPattern({factor, factor, factor});
    const std::array<float2, 4> corners{float2(0, 0), float2(1, 0), float2(1, 1), float2(0, 1)};
    std::vector<std::array<float2, 3>> sources{{corners[0], corners[1], corners[2]}, {corners[0], corners[2], corners[3]}};
    // Every edge remains larger than the leaf budget in this fixed 128-pixel
    // orthographic fixture, including both recursive levels tested below.
    for (uint32_t depth = 0; depth < splitDepth; ++depth) {
        std::vector<std::array<float2, 3>> children;
        for (const auto& source : sources) {
            const auto ab = (source[0] + source[1]) * .5f;
            const auto bc = (source[1] + source[2]) * .5f;
            const auto ca = (source[2] + source[0]) * .5f;
            children.push_back({source[0], ab, ca}); children.push_back({ab, source[1], bc});
            children.push_back({ca, bc, source[2]}); children.push_back({ab, bc, ca});
        }
        sources = std::move(children);
    }
    for (const auto& source : sources) {
        std::vector<float3> vertices;
        for (const auto& packed : pattern.vertices) {
            std::array<float, 3> bary{packed[0] / 65535.f, packed[1] / 65535.f, packed[2] / 65535.f};
            for (uint32_t zero = 0; zero < 3; ++zero) {
                if (packed[zero] != 0) { continue; }
                const uint32_t a = (zero + 1) % 3, b = (zero + 2) % 3;
                const float t = std::round(bary[b] * float(factor)) / float(factor);
                bary = {}; bary[a] = 1.f - t; bary[b] = t;
                break;
            }
            const float2 uv = source[0] * bary[0] + source[1] * bary[1] + source[2] * bary[2];
            const float coordinate = uv.x * kHeightWidth - .5f;
            const int base = int(std::floor(coordinate));
            const auto texel = [](int index) { return heightTexel(uint32_t(std::clamp(index, 0, int(kHeightWidth) - 1))) / 255.f; };
            const float height = std::lerp(texel(base), texel(base + 1), coordinate - std::floor(coordinate)) * .4f;
            vertices.emplace_back(uv.x * 2.f - 1.f, uv.y * 2.f - 1.f, height);
        }
        for (const auto& indices : pattern.triangles) {
            const auto a = vertices[indices[0]], b = vertices[indices[1]], c = vertices[indices[2]];
            const float denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
            const auto normal = normalize(cross(b - a, c - a));
            for (uint32_t ray = 0; ray + 1 < kRayCount; ++ray) {
                const auto uv = rayUV(ray, comparisonRays);
                const float x = uv.x * 2.f - 1.f, y = uv.y * 2.f - 1.f;
                const float wa = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / denominator;
                const float wb = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / denominator;
                const float wc = 1.f - wa - wb;
                if (wa >= -.0001f && wb >= -.0001f && wc >= -.0001f) {
                    result[ray].push_back({wa * a.z + wb * b.z + wc * c.z, normal});
                }
            }
        }
    }
    return result;
}

class CLASTessellationProbePass final : public ComputePass {
public:
    SceneStreamingRequirements sceneResourcesRequired(const RenderGraphCompileContext&) const override
    {
        return {.features = SceneResourceFeatureBits::Materials | SceneResourceFeatureBits::MaterialTextures,
            .textureFeedback = true};
    }
    std::span<const RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array required{GPUSceneSubsystem::kSubsystemId};
        return required;
    }
    RenderGraphSceneDependency sceneDependency() const override
    {
        return {RenderGraphSceneSource::Input, {"accelerationStructure"}};
    }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addAccelerationStructureInput("accelerationStructure").accelerationStructureRead();
        reflection.addBufferOutput("hits").buffer(kOutputBytes, 4 * sizeof(float))
            .storageWrite().hostReadback();
        return reflection;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        const auto& material = *context.preparedScene->snapshot->pathTraceResources;
        const char* capabilities[]{"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure",
            "spvRayTracingClusterAccelerationStructureNV"};
        const SlangMacroDefine defines[]{{"METALLIC_PT_PATH_DIAGNOSTICS", "1"}};
        auto shader = compileSlangShaderToSpirv({.moduleName = "Features/SmokeTests/CLASTessellationProbe",
            .entryPointName = "clasTessellationProbeMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .capabilities = capabilities, .macroDefines = defines, .descriptorHeapMode = SlangDescriptorHeapMode::Mapped}, log);
        if (!shader) { return makeError(shader.error()); }
        std::vector<ComputeResourceBindingDesc> layout{
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, scene), ComputeResourceBindingKind::AccelerationStructure},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureSamplingMetadata)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandMip)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandHits)},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures), ComputeResourceBindingKind::SampledImage,
                material.materialTextureCount()},
            {METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput)}};
        builtin_pass::appendStreamRayLayout(layout);
        return program_.initialize(*context.device, {.spirv = shader->spirv, .pushConstantSize = 8 * sizeof(float),
            .bindings = layout, .requiresRayQuery = true, .resourceParameterSize = sizeof(SceneResourceParameters)}, log);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const auto* prepared = context.preparedScene();
        const auto geometry = tessellationProducer.lock();
        if (!geometry || geometry->accelerationStructure() != context.inputAccelerationStructure("accelerationStructure")) {
            return makeError(Error::InvalidArgument);
        }
        const auto stream = geometry->deferredGpuResources();
        const auto& material = *prepared->snapshot->pathTraceResources;
        auto values = material.materialBinding();
        std::vector<ComputeDispatchBinding> bindings{
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, scene),
                .accelerationStructure = context.inputAccelerationStructure("accelerationStructure")},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials), .buffer = material.materialBuffer()},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues),
                .buffer = values->valueBuffer() ? values->valueBuffer() : material.materialBuffer()},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureSamplingMetadata), .buffer = prepared->textureFeedback.samplingMetadata},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandMip), .buffer = prepared->textureFeedback.demandMip},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, textureDemandHits), .buffer = prepared->textureFeedback.demandHits},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialTextures), .textureViews = material.materialTextureViews(),
                .sampledImages = material.materialTextureSnapshot()},
            {.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput), .buffer = context.outputBuffer("hits").buffer()}};
        builtin_pass::appendStreamRayBindings(bindings, stream, geometry->shadingChangeBuffer());
        const auto& transform = context.runtimeScene()->nodes()[0].localMatrix;
        const std::array<float, 8> push{transform.a00, transform.a11, transform.a22, transform.a03,
            transform.a13, transform.a23, properties().value("comparisonRays", false) ? 1.f : 0.f, 0};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .pushData = push.data(), .pushDataSize = sizeof(push), .groupCountX = (kRayCount + kImageSize * kImageSize) / 8});
    }
private:
    ComputeProgram program_;
};

bool writeCLASTessellationFixture(const std::filesystem::path& directory, std::string& log)
{
    // A tent ridge is linear on each side of u=0.5. Its source corners remain
    // flat, so an undisplaced two-triangle BLAS cannot satisfy interior rays.
    constexpr uint32_t width = kHeightWidth;
    std::vector<uint8_t> pixels(width * width * 4);
    for (uint32_t y = 0; y < width; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t h = heightTexel(x);
            const size_t offset = (y * width + x) * 4;
            pixels[offset] = pixels[offset + 1] = pixels[offset + 2] = h;
            pixels[offset + 3] = 255;
        }
    }
    if (!saveRgba8Png(directory / "Height.png", pixels.data(), width, width, log)) { return false; }
    struct Vertex { float x, y, z, nx, ny, nz, u, v, tx, ty, tz, sign; };
    const Vertex vertices[] = {
        {-1, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1},
        {1, -1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1},
        {1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1},
        {-1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1}};
    const uint32_t indices[]{0, 1, 2, 0, 2, 3};
    std::ofstream binary(directory / "mesh.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(vertices), sizeof(vertices));
    binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
    binary.close();
    std::ofstream gltf(directory / "scene.gltf");
    gltf << R"json({
        "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2,"TANGENT":3},"indices":4,"material":0}]}],
        "buffers":[{"uri":"mesh.bin","byteLength":216}],
        "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":192,"byteStride":48},{"buffer":0,"byteOffset":192,"byteLength":24}],
        "accessors":[
            {"bufferView":0,"byteOffset":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1,-1,0],"max":[1,1,0]},
            {"bufferView":0,"byteOffset":12,"componentType":5126,"count":4,"type":"VEC3"},
            {"bufferView":0,"byteOffset":24,"componentType":5126,"count":4,"type":"VEC2"},
            {"bufferView":0,"byteOffset":32,"componentType":5126,"count":4,"type":"VEC4"},
            {"bufferView":1,"componentType":5125,"count":6,"type":"SCALAR"}],
        "images":[{"uri":"Height.png"}],"samplers":[{"magFilter":9729,"minFilter":9729,"wrapS":33071,"wrapT":33071}],
        "textures":[{"source":0,"sampler":0}],
        "materials":[{"doubleSided":true,"pbrMetallicRoughness":{"metallicFactor":0,"roughnessFactor":1},
            "extras":{"METALLIC_displacement":{"magnitude":0.4,"center":0,"texture":{"index":0}}}}]
    })json";
    return bool(binary) && bool(gltf);
}

class CLASTessellationTest : public RHITest {
public:
    explicit CLASTessellationTest(bool templates = false) : templates_(templates)
    {
        type = RHITestType::Rendering;
        name = templates ? "scene_clas_tessellation_templates" : "scene_clas_tessellation";
    }
    std::optional<bench::Metadata> metadata() const override
    {
        auto result = bench::gpuMetadata({"clas.tessellation.production.displacement.attributes.lifecycle"},
            bench::Layer::Core, "ray-query", "core", {"readback.bin"});
        if (templates_) {
            result.coverage.push_back("clas.tessellation.templates.triangles.surface.equivalence");
            result.artifacts.push_back("template-comparison.json");
        }
        result.timeout = std::chrono::seconds(180);
        return result;
    }
    RHITestResult run(RHITestContext& context) override
    {
        std::string log;
        const auto directory = std::filesystem::absolute(context.outputDirectory / name);
        std::filesystem::create_directories(directory);
        if (!writeCLASTessellationFixture(directory, log)) { return RHITestResult::fail("Fixture write failed: " + log); }
        auto created = bench::createTestDevice(context, {.applicationName = "Production CLAS tessellation",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableRayQuery = true, .enableClusterAccelerationStructure = true});
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("CLAS/ray query unavailable"); }
        if (!created) { return RHITestResult::fail("Device creation failed"); }
        auto& device = **created;
        const auto check = [](bool condition, const std::string& message) {
            if (!condition) { throw std::runtime_error(message); }
        };
        const auto checked = [&](const Result<>& result, const char* operation) {
            check(bool(result), std::string(operation) + ": " + toString(result) + " " + log);
        };
        scene::Scene scene;
        check(scene.load(directory / "scene.gltf"), scene.lastLoadResult().error);
        static const bool registered = registerRenderGraphPassType("CLASTessellationProbe", "CLAS tessellation surface probe",
            [] { return std::make_unique<CLASTessellationProbePass>(); });
        static const bool registeredBuilder = registerRenderGraphPassType("CLASTessellationBuild", "Production CLAS builder with probe bridge",
            [] { return std::make_unique<CLASTessellationBuildPass>(); });
        (void)registered;
        (void)registeredBuilder;
        tessellationProducer.reset();
        registerBuiltInRenderGraphPasses();
        RenderGraph graph;
        const uint32_t build = graph.addNode("CLASTessellationBuild", "Build", {
            {"path", (directory / "scene.gltf").string()}, {"AsyncComputePreferred", false},
            {"rtTessellation", false}, {"rtTessellationEdgePixels", 1.f}, {"rtTessellationMaxFactor", 4},
            {"rtTessellationMaxSplitDepth", 0}, {"rtTessellationMaxPatches", 64},
            {"rtTessellationBuildMode", templates_ ? "templates" : "triangles"},
            {"camera", {{"eye", {0, 0, 4}}, {"center", {0, 0, 0}}, {"projection", "orthographic"},
                {"orthoHeight", 4.f}, {"znear", .001f}, {"zfar", 100.f}}}})->id;
        graph.addNode("CLASTessellationProbe", "Probe", {{"comparisonRays", templates_}});
        check(graph.addEdge("Build.accelerationStructure", "Probe.accelerationStructure"), "AS graph edge failed");
        check(graph.markOutput("Probe.hits"), "Probe output failed");
        RenderGraphExecutor executor;
        executor.bindRuntimeScene(&scene);
        checked(executor.compile(device, graph, 128, 128, log), "Graph compilation");
        graph.clearDirty();
        const auto setting = [&](const char* key, const nlohmann::json& value) {
            check(graph.setNodeRuntimeProperty(build, key, value), std::string("Setting rejected: ") + key);
            if (graph.dirty()) {
                checked(executor.compile(device, graph, 128, 128, log), "Settings compilation");
                graph.clearDirty();
            }
            else { check(executor.syncRuntimeProperties(graph), std::string("Setting did not reach executor: ") + key); }
        };
        const auto renderRaw = [&](uint32_t frames, const char* imageName, std::vector<float>* complete) {
            for (uint32_t frame = 0; frame < frames; ++frame) {
                checked(executor.execute({.graphicsQueue = device.getQueue(QueueType::Graphics),
                    .computeQueue = device.getQueue(QueueType::Compute)}), "Graph execution");
                checked(executor.waitForSubmittedWork(10'000'000'000ull), "Graph completion");
            }
            auto* buffer = executor.outputResource("Probe.hits")->buffer;
            const auto* mapped = buffer->map();
            check(mapped != nullptr, "Probe readback map failed");
            buffer->invalidate();
            TessellationObservations result;
            std::memcpy(result.data(), mapped, sizeof(result));
            if (complete) {
                complete->assign(static_cast<const float*>(mapped), static_cast<const float*>(mapped) + kOutputBytes / sizeof(float));
            }
            // Keep the complete last GPU observation in legacy runs as well as
            // testbench runs, including the image pixels and diagnostic counters.
            std::ofstream binary(directory / "readback.bin", std::ios::binary | std::ios::trunc);
            binary.write(static_cast<const char*>(mapped), kOutputBytes);
            binary.close();
            check(bool(binary), "Probe binary readback write failed");
            if (imageName) {
                const auto* image = static_cast<const float*>(mapped) + result.size();
                std::vector<uint8_t> rgba(kImageSize * kImageSize * 4);
                for (uint32_t pixel = 0; pixel < kImageSize * kImageSize; ++pixel) {
                    for (uint32_t channel = 0; channel < 3; ++channel) {
                        rgba[pixel * 4 + channel] = uint8_t(std::lround(std::pow(std::clamp(image[pixel * 4 + channel], 0.f, 1.f), 1.f / 2.2f) * 255.f));
                    }
                    rgba[pixel * 4 + 3] = 255;
                }
                check(saveRgba8Png(directory / imageName, rgba.data(), kImageSize, kImageSize, log), "Probe image write failed: " + log);
            }
            buffer->unmap();
            bench::readbackEvidence(context, "readback.bin", std::span<const float>(result));
            return result;
        };
        auto reference = referenceTessellationHits(4, 0, templates_);
        for (uint32_t ray = 0; ray + 1 < kRayCount; ++ray) {
            check(!reference[ray].empty() && reference[ray][0].height > .05f, "CPU reference lacks displaced interior coverage");
        }
        const auto verify = [&](const TessellationObservations& observations, bool displaced, const char* phase) {
            const auto& transform = scene.nodes()[0].localMatrix;
            for (uint32_t ray = 0; ray < kRayCount; ++ray) {
                const auto* value = observations.data() + ray * kRayWords;
                const std::string label = std::string(phase) + " ray " + std::to_string(ray) +
                    " [semantic=" + std::to_string(observations[24]) + ", patches=" + std::to_string(observations[25]) +
                    ", rejected=" + std::to_string(observations[26]) + ", hit=" + std::to_string(value[11]) +
                    ", position=" + std::to_string(value[0]) + "," + std::to_string(value[1]) + "," + std::to_string(value[2]) +
                    ", uv=" + std::to_string(value[20]) + "," + std::to_string(value[21]) + "]";
                check(value[11] == (ray + 1 == kRayCount ? 0.f : 1.f), label + " lost surface coverage or hit outside geometry");
                if (ray + 1 == kRayCount) { continue; }
                const auto uv = rayUV(ray, templates_);
                const float u = uv.x, v = uv.y;
                const float h = displaced ? reference[ray][0].height : 0.f;
                const bool back = ray == 6;
                const auto near = [](float a, float b, float epsilon) {
                    return std::isfinite(a) && std::abs(a - b) < epsilon;
                };
                check(near(value[0], (u * 2.f - 1.f) * transform.a00 + transform.a03, .001f) &&
                    near(value[1], (v * 2.f - 1.f) * transform.a11 + transform.a13, .001f) &&
                    near(value[2], h + transform.a23, .006f) &&
                    near(value[3], 2.f * transform.a22 + (back ? h : -h), .006f),
                    label + " wrong displaced hit/t: z=" + std::to_string(value[2]) + " expected=" +
                        std::to_string(h + transform.a23));
                check(near(value[20], u, .001f) && near(value[21], v, .001f), label + " lost source UV/domain mapping");
                check(std::isfinite(value[28]) && std::isfinite(value[29]) && value[28] >= -.0001f &&
                    value[29] >= -.0001f && value[28] + value[29] <= 1.0001f,
                    label + " generated triangle did not map to a valid source domain");
                check(value[7] == (back ? 0.f : 1.f), label + " changed authored front/back orientation");
                for (uint32_t offset : {4u, 8u}) {
                    const float norm = std::sqrt(value[offset] * value[offset] + value[offset + 1] * value[offset + 1] +
                        value[offset + 2] * value[offset + 2]);
                    check(near(norm, 1.f, .002f), label + " invalid surface normal");
                }
                // At an internal/shared edge either adjacent microtriangle is
                // legal. Geometry normals retain authored back-face orientation.
                const float3 actual(value[8], value[9], value[10]);
                const bool correctNormal = !displaced ? dot(actual, float3(0, 0, 1)) > .999f :
                    std::any_of(reference[ray].begin(), reference[ray].end(), [&](const auto& expected) {
                        return dot(actual, normalize(float3(expected.normal.x / transform.a00,
                            expected.normal.y / transform.a11, expected.normal.z))) > .995f;
                    });
                check(correctNormal, label + " geometry normal did not follow the displaced microtriangle");
            }
        };

        uint32_t pairedStates = 0;
        float maximumAttributeDifference = 0.f, maximumImageDifference = 0.f;
        const auto compare = [&](std::span<const float> expected, std::span<const float> actual, const char* mode) {
            check(expected.size() == actual.size(), "Build-mode output size mismatch");
            for (size_t word = 0; word < expected.size(); ++word) {
                // Only the first accepted frame after a scene edit invalidates
                // history. The paired mode switch must separately remain zero.
                if (word < kRayCount * kRayWords && word % kRayWords == 24) { continue; }
                const float difference = std::abs(expected[word] - actual[word]);
                check(std::isfinite(actual[word]) && difference <= .0002f,
                    std::string("Build-mode ") + mode + " differs at GPU word " + std::to_string(word) +
                    ": expected=" + std::to_string(expected[word]) + " actual=" + std::to_string(actual[word]));
                auto& maximum = word < kRayCount * kRayWords ? maximumAttributeDifference : maximumImageDifference;
                maximum = std::max(maximum, difference);
            }
        };
        const auto render = [&](uint32_t frames = 4, const char* imageName = nullptr) {
            std::vector<float> templateOutput;
            const auto result = renderRaw(frames, imageName, templates_ ? &templateOutput : nullptr);
            if (templates_) {
                std::vector<float> triangleOutput, restoredOutput;
                setting("rtTessellationBuildMode", "triangles");
                const auto triangleHits = renderRaw(1, imageName ? "CLASPairedTriangles.png" : nullptr, &triangleOutput);
                check(triangleHits[24] == 0.f, "Template-to-triangle mode switch invalidated unchanged shading history");
                compare(templateOutput, triangleOutput, "triangles");
                setting("rtTessellationBuildMode", "templates");
                const auto restoredHits = renderRaw(1, nullptr, &restoredOutput);
                check(restoredHits[24] == 0.f, "Triangle-to-template mode switch invalidated unchanged shading history");
                compare(templateOutput, restoredOutput, "templates restored");
                ++pairedStates;
            }
            return result;
        };

        verify(render(4, "CLASBase.png"), false, "disabled");
        setting("rtTessellation", true);
        const auto displaced = render(4, "CLASTessellation.png");
        verify(displaced, true, "enabled");
        check(displaced[25] == 2.f && displaced[26] == 0.f, "Two source triangles did not publish two tessellated patches");
        const auto stationary = render(1);
        check(stationary == displaced && stationary[24] == 0.f,
            "Stationary tessellated geometry changed hits or invalidated shading history");

        setting("rtTessellationMaxFactor", 2);
        reference = referenceTessellationHits(2, 0, templates_);
        verify(render(), true, "factor two");
        setting("rtTessellationMaxFactor", 4);
        for (uint32_t depth : {1u, 2u}) {
            setting("rtTessellationMaxSplitDepth", depth);
            reference = referenceTessellationHits(4, depth, templates_);
            const auto recursive = render();
            verify(recursive, true, "recursive subdivision");
            check(recursive[25] == float(2u << (2u * depth)) && recursive[26] == 0.f,
                "Recursive CLAS patch count or admission was incorrect");
            check(render(1) == recursive && recursive[24] == 0.f,
                "Stationary recursive geometry invalidated shading history");
        }
        setting("rtTessellationMaxSplitDepth", 0);
        reference = referenceTessellationHits(4, 0, templates_);
        check(render() == displaced, "Restoring tessellation quality changed geometry or history");
        const auto originalMaterial = scene.materials()[0];
        auto flatMaterial = originalMaterial;
        flatMaterial.displacementMagnitude = 0;
        check(scene.setMaterialProperties(0, flatMaterial), "Disable displacement edit failed");
        verify(render(), false, "material disabled");
        check(scene.setMaterialProperties(0, originalMaterial), "Restore displacement edit failed");
        const auto restored = render();
        verify(restored, true, "material restored");
        check(restored == displaced, "Restoring displacement changed the original surface");

        auto transform = scene.nodes()[0].localMatrix;
        transform.a00 = 2; transform.a11 = 3; transform.a22 = 4;
        transform.a03 = 3; transform.a13 = -2; transform.a23 = 1;
        check(scene.setNodeLocalMatrix(0, transform), "Nonuniform transform edit failed");
        verify(render(), true, "nonuniform scale");
        transform.a00 = -2;
        check(scene.setNodeLocalMatrix(0, transform), "Mirrored transform edit failed");
        verify(render(), true, "mirrored scale");
        setting("camera.eye", nlohmann::json::array({1000, 0, 4}));
        setting("camera.center", nlohmann::json::array({1000, 0, 0}));
        verify(render(), true, "offscreen secondary rays");

        // A recorded replacement must not publish CPU history before submission.
        transform.a03 = 3.5f;
        check(scene.setNodeLocalMatrix(0, transform), "Cancellation transform edit failed");
        auto pool = device.createCommandPool(*device.getQueue(QueueType::Graphics));
        check(bool(pool), "Cancellation command pool failed");
        auto commands = (*pool)->createCommandBuffer();
        check(bool(commands), "Cancellation command buffer failed");
        checked((*commands)->begin(), "Cancellation begin");
        checked(executor.execute(**commands), "Cancelled graph recording");
        checked((*commands)->end(), "Cancellation end");
        commands->reset();
        pool->reset();
        verify(render(), true, "cancelled recording retry");

        setting("rtTessellationMaxPatches", 1);
        const auto rejected = render(1);
        verify(rejected, false, "whole-instance capacity fallback");
        check(rejected[26] == 1.f, "Capacity fallback did not report its rejected instance");
        check(rejected[24] > 0.f, "First capacity-rejection frame failed to invalidate the displaced surface history");
        const auto stableFallback = render(3);
        verify(stableFallback, false, "stable capacity fallback");
        check(stableFallback[24] == 0.f && stableFallback[26] == 1.f,
            "Stable capacity fallback repeatedly invalidated shading history or lost rejection diagnostics");
        setting("rtTessellationMaxPatches", 64);
        const auto recovered = render(1);
        verify(recovered, true, "capacity recovery");
        check(recovered[24] > 0.f, "First capacity-recovery frame failed to invalidate the fallback surface history");
        const auto stableRecovery = render(3);
        verify(stableRecovery, true, "stable capacity recovery");
        check(stableRecovery[24] == 0.f && stableRecovery[26] == 0.f,
            "Stable capacity recovery repeatedly invalidated shading history or retained rejection diagnostics");
        // Reallocating a sufficient arena changes addresses, not the surface.
        // Keep the same scene and quality while replacing both capacity owners.
        for (uint32_t capacity : {32u, 64u}) {
            setting("rtTessellationMaxPatches", capacity);
            const auto replaced = render(1);
            verify(replaced, true, "sufficient capacity replacement");
            check(replaced[24] == 0.f && replaced[25] == 2.f && replaced[26] == 0.f,
                "Address-only capacity replacement invalidated an unchanged surface");
        }
        setting("rtTessellation", false);
        verify(render(), false, "runtime disabled");
        if (templates_) {
            const nlohmann::json comparison{{"pairedStates", pairedStates}, {"preciseRaysPerState", kRayCount},
                {"imagePixelsPerState", kImageSize * kImageSize}, {"maximumAttributeDifference", maximumAttributeDifference},
                {"maximumImageDifference", maximumImageDifference}, {"tolerance", .0002f},
                {"modeSwitchSemanticChanges", 0}};
            std::ofstream(directory / "TemplateComparison.json") << comparison.dump(2);
            if (context.evidence) { context.evidence->json("template-comparison.json", comparison); }
        }
        return RHITestResult::pass("Production tessellated CLAS hits match a tent-ridge oracle; source UV, microtriangle normals, "
            "material edits, mirrored/nonuniform transforms, offscreen rays, cancellation and whole-instance capacity fallback" +
            std::string(templates_ ? "; template/triangle source-domain and full-image equivalence with history-preserving mode switches" : ""));
    }
private:
    bool templates_;
};
METALLIC_REGISTER_RHI_TEST(CLASTessellationTest);
class CLASTessellationTemplatesTest final : public CLASTessellationTest {
public:
    CLASTessellationTemplatesTest() : CLASTessellationTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(CLASTessellationTemplatesTest);

} // namespace
} // namespace metallic::tests
