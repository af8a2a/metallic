#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMapPrototype.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/TessellationPatterns.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/MeshletStreamReferenceCodec.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
using Triangle = VirtualShadowMapPrototypeTriangle;
constexpr uint32_t kWidth = 129, kHeight = 97;
constexpr uint32_t kPixels = kWidth * kHeight;
constexpr uint64_t kProbeABI = 0x56534d434f4e5434ull;

void check(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

template<class T> void check(const Result<T>& result, const std::string& message)
{
    check(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

struct ContentCommands
{
    RenderFrameContext frame;
    QueueSubmissionTracker tracker;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    uint64_t index = 0;

    explicit ContentCommands(Device& device)
    {
        auto& queue = *device.getQueue(QueueType::Graphics);
        check(tracker.initialize(device, queue), "Content submission tracker");
        check(device.createCommandPool(queue).transform([&](auto p) { pool = std::move(p); }), "Content pool");
        check(pool->createCommandBuffer().transform([&](auto c) { commands = std::move(c); }), "Content commands");
    }
    ~ContentCommands()
    {
        frame.cancel();
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        (void)pool->reset();
        (void)frame.reset();
    }
    void begin()
    {
        check(frame.begin(index++), "Content frame begin");
        check(pool->reset(), "Content pool reset");
        check(commands->begin(frame.submissionContext()), "Content command begin");
    }
    void finish()
    {
        check(commands->end(), "Content command end");
        CommandBuffer* list[]{commands.get()};
        check(tracker.submit({.commandBuffers = list}, frame), "Content submit");
        check(frame.wait(), "Content wait");
    }
};

struct HeightImage
{
    uint32_t width = 0;
    std::vector<uint8_t> pixels;
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
};

HeightImage makeHeight(Device& device, ContentCommands& gpu, uint32_t width, bool inverted)
{
    HeightImage image;
    image.width = width;
    image.pixels.resize(width * width * 4);
    for (uint32_t y = 0; y < width; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const double h = .5 + .38 * std::sin(double(x) * 6.283185307179586 / width) *
                std::cos(double(y) * 6.283185307179586 / width);
            auto* p = image.pixels.data() + (y * width + x) * 4;
            p[0] = uint8_t(std::lround((inverted ? 1 - h : h) * 255));
            p[1] = p[2] = p[0];
            p[3] = inverted ? 0 : (x < width / 2 ? 0 : 255);
        }
    }
    check(device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
        .format = Format::RGBA8Unorm, .width = width, .height = width})
        .transform([&](auto t) { image.texture = std::move(t); }), "Height image");
    check(device.createTextureView(*image.texture, {.format = Format::RGBA8Unorm})
        .transform([&](auto v) { image.view = std::move(v); }), "Height view");
    auto upload = device.createBuffer({.size = image.pixels.size(), .usage = BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::HostUpload});
    check(upload, "Height staging");
    auto* data = (*upload)->map(); check(data != nullptr, "Height map");
    std::memcpy(data, image.pixels.data(), image.pixels.size()); (*upload)->flush(); (*upload)->unmap();
    gpu.begin();
    const TextureBarrierDesc before{.texture = image.texture.get(), .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
    check(gpu.commands->synchronize({.textures = {&before, 1}}), "Height upload barrier");
    check((*upload)->slice().and_then([&](const auto& slice) {
        return gpu.commands->copyBufferToTexture({.texture = image.texture.get(), .buffer = slice, .width = width, .height = width});
    }), "Height upload");
    const TextureBarrierDesc after{.texture = image.texture.get(), .oldLayout = TextureLayout::TransferDestination,
        .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}};
    check(gpu.commands->synchronize({.textures = {&after, 1}}), "Height ready");
    gpu.finish();
    return image;
}

double referenceHeight(const HeightImage& image, double u, double v, const scene::RenderMaterial& material)
{
    const auto& transform = material.displacementTexture.uvTransform;
    const double x = (u * transform[0] + v * transform[1] + transform[2]);
    const double y = (u * transform[3] + v * transform[4] + transform[5]);
    const double tx = (x - std::floor(x)) * image.width - .5;
    const double ty = (y - std::floor(y)) * image.width - .5;
    const int ix = int(std::floor(tx)), iy = int(std::floor(ty)), size = int(image.width);
    const auto fetch = [&](int a, int b) {
        return image.pixels[((b + size) % size * size + (a + size) % size) * 4] / 255.0;
    };
    const double value = std::lerp(std::lerp(fetch(ix, iy), fetch(ix + 1, iy), tx - ix),
        std::lerp(fetch(ix, iy + 1), fetch(ix + 1, iy + 1), tx - ix), ty - iy);
    return (value - material.displacementCenter) * material.displacementMagnitude;
}

std::filesystem::path writeContentFixture(const std::filesystem::path& directory, const char* missingAttribute = nullptr)
{
    std::filesystem::create_directories(directory);
    const float positions[]{-4,-4,0, 4,-4,0, 4,4,0, -4,4,0,
        -.75f,-.75f,1, .75f,-.75f,1, .75f,.75f,1, -.75f,.75f,1};
    const float normals[]{0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1};
    const float uvs[]{0,0, 1,0, 1,1, 0,1, 0,0, 1,0, 1,1, 0,1};
    const uint32_t indices[]{0,1,2, 0,2,3};
    { std::ofstream binary(directory / "Geometry.bin", std::ios::binary);
      binary.write(reinterpret_cast<const char*>(positions), sizeof(positions));
      binary.write(reinterpret_cast<const char*>(normals), sizeof(normals));
      binary.write(reinterpret_cast<const char*>(uvs), sizeof(uvs));
      binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
      check(binary.good(), "Write content geometry"); }
    Json doc = Json::parse(R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1]}],
      "nodes":[{"mesh":0},{"mesh":1}],"buffers":[{"uri":"Geometry.bin","byteLength":280}],
      "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":96},
        {"buffer":0,"byteOffset":96,"byteLength":96},{"buffer":0,"byteOffset":192,"byteLength":64},
        {"buffer":0,"byteOffset":256,"byteLength":24}],
      "accessors":[{"bufferView":0,"byteOffset":0,"componentType":5126,"count":4,"type":"VEC3","min":[-4,-4,0],"max":[4,4,0]},
        {"bufferView":0,"byteOffset":48,"componentType":5126,"count":4,"type":"VEC3","min":[-0.75,-0.75,1],"max":[0.75,0.75,1]},
        {"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"},
        {"bufferView":2,"componentType":5126,"count":4,"type":"VEC2"},
        {"bufferView":3,"componentType":5125,"count":6,"type":"SCALAR"}],
      "materials":[{"doubleSided":true},{"doubleSided":true}],
      "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":2,"TEXCOORD_0":3},"indices":4,"material":0}]},
        {"primitives":[{"attributes":{"POSITION":1,"NORMAL":2,"TEXCOORD_0":3},"indices":4,"material":1}]}]})");
    std::string log;
    const uint8_t pixel[]{128, 128, 128, 255};
    check(saveRgba8Png(directory / "Height.png", pixel, 1, 1, log), log);
    doc["images"] = {{{"uri", "Height.png"}}};
    doc["textures"] = {{{"source", 0}}};
    doc["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"] = {{"index", 0}};
    doc["materials"][1]["extras"]["METALLIC_displacement"] = {{"magnitude", .6}, {"center", .5},
        {"texture", {{"index", 0}, {"extensions", {{"KHR_texture_transform",
            {{"scale", {1.1, .9}}, {"offset", {.13, .23}}, {"rotation", .2}}}}}}}};
    if (missingAttribute) { doc["meshes"][1]["primitives"][0]["attributes"].erase(missingAttribute); }
    const auto path = directory / "Content.gltf";
    std::ofstream file(path); file << doc.dump(2); check(file.good(), "Write content fixture");
    return path;
}

struct ContentSnapshot
{
    std::vector<uint8_t> shadow;
    std::array<uint32_t, 32> stats{};
    std::vector<Triangle> triangles;
    uint64_t dependency = 0;
};

class VirtualShadowMapContentTest final : public RHITest
{
public:
    VirtualShadowMapContentTest() { name = "virtual_shadow_map_content"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapContent");
        Json report{{"passed", false}, {"frames", Json::array()}, {"rayQuery", false}};
        bench::ValidationRecorder validation;
        try {
            const auto path = writeContentFixture(directory);
            const auto streamPath = directory / "Content.meshstream.bin";
            std::string log;
            check(scene::buildMeshletStreamAssetOffline({.sourcePath = path, .outputPath = streamPath,
                .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log), "Cook content: " + log);
            auto deviceResult = createDevice({.applicationName = "VSM P4 content",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(deviceResult, Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and bindless"); }
            check(deviceResult, "Content device");
            auto& device = **deviceResult;
            ContentCommands gpu(device);
            auto heightA = makeHeight(device, gpu, 9, false);
            auto heightB = makeHeight(device, gpu, 5, true);
            auto registry = ResourceRegistry::forDevice(device); check(registry, "Content registry");
            const auto shader = compileSlangShaderToSpirv({.moduleName = "VirtualShadowMapContent", .entryPointName = "contentMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
            check(shader, "Content probe shader: " + log);
            struct ProbePush { uint32_t geometry, output, count, reserved; };
            ComputeKernel probe;
            check(probe.initialize(device, {.spirv = shader->spirv,
                .parameters = parameterAbi<ProbePush>(kProbeABI, ParameterTransport::InlinePush)}, log), "Content probe: " + log);
            std::unique_ptr<Texture> depth;
            std::unique_ptr<TextureView> depthView;
            check(device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::D32Sfloat, .width = kWidth, .height = kHeight})
                .transform([&](auto t) { depth = std::move(t); }), "Content depth");
            check(device.createTextureView(*depth, {.format = Format::D32Sfloat})
                .transform([&](auto v) { depthView = std::move(v); }), "Content depth view");
            RenderView camera;
            check(camera.setCamera({.eye = {0,0,4}, .center = {0,0,0}, .nearPlane = .01f,
                .farPlane = 10, .orthoHeight = 4, .orthographic = true}), "Content camera");
            auto depthUpload = device.createBuffer({.size = kPixels * sizeof(float), .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload});
            check(depthUpload, "Content depth upload");
            auto* depthValues = static_cast<float*>((*depthUpload)->map());
            check(depthValues != nullptr, "Map content depth");
            std::fill(depthValues, depthValues + kPixels, (10.f - 4.f) / (10.f - .01f));
            (*depthUpload)->flush(); (*depthUpload)->unmap();
            gpu.begin();
            const TextureBarrierDesc depthBegin{.texture = depth.get(), .oldLayout = TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
            check(gpu.commands->synchronize({.textures = {&depthBegin, 1}}), "Content depth begin");
            check((*depthUpload)->slice().and_then([&](const auto& slice) {
                return gpu.commands->copyBufferToTexture({.texture = depth.get(), .buffer = slice, .width = kWidth, .height = kHeight});
            }), "Content ground depth");
            const TextureBarrierDesc depthEnd{.texture = depth.get(), .oldLayout = TextureLayout::TransferDestination,
                .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
            check(gpu.commands->synchronize({.textures = {&depthEnd, 1}}), "Content depth ready");
            gpu.finish();
            std::unique_ptr<Streamer> streamer;
            check(createStreamer(device, {.constantBufferSize = 1u << 20}).transform([&](auto s) { streamer = std::move(s); }), "Content streamer");
            for (bool streamed : {false, true}) {
                scene::SceneDocument scene;
                check(streamed ? scene.loadStreamMetadata(path) : scene.load(path), "Load content fixture");
                std::shared_ptr<MeshletStreamRuntime> runtime;
                if (streamed) {
                    runtime = std::make_shared<MeshletStreamRuntime>();
                    check(runtime->initialize(device, {.sourcePath = path, .streamAssetPath = streamPath,
                        .maxResidentBytes = 4ull << 20, .maxResidentPages = 32, .maxLockedFallbackPages = 32,
                        .maxGpuPageRequests = 128, .maxGpuPageUnloadRequests = 128, .maxActiveGroups = 128,
                        .maxRasterCandidates = 128, .maxTraversalWorkers = 32, .maxTraversalWorkItems = 256,
                        .pageLoadConcurrency = 0, .queuedFrameCount = 1}, log), "Content stream runtime: " + log);
                    check(runtime->syncRuntimeScene(scene, log), "Content stream mapping: " + log);
                    for (uint32_t attempt = 0; attempt < 32 && !runtime->sceneReady(); ++attempt) {
                        gpu.begin(); check(streamer->beginFrame(gpu.frame), "Content stream frame");
                        check(runtime->cmdLoadInitialResources(*gpu.commands, *streamer), "Content initial load");
                        gpu.finish(); streamer->endFrame(); runtime->prepareMaintenance();
                    }
                    check(runtime->sceneReady(), "Content terminal pages not ready");
                }
                VirtualShadowMaps cached, forced;
                auto material = scene.materials()[1];
                material.displacementTexture.textureIndex = 0;
                material.displacementMagnitude = .8f;
                material.displacementCenter = .5f;
                check(scene.setMaterialProperties(1, material), "Set initial displacement");
                VirtualShadowMapCasterSettings casterSettings{.tessellationEnabled = true, .tessellationFactor = 4,
                    .texturePublicationRevision = 101};
                HeightImage* currentHeight = &heightA;
                auto record = [&](VirtualShadowMaps& shadows, bool force, bool captureGeometry) {
                    gpu.begin();
                    if (runtime) { check(runtime->syncRuntimeScene(scene, log), "Synchronize content edit: " + log); }
                    auto stream = runtime ? runtime->shadowGeometryResources() : MeshletStreamShadowGeometryView{};
                    stream.owner = runtime;
                    std::array<TextureView*, 2> heightViews{nullptr, currentHeight->view.get()};
                    casterSettings.displacementTextures = heightViews;
                    auto casters = prepareVirtualShadowMapCasters(device, *gpu.commands, scene, runtime ? &stream : nullptr, log, casterSettings);
                    check(casters, "Prepare content: " + log);
                    ContentSnapshot result; result.dependency = casters->dependencyKey;
                    std::unique_ptr<Buffer> decoded;
                    if (captureGeometry) {
                        check(device.createBuffer({.size = uint64_t(casters->triangleCapacity) * sizeof(Triangle), .structureStride = sizeof(Triangle),
                            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
                            .transform([&](auto b) { decoded = std::move(b); }), "Content decode output");
                        ParameterWriter writer(device, **registry);
                        ProbePush push{writer.buffer(casters->parameters).index, writer.buffer(decoded.get()).index, casters->triangleCapacity, 0};
                        auto encoded = writer.encode(push, kProbeABI, ParameterTransport::InlinePush); check(encoded, "Content probe parameters");
                        gpu.commands->hostWriteBarrier();
                        check(probe.dispatch(*gpu.commands, *encoded, (push.count + 63) / 64), "Content probe dispatch");
                    }
                    VirtualShadowMapLight light;
                    light.direction = {1,0,-1};
                    VirtualShadowMapSettings settings;
                    settings.pageSize = 32; settings.virtualPagesPerAxis = 8; settings.clipmapCount = 3;
                    settings.physicalPageCount = 256; settings.cacheEnabled = true; settings.forceRedraw = force;
                    settings.maxWorkItems = 65535;
                    std::array<TextureView*, 2> alphaViews{nullptr, material.alphaMode == "MASK" ? currentHeight->view.get() : nullptr};
                    const auto view = camera.constants(gpu.index, kWidth, kHeight, kWidth, kHeight);
                    auto shadow = shadows.record(device, *gpu.commands, *depthView, view, light, *casters,
                        scene.materials(), alphaViews, settings, log);
                    check(shadow, "Content shadow record: " + log);
                    constexpr uint64_t statsOffset = (kPixels + 3u) & ~3u;
                    auto readback = device.createBuffer({.size = statsOffset + sizeof(result.stats),
                        .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback});
                    check(readback, "Content readback");
                    const TextureBarrierDesc before{.texture = shadow->texture, .oldLayout = TextureLayout::ShaderRead,
                        .newLayout = TextureLayout::TransferSource, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                    const BufferBarrierDesc stats{.buffer = shadow->statistics,
                        .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                    check(gpu.commands->synchronize({.textures = {&before, 1}, .buffers = {&stats, 1}}), "Content readback barriers");
                    check((*readback)->slice({0, kPixels}).and_then([&](const auto& slice) {
                        return gpu.commands->copyTextureToBuffer({.texture = shadow->texture, .buffer = slice, .width = kWidth, .height = kHeight});
                    }), "Content shadow copy");
                    auto from = shadow->statistics->slice(), to = (*readback)->slice({statsOffset, sizeof(result.stats)});
                    check(from, "Content statistics source"); check(to, "Content statistics destination");
                    check(gpu.commands->copyBuffer(*from, *to), "Content statistics copy");
                    gpu.finish();
                    (*readback)->invalidate(); const auto* bytes = static_cast<const uint8_t*>((*readback)->map());
                    check(bytes != nullptr, "Map content shadow");
                    result.shadow.assign(bytes, bytes + kPixels); std::memcpy(result.stats.data(), bytes + statsOffset, sizeof(result.stats));
                    (*readback)->unmap();
                    if (decoded) {
                        decoded->invalidate(); const auto* triangles = static_cast<const Triangle*>(decoded->map());
                        check(triangles != nullptr, "Map content geometry");
                        for (uint32_t i = 0; i < casters->triangleCapacity; ++i) {
                            if (triangles[i].reserved[1] != 0) { result.triangles.push_back(triangles[i]); }
                        }
                        decoded->unmap();
                    }
                    return result;
                };
                const auto compare = [&](const std::string& name, bool geometry) {
                    validation.phase((streamed ? "Stream-" : "Resident-") + name);
                    report["activeCase"] = {{"name", name}, {"stream", streamed}};
                    auto actual = record(cached, false, geometry), reference = record(forced, true, false);
                    check(actual.shadow == reference.shadow, "Stale content cache at " + name);
                    double maximumError = 0;
                    uint32_t displacedVertices = 0;
                    uint32_t windingErrors = 0;
                    std::array<double, 3> casterUVCoverage{};
                    if (geometry) {
                        const auto& matrix = scene.renderNodes()[1].worldMatrix.a;
                        float sourceNormal[]{0, 0, 1};
                        // The source fixture uses the Reference codec's 22-bit
                        // normal contract. Canonicalize the CPU input once;
                        // testing against an unquantized +Z would incorrectly
                        // reject the real raster normal's small X/Y displacement.
                        // Source positions/UV corners are exact binary values.
                        if (streamed) { scene::canonicalizeMeshletStreamReferenceVertex(nullptr, sourceNormal, nullptr, nullptr); }
                        std::array<std::array<double, 3>, 3> cofactors{};
                        for (uint32_t column = 0; column < 3; ++column) {
                            const uint32_t a = ((column + 1) % 3) * 4, b = ((column + 2) % 3) * 4;
                            cofactors[column] = {double(matrix[a+1])*matrix[b+2] - double(matrix[a+2])*matrix[b+1],
                                double(matrix[a+2])*matrix[b] - double(matrix[a])*matrix[b+2],
                                double(matrix[a])*matrix[b+1] - double(matrix[a+1])*matrix[b]};
                        }
                        std::array<double, 3> normal{};
                        for (uint32_t axis = 0; axis < 3; ++axis) {
                            for (uint32_t column = 0; column < 3; ++column) { normal[axis] += cofactors[column][axis] * sourceNormal[column]; }
                        }
                        const double normalLength = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
                        const double sign = cofactors[2][0]*matrix[8] + cofactors[2][1]*matrix[9] + cofactors[2][2]*matrix[10] < 0 ? -1 : 1;
                        report["activeCase"]["sourceNormal"] = {sourceNormal[0], sourceNormal[1], sourceNormal[2]};
                        report["sourceNormals"][streamed ? "StreamReferenceCodec" : "Resident"] =
                            {sourceNormal[0], sourceNormal[1], sourceNormal[2]};
                        Json errors = Json::array();
                        for (const auto& triangle : actual.triangles) {
                            // Both authored fixture triangles have positive UV
                            // area. A reflected instance must reverse that order,
                            // independently of height sampling or subdivision.
                            const auto& world = scene.renderNodes()[triangle.materialIndex].worldMatrix.a;
                            const double determinant = double(world[0]) * (double(world[5])*world[10] - double(world[6])*world[9]) -
                                double(world[4]) * (double(world[1])*world[10] - double(world[2])*world[9]) +
                                double(world[8]) * (double(world[1])*world[6] - double(world[2])*world[5]);
                            const auto& a = triangle.vertices[0].uv;
                            const auto& b = triangle.vertices[1].uv;
                            const auto& c = triangle.vertices[2].uv;
                            const double uvArea = (double(b[0])-a[0]) * (double(c[1])-a[1]) -
                                (double(b[1])-a[1]) * (double(c[0])-a[0]);
                            if (uvArea * (determinant < 0 ? -1 : 1) <= 0) { ++windingErrors; }
                            if (triangle.materialIndex == 1) {
                                const double area = std::abs(uvArea) * .5;
                                casterUVCoverage[0] += area;
                                casterUVCoverage[1] += area * (double(a[0]) + b[0] + c[0]) / 3;
                                casterUVCoverage[2] += area * (double(a[1]) + b[1] + c[1]) / 3;
                            }
                            for (const auto& vertex : triangle.vertices) {
                                const auto& p = vertex.position;
                                check(std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]), "Nonfinite content geometry");
                                if (triangle.materialIndex != 1) { continue; }
                                ++displacedVertices;
                                const double height = casterSettings.tessellationEnabled ?
                                    referenceHeight(*currentHeight, vertex.uv[0], vertex.uv[1], material) : 0;
                                std::array<double, 3> expected{};
                                double error = 0;
                                for (uint32_t axis = 0; axis < 3; ++axis) {
                                    expected[axis] = matrix[axis] * (vertex.uv[0] * 1.5 - .75) +
                                        matrix[4 + axis] * (vertex.uv[1] * 1.5 - .75) + matrix[8 + axis] + matrix[12 + axis] +
                                        height * normal[axis] / normalLength * sign;
                                    error = std::max(error, std::abs(p[axis] - expected[axis]));
                                }
                                maximumError = std::max(maximumError, error);
                                if (error >= 2e-5 && errors.size() < 16) {
                                    errors.push_back({{"actual", p}, {"uv", vertex.uv}, {"expected", expected}, {"error", error}});
                                }
                            }
                        }
                        report["activeCase"]["geometryErrors"] = std::move(errors);
                        report["activeCase"]["windingErrors"] = windingErrors;
                        const auto pattern = makeTessellationPattern({casterSettings.tessellationFactor,
                            casterSettings.tessellationFactor, casterSettings.tessellationFactor});
                        const uint32_t patchCount = 1u << (casterSettings.subdivisionDepth * 2);
                        const uint32_t expectedTriangles = casterSettings.tessellationEnabled && material.displacementMagnitude != 0 ?
                            uint32_t(pattern.triangles.size() * patchCount * 2 + 2) : 4;
                        check(actual.triangles.size() == expectedTriangles && displacedVertices > 0, "Incomplete content tessellation");
                        check(maximumError < 2e-5, "Independent height reference mismatch: " + std::to_string(maximumError));
                        check(windingErrors == 0, "Mirrored caster did not preserve authored front faces at " + name);
                        check(std::abs(casterUVCoverage[0] - 1) < 2e-5 && std::abs(casterUVCoverage[1] - .5) < 2e-5 &&
                            std::abs(casterUVCoverage[2] - .5) < 2e-5, "Shadow patches do not cover the authored UV square at " + name);
                    }
                    report["frames"].push_back({{"name", name}, {"stream", streamed}, {"equal", true},
                        {"statistics", actual.stats}, {"triangles", actual.triangles.size()}, {"maximumGeometryError", maximumError},
                        {"windingErrors", windingErrors}, {"tessellationFactor", casterSettings.tessellationFactor},
                        {"casterUVAreaAndFirstMoments", casterUVCoverage},
                        {"subdivisionDepth", casterSettings.subdivisionDepth},
                        {"displacedTriangleMultiplier", 1u << (casterSettings.subdivisionDepth * 2)}});
                    std::vector<uint8_t> png(kPixels * 4, 255);
                    for (uint32_t i = 0; i < kPixels; ++i) { png[i*4] = png[i*4+1] = png[i*4+2] = actual.shadow[i]; }
                    check(saveRgba8Png(directory / ((streamed ? "Stream-" : "Resident-") + name + ".png"), png.data(), kWidth, kHeight, log), log);
                    return actual;
                };
                const auto initial = compare("Displaced", true);
                auto stable = compare("Stable", false);
                check(stable.stats[16] == 0 && stable.stats[21] == 0, "Stable displacement redrew depth");
                material.displacementMagnitude = -.9f; material.displacementCenter = .25f;
                check(scene.setMaterialProperties(1, material), "Negative height material edit");
                const auto negative = compare("NegativeMagnitude", true);
                check(negative.shadow != initial.shadow && negative.stats[18] > 0, "Height edit failed to invalidate and move shadows");
                material.displacementCenter = .7f;
                check(scene.setMaterialProperties(1, material), "Edit height center");
                (void)compare("HeightCenter", true);
                currentHeight = &heightB; casterSettings.texturePublicationRevision = 102;
                const auto published = compare("PublishedImageMip", true);
                check(published.dependency != negative.dependency && published.stats[22] > 0, "Height image publication retained stale pages");
                stable = compare("PublishedStable", false);
                check(stable.stats[16] == 0 && stable.stats[21] == 0, "Published height cache never became stable");
                casterSettings.tessellationFactor = 8;
                const auto factor8 = compare("Factor8", true);
                uint64_t previousDependency = factor8.dependency;
                for (uint32_t depth = 1; depth <= 2; ++depth) {
                    casterSettings.subdivisionDepth = depth;
                    const auto refined = compare("Subdivision" + std::to_string(depth), true);
                    check(refined.dependency != previousDependency && refined.stats[22] > 0,
                        "Shadow subdivision edit retained stale pages");
                    previousDependency = refined.dependency;
                    stable = compare("Subdivision" + std::to_string(depth) + "Stable", false);
                    check(stable.stats[16] == 0 && stable.stats[21] == 0, "Stable subdivision redrew depth");
                }
                casterSettings.subdivisionDepth = 0;
                const auto restored = compare("SubdivisionRestored", true);
                check(restored.dependency == factor8.dependency && restored.shadow == factor8.shadow && restored.stats[22] > 0,
                    "Returning to depth zero changed the original shadow or retained stale pages");
                const auto originalMatrix = scene.nodes()[1].localMatrix;
                float4x4 mirrored;
                mirrored.SetupByScale(float3(-1.25f, .8f, 1.6f));
                check(scene.setNodeLocalMatrix(1, mirrored), "Mirror and nonuniformly scale displaced caster");
                (void)compare("MirroredNonuniform", true);
                casterSettings.subdivisionDepth = 2;
                (void)compare("Subdivision2Mirrored", true);
                casterSettings.subdivisionDepth = 0;
                material.doubleSided = false;
                check(scene.setMaterialProperties(1, material), "Make mirrored displaced caster single-sided");
                (void)compare("MirroredSingleSided", true);
                casterSettings.tessellationEnabled = false;
                const auto mirroredBase = compare("MirroredSingleSidedProducerDisabled", true);
                check(std::ranges::any_of(mirroredBase.shadow, [](uint8_t value) { return value == 0; }),
                    "Mirrored single-sided base caster lost its shadow");
                casterSettings.tessellationEnabled = true;
                const auto originalGroundMatrix = scene.nodes()[0].localMatrix;
                const auto originalGroundMaterial = scene.materials()[0];
                auto groundMaterial = originalGroundMaterial;
                groundMaterial.doubleSided = false;
                check(scene.setMaterialProperties(0, groundMaterial), "Make mixed base material single-sided");
                float4x4 mirroredGround;
                mirroredGround.SetupByScale(float3(-1, 1, 1));
                check(scene.setNodeLocalMatrix(0, mirroredGround), "Mirror mixed base caster");
                (void)compare("MirroredMixedBaseMaterial", true);
                check(scene.setMaterialProperties(0, originalGroundMaterial), "Restore ground material");
                check(scene.setNodeLocalMatrix(0, originalGroundMatrix), "Restore ground transform");
                material.doubleSided = true;
                check(scene.setMaterialProperties(1, material), "Restore displaced caster sidedness");
                check(scene.setNodeLocalMatrix(1, originalMatrix), "Restore displaced caster transform");
                (void)compare("TransformRestored", true);
                casterSettings.tessellationEnabled = false;
                const auto disabled = compare("ProducerTessellationDisabled", true);
                check(disabled.shadow != published.shadow, "Producer tessellation disable retained displaced shadows");
                material.displacementMagnitude = 0; material.alphaMode = "MASK";
                material.baseColorTexture.textureIndex = 0;
                check(scene.setMaterialProperties(1, material), "Enable textured MASK");
                currentHeight = &heightA; casterSettings.texturePublicationRevision = 103;
                const auto masked = compare("MaskImage", false);
                currentHeight = &heightB; casterSettings.texturePublicationRevision = 104;
                const auto transparent = compare("MaskImagePublication", false);
                check(masked.shadow != transparent.shadow && transparent.stats[22] > 0,
                    "Alpha image publication retained stale Coverage pages");
                check(std::ranges::all_of(transparent.shadow, [](uint8_t value) { return value == 255; }), "Zero alpha caster still casts shadow");
                material.displacementMagnitude = .8f;
                check(scene.setMaterialProperties(1, material), "Set unsupported masked displacement");
                const auto reject = [&](VirtualShadowMapCasterSettings rejected, Error expected, const char* name) {
                    gpu.begin();
                    auto stream = runtime ? runtime->shadowGeometryResources() : MeshletStreamShadowGeometryView{};
                    stream.owner = runtime;
                    std::array<TextureView*, 2> views{nullptr, heightA.view.get()};
                    if (rejected.texturePublicationRevision != 0) { rejected.displacementTextures = views; }
                    auto prepared = prepareVirtualShadowMapCasters(device, *gpu.commands, scene, runtime ? &stream : nullptr, log, rejected);
                    check(hasError(prepared, expected), std::string("Accepted invalid caster boundary: ") + name);
                    gpu.finish();
                    report["rejected"].push_back({{"name", name}, {"stream", streamed}});
                };
                reject({.tessellationEnabled = true, .texturePublicationRevision = 105}, Error::Unsupported, "MASK displacement");
                material.alphaMode = "OPAQUE";
                check(scene.setMaterialProperties(1, material), "Restore opaque displacement for validation");
                reject({.tessellationEnabled = true}, Error::InvalidArgument, "Missing height publication");
                reject({.tessellationEnabled = true, .tessellationFactor = 9, .texturePublicationRevision = 105},
                    Error::InvalidArgument, "Unbounded subdivision factor");
                reject({.tessellationEnabled = true, .subdivisionDepth = 3, .texturePublicationRevision = 105},
                    Error::InvalidArgument, "Unbounded subdivision depth");
                if (!streamed) {
                    for (const auto* attribute : {"NORMAL", "TEXCOORD_0"}) {
                        scene::SceneDocument incomplete;
                        check(incomplete.load(writeContentFixture(directory / attribute, attribute)), "Load incomplete vertex streams");
                        gpu.begin();
                        std::array<TextureView*, 2> views{nullptr, heightA.view.get()};
                        const auto rejected = prepareVirtualShadowMapCasters(device, *gpu.commands, incomplete, nullptr, log,
                            {.tessellationEnabled = true, .displacementTextures = views, .texturePublicationRevision = 105});
                        check(hasError(rejected, Error::Unsupported), std::string("Accepted displaced geometry without ") + attribute);
                        gpu.finish();
                        report["rejected"].push_back({{"name", std::string("Missing ") + attribute}, {"stream", false}});
                    }
                }
            }
            report.erase("activeCase");
            report["validation"] = validation.snapshot();
            check(!validation.failed(), "Vulkan validation reported content errors");
            report["passed"] = true;
            std::ofstream(directory / "VirtualShadowMapContent.json") << report.dump(2);
            return RHITestResult::pass("Resident/Stream displaced geometry agrees with CPU height oracle; material, height/alpha publication and factor cache edits match redraw");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot();
            std::filesystem::create_directories(directory);
            std::ofstream(directory / "VirtualShadowMapContent.json") << report.dump(2);
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapContentTest);
} // namespace
} // namespace metallic::tests
