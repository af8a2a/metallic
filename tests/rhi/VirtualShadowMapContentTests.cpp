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
#include <tuple>

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
    void submit(Semaphore* gate = nullptr)
    {
        check(commands->end(), "Content command end");
        CommandBuffer* list[]{commands.get()};
        const SemaphoreSubmitDesc wait{.semaphore = gate, .value = 1};
        check(tracker.submit({.waitSemaphores = {gate ? &wait : nullptr, gate ? 1u : 0u},
            .commandBuffers = list}, frame), "Content submit");
    }
    void finish(bool cancel = false)
    {
        if (cancel) {
            check(commands->end(), "Content cancelled command end");
            frame.cancel();
            return;
        }
        submit();
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
    Json doc = Json::parse(R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0,1,2]}],
      "nodes":[{"mesh":0},{"mesh":1},{"camera":0,"translation":[0,0,4]}],
      "cameras":[{"type":"perspective","perspective":{"yfov":0.7,"znear":0.01,"zfar":10}}],
      "buffers":[{"uri":"Geometry.bin","byteLength":280}],
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
    std::vector<uint8_t> geometryBytes;
    std::vector<uint64_t> contentKeys;
    std::shared_ptr<void> casterOwner;
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
            const auto decodeGeometry = [&](ContentCommands& execution, const VirtualShadowMapCasterResources& casters) {
                std::unique_ptr<Buffer> decoded;
                check(device.createBuffer({.size = uint64_t(casters.triangleCapacity) * sizeof(Triangle), .structureStride = sizeof(Triangle),
                    .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostReadback})
                    .transform([&](auto b) { decoded = std::move(b); }), "Content decode output");
                ParameterWriter writer(device, **registry);
                ProbePush push{writer.buffer(casters.parameters).index, writer.buffer(decoded.get()).index, casters.triangleCapacity, 0};
                auto encoded = writer.encode(push, kProbeABI, ParameterTransport::InlinePush); check(encoded, "Content probe parameters");
                execution.commands->hostWriteBarrier();
                check(probe.dispatch(*execution.commands, *encoded, (push.count + 63) / 64), "Content probe dispatch");
                return decoded;
            };
            const auto readGeometry = [&](Buffer& decoded, uint32_t count) {
                decoded.invalidate();
                const auto* bytes = static_cast<const uint8_t*>(decoded.map());
                check(bytes != nullptr, "Map content geometry");
                std::vector<uint8_t> result(bytes, bytes + uint64_t(count) * sizeof(Triangle));
                decoded.unmap();
                return result;
            };
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
                VirtualShadowMapCasterCache casterCache;
                std::shared_ptr<void> previousCasterOwner;
                VirtualShadowMapLight light;
                light.direction = {1,0,-1};
                auto material = scene.materials()[1];
                material.displacementTexture.textureIndex = 0;
                material.displacementMagnitude = .8f;
                material.displacementCenter = .5f;
                check(scene.setMaterialProperties(1, material), "Set initial displacement");
                VirtualShadowMapCasterSettings casterSettings{.tessellationEnabled = true, .tessellationFactor = 4,
                    .texturePublicationRevision = 101};
                HeightImage* currentHeight = &heightA;
                auto record = [&](VirtualShadowMaps& shadows, bool force, bool cancel = false) {
                    gpu.begin();
                    if (runtime) { check(runtime->syncRuntimeScene(scene, log), "Synchronize content edit: " + log); }
                    auto stream = runtime ? runtime->shadowGeometryResources() : MeshletStreamShadowGeometryView{};
                    stream.owner = runtime;
                    std::array<TextureView*, 2> heightViews{nullptr, currentHeight->view.get()};
                    casterSettings.displacementTextures = heightViews;
                    auto casters = force ?
                        prepareVirtualShadowMapCasters(device, *gpu.commands, scene, runtime ? &stream : nullptr, log, casterSettings) :
                        casterCache.prepare(device, *gpu.commands, scene, runtime ? &stream : nullptr, log, casterSettings);
                    check(casters, "Prepare content: " + log);
                    ContentSnapshot result; result.dependency = casters->dependencyKey;
                    result.casterOwner = casters->owner;
                    for (const auto& state : casters->states) { result.contentKeys.push_back(state.contentKey); }
                    auto decoded = decodeGeometry(gpu, *casters);
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
                    gpu.finish(cancel);
                    if (cancel) { return result; }
                    (*readback)->invalidate(); const auto* bytes = static_cast<const uint8_t*>((*readback)->map());
                    check(bytes != nullptr, "Map content shadow");
                    result.shadow.assign(bytes, bytes + kPixels); std::memcpy(result.stats.data(), bytes + statsOffset, sizeof(result.stats));
                    (*readback)->unmap();
                    result.geometryBytes = readGeometry(*decoded, casters->triangleCapacity);
                    for (uint32_t i = 0; i < casters->triangleCapacity; ++i) {
                        Triangle triangle;
                        std::memcpy(&triangle, result.geometryBytes.data() + uint64_t(i) * sizeof(Triangle), sizeof(Triangle));
                        if (triangle.reserved[1] != 0) { result.triangles.push_back(triangle); }
                    }
                    return result;
                };
                const auto compare = [&](const std::string& name, bool geometry, bool expectReuse = false) {
                    validation.phase((streamed ? "Stream-" : "Resident-") + name);
                    report["activeCase"] = {{"name", name}, {"stream", streamed}};
                    auto actual = record(cached, false), reference = record(forced, true);
                    check(actual.shadow == reference.shadow, "Stale content cache at " + name);
                    check(actual.geometryBytes == reference.geometryBytes, "Cached GPU geometry differs from uncached preparation at " + name);
                    check(actual.contentKeys == reference.contentKeys && actual.dependency == reference.dependency,
                        "Cached caster metadata differs from uncached preparation at " + name);
                    const bool reused = actual.casterOwner == previousCasterOwner;
                    check(reused == (!streamed && expectReuse), "Unexpected caster package reuse at " + name);
                    previousCasterOwner = actual.casterOwner;
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
                        {"geometryBytesEqual", true}, {"casterPackageReused", reused},
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
                auto stable = compare("Stable", false, true);
                check(stable.stats[16] == 0 && stable.stats[21] == 0, "Stable displacement redrew depth");
                if (!streamed) {
                    const auto originalCameraMatrix = scene.nodes()[2].localMatrix;
                    const auto originalCamera = scene.objectForNode(2).getComponent<scene::CameraComponent>().properties;
                    auto editedCamera = originalCamera;
                    editedCamera.yfov = .9;
                    float4x4 cameraMatrix;
                    cameraMatrix.SetupByTranslation(float3(.25f, 0, 4));
                    check(scene.setNodeLocalMatrix(2, cameraMatrix), "Move camera without changing geometry");
                    check(scene.setObjectCameraProperties(scene.objectForNode(2).entity(), editedCamera), "Edit camera projection");
                    check(camera.setCamera({.eye = {.25f,0,4}, .center = {.25f,0,0}, .nearPlane = .01f,
                        .farPlane = 10, .orthoHeight = 4, .orthographic = true}), "Move content camera");
                    auto lighting = scene.lighting();
                    lighting.lights.push_back({.name = "Cache independence", .position = {2,3,4}});
                    check(scene.setLighting(lighting), "Edit scene light without changing geometry");
                    light.direction = {.8f,.2f,-1};
                    (void)compare("CameraAndLightEdit", true, true);
                    check(scene.setNodeLocalMatrix(2, originalCameraMatrix), "Restore camera transform");
                    check(scene.setObjectCameraProperties(scene.objectForNode(2).entity(), originalCamera), "Restore camera projection");
                    check(camera.setCamera({.eye = {0,0,4}, .center = {0,0,0}, .nearPlane = .01f,
                        .farPlane = 10, .orthoHeight = 4, .orthographic = true}), "Restore content camera");
                    light.direction = {1,0,-1};
                    (void)compare("CameraAndLightRestored", false, true);
                    material.baseColorFactor.x = .25f;
                    check(scene.setMaterialProperties(1, material), "Edit RGB through material setter");
                    const auto rgb = compare("RGBOnlyMaterialEdit", false);
                    check(rgb.contentKeys == initial.contentKeys && rgb.dependency == initial.dependency &&
                        rgb.shadow == initial.shadow && rgb.stats[16] == 0 && rgb.stats[21] == 0,
                        "RGB-only edit changed shadow content or dirtied depth");
                    check(scene.setObjectVisible(scene.objectForNode(1).entity(), false), "Hide resident caster");
                    const auto hidden = compare("HiddenCaster", false);
                    check(std::ranges::all_of(hidden.shadow, [](uint8_t value) { return value == 255; }), "Hidden caster retained shadow");
                    check(scene.setObjectVisible(scene.objectForNode(1).entity(), true), "Restore resident caster visibility");
                    (void)compare("VisibleCaster", true);
                    const auto originalCaster = scene.nodes()[1].localMatrix;
                    float4x4 movedCaster;
                    movedCaster.SetupByTranslation(float3(.35f, 0, 0));
                    check(scene.setNodeLocalMatrix(1, movedCaster * originalCaster), "Move resident caster before cancellation");
                    const auto cancelled = record(cached, false, true);
                    check(cancelled.casterOwner != previousCasterOwner, "Cancelled edit did not rebuild caster package");
                    previousCasterOwner = cancelled.casterOwner;
                    const auto retry = compare("CancelledPreparationRetry", true, true);
                    check(retry.stats[28] > 0 && retry.stats[16] > 0, "Cancelled depth was reused instead of redrawn");
                    (void)compare("CancelledPreparationStable", false, true);
                    check(scene.setNodeLocalMatrix(1, originalCaster), "Restore resident caster after cancellation");
                    (void)compare("CancelledPreparationRestored", true);

                    std::array<TextureView*, 2> validViews{nullptr, heightA.view.get()}, missingViews{};
                    std::array<TextureView*, 1> shortViews{heightA.view.get()};
                    auto validSettings = casterSettings;
                    validSettings.displacementTextures = validViews;
                    const auto rejectHot = [&](const char* name, VirtualShadowMapCasterSettings invalid) {
                        gpu.begin();
                        const auto actual = casterCache.prepare(device, *gpu.commands, scene, nullptr, log, invalid);
                        const auto reference = prepareVirtualShadowMapCasters(device, *gpu.commands, scene, nullptr, log, invalid);
                        check(hasError(actual, Error::InvalidArgument) && hasError(reference, Error::InvalidArgument),
                            std::string("Invalid settings bypassed cached boundary: ") + name);
                        const auto restored = casterCache.prepare(device, *gpu.commands, scene, nullptr, log, validSettings);
                        check(restored, "Restore valid settings after cached rejection");
                        check(restored->owner == previousCasterOwner, "Rejected settings replaced successful caster package");
                        gpu.finish();
                        report["rejected"].push_back({{"name", name}, {"stream", false}, {"retainedSuccessfulPackage", true}});
                    };
                    auto invalidSettings = validSettings;
                    invalidSettings.displacementTextures = shortViews;
                    rejectHot("Cached wrong slot count", invalidSettings);
                    invalidSettings.displacementTextures = {};
                    rejectHot("Cached missing slots", invalidSettings);
                    invalidSettings.displacementTextures = missingViews;
                    rejectHot("Cached missing height view", invalidSettings);
                    invalidSettings = validSettings; invalidSettings.tessellationFactor = 0;
                    rejectHot("Cached zero tessellation factor", invalidSettings);
                }
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
                stable = compare("PublishedStable", false, true);
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
                    stable = compare("Subdivision" + std::to_string(depth) + "Stable", false, true);
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
                    const auto prepared = casterCache.prepare(device, *gpu.commands, scene, runtime ? &stream : nullptr, log, rejected);
                    const auto reference = prepareVirtualShadowMapCasters(device, *gpu.commands, scene, runtime ? &stream : nullptr, log, rejected);
                    check(hasError(prepared, expected) && hasError(reference, expected), std::string("Accepted invalid caster boundary: ") + name);
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
                        const auto rejected = casterCache.prepare(device, *gpu.commands, incomplete, nullptr, log,
                            {.tessellationEnabled = true, .displacementTextures = views, .texturePublicationRevision = 105});
                        check(hasError(rejected, Error::Unsupported), std::string("Accepted displaced geometry without ") + attribute);
                        gpu.finish();
                        report["rejected"].push_back({{"name", std::string("Missing ") + attribute}, {"stream", false}});
                    }
                }
            }
            {
                validation.phase("Resident-cached-package-in-flight");
                scene::SceneDocument lifetimeScene;
                check(lifetimeScene.load(path), "Load resident package lifetime fixture");
                auto lifetimeCache = std::make_unique<VirtualShadowMapCasterCache>();
                std::weak_ptr<void> oldOwner;
                gpu.begin();
                {
                    const auto warmed = lifetimeCache->prepare(device, *gpu.commands, lifetimeScene, nullptr, log);
                    check(warmed, "Warm resident package lifetime cache");
                    oldOwner = warmed->owner;
                }
                gpu.finish();
                check(gpu.pool->reset(), "Release warmup recording package");
                check(gpu.frame.reset(), "Release warmup frame package");

                auto gate = device.createSemaphore();
                check(gate, "Resident package lifetime gate");
                ContentCommands earlier(device), later(device);
                std::unique_ptr<Buffer> oldDecoded, oldReference, newDecoded, newReference;
                uint32_t oldCount = 0, newCount = 0;
                // Every failure releases the gate before local GPU resources die.
                struct Drain
                {
                    Queue& queue;
                    Semaphore& gate;
                    ~Drain()
                    {
                        if (gate.currentValue() < 1) { (void)gate.signal(1); }
                        (void)queue.waitIdle();
                    }
                } drain{*device.getQueue(QueueType::Graphics), **gate};
                earlier.begin();
                {
                    const auto actual = lifetimeCache->prepare(device, *earlier.commands, lifetimeScene, nullptr, log);
                    const auto reference = prepareVirtualShadowMapCasters(device, *earlier.commands, lifetimeScene, nullptr, log);
                    check(actual, "Reuse old resident package"); check(reference, "Prepare old uncached reference");
                    check(actual->owner == oldOwner.lock(), "In-flight lifetime path did not hit resident cache");
                    check(actual->triangleCapacity == reference->triangleCapacity, "Old package/reference capacity differs");
                    oldCount = actual->triangleCapacity;
                    oldDecoded = decodeGeometry(earlier, *actual);
                    oldReference = decodeGeometry(earlier, *reference);
                }
                earlier.submit(gate->get());
                check(!earlier.frame.completion().isComplete(), "Lifetime gate did not retain in-flight work");
                float4x4 moved;
                moved.SetupByTranslation(float3(.5f, 0, .25f));
                check(lifetimeScene.setNodeLocalMatrix(1, moved), "Replace in-flight resident geometry");
                later.begin();
                {
                    const auto actual = lifetimeCache->prepare(device, *later.commands, lifetimeScene, nullptr, log);
                    const auto reference = prepareVirtualShadowMapCasters(device, *later.commands, lifetimeScene, nullptr, log);
                    check(actual, "Prepare replacement resident package"); check(reference, "Prepare replacement reference");
                    check(actual->owner != oldOwner.lock(), "Geometry edit reused in-flight package");
                    check(actual->triangleCapacity == reference->triangleCapacity, "Replacement package/reference capacity differs");
                    newCount = actual->triangleCapacity;
                    newDecoded = decodeGeometry(later, *actual);
                    newReference = decodeGeometry(later, *reference);
                }
                lifetimeCache.reset();
                check(!oldOwner.expired(), "Cache hit did not retain old package for its submitted command");
                later.submit();
                check((*gate)->signal(1), "Release resident package lifetime gate");
                check(earlier.frame.wait(), "Wait old resident package");
                check(later.frame.wait(), "Wait replacement resident package");
                const auto oldBytes = readGeometry(*oldDecoded, oldCount);
                const auto newBytes = readGeometry(*newDecoded, newCount);
                check(oldBytes == readGeometry(*oldReference, oldCount), "In-flight old package bytes changed after replacement");
                check(newBytes == readGeometry(*newReference, newCount), "Replacement package differs from uncached geometry");
                check(oldBytes != newBytes, "Lifetime transform fixture did not change GPU geometry");
                report["residentPackageLifetime"] = {{"cacheHit", true}, {"submittedGateHeld", true},
                    {"ownerRetainedAfterCacheDestruction", true}, {"oldGeometryBytesEqual", true},
                    {"replacementGeometryBytesEqual", true}, {"geometryChanged", true}};
            }
            {
                validation.phase("VSM-resource-pool");
                scene::SceneDocument poolScene;
                check(poolScene.load(path), "Load resource pool fixture");
                auto pooled = std::make_unique<VirtualShadowMaps>();
                VirtualShadowMaps reference;
                VirtualShadowMapLight light;
                light.direction = {1, 0, -1};
                VirtualShadowMapSettings settings;
                settings.pageSize = 32; settings.virtualPagesPerAxis = 8; settings.clipmapCount = 3;
                settings.physicalPageCount = 256; settings.cacheEnabled = true; settings.maxWorkItems = 65535;
                constexpr uint64_t statsOffset = (kPixels + 3u) & ~3u;
                struct Capture
                {
                    std::unique_ptr<Buffer> readback;
                    VirtualShadowMapResult resources;
                    std::array<uint32_t, 32> statistics{};
                };
                const auto record = [&](ContentCommands& execution, VirtualShadowMaps& shadows, bool force) {
                    const auto casters = prepareVirtualShadowMapCasters(device, *execution.commands, poolScene, nullptr, log);
                    check(casters, "Prepare resource pool casters: " + log);
                    auto current = settings;
                    current.forceRedraw = force;
                    const auto view = camera.constants(execution.index, kWidth, kHeight, kWidth, kHeight);
                    const auto shadow = shadows.record(device, *execution.commands, *depthView, view, light, *casters,
                        poolScene.materials(), {}, current, log);
                    check(shadow, "Record resource pool shadow: " + log);
                    auto buffer = device.createBuffer({.size = statsOffset + 128, .usage = BufferUsageBits::TransferDestination,
                        .memoryLocation = MemoryLocation::HostReadback});
                    check(buffer, "Resource pool shadow readback");
                    Capture capture{std::move(*buffer), *shadow};
                    const TextureBarrierDesc before{.texture = shadow->texture, .oldLayout = TextureLayout::ShaderRead,
                        .newLayout = TextureLayout::TransferSource, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                    const BufferBarrierDesc statisticsReady{.buffer = shadow->statistics,
                        .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                    check(execution.commands->synchronize({.textures = {&before, 1}, .buffers = {&statisticsReady, 1}}),
                        "Resource pool readback barrier");
                    check(capture.readback->slice({0, kPixels}).and_then([&](const auto& slice) {
                        return execution.commands->copyTextureToBuffer({.texture = shadow->texture, .buffer = slice,
                            .width = kWidth, .height = kHeight});
                    }), "Resource pool shadow copy");
                    auto from = shadow->statistics->slice({0, sizeof(capture.statistics)});
                    auto to = capture.readback->slice({statsOffset, sizeof(capture.statistics)});
                    check(from, "Resource pool statistics source"); check(to, "Resource pool statistics destination");
                    check(execution.commands->copyBuffer(*from, *to), "Resource pool statistics copy");
                    return capture;
                };
                const auto read = [&](Capture& capture) {
                    capture.readback->invalidate();
                    const auto* bytes = static_cast<const uint8_t*>(capture.readback->map());
                    check(bytes != nullptr, "Map resource pool shadow");
                    std::vector<uint8_t> pixels(bytes, bytes + kPixels);
                    std::memcpy(capture.statistics.data(), bytes + statsOffset, sizeof(capture.statistics));
                    capture.readback->unmap();
                    return pixels;
                };
                const auto serial = [&](const char* name) {
                    validation.phase(name);
                    gpu.begin();
                    auto actual = record(gpu, *pooled, false);
                    auto expected = record(gpu, reference, true);
                    gpu.finish();
                    check(read(actual) == read(expected), std::string("Pooled/forced shadow mismatch: ") + name);
                    return actual;
                };
                for (uint32_t frame = 0; frame < 6; ++frame) { (void)serial("PoolWarmup"); }
                const auto warmed = pooled->resourcePoolStats();
                for (uint32_t frame = 0; frame < 8; ++frame) { (void)serial("PoolSteady"); }
                const auto steady = pooled->resourcePoolStats();
                check(steady.bufferCreates == warmed.bufferCreates, "Steady VSM still creates buffers");
                check(steady.textureCreates == warmed.textureCreates, "Steady VSM still creates textures");
                check(steady.bufferReuses > warmed.bufferReuses && steady.textureReuses > warmed.textureReuses,
                    "Steady VSM did not reuse pooled resources");

                const auto small = settings;
                settings.virtualPagesPerAxis = 16; settings.physicalPageCount = 512; settings.maxWorkItems = 131070;
                (void)serial("PoolCapacityGrowth");
                (void)serial("PoolCapacityGrowthStable");
                const auto grown = pooled->resourcePoolStats();
                check(grown.bufferCreates > steady.bufferCreates && grown.textureCreates > steady.textureCreates,
                    "Resource pool growth fixture did not exercise new capacity");
                settings = small;
                for (uint32_t frame = 0; frame < 4; ++frame) { (void)serial("PoolCapacityShrink"); }
                const auto shrunk = pooled->resourcePoolStats();
                for (uint32_t frame = 0; frame < 4; ++frame) { (void)serial("PoolCapacityShrinkStable"); }
                const auto restabilized = pooled->resourcePoolStats();
                check(restabilized.bufferCreates == shrunk.bufferCreates && restabilized.textureCreates == shrunk.textureCreates,
                    "VSM capacity change did not return to steady resource reuse");
                check(restabilized.retainedTextureBytes <= steady.retainedTextureBytes,
                    "Resource pool retained the old atlas after capacity shrink");

                const auto original = poolScene.nodes()[1].localMatrix;
                float4x4 moved;
                moved.SetupByTranslation(float3(.4f, 0, 0));
                check(poolScene.setNodeLocalMatrix(1, moved * original), "Edit geometry before pooled cancellation");
                gpu.begin();
                auto cancelled = record(gpu, *pooled, false);
                gpu.finish(true);
                check(gpu.frame.completion().isCancelled(), "Resource pool cancellation did not cancel completion");
                cancelled.resources.resourceOwner.reset();
                {
                    auto retry = serial("PoolCancelledRetry");
                    check(retry.statistics[28] == 1, "Pooled retry did not report cancellation invalidation");
                }
                check(poolScene.setNodeLocalMatrix(1, original), "Restore geometry after pooled cancellation");
                auto completed = serial("PoolCompletedPreviousPage");

                auto gate = device.createSemaphore();
                check(gate, "Resource pool lifetime gate");
                ContentCommands earlier(device), later(device);
                VirtualShadowMaps earlierReference, laterReference;
                Capture old, oldReference, next, nextReference;
                // Release the gate before any retained commands/resources unwind.
                struct Drain
                {
                    Queue& queue;
                    Semaphore& gate;
                    ~Drain()
                    {
                        if (gate.currentValue() < 1) { (void)gate.signal(1); }
                        (void)queue.waitIdle();
                    }
                } drain{*device.getQueue(QueueType::Graphics), **gate};
                check(poolScene.setNodeLocalMatrix(1, moved * original), "First in-flight pooled geometry edit");
                earlier.begin();
                old = record(earlier, *pooled, false);
                oldReference = record(earlier, earlierReference, true);
                check(old.resources.pageTable != completed.resources.pageTable,
                    "Current VSM overwrote its previous page table");
                old.resources.resourceOwner.reset();
                earlier.submit(gate->get());
                check(!earlier.frame.completion().isComplete(), "Resource pool gate did not hold GPU work");
                // A has completed; B is still reading A behind the gate. Remove
                // A's command/frame owners so B's cache reader protects A alone.
                completed.resources.resourceOwner.reset();
                check(gpu.pool->reset(), "Release completed VSM command owners");
                check(gpu.frame.reset(), "Release completed VSM frame owners");
                moved.SetupByTranslation(float3(.9f, 0, 0));
                check(poolScene.setNodeLocalMatrix(1, moved * original), "Second in-flight pooled geometry edit");
                later.begin();
                next = record(later, *pooled, false);
                nextReference = record(later, laterReference, true);
                check(next.resources.pageTable != completed.resources.pageTable && next.resources.pageTable != old.resources.pageTable,
                    "Pooled page table reused while a completed producer still had an in-flight reader");
                check(next.resources.projections != old.resources.projections && next.resources.statistics != old.resources.statistics &&
                    next.resources.texture != old.resources.texture,
                    "Pooled upload, statistics or output resource reused before GPU completion");
                const auto inFlight = pooled->resourcePoolStats();
                next.resources.resourceOwner.reset();
                later.submit();
                pooled.reset();
                check((*gate)->signal(1), "Release resource pool lifetime gate");
                check(earlier.frame.wait(), "Wait first pooled recording");
                check(later.frame.wait(), "Wait second pooled recording");
                const auto oldPixels = read(old), nextPixels = read(next);
                check(old.statistics[22] == 0 && next.statistics[22] == 0,
                    "Gated resource pool fixture did not consume previous page tables");
                check(old.statistics[25] > 0 && next.statistics[25] > 0,
                    "Gated resource pool geometry edits did not exercise dirty bounds");
                check(oldPixels == read(oldReference), "First in-flight pooled shadow corrupted after replacement/destruction");
                check(nextPixels == read(nextReference), "Second in-flight pooled shadow differs from forced redraw");
                check(oldPixels != nextPixels, "Resource pool geometry edits did not change the shadow");
                report["resourcePool"] = {{"steadyFrames", 8}, {"steadyBufferCreates", steady.bufferCreates - warmed.bufferCreates},
                    {"steadyTextureCreates", steady.textureCreates - warmed.textureCreates},
                    {"steadyBufferReuses", steady.bufferReuses - warmed.bufferReuses},
                    {"steadyTextureReuses", steady.textureReuses - warmed.textureReuses},
                    {"growthBufferCreates", grown.bufferCreates - steady.bufferCreates},
                    {"growthTextureCreates", grown.textureCreates - steady.textureCreates},
                    {"steadyRetainedTextureBytes", steady.retainedTextureBytes},
                    {"grownRetainedTextureBytes", grown.retainedTextureBytes},
                    {"shrunkRetainedTextureBytes", restabilized.retainedTextureBytes},
                    {"retainedBuffers", inFlight.retainedBuffers}, {"retainedBufferBytes", inFlight.retainedBufferBytes},
                    {"retainedTextures", inFlight.retainedTextures}, {"retainedTextureBytes", inFlight.retainedTextureBytes},
                    {"inFlightBufferCreates", inFlight.bufferCreates}, {"cancelledRetryEqual", true},
                    {"capacityChangesEqual", true}, {"completedProducerReaderProtected", true},
                    {"pendingUploadProtected", true}, {"rendererDestroyedBeforeGPUCompletion", true},
                    {"firstInFlightStatistics", old.statistics}, {"secondInFlightStatistics", next.statistics},
                    {"bothInFlightShadowsEqual", true}, {"inFlightShadowsChanged", true}};
            }
            report.erase("activeCase");
            report["validation"] = validation.snapshot();
            check(!validation.failed(), "Vulkan validation reported content errors");
            report["passed"] = true;
            std::ofstream(directory / "VirtualShadowMapContent.json") << report.dump(2);
            return RHITestResult::pass("Resident cache/uncached GPU geometry and shadows agree; resource pooling, capacity changes, cancellation and gated in-flight readers verified with Resident/Stream height oracle");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot();
            std::filesystem::create_directories(directory);
            std::ofstream(directory / "VirtualShadowMapContent.json") << report.dump(2);
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapContentTest);

class VirtualShadowMapScrollPrecisionTest final : public RHITest
{
public:
    VirtualShadowMapScrollPrecisionTest() { name = "virtual_shadow_map_scroll_precision"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapScrollPrecision");
        std::filesystem::create_directories(directory);
        Json report{{"passed", false}, {"cases", Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&]() { std::ofstream(directory / "VirtualShadowMapScrollPrecision.json") << report.dump(2); };
        try {
            auto deviceResult = createDevice({.applicationName = "VSM scroll precision",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(deviceResult, Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and bindless"); }
            check(deviceResult, "Scroll device");
            auto& device = **deviceResult;
            ContentCommands gpu(device);
            std::string log;
            const auto unit = [](std::array<float, 3> v) {
                const float length = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                for (float& x : v) { x /= length; }
                return v;
            };
            const auto direction = unit({-.3f, -.94f, -.16f});
            const auto right = unit({direction[2], 0, -direction[0]});
            bool exact = true;
            for (float extent : {4.f, .01f}) { for (uint32_t axis : {128u, 127u}) {
                const std::string label = "Extent-" + std::to_string(extent) + "-Axis-" + std::to_string(axis);
                validation.phase(label);
                const auto fixtureDirectory = directory / label;
                std::filesystem::create_directories(fixtureDirectory);
                // Non-binary vertices and an oblique light exercise depth and
                // clipping interpolation; an axis-aligned constant-Z quad does not.
                std::array<float, 12> positions{-1.137f,-1.291f,0, 1.423f,-1.291f,0,
                    1.423f,1.173f,0, -1.137f,1.173f,0};
                for (uint32_t vertex = 0; vertex < 4; ++vertex) {
                    positions[vertex * 3 + 2] = .213f + .173f * positions[vertex * 3] - .117f * positions[vertex * 3 + 1];
                    for (uint32_t component = 0; component < 3; ++component) { positions[vertex * 3 + component] *= extent; }
                }
                const uint32_t indices[]{0,1,2, 0,2,3};
                {
                    std::ofstream binary(fixtureDirectory / "Geometry.bin", std::ios::binary);
                    binary.write(reinterpret_cast<const char*>(positions.data()), sizeof(positions));
                    binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
                    check(binary.good(), "Write scroll geometry");
                }
                Json document = Json::parse(R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
                    "nodes":[{"mesh":0}],"buffers":[{"uri":"Geometry.bin","byteLength":72}],
                    "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":24}],
                    "accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3"},
                        {"bufferView":1,"componentType":5125,"count":6,"type":"SCALAR"}],
                    "materials":[{"doubleSided":true}],
                    "meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1,"material":0}]}]})");
                for (uint32_t component = 0; component < 3; ++component) {
                    float low = positions[component], high = low;
                    for (uint32_t vertex = 1; vertex < 4; ++vertex) {
                        low = std::min(low, positions[vertex * 3 + component]); high = std::max(high, positions[vertex * 3 + component]);
                    }
                    document["accessors"][0]["min"].push_back(low); document["accessors"][0]["max"].push_back(high);
                }
                const auto path = fixtureDirectory / "Scroll.gltf";
                { std::ofstream file(path); file << document.dump(2); check(file.good(), "Write scroll fixture"); }
                scene::SceneDocument scene;
                check(scene.load(path), "Load scroll fixture");
                VirtualShadowMapCasterCache casterCache;
                VirtualShadowMaps shadows;
                VirtualShadowMapLight light; light.direction = direction;
                VirtualShadowMapSettings settings;
                settings.pageSize = 32; settings.virtualPagesPerAxis = axis; settings.clipmapCount = 1;
                settings.firstClipmapExtent = extent; settings.physicalPageCount = 512;
                settings.receiverFootprint = false; settings.cacheEnabled = true;
                RenderView camera;
                std::unique_ptr<Texture> depth;
                std::unique_ptr<TextureView> depthView;
                check(device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                    .format = Format::D32Sfloat, .width = kWidth, .height = kHeight})
                    .transform([&](auto t) { depth = std::move(t); }), "Scroll receiver depth");
                check(device.createTextureView(*depth, {.format = Format::D32Sfloat})
                    .transform([&](auto v) { depthView = std::move(v); }), "Scroll depth view");
                auto upload = device.createBuffer({.size = kPixels * sizeof(float), .usage = BufferUsageBits::TransferSource,
                    .memoryLocation = MemoryLocation::HostUpload});
                check(upload, "Scroll receiver upload");
                auto* values = static_cast<float*>((*upload)->map()); check(values != nullptr, "Map scroll receiver");
                // The depth fixture is a plane perpendicular to the camera,
                // two extent units away; it requests a fixed visible page patch.
                std::fill(values, values + kPixels, (4.f - 2.f) / (4.f - .01f));
                (*upload)->flush(); (*upload)->unmap();
                gpu.begin();
                const TextureBarrierDesc begin{.texture = depth.get(), .oldLayout = TextureLayout::Undefined,
                    .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
                check(gpu.commands->synchronize({.textures = {&begin, 1}}), "Scroll depth begin");
                check((*upload)->slice().and_then([&](const auto& slice) {
                    return gpu.commands->copyBufferToTexture({.texture = depth.get(), .buffer = slice, .width = kWidth, .height = kHeight});
                }), "Scroll depth upload");
                const TextureBarrierDesc ready{.texture = depth.get(), .oldLayout = TextureLayout::TransferDestination,
                    .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
                check(gpu.commands->synchronize({.textures = {&ready, 1}}), "Scroll depth ready");
                gpu.finish();
                struct Capture {
                    std::vector<uint32_t> atlas;
                    std::vector<std::array<uint32_t,4>> pages;
                    std::array<uint32_t,32> stats{};
                    std::vector<uint8_t> visibility;
                    uint32_t atlasWidth = 0, physicalPagesX = 0;
                };
                const auto record = [&](uint32_t scroll, bool force) {
                    const float shift = float(scroll) * (2 * extent / axis);
                    std::array<float,3> center{.337f * extent, -.193f * extent, (.213f + .173f * .337f + .117f * .193f) * extent};
                    std::array<float,3> eye{};
                    for (uint32_t i = 0; i < 3; ++i) { center[i] += right[i] * shift; eye[i] = center[i] - direction[i] * (2 * extent); }
                    check(camera.setCamera({.eye = {eye[0],eye[1],eye[2]}, .center = {center[0],center[1],center[2]},
                        .nearPlane = .01f * extent, .farPlane = 4 * extent, .orthoHeight = .24f * extent, .orthographic = true}), "Scroll camera");
                    gpu.begin();
                    auto casters = casterCache.prepare(device, *gpu.commands, scene, nullptr, log); check(casters, "Scroll casters: " + log);
                    settings.forceRedraw = force;
                    const auto shadow = shadows.record(device, *gpu.commands, *depthView,
                        camera.constants(gpu.index, kWidth, kHeight, kWidth, kHeight), light, *casters, scene.materials(), {}, settings, log);
                    check(shadow, "Scroll shadow: " + log);
                    Capture capture;
                    const auto atlasDesc = shadow->atlas->desc();
                    capture.atlasWidth = atlasDesc.width; capture.physicalPagesX = shadow->physicalPagesX;
                    capture.atlas.resize(size_t(atlasDesc.width) * atlasDesc.height); capture.pages.resize(axis * axis + 1);
                    capture.visibility.resize(kPixels);
                    const uint64_t atlasBytes = capture.atlas.size() * sizeof(uint32_t), tableBytes = capture.pages.size() * sizeof(capture.pages[0]);
                    const uint64_t statsOffset = atlasBytes + tableBytes, visibilityOffset = statsOffset + sizeof(capture.stats);
                    auto readback = device.createBuffer({.size = visibilityOffset + kPixels, .usage = BufferUsageBits::TransferDestination,
                        .memoryLocation = MemoryLocation::HostReadback});
                    check(readback, "Scroll atlas readback");
                    const TextureBarrierDesc textures[]{
                        {.texture = shadow->atlas, .oldLayout = TextureLayout::ShaderRead, .newLayout = TextureLayout::TransferSource,
                            .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::DepthStencilWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}},
                        {.texture = shadow->texture, .oldLayout = TextureLayout::ShaderRead, .newLayout = TextureLayout::TransferSource,
                            .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}}};
                    const BufferBarrierDesc buffers[]{
                        {.buffer = shadow->pageTable, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}},
                        {.buffer = shadow->statistics, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}}};
                    check(gpu.commands->synchronize({.textures = textures, .buffers = buffers}), "Scroll readback barriers");
                    check((*readback)->slice({0, atlasBytes}).and_then([&](const auto& slice) {
                        return gpu.commands->copyTextureToBuffer({.texture = shadow->atlas, .buffer = slice, .width = atlasDesc.width, .height = atlasDesc.height});
                    }), "Scroll atlas copy");
                    check((*readback)->slice({visibilityOffset, kPixels}).and_then([&](const auto& slice) {
                        return gpu.commands->copyTextureToBuffer({.texture = shadow->texture, .buffer = slice, .width = kWidth, .height = kHeight});
                    }), "Scroll visibility copy");
                    for (const auto& [source, offset, bytes] : {std::tuple{shadow->pageTable, atlasBytes, tableBytes},
                        std::tuple{shadow->statistics, statsOffset, uint64_t(sizeof(capture.stats))}}) {
                        auto from = source->slice({0, bytes}), to = (*readback)->slice({offset, bytes});
                        check(from, "Scroll buffer source"); check(to, "Scroll buffer destination");
                        check(gpu.commands->copyBuffer(*from, *to), "Scroll buffer copy");
                    }
                    const TextureBarrierDesc restore[]{
                        {.texture = shadow->atlas, .oldLayout = TextureLayout::TransferSource, .newLayout = TextureLayout::ShaderRead,
                            .before = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}},
                        {.texture = shadow->texture, .oldLayout = TextureLayout::TransferSource, .newLayout = TextureLayout::ShaderRead,
                            .before = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}}};
                    check(gpu.commands->synchronize({.textures = restore}), "Restore scroll cache layouts");
                    gpu.finish();
                    (*readback)->invalidate(); const auto* bytes = static_cast<const std::byte*>((*readback)->map());
                    check(bytes != nullptr, "Map scroll atlas");
                    std::memcpy(capture.atlas.data(), bytes, atlasBytes); std::memcpy(capture.pages.data(), bytes + atlasBytes, tableBytes);
                    std::memcpy(capture.stats.data(), bytes + statsOffset, sizeof(capture.stats));
                    std::memcpy(capture.visibility.data(), bytes + visibilityOffset, kPixels); (*readback)->unmap();
                    check(capture.stats[2] == 0 && capture.stats[5] == 0 && capture.stats[7] == 0 && capture.stats[9] == 0,
                        "Scroll fixture must use fully resident fine pages");
                    return capture;
                };
                (void)record(0, false);
                for (uint32_t scroll : {1u, 2u, 7u}) {
                    const auto cached = record(scroll, false), forced = record(scroll, true);
                    check(cached.stats[17] > 0 && cached.stats[19] > 0, "Scroll must retain and remap pages");
                    check(cached.atlasWidth == forced.atlasWidth && cached.physicalPagesX == forced.physicalPagesX, "Scroll atlas layout changed");
                    uint32_t pages = 0, occupied = 0, depthDifferences = 0, visibilityDifferences = 0;
                    // Inspect fine pages only; coarse coverage must not make
                    // an otherwise empty precision fixture pass its coverage gate.
                    for (size_t page = 0; page + 1 < cached.pages.size(); ++page) {
                        const auto& a = cached.pages[page]; const auto& b = forced.pages[page];
                        check((a[1] & 2u) == (b[1] & 2u), "Scroll admission changed during force redraw");
                        if ((a[1] & 2u) == 0u) { continue; }
                        check(a[0] == b[0] && a[0] != 0xffffffffu && (a[3] & 1u) != 0u && (b[3] & 1u) != 0u,
                            "Scroll physical page or valid depth changed during force redraw");
                        ++pages;
                        for (uint32_t y = 0; y < settings.pageSize; ++y) { for (uint32_t x = 0; x < settings.pageSize; ++x) {
                            const size_t texel = size_t(a[0] / cached.physicalPagesX * settings.pageSize + y) * cached.atlasWidth +
                                a[0] % cached.physicalPagesX * settings.pageSize + x;
                            occupied += cached.atlas[texel] != 0; depthDifferences += cached.atlas[texel] != forced.atlas[texel];
                        }}
                    }
                    for (size_t pixel = 0; pixel < kPixels; ++pixel) { visibilityDifferences += cached.visibility[pixel] != forced.visibility[pixel]; }
                    check(pages > 20 && occupied > 1000, "Scroll fixture has insufficient raster coverage");
                    exact &= depthDifferences == 0 && visibilityDifferences == 0;
                    report["cases"].push_back({{"extent",extent},{"axis",axis},{"scrollPages",scroll},
                        {"admittedPages",pages},{"occupiedTexels",occupied},{"reusedPages",cached.stats[17]},{"remappedPages",cached.stats[19]},
                        {"changedDepthTexels",depthDifferences},{"changedVisibilityBytes",visibilityDifferences}});
                    save();
                }
            }}
            report["validation"] = validation.snapshot();
            check(!validation.failed(), "Scroll precision Vulkan validation error");
            check(exact, "Cached and forced D32 pages differ after clipmap scrolling");
            report["passed"] = true; save();
            return RHITestResult::pass("Scrolled directional pages retain bit-identical D32 depth at binary/non-binary extents and page grids");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapScrollPrecisionTest);
} // namespace
} // namespace metallic::tests
