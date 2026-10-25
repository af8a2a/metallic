#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "harness/VulkanDiagnostics.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace metallic::tests {
namespace {

using namespace render;
using Json = nlohmann::json;
using Point = std::array<double, 3>;
constexpr uint32_t kWidth = 257, kHeight = 193;
constexpr size_t kPixels = size_t(kWidth) * kHeight;
constexpr uint32_t kPageSize = 128, kPagesPerAxis = 16, kClipmapCount = 4;
constexpr float kFirstExtent = 4;

void require(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

template<class T>
void require(const Result<T>& result, const std::string& message)
{
    require(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

Point add(Point a, Point b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Point scale(Point a, double amount)
{
    return {a[0] * amount, a[1] * amount, a[2] * amount};
}

double dot(Point a, Point b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Point cross(Point a, Point b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

Point normalized(Point value)
{
    return scale(value, 1 / std::sqrt(dot(value, value)));
}

Point point(const float* value)
{
    return {value[0], value[1], value[2]};
}

class VSMDepthFixturePass final : public RasterPass {
public:
    RenderGraphSceneDependency sceneDependency() const override
    {
        return {RenderGraphSceneSource::Input, {"depth", "rasterInfo"}};
    }

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addBindlessSampledInput("depth").format = Format::D32Sfloat;
        reflection.addBufferInput("rasterInfo").buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo))
            .shaderRead().memoryLocation = MemoryLocation::HostUpload;
        reflection.addTextureOutput("depth").depthStencilWrite()
            .transient(RenderGraphInitialization::FullOverwrite).format = Format::D32Sfloat;
        reflection.addBufferOutput("rasterInfo").buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo))
            .shaderRead().memoryLocation = MemoryLocation::HostUpload;
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        const char* entries[]{"vsmDepthFixtureVertexMain", "vsmDepthFixtureFragmentMain"};
        for (size_t index = 0; index < shaders_.size(); ++index) {
            auto compiled = compileSlangShaderToSpirv({.moduleName = "VirtualShadowMapDepthFixture",
                .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
            if (!compiled) { return std::unexpected(compiled.error()); }
            auto shader = context.device->createShaderModule({.spirv = compiled->spirv});
            if (!shader) { return std::unexpected(shader.error()); }
            shaders_[index] = std::move(*shader);
        }
        return context.device->createGraphicsPipeline({.vertexShader = {shaders_[0].get()},
            .fragmentShader = {shaders_[1].get()}, .depthStencilFormat = Format::D32Sfloat,
            .rasterization = {.cullMode = CullMode::None},
            .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = CompareOp::Always},
            .usesBindlessHeap = true})
            .transform([&](auto pipeline) { pipeline_ = std::move(pipeline); });
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        VisibilityBufferFrameInfo info;
        auto* input = context.inputBuffer("rasterInfo").buffer();
        const void* mapped = input->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(&info, mapped, sizeof(info));
        input->unmap();
        const bool standard = context.properties().value("standardDepth", false);
        info.clipOrtho[3] = standard ? 0.0f : 1.0f;
        auto* output = context.outputBuffer("rasterInfo").buffer();
        void* destination = output->map();
        if (!destination) { return makeError(Error::Failure); }
        std::memcpy(destination, &info, sizeof(info));
        output->flush();
        output->unmap();

        auto& commands = context.commandBuffer();
        const Rect area{.width = context.width(), .height = context.height()};
        const RenderingAttachmentDesc depth{.view = context.outputTexture("depth").view(),
            .layout = TextureLayout::DepthStencilAttachment, .loadOp = LoadOp::DontCare, .storeOp = StoreOp::Store};
        if (auto result = commands.beginRendering({.renderArea = area, .depthStencilAttachment = &depth}); !result) { return result; }
        if (auto result = commands.setViewport({.width = float(area.width), .height = float(area.height), .maxDepth = 1}); !result) { return result; }
        commands.setScissor(area);
        const uint32_t push[]{context.bindlessInput("depth")->shaderIndex, standard ? 1u : 0u};
        if (auto result = commands.bindExecution(pipeline_->execution(), push, sizeof(push)); !result) { return result; }
        if (auto result = commands.draw(3); !result) { return result; }
        commands.endRendering();
        return {};
    }

private:
    std::array<std::unique_ptr<ShaderModule>, 2> shaders_;
    std::unique_ptr<GraphicsPipeline> pipeline_;
};

// Graph-owned readback survives until the next graph compilation. Each snapshot
// is copied after PreviewRenderer has waited for its GPU submission.
struct EvidenceState {
    Buffer* readback = nullptr;
    std::map<std::string, Buffer*> namedReadbacks;
    VisibilityBufferFrameInfo camera;
};

std::shared_ptr<EvidenceState> activeEvidence;

struct ReadbackLayout {
    uint64_t shadow = 0;
    uint64_t debug = (kPixels + 3) & ~uint64_t(3);
    uint64_t rasterDepth = debug + kPixels * 4;
    uint64_t receiverDepth = rasterDepth + kPixels * 4;
    uint64_t statistics = receiverDepth + kPixels * 4;
    uint64_t parameters = statistics + 32 * sizeof(uint32_t);
    uint64_t size = parameters + sizeof(ScreenSpaceShadowParameters);
};

constexpr ReadbackLayout kReadback;

class VSMEvidencePass final : public UnsafePass {
public:
    explicit VSMEvidencePass(std::shared_ptr<EvidenceState> state) : state_(std::move(state)) {}

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("shadow").transferRead().format = Format::R8Unorm;
        reflection.addTextureInput("debug").transferRead().format = Format::RGBA8Unorm;
        reflection.addTextureInput("rasterDepth").transferRead().format = Format::D32Sfloat;
        reflection.addTextureInput("receiverDepth").transferRead().format = Format::D32Sfloat;
        reflection.addBufferInput("statistics").buffer(32 * sizeof(uint32_t), sizeof(uint32_t)).transferRead();
        reflection.addBufferInput("parameters").buffer(sizeof(ScreenSpaceShadowParameters), sizeof(ScreenSpaceShadowParameters)).transferRead();
        reflection.addBufferInput("rasterInfo").buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo))
            .shaderRead().memoryLocation = MemoryLocation::HostUpload;
        reflection.addBufferOutput("data").buffer(kReadback.size).transferWrite().hostReadback();
        reflection.addTextureOutput("color").transferWrite().format = Format::RGBA8Unorm;
        return reflection;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto& commands = context.commandBuffer();
        auto* buffer = context.outputBuffer("data").buffer();
        state_->readback = buffer;
        if (context.properties().contains("evidenceName")) {
            state_->namedReadbacks[context.properties().at("evidenceName").get<std::string>()] = buffer;
        }
        const char* fields[]{"shadow", "debug", "rasterDepth", "receiverDepth"};
        const uint64_t offsets[]{kReadback.shadow, kReadback.debug, kReadback.rasterDepth, kReadback.receiverDepth};
        const uint64_t sizes[]{kPixels, kPixels * 4, kPixels * 4, kPixels * 4};
        for (size_t index = 0; index < std::size(fields); ++index) {
            auto slice = buffer->slice({offsets[index], sizes[index]});
            if (!slice) { return std::unexpected(slice.error()); }
            if (auto result = commands.copyTextureToBuffer({.texture = context.inputTexture(fields[index]).texture(),
                .buffer = *slice, .width = kWidth, .height = kHeight}); !result) { return result; }
        }
        const char* bufferFields[]{"statistics", "parameters"};
        const uint64_t bufferOffsets[]{kReadback.statistics, kReadback.parameters};
        const uint64_t bufferSizes[]{32 * sizeof(uint32_t), sizeof(ScreenSpaceShadowParameters)};
        for (size_t index = 0; index < std::size(bufferFields); ++index) {
            auto source = context.inputBuffer(bufferFields[index]).buffer()->slice({0, bufferSizes[index]});
            if (!source) { return std::unexpected(source.error()); }
            auto destination = buffer->slice({bufferOffsets[index], bufferSizes[index]});
            if (!destination) { return std::unexpected(destination.error()); }
            if (auto result = commands.copyBuffer(*source, *destination); !result) { return result; }
        }
        auto* infoBuffer = context.inputBuffer("rasterInfo").buffer();
        const void* info = infoBuffer->map();
        if (!info) { return makeError(Error::Failure); }
        std::memcpy(&state_->camera, info, sizeof(state_->camera));
        infoBuffer->unmap();
        return commands.copyTexture({.source = context.inputTexture("debug").texture(),
            .destination = context.outputTexture("color").texture(), .width = kWidth, .height = kHeight, .depth = 1});
    }

private:
    std::shared_ptr<EvidenceState> state_;
};

void registerFixturePasses()
{
    static const bool registered = [] {
        return registerRenderGraphPassType("VSMDepthFixturePass", "Test-only actual raster depth conversion",
            [] { return std::make_unique<VSMDepthFixturePass>(); }) &&
            registerRenderGraphPassType("VSMEvidencePass", "Test-only VSM image and statistics readback",
            [] { return std::make_unique<VSMEvidencePass>(activeEvidence); });
    }();
    require(registered, "Register VSM fixture passes");
}

void requireValidPreview(const Json& capture)
{
    require(!capture.at("captureFailed").get<bool>(), "Preview validation message capture failed");
    for (const auto& message : capture.at("messages")) {
        const auto id = message.at("id").get<std::string>();
        // Match the legacy RHI test runner's VUID failures and also fail
        // synchronization hazards, while preserving all diagnostics in JSON.
        require(id.find("VUID-") == std::string::npos && id.find("SYNC-HAZARD") == std::string::npos,
            "Preview validation: " + id + ": " + message.at("text").get<std::string>());
    }
}

void renderAndRequire(RenderGraphPreviewRenderer& preview, RenderGraph& graph,
    std::string_view output = {})
{
    const auto result = preview.render(graph, kWidth, kHeight,
        output.empty() ? graph.firstOutputName() : output);
    require(result, "VSM graph render: " + preview.lastLog());
    require(preview.lastLog().find("Initial material compilation failed") == std::string::npos,
        "VSM fixture cannot accept the error-material fallback:\n" + preview.lastLog());
}

std::filesystem::path writeFixture(const std::filesystem::path& directory, bool caster)
{
    std::filesystem::create_directories(directory);
    const float attributes[]{
        -1,0,-1, 1,0,-1, 1,0,1, -1,0,1,
        0,1,0, 0,1,0, 0,1,0, 0,1,0,
        0,0, 1,0, 1,1, 0,1};
    const uint16_t indices[]{0,2,1,0,3,2};
    {
        std::ofstream binary(directory / "Scene.bin", std::ios::binary);
        binary.write(reinterpret_cast<const char*>(attributes), sizeof(attributes));
        binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
        require(bool(binary), "Write VSM fixture geometry");
    }
    Json nodes = Json::array({{{"name", "Receiver"}, {"mesh", 0}, {"scale", {20,1,20}}}});
    Json nodeIndices = Json::array({0});
    if (caster) {
        nodes.push_back({{"name", "OffscreenCaster"}, {"mesh", 0}, {"translation", {-8,8,0}}, {"scale", {1.5,1,1.5}}});
        nodeIndices.push_back(1);
    }
    Json gltf;
    gltf["asset"] = {{"version", "2.0"}};
    gltf["scene"] = 0;
    gltf["scenes"] = Json::array({Json{{"nodes", nodeIndices}}});
    gltf["nodes"] = std::move(nodes);
    gltf["buffers"] = Json::array({Json{{"uri", "Scene.bin"}, {"byteLength", sizeof(attributes) + sizeof(indices)}}});
    gltf["bufferViews"] = Json::array({
        Json{{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 48}},
        Json{{"buffer", 0}, {"byteOffset", 48}, {"byteLength", 48}},
        Json{{"buffer", 0}, {"byteOffset", 96}, {"byteLength", 32}},
        Json{{"buffer", 0}, {"byteOffset", 128}, {"byteLength", 12}}});
    gltf["accessors"] = Json::array({
        Json{{"bufferView", 0}, {"componentType", 5126}, {"count", 4}, {"type", "VEC3"}, {"min", {-1,0,-1}}, {"max", {1,0,1}}},
        Json{{"bufferView", 1}, {"componentType", 5126}, {"count", 4}, {"type", "VEC3"}},
        Json{{"bufferView", 2}, {"componentType", 5126}, {"count", 4}, {"type", "VEC2"}},
        Json{{"bufferView", 3}, {"componentType", 5123}, {"count", 6}, {"type", "SCALAR"}}});
    Json primitive;
    primitive["attributes"] = {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}};
    primitive["indices"] = 3;
    primitive["material"] = 0;
    Json mesh;
    mesh["primitives"] = Json::array({std::move(primitive)});
    gltf["meshes"] = Json::array({std::move(mesh)});
    Json material;
    material["name"] = "OpaqueDoubleSided";
    material["doubleSided"] = true;
    material["pbrMetallicRoughness"] = {{"baseColorFactor", {.7,.7,.7,1}}, {"metallicFactor", 0}, {"roughnessFactor", .8}};
    gltf["materials"] = Json::array({std::move(material)});
    const auto path = directory / "Scene.gltf";
    std::ofstream output(path);
    output << gltf.dump(2) << '\n';
    require(bool(output), "Write VSM fixture glTF");
    return path;
}

std::filesystem::path writeMaskFixture(const std::filesystem::path& directory)
{
    // Keep the opaque fixture's shared mesh unchanged. Only this separate
    // source assigns the offscreen caster its own fully rejected material.
    const auto path = writeFixture(directory, true);
    Json gltf;
    {
        std::ifstream input(path);
        input >> gltf;
    }
    Json casterMesh = gltf["meshes"][0];
    casterMesh["primitives"][0]["material"] = 1;
    gltf["meshes"].push_back(std::move(casterMesh));
    gltf["nodes"][1]["mesh"] = 1;
    Json casterMaterial = gltf["materials"][0];
    casterMaterial["name"] = "RejectedMaskCaster";
    casterMaterial["alphaMode"] = "MASK";
    casterMaterial["alphaCutoff"] = .5;
    casterMaterial["pbrMetallicRoughness"]["baseColorFactor"][3] = 0;
    gltf["materials"].push_back(std::move(casterMaterial));
    std::ofstream output(path);
    output << gltf.dump(2) << '\n';
    require(bool(output), "Write independent VSM MASK fixture");
    return path;
}

struct Case {
    bool orthographic = false;
    bool standardDepth = false;
    float cameraShift = 0;
    uint32_t pool = 256;
    std::string name;
    uint32_t maxWorkItems = 65535;
};

RenderGraph makeGraph(const std::filesystem::path& source, const std::filesystem::path& cache, const Case& test)
{
    RenderGraph graph;
    const Json camera{{"eye", {test.cameraShift, 4, 6}}, {"center", {test.cameraShift, 0, 0}}, {"up", {0,1,0}},
        {"fovDegrees", 50}, {"znear", .1}, {"zfar", 100}, {"orthoHeight", 7},
        {"projection", test.orthographic ? "orthographic" : "perspective"}};
    graph.setViewProperties({{"camera", camera}, {"temporalJitter", false},
        {"renderResolution", {{"width", kWidth}, {"height", kHeight}, {"adaptive", false}}}});
    graph.addNode("VisibilityBufferPass", "Raster", {{"path", source.generic_string()}, {"streamAssetPath", cache.generic_string()},
        {"autoBuildStreamAsset", false}, {"maxResidentPages", 32}, {"maxLockedFallbackPages", 32},
        {"maxActiveGroups", 128}, {"maxGpuPageRequests", 128}, {"maxTraversalWorkers", 32}, {"maxTraversalWorkItems", 256},
        {"autoLod", false}, {"lodLevel", 0}, {"instanceHzbCull", false}, {"meshletHzbCull", false},
        {"meshletNormalConeCull", false}, {"hybridRaster", false}, {"enableClas", false}, {"enableClusterRtx", false}, {"camera", camera}});
    graph.addNode("VSMDepthFixturePass", "Receiver", {{"standardDepth", test.standardDepth}});
    graph.addEdge("Raster.depth", "Receiver.depth");
    graph.addEdge("Raster.rasterInfo", "Receiver.rasterInfo");
    graph.addNode("VirtualShadowMapPass", "Shadows", {{"pageSize", kPageSize}, {"virtualPagesPerAxis", kPagesPerAxis},
        {"clipmapCount", kClipmapCount}, {"firstClipmapExtent", kFirstExtent}, {"physicalPageCount", test.pool},
        {"maxWorkItems", test.maxWorkItems}, {"shadowBias", .002}, {"shadowDebug", true}});
    graph.addEdge("Receiver.depth", "Shadows.depth");
    graph.addEdge("Receiver.rasterInfo", "Shadows.rasterInfo");
    graph.addNode("VSMEvidencePass", "Evidence");
    graph.addEdge("Raster.depth", "Evidence.rasterDepth");
    graph.addEdge("Receiver.depth", "Evidence.receiverDepth");
    graph.addEdge("Receiver.rasterInfo", "Evidence.rasterInfo");
    graph.addEdge("Shadows.shadow", "Evidence.shadow");
    graph.addEdge("Shadows.debug", "Evidence.debug");
    graph.addEdge("Shadows.statistics", "Evidence.statistics");
    graph.addEdge("Shadows.parameters", "Evidence.parameters");
    graph.markOutput("Evidence.color");
    return graph;
}

struct Snapshot {
    std::vector<uint8_t> shadow;
    std::vector<uint32_t> debug;
    std::vector<float> rasterDepth, receiverDepth;
    std::array<uint32_t, 32> statistics{};
    VisibilityBufferFrameInfo camera;
    ScreenSpaceShadowParameters parameters;
};

Snapshot snapshot(const EvidenceState& evidence, const std::string& name = {})
{
    auto* readback = name.empty() ? evidence.readback : evidence.namedReadbacks.at(name);
    require(readback != nullptr, "VSM evidence pass did not execute");
    require(evidence.camera.width == kWidth && evidence.camera.height == kHeight &&
        evidence.camera.viewport[1] == float(kWidth) && evidence.camera.viewport[2] == float(kHeight),
        "VSM fixture raster extent differs from the requested receiver extent");
    Snapshot result;
    result.shadow.resize(kPixels); result.debug.resize(kPixels);
    result.rasterDepth.resize(kPixels); result.receiverDepth.resize(kPixels);
    readback->invalidate();
    const auto* mapped = static_cast<const std::byte*>(readback->map());
    require(mapped != nullptr, "Map VSM scene evidence");
    std::memcpy(result.shadow.data(), mapped + kReadback.shadow, kPixels);
    std::memcpy(result.debug.data(), mapped + kReadback.debug, kPixels * 4);
    std::memcpy(result.rasterDepth.data(), mapped + kReadback.rasterDepth, kPixels * 4);
    std::memcpy(result.receiverDepth.data(), mapped + kReadback.receiverDepth, kPixels * 4);
    std::memcpy(result.statistics.data(), mapped + kReadback.statistics, sizeof(result.statistics));
    std::memcpy(&result.parameters, mapped + kReadback.parameters, sizeof(result.parameters));
    readback->unmap();
    require(result.parameters.view.current.viewport[1] == float(kWidth) &&
        result.parameters.view.current.viewport[2] == float(kHeight),
        "VSM shadow view extent differs from its raster metadata");
    result.camera = evidence.camera;
    return result;
}

struct FloorHit {
    Point position;
    double expectedDepth = 0;
    bool visible = false;
};

// Independent camera-ray / authored plane intersection. The reference does not
// use VSM projection data, page tables, physical tiles, or allocation logic.
FloorHit floorHit(const VisibilityBufferFrameInfo& camera, uint32_t x, uint32_t y)
{
    const Point eye = point(camera.eye);
    const Point forward = normalized(add(point(camera.center), scale(eye, -1)));
    const Point right = normalized(cross(forward, point(camera.upProjection)));
    const Point up = cross(right, forward);
    const double nx = 2 * (double(x) + .5 + camera.jitter[0]) / kWidth - 1;
    const double ny = 1 - 2 * (double(y) + .5 + camera.jitter[1]) / kHeight;
    Point origin = eye, direction = forward;
    const bool orthographic = camera.upProjection[3] > .5f;
    const double halfHeight = orthographic ? camera.clipOrtho[2] * .5 : std::tan(camera.viewport[3] * .5);
    const Point lateral = add(scale(right, nx * camera.viewport[0] * halfHeight), scale(up, ny * halfHeight));
    if (orthographic) { origin = add(origin, lateral); }
    else { direction = add(direction, lateral); }
    const double distance = -origin[1] / direction[1];
    FloorHit result;
    result.position = add(origin, scale(direction, distance));
    const double viewZ = dot(add(result.position, scale(eye, -1)), forward);
    const double nearPlane = camera.clipOrtho[0], farPlane = camera.clipOrtho[1];
    result.visible = distance > 0 && std::abs(result.position[0]) < 20 && std::abs(result.position[2]) < 20 &&
        viewZ >= nearPlane && viewZ <= farPlane;
    result.expectedDepth = orthographic ? (farPlane - viewZ) / (farPlane - nearPlane) :
        nearPlane * (farPlane - viewZ) / (viewZ * (farPlane - nearPlane));
    return result;
}

void requireCasterOffscreen(const VisibilityBufferFrameInfo& camera)
{
    const Point eye = point(camera.eye);
    const Point forward = normalized(add(point(camera.center), scale(eye, -1)));
    const Point right = normalized(cross(forward, point(camera.upProjection)));
    for (double x : {-9.5, -6.5}) {
        for (double z : {-1.5, 1.5}) {
            const Point delta = add(Point{x,8,z}, scale(eye, -1));
            const double halfWidth = camera.viewport[0] * (camera.upProjection[3] > .5f ? camera.clipOrtho[2] * .5 :
                dot(delta, forward) * std::tan(camera.viewport[3] * .5));
            require(dot(delta, right) < -halfWidth, "Caster fixture entered the camera frustum");
        }
    }
}

uint32_t rgba(uint8_t r, uint8_t g, uint8_t b)
{
    return uint32_t(r) | uint32_t(g) << 8 | uint32_t(b) << 16 | 0xff000000u;
}

void saveImage(const std::filesystem::path& path, const std::vector<uint32_t>& pixels)
{
    std::string log;
    require(saveRgba8Png(path, reinterpret_cast<const uint8_t*>(pixels.data()), kWidth, kHeight, log), log);
}

Json verifyAndSave(const std::filesystem::path& directory, const std::string& name,
    const Snapshot& image, const Case& test, bool hasCaster)
{
    requireCasterOffscreen(image.camera);
    size_t checked = 0, mismatched = 0, shadowed = 0, lit = 0, edgeExcluded = 0, depthMismatch = 0;
    size_t finePixels = 0, coarsePixels = 0, pageBoundaryPixels = 0, clipmapBoundaryPixels = 0;
    std::set<uint8_t> levels;
    std::vector<uint32_t> reference(kPixels), error(kPixels), shadow(kPixels), depth(kPixels);
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            const size_t index = size_t(y) * kWidth + x;
            const auto hit = floorHit(image.camera, x, y);
            const uint8_t actual = image.shadow[index];
            shadow[index] = rgba(actual, actual, actual);
            const uint8_t depthShade = uint8_t(std::clamp(image.rasterDepth[index] * 2500.0f, 0.0f, 255.0f));
            depth[index] = rgba(depthShade, depthShade, depthShade);
            const uint8_t red = uint8_t(image.debug[index]), green = uint8_t(image.debug[index] >> 8), blue = uint8_t(image.debug[index] >> 16);
            const bool coarse = blue > 200 && red < 30 && green < 30;
            const bool fine = green > 100 && green < 160 && blue < 30;
            if (coarse) { ++coarsePixels; }
            if (fine) { ++finePixels; levels.insert(red); }
            // Shadow from the horizontal quad at y=8 along travel direction
            // normalize(1,-1,0): its footprint on y=0 is exactly [-1.5,1.5]^2.
            const bool blocked = hasCaster && std::abs(hit.position[0]) < 1.5 && std::abs(hit.position[2]) < 1.5;
            reference[index] = blocked ? rgba(0,0,0) : rgba(255,255,255);
            error[index] = rgba(0,0,0);
            if (!hit.visible) { continue; }
            const float conversion = test.standardDepth ? 1.0f - image.rasterDepth[index] : image.rasterDepth[index];
            depthMismatch += std::abs(image.rasterDepth[index] - hit.expectedDepth) > 2e-5 ||
                image.receiverDepth[index] != conversion;
            // Exclude geometric shadow edges by a conservative world-space
            // texel margin. Coarse projection covers the complete 40 m fixture.
            const double margin = coarse ? .9 : .08;
            const bool nearEdge = hasCaster &&
                ((std::abs(std::abs(hit.position[0]) - 1.5) < margin && std::abs(hit.position[2]) < 1.5 + margin) ||
                 (std::abs(std::abs(hit.position[2]) - 1.5) < margin && std::abs(hit.position[0]) < 1.5 + margin));
            if (nearEdge) { ++edgeExcluded; error[index] = rgba(80,80,80); continue; }
            ++checked;
            if (blocked) { ++shadowed; } else { ++lit; }
            const bool mismatch = actual != (blocked ? 0 : 255);
            if (mismatch) { ++mismatched; error[index] = rgba(255,0,0); }
            // In this light basis x=-world.z, and the clipmap center snaps to
            // whole pages. Decode the actual selected level from its debug
            // palette, then test its page grid without relaxing the oracle.
            if (fine) {
                const int level = int(std::lround(double(red) * kClipmapCount / 255.0)) - 1;
                const double pageWidth = std::ldexp(2.0 * kFirstExtent / kPagesPerAxis, level);
                const double pageCoordinate = hit.position[2] / pageWidth;
                if (std::abs(pageCoordinate - std::round(pageCoordinate)) < .05) { ++pageBoundaryPixels; }
            }
            if (x > 0 && image.debug[index] != image.debug[index - 1] && fine) { ++clipmapBoundaryPixels; }
            if (y > 0 && image.debug[index] != image.debug[index - kWidth] && fine) { ++clipmapBoundaryPixels; }
        }
    }
    saveImage(directory / (name + "-Shadow.png"), shadow);
    saveImage(directory / (name + "-Reference.png"), reference);
    saveImage(directory / (name + "-Difference.png"), error);
    saveImage(directory / (name + "-Levels.png"), image.debug);
    saveImage(directory / (name + "-RasterDepth.png"), depth);
    const Json report{{"name", name}, {"statistics", image.statistics}, {"standardReceiverDepth", test.standardDepth},
        {"projection", test.orthographic ? "orthographic" : "perspective"}, {"cameraShift", test.cameraShift},
        {"physicalPageCapacity", test.pool}, {"hasCaster", hasCaster}, {"checked", checked}, {"mismatched", mismatched},
        {"maxWorkItems", test.maxWorkItems},
        {"lit", lit}, {"shadowed", shadowed}, {"edgeExcluded", edgeExcluded}, {"depthMismatch", depthMismatch},
        {"finePixels", finePixels}, {"coarsePixels", coarsePixels}, {"fineLevelCodes", levels},
        {"pageBoundaryPixels", pageBoundaryPixels}, {"clipmapBoundaryPixels", clipmapBoundaryPixels}};
    std::ofstream(directory / (name + ".json")) << report.dump(2) << '\n';
    require(depthMismatch == 0, name + ": actual raster/depth conversion disagrees with authored receiver plane");
    require(checked > 10000 && lit > 10000 && (!hasCaster || shadowed > 60), name + ": reference coverage too small");
    require(mismatched == 0, name + ": hard shadow differs from independent light-ray reference at " + std::to_string(mismatched) + " stable pixels");
    require(image.camera.clipOrtho[3] == (test.standardDepth ? 0 : 1) &&
        image.parameters.view.current.clipOrtho[3] == image.camera.clipOrtho[3], name + ": receiver depth mode metadata lost");
    require(image.statistics[9] == 0 && image.statistics[14] == 0,
        name + ": outside-all receiver or unexpectedly disabled VSM");
    if (test.maxWorkItems == 1) {
        require(image.statistics[3] > 1 && image.statistics[4] == 1 && image.statistics[5] > 0,
            name + ": bounded work overflow was not exercised");
    } else {
        require(image.statistics[5] == 0, name + ": unexpected work overflow");
    }
    require(image.statistics[0] > 1 && image.statistics[6] > 10000 && image.statistics[11] == kClipmapCount &&
        image.statistics[12] == test.pool, name + ": demand/receiver/clipmap/pool statistics invalid");
    if (test.pool == 1 || test.maxWorkItems == 1) {
        require(image.statistics[7] > 10000 && coarsePixels > 10000,
            name + ": overflow did not use the guaranteed coarse page");
        if (test.pool == 1) {
            require(image.statistics[1] == 0 && image.statistics[2] > 0,
                name + ": tiny pool allocation did not report overflow");
        }
    } else {
        require(pageBoundaryPixels > 100, name + ": virtual page boundary coverage not exercised");
        require(image.statistics[1] > 0 && finePixels > 10000 && levels.size() >= 2 && clipmapBoundaryPixels > 20,
            name + ": fine pages and clipmap boundary coverage not exercised");
    }
    return report;
}

Json verifyDeferred(const std::filesystem::path& directory, RenderGraphPreviewRenderer& preview,
    const std::filesystem::path& source, const std::filesystem::path& cache)
{
    activeEvidence = std::make_shared<EvidenceState>();
    auto graph = makeGraph(source, cache, Case{});
    graph.addNode("VisibilityBufferDeferredPass", "Deferred", {{"lightingMode", "realtime"}, {"debugView", "final"},
        {"materialBinning", false}, {"shadow.enabled", false}, {"accumulate", false}});
    graph.addEdge("Raster.visibility", "Deferred.visibility");
    graph.addEdge("Raster.depth", "Deferred.depth");
    graph.addEdge("Raster.rasterInfo", "Deferred.rasterInfo");
    graph.addEdge("Shadows.shadow", "Deferred.shadow");
    graph.addEdge("Shadows.parameters", "Deferred.shadowParameters");
    graph.addNode("FinalBlitPass", "Output");
    graph.addEdge("Deferred.color", "Output.source");
    graph.markOutput("Output.color");
    for (uint32_t frame = 0; frame < 8; ++frame) {
        renderAndRequire(preview, graph, "Output.color");
    }
    const auto shadowOn = preview.pixels();
    const auto evidenceOn = snapshot(*activeEvidence);
    saveImage(directory / "Deferred-ShadowOn.png", shadowOn);
    require(evidenceOn.parameters.control[0] == 1 && evidenceOn.parameters.control[1] == 1 &&
        evidenceOn.parameters.control[2] == kCelestialLightSourceTag, "Deferred VSM did not publish the enabled Sun identity");
    require(graph.setNodeRuntimeProperty(graph.findNode("Shadows")->id, "enabled", false), "Disable VSM for deferred control");
    for (uint32_t frame = 0; frame < 2; ++frame) {
        renderAndRequire(preview, graph, "Output.color");
    }
    const auto shadowOff = preview.pixels();
    const auto evidenceOff = snapshot(*activeEvidence);
    saveImage(directory / "Deferred-ShadowOff.png", shadowOff);
    require(evidenceOff.parameters.control[0] == 0, "Deferred VSM control did not disable the Sun shadow");
    require(evidenceOn.rasterDepth == evidenceOff.rasterDepth, "Deferred shadow toggle changed receiver geometry");
    size_t shadowRegion = 0, shadowDarker = 0, litRegion = 0, litChanged = 0;
    std::vector<uint32_t> difference(kPixels);
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            const size_t index = size_t(y) * kWidth + x;
            const auto hit = floorHit(evidenceOn.camera, x, y);
            int on = 0, off = 0, maxDifference = 0;
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const int a = int((shadowOn[index] >> (channel * 8)) & 255u);
                const int b = int((shadowOff[index] >> (channel * 8)) & 255u);
                on += a; off += b; maxDifference = std::max(maxDifference, std::abs(a - b));
            }
            difference[index] = rgba(uint8_t(maxDifference), uint8_t(maxDifference), uint8_t(maxDifference));
            if (!hit.visible) { continue; }
            if (std::abs(hit.position[0]) < 1.3 && std::abs(hit.position[2]) < 1.3) {
                ++shadowRegion;
                shadowDarker += off > on + 30;
            } else if (std::abs(hit.position[0]) > 1.8 || std::abs(hit.position[2]) > 1.8) {
                ++litRegion;
                litChanged += maxDifference > 2;
            }
        }
    }
    saveImage(directory / "Deferred-Difference.png", difference);
    const Json report{{"shadowRegionPixels", shadowRegion}, {"shadowDarkerPixels", shadowDarker},
        {"litRegionPixels", litRegion}, {"litChangedPixels", litChanged}, {"deviceRayQueryEnabled", false}};
    std::ofstream(directory / "Deferred.json") << report.dump(2) << '\n';
    require(shadowRegion > 100 && shadowDarker == shadowRegion, "VSM shadow did not darken the deferred Sun direct-light region");
    require(litRegion > 10000 && litChanged == 0, "VSM toggle changed deferred lighting outside the shadow region");
    return report;
}

Json verifyMaskScene(RHITestContext& context, const std::filesystem::path& directory,
    const std::filesystem::path& source, const std::filesystem::path& cache,
    bool metadataOnly, bench::ValidationRecorder& validation)
{
    scene::SceneDocument scene;
    require(metadataOnly ? scene.loadStreamMetadata(source) : scene.load(source), scene.lastLoadResult().error);
    require(scene.hasStreamGeometry() == metadataOnly && scene.materials().size() == 2 &&
        scene.materials()[0].alphaMode == "OPAQUE" && scene.materials()[1].alphaMode == "MASK" &&
        scene.materials()[1].baseColorFactor.w == 0 && scene.renderNodes()[1].materialIndex == 1,
        "MASK fixture did not preserve its opaque receiver and separate rejected caster");
    RenderGraphPreviewRenderer preview;
    preview.bindRuntimeScene(&scene);
    validation.phase("MASK Preview initialize");
    require(preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink()),
        "MASK preview initialize");
    preview.setEnvironment({.enabled = false});
    auto lighting = scene.lighting();
    lighting.autoExposure.enabled = false; lighting.exposureEV100 = 0;
    require(preview.setLighting(lighting), "Set MASK fixture lighting");
    environment::WorldEnvironment world;
    world.sun.enabled = true; world.sun.illuminance = 3; world.sun.angularRadius = 0;
    world.sun.direction = float3(1, -1, 0); world.moon.enabled = false;
    require(preview.setWorldEnvironment(world), "Set MASK fixture Sun");
    Json report = Json::array();
    for (uint32_t pool : {256u, 1u}) {
        const Case test{false, false, 0, pool, pool == 1 ? "MaskRejectedCoarse" : "MaskRejectedFine"};
        const std::string name = std::string(metadataOnly ? "Stream-" : "Resident-") + test.name;
        validation.phase(name);
        activeEvidence = std::make_shared<EvidenceState>();
        auto graph = makeGraph(source, cache, test);
        for (uint32_t frame = 0; frame < 8; ++frame) { renderAndRequire(preview, graph); }
        auto& device = *preview.subsystemHost()->device();
        require(!device.capabilities().rayQuery && !device.capabilities().rayTracingAccelerationStructure,
            "MASK fixture accidentally enabled ray tracing");
        require(!context.enableValidation || bench::activeValidation(device) == bench::Validation::Synchronization,
            "MASK preview synchronization validation did not activate");
        const auto image = snapshot(*activeEvidence);
        report.push_back(verifyAndSave(directory, name, image, test, false));
        require(metadataOnly ? image.statistics[10] >= 128 : image.statistics[10] == 4,
            name + ": MASK did not exercise its intended caster source");
        require(std::all_of(image.shadow.begin(), image.shadow.end(), [](uint8_t visibility) { return visibility == 255; }),
            name + ": a rejected MASK caster or an empty depth sample produced a shadow");
        const auto captured = validation.snapshot();
        std::ofstream(directory / "PreviewValidation.json") << captured.dump(2) << '\n';
        requireValidPreview(captured);
    }
    return report;
}

class VirtualShadowMapSceneTest final : public RHITest {
public:
    VirtualShadowMapSceneTest() { type = RHITestType::Rendering; name = "virtual_shadow_map_scene"; }

    RHITestResult run(RHITestContext& context) override
    {
        bench::ValidationRecorder validation;
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapScene";
        try {
            registerFixturePasses();
            std::filesystem::create_directories(directory);
            std::array<std::filesystem::path, 2> sources, caches;
            std::string log;
            for (uint32_t caster = 0; caster < 2; ++caster) {
                const auto fixtureDirectory = directory / (caster ? "Caster" : "ReceiverOnly");
                sources[caster] = writeFixture(fixtureDirectory, caster != 0);
                caches[caster] = fixtureDirectory / "Scene.meshstream.bin";
                require(scene::buildMeshletStreamAssetOffline({.sourcePath = sources[caster], .outputPath = caches[caster],
                    .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log), log);
            }
            const auto maskSource = writeMaskFixture(directory / "MaskCaster");
            const auto maskCache = maskSource.parent_path() / "Scene.meshstream.bin";
            require(scene::buildMeshletStreamAssetOffline({.sourcePath = maskSource, .outputPath = maskCache,
                .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log), log);
            std::vector<Case> cases;
            for (bool orthographic : {false, true}) {
                for (bool standard : {false, true}) {
                    const std::string name = std::string(orthographic ? "Ortho" : "Perspective") + (standard ? "Standard" : "Reversed");
                    cases.push_back({orthographic, standard, 0, 256, name});
                    cases.push_back({orthographic, standard, .51f, 256, name + "Shifted"});
                    cases.push_back({orthographic, standard, 0, 1, name + "PoolOverflow"});
                    cases.push_back({orthographic, standard, 0, 256, name + "WorkOverflow", 1});
                }
            }
            cases.push_back({true, false, 19, 1, "OrthoSceneEdge"});
            Json report{{"scope", "P1 actual glTF visibility depth to VSM hard shadows; no RT device features and no temporal VSM cache"},
                {"width", kWidth}, {"height", kHeight}, {"cases", Json::array()}, {"validationEnabled", context.enableValidation},
                {"standardDepthSource", "GPU fullscreen D32 conversion 1-depth from actual VisibilityBuffer depth, with matching camera metadata"},
                {"rasterGeometry", "Both camera-raster paths use cooked StreamAsset geometry; resident/metadata-only distinguish VSM caster acquisition"}};
            std::map<std::string, Snapshot> decoded;
            for (bool metadataOnly : {false, true}) {
                std::map<std::string, Snapshot> receiverOnly;
                for (uint32_t caster = 0; caster < 2; ++caster) {
                    scene::SceneDocument scene;
                    require(metadataOnly ? scene.loadStreamMetadata(sources[caster]) : scene.load(sources[caster]), scene.lastLoadResult().error);
                    require(scene.hasStreamGeometry() == metadataOnly, "Fixture scene did not select the intended caster source");
                    RenderGraphPreviewRenderer preview;
                    preview.bindRuntimeScene(&scene);
                    validation.phase("Preview initialize");
                    const auto initialized = preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink());
                    if (hasError(initialized, Error::Unsupported)) { return RHITestResult::skip("VSM scene requires Vulkan mesh shaders and native bindless descriptors"); }
                    require(initialized, "VSM preview initialize: " + preview.lastLog());
                    preview.setEnvironment({.enabled = false});
                    auto lighting = scene.lighting();
                    lighting.autoExposure.enabled = false; lighting.exposureEV100 = 0;
                    require(preview.setLighting(lighting), "Set fixture lighting");
                    environment::WorldEnvironment world;
                    world.sun.enabled = true; world.sun.illuminance = 3; world.sun.angularRadius = 0;
                    world.sun.direction = float3(1, -1, 0);
                    world.moon.enabled = false;
                    require(preview.setWorldEnvironment(world), "Set fixture Sun");
                    std::map<std::string, std::vector<uint8_t>> coarsePool;
                    for (const auto& test : cases) {
                        // Receiver-only is a focused offscreen-caster control;
                        // full parameter coverage belongs to the caster scene.
                        const bool sceneEdge = test.name == "OrthoSceneEdge";
                        if (caster && sceneEdge) { continue; }
                        if (!caster && !sceneEdge &&
                            (test.standardDepth || test.cameraShift != 0 || test.pool == 1 || test.maxWorkItems == 1)) { continue; }
                        activeEvidence = std::make_shared<EvidenceState>();
                        auto graph = makeGraph(sources[caster], caches[caster], test);
                        validation.phase(test.name);
                        for (uint32_t frame = 0; frame < 8; ++frame) {
                            renderAndRequire(preview, graph);
                        }
                        // The subsystem host receives its Device during graph
                        // compilation, not PreviewRenderer::initialize().
                        auto& device = *preview.subsystemHost()->device();
                        const auto& capabilities = device.capabilities();
                        require(!capabilities.rayQuery && !capabilities.rayTracingAccelerationStructure,
                            "VSM scene fixture accidentally enabled ray tracing");
                        const auto activeValidation = bench::activeValidation(device);
                        require(!context.enableValidation || activeValidation == bench::Validation::Synchronization,
                            "Preview synchronization validation was requested but did not activate");
                        report["activePreviewValidation"] = bench::name(activeValidation);
                        const auto captured = validation.snapshot();
                        std::ofstream(directory / "PreviewValidation.json") << captured.dump(2) << '\n';
                        requireValidPreview(captured);
                        auto image = snapshot(*activeEvidence);
                        const std::string name = std::string(metadataOnly ? "Stream-" : "Resident-") +
                            (caster ? "Caster-" : "ReceiverOnly-") + test.name;
                        report["cases"].push_back(verifyAndSave(directory, name, image, test, caster != 0));
                        require(metadataOnly ? image.statistics[10] >= 128 : image.statistics[10] == (caster ? 4 : 2),
                            name + ": triangle capacity proves the wrong caster source was exercised");
                        if (test.pool == 1) { coarsePool[test.name] = image.shadow; }
                        if (test.maxWorkItems == 1) {
                            const auto baselineName = test.name.substr(0, test.name.size() - std::string_view("WorkOverflow").size()) + "PoolOverflow";
                            require(image.shadow == coarsePool.at(baselineName),
                                name + ": work overflow differs from the independently complete coarse-page result");
                        }
                        std::ofstream(directory / "VirtualShadowMapScene.json") << report.dump(2) << '\n';
                        if (!caster) { receiverOnly[test.name] = image; continue; }
                        if (const auto control = receiverOnly.find(test.name); control != receiverOnly.end()) {
                            require(image.rasterDepth == control->second.rasterDepth,
                                name + ": offscreen caster changed main-view geometry depth");
                            size_t changed = 0;
                            for (size_t index = 0; index < kPixels; ++index) { changed += image.shadow[index] != control->second.shadow[index]; }
                            require(changed > 100, name + ": offscreen caster did not affect shadows");
                            report["offscreenCasterChangedPixels"][name] = changed;
                        }
                        if (!metadataOnly) { decoded[test.name] = image; }
                        else {
                            require(image.rasterDepth == decoded.at(test.name).rasterDepth, name + ": decoded/stream receiver geometry differs");
                            size_t differences = 0;
                            for (size_t index = 0; index < kPixels; ++index) { differences += image.shadow[index] != decoded.at(test.name).shadow[index]; }
                            report["residentStreamDifferentPixels"][test.name] = differences;
                            require(differences == 0, name + ": resident and StreamAsset caster paths disagree");
                        }
                        if (!test.standardDepth && test.pool == 256 && test.maxWorkItems != 1 && test.cameraShift == 0) {
                            renderAndRequire(preview, graph);
                            const auto repeated = snapshot(*activeEvidence);
                            require(repeated.shadow == image.shadow && repeated.debug == image.debug &&
                                std::equal(image.statistics.begin(), image.statistics.begin() + 16, repeated.statistics.begin()),
                                name + ": uncached fixed scene is not deterministic across frames");
                        }
                    }
                    if (!metadataOnly && caster) {
                        report["deferred"] = verifyDeferred(directory, preview, sources[caster], caches[caster]);
                    }
                }
            }
            report["mask"] = Json::array();
            for (bool metadataOnly : {false, true}) {
                for (auto& item : verifyMaskScene(context, directory, maskSource, maskCache, metadataOnly, validation)) {
                    report["mask"].push_back(std::move(item));
                }
            }
            activeEvidence.reset();
            report["previewValidation"] = validation.snapshot();
            std::ofstream(directory / "PreviewValidation.json") << report["previewValidation"].dump(2) << '\n';
            requireValidPreview(report["previewValidation"]);
            report["passed"] = true;
            std::ofstream(directory / "VirtualShadowMapScene.json") << report.dump(2) << '\n';
            return RHITestResult::pass("Real glTF raster depth, offscreen caster, resident/StreamAsset, perspective/orthographic, standard/reversed receivers, page/clipmap boundaries and coarse-page pool overflow; PNG/JSON exported");
        } catch (const std::exception& error) {
            activeEvidence.reset();
            std::ofstream(directory / "PreviewValidation.json") << validation.snapshot().dump(2) << '\n';
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapSceneTest);

std::filesystem::path writeCacheFixture(const std::filesystem::path& directory)
{
    const auto path = writeMaskFixture(directory);
    Json gltf;
    { std::ifstream input(path); input >> gltf; }
    gltf["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"][3] = 1;
    std::ofstream output(path);
    output << gltf.dump(2) << '\n';
    require(bool(output), "Write VSM cache fixture");
    return path;
}

void addCacheEvidence(RenderGraph& graph, const std::string& shadow, const std::string& evidence)
{
    graph.addNode("VSMEvidencePass", evidence, {{"evidenceName", shadow}});
    graph.addEdge("Raster.depth", evidence + ".rasterDepth");
    graph.addEdge("Receiver.depth", evidence + ".receiverDepth");
    graph.addEdge("Receiver.rasterInfo", evidence + ".rasterInfo");
    graph.addEdge(shadow + ".shadow", evidence + ".shadow");
    graph.addEdge(shadow + ".debug", evidence + ".debug");
    graph.addEdge(shadow + ".statistics", evidence + ".statistics");
    graph.addEdge(shadow + ".parameters", evidence + ".parameters");
    graph.markOutput(evidence + ".color");
}

RenderGraph makeCacheGraph(const std::filesystem::path& source, const std::filesystem::path& cache)
{
    auto graph = makeGraph(source, cache, Case{.pool = 512});
    auto settings = graph.findNode("Shadows")->properties;
    settings["cacheEnabled"] = true;
    settings["forceRedraw"] = false;
    settings["lightIndex"] = 0;
    require(graph.setNodeProperties(graph.findNode("Shadows")->id, settings), "Set cached Sun");
    require(graph.setNodeProperties(graph.findNode("Evidence")->id, {{"evidenceName", "Shadows"}}), "Name cached Sun evidence");
    for (const auto& name : {"SunReference", "Moon", "MoonReference"}) {
        auto lightSettings = settings;
        lightSettings["lightIndex"] = std::string_view(name).starts_with("Moon") ? 1 : 0;
        lightSettings["forceRedraw"] = std::string_view(name).ends_with("Reference");
        graph.addNode("VirtualShadowMapPass", name, lightSettings);
        graph.addEdge("Receiver.depth", std::string(name) + ".depth");
        graph.addEdge("Receiver.rasterInfo", std::string(name) + ".rasterInfo");
        addCacheEvidence(graph, name, std::string(name) + "Evidence");
    }
    return graph;
}

void saveVisibility(const std::filesystem::path& path, const std::vector<uint8_t>& visibility)
{
    std::vector<uint32_t> image(kPixels);
    for (size_t index = 0; index < kPixels; ++index) {
        image[index] = rgba(visibility[index], visibility[index], visibility[index]);
    }
    saveImage(path, image);
}

struct CacheFrame {
    Snapshot sun, moon, sunReference, moonReference;
};

CacheFrame compareCacheFrame(RenderGraphPreviewRenderer& preview, RenderGraph& graph,
    const std::filesystem::path& directory, const std::string& name, Json& report, bool allowWorkOverflow = false)
{
    renderAndRequire(preview, graph);
    CacheFrame frame{snapshot(*activeEvidence, "Shadows"), snapshot(*activeEvidence, "Moon"),
        snapshot(*activeEvidence, "SunReference"), snapshot(*activeEvidence, "MoonReference")};
    Json step{{"name", name}, {"sun", frame.sun.statistics}, {"moon", frame.moon.statistics},
        {"sunReference", frame.sunReference.statistics}, {"moonReference", frame.moonReference.statistics}};
    for (const auto& [label, cached, reference] : {
        std::tuple{"Sun", &frame.sun, &frame.sunReference}, std::tuple{"Moon", &frame.moon, &frame.moonReference}}) {
        size_t mismatches = 0;
        for (size_t index = 0; index < kPixels; ++index) { mismatches += cached->shadow[index] != reference->shadow[index]; }
        step[std::string(label) + "Mismatches"] = mismatches;
        saveVisibility(directory / (name + "-" + label + "-Cache.png"), cached->shadow);
        saveVisibility(directory / (name + "-" + label + "-Redraw.png"), reference->shadow);
        saveImage(directory / (name + "-" + label + "-Levels.png"), cached->debug);
        require(cached->rasterDepth == reference->rasterDepth, name + ": cache/reference receiver depth differs");
        require(cached->statistics[27] == 1 && reference->statistics[26] == 1,
            name + ": cache/reference settings did not reach the GPU");
        require(cached->statistics[9] == 0 && (allowWorkOverflow || cached->statistics[5] == 0),
            name + ": invalid cache receiver or unexpected work overflow");
    }
    report["frames"].push_back(step);
    std::ofstream(directory / "VirtualShadowMapCache.json") << report.dump(2) << '\n';
    require(frame.sun.shadow == frame.sunReference.shadow && frame.moon.shadow == frame.moonReference.shadow,
        name + ": cached visibility differs from a forced redraw");
    require(frame.sun.parameters.control[2] == kCelestialLightSourceTag &&
        frame.moon.parameters.control[2] == (kCelestialLightSourceTag | 1u), name + ": Sun/Moon shadow identity mismatch");
    return frame;
}

void requireCacheReused(const Snapshot& image, const std::string& name)
{
    require(image.statistics[16] == 0 && image.statistics[17] > 0 && image.statistics[21] == 0,
        name + ": a stable complete cache redrew depth");
}

size_t visibilityDifference(const Snapshot& first, const Snapshot& second)
{
    size_t differences = 0;
    for (size_t index = 0; index < kPixels; ++index) { differences += first.shadow[index] != second.shadow[index]; }
    return differences;
}

void configureCachePreview(RenderGraphPreviewRenderer& preview, scene::SceneDocument& scene,
    RenderView& view, RHITestContext& context, bench::ValidationRecorder& validation,
    const environment::WorldEnvironment& world)
{
    preview.bindRuntimeScene(&scene);
    require(view.setRenderResolution(kWidth, kHeight), "Set cache render extent");
    require(view.setCamera({.eye = {0,4,6}, .center = {0,0,0}, .fovDegrees = 50,
        .nearPlane = .1f, .farPlane = 100, .orthoHeight = 7}), "Set cache camera");
    preview.bindRenderView(&view);
    require(preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink()),
        "Initialize cache preview");
    preview.setEnvironment({.enabled = false});
    auto lighting = scene.lighting();
    lighting.autoExposure.enabled = false; lighting.exposureEV100 = 0;
    require(preview.setLighting(lighting), "Set cache lighting");
    require(preview.setWorldEnvironment(world), "Set cache Sun/Moon");
}

Json verifyCacheConsumers(RenderGraphPreviewRenderer& preview, const std::filesystem::path& source,
    const std::filesystem::path& cache, const std::filesystem::path& directory)
{
    Json report;
    for (bool binned : {false, true}) {
        // A graph has one presentation output. Compile each consumer in its
        // own graph so both use the ordinary FinalBlit presentation contract.
        activeEvidence = std::make_shared<EvidenceState>();
        auto graph = makeCacheGraph(source, cache);
        const std::string consumer = binned ? "Binned" : "Flat";
        graph.addNode("VisibilityBufferDeferredPass", consumer, {{"lightingMode", "realtime"}, {"debugView", "final"},
            {"materialBinning", binned}, {"shadow.enabled", false}, {"accumulate", false}});
        graph.addEdge("Raster.visibility", consumer + ".visibility");
        graph.addEdge("Raster.depth", consumer + ".depth");
        graph.addEdge("Raster.rasterInfo", consumer + ".rasterInfo");
        graph.addEdge("Shadows.shadow", consumer + ".shadow");
        graph.addEdge("Shadows.parameters", consumer + ".shadowParameters");
        graph.addEdge("Moon.shadow", consumer + ".moonShadow");
        graph.addEdge("Moon.parameters", consumer + ".moonShadowParameters");
        graph.addNode("FinalBlitPass", consumer + "Output");
        graph.addEdge(consumer + ".color", consumer + "Output.source");
        graph.markOutput(consumer + "Output.color");
        std::array<std::vector<uint32_t>, 3> images;
        for (uint32_t mode = 0; mode < 3; ++mode) {
            require(graph.setNodeRuntimeProperty(graph.findNode("Shadows")->id, "enabled", mode != 1), "Set Sun consumer control");
            require(graph.setNodeRuntimeProperty(graph.findNode("Moon")->id, "enabled", mode != 2), "Set Moon consumer control");
            for (uint32_t frame = 0; frame < 3; ++frame) { renderAndRequire(preview, graph, consumer + "Output.color"); }
            images[mode] = preview.pixels();
            saveImage(directory / ("Consumer-" + consumer + "-" + std::to_string(mode) + ".png"), images[mode]);
        }
        std::array<size_t, 2> changed{};
        for (uint32_t mode = 1; mode < 3; ++mode) {
            for (size_t pixel = 0; pixel < kPixels; ++pixel) { changed[mode - 1] += images[mode][pixel] != images[0][pixel]; }
            require(changed[mode - 1] > 100, consumer + ": independently disabling a celestial shadow did not affect lighting");
        }
        report[consumer] = {{"sunChangedPixels", changed[0]}, {"moonChangedPixels", changed[1]}};
    }
    return report;
}

// Bind every recording to a real frame. In particular, accepting an unrelated
// prefix must not make a cancelled VSM tail eligible for cache publication.
struct CacheCommands {
    RenderFrameContext frame;
    QueueSubmissionTracker tracker;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    std::unique_ptr<CommandBuffer> prefix;
    uint64_t index = 0;

    explicit CacheCommands(Device& device)
    {
        auto& queue = *device.getQueue(QueueType::Graphics);
        require(tracker.initialize(device, queue), "Cache submission tracker");
        require(device.createCommandPool(queue).transform([&](auto value) { pool = std::move(value); }), "Cache command pool");
        require(pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }), "Cache commands");
        require(pool->createCommandBuffer().transform([&](auto value) { prefix = std::move(value); }), "Cache prefix commands");
    }
    ~CacheCommands()
    {
        frame.cancel();
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        if (pool) { (void)pool->reset(); }
        (void)frame.reset();
    }
    void begin()
    {
        require(frame.begin(index++), "Begin cache frame");
        require(pool->reset(), "Reset cache pool");
        require(commands->begin(frame.submissionContext()), "Begin cache recording");
    }
    void finish(uint32_t cancelMode = 0)
    {
        require(commands->end(), "End cache recording");
        if (cancelMode == 2) {
            require(prefix->begin(frame.submissionContext()), "Begin unrelated prefix");
            require(prefix->end(), "End unrelated prefix");
            CommandBuffer* accepted[]{prefix.get()};
            require(tracker.submitSegment({.commandBuffers = accepted}, frame), "Submit unrelated prefix");
        }
        if (cancelMode != 0) {
            frame.cancel();
            if (cancelMode == 2) {
                require(frame.completion().isSubmitted(), "Partial cancellation lacks accepted prefix");
                require(frame.wait(), "Wait accepted prefix");
            }
            return;
        }
        CommandBuffer* buffers[]{commands.get()};
        require(tracker.submit({.commandBuffers = buffers}, frame), "Submit cache recording");
        require(frame.wait(), "Wait cache recording");
    }
};

Json verifyCacheCancellation(Device& device, scene::SceneDocument& scene, const Snapshot& receiver,
    const std::filesystem::path& directory, const std::filesystem::path& receiverOnlySource)
{
    std::unique_ptr<Texture> depth;
    std::unique_ptr<TextureView> depthView;
    require(device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
        .format = Format::D32Sfloat, .width = kWidth, .height = kHeight})
        .transform([&](auto value) { depth = std::move(value); }), "Actual raster depth upload texture");
    require(device.createTextureView(*depth, {.format = Format::D32Sfloat})
        .transform([&](auto value) { depthView = std::move(value); }), "Actual raster depth upload view");
    auto upload = device.createBuffer({.size = kPixels * sizeof(float), .usage = BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::HostUpload});
    require(upload, "Actual raster depth upload buffer");
    auto* mapped = (*upload)->map();
    require(mapped != nullptr, "Map actual raster depth upload");
    std::memcpy(mapped, receiver.rasterDepth.data(), kPixels * sizeof(float));
    (*upload)->flush(); (*upload)->unmap();
    constexpr uint64_t statisticsOffset = (kPixels + 3) & ~uint64_t(3);
    auto readback = device.createBuffer({.size = statisticsOffset + 32 * sizeof(uint32_t),
        .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback});
    require(readback, "Cancellation result readback");
    VirtualShadowMaps cached, reference;
    CacheCommands gpu(device);
    gpu.begin();
    const TextureBarrierDesc uploadBegin{.texture = depth.get(), .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
    require(gpu.commands->synchronize({.textures = {&uploadBegin, 1}}), "Actual depth upload barrier");
    require((*upload)->slice().and_then([&](const auto& slice) {
        return gpu.commands->copyBufferToTexture({.texture = depth.get(), .buffer = slice, .width = kWidth, .height = kHeight});
    }), "Upload actual raster receiver depth");
    const TextureBarrierDesc uploadEnd{.texture = depth.get(), .oldLayout = TextureLayout::TransferDestination,
        .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
    require(gpu.commands->synchronize({.textures = {&uploadEnd, 1}}), "Uploaded receiver depth ready");
    gpu.finish();

    GPUCelestialLight light;
    light.direction[0] = std::sqrt(.5f); light.direction[1] = -std::sqrt(.5f);
    light.flags = kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow;
    light.irradiance[0] = light.irradiance[1] = light.irradiance[2] = 3;
    VirtualShadowMapSettings settings;
    settings.physicalPageCount = 512;
    settings.cacheEnabled = true;
    std::string log;
    auto record = [&](VirtualShadowMaps& renderer, bool force, uint32_t cancelMode) {
        gpu.begin();
        auto view = receiver.parameters.view;
        view.frame[0] = uint32_t(gpu.index);
        auto casters = prepareVirtualShadowMapCasters(device, *gpu.commands, scene, nullptr, log);
        require(casters, "Cancellation caster preparation: " + log);
        settings.forceRedraw = force;
        auto result = renderer.record(device, *gpu.commands, *depthView, view, light, *casters,
            scene.materials(), {}, settings, log);
        require(result, "Cancellation shadow record: " + log);
        const TextureBarrierDesc shadowRead{.texture = result->texture, .oldLayout = TextureLayout::ShaderRead,
            .newLayout = TextureLayout::TransferSource, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
        const BufferBarrierDesc statsRead{.buffer = result->statistics,
            .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite},
            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
        require(gpu.commands->synchronize({.textures = {&shadowRead, 1}, .buffers = {&statsRead, 1}}), "Cancellation readback barrier");
        require((*readback)->slice({0, kPixels}).and_then([&](const auto& slice) {
            return gpu.commands->copyTextureToBuffer({.texture = result->texture, .buffer = slice, .width = kWidth, .height = kHeight});
        }), "Cancellation shadow readback");
        auto source = result->statistics->slice({0, 32 * sizeof(uint32_t)});
        auto destination = (*readback)->slice({statisticsOffset, 32 * sizeof(uint32_t)});
        require(source, "Cancellation statistics source"); require(destination, "Cancellation statistics destination");
        require(gpu.commands->copyBuffer(*source, *destination), "Cancellation statistics readback");
        const TextureBarrierDesc shadowRestore{.texture = result->texture, .oldLayout = TextureLayout::TransferSource,
            .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
            .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}};
        const BufferBarrierDesc statsRestore{.buffer = result->statistics,
            .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
            .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
        require(gpu.commands->synchronize({.textures = {&shadowRestore, 1}, .buffers = {&statsRestore, 1}}), "Restore cache result states");
        gpu.finish(cancelMode);
        Snapshot image;
        if (cancelMode == 0) {
            (*readback)->invalidate();
            const auto* bytes = static_cast<const uint8_t*>((*readback)->map());
            require(bytes != nullptr, "Map cancellation readback");
            image.shadow.assign(bytes, bytes + kPixels);
            std::memcpy(image.statistics.data(), bytes + statisticsOffset, sizeof(image.statistics));
            (*readback)->unmap();
        }
        return image;
    };
    const auto original = scene.nodes()[1].localMatrix;
    Json report;
    (void)record(cached, false, 0);
    auto stable = record(cached, false, 0);
    requireCacheReused(stable, "Low-level stable cache");
    (void)record(reference, true, 0);
    for (uint32_t cancelMode : {1u, 2u}) {
        float4x4 translation;
        translation.SetupByTranslation(float3(float(cancelMode) * 1.25f, 0, 0));
        require(scene.setNodeLocalMatrix(1, translation * original), "Move cancellation caster");
        (void)record(cached, false, cancelMode);
        const auto retry = record(cached, false, 0);
        const auto redraw = record(reference, true, 0);
        const std::string name = cancelMode == 1 ? "CancelledRecording" : "AcceptedPrefixCancelledTail";
        saveVisibility(directory / (name + "-Retry.png"), retry.shadow);
        saveVisibility(directory / (name + "-Redraw.png"), redraw.shadow);
        report[name] = {{"retryStatistics", retry.statistics}, {"redrawStatistics", redraw.statistics},
            {"equal", retry.shadow == redraw.shadow}, {"acceptedPrefix", cancelMode == 2}};
        std::ofstream(directory / "Cancellation.json") << report.dump(2) << '\n';
        require(retry.shadow == redraw.shadow, name + ": retry reused unsubmitted depth");
        require(retry.statistics[16] > 0 && retry.statistics[28] > 0, name + ": cancelled dirty pages were not discarded and redrawn");
        requireCacheReused(record(cached, false, 0), name + " stable retry");
    }
    require(scene.setNodeLocalMatrix(1, original), "Restore cancellation caster");
    const auto originalSource = scene.sourcePath();
    require(scene.load(receiverOnlySource), "Replace scene after cancellation");
    const auto replaced = record(cached, false, 0);
    const auto replacedReference = record(reference, true, 0);
    saveVisibility(directory / "SceneReplacement.png", replaced.shadow);
    report["SceneReplacement"] = {{"statistics", replaced.statistics}, {"equal", replaced.shadow == replacedReference.shadow}};
    require(replaced.shadow == replacedReference.shadow &&
        std::all_of(replaced.shadow.begin(), replaced.shadow.end(), [](auto value) { return value == 255; }),
        "Scene replacement retained removed caster depth");
    require(replaced.statistics[22] > 0, "Scene replacement did not reset cache identity");
    require(scene.load(originalSource), "Restore original scene after cancellation");
    std::ofstream(directory / "Cancellation.json") << report.dump(2) << '\n';
    return report;
}

class VirtualShadowMapCacheTest final : public RHITest {
public:
    VirtualShadowMapCacheTest() { type = RHITestType::Rendering; name = "virtual_shadow_map_cache"; }

    RHITestResult run(RHITestContext& context) override
    {
        bench::ValidationRecorder validation;
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapCache";
        Json report{{"scope", "P2 real raster receivers; independent cached and force-redraw Sun/Moon; exact hard-shadow equality"},
            {"width", kWidth}, {"height", kHeight}, {"frames", Json::array()}, {"validationEnabled", context.enableValidation},
            {"jitter", false}, {"passed", false}};
        try {
            registerFixturePasses();
            std::filesystem::create_directories(directory);
            const auto source = writeCacheFixture(directory / "Caster");
            const auto receiverSource = writeFixture(directory / "ReceiverOnly", false);
            const auto cache = source.parent_path() / "Scene.meshstream.bin";
            std::string log;
            require(scene::buildMeshletStreamAssetOffline({.sourcePath = source, .outputPath = cache,
                .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log), "Cook cache fixture: " + log);
            for (bool metadataOnly : {false, true}) {
                const std::string mode = metadataOnly ? "Stream" : "Resident";
                const auto modeDirectory = directory / mode;
                std::filesystem::create_directories(modeDirectory);
                scene::SceneDocument scene;
                require(metadataOnly ? scene.loadStreamMetadata(source) : scene.load(source), "Load cache fixture");
                require(scene.hasStreamGeometry() == metadataOnly, "Cache fixture selected the wrong caster source");
                environment::WorldEnvironment world;
                world.sun.enabled = true; world.sun.direction = float3(1, -1, 0);
                world.sun.illuminance = 3; world.sun.angularRadius = 0; world.sun.color = float3(1, .3f, .15f);
                world.moon.enabled = true; world.moon.direction = float3(1, -1, -.3f);
                world.moon.illuminance = 3; world.moon.angularRadius = 0; world.moon.color = float3(.15f, .3f, 1);
                RenderView view;
                RenderGraphPreviewRenderer preview;
                configureCachePreview(preview, scene, view, context, validation, world);
                activeEvidence = std::make_shared<EvidenceState>();
                auto graph = makeCacheGraph(source, cache);
                auto compare = [&](const std::string& name, bool allowWorkOverflow = false) {
                    validation.phase(mode + "-" + name);
                    return compareCacheFrame(preview, graph, modeDirectory, mode + "-" + name, report, allowWorkOverflow);
                };
                auto initial = compare("Initial");
                auto* device = preview.subsystemHost()->device();
                require(!device->capabilities().rayTracingPipeline && !device->capabilities().rayQuery,
                    "VSM cache fixture unexpectedly enabled ray tracing");
                report[mode + "Validation"] = bench::name(bench::activeValidation(*device));
                require(metadataOnly ? initial.sun.statistics[10] >= 128 : initial.sun.statistics[10] == 4,
                    "P2 caster source capacity is incorrect");
                require(initial.sun.statistics[16] > 0 && initial.moon.statistics[16] > 0, "Initial cache did not render fine pages");
                for (uint32_t frame = 0; frame < 3; ++frame) {
                    auto stable = compare("Stable" + std::to_string(frame));
                    requireCacheReused(stable.sun, "Stable Sun"); requireCacheReused(stable.moon, "Stable Moon");
                }
                world.sun.illuminance = 4; world.sun.color = float3(.5f, 1, .2f);
                world.moon.illuminance = 5; world.moon.color = float3(.2f, .5f, 1);
                require(preview.setWorldEnvironment(world), "Change light radiometry");
                auto radiometry = compare("LightColorIntensity");
                requireCacheReused(radiometry.sun, "Sun radiometry"); requireCacheReused(radiometry.moon, "Moon radiometry");
                require(radiometry.sun.shadow == initial.sun.shadow && radiometry.moon.shadow == initial.moon.shadow,
                    "Light color/intensity changed geometric visibility");

                auto material = scene.materials()[1];
                material.baseColorFactor.x = .25f;
                require(scene.setMaterialProperties(1, material), "Change caster RGB only");
                auto rgb = compare("MaterialRGB");
                requireCacheReused(rgb.sun, "Caster RGB Sun"); requireCacheReused(rgb.moon, "Caster RGB Moon");

                const auto originalCamera = view.camera();
                auto camera = originalCamera;
                camera.eye[0] += .55f; camera.center[0] += .55f;
                camera.eye[2] += .55f; camera.center[2] += .55f;
                require(view.setCamera(camera), "Shift cache camera across page boundaries");
                auto shifted = compare("CameraPageShift");
                require(shifted.sun.statistics[19] > 0 && shifted.sun.statistics[17] > 0,
                    "Camera shift did not retain and remap Sun pages");
                require(shifted.moon.statistics[19] > 0 && shifted.moon.statistics[17] > 0,
                    "Camera shift did not retain and remap Moon pages");
                // A horizontal translation over the same plane can preserve
                // every depth value, so verify the actual raster camera ABI.
                require(std::abs(shifted.sun.camera.eye[0] - camera.eye[0]) < 1e-6f &&
                    std::abs(shifted.sun.camera.eye[2] - camera.eye[2]) < 1e-6f,
                    "External camera edit did not reach actual raster metadata");
                require(view.setCamera(originalCamera), "Return camera to cached page region");
                auto returned = compare("CameraPageReturn");
                require(returned.sun.statistics[19] > 0 && returned.sun.statistics[17] > 0 &&
                    returned.moon.statistics[19] > 0 && returned.moon.statistics[17] > 0,
                    "Returning camera did not reuse remapped page contents");
                require(returned.sun.shadow == initial.sun.shadow && returned.moon.shadow == initial.moon.shadow,
                    "Camera return did not reproduce its original visibility");
                require(view.setCamera(camera), "Shift camera again for invalidation controls");
                shifted = compare("CameraPageShiftAgain");
                const auto originalMatrix = scene.nodes()[1].localMatrix;
                float4x4 translation;
                translation.SetupByTranslation(float3(2.0f, 0, 0));
                require(scene.setNodeLocalMatrix(1, translation * originalMatrix), "Move cached offscreen caster");
                auto moved = compare("CasterMoved");
                require(visibilityDifference(moved.sun, shifted.sun) > 100, "Moved caster did not change visible shadows");
                require(moved.sun.statistics[18] > 0 && moved.moon.statistics[18] > 0,
                    "Moved caster did not invalidate old and new bounds");
                require(moved.sun.statistics[17] > 0 && moved.moon.statistics[17] > 0,
                    "Local caster movement unnecessarily discarded all cached fine pages");

                require(scene.setObjectVisible(scene.objectForNode(1).entity(), false), "Hide cached caster");
                auto hidden = compare("CasterHidden");
                require(std::all_of(hidden.sun.shadow.begin(), hidden.sun.shadow.end(), [](auto value) { return value == 255; }) &&
                    std::all_of(hidden.moon.shadow.begin(), hidden.moon.shadow.end(), [](auto value) { return value == 255; }),
                    "Hidden caster retained stale Sun/Moon shadows");
                require(hidden.sun.statistics[18] > 0 && hidden.moon.statistics[18] > 0, "Hidden caster old bounds were not invalidated");
                require(scene.setObjectVisible(scene.objectForNode(1).entity(), true), "Show cached caster");
                (void)compare("CasterShown");
                material.baseColorFactor.w = 0;
                require(scene.setMaterialProperties(1, material), "Reject cached MASK caster");
                auto rejected = compare("CoverageRejected");
                require(rejected.sun.shadow == hidden.sun.shadow && rejected.moon.shadow == hidden.moon.shadow,
                    "Coverage change retained opaque cached depth");
                require(rejected.sun.statistics[18] > 0 && rejected.moon.statistics[18] > 0, "Coverage change did not invalidate caster bounds");
                material.baseColorFactor.w = 1;
                require(scene.setMaterialProperties(1, material), "Restore cached MASK caster");
                (void)compare("CoverageRestored");
                world.sun.direction = float3(.9f, -1, 0);
                require(preview.setWorldEnvironment(world), "Change Sun direction only");
                auto sunDirection = compare("SunDirection");
                require(sunDirection.sun.statistics[16] > 0 && sunDirection.sun.statistics[21] == 1, "Sun direction did not redraw depth");
                requireCacheReused(sunDirection.moon, "Sun direction must preserve Moon cache");
                world.moon.direction = float3(.9f, -1, -.25f);
                require(preview.setWorldEnvironment(world), "Change Moon direction only");
                auto moonDirection = compare("MoonDirection");
                require(moonDirection.moon.statistics[16] > 0 && moonDirection.moon.statistics[21] == 1, "Moon direction did not redraw depth");
                requireCacheReused(moonDirection.sun, "Moon direction must preserve Sun cache");

                for (const auto& name : {"Shadows", "SunReference"}) {
                    require(graph.setNodeRuntimeProperty(graph.findNode(name)->id, "maxWorkItems", 1), "Constrain dirty-page work");
                }
                translation.SetupByTranslation(float3(2.75f, 0, 0));
                require(scene.setNodeLocalMatrix(1, translation * originalMatrix), "Dirty pages for work overflow");
                auto workOverflow = compare("DirtyWorkOverflow", true);
                require(workOverflow.sun.statistics[5] > 0 && workOverflow.sun.statistics[7] == workOverflow.sun.statistics[6],
                    "Dirty work overflow did not use complete coarse coverage");
                auto overflowRetry = compare("DirtyWorkOverflowNextFrame", true);
                require(overflowRetry.sun.statistics[5] > 0 && overflowRetry.sun.statistics[16] > 0 &&
                    overflowRetry.sun.statistics[21] == 0 && overflowRetry.sun.statistics[7] == overflowRetry.sun.statistics[6],
                    "Incomplete fine pages became valid or stable coarse depth was redrawn after work overflow");
                for (const auto& name : {"Shadows", "SunReference"}) {
                    require(graph.setNodeRuntimeProperty(graph.findNode(name)->id, "maxWorkItems", 65535), "Restore dirty-page work budget");
                }
                auto workRecovered = compare("DirtyWorkRecovered");
                require(workRecovered.sun.statistics[16] > 0 && workRecovered.sun.statistics[7] < workRecovered.sun.statistics[6],
                    "Fine-page cache did not recover after restoring work budget");
                auto workStable = compare("DirtyWorkRecoveredStable");
                requireCacheReused(workStable.sun, "Recovered Sun cache");
                requireCacheReused(workStable.moon, "Recovered Moon cache");

                for (const auto& name : {"Moon", "MoonReference"}) {
                    require(graph.setNodeRuntimeProperty(graph.findNode(name)->id, "physicalPageCount", 1), "Constrain Moon pool");
                }
                auto overflow = compare("IndependentMoonPoolOverflow");
                require(overflow.moon.statistics[2] > 0 && overflow.moon.statistics[7] > 0 && overflow.moon.statistics[12] == 1,
                    "Moon pool overflow did not use complete coarse fallback");
                requireCacheReused(overflow.sun, "Moon pool change must preserve Sun cache");
                auto overflowStable = compare("IndependentMoonPoolStable");
                require(overflowStable.moon.statistics[21] == 0, "Stable coarse-only Moon cache redrew depth");
                requireCacheReused(overflowStable.sun, "Moon pool overflow must preserve Sun cache");

                for (const auto& name : {"Shadows", "SunReference"}) {
                    require(graph.setNodeRuntimeProperty(graph.findNode(name)->id, "physicalPageCount", 64), "Constrain Sun eviction pool");
                }
                auto fullPool = compare("SunEvictionPoolFilled");
                require(fullPool.sun.statistics[2] > 0 && fullPool.sun.statistics[24] == 63,
                    "Eviction fixture did not fill all fine physical slots");
                // Rotate around a fixed eye: all clipmap origins stay fixed,
                // so new allocations in this full pool must evict retained
                // unrequested owners, rather than use pages lost by scrolling.
                auto rotatedCamera = camera;
                rotatedCamera.center[0] += 2;
                require(view.setCamera(rotatedCamera), "Rotate receivers into unrequested cached pages");
                auto evicted = compare("SunEvictionDemandChanged");
                require(evicted.sun.statistics[19] == 0 && evicted.sun.statistics[20] > 0 && evicted.sun.statistics[17] > 0,
                    "Changing demand in a full fixed-origin pool did not exercise eviction and reuse");
                require(view.setCamera(camera), "Return receivers after page eviction");
                auto evictionReturn = compare("SunEvictionReturn");
                require(evictionReturn.sun.statistics[17] > 0 && evictionReturn.sun.statistics[20] > 0,
                    "Returning demand did not reuse retained pages and recover evicted pages");
                require(evictionReturn.sun.shadow == fullPool.sun.shadow, "Returning after eviction changed the original visibility");

                // Restore the authored view and caster before exercising the
                // actual direct-light consumers and low-level cancellation.
                require(view.setCamera(originalCamera), "Restore consumer camera");
                require(scene.setNodeLocalMatrix(1, originalMatrix), "Restore consumer caster");
                world.sun.direction = float3(1, -1, 0); world.moon.direction = float3(1, -1, -.3f);
                require(preview.setWorldEnvironment(world), "Restore consumer directions");
                auto restored = compare("Restored");
                if (!metadataOnly) {
                    report["cancellation"] = verifyCacheCancellation(*device, scene, restored.sun, modeDirectory, receiverSource);
                    // The low-level replacement test reloads this document;
                    // republish its lifetime before restoring authored lights.
                    preview.bindRuntimeScene(&scene);
                    require(preview.setWorldEnvironment(world), "Restore Sun/Moon after scene replacement");
                    report["consumers"] = verifyCacheConsumers(preview, source, cache, modeDirectory);
                }
                requireValidPreview(validation.snapshot());
            }
            activeEvidence.reset();
            report["previewValidation"] = validation.snapshot();
            requireValidPreview(report["previewValidation"]);
            report["passed"] = true;
            std::ofstream(directory / "VirtualShadowMapCache.json") << report.dump(2) << '\n';
            std::ofstream(directory / "PreviewValidation.json") << report["previewValidation"].dump(2) << '\n';
            return RHITestResult::pass("VSM cache/redraw image equality, resident/StreamAsset invalidation, Sun/Moon isolation, cancellation and flat/binned consumers; PNG/JSON exported");
        } catch (const std::exception& error) {
            activeEvidence.reset();
            report["error"] = error.what();
            report["previewValidation"] = validation.snapshot();
            std::ofstream(directory / "VirtualShadowMapCache.json") << report.dump(2) << '\n';
            std::ofstream(directory / "PreviewValidation.json") << report["previewValidation"].dump(2) << '\n';
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapCacheTest);

} // namespace
} // namespace metallic::tests
