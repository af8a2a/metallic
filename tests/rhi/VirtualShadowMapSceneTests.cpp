#include "RHITest.h"
#include "Runtime/Debug/DebugHash.h"
#include "harness/ValidationRecorder.h"
#include "harness/VulkanDiagnostics.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/ScreenSpaceShadows.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLights.h"
#include "Runtime/Render/Shadow/VirtualShadowMapProfiling.h"
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
    // These page-boundary and cache-eviction fixtures require the finest cut.
    // Receiver-matched resolution is covered by virtual_shadow_map_filter_demand.
    graph.addNode("VirtualShadowMapPass", "Shadows", {{"pageSize", kPageSize}, {"virtualPagesPerAxis", kPagesPerAxis},
        {"clipmapCount", kClipmapCount}, {"firstClipmapExtent", kFirstExtent}, {"physicalPageCount", test.pool},
        {"maxWorkItems", test.maxWorkItems}, {"shadowBias", .002}, {"shadowDebug", true}, {"receiverFootprint", false}});
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

constexpr uint32_t kLocalCapacity = 8, kLocalLayers = kLocalCapacity * 6;
constexpr uint64_t kLocalShadowStride = (kPixels + 3) & ~uint64_t(3);
constexpr uint64_t kLocalDebugOffset = kLocalShadowStride * kLocalLayers;
constexpr uint64_t kLocalDepthOffset = kLocalDebugOffset + kPixels * 4 * kLocalLayers;
constexpr uint64_t kLocalStatsOffset = kLocalDepthOffset + kPixels * 4;
constexpr uint64_t kLocalLightsOffset = kLocalStatsOffset + kLocalLayers * kVirtualShadowMapFaceStatisticsSize;
constexpr uint64_t kLocalTableSize = sizeof(VirtualShadowMapLightsHeader) + kLocalCapacity * sizeof(VirtualShadowMapLightEntry);
constexpr uint64_t kLocalReadbackSize = kLocalLightsOffset + kLocalTableSize;

class VSMLocalEvidencePass final : public UnsafePass {
public:
    explicit VSMLocalEvidencePass(std::shared_ptr<EvidenceState> state) : state_(std::move(state)) {}
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        auto& shadow = reflection.addTextureInput("shadows").transferRead();
        shadow.format = Format::R8Unorm; shadow.layerCount = kLocalLayers;
        auto& debug = reflection.addTextureInput("debug").transferRead();
        debug.format = Format::RGBA8Unorm; debug.layerCount = kLocalLayers;
        reflection.addTextureInput("depth").transferRead().format = Format::D32Sfloat;
        reflection.addBufferInput("statistics").buffer(kLocalLayers * kVirtualShadowMapFaceStatisticsSize).transferRead();
        reflection.addBufferInput("lights").buffer(kLocalTableSize).transferRead();
        reflection.addBufferInput("rasterInfo").buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo))
            .shaderRead().memoryLocation = MemoryLocation::HostUpload;
        reflection.addBufferOutput("data").buffer(kLocalReadbackSize).transferWrite().hostReadback();
        reflection.addTextureOutput("color").transferWrite().format = Format::RGBA8Unorm;
        return reflection;
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto& commands = context.commandBuffer();
        auto* buffer = context.outputBuffer("data").buffer();
        state_->namedReadbacks[context.properties().at("evidenceName").get<std::string>()] = buffer;
        for (uint32_t layer = 0; layer < kLocalLayers; ++layer) {
            for (bool debug : {false, true}) {
                const uint64_t offset = debug ? kLocalDebugOffset + layer * kPixels * 4 : layer * kLocalShadowStride;
                auto slice = buffer->slice({offset, kPixels * (debug ? 4 : 1)});
                if (!slice) { return std::unexpected(slice.error()); }
                if (auto result = commands.copyTextureToBuffer({.texture = context.inputTexture(debug ? "debug" : "shadows").texture(),
                    .buffer = *slice, .width = kWidth, .height = kHeight, .baseLayer = layer}); !result) { return result; }
            }
        }
        auto depth = buffer->slice({kLocalDepthOffset, kPixels * 4});
        if (!depth) { return std::unexpected(depth.error()); }
        if (auto result = commands.copyTextureToBuffer({.texture = context.inputTexture("depth").texture(),
            .buffer = *depth, .width = kWidth, .height = kHeight}); !result) { return result; }
        for (const auto& [name, offset, size] : {std::tuple{"statistics", kLocalStatsOffset, uint64_t(kLocalLayers * kVirtualShadowMapFaceStatisticsSize)},
            std::tuple{"lights", kLocalLightsOffset, kLocalTableSize}}) {
            auto source = context.inputBuffer(name).buffer()->slice({0, size});
            auto destination = buffer->slice({offset, size});
            if (!source) { return std::unexpected(source.error()); }
            if (!destination) { return std::unexpected(destination.error()); }
            if (auto result = commands.copyBuffer(*source, *destination); !result) { return result; }
        }
        auto* info = context.inputBuffer("rasterInfo").buffer();
        const void* mapped = info->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(&state_->camera, mapped, sizeof(state_->camera));
        info->unmap();
        return commands.copyTexture({.source = context.inputTexture("debug").texture(),
            .destination = context.outputTexture("color").texture(), .width = kWidth, .height = kHeight,
            .sourceBaseLayer = 12});
    }
private:
    std::shared_ptr<EvidenceState> state_;
};

struct LocalSnapshot {
    VirtualShadowMapLightsHeader header{};
    std::array<VirtualShadowMapLightEntry, kLocalCapacity> entries{};
    std::array<std::array<uint32_t, 32>, kLocalLayers> statistics{};
    std::array<std::vector<uint8_t>, kLocalLayers> shadows;
    std::array<std::vector<uint32_t>, kLocalLayers> debug;
    std::vector<float> depth;
    VisibilityBufferFrameInfo camera;
};

LocalSnapshot localSnapshot(const std::string& name)
{
    LocalSnapshot result;
    auto* buffer = activeEvidence->namedReadbacks.at(name);
    buffer->invalidate();
    const auto* mapped = static_cast<const std::byte*>(buffer->map());
    require(mapped != nullptr, "Map local-light array evidence");
    for (uint32_t layer = 0; layer < kLocalLayers; ++layer) {
        result.shadows[layer].resize(kPixels); result.debug[layer].resize(kPixels);
        std::memcpy(result.shadows[layer].data(), mapped + layer * kLocalShadowStride, kPixels);
        std::memcpy(result.debug[layer].data(), mapped + kLocalDebugOffset + layer * kPixels * 4, kPixels * 4);
    }
    result.depth.resize(kPixels);
    std::memcpy(result.depth.data(), mapped + kLocalDepthOffset, kPixels * 4);
    std::memcpy(result.statistics.data(), mapped + kLocalStatsOffset, sizeof(result.statistics));
    std::memcpy(&result.header, mapped + kLocalLightsOffset, sizeof(result.header));
    std::memcpy(result.entries.data(), mapped + kLocalLightsOffset + sizeof(result.header), sizeof(result.entries));
    buffer->unmap();
    result.camera = activeEvidence->camera;
    require(result.header.version == kVirtualShadowMapLightsVersion && result.header.capacity == kLocalCapacity &&
        result.header.count <= kLocalCapacity, "Local light publication ABI mismatch");
    require(result.camera.width == kWidth && result.camera.height == kHeight, "Local light raster resolution mismatch");
    return result;
}

struct AuthoredTriangle { Point a, b, c; };
struct LocalFixture {
    std::vector<AuthoredTriangle> triangles;
    std::filesystem::path source, cache;
};

void addFixtureQuad(LocalFixture& fixture, Point center, Point u, Point v)
{
    const Point a = add(add(center, scale(u, -1)), scale(v, -1));
    const Point b = add(add(center, u), scale(v, -1));
    const Point c = add(add(center, u), v), d = add(add(center, scale(u, -1)), v);
    fixture.triangles.push_back({a,b,c}); fixture.triangles.push_back({a,c,d});
}

LocalFixture writeLocalFixture(const std::filesystem::path& directory, bool room, bool nearCrossing = false)
{
    std::filesystem::create_directories(directory);
    LocalFixture fixture;
    if (!room) {
        addFixtureQuad(fixture, {0,0,0}, {8,0,0}, {0,0,-8});
        if (nearCrossing) {
            fixture.triangles.push_back({{-2,3.2,-1},{2,2,-1},{0,2,2}});
            addFixtureQuad(fixture, {0,2.9,0}, {.08,0,0}, {0,0,.08});
        }
        else { addFixtureQuad(fixture, {0,1.5,0}, {.8,0,0}, {0,0,.8}); }
    } else {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            for (double sign : {-1., 1.}) {
                Point center{}, u{}, v{}; center[axis] = sign * 6;
                u[(axis + 1) % 3] = 6; v[(axis + 2) % 3] = 6;
                addFixtureQuad(fixture, center, u, v);
                center[axis] = sign * 2;
                addFixtureQuad(fixture, center, scale(u, .1), scale(v, .1));
            }
        }
        // Each cube seam has a blocker whose shadow crosses the face boundary.
        // Its normal points away from the light, so both faces see the same quad.
        for (uint32_t first = 0; first < 3; ++first) {
            for (uint32_t second = first + 1; second < 3; ++second) {
                for (double a : {-1., 1.}) { for (double b : {-1., 1.}) {
                    Point center{}, u{}, v{}; center[first] = a * 2; center[second] = b * 2;
                    u[first] = .45 * a; u[second] = -.45 * b; v[3 - first - second] = .65;
                    addFixtureQuad(fixture, center, u, v);
                }}
            }
        }
    }
    std::vector<float> positions, normals, uv;
    std::vector<uint32_t> indices;
    Point minimum{1e30,1e30,1e30}, maximum{-1e30,-1e30,-1e30};
    for (const auto& triangle : fixture.triangles) {
        const Point normal = normalized(cross(add(triangle.b, scale(triangle.a, -1)), add(triangle.c, scale(triangle.a, -1))));
        for (Point vertex : {triangle.a, triangle.b, triangle.c}) {
            for (uint32_t axis = 0; axis < 3; ++axis) {
                positions.push_back(float(vertex[axis])); normals.push_back(float(normal[axis]));
                minimum[axis] = std::min(minimum[axis], vertex[axis]); maximum[axis] = std::max(maximum[axis], vertex[axis]);
            }
            uv.insert(uv.end(), {0,0}); indices.push_back(uint32_t(indices.size()));
        }
    }
    const uint64_t positionBytes = positions.size() * 4, normalBytes = normals.size() * 4, uvBytes = uv.size() * 4;
    {
        std::ofstream binary(directory / "Scene.bin", std::ios::binary);
        for (const auto& values : {positions, normals, uv}) { binary.write(reinterpret_cast<const char*>(values.data()), values.size() * 4); }
        binary.write(reinterpret_cast<const char*>(indices.data()), indices.size() * 4);
        require(bool(binary), "Write local-light authored triangles");
    }
    Json gltf;
    gltf["asset"] = {{"version", "2.0"}}; gltf["scene"] = 0;
    gltf["scenes"] = Json::array({Json{{"nodes", {0}}}});
    gltf["nodes"] = Json::array({Json{{"name", "AuthoredGeometry"}, {"mesh", 0}}});
    gltf["buffers"] = Json::array({Json{{"uri", "Scene.bin"}, {"byteLength", positionBytes + normalBytes + uvBytes + indices.size() * 4}}});
    gltf["bufferViews"] = Json::array();
    uint64_t offset = 0;
    for (uint64_t bytes : {positionBytes, normalBytes, uvBytes, uint64_t(indices.size() * 4)}) {
        gltf["bufferViews"].push_back({{"buffer", 0}, {"byteOffset", offset}, {"byteLength", bytes}}); offset += bytes;
    }
    gltf["accessors"] = Json::array({
        Json{{"bufferView",0},{"componentType",5126},{"count",indices.size()},{"type","VEC3"},{"min",minimum},{"max",maximum}},
        Json{{"bufferView",1},{"componentType",5126},{"count",indices.size()},{"type","VEC3"}},
        Json{{"bufferView",2},{"componentType",5126},{"count",indices.size()},{"type","VEC2"}},
        Json{{"bufferView",3},{"componentType",5125},{"count",indices.size()},{"type","SCALAR"}}});
    Json primitive{{"attributes",{{"POSITION",0},{"NORMAL",1},{"TEXCOORD_0",2}}},{"indices",3},{"material",0}};
    gltf["meshes"] = Json::array({Json{{"primitives", Json::array({primitive})}}});
    Json material{{"name","LocalShadowOpaque"},{"doubleSided",true}};
    material["pbrMetallicRoughness"] = {{"baseColorFactor",{.7,.7,.7,1}},{"metallicFactor",0},{"roughnessFactor",1}};
    gltf["materials"] = Json::array({material});
    fixture.source = directory / "Scene.gltf"; fixture.cache = directory / "Scene.meshstream.bin";
    std::ofstream(fixture.source) << gltf.dump(2) << '\n';
    std::string log;
    const auto cooked = scene::buildMeshletStreamAssetOffline({.sourcePath = fixture.source, .outputPath = fixture.cache,
        .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log);
    require(cooked, "Cook local-light fixture: " + log);
    return fixture;
}

RenderGraph makeLocalGraph(const LocalFixture& fixture, float nearPlane = .001f)
{
    RenderGraph graph;
    graph.setName("VirtualShadowMapLocalLights");
    graph.setViewProperties({{"temporalJitter",false},
        {"renderResolution",{{"width",kWidth},{"height",kHeight},{"adaptive",false}}}});
    graph.addNode("VisibilityBufferPass", "Raster", {{"path",fixture.source.generic_string()}, {"streamAssetPath",fixture.cache.generic_string()},
        {"autoBuildStreamAsset",false},{"maxResidentPages",64},{"maxLockedFallbackPages",64},
        {"maxActiveGroups",256},{"maxGpuPageRequests",256},{"maxTraversalWorkers",64},{"maxTraversalWorkItems",512},
        {"autoLod",false},{"lodLevel",0},{"instanceHzbCull",false},{"meshletHzbCull",false},
        {"meshletNormalConeCull",false},{"hybridRaster",false},{"enableClas",false},{"enableClusterRtx",false}});
    for (const char* name : {"Cached", "Redraw"}) {
        // Keep the fixed-resolution geometry/seam oracle and its edge margins.
        // Filter-demand and quality tests exercise receiver-matched resolution.
        graph.addNode("VirtualShadowMapLightsPass", name, {{"lightCapacity",kLocalCapacity},{"physicalPageBudget",512},
            {"pageSize",kPageSize},{"virtualPagesPerAxis",kPagesPerAxis},{"clipmapCount",kClipmapCount},
            {"maxWorkItems",65535},{"shadowBias",.002},{"nearPlane",nearPlane},{"cacheEnabled",true},
            {"forceRedraw",std::string_view(name) == "Redraw"},{"receiverFootprint",false}});
        graph.addEdge("Raster.depth",std::string(name) + ".depth");
        graph.addEdge("Raster.rasterInfo",std::string(name) + ".rasterInfo");
        const std::string evidence = std::string(name) + "Evidence";
        graph.addNode("VSMLocalEvidencePass", evidence, {{"evidenceName",name}});
        for (const char* field : {"shadows","debug","lights","statistics"}) {
            graph.addEdge(std::string(name) + "." + field, evidence + "." + field);
        }
        graph.addEdge("Raster.depth",evidence + ".depth"); graph.addEdge("Raster.rasterInfo",evidence + ".rasterInfo");
        graph.markOutput(evidence + ".color");
    }
    return graph;
}

double triangleDistance(Point origin, Point direction, const AuthoredTriangle& triangle)
{
    const Point e1 = add(triangle.b, scale(triangle.a, -1)), e2 = add(triangle.c, scale(triangle.a, -1));
    const Point p = cross(direction, e2);
    const double determinant = dot(e1, p);
    if (std::abs(determinant) < 1e-12) { return 1e30; }
    const Point t = add(origin, scale(triangle.a, -1));
    const double u = dot(t, p) / determinant;
    const Point q = cross(t, e1);
    const double v = dot(direction, q) / determinant, distance = dot(e2, q) / determinant;
    // A ray exactly on the room's shared edge can round u+v to
    // 1.0000000000000002 for both adjacent walls. Keep that closed boundary;
    // this tolerance is in double barycentrics, not a screen-pixel exclusion.
    constexpr double kBarycentricRoundoff = 1e-12;
    return u >= -kBarycentricRoundoff && v >= -kBarycentricRoundoff &&
        u + v <= 1 + kBarycentricRoundoff && distance > 1e-7 ? distance : 1e30;
}

struct LocalHit { Point position{}, normal{}; double depth = 0; bool valid = false; };

LocalHit localCameraHit(const VisibilityBufferFrameInfo& camera, uint32_t x, uint32_t y, const LocalFixture& fixture)
{
    const Point eye = point(camera.eye), forward = normalized(add(point(camera.center),scale(eye,-1)));
    const Point right = normalized(cross(forward,point(camera.upProjection))), up = cross(right,forward);
    const bool orthographic = camera.upProjection[3] > .5f;
    const double halfHeight = orthographic ? camera.clipOrtho[2] * .5 : std::tan(camera.viewport[3] * .5);
    const Point lateral = add(scale(right,(2 * (x + .5 + camera.jitter[0]) / kWidth - 1) * camera.viewport[0] * halfHeight),
        scale(up,(1 - 2 * (y + .5 + camera.jitter[1]) / kHeight) * halfHeight));
    const Point origin = orthographic ? add(eye,lateral) : eye, direction = orthographic ? forward : add(forward,lateral);
    LocalHit result; double nearest = 1e30;
    for (const auto& triangle : fixture.triangles) {
        const double distance = triangleDistance(origin,direction,triangle);
        const Point position = add(origin,scale(direction,distance));
        const double viewZ = dot(add(position,scale(eye,-1)),forward);
        if (distance >= nearest || viewZ < camera.clipOrtho[0] || viewZ > camera.clipOrtho[1]) { continue; }
        nearest = distance; result.valid = true; result.position = position;
        result.normal = normalized(cross(add(triangle.b,scale(triangle.a,-1)),add(triangle.c,scale(triangle.a,-1))));
        const double n = camera.clipOrtho[0], f = camera.clipOrtho[1];
        result.depth = orthographic ? (f-viewZ)/(f-n) : n*(f-viewZ)/(viewZ*(f-n));
    }
    return result;
}

uint32_t pointFace(Point direction)
{
    uint32_t axis = 0;
    if (std::abs(direction[1]) > std::abs(direction[axis])) { axis = 1; }
    if (std::abs(direction[2]) > std::abs(direction[axis])) { axis = 2; }
    return axis * 2 + (direction[axis] < 0 ? 1 : 0);
}

bool localOccluded(Point receiver, const scene::PunctualLight& light, const LocalFixture& fixture, double nearPlane)
{
    const Point origin{light.position.x,light.position.y,light.position.z};
    const Point ray = add(receiver,scale(origin,-1));
    const Point lightDirection{light.direction.x,light.direction.y,light.direction.z};
    for (const auto& triangle : fixture.triangles) {
        const double t = triangleDistance(origin,ray,triangle);
        if (t >= 1 - .004 / std::sqrt(dot(ray,ray))) { continue; }
        const Point fromLight = scale(ray,t);
        const double projected = light.properties.type == "point" ?
            std::max({std::abs(fromLight[0]),std::abs(fromLight[1]),std::abs(fromLight[2])}) : dot(fromLight,normalized(lightDirection));
        if (projected >= nearPlane) { return true; }
    }
    return false;
}

uint64_t localEntryId(const VirtualShadowMapLightEntry& entry)
{
    return uint64_t(entry.stableId[0]) | (uint64_t(entry.stableId[1]) << 32);
}

const VirtualShadowMapLightEntry& localEntry(const LocalSnapshot& image, uint64_t id)
{
    for (uint32_t index = 0; index < image.header.count; ++index) {
        if (localEntryId(image.entries[index]) == id) { return image.entries[index]; }
    }
    throw std::runtime_error("Published light table lost stable identity " + std::to_string(id));
}

Json verifyLocalOracle(const LocalSnapshot& image, const LocalFixture& fixture, const scene::LightingSettings& lighting,
    const std::filesystem::path& directory, const std::string& name, double nearPlane)
{
    Json report = Json::array();
    std::vector<LocalHit> hits(kPixels);
    size_t depthMismatches = 0, visible = 0;
    Json depthErrors=Json::array();
    for (uint32_t y = 0; y < kHeight; ++y) { for (uint32_t x = 0; x < kWidth; ++x) {
        const size_t pixel = size_t(y)*kWidth+x;
        hits[pixel] = localCameraHit(image.camera,x,y,fixture);
        visible += hits[pixel].valid;
        if (std::abs(image.depth[pixel] - hits[pixel].depth) > 2e-5) {
            ++depthMismatches;
            depthErrors.push_back({{"pixel",{x,y}},{"actual",image.depth[pixel]},{"expected",hits[pixel].depth},
                {"position",hits[pixel].position}});
        }
    }}
    if (depthMismatches) {
        std::ofstream(directory/(name+"-DepthErrors.json")) << Json{{"errors",depthErrors},
            {"eye",{image.camera.eye[0],image.camera.eye[1],image.camera.eye[2]}},
            {"center",{image.camera.center[0],image.camera.center[1],image.camera.center[2]}},
            {"up",{image.camera.upProjection[0],image.camera.upProjection[1],image.camera.upProjection[2]}},
            {"viewport",{image.camera.viewport[0],image.camera.viewport[1],image.camera.viewport[2],image.camera.viewport[3]}}}.dump(2) << '\n';
    }
    require(visible > 1000 && depthMismatches == 0, name + ": actual raster depth differs from authored triangle rays: " + std::to_string(depthMismatches));
    for (const auto& light : lighting.lights) {
        const auto& entry = localEntry(image,light.stableId);
        require(entry.kind == (light.properties.type == "point" ? 2u : 1u), name + ": wrong published light kind");
        require(entry.faceCount == (entry.kind == 2 ? 6u : 1u), name + ": incomplete point-face publication");
        if (entry.status != kVirtualShadowMapLightAdmitted) { continue; }
        size_t checked=0,mismatches=0,shadowed=0,edgeExcluded=0,seamChecked=0,seamShadowed=0;
        size_t coarseChecked=0,coarseEdgeExcluded=0;
        std::set<uint32_t> faces;
        std::vector<uint8_t> actual(kPixels,255), reference(kPixels,255);
        std::vector<uint32_t> error(kPixels,rgba(0,0,0));
        const Point origin{light.position.x,light.position.y,light.position.z};
        const Point direction = normalized(Point{light.direction.x,light.direction.y,light.direction.z});
        for (size_t pixel=0; pixel<kPixels; ++pixel) {
            const auto& hit = hits[pixel]; if (!hit.valid) { continue; }
            const Point fromLight = add(hit.position,scale(origin,-1));
            const double distance = std::sqrt(dot(fromLight,fromLight));
            if (light.properties.range > 0 && distance >= light.properties.range) { continue; }
            if (entry.kind == 1 && dot(normalized(fromLight),direction) <= std::cos(light.properties.outerConeAngle) + .002) { continue; }
            const uint32_t face = entry.kind == 2 ? pointFace(fromLight) : 0;
            const double projected = entry.kind == 2 ? std::abs(fromLight[face/2]) : dot(fromLight,direction);
            if (projected <= nearPlane + .01) { continue; }
            actual[pixel] = image.shadows[entry.baseLayer+face][pixel];
            const bool blocked = localOccluded(hit.position,light,fixture,nearPlane);
            reference[pixel] = blocked ? 0 : 255;
            // Raster sample coverage differs within one texel of a geometric
            // shadow edge. Exclude only those edges using world-space rays;
            // cube-face seams themselves are never excluded.
            const Point tangent = normalized(cross(hit.normal,std::abs(hit.normal[1]) < .9 ? Point{0,1,0} : Point{1,0,0}));
            const Point bitangent = cross(hit.normal,tangent);
            const uint32_t debug=image.debug[entry.baseLayer+face][pixel];
            const bool coarse=(debug&255)<30 && ((debug>>8)&255)<30 && ((debug>>16)&255)>200;
            const double margin = coarse ? .24 : .035;
            bool edge = false;
            for (Point offset : {scale(tangent,margin),scale(tangent,-margin),scale(bitangent,margin),scale(bitangent,-margin)}) {
                edge |= localOccluded(add(hit.position,offset),light,fixture,nearPlane) != blocked;
            }
            if (edge) { ++edgeExcluded; coarseEdgeExcluded+=coarse; error[pixel]=rgba(50,50,50); continue; }
            ++checked; shadowed += blocked; faces.insert(face);
            coarseChecked+=coarse;
            std::array<double,3> magnitudes{std::abs(fromLight[0]),std::abs(fromLight[1]),std::abs(fromLight[2])};
            std::sort(magnitudes.begin(),magnitudes.end());
            const bool seam = entry.kind == 2 && magnitudes[2] - magnitudes[1] < .06;
            seamChecked += seam; seamShadowed += seam && blocked;
            const bool mismatch = actual[pixel] != reference[pixel]; mismatches += mismatch;
            error[pixel] = mismatch ? rgba(255,0,255) : seam ? rgba(0,160,255) : rgba(0,60,0);
        }
        const std::string label = name + "-" + light.name;
        saveVisibility(directory/(label+"-GPU.png"),actual); saveVisibility(directory/(label+"-CPU.png"),reference);
        saveImage(directory/(label+"-Error.png"),error);
        if (mismatches != 0) {
            for (uint32_t face=0; face<entry.faceCount; ++face) {
                saveImage(directory/(label+"-Face"+std::to_string(face)+"-Debug.png"),image.debug[entry.baseLayer+face]);
            }
        }
        report.push_back({{"light",light.name},{"stableId",light.stableId},{"checked",checked},{"shadowed",shadowed},
            {"mismatches",mismatches},{"shadowEdgeExcluded",edgeExcluded},{"coarseChecked",coarseChecked},
            {"coarseShadowEdgeExcluded",coarseEdgeExcluded},{"fineEdgeMarginMetres",.035},{"coarseEdgeMarginMetres",.24},
            {"seamChecked",seamChecked},{"seamShadowed",seamShadowed},{"faces",faces}});
        std::ofstream(directory/(name+"-Oracle.json")) << report.dump(2) << '\n';
        require(checked > 100 && mismatches == 0, label + ": light-ray oracle mismatch count " + std::to_string(mismatches));
    }
    return report;
}

LocalSnapshot compareLocalFrame(RenderGraphPreviewRenderer& preview, RenderGraph& graph, const LocalFixture& fixture,
    const scene::LightingSettings& lighting, const std::filesystem::path& directory, const std::string& name, Json& report,
    double nearPlane = .001)
{
    renderAndRequire(preview,graph,"CachedEvidence.color");
    auto cached = localSnapshot("Cached"), redraw = localSnapshot("Redraw");
    size_t mismatches = 0;
    for (uint32_t layer=0; layer<kLocalLayers; ++layer) {
        for (size_t pixel=0; pixel<kPixels; ++pixel) { mismatches += cached.shadows[layer][pixel] != redraw.shadows[layer][pixel]; }
    }
    Json frame{{"name",name},{"cacheRedrawMismatches",mismatches},{"admitted",cached.header.admittedCount},
        {"dropped",cached.header.budgetDroppedCount},{"statistics",cached.statistics}};
    report["frames"].push_back(frame);
    std::ofstream(directory/"VirtualShadowMapLocalLights.json") << report.dump(2) << '\n';
    require(mismatches == 0 && cached.depth == redraw.depth, name + ": cached local-light arrays differ from forced redraw");
    for (uint32_t index=0; index<cached.header.count; ++index) {
        const auto& entry = cached.entries[index];
        require(entry.baseLayer == index*6, name+": table layer does not match its source entry");
        if (entry.status == kVirtualShadowMapLightAdmitted) {
            for (uint32_t face=0; face<entry.faceCount; ++face) {
                require(cached.statistics[entry.baseLayer+face][27] == 1 && redraw.statistics[entry.baseLayer+face][26] == 1,
                    name+": cache/force-redraw control did not reach every admitted face");
            }
        }
    }
    report["frames"].back()["oracle"] = verifyLocalOracle(cached,fixture,lighting,directory,name,nearPlane);
    return cached;
}

void requireLocalReused(const LocalSnapshot& image, uint64_t identity, const std::string& name)
{
    const auto& entry = localEntry(image,identity);
    for (uint32_t face=0; face<entry.faceCount; ++face) {
        const auto& statistics = image.statistics[entry.baseLayer+face];
        require(statistics[16] == 0 && statistics[21] == 0, name+": stable light face redrew depth");
    }
}

scene::PunctualLight localLight(std::string name, bool spot, Point position)
{
    scene::PunctualLight light; light.name = std::move(name);
    light.position = float3(float(position[0]),float(position[1]),float(position[2]));
    light.direction = float3(0,-1,0); light.properties.type = spot ? "spot" : "point";
    light.properties.range = 20; light.properties.intensity = 80;
    light.properties.outerConeAngle = 1.0; light.properties.innerConeAngle = .5;
    return light;
}

std::vector<float> localHDR(const RenderGraphPreviewRenderer& preview)
{
    const auto& bytes=preview.readbackBytes();
    std::vector<float> values;
    if (preview.readbackFormat()==Format::RGBA32Sfloat) {
        values.resize(bytes.size()/4); std::memcpy(values.data(),bytes.data(),bytes.size());
    } else if (preview.readbackFormat()==Format::RGBA16Sfloat) {
        for (size_t offset=0; offset<bytes.size(); offset+=2) {
            uint16_t bits; std::memcpy(&bits,bytes.data()+offset,2);
            const uint32_t exponent=(bits>>10)&31, mantissa=bits&1023;
            require(exponent!=31,"Local consumer produced nonfinite HDR");
            const float value=exponent==0 ? std::ldexp(float(mantissa),-24) : std::ldexp(float(1024+mantissa),int(exponent)-25);
            values.push_back(bits&0x8000 ? -value : value);
        }
    }
    require(values.size()==kPixels*4,"Local consumer requires full linear floating-point readback");
    require(std::all_of(values.begin(),values.end(),[](float v) { return std::isfinite(v); }),"Local consumer nonfinite HDR");
    return values;
}

Json verifyLocalConsumers(RenderGraphPreviewRenderer& preview, scene::SceneDocument& scene,
    const LocalFixture& fixture, const std::filesystem::path& directory)
{
    const auto authored=scene.lighting();
    require(authored.lights.size()==2,"Local consumer needs exactly one point and one spot");
    Json report;
    for (bool binned : {false,true}) {
        const std::string label=binned ? "Binned" : "Flat";
        activeEvidence=std::make_shared<EvidenceState>();
        auto graph=makeLocalGraph(fixture);
        graph.addNode("VisibilityBufferDeferredPass","Deferred",{{"lightingMode","realtime"},{"debugView","final"},
            {"materialBinning",binned},{"shadow.enabled",false},{"accumulate",false},{"outputLinear",true}});
        for (const char* field : {"visibility","depth","rasterInfo"}) { graph.addEdge(std::string("Raster.")+field,std::string("Deferred.")+field); }
        graph.addEdge("Cached.shadows","Deferred.virtualShadows");
        graph.addEdge("Cached.lights","Deferred.virtualShadowLights");
        graph.markOutput("Deferred.color");
        preview.setRawReadbackEnabled(true);
        std::array<std::vector<float>,7> images;
        LocalSnapshot masks;
        const char* modes[]{"AllCached","AllRedraw","AllUnshadowed","FirstOnly","SecondOnly","FirstUnshadowed","SecondUnshadowed"};
        for (uint32_t mode=0; mode<std::size(modes); ++mode) {
            auto lighting=authored;
            // Keep both lights enabled and geometrically identical, so the
            // page allocation stays fixed while isolating direct contributions.
            if (mode==3 || mode==5) { lighting.lights[1].properties.intensity=1e-10; }
            if (mode==4 || mode==6) { lighting.lights[0].properties.intensity=1e-10; }
            (void)scene.setLighting(lighting); (void)preview.setLighting(scene.lighting());
            const bool unshadowed=mode==2 || mode>=5;
            (void)graph.setNodeRuntimeProperty(graph.findNode("Cached")->id,"physicalPageBudget",unshadowed ? 0 : 512);
            (void)graph.setNodeRuntimeProperty(graph.findNode("Cached")->id,"forceRedraw",mode==1);
            renderAndRequire(preview,graph,"Deferred.color");
            images[mode]=localHDR(preview);
            if (mode==0) { masks=localSnapshot("Cached"); }
            saveImage(directory/("Consumer-"+label+"-"+modes[mode]+".png"),preview.pixels());
        }
        require(images[0]==images[1],label+": actual Deferred cache/force HDR differs");
        size_t sumErrors=0,allChanged=0;
        double maximumSumError=0;
        const double tolerance=preview.readbackFormat()==Format::RGBA16Sfloat ? .002 : 2e-5;
        for (size_t pixel=0; pixel<kPixels; ++pixel) {
            bool changed=false;
            for (uint32_t channel=0; channel<3; ++channel) {
                const size_t index=pixel*4+channel;
                const double difference=std::abs(double(images[0][index])-images[3][index]-images[4][index]);
                maximumSumError=std::max(maximumSumError,difference);
                sumErrors+=difference>tolerance*std::max(1.,std::abs(double(images[0][index])));
                changed|=std::abs(images[0][index]-images[2][index])>tolerance;
            }
            allChanged+=changed;
        }
        Json controls=Json::array();
        for (uint32_t lightIndex=0; lightIndex<2; ++lightIndex) {
            const auto& light=authored.lights[lightIndex]; const auto& entry=localEntry(masks,light.stableId);
            size_t changed=0,lit=0,leaks=0;
            for (uint32_t y=0; y<kHeight; ++y) { for (uint32_t x=0; x<kWidth; ++x) {
                const size_t pixel=size_t(y)*kWidth+x; const auto hit=localCameraHit(masks.camera,x,y,fixture);
                if (!hit.valid) { continue; }
                const Point delta=add(hit.position,Point{-light.position.x,-light.position.y,-light.position.z});
                const uint32_t face=entry.kind==2 ? pointFace(delta) : 0;
                bool different=false;
                for (uint32_t channel=0; channel<3; ++channel) {
                    const size_t index=pixel*4+channel;
                    different|=std::abs(images[3+lightIndex][index]-images[5+lightIndex][index])>
                        tolerance*std::max(1.,std::abs(double(images[5+lightIndex][index])));
                }
                if (masks.shadows[entry.baseLayer+face][pixel]==255) { ++lit; leaks+=different; }
                else { changed+=different; }
            }}
            controls.push_back({{"light",light.name},{"changedShadowedPixels",changed},{"litPixels",lit},{"changedLitPixels",leaks}});
            require(changed>50 && lit>1000 && leaks==0,label+": per-light direct visibility was absent or applied to a different light");
        }
        report[label]={{"sumErrors",sumErrors},{"maximumSumError",maximumSumError},{"allShadowChangedPixels",allChanged},{"controls",controls}};
        require(sumErrors==0 && allChanged>100,label+": overlapping local-light contributions are not additive or shadows were not consumed");

        auto reordered = authored;
        std::reverse(reordered.lights.begin(), reordered.lights.end());
        require(scene.setLighting(reordered), "Reorder consumer source slots");
        (void)preview.setLighting(scene.lighting());
        require(graph.setNodeRuntimeProperty(graph.findNode("Cached")->id, "physicalPageBudget", 512), "Restore reordered consumer budget");
        require(graph.setNodeRuntimeProperty(graph.findNode("Cached")->id, "forceRedraw", false), "Restore reordered cache");
        renderAndRequire(preview, graph, "Deferred.color");
        const auto reorderedHDR = localHDR(preview);
        const auto reorderedMasks = localSnapshot("Cached");
        for (const auto& light : authored.lights) {
            const auto& before = localEntry(masks, light.stableId);
            const auto& after = localEntry(reorderedMasks, light.stableId);
            require(before.source != after.source && before.faceCount == after.faceCount, label + ": reordered source identity was not exercised");
            for (uint32_t face = 0; face < before.faceCount; ++face) {
                require(masks.shadows[before.baseLayer + face] == reorderedMasks.shadows[after.baseLayer + face],
                    label + ": source reordering changed a stable light's shadow mask");
            }
        }
        // Reordering direct lights changes floating-point accumulation order.
        // Keep the established HDR format tolerance; stable-ID visibility above
        // remains exact, as does same-frame projection equivalence elsewhere.
        for (size_t index = 0; index < reorderedHDR.size(); ++index) {
            require(std::abs(reorderedHDR[index] - images[0][index]) <= tolerance * std::max(1., std::abs(double(images[0][index]))),
                label + ": source reordering changed Deferred lighting");
        }
        report[label]["sourceReorder"] = "Stable-ID shadow masks exact; HDR uses the existing format-dependent accumulation tolerance";

        // Keep all four source slots enabled while isolating each celestial's
        // ordinary, measurable illumination. Tiny other contributions preserve
        // table admission; they are not counted as lighting coverage.
        for (auto& light : reordered.lights) { light.properties.intensity = 1e-10; }
        require(scene.setLighting(reordered), "Isolate celestial consumer contributions");
        (void)preview.setLighting(scene.lighting());
        environment::WorldEnvironment world;
        world.sun.enabled = world.moon.enabled = true;
        world.sun.direction = float3(1, -1, 0); world.moon.direction = float3(-.6f, -1, .2f);
        for (uint32_t celestial = 0; celestial < 2; ++celestial) {
            world.sun.illuminance = celestial == 0 ? 3 : 1e-10f;
            world.moon.illuminance = celestial == 1 ? 3 : 1e-10f;
            (void)preview.setWorldEnvironment(world);
            std::array<std::vector<float>, 2> hdr;
            LocalSnapshot celestialMasks;
            for (uint32_t unshadowed = 0; unshadowed < 2; ++unshadowed) {
                require(graph.setNodeRuntimeProperty(graph.findNode("Cached")->id, "physicalPageBudget", unshadowed ? 0 : 512),
                    "Set celestial consumer budget");
                renderAndRequire(preview, graph, "Deferred.color");
                hdr[unshadowed] = localHDR(preview);
                if (!unshadowed) { celestialMasks = localSnapshot("Cached"); }
            }
            require(celestialMasks.header.count == 4 && celestialMasks.header.admittedCount == 4, label + ": mixed consumer omitted a source");
            const auto entry = std::find_if(celestialMasks.entries.begin(), celestialMasks.entries.begin() + celestialMasks.header.count,
                [&](const auto& value) { return value.source == (kCelestialLightSourceTag | celestial); });
            require(entry != celestialMasks.entries.begin() + celestialMasks.header.count && entry->kind == 0,
                label + ": celestial source lookup omitted Sun or Moon");
            size_t lit = 0, changed = 0, leaks = 0;
            for (size_t pixel = 0; pixel < kPixels; ++pixel) {
                if (celestialMasks.depth[pixel] == 0) { continue; }
                bool different = false;
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    const size_t index = pixel * 4 + channel;
                    different |= std::abs(hdr[0][index] - hdr[1][index]) > tolerance * std::max(1., std::abs(double(hdr[1][index])));
                }
                if (celestialMasks.shadows[entry->baseLayer][pixel] == 255) { ++lit; leaks += different; }
                else { changed += different; }
            }
            report[label]["celestials"].push_back({{"source",celestial},{"changedShadowedPixels",changed},{"litPixels",lit},{"changedLitPixels",leaks}});
            std::ofstream(directory/"Consumers.json") << report.dump(2) << '\n';
            require(changed > 50 && lit > 1000 && leaks == 0, label + ": celestial visibility was absent or applied to another source");
        }
        world.sun.enabled = world.moon.enabled = false;
        (void)preview.setWorldEnvironment(world);
        (void)scene.setLighting(authored); (void)preview.setLighting(scene.lighting());
        std::ofstream(directory/"Consumers.json") << report.dump(2) << '\n';
    }
    (void)scene.setLighting(authored); (void)preview.setLighting(scene.lighting());
    preview.setRawReadbackEnabled(false);
    return report;
}

class VirtualShadowMapLocalLightsTest final : public RHITest {
public:
    VirtualShadowMapLocalLightsTest() { type=RHITestType::Rendering; name="virtual_shadow_map_local_lights"; }
    RHITestResult run(RHITestContext& context) override
    {
        bench::ValidationRecorder validation;
        const auto directory = std::filesystem::absolute(context.outputDirectory)/"VirtualShadowMapLocalLights";
        Json report{{"scope","P3 real raster receivers and independent authored triangle light rays"},{"passed",false},{"frames",Json::array()}};
        try {
            static const bool registered = registerRenderGraphPassType("VSMLocalEvidencePass","Test-only local VSM array readback",
                [] { return std::make_unique<VSMLocalEvidencePass>(activeEvidence); });
            require(registered,"Register local VSM evidence pass");
            std::filesystem::create_directories(directory);
            const auto floor = writeLocalFixture(directory/"Floor",false);
            const auto room = writeLocalFixture(directory/"Room",true);
            const auto nearFixture = writeLocalFixture(directory/"NearCrossing",false,true);
            for (bool metadataOnly : {false,true}) {
                const std::string mode = metadataOnly ? "Stream" : "Resident";
                for (const LocalFixture* fixture : {&floor,&room,&nearFixture}) {
                    const bool isRoom = fixture == &room, isNear = fixture == &nearFixture;
                    const std::string prefix = mode + (isRoom ? "-Room" : isNear ? "-Near" : "-Floor");
                    const auto output = directory/prefix; std::filesystem::create_directories(output);
                    scene::SceneDocument scene;
                    require(metadataOnly ? scene.loadStreamMetadata(fixture->source) : scene.load(fixture->source),"Load local VSM fixture");
                    environment::WorldEnvironment world; world.sun.enabled=false; world.moon.enabled=false;
                    RenderView view; RenderGraphPreviewRenderer preview;
                    configureCachePreview(preview,scene,view,context,validation,world);
                    scene::LightingSettings lighting; lighting.autoExposure.enabled=false;
                    if (isRoom) { lighting.lights.push_back(localLight("Point",false,{0,0,0})); }
                    else if (isNear) { lighting.lights.push_back(localLight("NearSpot",true,{0,3,0})); }
                    else {
                        lighting.lights.push_back(localLight("Point",false,{-1,4,0}));
                        lighting.lights.push_back(localLight("Spot",true,{1,4,0}));
                    }
                    auto publishLighting = [&] {
                        require(scene.setLighting(lighting),"Publish local light edits");
                        lighting=scene.lighting();
                        (void)preview.setLighting(lighting);
                    };
                    publishLighting();
                    activeEvidence=std::make_shared<EvidenceState>();
                    const float nearPlane=isNear ? .25f : .001f;
                    auto graph=makeLocalGraph(*fixture,nearPlane);
                    LocalFixture oracleFixture=*fixture;
                    auto compare=[&](const std::string& step) {
                        validation.phase(prefix+"-"+step);
                        return compareLocalFrame(preview,graph,oracleFixture,scene.lighting(),output,prefix+"-"+step,report,nearPlane);
                    };
                    if (isRoom) {
                        uint32_t viewIndex=0; size_t seamChecked=0,seamShadowed=0;
                        auto renderView=[&](Point eye,Point target,Point up) {
                            ViewCamera camera;
                            for (uint32_t axis=0; axis<3; ++axis) { camera.eye[axis]=float(eye[axis]); camera.center[axis]=float(target[axis]); camera.up[axis]=float(up[axis]); }
                            camera.fovDegrees=65; camera.nearPlane=.05f; camera.farPlane=30;
                            (void)view.setCamera(camera);
                            const auto image=compare("View"+std::to_string(viewIndex++));
                            for (const auto& oracle : report["frames"].back()["oracle"]) {
                                seamChecked+=oracle["seamChecked"].get<size_t>(); seamShadowed+=oracle["seamShadowed"].get<size_t>();
                            }
                            if (viewIndex==1) {
                                const auto stable=compare("Stable");
                                requireLocalReused(stable,lighting.lights[0].stableId,"Room stable point");
                            }
                        };
                        for (uint32_t axis=0; axis<3; ++axis) { for (double sign : {-1.,1.}) {
                            Point eye{},target{},up{}; eye[(axis+1)%3]=3; target[axis]=6*sign; up[(axis+2)%3]=1;
                            renderView(eye,target,up);
                        }}
                        for (uint32_t first=0; first<3; ++first) { for (uint32_t second=first+1; second<3; ++second) {
                            for (double a : {-1.,1.}) { for (double b : {-1.,1.}) {
                                Point eye{},target{},up{}; const uint32_t third=3-first-second;
                                eye[third]=3; target[first]=6*a; target[second]=6*b; up[third]=1;
                                renderView(eye,target,up);
                            }}
                        }}
                        require(seamChecked>100 && seamShadowed>100,"Room fixture did not check shadowed cube seams");
                        report[prefix+"Seams"]={{"checked",seamChecked},{"shadowed",seamShadowed}};
                    } else {
                        auto initial=compare("Initial");
                        auto stable=compare("Stable");
                        for (const auto& light : lighting.lights) { requireLocalReused(stable,light.stableId,"Stable local light"); }
                        if (isNear) {
                            require(initial.statistics[localEntry(initial,lighting.lights[0].stableId).baseLayer][29]>0,
                                "Explicit local near-plane receiver limit was not counted");
                            report[prefix+"NearPlaneContract"]="Receivers inside the configured near plane are counted; casters inside it are explicitly truncated";
                            for (double angle : {1e-8,1.5707963267948966}) {
                                lighting.lights[0].properties.innerConeAngle=0;
                                lighting.lights[0].properties.outerConeAngle=angle;
                                publishLighting();
                                const auto rejected=preview.render(graph,kWidth,kHeight,"CachedEvidence.color");
                                report[prefix+"UnsupportedCones"].push_back({{"outerHalfAngle",angle},{"result",resultToString(rejected)},
                                    {"log",preview.lastLog()}});
                                require(!rejected && rejected.error()==Error::Unsupported,
                                    "Unrepresentable or 180-degree spot projection was silently clamped");
                            }
                        }
                        if (!isNear) {
                            const uint64_t pointId=lighting.lights[0].stableId,spotId=lighting.lights[1].stableId;
                            const auto originalLighting=lighting;
                            auto requireRedrawn=[&](const LocalSnapshot& image,uint64_t id,const std::string& label) {
                                const auto& entry=localEntry(image,id);
                                for (uint32_t face=0; face<entry.faceCount; ++face) {
                                    require(image.statistics[entry.baseLayer+face][21]==1,label+": changed projection retained old coarse depth");
                                }
                            };
                            lighting.lights[0].position.x+=.6f; publishLighting();
                            auto movedPoint=compare("PointMoved");
                            requireRedrawn(movedPoint,pointId,"Moved point"); requireLocalReused(movedPoint,spotId,"Point move preserves spot");
                            lighting=originalLighting; publishLighting();
                            auto restoredPoint=compare("PointRestored");
                            requireRedrawn(restoredPoint,pointId,"Restored point"); requireLocalReused(restoredPoint,spotId,"Point restore preserves spot");
                            lighting.lights[1].direction=float3(.18f,-1,.12f); publishLighting();
                            auto turnedSpot=compare("SpotDirection");
                            requireRedrawn(turnedSpot,spotId,"Turned spot"); requireLocalReused(turnedSpot,pointId,"Spot direction preserves point");
                            lighting.lights[1].properties.outerConeAngle=.85;
                            lighting.lights[1].properties.innerConeAngle=.4; publishLighting();
                            auto coneSpot=compare("SpotCone");
                            requireRedrawn(coneSpot,spotId,"Changed spot cone"); requireLocalReused(coneSpot,pointId,"Spot cone preserves point");
                            lighting.lights[1].properties.range=12; publishLighting();
                            auto rangedSpot=compare("SpotRange");
                            requireRedrawn(rangedSpot,spotId,"Changed spot range"); requireLocalReused(rangedSpot,pointId,"Spot range preserves point");
                            lighting=originalLighting; publishLighting();
                            auto restoredSpot=compare("SpotRestored");
                            requireRedrawn(restoredSpot,spotId,"Restored spot"); requireLocalReused(restoredSpot,pointId,"Spot restore preserves point");

                            // This fixture uses one authored mesh. Move both the
                            // receiver and blocker, and independently transform
                            // the CPU triangles; no GPU-derived geometry enters
                            // the old/new perspective dirty-bounds oracle.
                            const auto originalMatrix=scene.nodes()[0].localMatrix;
                            const Point translation{.35,0,.2};
                            float4x4 transform; transform.SetupByTranslation(float3(.35f,0,.2f));
                            require(scene.setNodeLocalMatrix(0,transform*originalMatrix),"Translate local fixture geometry");
                            for (auto& triangle : oracleFixture.triangles) {
                                triangle.a=add(triangle.a,translation); triangle.b=add(triangle.b,translation); triangle.c=add(triangle.c,translation);
                            }
                            const auto movedGeometry=compare("GeometryTranslated");
                            for (uint64_t id : {pointId,spotId}) {
                                const auto& entry=localEntry(movedGeometry,id); uint32_t invalidated=0;
                                for (uint32_t face=0; face<entry.faceCount; ++face) { invalidated+=movedGeometry.statistics[entry.baseLayer+face][18]; }
                                require(invalidated>0,"Translated local caster old/new bounds did not invalidate cached perspective pages");
                            }
                            for (const char* node : {"Cached","Redraw"}) {
                                require(graph.setNodeRuntimeProperty(graph.findNode(node)->id,"maxWorkItems",1),"Constrain local fine-page work");
                            }
                            require(scene.setNodeLocalMatrix(0,originalMatrix),"Restore geometry while fine-page work is constrained");
                            oracleFixture=*fixture;
                            auto requireOverflow=[&](const LocalSnapshot& image,bool retry) {
                                for (uint64_t id : {pointId,spotId}) {
                                    const auto& entry=localEntry(image,id); uint32_t overflow=0;
                                    for (uint32_t face=0; face<entry.faceCount; ++face) {
                                        const auto& statistics=image.statistics[entry.baseLayer+face]; overflow+=statistics[5];
                                        if (statistics[5]>0) {
                                            const auto& pixels=image.debug[entry.baseLayer+face];
                                            const auto coarse=std::count(pixels.begin(),pixels.end(),rgba(0,0,255));
                                            const auto fine=std::count_if(pixels.begin(),pixels.end(),[](uint32_t p) { return (p & 0x0000ff00u)!=0; });
                                            // stats6 counts all screen receivers before range/face
                                            // selection. Only this face's queried pixels must be coarse.
                                            require(coarse>0 && coarse==statistics[7] && fine==0,
                                                "Local fine work overflow did not query complete coarse coverage");
                                            require(!retry || statistics[21]==0,"Local overflow retry unnecessarily redrew valid coarse depth");
                                        }
                                    }
                                    require(overflow>0,"Local-light work budget fixture did not overflow");
                                }
                            };
                            requireOverflow(compare("DirtyWorkOverflow"),false);
                            requireOverflow(compare("DirtyWorkOverflowRetry"),true);
                            for (const char* node : {"Cached","Redraw"}) {
                                require(graph.setNodeRuntimeProperty(graph.findNode(node)->id,"maxWorkItems",65535),"Restore local fine-page work budget");
                            }
                            auto recovered=compare("DirtyWorkRecovered");
                            for (uint64_t id : {pointId,spotId}) {
                                const auto& entry=localEntry(recovered,id); uint32_t fineReceivers=0;
                                for (uint32_t face=0; face<entry.faceCount; ++face) {
                                    const auto& statistics=recovered.statistics[entry.baseLayer+face];
                                    require(statistics[5]==0,"Local fine work did not recover");
                                    const auto& pixels=recovered.debug[entry.baseLayer+face];
                                    fineReceivers+=uint32_t(std::count_if(pixels.begin(),pixels.end(),[](uint32_t p) { return (p & 0x0000ff00u)!=0; }));
                                }
                                require(fineReceivers>0,"Recovered local light remained entirely at coarse resolution");
                            }
                            const auto recoveredStable=compare("DirtyWorkRecoveredStable");
                            requireLocalReused(recoveredStable,pointId,"Recovered point"); requireLocalReused(recoveredStable,spotId,"Recovered spot");
                            std::swap(lighting.lights[0],lighting.lights[1]);
                            publishLighting();
                            auto reordered=compare("Reordered");
                            requireLocalReused(reordered,pointId,"Reordered point"); requireLocalReused(reordered,spotId,"Reordered spot");
                            auto clone=lighting.lights[0]; clone.name="CopiedSpot"; clone.position.x=-2;
                            lighting.lights.push_back(clone); publishLighting();
                            const uint64_t cloneId=lighting.lights.back().stableId;
                            require(cloneId!=spotId && cloneId!=pointId,"Copied light did not acquire a new runtime identity");
                            auto added=compare("CopiedLightAdded");
                            require(added.statistics[localEntry(added,cloneId).baseLayer][21]==1,"New identity inherited stale coarse cache");
                            auto addedStable=compare("CopiedLightStable");
                            requireLocalReused(addedStable,pointId,"Stable after add point"); requireLocalReused(addedStable,spotId,"Stable after add spot");
                            lighting.lights.pop_back(); publishLighting();
                            (void)compare("CopiedLightRemoved");
                            auto removedStable=compare("RemovedStable");
                            requireLocalReused(removedStable,pointId,"Stable after remove point"); requireLocalReused(removedStable,spotId,"Stable after remove spot");
                            lighting.lights[0].properties.color=float3(.2f,.6f,1); lighting.lights[0].properties.intensity*=2;
                            publishLighting();
                            auto radiometry=compare("Radiometry");
                            requireLocalReused(radiometry,pointId,"Radiometry point"); requireLocalReused(radiometry,spotId,"Radiometry spot");
                            for (uint32_t budget : {7u,6u,0u,512u}) {
                                for (const char* node : {"Cached","Redraw"}) {
                                    require(graph.setNodeRuntimeProperty(graph.findNode(node)->id,"physicalPageBudget",budget),"Change global local-light budget");
                                }
                                const auto constrained=compare("Budget"+std::to_string(budget));
                                require(constrained.header.admittedCount==(budget>=7 ? 2u : budget==6 ? 1u : 0u),"Whole-light coarse admission count mismatch");
                                require(constrained.header.budgetDroppedCount==(budget>=7 ? 0u : budget==6 ? 1u : 2u),"Budget rejection was not explicitly published");
                                if (budget==7) {
                                    require(localEntry(constrained,pointId).pageBudget==6 && localEntry(constrained,spotId).pageBudget==1,
                                        "Point coarse budget was split or an incomplete face set was admitted");
                                }
                            }
                            report[prefix+"Consumers"]=verifyLocalConsumers(preview,scene,*fixture,output);
                        }
                    }
                    requireValidPreview(validation.snapshot());
                }
            }
            activeEvidence.reset(); report["validation"]=validation.snapshot(); requireValidPreview(report["validation"]);
            report["passed"]=true;
            std::ofstream(directory/"VirtualShadowMapLocalLights.json") << report.dump(2) << '\n';
            return RHITestResult::pass("Local spot/point real-depth ray oracle, six faces/seams, stable identities, cache/redraw and whole-light budgets; PNG/JSON exported");
        } catch (const std::exception& error) {
            activeEvidence.reset(); report["error"]=error.what(); report["validation"]=validation.snapshot();
            std::ofstream(directory/"VirtualShadowMapLocalLights.json") << report.dump(2) << '\n';
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapLocalLightsTest);

struct ProfilingReadbackLayout
{
    uint32_t layers, faces;
    uint64_t shadowStride = (kPixels + 3) & ~uint64_t(3);
    uint64_t debug, depth, statistics, workload, buildDiagnostics, size;

    ProfilingReadbackLayout(uint32_t layerCount, uint32_t faceCount, bool collectWork)
        : layers(layerCount), faces(faceCount), debug(shadowStride * layers),
          depth(debug + kPixels * 4 * layers), statistics(depth + kPixels * 4),
          workload((statistics + faces * kVirtualShadowMapFaceStatisticsSize + 7) & ~uint64_t(7)),
          buildDiagnostics(workload + (collectWork ? faces * kVirtualShadowMapWorkloadSize : 0)),
          size(buildDiagnostics + (collectWork ? faces * kVirtualShadowMapBuildDiagnosticSize : 0)) {}
};

class VSMProfilingEvidencePass final : public UnsafePass
{
public:
    explicit VSMProfilingEvidencePass(std::shared_ptr<EvidenceState> state) : state_(std::move(state)) {}

    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        const bool collectWork = properties().value("collectWork", false);
        const ProfilingReadbackLayout layout(properties().value("layers", 1u), properties().value("faces", 1u), collectWork);
        RenderPassReflection reflection;
        auto& shadow = reflection.addTextureInput("shadow").transferRead();
        shadow.format = Format::R8Unorm; shadow.layerCount = layout.layers;
        auto& debug = reflection.addTextureInput("debug").transferRead();
        debug.format = Format::RGBA8Unorm; debug.layerCount = layout.layers;
        reflection.addTextureInput("depth").transferRead().format = Format::D32Sfloat;
        reflection.addBufferInput("statistics").buffer(layout.faces * kVirtualShadowMapFaceStatisticsSize).transferRead();
        if (collectWork) {
            reflection.addBufferInput("workload").buffer(layout.faces * kVirtualShadowMapWorkloadSize).transferRead();
            reflection.addBufferInput("buildDiagnostics").buffer(layout.faces * kVirtualShadowMapBuildDiagnosticSize).transferRead();
        }
        reflection.addBufferOutput("data").buffer(layout.size).transferWrite().hostReadback();
        reflection.addTextureOutput("color").transferWrite().format = Format::RGBA8Unorm;
        return reflection;
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const bool collectWork = context.properties().value("collectWork", false);
        const ProfilingReadbackLayout layout(context.properties().value("layers", 1u), context.properties().value("faces", 1u), collectWork);
        auto& commands = context.commandBuffer();
        auto* buffer = context.outputBuffer("data").buffer();
        state_->namedReadbacks[context.properties().at("evidenceName").get<std::string>()] = buffer;
        for (uint32_t layer = 0; layer < layout.layers; ++layer) {
            for (bool debug : {false, true}) {
                const uint64_t offset = debug ? layout.debug + layer * kPixels * 4 : layer * layout.shadowStride;
                auto slice = buffer->slice({offset, kPixels * (debug ? 4 : 1)});
                if (!slice) { return std::unexpected(slice.error()); }
                if (auto result = commands.copyTextureToBuffer({.texture = context.inputTexture(debug ? "debug" : "shadow").texture(),
                    .buffer = *slice, .width = kWidth, .height = kHeight, .baseLayer = layer}); !result) { return result; }
            }
        }
        auto depth = buffer->slice({layout.depth, kPixels * 4});
        if (!depth) { return std::unexpected(depth.error()); }
        if (auto result = commands.copyTextureToBuffer({.texture = context.inputTexture("depth").texture(),
            .buffer = *depth, .width = kWidth, .height = kHeight}); !result) { return result; }
        const std::array names{"statistics", "workload", "buildDiagnostics"};
        const std::array strides{kVirtualShadowMapFaceStatisticsSize, kVirtualShadowMapWorkloadSize, kVirtualShadowMapBuildDiagnosticSize};
        const std::array offsets{layout.statistics, layout.workload, layout.buildDiagnostics};
        for (uint32_t index = 0; index < names.size(); ++index) {
            if (index != 0 && !collectWork) { continue; }
            const uint64_t size = uint64_t(layout.faces) * strides[index];
            auto source = context.inputBuffer(names[index]).buffer()->slice({0, size});
            auto destination = buffer->slice({offsets[index], size});
            if (!source) { return std::unexpected(source.error()); }
            if (!destination) { return std::unexpected(destination.error()); }
            if (auto result = commands.copyBuffer(*source, *destination); !result) { return result; }
        }
        return commands.copyTexture({.source = context.inputTexture("debug").texture(),
            .destination = context.outputTexture("color").texture(), .width = kWidth, .height = kHeight});
    }
private:
    std::shared_ptr<EvidenceState> state_;
};

struct ProfilingSnapshot
{
    std::vector<uint8_t> shadow;
    std::vector<uint32_t> debug, depth;
    std::vector<std::array<uint32_t, 32>> statistics;
    std::vector<std::array<uint64_t, kVirtualShadowMapWorkCounterCount>> workload;
    std::vector<std::array<uint64_t, kVirtualShadowMapBuildDiagnosticCounterCount>> buildDiagnostics;
};

ProfilingSnapshot profilingSnapshot(const std::string& name, const ProfilingReadbackLayout& layout, bool collectWork)
{
    ProfilingSnapshot result;
    auto* buffer = activeEvidence->namedReadbacks.at(name);
    buffer->invalidate();
    const auto* bytes = static_cast<const std::byte*>(buffer->map());
    require(bytes != nullptr, "Map VSM profiling evidence");
    result.shadow.resize(kPixels * layout.layers); result.debug.resize(kPixels * layout.layers); result.depth.resize(kPixels);
    result.statistics.resize(layout.faces);
    for (uint32_t layer = 0; layer < layout.layers; ++layer) {
        std::memcpy(result.shadow.data() + layer * kPixels, bytes + layer * layout.shadowStride, kPixels);
    }
    std::memcpy(result.debug.data(), bytes + layout.debug, result.debug.size() * 4);
    std::memcpy(result.depth.data(), bytes + layout.depth, kPixels * 4);
    std::memcpy(result.statistics.data(), bytes + layout.statistics, layout.faces * kVirtualShadowMapFaceStatisticsSize);
    if (collectWork) {
        result.workload.resize(layout.faces);
        std::memcpy(result.workload.data(), bytes + layout.workload, layout.faces * kVirtualShadowMapWorkloadSize);
        result.buildDiagnostics.resize(layout.faces);
        std::memcpy(result.buildDiagnostics.data(), bytes + layout.buildDiagnostics, layout.faces * kVirtualShadowMapBuildDiagnosticSize);
    }
    buffer->unmap();
    return result;
}

void requireBuildDiagnosticInvariants(const ProfilingSnapshot& snapshot, bool clusterCulling, const std::string& label,
    bool clusterPageBitset = false)
{
    require(snapshot.buildDiagnostics.size() == snapshot.statistics.size(), label + ": missing build diagnostic faces");
    for (size_t face = 0; face < snapshot.statistics.size(); ++face) {
        const auto& stats = snapshot.statistics[face];
        const auto& counters = snapshot.buildDiagnostics[face];
        const std::string prefix = label + "/face" + std::to_string(face);
        if (!clusterCulling || stats[11] == 0 || stats[16] == 0) {
            require(std::ranges::all_of(counters, [](uint64_t value) { return value == 0; }),
                prefix + ": inactive, warm or cull-disabled face wrote build diagnostics");
            continue;
        }
        const uint64_t groups = counters[kVSMBuildDiagnosticClusterGroups];
        const uint64_t valid = counters[kVSMBuildDiagnosticClusterBoundsValidGroups];
        const uint64_t surviving = counters[kVSMBuildDiagnosticClusterSurvivingGroups];
        const uint64_t levels = counters[kVSMBuildDiagnosticClusterLevelTests];
        const uint64_t pages = counters[kVSMBuildDiagnosticClusterPageTests], hits = counters[kVSMBuildDiagnosticClusterPageHits];
        const uint64_t near = counters[kVSMBuildDiagnosticClusterNearPlaneKeeps];
        const uint64_t outside = counters[kVSMBuildDiagnosticClusterOutsideLevels], empty = counters[kVSMBuildDiagnosticClusterEmptyLevels];
        const uint64_t bitsetLevels = counters[kVSMBuildDiagnosticClusterBitsetLevelTests];
        const uint64_t words = counters[kVSMBuildDiagnosticClusterBitsetWordTests];
        require(valid <= groups && surviving <= valid, prefix + ": invalid cluster group counts");
        require(levels == valid * stats[11], prefix + ": bounds-valid groups did not test every face level");
        require(hits + near + outside + empty == levels, prefix + ": cluster level outcomes do not partition tests");
        require(bitsetLevels == (clusterPageBitset ? levels : 0), prefix + ": build scan method does not match its diagnostic control");
        require(counters[kVSMBuildDiagnosticClusterPageScanHistogram] + counters[kVSMBuildDiagnosticClusterBitsetScanHistogram] == near + outside,
            prefix + ": zero-length scans differ from near/outside outcomes");
        for (bool bitset : {false, true}) {
            const uint32_t base = bitset ? kVSMBuildDiagnosticClusterBitsetScanHistogram : kVSMBuildDiagnosticClusterPageScanHistogram;
            const uint64_t maximumScan = bitset ? 512 : 16384;
            uint64_t histogramCount = 0, minimumLoads = 0, maximumLoads = 0;
            for (uint32_t bin = 0; bin < kVirtualShadowMapBuildDiagnosticHistogramCount; ++bin) {
                const uint64_t count = counters[base + bin];
                const uint64_t lower = bin == 0 ? 0 : uint64_t(1) << (bin - 1);
                const uint64_t upper = bin == 0 ? 0 : std::min(maximumScan, (uint64_t(1) << bin) - 1);
                require(lower <= maximumScan || count == 0, prefix + ": scan histogram exceeds the maximum rectangle");
                histogramCount += count;
                minimumLoads += count * lower;
                maximumLoads += count * upper;
            }
            require(histogramCount == (bitset ? bitsetLevels : levels - bitsetLevels),
                prefix + ": scan histogram does not cover its selected level tests");
            const uint64_t actualLoads = bitset ? words : pages;
            require(minimumLoads <= actualLoads && actualLoads <= maximumLoads,
                prefix + ": actual page/word scans exceed histogram bounds");
        }
        if (stats[10] > 0) { require(groups > 0, prefix + ": dirty face with casters did not enter cluster precheck"); }
    }
}

void registerProfilingEvidencePass()
{
    static const bool registered = registerRenderGraphPassType("VSMProfilingEvidencePass", "Test-only VSM profiling readback",
        [] { return std::make_unique<VSMProfilingEvidencePass>(activeEvidence); });
    require(registered, "Register VSM profiling evidence pass");
}

class VirtualShadowMapProfilingTest final : public RHITest
{
public:
    VirtualShadowMapProfilingTest() { name = "virtual_shadow_map_profiling"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        bench::ValidationRecorder validation;
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapProfiling";
        Json report{{"passed", false}, {"frames", Json::array()}, {"counterNames", kVirtualShadowMapWorkCounterNames},
            {"buildDiagnosticCounterNames", kVirtualShadowMapBuildDiagnosticCounterNames}};
        try {
            std::filesystem::create_directories(directory);
            registerProfilingEvidencePass();
            require(VirtualShadowMapSettings{}.clusterPageBitset, "Production VSM settings must default to dirty-page bitsets");
            const auto fixture = writeLocalFixture(directory / "Fixture", false);
            constexpr std::array modes{"Off", "Timing", "Work", "Both"};
            constexpr std::array stages{"VSM reset", "VSM mark", "VSM allocator", "VSM build", "VSM raster", "VSM projection"};
            for (uint32_t variant = 0; variant < 5; ++variant) {
                const bool local = variant == 1 || variant == 2, compact = variant == 2, streamed = variant == 3, tiny = variant == 4;
                const std::string label = local ? (compact ? "CompactSMRT" : "LegacyPCF") : streamed ? "StreamSingleHard" :
                    tiny ? "TinySingleHard" : "SingleHard";
                // Small allocations exercise reset's diagnostic-driven dispatch size.
                // These short scans do not prove reuse of previously nonzero high histogram bins.
                const uint32_t pagesPerAxis = tiny ? 2 : kPagesPerAxis;
                constexpr uint32_t capacity = 4;
                const uint32_t faces = local ? capacity * 6 : 1, layers = local ? (compact ? capacity : faces) : 1;
                scene::SceneDocument scene;
                require(streamed ? scene.loadStreamMetadata(fixture.source) : scene.load(fixture.source), "Load profiling fixture");
                require(scene.hasStreamGeometry() == streamed, "Profiling fixture selected the wrong caster source");
                environment::WorldEnvironment world;
                world.sun.enabled = true; world.sun.direction = float3(1, -1, 0); world.sun.angularRadius = .01f;
                world.sun.illuminance = 3;
                world.moon.enabled = false;
                RenderView view;
                RenderGraphPreviewRenderer preview;
                configureCachePreview(preview, scene, view, context, validation, world);
                scene::LightingSettings lighting; lighting.autoExposure.enabled = false;
                lighting.lights = {localLight("Point", false, {-1, 4, 0}), localLight("Spot", true, {1, 4, 0})};
                for (auto& light : lighting.lights) { light.properties.sourceRadius = .05f; }
                require(scene.setLighting(lighting), "Publish profiling lights");
                require(preview.setLighting(scene.lighting()), "Set profiling lights");
                activeEvidence = std::make_shared<EvidenceState>();
                RenderGraph graph;
                graph.setViewProperties({{"temporalJitter", false},
                    {"renderResolution", {{"width", kWidth}, {"height", kHeight}, {"adaptive", false}}}});
                graph.addNode("VisibilityBufferPass", "Raster", {{"path", fixture.source.generic_string()}, {"streamAssetPath", fixture.cache.generic_string()},
                    {"autoBuildStreamAsset", false}, {"maxResidentPages", 64}, {"maxLockedFallbackPages", 64},
                    {"maxActiveGroups", 256}, {"maxGpuPageRequests", 256}, {"maxTraversalWorkers", 64}, {"maxTraversalWorkItems", 512},
                    {"autoLod", false}, {"lodLevel", 0}, {"instanceHzbCull", false}, {"meshletHzbCull", false},
                    {"meshletNormalConeCull", false}, {"hybridRaster", false}, {"enableClas", false}, {"enableClusterRtx", false}});
                for (uint32_t mode = 0; mode < modes.size(); ++mode) {
                    const std::string name = modes[mode], evidence = name + "Evidence";
                    const bool timing = (mode & 1) != 0, work = (mode & 2) != 0;
                    graph.addNode(local ? "VirtualShadowMapLightsPass" : "VirtualShadowMapPass", name,
                        {{"lightCapacity", capacity}, {"compactOutputs", compact}, {"physicalPageBudget", 512}, {"physicalPageCount", tiny ? 4 : 512},
                         {"pageSize", kPageSize}, {"virtualPagesPerAxis", pagesPerAxis}, {"clipmapCount", kClipmapCount},
                         {"firstClipmapExtent", kFirstExtent}, {"maxWorkItems", 65535}, {"shadowBias", .002}, {"shadowDebug", true},
                         {"cacheEnabled", true}, {"receiverFootprint", false}, {"filter", compact ? "smrt" : local ? "pcf" : "hard"},
                         {"smrtRayCount", 2}, {"smrtSamplesPerRay", 2}, {"smrtTemporalEnabled", false},
                         {"benchmarkGPUProfile", timing}, {"benchmarkGPUWork", work}});
                    graph.addEdge("Raster.depth", name + ".depth"); graph.addEdge("Raster.rasterInfo", name + ".rasterInfo");
                    graph.addNode("VSMProfilingEvidencePass", evidence,
                        {{"layers", layers}, {"faces", faces}, {"collectWork", work}, {"evidenceName", name}});
                    graph.addEdge(name + (local ? ".shadows" : ".shadow"), evidence + ".shadow");
                    graph.addEdge(name + ".debug", evidence + ".debug"); graph.addEdge("Raster.depth", evidence + ".depth");
                    graph.addEdge(name + ".statistics", evidence + ".statistics");
                    if (work) {
                        graph.addEdge(name + ".workload", evidence + ".workload");
                        graph.addEdge(name + ".buildDiagnostics", evidence + ".buildDiagnostics");
                    }
                    graph.markOutput(evidence + ".color");
                }
                const auto originalMatrix = scene.nodes()[0].localMatrix;
                for (uint32_t frame = 0; frame < 4; ++frame) {
                    const std::string phase = std::array{"Cold", "Warm", "Dirty", "Overflow"}[frame];
                    validation.phase(label + "-" + phase);
                    if (frame == 2) {
                        float4x4 translation; translation.SetupByTranslation(float3(.35f, 0, .2f));
                        require(scene.setNodeLocalMatrix(0, translation * originalMatrix), "Move profiling caster");
                    } else if (frame == 3) {
                        for (const char* mode : modes) {
                            require(graph.setNodeRuntimeProperty(graph.findNode(mode)->id, "maxWorkItems", 1), "Constrain profiling raster work");
                        }
                        require(scene.setNodeLocalMatrix(0, originalMatrix), "Restore profiling caster under overflow");
                    }
                    renderAndRequire(preview, graph, "OffEvidence.color");
                    auto completed = preview.collectCompletedGpuExecutionStats();
                    require(completed, "Read profiling timestamps");
                    require(completed->size() == 1 && completed->front().gpuTimingAvailable && !completed->front().profilingOverflow,
                        "Missing or overflowed profiling timestamp frame");
                    const auto& timings = completed->front();
                    std::array<ProfilingSnapshot, 4> snapshots;
                    report["frames"].push_back({{"variant", label}, {"phase", phase}, {"timings", Json::object()}, {"arms", Json::object()}});
                    auto& entry = report["frames"].back();
                    for (uint32_t mode = 0; mode < modes.size(); ++mode) {
                        const bool timing = (mode & 1) != 0, work = (mode & 2) != 0;
                        snapshots[mode] = profilingSnapshot(modes[mode], {layers, faces, work}, work);
                        entry["arms"][modes[mode]] = {{"statistics", snapshots[mode].statistics}, {"workload", snapshots[mode].workload},
                            {"buildDiagnostics", snapshots[mode].buildDiagnostics},
                            {"nonzeroDepthPixels", std::count_if(snapshots[mode].depth.begin(), snapshots[mode].depth.end(), [](uint32_t value) { return value != 0; })}};
                        const auto node = std::ranges::find(timings.nodes, modes[mode], &RenderGraphNodeExecutionStat::name);
                        require(node != timings.nodes.end(), "Missing profiling graph node");
                        const size_t activeFaces = std::count_if(snapshots[mode].statistics.begin(), snapshots[mode].statistics.end(),
                            [](const auto& stats) { return stats[11] != 0; });
                        require(activeFaces == (local ? 8u : 1u), "Profiling fixture omitted an admitted face");
                        Json durations = Json::object();
                        for (const char* stage : stages) {
                            size_t count = 0; double total = 0;
                            for (const auto& section : node->sections) {
                                if (section.name != stage) { continue; }
                                ++count;
                                require(!section.cpuOnly && section.gpuTimingAvailable && std::isfinite(section.gpuMilliseconds) && section.gpuMilliseconds >= 0,
                                    std::string("Invalid GPU stage timing: ") + stage);
                                require(section.parent == UINT32_MAX || section.parent < node->sections.size(), "Invalid GPU stage parent");
                                total += section.gpuMilliseconds;
                            }
                            require(count == (timing ? activeFaces : 0), std::string("GPU stage toggle/count mismatch: ") + stage);
                            durations[stage] = {{"count", count}, {"gpuMs", total}};
                        }
                        entry["timings"][modes[mode]] = std::move(durations);
                        if (mode != 0) {
                            require(snapshots[mode].shadow == snapshots[0].shadow && snapshots[mode].debug == snapshots[0].debug &&
                                snapshots[mode].depth == snapshots[0].depth && snapshots[mode].statistics == snapshots[0].statistics,
                                label + "-" + phase + ": instrumentation changed images or the 32-word statistics ABI");
                        }
                        if (work) { requireBuildDiagnosticInvariants(snapshots[mode], true, label + "-" + phase + "-" + modes[mode], true); }
                    }
                    for (uint32_t face = 0; face < faces; ++face) {
                        for (uint32_t counter = 0; counter < kVirtualShadowMapWorkCounterCount; ++counter) {
                            // Shared mark-cache collisions can change repeated OR counts with execution order.
                            if (counter == kVSMWorkMarkPageRequests) { continue; }
                            require(snapshots[2].workload[face][counter] == snapshots[3].workload[face][counter],
                                "GPU timing changed a deterministic workload counter");
                        }
                        for (uint32_t mode : {2u, 3u}) {
                            const auto& work = snapshots[mode].workload[face];
                            require(work[kVSMWorkMarkPageRequests] <= work[kVSMWorkMarkLevelTests],
                                "Mark requests exceed one possible page write per tested level");
                            if (snapshots[mode].statistics[face][0] > 0) {
                                require(work[kVSMWorkMarkPageRequests] > 0, "Requested fine pages have no mark atomic writes");
                            }
                        }
                    }
                    require(snapshots[2].buildDiagnostics == snapshots[3].buildDiagnostics, "GPU timing changed build diagnostics");
                    std::array<uint64_t, kVirtualShadowMapWorkCounterCount> totals{};
                    uint64_t dirtyPages = 0, overflow = 0, defaultBitsetLevels = 0;
                    for (uint32_t face = 0; face < faces; ++face) {
                        for (uint32_t index = 0; index < totals.size(); ++index) { totals[index] += snapshots[3].workload[face][index]; }
                        dirtyPages += snapshots[3].statistics[face][16]; overflow += snapshots[3].statistics[face][5];
                        defaultBitsetLevels += snapshots[3].buildDiagnostics[face][kVSMBuildDiagnosticClusterBitsetLevelTests];
                    }
                    entry["statistics"] = snapshots[3].statistics; entry["workload"] = snapshots[3].workload; entry["totals"] = totals;
                    entry["buildDiagnostics"] = snapshots[3].buildDiagnostics;
                    entry["defaultBitsetTestedLevels"] = defaultBitsetLevels;
                    std::ofstream(directory / "VirtualShadowMapProfiling.json") << report.dump(2) << '\n';
                    for (uint32_t face = 0; face < faces; ++face) {
                        const auto& counters = snapshots[3].workload[face];
                        const auto& stats = snapshots[3].statistics[face];
                        if (stats[11] == 0) {
                            require(std::ranges::all_of(counters, [](uint64_t value) { return value == 0; }), "Inactive face published workload");
                        }
                        require(counters[kVSMWorkBuildDecodedTriangles] <= counters[kVSMWorkBuildActiveTriangleSlots], "Decoded more build triangles than active slots");
                        require(counters[kVSMWorkRasterFineGroups] == stats[4], "Fine raster groups disagree with admitted work");
                        if (frame == 1) {
                            require(stats[16] == 0 && stats[21] == 0 && counters[kVSMWorkBuildActiveTriangleSlots] == 0 &&
                                counters[kVSMWorkRasterCoarseGroups] == 0 && counters[kVSMWorkRasterFineGroups] == 0 &&
                                counters[kVSMWorkRasterDecodedTriangles] == 0 && counters[kVSMWorkRasterOutputPrimitives] == 0 &&
                                counters[kVSMWorkRasterClearedPages] == 0, "Warm cache reported depth rendering work");
                        }
                    }
                    require(totals[kVSMWorkMarkReceiverDispatches] > 0 && totals[kVSMWorkProjectionReceivers] > 0 &&
                        totals[kVSMWorkProjectionAtlasLoads] > 0, "Profiling fixture has no receiver/projection work");
                    require((compact && totals[kVSMWorkProjectionRays] > 0) || (!compact && totals[kVSMWorkProjectionRays] == 0), "SMRT ray workload does not match filter");
                    if (frame != 1) { require(dirtyPages > 0 && totals[kVSMWorkBuildDecodedTriangles] > 0, "Cold/dirty fixture did not build depth work"); }
                    require(frame == 1 ? defaultBitsetLevels == 0 : defaultBitsetLevels > 0,
                        label + "-" + phase + ": production default did not select the actual bitset build path");
                    if (frame == 3) { require(overflow > 0, "Profiling overflow fixture did not overflow"); }
                }
                auto incompatible = graph.findNode("Off")->properties;
                incompatible["benchmarkCPUProfile"] = true;
                incompatible["benchmarkGPUProfile"] = true;
                require(graph.setNodeProperties(graph.findNode("Off")->id, incompatible), "Set incompatible profiling controls");
                const auto rejected = preview.render(graph, kWidth, kHeight, "OffEvidence.color");
                require(!rejected && preview.lastLog().find("profiling must run separately") != std::string::npos,
                    "Simultaneous CPU attribution and detailed GPU timing was not explicitly rejected");
                report[label + "CPUProfileRejection"] = preview.lastLog();
                requireValidPreview(validation.snapshot());
            }
            activeEvidence.reset(); report["validation"] = validation.snapshot(); report["passed"] = true;
            std::ofstream(directory / "VirtualShadowMapProfiling.json") << report.dump(2) << '\n';
            return RHITestResult::pass("Independent VSM timing/work toggles preserve images and statistics; six stages and cold/warm/dirty/overflow workloads verified for single/legacy/compact lights");
        } catch (const std::exception& error) {
            activeEvidence.reset(); report["error"] = error.what(); report["validation"] = validation.snapshot();
            std::ofstream(directory / "VirtualShadowMapProfiling.json") << report.dump(2) << '\n';
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapProfilingTest);

class VirtualShadowMapEquivalenceTest : public RHITest
{
public:
    explicit VirtualShadowMapEquivalenceTest(bool bitsetComparison) : bitsetComparison_(bitsetComparison)
    {
        name = bitsetComparison ? "virtual_shadow_map_cluster_page_bitset_equivalence" : "virtual_shadow_map_optimization_equivalence";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        bench::ValidationRecorder validation;
        const std::string reportName = bitsetComparison_ ? "VirtualShadowMapClusterPageBitsetEquivalence" : "VirtualShadowMapOptimizationEquivalence";
        const auto directory = std::filesystem::absolute(context.outputDirectory) / reportName;
        Json report{{"passed", false}, {"frames", Json::array()}, {"counterNames", kVirtualShadowMapWorkCounterNames},
            {"buildDiagnosticCounterNames", kVirtualShadowMapBuildDiagnosticCounterNames}};
        const auto save = [&] { std::ofstream(directory / (reportName + ".json")) << report.dump(2) << '\n'; };
        try {
            std::filesystem::create_directories(directory);
            registerProfilingEvidencePass();
            const auto floor = writeLocalFixture(directory / "Floor", false);
            const auto near = writeLocalFixture(directory / "Near", false, true);
            const auto room = writeLocalFixture(directory / "Room", true);
            LocalFixture offscreen;
            offscreen.source = writeFixture(directory / "Offscreen", true);
            offscreen.cache = directory / "Offscreen/Scene.meshstream.bin";
            // A distant independent instance exercises cluster rejection without
            // replacing the off-camera caster that shadows the visible floor.
            Json document;
            { std::ifstream input(offscreen.source); input >> document; }
            document["nodes"].push_back({{"name", "DistantCaster"}, {"mesh", 0}, {"translation", {128,0,128}}});
            document["scenes"][0]["nodes"].push_back(2);
            { std::ofstream output(offscreen.source); output << document.dump(2) << '\n'; }
            std::string log;
            require(scene::buildMeshletStreamAssetOffline({.sourcePath = offscreen.source, .outputPath = offscreen.cache,
                .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log), "Cook optimization offscreen fixture: " + log);

            constexpr uint32_t capacity = 4;
            const std::vector<const char*> baseArms = bitsetComparison_ ? std::vector<const char*>{"Baseline", "Bitset"} :
                std::vector<const char*>{"Legacy", "MarkOnly", "ClusterOnly", "Both"};
            if (bitsetComparison_) {
                report["selectionControls"] = {
                    {"Baseline",{{"clusterPageBitset",true},{"benchmarkClusterPageBitset",false}}},
                    {"Bitset",Json::object()},
                    {"ProductionOff",{{"clusterPageBitset",false}}},
                    {"CullDisabled",{{"clusterPageBitset",true},{"benchmarkClusterPageBitset",true},{"benchmarkDisableClusterCull",true}}}};
            }
            uint64_t markReducedStates = 0, clusterReducedStates = 0, bitsetTestedLevels = 0, bitsetControlStates = 0;
            std::array<uint64_t, 2> defaultBitsetLevelsByPass{};
            for (bool streamed : {false, true}) {
                for (const LocalFixture* fixture : std::array<const LocalFixture*, 4>{&floor, &near, &offscreen, &room}) {
                    const bool mainMatrix = fixture == &floor, nearCase = fixture == &near, roomCase = fixture == &room;
                    const std::vector<std::string> filters = mainMatrix ? (bitsetComparison_ ?
                        std::vector<std::string>{"pcf", "smrt", "single-hard", "single-hard-31", "single-hard-33"} :
                        std::vector<std::string>{"pcf", "smrt"}) :
                        std::vector<std::string>{nearCase ? "smrt" : roomCase ? "pcf" : "hard"};
                    for (const auto& filter : filters) {
                        const bool single = filter.starts_with("single-hard");
                        const bool bitsetControls = bitsetComparison_ && mainMatrix && !streamed &&
                            (filter == "single-hard" || filter == "smrt");
                        auto arms = baseArms;
                        if (bitsetControls) { arms.insert(arms.end(), {"ProductionOff", "CullDisabled"}); }
                        const uint32_t pagesPerAxis = filter == "single-hard-31" ? 31 : filter == "single-hard-33" ? 33 : kPagesPerAxis;
                        const uint32_t faces = single ? 1 : capacity * 6, layers = single ? 1 : capacity;
                        const std::string label = std::string(streamed ? "Stream-" : "Resident-") +
                            (mainMatrix ? "Floor-" : nearCase ? "Near-" : roomCase ? "Room-" : "Offscreen-") + filter;
                        scene::SceneDocument scene;
                        require(streamed ? scene.loadStreamMetadata(fixture->source) : scene.load(fixture->source), "Load optimization fixture");
                        require(scene.hasStreamGeometry() == streamed, "Optimization fixture selected wrong caster source");
                        environment::WorldEnvironment world;
                        world.sun.enabled = true; world.sun.direction = float3(1,-1,0); world.sun.angularRadius = .01f;
                        world.sun.illuminance = 3; world.moon.enabled = false;
                        RenderView view;
                        RenderGraphPreviewRenderer preview;
                        configureCachePreview(preview, scene, view, context, validation, world);
                        if (roomCase) {
                            require(view.setCamera({.eye = {0,0,3}, .center = {6,6,0}, .fovDegrees = 65,
                                .nearPlane = .05f, .farPlane = 30}), "Set point seam optimization view");
                        }
                        scene::LightingSettings lighting; lighting.autoExposure.enabled = false;
                        lighting.lights = {localLight("Point", false, roomCase ? Point{0,0,0} : nearCase ? Point{0,3,0} : Point{-1,4,0}),
                            localLight("Spot", true, nearCase ? Point{0,3,0} : Point{1,4,0})};
                        for (auto& light : lighting.lights) { light.properties.sourceRadius = .05f; }
                        require(scene.setLighting(lighting), "Publish optimization lights");
                        require(preview.setLighting(scene.lighting()), "Set optimization lights");
                        activeEvidence = std::make_shared<EvidenceState>();
                        RenderGraph graph;
                        graph.setViewProperties({{"temporalJitter", false},
                            {"renderResolution", {{"width", kWidth}, {"height", kHeight}, {"adaptive", false}}}});
                        graph.addNode("VisibilityBufferPass", "Raster", {{"path", fixture->source.generic_string()},
                            {"streamAssetPath", fixture->cache.generic_string()}, {"autoBuildStreamAsset", false},
                            {"maxResidentPages", 64}, {"maxLockedFallbackPages", 64}, {"maxActiveGroups", 256},
                            {"maxGpuPageRequests", 256}, {"maxTraversalWorkers", 64}, {"maxTraversalWorkItems", 512},
                            {"autoLod", false}, {"lodLevel", 0}, {"instanceHzbCull", false}, {"meshletHzbCull", false},
                            {"meshletNormalConeCull", false}, {"hybridRaster", false}, {"enableClas", false}, {"enableClusterRtx", false}});
                        for (uint32_t arm = 0; arm < arms.size(); ++arm) {
                            const std::string name = arms[arm], evidence = name + "Evidence";
                            Json settings{{"lightCapacity", capacity}, {"compactOutputs", true}, {"physicalPageBudget", 512}, {"physicalPageCount", 512},
                                 {"pageSize", kPageSize}, {"virtualPagesPerAxis", pagesPerAxis}, {"clipmapCount", kClipmapCount},
                                 {"firstClipmapExtent", kFirstExtent}, {"maxWorkItems", 65535}, {"shadowBias", .002},
                                 {"cacheEnabled", true}, {"receiverFootprint", true}, {"filter", single ? "hard" : filter}, {"pcfRadius", 1.5},
                                 {"nearPlane", nearCase ? .25 : .001}, {"smrtRayCount", 2}, {"smrtSamplesPerRay", 8},
                                 {"smrtTemporalEnabled", false}, {"benchmarkGPUWork", true},
                                 {"benchmarkLegacyMark", bitsetComparison_ ? false : (arm & 1) == 0},
                                 {"benchmarkDisableClusterCull", bitsetComparison_ ? false : (arm & 2) == 0},
                                 {"benchmarkClusterPageBitset", false}};
                            if (bitsetComparison_) {
                                // The baseline explicitly preserves legacy scanning. A conflicting
                                // production value proves the benchmark override takes precedence.
                                // The Bitset arm deliberately omits both values to test the default.
                                settings.erase("benchmarkClusterPageBitset");
                                if (arm == 0) {
                                    settings["clusterPageBitset"] = true;
                                    settings["benchmarkClusterPageBitset"] = false;
                                } else if (arm == 2) {
                                    settings["clusterPageBitset"] = false;
                                } else if (arm == 3) {
                                    settings["clusterPageBitset"] = true;
                                    settings["benchmarkClusterPageBitset"] = true;
                                    settings["benchmarkDisableClusterCull"] = true;
                                }
                            }
                            graph.addNode(single ? "VirtualShadowMapPass" : "VirtualShadowMapLightsPass", name, settings);
                            graph.addEdge("Raster.depth", name + ".depth"); graph.addEdge("Raster.rasterInfo", name + ".rasterInfo");
                            graph.addNode("VSMProfilingEvidencePass", evidence,
                                {{"layers", layers}, {"faces", faces}, {"collectWork", true}, {"evidenceName", name}});
                            graph.addEdge(name + (single ? ".shadow" : ".shadows"), evidence + ".shadow");
                            graph.addEdge(name + ".debug", evidence + ".debug");
                            graph.addEdge("Raster.depth", evidence + ".depth");
                            graph.addEdge(name + ".statistics", evidence + ".statistics");
                            graph.addEdge(name + ".workload", evidence + ".workload");
                            graph.addEdge(name + ".buildDiagnostics", evidence + ".buildDiagnostics");
                            graph.markOutput(evidence + ".color");
                        }
                        const auto original = scene.nodes()[0].localMatrix;
                        const uint32_t frameCount = mainMatrix ? 5 : 2;
                        for (uint32_t frame = 0; frame < frameCount; ++frame) {
                            const std::string phase = std::array{"Cold", "Warm", "Dirty", "Overflow", "Recover"}[frame];
                            validation.phase(label + "-" + phase);
                            if (frame == 2) {
                                float4x4 translation; translation.SetupByTranslation(float3(.35f,0,.2f));
                                require(scene.setNodeLocalMatrix(0, translation * original), "Move optimization caster");
                            } else if (frame >= 3) {
                                for (const char* arm : arms) {
                                    require(graph.setNodeRuntimeProperty(graph.findNode(arm)->id, "maxWorkItems", frame == 3 ? 1 : 65535),
                                        "Change optimization overflow limit");
                                }
                                if (frame == 3) { require(scene.setNodeLocalMatrix(0, original), "Restore optimization caster"); }
                            }
                            renderAndRequire(preview, graph, std::string(arms.front()) + "Evidence.color");
                            std::vector<ProfilingSnapshot> snapshots(arms.size());
                            std::vector<std::array<uint64_t, kVirtualShadowMapWorkCounterCount>> work(arms.size());
                            report["frames"].push_back({{"variant", label}, {"phase", phase}, {"arms", Json::object()}});
                            auto& entry = report["frames"].back();
                            for (uint32_t arm = 0; arm < arms.size(); ++arm) {
                                snapshots[arm] = profilingSnapshot(arms[arm], {layers, faces, true}, true);
                                const auto& snapshot = snapshots[arm];
                                for (const auto& face : snapshot.workload) {
                                    for (uint32_t counter = 0; counter < work[arm].size(); ++counter) { work[arm][counter] += face[counter]; }
                                }
                                entry["arms"][arms[arm]] = {{"statistics", snapshot.statistics}, {"workload", snapshot.workload},
                                    {"buildDiagnostics", snapshot.buildDiagnostics},
                                    {"workTotals", work[arm]}, {"shadowSha256", debug::debugSha256(snapshot.shadow)},
                                    {"debugSha256", debug::debugSha256(std::span(reinterpret_cast<const uint8_t*>(snapshot.debug.data()), snapshot.debug.size() * 4))},
                                    {"depthSha256", debug::debugSha256(std::span(reinterpret_cast<const uint8_t*>(snapshot.depth.data()), snapshot.depth.size() * 4))}};
                                save();
                                requireBuildDiagnosticInvariants(snapshot, bitsetComparison_ ? arm != 3 : (arm & 2) != 0,
                                    label + "-" + phase + "-" + arms[arm], bitsetComparison_ && arm == 1);
                                if (arm != 0) {
                                    require(snapshot.shadow == snapshots[0].shadow && snapshot.debug == snapshots[0].debug &&
                                        snapshot.depth == snapshots[0].depth && snapshot.statistics == snapshots[0].statistics,
                                        label + "-" + phase + ": " + arms[arm] + " changed images/depth or the 32-word statistics ABI");
                                }
                            }
                            if (bitsetComparison_) {
                                bitsetControlStates += bitsetControls;
                                for (uint32_t face = 0; face < faces; ++face) {
                                    for (uint32_t arm = 1; arm < arms.size(); ++arm) {
                                        // Disabling cluster culling intentionally restores rejected
                                        // build work; its image/statistics checks above remain exact.
                                        if (arm == 3) { continue; }
                                        for (uint32_t counter = 0; counter < kVirtualShadowMapWorkCounterCount; ++counter) {
                                            // Shared mark hash collisions can reorder repeated ORs.
                                            if (counter == kVSMWorkMarkPageRequests) { continue; }
                                            require(snapshots[0].workload[face][counter] == snapshots[arm].workload[face][counter],
                                                label + "-" + phase + ": bitset selection changed deterministic GPU work");
                                        }
                                    }
                                    for (uint32_t arm = 0; arm < arms.size(); ++arm) {
                                        const auto& counts = snapshots[arm].workload[face];
                                        require(counts[kVSMWorkMarkPageRequests] <= counts[kVSMWorkMarkLevelTests] &&
                                            (snapshots[arm].statistics[face][0] == 0 || counts[kVSMWorkMarkPageRequests] > 0),
                                            label + "-" + phase + ": invalid shared mark request count");
                                    }
                                    bitsetTestedLevels += snapshots[1].buildDiagnostics[face][kVSMBuildDiagnosticClusterBitsetLevelTests];
                                    defaultBitsetLevelsByPass[single ? 0 : 1] +=
                                        snapshots[1].buildDiagnostics[face][kVSMBuildDiagnosticClusterBitsetLevelTests];
                                }
                            } else {
                                for (uint32_t arm = 1; arm < arms.size(); ++arm) {
                                    for (uint32_t counter : {kVSMWorkMarkLevelTests, kVSMWorkMarkPageRequests,
                                         kVSMWorkBuildDecodedTriangles, kVSMWorkBuildLevelTests, kVSMWorkBuildPageTests}) {
                                        require(work[arm][counter] <= work[0][counter], label + "-" + phase + ": optimized GPU work increased");
                                    }
                                }
                                markReducedStates += work[1][kVSMWorkMarkLevelTests] < work[0][kVSMWorkMarkLevelTests] ||
                                    work[1][kVSMWorkMarkPageRequests] < work[0][kVSMWorkMarkPageRequests];
                                clusterReducedStates += work[2][kVSMWorkBuildDecodedTriangles] < work[0][kVSMWorkBuildDecodedTriangles];
                            }
                            uint64_t dirty = 0, overflow = 0, nearRejected = 0;
                            for (const auto& stats : snapshots[0].statistics) { dirty += stats[16]; overflow += stats[5]; nearRejected += stats[29]; }
                            require(work[0][kVSMWorkProjectionReceivers] > 0 && work[0][kVSMWorkProjectionAtlasLoads] > 0,
                                label + ": optimization fixture lacks actual projection work");
                            if (frame == 1) { require(dirty == 0 && work[0][kVSMWorkBuildDecodedTriangles] == 0, label + ": warm cache rebuilt depth"); }
                            if (frame == 3) { require(overflow > 0, label + ": overflow case did not overflow"); }
                            if (nearCase) { require(nearRejected > 0, label + ": near-plane exclusion was not exercised"); }
                            entry["equal"] = true;
                            save();
                        }
                        requireValidPreview(validation.snapshot());
                    }
                }
            }
            if (bitsetComparison_) {
                report["bitsetTestedLevels"] = bitsetTestedLevels;
                report["defaultBitsetLevelsByPass"] = {{"single",defaultBitsetLevelsByPass[0]},{"lights",defaultBitsetLevelsByPass[1]}};
                report["bitsetControlStates"] = bitsetControlStates;
                require(bitsetTestedLevels > 0, "Bitset matrix did not exercise a bitset level test");
                require(defaultBitsetLevelsByPass[0] > 0 && defaultBitsetLevelsByPass[1] > 0 && bitsetControlStates == 10,
                    "Bitset production defaults or explicit fallback controls were not exercised for both pass types");
            } else {
                report["markReducedStates"] = markReducedStates; report["clusterReducedStates"] = clusterReducedStates;
                require(markReducedStates > 0, "Optimization matrix did not exercise reduced mark work");
                require(clusterReducedStates > 0, "Optimization matrix did not exercise a rejected stream cluster");
            }
            activeEvidence.reset(); report["validation"] = validation.snapshot(); report["passed"] = true; save();
            return RHITestResult::pass(bitsetComparison_ ?
                "Dirty-page bitsets preserve images, receiver depth, statistics and deterministic work in single/multi-light production graphs" :
                "Four production VSM arms preserve images, receiver depth and statistics across resident/stream, PCF/SMRT, offscreen, near-plane, seam and cache/overflow states");
        } catch (const std::exception& error) {
            activeEvidence.reset(); report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
private:
    const bool bitsetComparison_;
};

class VirtualShadowMapOptimizationEquivalenceTest final : public VirtualShadowMapEquivalenceTest
{
public:
    VirtualShadowMapOptimizationEquivalenceTest() : VirtualShadowMapEquivalenceTest(false) {}
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapOptimizationEquivalenceTest);

class VirtualShadowMapClusterPageBitsetEquivalenceTest final : public VirtualShadowMapEquivalenceTest
{
public:
    VirtualShadowMapClusterPageBitsetEquivalenceTest() : VirtualShadowMapEquivalenceTest(true) {}
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapClusterPageBitsetEquivalenceTest);

} // namespace
} // namespace metallic::tests
