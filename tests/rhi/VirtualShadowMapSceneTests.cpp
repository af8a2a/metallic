#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "harness/VulkanDiagnostics.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/ScreenSpaceShadows.h"
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
    VisibilityBufferFrameInfo camera;
};

std::shared_ptr<EvidenceState> activeEvidence;

struct ReadbackLayout {
    uint64_t shadow = 0;
    uint64_t debug = (kPixels + 3) & ~uint64_t(3);
    uint64_t rasterDepth = debug + kPixels * 4;
    uint64_t receiverDepth = rasterDepth + kPixels * 4;
    uint64_t statistics = receiverDepth + kPixels * 4;
    uint64_t parameters = statistics + 16 * sizeof(uint32_t);
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
        reflection.addBufferInput("statistics").buffer(16 * sizeof(uint32_t), sizeof(uint32_t)).transferRead();
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
        const uint64_t bufferSizes[]{16 * sizeof(uint32_t), sizeof(ScreenSpaceShadowParameters)};
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
    std::array<uint32_t, 16> statistics{};
    VisibilityBufferFrameInfo camera;
    ScreenSpaceShadowParameters parameters;
};

Snapshot snapshot(const EvidenceState& evidence)
{
    require(evidence.readback != nullptr, "VSM evidence pass did not execute");
    require(evidence.camera.width == kWidth && evidence.camera.height == kHeight &&
        evidence.camera.viewport[1] == float(kWidth) && evidence.camera.viewport[2] == float(kHeight),
        "VSM fixture raster extent differs from the requested receiver extent");
    Snapshot result;
    result.shadow.resize(kPixels); result.debug.resize(kPixels);
    result.rasterDepth.resize(kPixels); result.receiverDepth.resize(kPixels);
    evidence.readback->invalidate();
    const auto* mapped = static_cast<const std::byte*>(evidence.readback->map());
    require(mapped != nullptr, "Map VSM scene evidence");
    std::memcpy(result.shadow.data(), mapped + kReadback.shadow, kPixels);
    std::memcpy(result.debug.data(), mapped + kReadback.debug, kPixels * 4);
    std::memcpy(result.rasterDepth.data(), mapped + kReadback.rasterDepth, kPixels * 4);
    std::memcpy(result.receiverDepth.data(), mapped + kReadback.receiverDepth, kPixels * 4);
    std::memcpy(result.statistics.data(), mapped + kReadback.statistics, sizeof(result.statistics));
    std::memcpy(&result.parameters, mapped + kReadback.parameters, sizeof(result.parameters));
    evidence.readback->unmap();
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
                                repeated.statistics == image.statistics, name + ": uncached fixed scene is not deterministic across frames");
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

} // namespace
} // namespace metallic::tests
