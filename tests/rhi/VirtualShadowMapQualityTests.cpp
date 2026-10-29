#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Render/GPUDrivenRaster.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLights.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
using Point = std::array<double, 3>;
constexpr uint32_t kWidth = 193, kHeight = 129;
constexpr size_t kPixels = size_t(kWidth) * kHeight;
constexpr uint32_t kCapacity = 4, kLayers = kCapacity * 6;
constexpr double kPi = 3.14159265358979323846;

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

struct EvidenceState {
    std::map<std::string, Buffer*> buffers;
    VisibilityBufferFrameInfo camera;
    std::map<std::string, uint32_t> layers;
};
std::shared_ptr<EvidenceState> evidence;
constexpr uint64_t kShadowStride = (kPixels + 3) & ~uint64_t(3);
constexpr uint64_t kDepthOffset = kShadowStride * kLayers;
constexpr uint64_t kStatsOffset = kDepthOffset + kPixels * 4;
constexpr uint64_t kTableOffset = kStatsOffset + kLayers * kVirtualShadowMapFaceStatisticsSize;
constexpr uint64_t kTableSize = sizeof(VirtualShadowMapLightsHeader) + kCapacity * sizeof(VirtualShadowMapLightEntry);
constexpr uint64_t kMemoryOffset = kTableOffset + kTableSize;
constexpr uint64_t kDebugOffset = kMemoryOffset + 8 * sizeof(uint64_t);
constexpr uint64_t kReadbackSize = kDebugOffset + kPixels * 4 * kLayers;

class VSMQualityEvidencePass final : public UnsafePass {
public:
    explicit VSMQualityEvidencePass(std::shared_ptr<EvidenceState> state) : state_(std::move(state)) {}
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        auto& shadows = reflection.addTextureInput("shadows").transferRead();
        shadows.format = Format::R8Unorm; shadows.layerCount = 0;
        auto& debug = reflection.addTextureInput("debug").transferRead();
        debug.format = Format::RGBA8Unorm; debug.layerCount = 0;
        reflection.addTextureInput("depth").transferRead().format = Format::D32Sfloat;
        reflection.addBufferInput("statistics").buffer(kLayers * kVirtualShadowMapFaceStatisticsSize).transferRead();
        reflection.addBufferInput("lights").buffer(kTableSize).transferRead();
        reflection.addBufferInput("memory").buffer(8 * sizeof(uint64_t)).transferRead();
        reflection.addBufferInput("rasterInfo").buffer(sizeof(VisibilityBufferFrameInfo), sizeof(VisibilityBufferFrameInfo))
            .shaderRead().memoryLocation = MemoryLocation::HostUpload;
        reflection.addBufferOutput("data").buffer(kReadbackSize).transferWrite().hostReadback();
        reflection.addTextureOutput("color").transferWrite().format = Format::RGBA8Unorm;
        return reflection;
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        auto& commands = context.commandBuffer();
        auto* buffer = context.outputBuffer("data").buffer();
        const auto name = context.properties().at("evidenceName").get<std::string>();
        state_->buffers[name] = buffer;
        const uint32_t layers = context.inputTexture("shadows").desc().layerCount;
        state_->layers[name] = layers;
        for (uint32_t layer = 0; layer < layers; ++layer) {
            auto destination = buffer->slice({layer * kShadowStride, kPixels});
            if (!destination) { return std::unexpected(destination.error()); }
            if (auto result = commands.copyTextureToBuffer({.texture = context.inputTexture("shadows").texture(),
                .buffer = *destination, .width = kWidth, .height = kHeight, .baseLayer = layer}); !result) { return result; }
            if (context.properties().value("captureDebug", false)) {
                auto debugDestination = buffer->slice({kDebugOffset + uint64_t(layer) * kPixels * 4, kPixels * 4});
                if (!debugDestination) { return std::unexpected(debugDestination.error()); }
                if (auto result = commands.copyTextureToBuffer({.texture = context.inputTexture("debug").texture(),
                    .buffer = *debugDestination, .width = kWidth, .height = kHeight, .baseLayer = layer}); !result) { return result; }
            }
        }
        auto depth = buffer->slice({kDepthOffset, kPixels * 4});
        if (!depth) { return std::unexpected(depth.error()); }
        if (auto result = commands.copyTextureToBuffer({.texture = context.inputTexture("depth").texture(),
            .buffer = *depth, .width = kWidth, .height = kHeight}); !result) { return result; }
        for (const auto& [name, offset, size] : {std::tuple{"statistics", kStatsOffset, uint64_t(kLayers * kVirtualShadowMapFaceStatisticsSize)},
            std::tuple{"lights", kTableOffset, kTableSize}, std::tuple{"memory", kMemoryOffset, uint64_t(8 * sizeof(uint64_t))}}) {
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
            .destination = context.outputTexture("color").texture(), .width = kWidth, .height = kHeight});
    }
private:
    std::shared_ptr<EvidenceState> state_;
};

struct Snapshot {
    VirtualShadowMapLightsHeader header{};
    std::array<VirtualShadowMapLightEntry, kCapacity> entries{};
    std::array<std::array<uint32_t, 32>, kLayers> statistics{};
    std::array<std::vector<uint8_t>, kLayers> shadows;
    std::array<std::vector<uint8_t>, kLayers> debug;
    std::array<uint64_t, 8> memory{};
    std::vector<float> depth;
    VisibilityBufferFrameInfo camera;
};
Snapshot snapshot(const std::string& name, bool captureDebug = false)
{
    Snapshot result;
    auto* buffer = evidence->buffers.at(name);
    buffer->invalidate();
    const auto* mapped = static_cast<const std::byte*>(buffer->map());
    require(mapped != nullptr, "Map VSM quality evidence");
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        result.shadows[layer].resize(kPixels, 255);
        if (layer < evidence->layers.at(name)) {
            std::memcpy(result.shadows[layer].data(), mapped + layer * kShadowStride, kPixels);
            if (captureDebug) {
                result.debug[layer].resize(kPixels * 4);
                std::memcpy(result.debug[layer].data(), mapped + kDebugOffset + uint64_t(layer) * kPixels * 4, kPixels * 4);
            }
        }
    }
    result.depth.resize(kPixels);
    std::memcpy(result.depth.data(), mapped + kDepthOffset, kPixels * 4);
    std::memcpy(result.statistics.data(), mapped + kStatsOffset, sizeof(result.statistics));
    std::memcpy(&result.header, mapped + kTableOffset, sizeof(result.header));
    std::memcpy(result.memory.data(), mapped + kMemoryOffset, sizeof(result.memory));
    std::memcpy(result.entries.data(), mapped + kTableOffset + sizeof(result.header), sizeof(result.entries));
    buffer->unmap(); result.camera = evidence->camera;
    require(result.header.version == kVirtualShadowMapLightsVersion && result.header.capacity == kCapacity,
        "VSM quality publication ABI mismatch");
    return result;
}

struct AuthoredTriangle { Point a, b, c; };
struct Fixture {
    std::vector<AuthoredTriangle> triangles;
    std::filesystem::path source, cache;
    uint32_t publishedPrimitives = 0, terminalTriangles = 0;
};

void addFixtureQuad(Fixture& fixture, Point center, Point u, Point v)
{
    const Point a = add(add(center, scale(u, -1)), scale(v, -1));
    const Point b = add(add(center, u), scale(v, -1));
    const Point c = add(add(center, u), v), d = add(add(center, scale(u, -1)), v);
    fixture.triangles.push_back({a,b,c}); fixture.triangles.push_back({a,c,d});
}

Fixture writeFixture(const std::filesystem::path& directory, bool room, double blockerHeight = 1.5,
    double blockerHalfExtent = .8, const Fixture* authored = nullptr)
{
    std::filesystem::create_directories(directory);
    Fixture fixture;
    if (authored) {
        fixture = *authored;
    } else if (!room) {
        addFixtureQuad(fixture, {0,0,0}, {8,0,0}, {0,0,-8});
        addFixtureQuad(fixture, {0,blockerHeight,0}, {blockerHalfExtent,0,0}, {0,0,blockerHalfExtent});
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
    if (room) {
        for (double x : {-1., 1.}) { for (double y : {-1., 1.}) { for (double z : {-1., 1.}) {
            const Point center{2*x,2*y,2*z}, normal = normalized(center);
            const Point tangent = normalized(cross(normal,Point{0,1,0}));
            addFixtureQuad(fixture,center,scale(tangent,.6),scale(cross(normal,tangent),.6));
        }}}
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
    Json primitives=Json::array();
    if (room || authored) {
        // Keep each authored quad in its own primitive. The quality reference
        // requires the complete geometry even on the locked stream terminal
        // cut; combining all quads can introduce a simplified parent.
        for (size_t quad=0; quad<fixture.triangles.size()/2; ++quad) {
            primitive["indices"]=gltf["accessors"].size();
            gltf["accessors"].push_back({{"bufferView",3},{"byteOffset",quad*6*sizeof(uint32_t)},
                {"componentType",5125},{"count",6},{"type","SCALAR"}});
            primitives.push_back(primitive);
        }
    } else { primitives.push_back(primitive); }
    fixture.publishedPrimitives=static_cast<uint32_t>(primitives.size());
    gltf["meshes"] = Json::array({Json{{"primitives",std::move(primitives)}}});
    Json material{{"name","LocalShadowOpaque"},{"doubleSided",true}};
    material["pbrMetallicRoughness"] = {{"baseColorFactor",{.7,.7,.7,1}},{"metallicFactor",0},{"roughnessFactor",1}};
    gltf["materials"] = Json::array({material});
    fixture.source = directory / "Scene.gltf"; fixture.cache = directory / "Scene.meshstream.bin";
    std::ofstream(fixture.source) << gltf.dump(2) << '\n';
    std::string log;
    const auto cooked = scene::buildMeshletStreamAssetOffline({.sourcePath = fixture.source, .outputPath = fixture.cache,
        .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log);
    require(cooked, "Cook local-light fixture: " + log);
    scene::MeshletStreamAsset asset;
    require(asset.open(fixture.cache,log),"Open quality stream geometry: "+log);
    require(asset.primitiveCount()==fixture.publishedPrimitives,"Quality primitive publication changed");
    for (const auto& instance : asset.instances()) {
        for (uint32_t group : asset.primitiveTerminalGroups(instance.primitiveIndex)) {
            const uint32_t page=asset.groups()[group].pageIndex;
            std::vector<uint8_t> scratch;
            std::span<const uint8_t> payload;
            require(scene::decodeMeshletStreamPayloadForDevice(asset.pages()[page],asset.pagePayload(page),scratch,payload,log),
                "Decode quality terminal geometry: "+log);
            scene::MeshletStreamPayloadHeader header;
            require(payload.size()>=sizeof(header),"Quality terminal payload lacks header");
            std::memcpy(&header,payload.data(),sizeof(header));
            for (uint32_t cluster=0; cluster<header.clusterCount; ++cluster) {
                scene::MeshletStreamPayloadCluster value;
                const size_t offset=header.clusterOffsetBytes+cluster*sizeof(value);
                require(offset+sizeof(value)<=payload.size(),"Quality terminal cluster is outside payload");
                std::memcpy(&value,payload.data()+offset,sizeof(value));
                fixture.terminalTriangles+=value.triangleCount;
            }
        }
    }
    require(fixture.terminalTriangles==fixture.triangles.size(),"Quality stream terminal cut differs from authored geometry");
    return fixture;
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

struct Hit { Point position{}, normal{}; double depth = 0; bool valid = false; size_t triangleIndex = 0; };

Hit cameraHit(const VisibilityBufferFrameInfo& camera, uint32_t x, uint32_t y, const Fixture& fixture)
{
    const Point eye = point(camera.eye), forward = normalized(add(point(camera.center),scale(eye,-1)));
    const Point right = normalized(cross(forward,point(camera.upProjection))), up = cross(right,forward);
    const bool orthographic = camera.upProjection[3] > .5f;
    const double halfHeight = orthographic ? camera.clipOrtho[2] * .5 : std::tan(camera.viewport[3] * .5);
    const Point lateral = add(scale(right,(2 * (x + .5 + camera.jitter[0]) / kWidth - 1) * camera.viewport[0] * halfHeight),
        scale(up,(1 - 2 * (y + .5 + camera.jitter[1]) / kHeight) * halfHeight));
    const Point origin = orthographic ? add(eye,lateral) : eye, direction = orthographic ? forward : add(forward,lateral);
    Hit result; double nearest = 1e30;
    for (size_t triangleIndex = 0; triangleIndex < fixture.triangles.size(); ++triangleIndex) {
        const auto& triangle = fixture.triangles[triangleIndex];
        const double distance = triangleDistance(origin,direction,triangle);
        const Point position = add(origin,scale(direction,distance));
        const double viewZ = dot(add(position,scale(eye,-1)),forward);
        if (distance >= nearest || viewZ < camera.clipOrtho[0] || viewZ > camera.clipOrtho[1]) { continue; }
        nearest = distance; result.valid = true; result.position = position; result.triangleIndex = triangleIndex;
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

struct Source {
    std::string kind;
    Point position{-2,4,0};
    Point direction{.6,-1,0};
    double size = .35;
};

RenderGraph makeGraph(const Fixture& fixture, const std::string& filter = "smrt", bool compactComparison = false,
    uint32_t virtualPagesPerAxis = 16)
{
    RenderGraph graph;
    graph.setName("VirtualShadowMapQuality");
    graph.setViewProperties({{"temporalJitter",false},
        {"renderResolution",{{"width",kWidth},{"height",kHeight},{"adaptive",false}}}});
    graph.addNode("VisibilityBufferPass", "Raster", {{"path",fixture.source.generic_string()},
        {"streamAssetPath",fixture.cache.generic_string()},{"autoBuildStreamAsset",false},
        {"maxResidentPages",128},{"maxLockedFallbackPages",128},{"maxActiveGroups",512},
        {"maxGpuPageRequests",512},{"maxTraversalWorkers",64},{"maxTraversalWorkItems",1024},
        {"autoLod",false},{"lodLevel",0},{"instanceFrustumCull",false},{"meshletFrustumCull",false},
        {"instanceHzbCull",false},{"meshletHzbCull",false},
        {"meshletNormalConeCull",false},{"hybridRaster",false},{"enableClas",false},{"enableClusterRtx",false}});
    const std::vector<const char*> names = compactComparison ? std::vector<const char*>{"Cached", "Compact"} :
        std::vector<const char*>{"Cached", "Redraw", "Hard"};
    for (const char* name : names) {
        graph.addNode("VirtualShadowMapLightsPass", name, {{"lightCapacity",kCapacity},{"physicalPageBudget",2048},
            {"pageSize",128},{"virtualPagesPerAxis",virtualPagesPerAxis},{"clipmapCount",4},{"firstClipmapExtent",4},
            {"maxWorkItems",131072},{"shadowBias",.002},{"cacheEnabled",true},
            {"forceRedraw",std::string_view(name)=="Redraw"},{"filter",std::string_view(name)=="Hard" ? "hard" : filter},
            {"pcfRadius",1.5},{"smrtRayCount",32},{"smrtSamplesPerRay",32},{"smrtMaxRayLength",12},
            {"receiverFootprint",true},{"compactOutputs",std::string_view(name)=="Compact"}});
        graph.addEdge("Raster.depth",std::string(name)+".depth");
        graph.addEdge("Raster.rasterInfo",std::string(name)+".rasterInfo");
        const std::string readback=std::string(name)+"Evidence";
        graph.addNode("VSMQualityEvidencePass",readback,{{"evidenceName",name},{"captureDebug",compactComparison}});
        for (const char* field : {"shadows","debug","lights","statistics","memory"}) {
            graph.addEdge(std::string(name)+"."+field,readback+"."+field);
        }
        graph.addEdge("Raster.depth",readback+".depth");
        graph.addEdge("Raster.rasterInfo",readback+".rasterInfo");
        graph.markOutput(readback+".color");
    }
    return graph;
}

void setProperty(RenderGraph& graph, const char* property, const Json& value)
{
    for (const char* node : {"Cached","Redraw"}) {
        require(graph.setNodeRuntimeProperty(graph.findNode(node)->id,property,value),
            std::string("Set VSM quality ")+property);
    }
}

void publishSource(scene::SceneDocument& scene, RenderGraphPreviewRenderer& preview, const Source& source)
{
    environment::WorldEnvironment world;
    world.sun.enabled=source.kind=="Directional"; world.moon.enabled=false;
    world.sun.direction=float3(float(source.direction[0]),float(source.direction[1]),float(source.direction[2]));
    world.sun.illuminance=3; world.sun.angularRadius=float(source.size);
    require(environment::validWorldEnvironment(world),"Validate quality celestial source");
    (void)preview.setWorldEnvironment(world);
    auto lighting=scene.lighting(); lighting.autoExposure.enabled=false;
    if (source.kind=="Directional") { lighting.lights.clear(); }
    else {
        if (lighting.lights.empty()) { lighting.lights.emplace_back(); }
        auto& light=lighting.lights[0];
        light.name=source.kind; light.properties.type=source.kind=="Point" ? "point" : "spot";
        light.position=float3(float(source.position[0]),float(source.position[1]),float(source.position[2]));
        light.direction=float3(0,-1,0); light.properties.range=30; light.properties.intensity=80;
        light.properties.outerConeAngle=1.2; light.properties.innerConeAngle=.8;
        light.properties.sourceRadius=source.size;
    }
    require(scene.setLighting(lighting),"Publish quality local source");
    (void)preview.setLighting(scene.lighting());
}

void configurePreview(RenderGraphPreviewRenderer& preview, scene::SceneDocument& scene,
    RenderView& view, RHITestContext& context, bench::ValidationRecorder& validation)
{
    preview.bindRuntimeScene(&scene);
    require(view.setRenderResolution(kWidth,kHeight),"Set quality render extent");
    require(view.setCamera({.eye={0,5,7},.center={0,0,0},.fovDegrees=52,
        .nearPlane=.1f,.farPlane=100,.orthoHeight=7}),"Set quality camera");
    preview.bindRenderView(&view);
    require(preview.initialize(context.enableValidation,false,false,context.enableValidation,validation.sink()),
        "Initialize quality preview without ray query");
    preview.setEnvironment({.enabled=false});
}

const VirtualShadowMapLightEntry& activeEntry(const Snapshot& image)
{
    const VirtualShadowMapLightEntry* result=nullptr;
    for (uint32_t index=0; index<image.header.count; ++index) {
        if (image.entries[index].status==kVirtualShadowMapLightAdmitted) {
            require(result==nullptr,"Quality fixture unexpectedly admitted multiple sources");
            result=&image.entries[index];
        }
    }
    require(result!=nullptr,"Quality fixture did not admit its source");
    return *result;
}

std::vector<uint8_t> visibility(const Snapshot& image, const Fixture& fixture, const Source& source)
{
    const auto& entry=activeEntry(image);
    std::vector<uint8_t> result(kPixels,255);
    for (uint32_t y=0; y<kHeight; ++y) { for (uint32_t x=0; x<kWidth; ++x) {
        const size_t pixel=size_t(y)*kWidth+x;
        const auto hit=cameraHit(image.camera,x,y,fixture);
        if (!hit.valid) { continue; }
        const uint32_t face=entry.kind==2 ? pointFace(add(hit.position,scale(source.position,-1))) : 0;
        result[pixel]=image.shadows[entry.baseLayer+face][pixel];
    }}
    return result;
}

void saveGray(const std::filesystem::path& path, const std::vector<uint8_t>& pixels)
{
    std::vector<uint8_t> rgba(kPixels*4,255);
    for (size_t pixel=0; pixel<kPixels; ++pixel) {
        rgba[pixel*4]=rgba[pixel*4+1]=rgba[pixel*4+2]=pixels[pixel];
    }
    std::string error;
    require(saveRgba8Png(path,rgba.data(),kWidth,kHeight,error),"Save VSM quality image: "+error);
}

// Independent area-light reference: authored double triangles and a polar
// 16x16 stratification, unrelated to the GPU's finite-ray sampling sequence or
// virtual projections. It deliberately models geometric binary visibility,
// without BRDF, irradiance or any GPU readback other than receiver depth.
double referenceVisibility(const Hit& hit, const Fixture& fixture, const Source& source)
{
    const Point toLight=source.kind=="Directional" ? normalized(scale(source.direction,-1)) :
        normalized(add(source.position,scale(hit.position,-1)));
    const Point tangent=normalized(cross(toLight,std::abs(toLight[1])<.9 ? Point{0,1,0} : Point{1,0,0}));
    const Point bitangent=cross(toLight,tangent);
    // Keep the ground truth at the actual receiver. Moving it along the normal
    // changes the penumbra rather than merely preventing numerical self hits;
    // triangleDistance already rejects nonpositive/near-zero ray parameters.
    const Point origin=hit.position;
    const uint32_t count=source.size>0 ? 256 : 1;
    uint32_t visible=0;
    for (uint32_t sample=0; sample<count; ++sample) {
        const double radius=std::sqrt((sample/16+.5)/16.);
        const double angle=(sample%16+.5)*2*kPi/16;
        const Point disk=add(scale(tangent,radius*std::cos(angle)),scale(bitangent,radius*std::sin(angle)));
        Point direction; double maximum;
        if (source.kind=="Directional") {
            direction=normalized(add(toLight,scale(disk,std::tan(source.size)))); maximum=100;
        } else {
            const Point ray=add(add(source.position,scale(disk,source.size)),scale(origin,-1));
            maximum=std::sqrt(dot(ray,ray)); direction=scale(ray,1/maximum);
        }
        bool blocked=false;
        for (const auto& triangle : fixture.triangles) {
            if (triangleDistance(origin,direction,triangle)<maximum-.001) { blocked=true; break; }
        }
        visible+=!blocked;
    }
    return double(visible)/count;
}

Json measureOracle(const Snapshot& image, const Fixture& fixture, const Source& source,
    const std::filesystem::path& directory, const std::string& name, bool room, bool requirePenumbra = true)
{
    const auto actual=visibility(image,fixture,source);
    std::vector<uint8_t> reference(kPixels,255),error(kPixels,0);
    uint32_t checked=0,depthErrors=0,penumbra=0,interior=0,interiorErrors=0,softPixels=0,seams=0,corners=0;
    uint32_t seamPenumbra=0,cornerPenumbra=0,seamSoftPixels=0,cornerSoftPixels=0;
    uint32_t depthSamples=0,peripheralDepthDifferences=0;
    double totalError=0,penumbraError=0,seamError=0,cornerError=0;
    Json depthErrorDetails=Json::array();
    const uint32_t stride=room ? 3 : 1;
    for (uint32_t y=0; y<kHeight; y+=stride) { for (uint32_t x=0; x<kWidth; x+=stride) {
        const size_t pixel=size_t(y)*kWidth+x;
        const auto hit=cameraHit(image.camera,x,y,fixture);
        // The planar visibility oracle is defined on |x|<=5, |z|<=4. Validate
        // its receiver depth in the same region, including the blocker there.
        // The finite floor's outer silhouette is outside that contract: exact
        // double rays and hardware subpixel edge coverage need not agree there.
        // Keep those differences in evidence instead of silently counting them
        // as validated pixels. The room retains whole-image depth validation.
        const bool depthRegion=room || (hit.valid && std::abs(hit.position[0])<=5 && std::abs(hit.position[2])<=4);
        depthSamples+=depthRegion;
        if (std::abs(image.depth[pixel]-hit.depth)>2e-5) {
            if (depthRegion) { ++depthErrors; }
            else { ++peripheralDepthDifferences; }
            depthErrorDetails.push_back({{"pixel",{x,y}},{"actual",image.depth[pixel]},
                {"expected",hit.depth},{"hitValid",hit.valid},{"insideOracleRegion",depthRegion},
                {"position",hit.position},{"normal",hit.normal}});
        }
        if (!hit.valid || (!room && std::abs(hit.position[1])>.01)) { continue; }
        if (!room && (std::abs(hit.position[0])>5 || std::abs(hit.position[2])>4)) { continue; }
        // Spot source tests use the fully illuminated inner region. Outside the
        // emitter's influence neutral visibility does not mean a failed trace.
        if (source.kind=="Spot") {
            const auto fromLight=normalized(add(hit.position,scale(source.position,-1)));
            if (-fromLight[1]<std::cos(1.2)+.02) { continue; }
        }
        const double expected=referenceVisibility(hit,fixture,source), value=actual[pixel]/255.;
        const double difference=std::abs(value-expected);
        reference[pixel]=uint8_t(std::lround(expected*255)); error[pixel]=uint8_t(std::lround(difference*255));
        ++checked; totalError+=difference; softPixels+=value>.05 && value<.95;
        if (expected>.05 && expected<.95) { ++penumbra; penumbraError+=difference; }
        if (expected==0 || expected==1) { ++interior; interiorErrors+=difference>.1; }
        if (source.kind=="Point") {
            auto delta=add(hit.position,scale(source.position,-1));
            std::array<double,3> magnitude{std::abs(delta[0]),std::abs(delta[1]),std::abs(delta[2])};
            std::sort(magnitude.begin(),magnitude.end());
            if (magnitude[2]-magnitude[1]<.15) {
                ++seams; seamError+=difference;
                seamPenumbra+=expected>.05 && expected<.95; seamSoftPixels+=value>.05 && value<.95;
            }
            if (magnitude[2]-magnitude[0]<.3) {
                ++corners; cornerError+=difference;
                cornerPenumbra+=expected>.05 && expected<.95; cornerSoftPixels+=value>.05 && value<.95;
            }
        }
    }}
    saveGray(directory/(name+"-GPU.png"),actual); saveGray(directory/(name+"-CPU.png"),reference);
    saveGray(directory/(name+"-Error.png"),error);
    Json result{{"checked",checked},{"depthErrors",depthErrors},{"meanAbsoluteError",totalError/std::max(checked,1u)},
        {"depthSamples",depthSamples},{"peripheralDepthDifferences",peripheralDepthDifferences},
        {"penumbraSamples",penumbra},{"penumbraMeanAbsoluteError",penumbraError/std::max(penumbra,1u)},
        {"interiorSamples",interior},{"interiorErrorsOverPointOne",interiorErrors},{"softPixels",softPixels},
        {"seamSamples",seams},{"seamMeanAbsoluteError",seamError/std::max(seams,1u)},
        {"cornerSamples",corners},{"cornerMeanAbsoluteError",cornerError/std::max(corners,1u)},
        {"seamPenumbraSamples",seamPenumbra},{"cornerPenumbraSamples",cornerPenumbra},
        {"seamSoftPixels",seamSoftPixels},{"cornerSoftPixels",cornerSoftPixels},
        {"cpuLightSamples",source.size>0 ? 256 : 1},{"pixelStride",stride}};
    Json failures=Json::array();
    // Record all quality gates, then let the test finish its matrix before
    // failing. Receiver depth, cache equality and validation stay immediate.
    if (!room && source.size>0) {
        if (totalError/checked>=.04) { failures.push_back("planar mean absolute error exceeds 0.04"); }
        if ((requirePenumbra && penumbra<=20) || (penumbra>0 && penumbraError/penumbra>=.20)) {
            failures.push_back("planar penumbra error exceeds 0.20 or separated fixture has no penumbra");
        }
        if (double(interiorErrors)/std::max(interior,1u)>=.01) { failures.push_back("planar lit/umbra interior error fraction exceeds 0.01"); }
        if (requirePenumbra && softPixels<=20) { failures.push_back("finite emitter did not produce soft visibility"); }
    }
    result["qualityPassed"]=failures.empty(); result["qualityFailures"]=failures;
    std::ofstream(directory/(name+"-Oracle.json")) << result.dump(2) << '\n';
    if (!depthErrorDetails.empty()) {
        Json triangles=Json::array();
        for (const auto& triangle : fixture.triangles) { triangles.push_back({triangle.a,triangle.b,triangle.c}); }
        std::ofstream(directory/(name+"-DepthErrors.json")) << Json{{"errors",depthErrorDetails},{"triangles",triangles},
            {"eye",{image.camera.eye[0],image.camera.eye[1],image.camera.eye[2]}},
            {"center",{image.camera.center[0],image.camera.center[1],image.camera.center[2]}},
            {"upProjection",{image.camera.upProjection[0],image.camera.upProjection[1],image.camera.upProjection[2],image.camera.upProjection[3]}},
            {"viewport",{image.camera.viewport[0],image.camera.viewport[1],image.camera.viewport[2],image.camera.viewport[3]}},
            {"clipOrtho",{image.camera.clipOrtho[0],image.camera.clipOrtho[1],image.camera.clipOrtho[2],image.camera.clipOrtho[3]}}}.dump(2) << '\n';
    }
    require(checked>100 && depthErrors==0,name+": independent receiver depth mismatch");
    return result;
}

void requireReused(const Snapshot& image, const std::string& name)
{
    const auto& entry=activeEntry(image);
    for (uint32_t face=0; face<entry.faceCount; ++face) {
        require(image.statistics[entry.baseLayer+face][16]==0 && image.statistics[entry.baseLayer+face][21]==0,
            name+": stable cached face redrew depth");
    }
}

Snapshot compareFrame(RenderGraphPreviewRenderer& preview, RenderGraph& graph,
    const std::filesystem::path& directory, const std::string& name, Json& report, bool captureDebug = false)
{
    const auto rendered=preview.render(graph,kWidth,kHeight,"CachedEvidence.color");
    require(rendered,"VSM quality render: "+preview.lastLog());
    require(preview.lastLog().find("Initial material compilation failed")==std::string::npos,
        "Quality scene rendered the error material");
    auto cached=snapshot("Cached",captureDebug),redraw=snapshot("Redraw",captureDebug);
    size_t mismatches=0;
    for (uint32_t layer=0; layer<kLayers; ++layer) {
        for (size_t pixel=0; pixel<kPixels; ++pixel) { mismatches+=cached.shadows[layer][pixel]!=redraw.shadows[layer][pixel]; }
    }
    report["frames"].push_back({{"name",name},{"cacheRedrawMismatches",mismatches},{"statistics",cached.statistics}});
    std::ofstream(directory/"VirtualShadowMapQuality.json") << report.dump(2) << '\n';
    require(mismatches==0 && cached.depth==redraw.depth,name+": cached soft shadows differ from forced redraw");
    const auto& entry=activeEntry(cached);
    for (uint32_t face=0; face<entry.faceCount; ++face) {
        require(cached.statistics[entry.baseLayer+face][9]==0,name+": unresolved receiver/tap query");
        require(cached.statistics[entry.baseLayer+face][5]==0,name+": unexpected caster work overflow");
    }
    return cached;
}

void verifyUnadmittedFrame(RenderGraphPreviewRenderer& preview, RenderGraph& graph,
    const std::filesystem::path& directory, const std::string& name, Json& report)
{
    const auto rendered=preview.render(graph,kWidth,kHeight,"CachedEvidence.color");
    require(rendered,"VSM unadmitted quality render: "+preview.lastLog());
    const auto cached=snapshot("Cached"),redraw=snapshot("Redraw");
    require(cached.shadows==redraw.shadows && cached.depth==redraw.depth,
        name+": unadmitted cache/force outputs differ");
    for (const auto* image : {&cached,&redraw}) {
        require(image->header.admittedCount==0,name+": a shadow source was unexpectedly admitted");
        for (uint32_t layer=0; layer<kLayers; ++layer) {
            require(std::all_of(image->shadows[layer].begin(),image->shadows[layer].end(),
                [](uint8_t value) { return value==255; }),name+": unadmitted layer is not neutral visibility");
            require(std::all_of(image->statistics[layer].begin(),image->statistics[layer].end(),
                [](uint32_t value) { return value==0; }),name+": unadmitted face recorded caster work");
        }
    }
    report["frames"].push_back({{"name",name},{"cacheRedrawMismatches",0},{"admittedCount",0},
        {"allVisibilityNeutral",true},{"allFaceStatisticsZero",true},{"statistics",cached.statistics}});
    std::ofstream(directory/"VirtualShadowMapQuality.json") << report.dump(2) << '\n';
}

void registerQualityEvidence()
{
    static const bool registered = registerRenderGraphPassType("VSMQualityEvidencePass",
        "Test-only filtered VSM publication readback", [] { return std::make_unique<VSMQualityEvidencePass>(evidence); });
    require(registered, "Register VSM quality evidence pass");
}

class VirtualShadowMapQualityTest final : public RHITest {
public:
    VirtualShadowMapQualityTest() { type=RHITestType::Rendering; name="virtual_shadow_map_quality"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory=std::filesystem::absolute(context.outputDirectory)/"VirtualShadowMapQuality";
        Json report{{"scope","P4 deterministic SMRT, independent area-source geometry oracle and cache equivalence"},
            {"passed",false},{"frames",Json::array()},{"planarOracle",Json::array()},
            {"qualityFailures",Json::array()},
            {"settings",{{"width",kWidth},{"height",kHeight},{"pageSize",128},{"virtualPagesPerAxis",128},
                {"clipmapCount",4},{"firstClipmapExtent",4},{"physicalPageBudget",2048},
                {"smrtRayCount",32},{"smrtSamplesPerRay",32},{"smrtMaxRayLength",12},{"receiverFootprint",true},
                {"shadowBiasMetres",.002},{"cpuAreaSamples",256},{"oracleOriginOffset",0},{"triangleEpsilon",1e-7},
                {"receiverCullDisabled",true}}},
            {"thresholds",{{"planarMeanAbsoluteError",.04},{"planarPenumbraMeanAbsoluteError",.20},
                {"interiorFractionOverPointOne",.01},{"planarPointBoundaryMeanAbsoluteError",.20}}}};
        bench::ValidationRecorder validation;
        try {
            std::filesystem::create_directories(directory);
            registerQualityEvidence();
            const auto floor=writeFixture(directory/"Floor",false);
            const auto contact=writeFixture(directory/"Contact",false,.15);
            const auto cornerPlane=writeFixture(directory/"CornerPlane",false,1.5,.5);
            const auto room=writeFixture(directory/"Room",true);
            report["roomGeometry"]={{"authoredTriangles",room.triangles.size()},
                {"publishedPrimitives",room.publishedPrimitives},{"decodedTerminalTriangles",room.terminalTriangles}};
            auto oracle=[&](const Snapshot& image,const Fixture& fixture,const Source& source,
                const std::string& name,bool isRoom,bool requirePenumbra=true) {
                auto metric=measureOracle(image,fixture,source,directory,name,isRoom,requirePenumbra);
                if (!metric["qualityPassed"].get<bool>()) {
                    report["qualityFailures"].push_back({{"case",name},{"failures",metric["qualityFailures"]}});
                }
                return metric;
            };
            for (const std::string kind : {"Directional","Spot","Point"}) {
                uint32_t farPenumbra=0,nearPenumbra=0;
                for (bool nearContact : {false,true}) {
                    const auto& authored=nearContact ? contact : floor;
                    Fixture oracleFixture=authored;
                    const std::string prefix=kind+(nearContact ? "-Contact" : "-Separated");
                    validation.phase(prefix);
                    scene::SceneDocument scene; require(scene.load(authored.source),"Load planar quality fixture");
                    RenderView view; RenderGraphPreviewRenderer preview;
                    configurePreview(preview,scene,view,context,validation);
                    Source source{kind}; source.size=kind=="Directional" ? .08 : .35;
                    publishSource(scene,preview,source);
                    // Resolve the 24 mm directional contact penumbra even at
                    // the receiver-selected mip. The old 16-page axis left a
                    // 31 mm texel there and only looked sharp by requesting an
                    // incorrectly finer mip for the first trace segment.
                    evidence=std::make_shared<EvidenceState>(); auto graph=makeGraph(authored,"smrt",false,128);
                    const auto first=compareFrame(preview,graph,directory,prefix+"-Initial",report);
                    const auto metric=oracle(first,oracleFixture,source,prefix,false,!nearContact);
                    report["planarOracle"].push_back({{"name",prefix},{"metrics",metric}});
                    (nearContact ? nearPenumbra : farPenumbra)=metric["softPixels"].get<uint32_t>();
                    const auto stable=compareFrame(preview,graph,directory,prefix+"-Stable",report);
                    requireReused(stable,prefix+" stable");
                    require(first.shadows==stable.shadows,prefix+": deterministic soft sampling changed between stable frames");
                    if (nearContact) {
                        require(view.setCamera({.eye={.9f,5,0},.center={.9f,0,0},.up={0,0,-1},
                            .fovDegrees=52,.nearPlane=.1f,.farPlane=100,.orthoHeight=.8f,.orthographic=true}),
                            "Resolve contact penumbra in a close-up view");
                        const auto closeup=compareFrame(preview,graph,directory,prefix+"-Closeup",report);
                        const auto closeupMetric=oracle(closeup,oracleFixture,source,prefix+"-Closeup",false);
                        require(closeupMetric["penumbraSamples"].get<uint32_t>()>50,
                            prefix+": close-up lacks independent penumbra sample coverage");
                        report["planarOracle"].push_back({{"name",prefix+"-Closeup"},{"metrics",closeupMetric}});
                        continue;
                    }

                    source.size*=.5; publishSource(scene,preview,source);
                    const auto reduced=compareFrame(preview,graph,directory,prefix+"-HalfRadius",report);
                    const auto& reducedEntry=activeEntry(reduced);
                    for (uint32_t face=0; face<reducedEntry.faceCount; ++face) {
                        require(reduced.statistics[reducedEntry.baseLayer+face][21]==0,
                            prefix+": emitter size edit reset the entire depth cache");
                    }
                    source.size=0; publishSource(scene,preview,source);
                    const auto zero=compareFrame(preview,graph,directory,prefix+"-ZeroRadius",report);
                    require(zero.shadows==snapshot("Hard").shadows,prefix+": zero-size SMRT differs from hard shadow");
                    source.size=kind=="Directional" ? .08 : .35; publishSource(scene,preview,source);
                    (void)compareFrame(preview,graph,directory,prefix+"-RestoreRadius",report);

                    ViewCamera movedCamera{.eye={.15f,5,7},.center={.15f,0,0},.fovDegrees=52,
                        .nearPlane=.1f,.farPlane=100,.orthoHeight=7};
                    require(view.setCamera(movedCamera),"Move quality camera");
                    const auto moved=compareFrame(preview,graph,directory,prefix+"-CameraMove",report);
                    report["planarOracle"].push_back({{"name",prefix+"-CameraMove"},
                        {"metrics",oracle(moved,oracleFixture,source,prefix+"-CameraMove",false)}});
                    if (kind=="Directional") { source.direction={.68,-1,.04}; }
                    else { source.position[0]+=.2; }
                    publishSource(scene,preview,source);
                    const auto movedLight=compareFrame(preview,graph,directory,prefix+"-LightMove",report);
                    report["planarOracle"].push_back({{"name",prefix+"-LightMove"},
                        {"metrics",oracle(movedLight,oracleFixture,source,prefix+"-LightMove",false)}});

                    const auto matrix=scene.nodes()[0].localMatrix;
                    float4x4 translation; translation.SetupByTranslation(float3(.2f,0,.1f));
                    require(scene.setNodeLocalMatrix(0,translation*matrix),"Move quality caster and receiver");
                    for (auto& triangle : oracleFixture.triangles) {
                        triangle.a=add(triangle.a,{.2,0,.1}); triangle.b=add(triangle.b,{.2,0,.1}); triangle.c=add(triangle.c,{.2,0,.1});
                    }
                    const auto movedCaster=compareFrame(preview,graph,directory,prefix+"-CasterMove",report);
                    report["planarOracle"].push_back({{"name",prefix+"-CasterMove"},
                        {"metrics",oracle(movedCaster,oracleFixture,source,prefix+"-CasterMove",false)}});

                    setProperty(graph,"filter","pcf");
                    const auto pcf=compareFrame(preview,graph,directory,prefix+"-PCF",report);
                    const auto pcfPixels=visibility(pcf,oracleFixture,source);
                    saveGray(directory/(prefix+"-PCF.png"),pcfPixels);
                    require(std::any_of(pcfPixels.begin(),pcfPixels.end(),[](uint8_t value) { return value>0 && value<255; }),
                        prefix+": PCF did not filter a shadow edge");
                    setProperty(graph,"filter","smrt"); setProperty(graph,"physicalPageBudget",kind=="Point" ? 6 : 1);
                    const auto coarse=compareFrame(preview,graph,directory,prefix+"-CoarseOnly",report);
                    uint64_t coarseTaps=0;
                    const auto& coarseEntry=activeEntry(coarse);
                    for (uint32_t face=0; face<coarseEntry.faceCount; ++face) { coarseTaps+=coarse.statistics[coarseEntry.baseLayer+face][31]; }
                    require(coarseTaps>0,prefix+": filtered coarse fallback was not observed");
                    setProperty(graph,"physicalPageBudget",2048);
                    (void)compareFrame(preview,graph,directory,prefix+"-PoolRecovery",report);
                    if (kind=="Directional") {
                        // Invalid caster-only settings must not be consumed
                        // when admission leaves no depth work for this pass.
                        setProperty(graph,"enabled",false); setProperty(graph,"tessellationFactor",0);
                        verifyUnadmittedFrame(preview,graph,directory,prefix+"-DisabledSkipsCasters",report);
                        setProperty(graph,"enabled",true); setProperty(graph,"physicalPageBudget",0);
                        verifyUnadmittedFrame(preview,graph,directory,prefix+"-ZeroBudgetSkipsCasters",report);
                        setProperty(graph,"tessellationFactor",4); setProperty(graph,"physicalPageBudget",2048);
                    }
                }
                const bool narrowed=farPenumbra>nearPenumbra*1.3;
                report["contactHardening"][kind]={{"separatedSoftPixels",farPenumbra},{"contactSoftPixels",nearPenumbra},{"passed",narrowed}};
                if (!narrowed) { report["qualityFailures"].push_back({{"case",kind},{"failure","penumbra did not narrow near contact"}}); }
            }

            {
                // Projection ratio 4/(4-1.5)=1.6 places this blocker's shadow
                // at [0.4,2] x [0.4,2] on y=0. With the source at (-2,4,-2),
                // its right/top edges are the X/Y and Z/Y face seams, and the
                // corner (2,0,2) meets +X/-Y/+Z. The penumbra therefore crosses
                // actual shadow edges at the face boundary in a single layer.
                scene::SceneDocument scene; require(scene.load(cornerPlane.source),"Load planar point corner fixture");
                RenderView view; RenderGraphPreviewRenderer preview;
                configurePreview(preview,scene,view,context,validation);
                Source source{"Point",{-2,4,-2},{0,-1,0},.35}; publishSource(scene,preview,source);
                evidence=std::make_shared<EvidenceState>(); auto graph=makeGraph(cornerPlane,"smrt",false,128);
                const std::string name="PointCornerPlane"; validation.phase(name);
                const auto image=compareFrame(preview,graph,directory,name,report);
                const auto metric=oracle(image,cornerPlane,source,name,false);
                report["planarPointBoundaryOracle"]=metric;
                const bool boundarySamples=metric["seamPenumbraSamples"].get<uint32_t>()>0 && metric["cornerPenumbraSamples"].get<uint32_t>()>0 &&
                    metric["seamSoftPixels"].get<uint32_t>()>0 && metric["cornerSoftPixels"].get<uint32_t>()>0;
                const bool boundaryError=metric["seamMeanAbsoluteError"].get<double>()<.20 && metric["cornerMeanAbsoluteError"].get<double>()<.20;
                report["planarPointBoundaryOracle"]["boundaryQualityPassed"]=boundarySamples && boundaryError;
                if (!boundarySamples || !boundaryError) { report["qualityFailures"].push_back({{"case",name},
                    {"failure",!boundarySamples ? "no soft seam/corner samples" : "seam/corner mean absolute error exceeds 0.20"}}); }
                const auto stable=compareFrame(preview,graph,directory,name+"-Stable",report);
                requireReused(stable,name);
                require(image.shadows==stable.shadows,"Planar point seam/corner shadows are not stable");
            }

            // Room covers all twelve two-face seams and all eight three-face
            // corners. Its multi-layer approximation metrics are evidence,
            // while cache/force equality and resolved queries remain strict.
            for (bool stream : {false,true}) {
                scene::SceneDocument scene;
                require(stream ? scene.loadStreamMetadata(room.source) : scene.load(room.source),"Load point corner quality fixture");
                RenderView view; RenderGraphPreviewRenderer preview;
                configurePreview(preview,scene,view,context,validation);
                Source source{"Point",{0,0,0},{0,-1,0},.2}; publishSource(scene,preview,source);
                evidence=std::make_shared<EvidenceState>(); auto graph=makeGraph(room,"smrt",false,128);
                const std::string prefix=stream ? "StreamRoom" : "ResidentRoom";
                uint64_t seamSamples=0,cornerSamples=0;
                uint32_t viewIndex=0;
                auto renderView=[&](Point eye,Point target,Point up) {
                    ViewCamera camera;
                    for (uint32_t axis=0; axis<3; ++axis) {
                        camera.eye[axis]=float(eye[axis]); camera.center[axis]=float(target[axis]); camera.up[axis]=float(up[axis]);
                    }
                    camera.fovDegrees=65; camera.nearPlane=.05f; camera.farPlane=30;
                    require(view.setCamera(camera),"Set point seam/corner view");
                    const std::string name=prefix+"-View"+std::to_string(viewIndex++);
                    validation.phase(name);
                    const auto image=compareFrame(preview,graph,directory,name,report);
                    const auto metric=oracle(image,room,source,name,true);
                    seamSamples+=metric["seamSamples"].get<uint32_t>(); cornerSamples+=metric["cornerSamples"].get<uint32_t>();
                    report["pointBoundaryOracle"].push_back({{"name",name},{"metrics",metric}});
                };
                for (uint32_t first=0; first<3; ++first) { for (uint32_t second=first+1; second<3; ++second) {
                    for (double a : {-1.,1.}) { for (double b : {-1.,1.}) {
                        Point eye{},target{},up{}; const uint32_t third=3-first-second;
                        eye[third]=3; target[first]=6*a; target[second]=6*b; up[third]=1;
                        renderView(eye,target,up);
                    }}
                }}
                for (double x : {-1.,1.}) { for (double y : {-1.,1.}) { for (double z : {-1.,1.}) {
                    renderView({0,0,3*z},{6*x,6*y,6*z},{0,1,0});
                }}}
                require(seamSamples>100 && cornerSamples>20,prefix+": point boundary oracle lacked seam/corner samples");
                report[prefix]={{"seamSamples",seamSamples},{"cornerSamples",cornerSamples},{"views",viewIndex}};
            }
            report["validation"]=validation.snapshot();
            require(!validation.failed(),"VSM quality Vulkan validation reported an error");
            report["qualityPassed"]=report["qualityFailures"].empty();
            require(report["qualityPassed"].get<bool>(),"VSM quality thresholds failed; see the complete qualityFailures matrix");
            report["passed"]=true;
            std::ofstream(directory/"VirtualShadowMapQuality.json") << report.dump(2) << '\n';
            return RHITestResult::pass("VSM SMRT/PCF planar area-light oracle, contact hardening, cache motion and point boundaries");
        } catch (const std::exception& error) {
            report["error"]=error.what(); report["validation"]=validation.snapshot();
            std::ofstream(directory/"VirtualShadowMapQuality.json") << report.dump(2) << '\n';
            return RHITestResult::fail(error.what());
        }
    }
};
class VirtualShadowMapCompactTest final : public RHITest {
public:
    VirtualShadowMapCompactTest() { type = RHITestType::Rendering; name = "virtual_shadow_map_compact"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapCompact";
        Json report{{"passed", false}, {"frames", Json::array()}};
        bench::ValidationRecorder validation;
        try {
            std::filesystem::create_directories(directory);
            registerQualityEvidence();
            const auto floor = writeFixture(directory / "Floor", false);
            const auto room = writeFixture(directory / "Room", true);
            const auto compare = [&](RenderGraphPreviewRenderer& preview, RenderGraph& graph, const std::string& name) {
                validation.phase(name);
                require(preview.render(graph, kWidth, kHeight, "CachedEvidence.color"),
                    "Compact VSM render: " + preview.lastLog());
                const auto legacy = snapshot("Cached", true), compact = snapshot("Compact", true);
                require(legacy.depth == compact.depth, name + ": receiver depth mismatch");
                require(legacy.header.reserved[0] == 0 && compact.header.reserved[0] == 1, name + ": layout flags");
                require(legacy.memory[3] == compact.memory[3] * 6 && compact.memory[1] == 0,
                    name + ": compact memory accounting did not remove face outputs");
                require(legacy.memory[0] == compact.memory[0] && legacy.memory[4] == compact.memory[4] &&
                    legacy.memory[5] == compact.memory[5], name + ": compact projection changed atlas allocation");
                uint64_t visibilityMismatches = 0, debugMismatches = 0;
                for (uint32_t index = 0; index < legacy.header.count; ++index) {
                    const auto& a = legacy.entries[index];
                    const auto& b = compact.entries[index];
                    require(a.source == b.source && a.faceCount == b.faceCount && a.status == b.status &&
                        a.pageBudget == b.pageBudget && b.baseLayer == index, name + ": compact table identity");
                    for (uint32_t y = 0; y < kHeight; ++y) { for (uint32_t x = 0; x < kWidth; ++x) {
                        const auto pixel = size_t(y) * kWidth + x;
                        const uint32_t faceCount = a.kind == 2 && a.status == kVirtualShadowMapLightAdmitted ? 6 : 1;
                        uint8_t expectedShadow = 255;
                        std::array<uint8_t, 4> expectedDebug{};
                        uint32_t coloredFaces = 0;
                        for (uint32_t face = 0; face < faceCount; ++face) {
                            const auto layer = a.baseLayer + face;
                            expectedShadow = std::min(expectedShadow, legacy.shadows[layer][pixel]);
                            coloredFaces += legacy.debug[layer][pixel * 4] || legacy.debug[layer][pixel * 4 + 1] ||
                                legacy.debug[layer][pixel * 4 + 2];
                            for (uint32_t channel = 0; channel < 4; ++channel) {
                                expectedDebug[channel] = std::max(expectedDebug[channel], legacy.debug[layer][pixel * 4 + channel]);
                            }
                        }
                        require(coloredFaces <= 1, name + ": multiple point faces own the same receiver");
                        visibilityMismatches += expectedShadow != compact.shadows[b.baseLayer][pixel];
                        for (uint32_t channel = 0; channel < 4; ++channel) {
                            debugMismatches += expectedDebug[channel] != compact.debug[b.baseLayer][pixel * 4 + channel];
                        }
                    }}
                }
                report["frames"].push_back({{"name", name}, {"visibilityMismatches", visibilityMismatches},
                    {"debugMismatches", debugMismatches}, {"legacyMemory", legacy.memory}, {"compactMemory", compact.memory}});
                require(visibilityMismatches == 0 && debugMismatches == 0, name + ": compact differs from canonical legacy face");
            };
            for (const std::string filter : {"hard", "pcf", "smrt"}) {
                for (const std::string kind : {"Directional", "Spot", "Point"}) {
                    scene::SceneDocument scene; require(scene.load(floor.source), "Load compact VSM fixture");
                    RenderView view; RenderGraphPreviewRenderer preview;
                    configurePreview(preview, scene, view, context, validation);
                    Source source{kind}; source.size = kind == "Directional" ? .08 : .35;
                    publishSource(scene, preview, source);
                    evidence = std::make_shared<EvidenceState>(); auto graph = makeGraph(floor, filter, true);
                    const auto name = filter + "-" + kind;
                    compare(preview, graph, name + "-Cold");
                    compare(preview, graph, name + "-Warm");
                    if (filter == "hard" && kind == "Directional") {
                        for (const char* node : {"Cached", "Compact"}) {
                            require(graph.setNodeRuntimeProperty(graph.findNode(node)->id, "enabled", false), "Disable compact shadows");
                        }
                        compare(preview, graph, name + "-Disabled");
                        for (const char* node : {"Cached", "Compact"}) {
                            require(graph.setNodeRuntimeProperty(graph.findNode(node)->id, "enabled", true), "Enable compact shadows");
                            require(graph.setNodeRuntimeProperty(graph.findNode(node)->id, "physicalPageBudget", 0), "Remove compact budget");
                        }
                        compare(preview, graph, name + "-ZeroBudget");
                    }
                }
            }
            {
                scene::SceneDocument scene; require(scene.loadStreamMetadata(room.source), "Load compact Stream room");
                RenderView view; RenderGraphPreviewRenderer preview;
                configurePreview(preview, scene, view, context, validation);
                Source source{"Point", {0,0,0}, {0,-1,0}, .2}; publishSource(scene, preview, source);
                evidence = std::make_shared<EvidenceState>(); auto graph = makeGraph(room, "smrt", true);
                for (const auto target : {float3(6,6,0), float3(6,6,6)}) {
                    require(view.setCamera({.eye={0,0,3}, .center={target.x,target.y,target.z}, .up={0,0,1},
                        .fovDegrees=65, .nearPlane=.05f, .farPlane=30}), "Set compact seam/corner camera");
                    const auto name = target.z == 0 ? "PointRoom-Seam" : "PointRoom-Corner";
                    compare(preview, graph, std::string(name) + "-Cold");
                    compare(preview, graph, std::string(name) + "-Warm");
                }
            }
            report["validation"] = validation.snapshot();
            require(!validation.failed(), "Compact VSM validation error");
            report["passed"] = true;
            std::ofstream(directory / "VirtualShadowMapCompact.json") << report.dump(2) << '\n';
            return RHITestResult::pass("Compact VSM matches canonical legacy shadow/debug with no private screen outputs");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot();
            std::ofstream(directory / "VirtualShadowMapCompact.json") << report.dump(2) << '\n';
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapCompactTest);

class VirtualShadowMapFilterDemandTest final : public RHITest {
public:
    VirtualShadowMapFilterDemandTest() { type = RHITestType::Rendering; name = "virtual_shadow_map_filter_demand"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapFilterDemand";
        Json report{{"passed", false}, {"scope", "Shared receiver LOD, demand admission and zero-size filter reductions"},
            {"virtualPagesPerAxis", 128}, {"physicalPagesPerFace", 256}, {"orthographicHeight", 4},
            {"frames", Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&]() { std::ofstream(directory / "VirtualShadowMapFilterDemand.json") << report.dump(2) << '\n'; };
        try {
            std::filesystem::create_directories(directory);
            registerQualityEvidence();
            const auto fixture = writeFixture(directory / "Plane", false, 1.5, .5);
            for (const std::string kind : {"Directional", "Spot", "Point"}) {
                scene::SceneDocument scene;
                require(scene.load(fixture.source), "Load demand fixture");
                RenderView view;
                RenderGraphPreviewRenderer preview;
                configurePreview(preview, scene, view, context, validation);
                const float center = kind == "Point" ? 2.0f : 0.0f;
                ViewCamera camera{.eye = {center, 10, center}, .center = {center, 0, center}, .up = {0,0,-1},
                    .fovDegrees = 52, .nearPlane = .1f, .farPlane = 30, .orthoHeight = 4, .orthographic = true};
                require(view.setCamera(camera), "Set orthographic footprint camera");
                Source source{kind, {-2,4,-2}, {.6,-1,0}, kind == "Directional" ? .2 : 1.0};
                publishSource(scene, preview, source);
                evidence = std::make_shared<EvidenceState>();
                auto graph = makeGraph(fixture, "pcf");
                const uint32_t pageBudget = kind == "Point" ? 6 * 256 : 256;
                for (const char* node : {"Cached", "Redraw", "Hard"}) {
                    const auto id = graph.findNode(node)->id;
                    require(graph.setNodeRuntimeProperty(id, "virtualPagesPerAxis", 128), "Set demand virtual resolution");
                    require(graph.setNodeRuntimeProperty(id, "physicalPageBudget", pageBudget), "Set demand physical budget");
                    require(graph.setNodeRuntimeProperty(id, "smrtRayCount", 4), "Set demand ray count");
                    require(graph.setNodeRuntimeProperty(id, "smrtSamplesPerRay", 1), "Set demand segment count");
                    require(graph.setNodeRuntimeProperty(id, "smrtMaxRayLength", 1), "Set demand ray length");
                    require(graph.setNodeRuntimeProperty(graph.findNode(std::string(node) + "Evidence")->id,
                        "captureDebug", true), "Capture demand receiver levels");
                }
                const auto totalStatistics = [](const Snapshot& image) {
                    std::array<uint64_t, 32> totals{};
                    const auto& entry = activeEntry(image);
                    for (uint32_t face = 0; face < entry.faceCount; ++face) {
                        for (size_t index = 0; index < totals.size(); ++index) {
                            totals[index] += image.statistics[entry.baseLayer + face][index];
                        }
                    }
                    return totals;
                };
                const auto render = [&](const std::string& suffix, bool expectCompleteDemand, bool filtered = true,
                    bool expectLastFineLevel = true) {
                    const auto name = kind + "-" + suffix;
                    validation.phase(name);
                    require(preview.render(graph, kWidth, kHeight, "CachedEvidence.color"), "Demand render: " + preview.lastLog());
                    const auto cached = snapshot("Cached", true), redraw = snapshot("Redraw", true), hard = snapshot("Hard", true);
                    const auto totals = totalStatistics(cached), hardTotals = totalStatistics(hard);
                    const bool equal = cached.shadows == redraw.shadows && cached.depth == redraw.depth;
                    const auto& node = *graph.findNode("Cached");
                    auto settings = node.properties;
                    settings.merge_patch(node.runtimeProperties);
                    const auto filter = settings.at("filter").get<std::string>();
                    const bool actualFiltered = filter == "pcf" ? settings.at("pcfRadius").get<float>() > 0 :
                        filter == "smrt" && source.size > 0;
                    require(filtered == actualFiltered, name + ": filter oracle does not match the effective mode");
                    const uint64_t rays = settings.at("smrtRayCount").get<uint32_t>();
                    const uint64_t samples = settings.at("smrtSamplesPerRay").get<uint32_t>() + 1u;
                    Json queryEvidence = Json::array();
                    for (const auto* image : {&cached, &redraw}) {
                        const auto& entry = activeEntry(*image);
                        for (uint32_t face = 0; face < entry.faceCount; ++face) {
                            const auto layer = entry.baseLayer + face;
                            const auto& debugPixels = image->debug[layer];
                            uint64_t receivers = 0;
                            for (size_t pixel = 0; pixel < kPixels; ++pixel) {
                                // Fine green or valid coarse blue reaches filtering;
                                // domain/near-plane rejects stay black, invalid coarse is magenta.
                                receivers += debugPixels[pixel * 4 + 1] != 0 ||
                                    (debugPixels[pixel * 4] == 0 && debugPixels[pixel * 4 + 2] != 0);
                            }
                            const auto& stats = image->statistics[layer];
                            const uint64_t minimum = !actualFiltered ? 0 : receivers * (filter == "pcf" ? 9 : rays);
                            const uint64_t maximum = !actualFiltered ? 0 :
                                receivers * (filter == "pcf" ? 9 : rays * samples);
                            require(stats[30] >= minimum && stats[30] <= maximum,
                                name + ": incorrect filtered query count on receiver face " + std::to_string(face));
                            require(stats[31] <= stats[30], name + ": coarse taps exceed filtered queries");
                            if (image == &cached) {
                                queryEvidence.push_back({{"face", face}, {"receivers", receivers},
                                    {"minimumQueries", minimum}, {"maximumQueries", maximum},
                                    {"shadowSha256", debug::debugSha256(image->shadows[layer])},
                                    {"debugSha256", debug::debugSha256(debugPixels)}});
                            }
                        }
                    }
                    report["frames"].push_back({{"name", name}, {"cacheRedrawEqual", equal},
                        {"statistics", cached.statistics}, {"totals", totals}, {"hardTotals", hardTotals},
                        {"queryEvidence", std::move(queryEvidence)},
                        {"depthSha256", debug::debugSha256(std::span(
                            reinterpret_cast<const uint8_t*>(cached.depth.data()), cached.depth.size() * sizeof(float)))},
                        {"expectCompleteDemand", expectCompleteDemand}, {"filtered", filtered},
                        {"expectLastFineLevel", expectLastFineLevel}});
                    save();
                    require(equal, name + ": cached and forced redraw differ");
                    require(totals[5] == 0 && totals[9] == 0, name + ": raster overflow or unresolved coarse page");
                    if (expectCompleteDemand) {
                        // These orthographic receivers select coarse clipmap/mip
                        // levels. Their useful demand fits the declared pool; an
                        // unconditional level-zero request fills it first instead.
                        require(totals[0] > 0 && totals[0] == totals[1] && totals[2] == 0,
                            name + ": useful filtered demand was not fully admitted");
                        require((!filtered || totals[30] > 0) && totals[31] == 0,
                            name + ": filtered queries unexpectedly fell back to the complete coarse map");
                        require(hardTotals[0] > 0 && hardTotals[0] == hardTotals[1] && hardTotals[2] == 0,
                            name + ": Hard ignored the receiver footprint and exhausted the useful page budget");
                        require(hardTotals[5] == 0 && hardTotals[7] == 0 && hardTotals[9] == 0 && hardTotals[13] == 0,
                            name + ": Hard did not query its admitted receiver level");
                        // At 128 virtual pages per axis and 4 m / 129 pixels,
                        // every visible receiver selects the
                        // last of these four fine levels, including local-light
                        // receivers. Fine debug pixels encode (level + 1) / 4 in
                        // red. Kernel expansion can change requested page sets,
                        // but must not change the receiver's base resolution.
                        for (const auto* image : {&cached, &hard}) {
                            const auto& entry = activeEntry(*image);
                            uint32_t finePixels = 0, wrongLevelPixels = 0;
                            for (uint32_t face = 0; face < entry.faceCount; ++face) {
                                const auto& debug = image->debug[entry.baseLayer + face];
                                for (size_t pixel = 0; pixel < kPixels; ++pixel) {
                                    if (debug[pixel * 4 + 1] == 0) { continue; }
                                    ++finePixels;
                                    wrongLevelPixels += debug[pixel * 4] != 255;
                                }
                            }
                            require(finePixels > 100 && (!expectLastFineLevel || wrongLevelPixels == 0),
                                name + ": receiver resolution changed with the filter");
                        }
                    }
                    if (!filtered) {
                        require(totals[30] == 0 && cached.shadows == hard.shadows && cached.debug == hard.debug,
                            name + ": zero-size filter changed Hard visibility or receiver levels");
                        const auto& entry = activeEntry(cached);
                        for (uint32_t face = 0; face < entry.faceCount; ++face) {
                            for (uint32_t index : {0u, 1u, 2u, 7u, 13u}) {
                                require(cached.statistics[entry.baseLayer + face][index] ==
                                    hard.statistics[activeEntry(hard).baseLayer + face][index],
                                    name + ": zero-size filter changed Hard demand or fallback");
                            }
                        }
                    }
                    return cached;
                };
                const auto pcf = render("PCF-Cold", true);
                requireReused(render("PCF-Warm", true), kind + " PCF reuse");
                saveGray(directory / (kind + "-PCF.png"), visibility(pcf, fixture, source));
                if (kind == "Point") {
                    uint32_t seams = 0, corners = 0;
                    std::array<bool, 6> visibleFaces{};
                    for (uint32_t y = 0; y < kHeight; ++y) { for (uint32_t x = 0; x < kWidth; ++x) {
                        const auto hit = cameraHit(pcf.camera, x, y, fixture);
                        if (!hit.valid || std::abs(hit.position[1]) > .01) { continue; }
                        const auto delta = add(hit.position, scale(source.position, -1));
                        visibleFaces[pointFace(delta)] = true;
                        std::array<double, 3> magnitudes{std::abs(delta[0]), std::abs(delta[1]), std::abs(delta[2])};
                        std::sort(magnitudes.begin(), magnitudes.end());
                        seams += magnitudes[2] - magnitudes[1] < .15;
                        corners += magnitudes[2] - magnitudes[0] < .3;
                    }}
                    const auto faces = std::count(visibleFaces.begin(), visibleFaces.end(), true);
                    report["pointBoundaries"] = {{"seamSamples", seams}, {"cornerSamples", corners}, {"canonicalFaces", faces}};
                    require(seams > 100 && corners > 20 && faces >= 3, "Demand fixture missed point seams/corner");
                }
                camera.eye[0] += .07f; camera.center[0] += .07f;
                require(view.setCamera(camera), "Move demand camera across virtual page boundary");
                (void)render("PCF-Move", true);
                setProperty(graph, "pcfRadius", 0);
                const auto zeroPCF = render("PCF-ZeroRadius", true, false);
                require(zeroPCF.shadows == snapshot("Hard").shadows, kind + ": zero-radius PCF differs from Hard");
                const auto& zeroEntry = activeEntry(zeroPCF);
                for (uint32_t face = 0; face < zeroEntry.faceCount; ++face) {
                    require(zeroPCF.statistics[zeroEntry.baseLayer + face][30] == 0, "Zero-radius PCF still issued filtered taps");
                }
                setProperty(graph, "pcfRadius", 1.5);
                setProperty(graph, "filter", "smrt");
                (void)render("SMRT-Cold", true);
                requireReused(render("SMRT-Warm", true), kind + " SMRT reuse");
                setProperty(graph, "physicalPageBudget", kind == "Point" ? 6 : 1);
                const auto coarse = render("SMRT-CoarseOnly", false);
                const auto& coarseEntry = activeEntry(coarse);
                uint64_t coarseTaps = 0;
                for (uint32_t face = 0; face < coarseEntry.faceCount; ++face) { coarseTaps += coarse.statistics[coarseEntry.baseLayer + face][31]; }
                require(coarseTaps > 0, kind + ": coarse-only pool did not exercise filtered fallback");
                setProperty(graph, "physicalPageBudget", pageBudget);
                (void)render("SMRT-PoolRecovery", true);
                requireReused(render("SMRT-RecoveryWarm", true), kind + " recovered cache reuse");
                // A small source with the normal segment count must retain the
                // receiver's useful mip. Shrinking demand to a ray's first
                // angular stratum requests fine pages that exhaust this pool.
                source.size = kind == "Directional" ? .00465 : .02;
                publishSource(scene, preview, source);
                setProperty(graph, "smrtSamplesPerRay", 16);
                setProperty(graph, "smrtMaxRayLength", 10);
                const auto matchedSMRT = render("SMRT-SmallSource", true);
                requireReused(render("SMRT-SmallSourceWarm", true), kind + " small-source cache reuse");
                source.size = 0; publishSource(scene, preview, source);
                const auto zeroSMRT = render("SMRT-ZeroSource", true, false);
                require(zeroSMRT.shadows == snapshot("Hard").shadows, kind + ": zero-source SMRT differs from Hard");
                for (const char* node : {"Cached", "Redraw", "Hard"}) {
                    require(graph.setNodeRuntimeProperty(graph.findNode(node)->id, "receiverFootprint", false),
                        "Select explicit finest receiver resolution");
                }
                source.size = kind == "Directional" ? .00465 : .02;
                publishSource(scene, preview, source);
                setProperty(graph, "filter", "pcf");
                const auto finestPCF = render("Finest-PCF", false);
                require(totalStatistics(finestPCF)[0] > totalStatistics(pcf)[0],
                    kind + ": disabling footprint matching did not restore finest PCF demand");
                setProperty(graph, "pcfRadius", 0);
                (void)render("Finest-PCF-ZeroRadius", false, false);
                require(totalStatistics(snapshot("Hard"))[0] > totalStatistics(zeroSMRT)[0],
                    kind + ": Hard ignored the explicit finest-resolution setting");
                setProperty(graph, "filter", "smrt");
                const auto finestSMRT = render("Finest-SMRT", false);
                require(totalStatistics(finestSMRT)[0] > totalStatistics(matchedSMRT)[0],
                    kind + ": disabling footprint matching did not restore finest SMRT demand");
                source.size = 0; publishSource(scene, preview, source);
                (void)render("Finest-SMRT-ZeroSource", false, false);
                // Reduce virtual resolution so this same footprint selects an
                // intermediate level instead of clamping to the last fine mip.
                for (const char* node : {"Cached", "Redraw", "Hard"}) {
                    const auto id = graph.findNode(node)->id;
                    require(graph.setNodeRuntimeProperty(id, "virtualPagesPerAxis", 16), "Set intermediate mip resolution");
                    require(graph.setNodeRuntimeProperty(id, "receiverFootprint", true), "Restore receiver footprint matching");
                }
                source.size = kind == "Directional" ? .00465 : .02;
                publishSource(scene, preview, source);
                setProperty(graph, "pcfRadius", 1.5);
                for (const char* filter : {"pcf", "smrt"}) {
                    setProperty(graph, "filter", filter);
                    const auto image = render(std::string("Intermediate-") + filter, true, true, false);
                    const auto hard = snapshot("Hard", true);
                    const auto& entry = activeEntry(image);
                    uint32_t compared = 0, intermediate = 0, mismatches = 0;
                    for (uint32_t face = 0; face < entry.faceCount; ++face) {
                        const auto& actual = image.debug[entry.baseLayer + face];
                        const auto& expected = hard.debug[activeEntry(hard).baseLayer + face];
                        for (size_t pixel = 0; pixel < kPixels; ++pixel) {
                            if (actual[pixel * 4 + 1] == 0 || expected[pixel * 4 + 1] == 0) { continue; }
                            ++compared;
                            intermediate += expected[pixel * 4] < 255;
                            mismatches += actual[pixel * 4] != expected[pixel * 4];
                        }
                    }
                    report["frames"].back()["receiverLevelComparison"] = {{"pixels", compared},
                        {"nonLastLevelPixels", intermediate}, {"mismatches", mismatches}};
                    save();
                    require(compared > 100 && intermediate > 100 && mismatches == 0,
                        kind + ": filtered receiver level differs from Hard away from the last-mip clamp");
                }
            }
            report["validation"] = validation.snapshot();
            require(!validation.failed(), "Filtered demand validation error");
            report["passed"] = true; save();
            return RHITestResult::pass("High-resolution filtered demand fits its pool with no coarse taps; cache, point boundaries and Hard reductions match");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapFilterDemandTest);

class VirtualShadowMapSMRTDistanceTest final : public RHITest {
public:
    VirtualShadowMapSMRTDistanceTest() { type = RHITestType::Rendering; name = "virtual_shadow_map_smrt_distance"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapSMRTDistance";
        Json report{{"passed", false}, {"scope", "SMRT trace distance must not remove farther shadow casters"},
            {"samplesPerRay", 16}, {"cpuAreaSamples", 256}, {"localLightHeight", 20},
            {"cases", Json::array()}, {"failures", Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&]() { std::ofstream(directory / "VirtualShadowMapSMRTDistance.json") << report.dump(2) << '\n'; };
        try {
            std::filesystem::create_directories(directory);
            registerQualityEvidence();
            const auto nearFixture = writeFixture(directory / "NearBlocker", false, 1.5);
            const auto farFixture = writeFixture(directory / "FarBlocker", false, 12);
            for (const auto& fixture : {nearFixture, farFixture}) {
                const bool farBlocker = fixture.source == farFixture.source;
                for (const std::string kind : {"Directional", "Spot", "Point"}) {
                    scene::SceneDocument scene;
                    require(scene.load(fixture.source), "Load SMRT distance fixture");
                    RenderView view;
                    RenderGraphPreviewRenderer preview;
                    configurePreview(preview, scene, view, context, validation);
                    Source source{kind, {0,20,0}, {0,-1,0}, kind == "Directional" ? .00465 : .05};
                    publishSource(scene, preview, source);
                    evidence = std::make_shared<EvidenceState>();
                    auto graph = makeGraph(fixture);
                    setProperty(graph, "smrtSamplesPerRay", 16);
                    const auto runCase = [&](uint32_t rays, uint32_t distance, const char* suffix) {
                        setProperty(graph, "smrtRayCount", rays);
                        setProperty(graph, "smrtMaxRayLength", distance);
                        const std::string name = kind + (farBlocker ? "-Far" : "-Near") +
                            "-Rays" + std::to_string(rays) + "-Distance" + std::to_string(distance) + suffix;
                        validation.phase(name);
                        require(preview.render(graph, kWidth, kHeight, "CachedEvidence.color"),
                            name + ": render failed: " + preview.lastLog());
                        const auto first = snapshot("Cached"), redraw = snapshot("Redraw"), hard = snapshot("Hard");
                        require(first.shadows == redraw.shadows && first.depth == redraw.depth,
                            name + ": cached shadow differs from forced redraw");
                        const auto actual = visibility(first, fixture, source);
                        const auto hardPixels = visibility(hard, fixture, source);
                        uint32_t umbraSamples = 0, litSamples = 0, umbraErrors = 0, litErrors = 0;
                        uint32_t receiverDepthErrors = 0, hardErrors = 0;
                        for (uint32_t y = 0; y < kHeight; ++y) { for (uint32_t x = 0; x < kWidth; ++x) {
                            const auto hit = cameraHit(first.camera, x, y, fixture);
                            if (!hit.valid || std::abs(hit.position[1]) > .01) { continue; }
                            // Interior patches avoid finite-texel silhouette ambiguity.
                            // The independent area-source oracle still checks every
                            // selected receiver against all authored triangles.
                            const bool umbra = std::abs(hit.position[0]) < .35 && std::abs(hit.position[2]) < .35;
                            const bool lit = std::abs(hit.position[0]) > 3 && std::abs(hit.position[0]) < 4 &&
                                std::abs(hit.position[2]) < 1;
                            if (!umbra && !lit) { continue; }
                            const auto pixel = size_t(y) * kWidth + x;
                            const double expected = referenceVisibility(hit, fixture, source);
                            require(expected == (umbra ? 0.0 : 1.0), name + ": authored interior oracle is ambiguous");
                            receiverDepthErrors += std::abs(first.depth[pixel] - hit.depth) > 2e-5;
                            hardErrors += hardPixels[pixel] != (umbra ? 0 : 255);
                            umbraSamples += umbra; litSamples += lit;
                            umbraErrors += umbra && actual[pixel] != 0;
                            litErrors += lit && actual[pixel] != 255;
                        }}
                        require(umbraSamples > 20 && litSamples > 100, name + ": insufficient independent receiver coverage");
                        require(receiverDepthErrors == 0 && hardErrors == 0,
                            name + ": primary geometry or Hard shadow disagrees with the interior oracle");
                        saveGray(directory / (name + "-SMRT.png"), actual);
                        saveGray(directory / (name + "-Hard.png"), hardPixels);
                        require(preview.render(graph, kWidth, kHeight, "CachedEvidence.color"),
                            name + ": stable render failed: " + preview.lastLog());
                        const auto stable = snapshot("Cached");
                        requireReused(stable, name);
                        require(stable.shadows == first.shadows && stable.shadows == snapshot("Redraw").shadows,
                            name + ": stable shadow or redraw changed");
                        const auto& entry = activeEntry(stable);
                        for (uint32_t face = 0; face < entry.faceCount; ++face) {
                            require(stable.statistics[entry.baseLayer + face][5] == 0 &&
                                stable.statistics[entry.baseLayer + face][9] == 0,
                                name + ": raster overflow or unresolved shadow query");
                        }
                        report["cases"].push_back({{"name", name}, {"blockerHeight", farBlocker ? 12.0 : 1.5},
                            {"rayCount", rays}, {"traceDistance", distance}, {"sourceSize", source.size}, {"umbraSamples", umbraSamples},
                            {"litSamples", litSamples}, {"umbraErrors", umbraErrors}, {"litErrors", litErrors},
                            {"receiverDepthErrors", receiverDepthErrors}, {"hardErrors", hardErrors},
                            {"stableCacheAndRedrawEqual", true}, {"statistics", stable.statistics}});
                        if (umbraErrors || litErrors) {
                            report["failures"].push_back({{"name", name}, {"umbraErrors", umbraErrors}, {"litErrors", litErrors}});
                        }
                        save();
                    };
                    for (const uint32_t rays : {1u, 8u}) {
                        for (const uint32_t distance : {10u, 1u}) { runCase(rays, distance, ""); }
                    }
                    if (farBlocker) {
                        // A positive source must exercise SMRT instead of its exact-zero Hard reduction.
                        source.size = 1e-6;
                        publishSource(scene, preview, source);
                        runCase(1, 10, "-TinySource");
                    }
                }
            }
            report["validation"] = validation.snapshot();
            require(!validation.failed(), "SMRT distance Vulkan validation error");
            require(report["failures"].empty(), "SMRT distance lost distant occlusion or shadowed the lit control; see failures");
            report["passed"] = true; save();
            return RHITestResult::pass("Directional, spot and point SMRT retain distant umbra and lit controls with bounded tracing");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapSMRTDistanceTest);

class VirtualShadowMapSMRTPenumbraTest final : public RHITest {
public:
    VirtualShadowMapSMRTPenumbraTest() { type = RHITestType::Rendering; name = "virtual_shadow_map_smrt_penumbra"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapSMRTPenumbra";
        Json report{{"passed", false}, {"scope", "Far-blocker penumbra and spatially unbiased one-ray sampling"},
            {"blockerHeight", 12}, {"blockerHalfExtent", .8}, {"traceDistance", 32}, {"samplesPerRay", 32},
            {"cpuAreaSamples", 256}, {"maximumOneRayBandMeanError", .18},
            {"frames", Json::array()}, {"cases", Json::array()}, {"failures", Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&]() { std::ofstream(directory / "VirtualShadowMapSMRTPenumbra.json") << report.dump(2) << '\n'; };
        try {
            std::filesystem::create_directories(directory);
            registerQualityEvidence();
            const auto fixture = writeFixture(directory / "FarBlocker", false, 12);
            for (const std::string name : {"Directional-Sun", "Directional-Broad", "Spot", "Point"}) {
                const bool directional = name.starts_with("Directional");
                Source source{directional ? "Directional" : name, {0,20,0}, {0,-1,0},
                    directional ? (name == "Directional-Sun" ? .00465 : .04) : .1};
                // View a straight edge from below the elevated blocker. A
                // close orthographic view resolves even a solar-size penumbra.
                const double edge = directional ? .8 : 2.0;
                const double radius = directional ? 12 * std::tan(source.size) : 1.5 * source.size;
                const float height = directional ? (name == "Directional-Sun" ? 1.0f : 2.0f) : 1.5f;
                scene::SceneDocument scene;
                require(scene.load(fixture.source), "Load SMRT penumbra fixture");
                RenderView view;
                RenderGraphPreviewRenderer preview;
                configurePreview(preview, scene, view, context, validation);
                require(view.setCamera({.eye={float(edge),5,0}, .center={float(edge),0,0}, .up={0,0,-1},
                    .fovDegrees=52, .nearPlane=.1f, .farPlane=100, .orthoHeight=height, .orthographic=true}),
                    "Set SMRT penumbra edge camera");
                publishSource(scene, preview, source);
                evidence = std::make_shared<EvidenceState>();
                auto graph = makeGraph(fixture);
                setProperty(graph, "smrtMaxRayLength", 32);
                validation.phase(name + "-ManyRays");
                const auto many = compareFrame(preview, graph, directory, name + "-ManyRays", report);
                const auto metric = measureOracle(many, fixture, source, directory, name + "-ManyRays", false);
                if (!metric.at("qualityPassed").get<bool>()) {
                    report["failures"].push_back({{"name", name}, {"qualityFailures", metric.at("qualityFailures")}});
                }

                setProperty(graph, "smrtRayCount", 1);
                validation.phase(name + "-OneRay");
                const auto one = compareFrame(preview, graph, directory, name + "-OneRay", report);
                const auto pixels = visibility(one, fixture, source);
                saveGray(directory / (name + "-OneRay.png"), pixels);
                const auto stable = compareFrame(preview, graph, directory, name + "-OneRayStable", report);
                requireReused(stable, name + " one-ray stable");
                require(one.shadows == stable.shadows, name + ": fixed-frame spatial samples changed between renders");

                // A single ray is binary; per-pixel soft visibility is not a
                // valid assertion. Independent narrow bands across the edge
                // should nevertheless estimate the area-source visibility.
                // A globally shared disk point makes a shifted hard edge and
                // fails the outer bands even when its full-image mean agrees.
                Json bands = Json::array();
                for (const double offset : {-.5, 0., .5}) {
                    uint32_t samples = 0, depthErrors = 0;
                    double actualSum = 0, expectedSum = 0;
                    for (uint32_t y = 0; y < kHeight; ++y) { for (uint32_t x = 0; x < kWidth; ++x) {
                        const auto hit = cameraHit(one.camera, x, y, fixture);
                        if (!hit.valid || std::abs(hit.position[1]) > .01 || std::abs(hit.position[2]) > .25 ||
                            std::abs((hit.position[0] - edge) / radius - offset) > .15) { continue; }
                        const size_t pixel = size_t(y) * kWidth + x;
                        ++samples;
                        depthErrors += std::abs(one.depth[pixel] - hit.depth) > 2e-5;
                        actualSum += pixels[pixel] / 255.;
                        expectedSum += referenceVisibility(hit, fixture, source);
                    }}
                    require(samples > 50 && depthErrors == 0, name + ": insufficient independent edge-band coverage");
                    const double actual = actualSum / samples, expected = expectedSum / samples;
                    require(expected > .1 && expected < .9, name + ": edge band is not in the physical penumbra");
                    const double error = std::abs(actual - expected);
                    bands.push_back({{"offsetInPenumbraRadii", offset}, {"samples", samples},
                        {"gpuMean", actual}, {"cpuMean", expected}, {"absoluteMeanError", error}});
                    if (error > .18) {
                        report["failures"].push_back({{"name", name}, {"offsetInPenumbraRadii", offset},
                            {"error", "One-ray spatial mean differs from the geometric area-source oracle"},
                            {"absoluteMeanError", error}});
                    }
                }
                report["cases"].push_back({{"name", name}, {"sourceSize", source.size}, {"manyRayMetrics", metric},
                    {"oneRayBands", bands}, {"stableOneRayAndRedrawEqual", true}});
                save();
            }
            report["validation"] = validation.snapshot();
            require(!validation.failed(), "SMRT penumbra Vulkan validation error");
            require(report["failures"].empty(), "SMRT penumbra or spatial sampling disagrees with the independent oracle");
            report["passed"] = true; save();
            return RHITestResult::pass("Far SMRT penumbra matches area-source geometry; one-ray spatial bands remain unbiased");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapSMRTPenumbraTest);

// This oracle accelerates authored double-precision triangles only. Neither
// its bounds nor its leaves depend on a GPU depth sample or VSM projection.
class ReceiverOracle
{
    struct Node {
        Point low{1e30,1e30,1e30}, high{-1e30,-1e30,-1e30};
        uint32_t first = 0, count = 0, left = 0, right = 0;
    };
    const Fixture& fixture_;
    std::vector<uint32_t> indices_;
    std::vector<Node> nodes_;

    uint32_t build(uint32_t first, uint32_t count)
    {
        const uint32_t index = uint32_t(nodes_.size());
        nodes_.emplace_back();
        Node node; node.first = first; node.count = count;
        for (uint32_t i = first; i < first + count; ++i) {
            const auto& triangle = fixture_.triangles[indices_[i]];
            for (Point p : {triangle.a, triangle.b, triangle.c}) {
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    node.low[axis] = std::min(node.low[axis], p[axis]);
                    node.high[axis] = std::max(node.high[axis], p[axis]);
                }
            }
        }
        if (count > 4) {
            uint32_t axis = 0;
            for (uint32_t i = 1; i < 3; ++i) {
                if (node.high[i] - node.low[i] > node.high[axis] - node.low[axis]) { axis = i; }
            }
            const uint32_t middle = first + count / 2;
            std::nth_element(indices_.begin() + first, indices_.begin() + middle, indices_.begin() + first + count,
                [&](uint32_t a, uint32_t b) {
                    const auto& ta = fixture_.triangles[a]; const auto& tb = fixture_.triangles[b];
                    return ta.a[axis] + ta.b[axis] + ta.c[axis] < tb.a[axis] + tb.b[axis] + tb.c[axis];
                });
            node.left = build(first, middle - first); node.right = build(middle, first + count - middle);
            node.count = 0;
        }
        nodes_[index] = node;
        return index;
    }

    bool occluded(Point origin, Point direction, double maximum) const
    {
        // Median splitting gives depth <= ceil(log2(triangle count)); these
        // fixtures contain at most 66 triangles, including the blocker.
        std::array<uint32_t, 64> stack{}; uint32_t size = 1;
        while (size) {
            const auto& node = nodes_[stack[--size]];
            double near = 1e-7, far = maximum - .001;
            bool intersects = true;
            for (uint32_t axis = 0; axis < 3; ++axis) {
                if (std::abs(direction[axis]) < 1e-12) {
                    intersects &= origin[axis] >= node.low[axis] - 1e-9 && origin[axis] <= node.high[axis] + 1e-9;
                } else {
                    const double a = (node.low[axis] - 1e-9 - origin[axis]) / direction[axis];
                    const double b = (node.high[axis] + 1e-9 - origin[axis]) / direction[axis];
                    near = std::max(near, std::min(a,b)); far = std::min(far, std::max(a,b));
                }
            }
            if (!intersects || far < near) { continue; }
            if (node.count) {
                for (uint32_t i = node.first; i < node.first + node.count; ++i) {
                    if (triangleDistance(origin, direction, fixture_.triangles[indices_[i]]) < maximum - .001) { return true; }
                }
            } else { stack[size++] = node.left; stack[size++] = node.right; }
        }
        return false;
    }
public:
    explicit ReceiverOracle(const Fixture& fixture) : fixture_(fixture)
    {
        for (uint32_t i = 0; i < fixture.triangles.size(); ++i) { indices_.push_back(i); }
        build(0, uint32_t(indices_.size()));
    }
    double visibility(const Hit& hit, const Source& source) const
    {
        const Point toLight = source.kind == "Directional" ? normalized(scale(source.direction,-1)) :
            normalized(add(source.position,scale(hit.position,-1)));
        const Point tangent = normalized(cross(toLight,std::abs(toLight[1]) < .9 ? Point{0,1,0} : Point{1,0,0}));
        const Point bitangent = cross(toLight,tangent);
        const uint32_t samples = source.size > 0 ? 256 : 1;
        uint32_t visible = 0;
        for (uint32_t i = 0; i < samples; ++i) {
            const double radius = std::sqrt((i / 16 + .5) / 16.);
            const double angle = (i % 16 + .5) * 2 * kPi / 16;
            const Point disk = add(scale(tangent,radius * std::cos(angle)),scale(bitangent,radius * std::sin(angle)));
            Point direction; double maximum;
            if (source.kind == "Directional") {
                direction = normalized(add(toLight,scale(disk,std::tan(source.size)))); maximum = 100;
            } else {
                const Point ray = add(add(source.position,scale(disk,source.size)),scale(hit.position,-1));
                maximum = std::sqrt(dot(ray,ray)); direction = scale(ray,1 / maximum);
            }
            visible += !occluded(hit.position,direction,maximum);
        }
        return double(visible) / samples;
    }
};

constexpr double kReceiverBlockerHalfExtent = .6;

struct ReceiverFixture {
    Fixture geometry;
    size_t receiverTriangles = 0;
    Point tangent{1,0,0}, normal{0,1,0}, toLight{};
    double blockerDistance = 0, viewHeight = 1.4;
    bool grazing = false, planar = false;
};

ReceiverFixture writeReceiverFixture(const std::filesystem::path& directory, const std::string& shape, double gap)
{
    ReceiverFixture result; result.blockerDistance = gap; result.viewHeight = gap == .15 ? .4 : 1.4;
    result.grazing = shape == "Grazing";
    const bool curved = shape == "Convex" || shape == "Concave";
    result.planar = !curved;
    if (!curved) {
        result.tangent = {std::sqrt(.75),.5,0}; result.normal = {-.5,std::sqrt(.75),0};
        addFixtureQuad(result.geometry,{0,0,0},scale(result.tangent,8),{0,0,-8});
    } else {
        constexpr double radius = 3, halfWidth = 2.4;
        const double sign = shape == "Convex" ? 1 : -1;
        for (uint32_t segment = 0; segment < 32; ++segment) {
            const double x0 = -halfWidth + 2 * halfWidth * segment / 32;
            const double x1 = -halfWidth + 2 * halfWidth * (segment + 1) / 32;
            const double y0 = sign * (std::sqrt(radius * radius - x0 * x0) - radius);
            const double y1 = sign * (std::sqrt(radius * radius - x1 * x1) - radius);
            const Point a{x0,y0,4}, b{x1,y1,4}, c{x1,y1,-4}, d{x0,y0,-4};
            result.geometry.triangles.push_back({a,b,c}); result.geometry.triangles.push_back({a,c,d});
        }
    }
    result.receiverTriangles = result.geometry.triangles.size();
    const double cosine = result.grazing ? .1 : .8;
    result.toLight = add(scale(result.normal,cosine),scale(result.tangent,std::sqrt(1 - cosine * cosine)));
    if (gap > 0) {
        addFixtureQuad(result.geometry,scale(result.toLight,gap),scale(result.tangent,kReceiverBlockerHalfExtent),{0,0,-1.5});
    }
    result.geometry = writeFixture(directory,false,1.5,.8,&result.geometry);
    return result;
}

struct ReceiverReference {
    std::vector<double> area, hard, ndotl;
    std::vector<uint8_t> mask, faces;
    uint32_t depthSamples = 0, depthErrors = 0;
    bool hasAnalyticEdge = false;
    double analyticEdgePixels = 0;
    Json analyticEdgeValidation;
};

ReceiverReference receiverReference(const Snapshot& image, const ReceiverFixture& fixture, const Source& source)
{
    ReceiverReference reference;
    reference.area.resize(kPixels,1); reference.hard.resize(kPixels,1);
    reference.ndotl.resize(kPixels); reference.mask.resize(kPixels); reference.faces.resize(kPixels);
    ReceiverOracle oracle(fixture.geometry);
    Source pointSource = source; pointSource.size = 0;
    if (fixture.planar && fixture.blockerDistance > 0) {
        // The receiver and blocker are parallel planes. Their left-edge
        // projection is u=-halfExtent for a directional source, and
        // u=-halfExtent/(1-gap/sourceDistance) for a finite source. At that
        // point the blocker edge splits the uniform disk through its center,
        // giving exactly 50% visibility independent of a sampled CDF plateau.
        const double distance = std::sqrt(dot(source.position,source.position));
        const double u = -kReceiverBlockerHalfExtent /
            (source.kind == "Directional" ? 1 : 1 - fixture.blockerDistance / distance);
        const Point edgePoint = scale(fixture.tangent,u);
        const Point eye = point(image.camera.eye);
        const Point forward = normalized(add(point(image.camera.center),scale(eye,-1)));
        const Point right = normalized(cross(forward,point(image.camera.upProjection)));
        const double halfWidth = image.camera.clipOrtho[2] * .5 * image.camera.viewport[0];
        reference.analyticEdgePixels = (dot(add(edgePoint,scale(eye,-1)),right) / halfWidth + 1) *
            kWidth * .5 - .5 - image.camera.jitter[0];
        reference.hasAnalyticEdge = true;
        require(reference.analyticEdgePixels >= 0 && reference.analyticEdgePixels < kWidth - 1,
            "Analytic receiver edge lies outside the configured camera");
        Json checks = Json::array();
        constexpr double epsilon = 1e-4;
        for (double z : {-fixture.viewHeight * .35,0.,fixture.viewHeight * .35}) {
            Hit center; center.position = add(edgePoint,{0,0,z}); center.normal = fixture.normal; center.valid = true;
            Hit left = center, right = center;
            left.position = add(center.position,scale(fixture.tangent,-epsilon));
            right.position = add(center.position,scale(fixture.tangent,epsilon));
            const double leftVisibility = referenceVisibility(left,fixture.geometry,pointSource);
            const double rightVisibility = referenceVisibility(right,fixture.geometry,pointSource);
            const double centerAreaVisibility = referenceVisibility(center,fixture.geometry,source);
            checks.push_back({{"z",z},{"litSide",leftVisibility},{"shadowSide",rightVisibility},
                {"diskCenterVisibility",centerAreaVisibility}});
            require(leftVisibility == 1 && rightVisibility == 0 && centerAreaVisibility == .5,
                "Analytic planar edge disagrees with independent authored-triangle rays");
        }
        reference.analyticEdgeValidation = {{"receiverU",u},{"pixel",reference.analyticEdgePixels},
            {"centerRaySideOffsetMetres",epsilon},{"geometryChecks",checks},
            {"reference","Parallel authored planes and a uniform source disk; no GPU visibility input"}};
    }
    for (uint32_t y = 0; y < kHeight; ++y) { for (uint32_t x = 0; x < kWidth; ++x) {
        const size_t pixel = size_t(y) * kWidth + x;
        const auto hit = cameraHit(image.camera,x,y,fixture.geometry);
        if (!hit.valid || hit.triangleIndex >= fixture.receiverTriangles) { continue; }
        const double u = dot(hit.position,fixture.tangent);
        // At NdotL=0.1, the shallow directions from the 8 m local source stretch
        // the separated left penumbra to about u=-1.85. Include a lit interior.
        const double halfWidth = fixture.grazing && fixture.blockerDistance == 1.2 ? 1.35 :
            fixture.blockerDistance == .15 ? .25 : .7;
        if (std::abs(u + .6) > halfWidth || std::abs(hit.position[2]) > fixture.viewHeight * .4) { continue; }
        ++reference.depthSamples;
        reference.depthErrors += std::abs(image.depth[pixel] - hit.depth) > 2e-5;
        const Point toLight = source.kind == "Directional" ? fixture.toLight :
            normalized(add(source.position,scale(hit.position,-1)));
        reference.ndotl[pixel] = dot(hit.normal,toLight);
        // All source directions must lie above the local tangent hemisphere.
        // Outside it, backface irradiance and geometric visibility differ.
        if (reference.ndotl[pixel] <= .06) { continue; }
        reference.mask[pixel] = 1;
        reference.faces[pixel] = source.kind == "Point" ? uint8_t(pointFace(add(hit.position,scale(source.position,-1)))) : 0;
        reference.area[pixel] = oracle.visibility(hit,source);
        reference.hard[pixel] = oracle.visibility(hit,pointSource);
        // Cross-check the independent BVH against the original brute-force
        // triangle oracle on a fixed subset, including finite-source rays.
        if (x % 31 == 0 && y % 29 == 0) {
            require(reference.area[pixel] == referenceVisibility(hit,fixture.geometry,source),
                "Receiver BVH and brute-force authored geometry disagree");
        }
    }}
    return reference;
}

Json receiverMetrics(const std::vector<uint8_t>& actual, const ReceiverReference& reference,
    const std::string& filter, bool blocker, bool grazing)
{
    const auto& expected = filter == "SMRT" ? reference.area : reference.hard;
    uint32_t checked = 0, penumbra = 0, lit = 0, dark = 0, falseShadow = 0, leaks = 0, grazingSamples = 0;
    double error = 0, penumbraError = 0;
    std::array<std::array<uint32_t,4>,4> buckets{};
    std::vector<double> edgeDistances;
    std::vector<std::array<double,2>> discreteOracleEdges;
    for (uint32_t y = 0; y < kHeight; ++y) { for (uint32_t x = 0; x < kWidth; ++x) {
        const size_t pixel = size_t(y) * kWidth + x;
        if (!reference.mask[pixel]) { continue; }
        const double value = actual[pixel] / 255., difference = std::abs(value - expected[pixel]);
        ++checked; error += difference;
        grazingSamples += reference.ndotl[pixel] >= .06 && reference.ndotl[pixel] <= .15;
        if (expected[pixel] > .05 && expected[pixel] < .95) { ++penumbra; penumbraError += difference; }
        // Protect only the independently predicted transition band. PCF's
        // radius is 1 texel; receiver matching bounds its world texel by one
        // camera pixel here. Projecting the square kernel onto the receiver
        // adds at most sqrt(2)/NdotL pixels, plus two raster coverage pixels.
        const uint32_t radius = filter == "PCF" ? uint32_t(std::ceil(std::sqrt(2.) / reference.ndotl[pixel])) + 2 : 1;
        bool stable = expected[pixel] == 0 || expected[pixel] == 1;
        if (x < radius || y < radius || x + radius >= kWidth || y + radius >= kHeight) { stable = false; }
        if (stable) {
            for (int dy = -int(radius); dy <= int(radius) && stable; ++dy) {
                for (int dx = -int(radius); dx <= int(radius); ++dx) {
                    const size_t other = size_t(int(y) + dy) * kWidth + int(x) + dx;
                    if (!reference.mask[other] || expected[other] != expected[pixel]) { stable = false; break; }
                }
            }
        }
        if (stable) {
            const uint32_t bucket = reference.ndotl[pixel] <= .15 ? 0 : reference.ndotl[pixel] <= .35 ? 1 :
                reference.ndotl[pixel] <= .65 ? 2 : 3;
            if (expected[pixel] == 1) {
                ++lit; falseShadow += value < .9; ++buckets[bucket][0]; buckets[bucket][1] += value < .9;
            } else {
                ++dark; leaks += value > .1; ++buckets[bucket][2]; buckets[bucket][3] += value > .1;
            }
        }
        const bool discreteCrossing = blocker && x + 1 < kWidth && reference.mask[pixel + 1] &&
            expected[pixel] >= .5 && expected[pixel + 1] < .5;
        const double discreteEdge = discreteCrossing ?
            x + (expected[pixel] - .5) / (expected[pixel] - expected[pixel + 1]) : 0;
        if (discreteCrossing) { discreteOracleEdges.push_back({double(y),discreteEdge}); }
        const bool edgeRow = reference.hasAnalyticEdge ?
            blocker && x == uint32_t(std::floor(reference.analyticEdgePixels)) && reference.mask[pixel + 1] : discreteCrossing;
        if (edgeRow) {
            const double edge = reference.hasAnalyticEdge ? reference.analyticEdgePixels : discreteEdge;
            double nearest = 16;
            for (int offset = -16; offset <= 16; ++offset) {
                const int xx = int(x) + offset;
                if (xx < 0 || xx + 1 >= int(kWidth)) { continue; }
                const size_t other = size_t(y) * kWidth + xx;
                if (!reference.mask[other] || !reference.mask[other + 1]) { continue; }
                const double a = actual[other] / 255., b = actual[other + 1] / 255.;
                if (a >= .5 && b < .5) {
                    nearest = std::min(nearest,std::abs(xx + (a - .5) / (a - b) - edge));
                }
            }
            edgeDistances.push_back(nearest);
        }
    }}
    std::sort(edgeDistances.begin(),edgeDistances.end());
    const double edge95 = edgeDistances.empty() ? 0 : edgeDistances[size_t((edgeDistances.size() - 1) * .95)];
    Json discreteDifference = Json::array();
    if (reference.hasAnalyticEdge) {
        for (const auto& sample : discreteOracleEdges) { discreteDifference.push_back({sample[0],sample[1] - reference.analyticEdgePixels}); }
    }
    Json failures = Json::array();
    if (checked < 500 || lit < 100 || (blocker && filter != "PCF" && dark < 50)) { failures.push_back("insufficient receiver/interior coverage"); }
    if (grazing && grazingSamples < 500) { failures.push_back("insufficient measured grazing-angle coverage"); }
    if (filter != "PCF" && error / std::max(checked,1u) >= .04) { failures.push_back("receiver MAE >= 0.04"); }
    if (filter == "SMRT" && blocker && (penumbra < 20 || penumbraError / std::max(penumbra,1u) >= .20)) {
        failures.push_back("penumbra coverage < 20 or MAE >= 0.20");
    }
    if (double(falseShadow) / std::max(lit,1u) >= .01) { failures.push_back("false-shadow fraction >= 0.01"); }
    if (double(leaks) / std::max(dark,1u) >= .01) { failures.push_back("light-leak fraction >= 0.01"); }
    for (const auto& bucket : buckets) {
        if ((bucket[0] >= 100 && double(bucket[1]) / bucket[0] >= .01) ||
            (bucket[2] >= 100 && double(bucket[3]) / bucket[2] >= .01)) { failures.push_back("NdotL bucket interior error fraction >= 0.01"); }
    }
    if (blocker && (edgeDistances.size() < 20 || edge95 > 2)) { failures.push_back("edge coverage < 20 or 95th-percentile displacement > 2 pixels"); }
    return {{"checked",checked},{"meanAbsoluteError",error / std::max(checked,1u)},
        {"penumbraSamples",penumbra},{"penumbraMeanAbsoluteError",penumbraError / std::max(penumbra,1u)},
        {"litInteriorSamples",lit},{"umbraInteriorSamples",dark},{"falseShadowSamples",falseShadow},{"lightLeakSamples",leaks},
        {"grazingSamples",grazingSamples},{"NdotLBucketsLitFalseShadowUmbraLeak",buckets},
        {"edgeSamples",edgeDistances.size()},{"edgeDisplacement95Pixels",edge95},
        {"edgeReference",reference.hasAnalyticEdge ? "analytic planar disk median" : "authored-triangle visibility profile"},
        {"analyticEdgePixels",reference.hasAnalyticEdge ? Json(reference.analyticEdgePixels) : Json()},
        {"discreteOracleEdgesByRow",discreteOracleEdges},{"discreteOracleMinusAnalyticByRow",discreteDifference},
        {"qualityPassed",failures.empty()},{"qualityFailures",failures}};
}

Json receiverRowAverageEdge(const std::vector<uint8_t>& actual, const ReceiverReference& reference,
    const std::string& filter)
{
    const auto& expected = filter == "SMRT" ? reference.area : reference.hard;
    std::array<double,kWidth> actualProfile{}, expectedProfile{};
    std::array<uint32_t,kWidth> counts{};
    for (uint32_t y = 0; y < kHeight; ++y) { for (uint32_t x = 0; x < kWidth; ++x) {
        const size_t pixel = size_t(y) * kWidth + x;
        if (!reference.mask[pixel]) { continue; }
        actualProfile[x] += actual[pixel] / 255.; expectedProfile[x] += expected[pixel]; ++counts[x];
    }}
    for (uint32_t x = 0; x < kWidth; ++x) {
        if (counts[x]) { actualProfile[x] /= counts[x]; expectedProfile[x] /= counts[x]; }
    }
    double discreteEdge = -1, actualEdge = -1;
    for (uint32_t x = 0; x + 1 < kWidth; ++x) {
        if (counts[x] < 20 || counts[x + 1] < 20) { continue; }
        if (expectedProfile[x] >= .5 && expectedProfile[x + 1] < .5) {
            discreteEdge = x + (expectedProfile[x] - .5) / (expectedProfile[x] - expectedProfile[x + 1]);
            break;
        }
    }
    const double expectedEdge = reference.hasAnalyticEdge ? reference.analyticEdgePixels : discreteEdge;
    if (expectedEdge >= 0) {
        double nearest = 1e30;
        for (uint32_t x = 0; x + 1 < kWidth; ++x) {
            if (counts[x] < 20 || counts[x + 1] < 20) { continue; }
            if (actualProfile[x] >= .5 && actualProfile[x + 1] < .5) {
                const double edge = x + (actualProfile[x] - .5) / (actualProfile[x] - actualProfile[x + 1]);
                if (std::abs(edge - expectedEdge) < nearest) { nearest = std::abs(edge - expectedEdge); actualEdge = edge; }
            }
        }
    }
    const bool available = expectedEdge >= 0 && actualEdge >= 0;
    // Diagnostic only: averaging rows separates a systematic edge displacement
    // from per-pixel finite-ray noise. It never replaces the existing P95 gate.
    return {{"diagnosticOnly",true},{"available",available},{"columnSamples",counts},
        {"expectedProfile",expectedProfile},{"actualProfile",actualProfile},
        {"expectedEdgePixels",expectedEdge >= 0 ? Json(expectedEdge) : Json()},
        {"discreteOracleEdgePixels",discreteEdge >= 0 ? Json(discreteEdge) : Json()},
        {"analyticEdgePixels",reference.hasAnalyticEdge ? Json(reference.analyticEdgePixels) : Json()},
        {"discreteOracleMinusAnalyticPixels",reference.hasAnalyticEdge && discreteEdge >= 0 ?
            Json(discreteEdge - reference.analyticEdgePixels) : Json()},
        {"actualEdgePixels",actualEdge >= 0 ? Json(actualEdge) : Json()},
        {"signedDisplacementPixels",available ? Json(actualEdge - expectedEdge) : Json()},
        {"absoluteDisplacementPixels",available ? Json(std::abs(actualEdge - expectedEdge)) : Json()}};
}

class VirtualShadowMapReceiverQualityTest final : public RHITest {
public:
    VirtualShadowMapReceiverQualityTest() { type = RHITestType::Rendering; name = "virtual_shadow_map_receiver_quality"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory) / "VirtualShadowMapReceiverQuality";
        Json report{{"passed",false},{"frames",Json::array()},{"cases",Json::array()},{"qualityFailures",Json::array()},
            {"contract",{{"receiverFootprint",true},{"virtualPagesPerAxis",128},{"clipmapCount",4},{"cpuAreaSamples",256},
                {"triangleEpsilonMetres",1e-7},{"oracleOriginOffsetMetres",0},{"planeTiltDegrees",30},
                {"grazingCenterNdotL",.1},{"curveRadiusMetres",3},{"curveSegments",32},
                {"sourceAngularRadius",.04},{"localSourceDistanceMetres",8},{"cameraNormalDistanceMetres",2},
                {"receiverMAE",.04},{"penumbraMAE",.20},
                {"separateInteriorErrorFraction",.01},{"edgeDisplacement95Pixels",2},
                {"minimumOracleNdotL",.06},{"NdotLBucketUpperBounds",{.15,.35,.65,1.0}},
                {"pcfRadiusTexels",1},{"pcfInteriorGuardPixels","ceil(sqrt(2)/NdotL)+2"}}}};
        bench::ValidationRecorder validation;
        const auto save = [&]() { std::ofstream(directory / "VirtualShadowMapReceiverQuality.json") << report.dump(2) << '\n'; };
        try {
            std::filesystem::create_directories(directory); registerQualityEvidence();
            for (const std::string shape : {"Tilted","Grazing","Convex","Concave"}) {
                for (double gap : {0.,.15,1.2}) {
                    const std::string geometryName = shape + (gap == 0 ? "-Clear" : gap == .15 ? "-Contact" : "-Separated");
                    const auto fixture = writeReceiverFixture(directory / geometryName,shape,gap);
                    for (const std::string kind : {"Directional","Spot","Point"}) {
                        const std::string name = geometryName + "-" + kind; validation.phase(name);
                        scene::SceneDocument scene; require(scene.load(fixture.geometry.source),"Load receiver quality geometry");
                        RenderView view; RenderGraphPreviewRenderer preview;
                        configurePreview(preview,scene,view,context,validation);
                        // Translate the grazing separated view without changing
                        // its pixel footprint or the selected shadow resolution.
                        Point center = scale(fixture.tangent,fixture.grazing && gap == 1.2 ? -.95 : -.6);
                        if (shape == "Convex" || shape == "Concave") {
                            center[1] = (shape == "Convex" ? 1 : -1) * (std::sqrt(9 - .36) - 3);
                        }
                        const Point eye = add(center,scale(fixture.normal,2));
                        require(view.setCamera({.eye={float(eye[0]),float(eye[1]),float(eye[2])},
                            .center={float(center[0]),float(center[1]),float(center[2])},.up={0,0,-1},
                            .fovDegrees=52,.nearPlane=.1f,.farPlane=100,.orthoHeight=float(fixture.viewHeight),.orthographic=true}),
                            "Set receiver quality close-up camera");
                        Source source{kind,scale(fixture.toLight,8),scale(fixture.toLight,-1),kind == "Directional" ? .04 : .32};
                        publishSource(scene,preview,source);
                        if (kind == "Spot") {
                            auto lighting = scene.lighting(); auto& light = lighting.lights[0];
                            light.direction = float3(float(source.direction[0]),float(source.direction[1]),float(source.direction[2]));
                            light.properties.outerConeAngle = .6; light.properties.innerConeAngle = .4;
                            require(scene.setLighting(lighting),"Aim receiver spot at the authored surface");
                            require(preview.setLighting(scene.lighting()),"Publish aimed receiver spot");
                        }
                        evidence = std::make_shared<EvidenceState>(); auto graph = makeGraph(fixture.geometry,"smrt",false,128);
                        for (const char* node : {"Cached","Redraw","Hard"}) {
                            require(graph.setNodeRuntimeProperty(graph.findNode(node)->id,"physicalPageBudget",kind == "Point" ? 6 * 1024 : 1024),
                                "Set receiver quality page budget");
                            require(graph.setNodeRuntimeProperty(graph.findNode(node)->id,"smrtMaxRayLength",32),"Set receiver quality trace length");
                            require(graph.setNodeRuntimeProperty(graph.findNode(node)->id,"pcfRadius",1),"Set bounded receiver PCF kernel");
                            require(graph.setNodeRuntimeProperty(graph.findNode(std::string(node) + "Evidence")->id,"captureDebug",true),
                                "Capture receiver quality center levels");
                        }
                        const auto first = compareFrame(preview,graph,directory,name + "-SMRT",report,true);
                        const auto reference = receiverReference(first,fixture,source);
                        require(reference.depthSamples > 500 && reference.depthErrors == 0,name + ": authored receiver depth mismatch");
                        const auto stable = compareFrame(preview,graph,directory,name + "-Warm",report,true);
                        requireReused(stable,name); require(first.shadows == stable.shadows,name + ": stable visibility changed");
                        Json item{{"name",name},{"receiverTriangles",fixture.receiverTriangles},{"blockerDistance",gap},
                            {"viewHeight",fixture.viewHeight},{"depthSamples",reference.depthSamples},{"depthErrors",reference.depthErrors},
                            {"cameraEye",eye},{"cameraCenter",center},{"receiverTangent",fixture.tangent},
                            {"sourcePosition",source.position},{"sourceDirection",source.direction},{"sourceSize",source.size},
                            {"analyticEdgeValidation",reference.analyticEdgeValidation},
                            {"filters",Json::array()}};
                        const auto measure = [&](const Snapshot& image, const std::string& filter) {
                            const auto& entry = activeEntry(image);
                            for (uint32_t face = 0; face < entry.faceCount; ++face) {
                                const auto& stats = image.statistics[entry.baseLayer + face];
                                require(stats[2] == 0 && stats[5] == 0 && stats[7] == 0 && stats[9] == 0 && stats[31] == 0,
                                    name + ": quality fixture has dropped pages, unresolved depth or coarse fallback");
                            }
                            const auto pixels = visibility(image,fixture.geometry,source);
                            auto metric = receiverMetrics(pixels,reference,filter,gap > 0,fixture.grazing);
                            std::array<uint32_t,4> levels{}; uint32_t coarse = 0, other = 0;
                            for (size_t pixel = 0; pixel < kPixels; ++pixel) {
                                if (!reference.mask[pixel]) { continue; }
                                const auto& debug = image.debug[entry.baseLayer + reference.faces[pixel]];
                                if (debug[pixel * 4 + 1] > 100) {
                                    const uint32_t level = (uint32_t(debug[pixel * 4]) * 4 + 127) / 255 - 1;
                                    ++levels[level];
                                } else if (debug[pixel * 4 + 2] > 200 && debug[pixel * 4] < 30) { ++coarse; }
                                else { ++other; }
                            }
                            metric["centerLevelHistogram"] = levels;
                            metric["centerCoarsePixels"] = coarse; metric["centerOtherPixels"] = other;
                            if (fixture.grazing && gap == 1.2) {
                                metric["rowAverageEdgeDiagnostic"] = receiverRowAverageEdge(pixels,reference,filter);
                            }
                            std::vector<uint8_t> cpu(kPixels,255), errors(kPixels,0);
                            const auto& expected = filter == "SMRT" ? reference.area : reference.hard;
                            for (size_t pixel = 0; pixel < kPixels; ++pixel) {
                                if (!reference.mask[pixel]) { continue; }
                                cpu[pixel] = uint8_t(std::lround(expected[pixel] * 255));
                                errors[pixel] = uint8_t(std::lround(std::abs(pixels[pixel] / 255. - expected[pixel]) * 255));
                            }
                            saveGray(directory / (name + "-" + filter + "-GPU.png"),pixels);
                            saveGray(directory / (name + "-" + filter + "-CPU.png"),cpu);
                            saveGray(directory / (name + "-" + filter + "-Error.png"),errors);
                            item["filters"].push_back({{"filter",filter},{"metrics",metric}});
                            if (!metric["qualityPassed"].get<bool>()) {
                                report["qualityFailures"].push_back({{"case",name},{"filter",filter},{"failures",metric["qualityFailures"]}});
                            }
                        };
                        measure(first,"SMRT"); measure(snapshot("Hard",true),"Hard");
                        setProperty(graph,"filter","pcf");
                        measure(compareFrame(preview,graph,directory,name + "-PCF",report,true),"PCF");
                        if (gap > 0) {
                            std::vector<uint8_t> allLit(kPixels,255), allDark(kPixels,0), shifted(kPixels,255);
                            for (uint32_t y = 0; y < kHeight; ++y) { for (uint32_t x = 4; x < kWidth; ++x) {
                                shifted[size_t(y) * kWidth + x] = uint8_t(reference.hard[size_t(y) * kWidth + x - 4] * 255);
                            }}
                            const auto litControl = receiverMetrics(allLit,reference,"Hard",true,fixture.grazing);
                            const auto darkControl = receiverMetrics(allDark,reference,"Hard",true,fixture.grazing);
                            const auto shiftedControl = receiverMetrics(shifted,reference,"Hard",true,fixture.grazing);
                            require(litControl["lightLeakSamples"].get<uint32_t>() > 0 &&
                                darkControl["falseShadowSamples"].get<uint32_t>() > 0 &&
                                shiftedControl["edgeDisplacement95Pixels"].get<double>() > 2,
                                name + ": oracle negative controls did not reject missing, false or shifted shadows");
                            item["negativeControls"] = {{"allLitRejected",!litControl["qualityPassed"].get<bool>()},
                                {"allDarkRejected",!darkControl["qualityPassed"].get<bool>()},
                                {"fourPixelShiftRejected",!shiftedControl["qualityPassed"].get<bool>()}};
                        }
                        report["cases"].push_back(item); save();
                    }
                }
            }
            report["validation"] = validation.snapshot(); save();
            require(!validation.failed(),"Receiver quality Vulkan validation error");
            require(report["qualityFailures"].empty(),"Receiver quality gates failed; see complete per-filter matrix");
            report["passed"] = true; save();
            return RHITestResult::pass("Sloped, grazing, convex and concave receivers match independent geometry with separate leak, acne and edge gates");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(VirtualShadowMapReceiverQualityTest);

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapQualityTest);

} // namespace
} // namespace metallic::tests

