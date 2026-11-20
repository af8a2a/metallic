#include "MeshDistanceFieldFixture.h"
#include "RHITest.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/MeshCards.h"
#include "Runtime/Scene/MeshDistanceField.h"
#include "Runtime/Scene/SceneDocument.h"
#include "harness/ValidationRecorder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <optional>

namespace metallic::tests {
namespace {

constexpr uint32_t kAtlasEdge = 128;
constexpr uint32_t kTileEdge = 16;
constexpr uint32_t kCardsPerBox = 6;
constexpr uint32_t kCardCount = 2 * kCardsPerBox;
constexpr size_t kAtlasPixels = size_t(kAtlasEdge) * kAtlasEdge;

struct CapturedCard {
    std::array<float, 4> origin, axisX, axisY, axisZ;
    std::array<uint32_t, 4> metadata;
};
static_assert(sizeof(CapturedCard) == 80);

template<class Value, class Bytes>
Value readSurfaceValue(const Bytes& bytes, size_t index)
{
    Value value;
    std::memcpy(&value, bytes.data() + index * sizeof(Value), sizeof(Value));
    return value;
}

float halfToFloat(uint16_t value)
{
    const uint32_t exponent = (value >> 10) & 31u;
    const uint32_t mantissa = value & 1023u;
    if (exponent == 31) { return std::numeric_limits<float>::quiet_NaN(); }
    const float magnitude = exponent == 0 ? std::ldexp(float(mantissa), -24) :
        std::ldexp(1.0f + float(mantissa) / 1024.0f, int(exponent) - 15);
    return (value & 0x8000u) != 0 ? -magnitude : magnitude;
}

size_t tileCenter(uint32_t cardIndex)
{
    constexpr uint32_t tileCount = kAtlasEdge / kTileEdge;
    const uint32_t x = (cardIndex % tileCount) * kTileEdge + kTileEdge / 2;
    const uint32_t y = (cardIndex / tileCount) * kTileEdge + kTileEdge / 2;
    return size_t(y) * kAtlasEdge + x;
}

bool writeAnalyticBoxCards(const std::filesystem::path& path, const scene::Scene& document, int32_t meshIndex,
    std::string& log)
{
    scene::MeshDistanceFieldGeometry geometry;
    if (!scene::getMeshDistanceFieldGeometry(document, meshIndex, geometry, log)) { return false; }
    std::array<float, 3> minimum = geometry.positions.front();
    std::array<float, 3> maximum = minimum;
    for (const auto& position : geometry.positions) {
        for (size_t axis = 0; axis < minimum.size(); ++axis) {
            minimum[axis] = std::min(minimum[axis], position[axis]);
            maximum[axis] = std::max(maximum[axis], position[axis]);
        }
    }
    std::array<float, 3> center{}, halfExtent{};
    for (size_t axis = 0; axis < center.size(); ++axis) {
        center[axis] = (minimum[axis] + maximum[axis]) * 0.5f;
        halfExtent[axis] = (maximum[axis] - minimum[axis]) * 0.5f;
    }

    scene::MeshCards field;
    field.sourceFingerprint = geometry.fingerprint;
    field.sourceMeshIndex = meshIndex;
    // The six UE 5.7.4 cube OBBs captured from GenerateCardRepresentationData,
    // scaled along the axes of each analytic test box.
    for (uint32_t direction = 0; direction < kCardsPerBox; ++direction) {
        const uint32_t axis = direction / 2;
        const uint32_t x = axis == 0 ? 1 : 0;
        const uint32_t y = axis == 2 ? 1 : 2;
        const float sign = direction % 2 == 0 ? -1.0f : 1.0f;
        scene::MeshCardBuildRecord card{};
        card.origin = center;
        card.origin[axis] += sign * 0.9f * halfExtent[axis];
        card.axisX[x] = 1.0f;
        card.axisY[y] = 1.0f;
        card.axisZ[axis] = sign;
        card.extent = {halfExtent[x], halfExtent[y], 0.2f * halfExtent[axis]};
        card.axisAlignedDirectionIndex = direction;
        field.cards.push_back(card);
    }
    return scene::saveMeshCards(path, field, log);
}

bool prepareSurfaceFixture(const std::filesystem::path& fixture, const std::filesystem::path& cardPath,
    scene::SceneDocument& document, std::string& log, float backFaceEmission = 0.125f)
{
    const std::filesystem::path root(PROJECT_SOURCE_DIR);
    std::ifstream input(root / "Asset/MeshDistanceField/BoxInstances.gltf");
    if (!input) { log = "Cannot open the Surface Cache box fixture"; return false; }
    auto gltf = nlohmann::json::parse(input);
    gltf["extensionsUsed"] = {"KHR_materials_specular"};
    auto& materials = gltf["materials"] = nlohmann::json::array();
    const std::array materialValues{
        std::array{0.25f, 0.125f}, std::array{0.5625f, 0.5f}, std::array{0.25f, backFaceEmission}};
    for (const auto& [albedo, emissive] : materialValues) {
        materials.push_back({
            {"pbrMetallicRoughness", {{"baseColorFactor", {albedo, albedo, albedo, 1.0f}}, {"metallicFactor", 0.0f}}},
            {"extensions", {{"KHR_materials_specular", {{"specularFactor", 0.0f}}}}},
            {"emissiveFactor", {emissive, emissive, emissive}},
        });
    }
    // Source index order is -Z, +Z, -X, +X, -Y, +Y. Give each Z face its own material.
    constexpr std::array faceMaterials{2, 1, 0, 0, 0, 0};
    auto& accessors = gltf.at("accessors");
    const auto allIndices = accessors.at(1);
    auto& primitives = gltf.at("meshes").at(0).at("primitives");
    primitives.clear();
    for (uint32_t face = 0; face < kCardsPerBox; ++face) {
        auto indices = allIndices;
        indices["byteOffset"] = face * 6 * sizeof(uint16_t);
        indices["count"] = 6;
        indices.erase("min");
        indices.erase("max");
        primitives.push_back({{"attributes", {{"POSITION", 0}}}, {"indices", accessors.size()},
            {"material", faceMaterials[face]}});
        accessors.push_back(std::move(indices));
    }
    std::ofstream output(fixture, std::ios::binary);
    output << gltf;
    output.close();
    if (!output) { log = "Cannot write the Surface Cache material fixture"; return false; }
    if (!document.load(fixture)) { log = document.lastLoadResult().error; return false; }

    return writeAnalyticBoxCards(cardPath, document, 0, log);
}

std::string checkSurfaceCapture(const debug::DebugCapture& capture, uint64_t sceneIdentity)
{
    if (capture.artifacts.size() != 6) { return "Expected four Surface Cache atlases and two buffers"; }
    constexpr std::array layouts{"RGBA8", "RGBA8", "RGBA16F", "RGBA16F"};
    constexpr std::array strides{4u, 4u, 8u, 8u};
    for (size_t i = 0; i < layouts.size(); ++i) {
        const auto& atlas = capture.artifacts[i];
        if (atlas.layout.name != layouts[i] || atlas.metadata.value("width", 0u) != kAtlasEdge ||
            atlas.metadata.value("height", 0u) != kAtlasEdge || atlas.bytes.size() != kAtlasPixels * strides[i]) {
            return "Unexpected Surface Cache atlas layout " + atlas.metadata.dump();
        }
    }
    const auto& cards = capture.artifacts[4].bytes;
    const auto& scene = capture.artifacts[5].bytes;
    if (cards.size() != kCardCount * sizeof(CapturedCard) || scene.size() != 4 * sizeof(uint32_t)) {
        return "Unexpected Surface Cache card or scene buffer size";
    }
    const auto header = readSurfaceValue<std::array<uint32_t, 4>>(scene, 0);
    if (header != std::array{kCardCount, kTileEdge, uint32_t(sceneIdentity), uint32_t(sceneIdentity >> 32u)}) {
        return "Surface Cache scene record did not identify the fixture and twelve cards";
    }

    std::map<uint32_t, uint32_t> instanceDirections;
    uint32_t checkedFrontOrigins = 0;
    for (uint32_t index = 0; index < kCardCount; ++index) {
        const auto card = readSurfaceValue<CapturedCard>(cards, index);
        const uint32_t direction = card.metadata[2];
        const bool front = direction == 5;
        if (card.metadata[1] != 0 || direction >= kCardsPerBox) { return "Invalid Surface Cache card identity"; }
        const uint32_t bit = 1u << direction;
        if (instanceDirections[card.metadata[0]] & bit) { return "Duplicate card direction for one box instance"; }
        instanceDirections[card.metadata[0]] |= bit;
        if (front) {
            const std::array expectedOrigin = card.origin[0] < 0.0f ?
                std::array{-1.05f, 0.0f, 0.45f} : std::array{1.08f, 0.0f, -0.15f + 0.18f * std::sqrt(3.0f)};
            for (size_t axis = 0; axis < expectedOrigin.size(); ++axis) {
                if (std::abs(card.origin[axis] - expectedOrigin[axis]) > 1.0e-5f) {
                    return "Surface Cache did not transform the card with its instance";
                }
            }
            ++checkedFrontOrigins;
        }

        const auto albedo = readSurfaceValue<std::array<uint8_t, 4>>(capture.artifacts[0].bytes, tileCenter(index));
        const auto normal = readSurfaceValue<std::array<uint8_t, 4>>(capture.artifacts[1].bytes, tileCenter(index));
        const auto emission = readSurfaceValue<std::array<uint16_t, 4>>(capture.artifacts[2].bytes, tileCenter(index));
        const auto lighting = readSurfaceValue<std::array<uint16_t, 4>>(capture.artifacts[3].bytes, tileCenter(index));
        const int expectedAlbedo = front ? 191 : 128;
        const float expectedEmission = front ? 0.5f : 0.125f;
        const float normalX = float(normal[0]) * (2.0f / 255.0f) - 1.0f;
        const float normalY = float(normal[1]) * (2.0f / 255.0f) - 1.0f;
        const float normalZ = std::sqrt(std::max(1.0f - normalX * normalX - normalY * normalY, 0.0001f));
        const float noL = std::max(0.0f, card.axisX[2] * normalX + card.axisY[2] * normalY + card.axisZ[2] * normalZ);
        for (size_t channel = 0; channel < 3; ++channel) {
            const float emissionValue = halfToFloat(emission[channel]);
            const float lightingValue = halfToFloat(lighting[channel]);
            const float encoded = float(albedo[channel]) / 255.0f;
            const float expectedLighting = expectedEmission + 2.0f * noL * encoded * encoded / 3.1415926535897932f;
            if (std::abs(int(albedo[channel]) - expectedAlbedo) > 2 ||
                !std::isfinite(emissionValue) || std::abs(emissionValue - expectedEmission) > 0.002f ||
                !std::isfinite(lightingValue) || std::abs(lightingValue - expectedLighting) > 0.003f) {
                return "Unexpected Surface Cache material or light on card " + std::to_string(index) +
                    " (encoded albedo " + std::to_string(albedo[channel]) + ", coverage " + std::to_string(normal[3]) +
                    ", emission " + std::to_string(emissionValue) + ", lighting " + std::to_string(lightingValue) +
                    ", expected lighting " + std::to_string(expectedLighting) + ")";
            }
        }
        if (albedo[3] != 255u || normal[3] != 255u ||
            std::abs(int(normal[0]) - 128) > 2 || std::abs(int(normal[1]) - 128) > 2) {
            return "Opaque off-screen card center did not report coverage or a face-aligned normal";
        }
    }
    if (instanceDirections.size() != 2 || checkedFrontOrigins != 2) {
        return "The translated and mirrored instances were merged";
    }
    for (const auto& [instance, mask] : instanceDirections) {
        if (mask != (1u << kCardsPerBox) - 1) { return "Missing a card direction on an instance"; }
    }
    const auto unusedNormal = readSurfaceValue<std::array<uint8_t, 4>>(capture.artifacts[1].bytes, tileCenter(kCardCount));
    const auto unusedLighting = readSurfaceValue<std::array<uint16_t, 4>>(capture.artifacts[3].bytes, tileCenter(kCardCount));
    if (unusedNormal[3] != 0 || unusedLighting != std::array<uint16_t, 4>{}) {
        return "Empty Surface Cache tile retained coverage or outgoing light";
    }
    return {};
}

struct SurfaceCachePreview {
    SurfaceCachePreview()
    {
        preview.setDebugObserver(&inspector);
    }

    ~SurfaceCachePreview()
    {
        preview.setDebugObserver(nullptr);
    }

    std::shared_ptr<const debug::DebugCapture> captureAfterPass(render::RenderGraph& graph, std::string_view pass,
        debug::DebugValue resources, std::string_view output, std::string& log)
    {
        const auto request = inspector.core().dispatch({{"method", "capture.batch"}, {"params", {
            {"pass", pass}, {"checkpoint", "AfterPass"}, {"resources", std::move(resources)}}}});
        if (request.value("status", "") != "ok") {
            log = request.dump();
            return {};
        }
        if (!preview.render(graph, kAtlasEdge, kAtlasEdge, output)) {
            log = std::string(output) + " readback: " + preview.lastLog();
            return {};
        }
        inspector.poll();
        const auto job = request.at("result").at("job").get<std::string>();
        auto capture = inspector.core().completedCapture(job);
        if (!capture) {
            log = "Capture did not complete: " +
                inspector.core().dispatch({{"method", "jobs.get"}, {"params", {{"job", job}}}}).dump();
        }
        return capture;
    }

    // The inspector drains pending Vulkan captures before the preview destroys its device.
    render::RenderGraphPreviewRenderer preview;
    render::RenderDebugRuntime inspector{{}, render::RenderDebugDetail::Resources};
};

class SurfaceCacheTest final : public RHITest {
public:
    SurfaceCacheTest()
    {
        name = "lumen_surface_cache_capture";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const std::filesystem::path root(PROJECT_SOURCE_DIR);
        auto cardPath = context.outputDirectory / "SurfaceCacheBoxes.meshcards";
        scene::SceneDocument document;
        std::string log;
        if (!prepareSurfaceFixture(context.outputDirectory / "SurfaceCacheBoxes.gltf", cardPath, document, log)) {
            return RHITestResult::fail(log);
        }

        RenderGraph graph;
        if (!loadRenderGraphFromFile(root / "Pipelines/Samples/lumen_surface_cache.metallic_graph.json", graph, log)) {
            return RHITestResult::fail(log);
        }
        const auto* node = graph.findNode("SurfaceCache");
        if (!node || !graph.setNodeProperties(node->id, {{"meshCardPaths", {cardPath.generic_string()}},
                {"atlasSize", kAtlasEdge}, {"tileResolution", kTileEdge}})) {
            return RHITestResult::fail("Cannot configure the Surface Cache sample pass");
        }
        RenderView view;
        ViewCamera camera;
        camera.eye = {0.0f, 0.0f, 10.0f};
        camera.center = {0.0f, 0.0f, 11.0f};
        if (!view.setCamera(camera) || !view.setRenderResolution(kAtlasEdge, kAtlasEdge)) {
            return RHITestResult::fail("Invalid Surface Cache test camera");
        }

        bench::ValidationRecorder validation;
        SurfaceCachePreview capturePreview;
        auto& preview = capturePreview.preview;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        if (!preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink())) {
            return RHITestResult::fail("Surface Cache preview setup: " + preview.lastLog());
        }
        environment::WorldEnvironment world;
        world.sun.enabled = true;
        world.sun.direction = {0.0f, 0.0f, -1.0f};
        world.sun.illuminance = 2.0f;
        if (!preview.setWorldEnvironment(world)) { return RHITestResult::fail("Could not set the card lighting test Sun"); }
        preview.setRawReadbackEnabled(true);
        if (const auto result = preview.render(graph, kAtlasEdge, kAtlasEdge, "SurfaceCache.albedo"); !result) {
            const auto error = "Surface Cache capture: " + preview.lastLog();
            return result.error() == Error::Unsupported ? RHITestResult::skip(error) : RHITestResult::fail(error);
        }
        if (!saveRgba8Png(context.outputDirectory / "LumenSurfaceCacheAlbedo.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), kAtlasEdge, kAtlasEdge, log)) {
            return RHITestResult::fail(log);
        }

        debug::DebugValue resources = debug::DebugValue::array();
        for (const auto* atlas : {"albedo", "normal", "emissive", "lighting"}) {
            resources.push_back({{"id", "SurfaceCache." + std::string(atlas)},
                {"roi", {{"x", 0}, {"y", 0}, {"width", kAtlasEdge}, {"height", kAtlasEdge}}}});
        }
        resources.push_back({{"id", "SurfaceCache.cards"}, {"layout", "u32"},
            {"count", kCardCount * sizeof(CapturedCard) / sizeof(uint32_t)}});
        resources.push_back({{"id", "SurfaceCache.scene"}, {"layout", "u32"}, {"count", 4}});
        const auto capture = capturePreview.captureAfterPass(graph, "SurfaceCache", std::move(resources),
            "SurfaceCache.depth", log);
        if (!capture) { return RHITestResult::fail(log); }
        if (const auto error = checkSurfaceCapture(*capture, document.resourceIdentity()); !error.empty()) {
            return RHITestResult::fail(error);
        }
        const auto& depth = preview.readbackBytes();
        if (preview.readbackFormat() != Format::R16Unorm || depth.size() != kAtlasPixels * sizeof(uint16_t)) {
            return RHITestResult::fail("Expected R16 UNORM card depth");
        }
        for (uint32_t index = 0; index < kCardCount; ++index) {
            const int raw = readSurfaceValue<uint16_t>(depth, tileCenter(index));
            if (std::abs(raw - 16384) > 2) {
                return RHITestResult::fail("Card center depth did not encode a unit-box face: " + std::to_string(raw));
            }
        }
        if (readSurfaceValue<uint16_t>(depth, tileCenter(kCardCount)) != 0xffffu) {
            return RHITestResult::fail("Uncovered Surface Cache depth did not retain the UE sentinel");
        }
        if (validation.failed()) {
            return RHITestResult::fail("Surface Cache Vulkan validation: " + validation.snapshot().dump());
        }
        return RHITestResult::pass("Captured separate primitive materials, opaque depth, and outgoing light for twelve off-screen cards.");
    }
};

METALLIC_REGISTER_RHI_TEST(SurfaceCacheTest);

struct LumenRayResult {
    float hitDistance, surfaceExpansion, weightSum, radianceFactor;
    std::array<float, 4> normal;
    std::array<float, 4> radiance;
    uint32_t geometryClass, clipmap, steps, cacheStatus;
    std::array<uint32_t, 4> candidates;
    uint32_t cardIndex, cardObject;
    float atlasU, atlasV;
};
static_assert(sizeof(LumenRayResult) == 96);

constexpr uint32_t kInvalidRayIndex = ~0u;
constexpr uint32_t kRayOutside = 0;
constexpr uint32_t kRayMiss = 1;
constexpr uint32_t kRayHit = 2;
constexpr uint32_t kRayUnavailable = 3;
constexpr uint32_t kRayPartial = 4;
constexpr uint32_t kRayNoGeometryHit = 0;
constexpr uint32_t kRaySuppressed = 1;
constexpr uint32_t kRayNoCard = 2;
constexpr uint32_t kRaySampled = 3;
constexpr size_t kLumenRayCount = 8;

debug::DebugValue lumenRay(std::array<float, 3> origin, std::array<float, 3> direction, float maxDistance)
{
    return {{"origin", origin}, {"direction", direction}, {"minDistance", 0.0f}, {"maxDistance", maxDistance}};
}

debug::DebugValue describeLumenRay(const LumenRayResult& ray)
{
    return {{"class", ray.geometryClass}, {"clipmap", ray.clipmap}, {"steps", ray.steps},
        {"status", ray.cacheStatus}, {"distance", ray.hitDistance}, {"expansion", ray.surfaceExpansion},
        {"normal", {ray.normal[0], ray.normal[1], ray.normal[2]}}, {"weight", ray.weightSum},
        {"cardWeight", ray.normal[3]}, {"atlasUV", {ray.atlasU, ray.atlasV}}, {"radianceFactor", ray.radianceFactor},
        {"radiance", ray.radiance}, {"candidates", ray.candidates},
        {"card", ray.cardIndex}, {"object", ray.cardObject}};
}

std::string checkLumenHit(const LumenRayResult& ray, uint32_t clipmap, uint32_t object, uint32_t cardDirection,
    const std::vector<uint8_t>& cards, std::array<float, 3> expectedNormal, float expectedDistance,
    float expectedRadiance)
{
    constexpr float voxelSize = 3.0f / 56.0f;
    const float distanceTolerance = 1.5f * std::ldexp(voxelSize, int(clipmap));
    const float expectedExpansion = std::ldexp(voxelSize * 0.5f, int(clipmap));
    const float normalDot = ray.normal[0] * expectedNormal[0] + ray.normal[1] * expectedNormal[1] +
        ray.normal[2] * expectedNormal[2];
    // Face-center rays have four covered, depth-visible taps and one dominant cardinal card.
    // Its weight is cos² of the angle to the quantized GDF normal.
    if (ray.geometryClass != kRayHit || ray.clipmap != clipmap || ray.cacheStatus != kRaySampled ||
        ray.steps == 0 || !(std::abs(ray.hitDistance - expectedDistance) < distanceTolerance) ||
        !(std::abs(ray.surfaceExpansion - expectedExpansion) < 1.0e-6f) || !(normalDot > 0.98f) ||
        !(std::abs(ray.weightSum - 1.0f) < 0.04f) || !(std::abs(ray.normal[3] - 1.0f) < 0.04f) ||
        ray.radianceFactor != 1.0f || ray.candidates[0] != object || ray.cardObject != object ||
        ray.cardIndex >= kCardCount) {
        return "Unexpected Lumen hit: " + describeLumenRay(ray).dump();
    }
    const auto card = readSurfaceValue<CapturedCard>(cards, ray.cardIndex);
    const size_t center = tileCenter(ray.cardIndex);
    const float centerU = float(center % kAtlasEdge) / float(kAtlasEdge);
    const float centerV = float(center / kAtlasEdge) / float(kAtlasEdge);
    constexpr float uvTolerance = 0.125f / kAtlasEdge;
    if (card.metadata[0] != object || card.metadata[2] != cardDirection ||
        !(std::abs(ray.atlasU - centerU) < uvTolerance) || !(std::abs(ray.atlasV - centerV) < uvTolerance) ||
        !std::all_of(ray.radiance.begin(), ray.radiance.begin() + 3, [&](float value) {
            return std::abs(value - expectedRadiance) < 0.002f;
        })) {
        return "Unexpected card direction or scene-linear outgoing radiance: " + describeLumenRay(ray).dump();
    }
    return {};
}

bool emptyLumenSample(const LumenRayResult& ray)
{
    return ray.weightSum == 0.0f && ray.cardIndex == kInvalidRayIndex && ray.cardObject == kInvalidRayIndex &&
        ray.radiance[0] == 0.0f && ray.radiance[1] == 0.0f && ray.radiance[2] == 0.0f;
}

class LumenGlobalTraceTest final : public RHITest {
public:
    LumenGlobalTraceTest()
    {
        name = "lumen_global_tracing_surface_cache";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const std::filesystem::path root(PROJECT_SOURCE_DIR);
        const auto fixturePath = context.outputDirectory / "LumenTraceBoxes.gltf";
        const auto cardPath = context.outputDirectory / "LumenTraceBoxes.meshcards";
        const auto fieldPath = context.outputDirectory / "LumenTraceBoxes.meshdf";
        scene::SceneDocument document;
        std::string log;
        if (!prepareSurfaceFixture(fixturePath, cardPath, document, log, 0.0f) ||
            !writeAnalyticBoxField(fieldPath, document, log)) {
            return RHITestResult::fail(log);
        }

        uint32_t leftObject = kInvalidRayIndex;
        uint32_t rightObject = kInvalidRayIndex;
        for (const auto& node : document.renderNodes()) {
            auto& object = node.worldMatrix.a[12] < 0.0f ? leftObject : rightObject;
            object = uint32_t(node.object);
        }
        if (leftObject == rightObject || leftObject == kInvalidRayIndex || rightObject == kInvalidRayIndex) {
            return RHITestResult::fail("The Lumen ray fixture needs two distinct instances of the same source mesh");
        }

        const float cosAngle = std::sqrt(3.0f) * 0.5f;
        const std::array mirrorNormal{-cosAngle, 0.0f, 0.5f};
        const std::array front{0.0f, 0.0f, 1.0f};
        const std::array back{0.0f, 0.0f, -1.0f};
        // y=0 is a logical page boundary for both camera-centered clipmaps.
        const std::array rays{
            lumenRay({-1.05f, 0.0f, 1.25f}, back, 2.0f),
            lumenRay({0.9f - cosAngle, 0.0f, 0.35f}, {cosAngle, 0.0f, -0.5f}, 1.0f),
            lumenRay({-1.05f, 0.0f, -1.25f}, front, 2.0f),
            lumenRay({-1.05f, 0.0f, 2.5f}, back, 2.5f),
            lumenRay({0.0f, 1.0f, 1.25f}, back, 2.0f),
            lumenRay({3.5f, 0.0f, 1.25f}, back, 2.0f),
            lumenRay({0.0f, 1.0f, 1.25f}, back, 5.0f),
            lumenRay({-1.05f, 0.0f, 0.51f}, back, 1.0f)};
        static_assert(rays.size() == kLumenRayCount);

        RenderGraph graph;
        if (!loadRenderGraphFromFile(root / "Pipelines/Samples/lumen_global_tracing.metallic_graph.json", graph, log)) {
            return RHITestResult::fail(log);
        }
        const auto* fieldNode = graph.findNode("GlobalDistanceFieldPages");
        const auto* surfaceNode = graph.findNode("SurfaceCache");
        const auto* traceNode = graph.findNode("Trace");
        if (!fieldNode || !surfaceNode || !traceNode) {
            return RHITestResult::fail("Missing one of the Lumen global tracing sample passes");
        }
        const auto fieldId = fieldNode->id;
        const auto surfaceId = surfaceNode->id;
        nlohmann::json fieldProperties{{"distanceFieldPaths", {fieldPath.generic_string()}}, {"extent", 1.5f},
            {"resolution", 56}, {"maxPhysicalPages", 1024}, {"quantized", true}, {"clipmapCount", 2},
            {"buildObjectGrid", true}};
        nlohmann::json surfaceProperties{{"meshCardPaths", {cardPath.generic_string()}}, {"atlasSize", kAtlasEdge},
            {"tileResolution", kTileEdge}};
        if (!graph.setNodeProperties(fieldId, fieldProperties) ||
            !graph.setNodeProperties(surfaceId, surfaceProperties) ||
            !graph.setNodeProperties(traceNode->id, {{"rays", rays}})) {
            return RHITestResult::fail("Could not configure Lumen ray fixtures");
        }

        RenderView view;
        ViewCamera camera;
        camera.eye = {0.0f, 0.0f, 0.0f};
        camera.center = {0.0f, 0.0f, 1.0f};
        if (!view.setCamera(camera) || !view.setRenderResolution(kAtlasEdge, kAtlasEdge)) {
            return RHITestResult::fail("Invalid Lumen ray test camera");
        }

        bench::ValidationRecorder validation;
        SurfaceCachePreview capturePreview;
        auto& preview = capturePreview.preview;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        if (!preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink())) {
            return RHITestResult::fail("Lumen global tracing preview setup: " + preview.lastLog());
        }
        environment::WorldEnvironment world;
        world.sun.enabled = false;
        if (!preview.setWorldEnvironment(world)) { return RHITestResult::fail("Could not set the emissive-only world"); }
        preview.setRawReadbackEnabled(true);
        constexpr std::string_view output = "Trace.radiance";
        if (const auto rendered = preview.render(graph, kAtlasEdge, kAtlasEdge, output); !rendered) {
            const auto error = "Lumen global tracing preview: " + preview.lastLog();
            if (rendered.error() == Error::Unsupported &&
                (preview.lastLog().find("requires R8 storage image support") != std::string::npos ||
                    preview.lastLog().find("requires R16 storage image support") != std::string::npos)) {
                return RHITestResult::skip(error);
            }
            return RHITestResult::fail(error);
        }

        const auto surface = capturePreview.captureAfterPass(graph, "SurfaceCache", debug::DebugValue::array({
            {{"id", "SurfaceCache.cards"}, {"layout", "u32"}, {"count", kCardCount * sizeof(CapturedCard) / sizeof(uint32_t)}},
        }), output, log);
        if (!surface) { return RHITestResult::fail(log); }
        if (surface->artifacts.size() != 1 || surface->artifacts[0].bytes.size() != kCardCount * sizeof(CapturedCard)) {
            return RHITestResult::fail("Unexpected Lumen ray fixture card layout");
        }
        const auto& cards = surface->artifacts[0].bytes;
        const auto captureRays = [&](std::array<LumenRayResult, kLumenRayCount>& hits) {
            // Capture requests belong to the currently compiled graph generation.
            if (graph.dirty() && !preview.render(graph, kAtlasEdge, kAtlasEdge, output, false)) {
                log = preview.lastLog();
                return false;
            }
            const auto capture = capturePreview.captureAfterPass(graph, "Trace", debug::DebugValue::array({
                {{"id", "Trace.results"}, {"layout", "u32"}, {"count", sizeof(hits) / sizeof(uint32_t)}},
            }), output, log);
            if (!capture) { return false; }
            if (capture->artifacts.size() != 1 || capture->artifacts[0].bytes.size() != sizeof(hits)) {
                log = "Unexpected fixed-ray result buffer layout";
                return false;
            }
            std::memcpy(hits.data(), capture->artifacts[0].bytes.data(), sizeof(hits));
            return true;
        };

        std::array<LumenRayResult, kLumenRayCount> hits;
        if (!captureRays(hits)) { return RHITestResult::fail(log); }
        for (const auto& error : {
            checkLumenHit(hits[0], 0, leftObject, 5, cards, front, 0.75f, 0.5f),
            checkLumenHit(hits[1], 0, rightObject, 1, cards, mirrorNormal, 0.6f, 0.125f),
            checkLumenHit(hits[2], 0, leftObject, 4, cards, back, 0.75f, 0.0f),
            checkLumenHit(hits[3], 1, leftObject, 5, cards, front, 2.0f, 0.5f),
        }) {
            if (!error.empty()) { return RHITestResult::fail(error); }
        }
        constexpr std::array missClasses{kRayMiss, kRayOutside, kRayPartial};
        for (size_t i = 0; i < missClasses.size(); ++i) {
            const auto& hit = hits[4 + i];
            if (hit.geometryClass != missClasses[i] || hit.hitDistance != -1.0f || hit.clipmap != kInvalidRayIndex ||
                hit.cacheStatus != kRayNoGeometryHit || !emptyLumenSample(hit) ||
                hit.candidates[0] != kInvalidRayIndex) {
                return RHITestResult::fail("Incorrect Lumen miss classification: " + describeLumenRay(hit).dump());
            }
        }
        const auto& nearHit = hits.back();
        if (nearHit.geometryClass != kRayHit || nearHit.cacheStatus != kRaySuppressed ||
            nearHit.radianceFactor != 0.0f || !emptyLumenSample(nearHit)) {
            return RHITestResult::fail("Expected the UE near-origin radiance suppression: " + describeLumenRay(nearHit).dump());
        }
        const auto& pixels = preview.readbackBytes();
        if (preview.readbackFormat() != Format::RGBA16Sfloat || pixels.size() != kLumenRayCount * 4 * sizeof(uint16_t)) {
            return RHITestResult::fail("Expected one RGBA16F radiance texel per fixed ray");
        }
        for (size_t i = 0; i < hits.size(); ++i) {
            const auto rgba = readSurfaceValue<std::array<uint16_t, 4>>(pixels, i);
            for (size_t channel = 0; channel < 3; ++channel) {
                if (!(std::abs(halfToFloat(rgba[channel]) - hits[i].radiance[channel]) < 0.001f)) {
                    return RHITestResult::fail("The trace radiance texture disagrees with the GPU ray result");
                }
            }
        }
        if (!preview.render(graph, kAtlasEdge, kAtlasEdge, "FinalBlit.color")) {
            return RHITestResult::fail("Lumen fixed-ray preview: " + preview.lastLog());
        }
        if (!saveRgba8Png(context.outputDirectory / "LumenGlobalTraceRadiance.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), preview.width(), preview.height(), log)) {
            return RHITestResult::fail(log);
        }

        debug::DebugValue evidence{{"rays", rays}, {"fullBudget", debug::DebugValue::array()}};
        for (const auto& hit : hits) {
            evidence["fullBudget"].push_back(describeLumenRay(hit));
        }
        surfaceProperties["meshCardPaths"] = nlohmann::json::array();
        if (!graph.setNodeProperties(surfaceId, surfaceProperties) || !captureRays(hits)) {
            return RHITestResult::fail("Lumen geometry without cards: " + log);
        }
        if (hits[0].geometryClass != kRayHit || hits[0].cacheStatus != kRayNoCard ||
            hits[0].candidates[0] != leftObject || !emptyLumenSample(hits[0])) {
            return RHITestResult::fail("Missing card was confused with a miss or valid black: " + describeLumenRay(hits[0]).dump());
        }
        evidence["firstRayWithoutCards"] = describeLumenRay(hits[0]);

        fieldProperties["maxPhysicalPages"] = 0;
        surfaceProperties["meshCardPaths"] = {cardPath.generic_string()};
        if (!graph.setNodeProperties(fieldId, fieldProperties) || !graph.setNodeProperties(surfaceId, surfaceProperties) ||
            !captureRays(hits)) {
            return RHITestResult::fail("Lumen zero-budget object grid: " + log);
        }
        if ((hits[0].geometryClass != kRayUnavailable &&
                (hits[0].geometryClass != kRayHit || hits[0].cacheStatus != kRayNoCard)) ||
            hits[0].candidates[0] != kInvalidRayIndex || !emptyLumenSample(hits[0])) {
            return RHITestResult::fail("An overflow page supplied a false usable candidate: " + describeLumenRay(hits[0]).dump());
        }
        evidence["firstRayWithoutResidentPages"] = describeLumenRay(hits[0]);
        if (validation.failed()) {
            return RHITestResult::fail("Lumen global tracing Vulkan validation: " + validation.snapshot().dump());
        }
        const auto evidencePath = context.outputDirectory / "LumenGlobalTraceReadback.json";
        std::ofstream outputFile(evidencePath);
        outputFile << evidence.dump(2) << '\n';
        outputFile.close();
        if (!outputFile) { return RHITestResult::fail("Cannot write " + evidencePath.string()); }
        return RHITestResult::pass(
            "Validated clipmap hits, mirrored instance selection, card radiance, ray coverage and zero-budget pages.");
    }
};

METALLIC_REGISTER_RHI_TEST(LumenGlobalTraceTest);

constexpr uint32_t kSingleBounceEdge = 128;

struct BounceImageStats {
    std::array<double, 3> receiverAverage{};
    float maxRgb = 0.0f;
};

std::optional<BounceImageStats> readSingleBounceStats(const render::RenderGraphPreviewRenderer& preview,
    render::Format format, std::string& log)
{
    using HalfPixel = std::array<uint16_t, 4>;
    using FloatPixel = std::array<float, 4>;
    const bool fullPrecision = format == render::Format::RGBA32Sfloat;
    const size_t pixelStride = fullPrecision ? sizeof(FloatPixel) : sizeof(HalfPixel);
    const auto& bytes = preview.readbackBytes();
    if (preview.readbackFormat() != format ||
        bytes.size() != size_t(kSingleBounceEdge) * kSingleBounceEdge * pixelStride) {
        log = "Unexpected single-bounce evidence format or resolution";
        return std::nullopt;
    }
    BounceImageStats stats;
    // Interior of the receiver on the side under the blocker.
    constexpr uint32_t kStartX = 76, kEndX = 99, kStartY = 34, kEndY = 58;
    constexpr double kPixelCount = (kEndX - kStartX) * (kEndY - kStartY);
    for (uint32_t y = 0; y < kSingleBounceEdge; ++y) {
        for (uint32_t x = 0; x < kSingleBounceEdge; ++x) {
            const size_t index = size_t(y) * kSingleBounceEdge + x;
            FloatPixel pixel;
            if (fullPrecision) {
                pixel = readSurfaceValue<FloatPixel>(bytes, index);
            } else {
                const auto half = readSurfaceValue<HalfPixel>(bytes, index);
                std::transform(half.begin(), half.end(), pixel.begin(), halfToFloat);
            }
            for (size_t channel = 0; channel < 3; ++channel) {
                const float radiance = pixel[channel];
                if (!std::isfinite(radiance) || radiance < 0.0f) {
                    log = "Single-bounce evidence contains invalid radiance";
                    return std::nullopt;
                }
                stats.maxRgb = std::max(stats.maxRgb, radiance);
                if (x >= kStartX && x < kEndX && y >= kStartY && y < kEndY) {
                    stats.receiverAverage[channel] += radiance / kPixelCount;
                }
            }
        }
    }
    return stats;
}

class LumenSingleBounceTest final : public RHITest {
public:
    LumenSingleBounceTest()
    {
        name = "lumen_single_bounce_realtime";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const std::filesystem::path root(PROJECT_SOURCE_DIR);
        const auto asset = root / "Asset/Lumen/SingleBounce.gltf";
        const auto fixture = context.outputDirectory / "LumenSingleBounce.gltf";
        std::error_code fileError;
        std::filesystem::copy_file(asset, fixture, std::filesystem::copy_options::overwrite_existing, fileError);
        if (fileError) { return RHITestResult::fail("Could not copy the single-bounce fixture: " + fileError.message()); }
        scene::SceneDocument document;
        if (!document.load(fixture)) { return RHITestResult::fail(document.lastLoadResult().error); }

        std::string log;
        std::vector<std::string> fieldPaths, cardPaths;
        for (int32_t mesh = 0; mesh < 3; ++mesh) {
            const auto base = "LumenSingleBounce" + std::to_string(mesh);
            const auto field = context.outputDirectory / (base + ".meshdf");
            const auto cards = context.outputDirectory / (base + ".meshcards");
            if (!writeAnalyticBoxField(field, document, log, mesh) || !writeAnalyticBoxCards(cards, document, mesh, log)) {
                return RHITestResult::fail(log);
            }
            fieldPaths.push_back(field.generic_string());
            cardPaths.push_back(cards.generic_string());
        }

        scene::SceneEntity blocker = scene::kNullSceneEntity;
        scene::SceneEntity emitter = scene::kNullSceneEntity;
        int32_t emitterMaterial = -1;
        for (const auto& node : document.renderNodes()) {
            const auto& primitive = document.renderPrimitives()[node.renderPrimitiveIndex];
            if (primitive.meshIndex == 1) {
                emitter = node.object;
                emitterMaterial = node.materialIndex;
            }
            if (primitive.meshIndex == 2) { blocker = node.object; }
        }
        if (blocker == scene::kNullSceneEntity || emitter == scene::kNullSceneEntity || emitterMaterial < 0 ||
            !document.setObjectVisible(blocker, false)) {
            return RHITestResult::fail("Expected separately editable emitter and blocker objects");
        }

        RenderGraph graph;
        if (!loadRenderGraphFromFile(root / "Pipelines/Samples/lumen_single_bounce.metallic_graph.json", graph, log)) {
            return RHITestResult::fail(log);
        }
        const auto* fieldNode = graph.findNode("GlobalDistanceFieldPages");
        const auto* surfaceNode = graph.findNode("SurfaceCache");
        if (!fieldNode || !surfaceNode) {
            return RHITestResult::fail("Missing one of the single-bounce sample producers");
        }
        auto fieldProperties = fieldNode->properties;
        fieldProperties["distanceFieldPaths"] = fieldPaths;
        auto cardProperties = surfaceNode->properties;
        cardProperties["meshCardPaths"] = cardPaths;
        const uint32_t tileResolution = cardProperties.value("tileResolution", kTileEdge);
        if (!graph.setNodeProperties(fieldNode->id, std::move(fieldProperties)) ||
            !graph.setNodeProperties(surfaceNode->id, std::move(cardProperties))) {
            return RHITestResult::fail("Could not bind fixture fields and cards to the single-bounce sample");
        }

        RenderView view;
        ViewCamera camera;
        camera.eye = {0.0f, 0.0f, 3.0f};
        camera.orthographic = true;
        camera.orthoHeight = 1.25f;
        camera.farPlane = 8.0f;
        if (!view.setCamera(camera) || !view.setRenderResolution(kSingleBounceEdge, kSingleBounceEdge)) {
            return RHITestResult::fail("Invalid single-bounce test camera");
        }

        bench::ValidationRecorder validation;
        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        const auto initialized = preview.initialize(context.enableValidation, false, false,
            context.enableValidation, validation.sink());
        if (!initialized) { return RHITestResult::fail("Single-bounce preview setup: " + preview.lastLog()); }
        preview.setEnvironment({.enabled = false, .visible = false});
        if (!preview.setWorldEnvironment(environment::WorldEnvironment{})) {
            return RHITestResult::fail("Could not select an emissive-only world");
        }
        preview.setLighting(document.lighting());
        preview.setRawReadbackEnabled(true);

        const auto sample = [&](std::string_view output,
            Format format = Format::RGBA16Sfloat) -> std::optional<BounceImageStats> {
            if (!preview.render(graph, kSingleBounceEdge, kSingleBounceEdge, output)) {
                log = std::string(output) + ": " + preview.lastLog();
                return std::nullopt;
            }
            return readSingleBounceStats(preview, format, log);
        };
        constexpr std::string_view kIndirect = "Gather.indirectDiffuse";
        const auto baseline = sample(kIndirect);
        if (!baseline) { return RHITestResult::fail(log); }
        const auto directColor = sample("Deferred.color", Format::RGBA32Sfloat);
        if (!directColor) { return RHITestResult::fail(log); }
        const auto composite = sample("Gather.color", Format::RGBA32Sfloat);
        if (!composite) { return RHITestResult::fail(log); }
        if (!savePreview(preview, graph, context.outputDirectory / "LumenSingleBounceOpen.png", log)) {
            return RHITestResult::fail(log);
        }

        const auto emissive = document.materials().at(emitterMaterial);
        auto darkEmitter = emissive;
        darkEmitter.emissiveFactor = float3(0.0f);
        if (!document.setMaterialProperties(emitterMaterial, darkEmitter)) {
            return RHITestResult::fail("Could not extinguish the offscreen emitter");
        }
        const auto dark = sample(kIndirect);
        if (!dark) { return RHITestResult::fail(log); }
        if (!document.setMaterialProperties(emitterMaterial, emissive) ||
            !document.setObjectVisible(blocker, true)) {
            return RHITestResult::fail("Could not restore the emitter and enable the blocker");
        }
        const auto blocked = sample(kIndirect);
        if (!blocked) { return RHITestResult::fail(log); }
        if (!savePreview(preview, graph, context.outputDirectory / "LumenSingleBounceBlocked.png", log)) {
            return RHITestResult::fail(log);
        }

        if (!document.setObjectVisible(emitter, false) || !document.setObjectVisible(blocker, false)) {
            return RHITestResult::fail("Could not isolate the receiver for Sun shadow validation");
        }
        environment::WorldEnvironment world;
        world.sun.enabled = true;
        world.sun.direction = float3(0.0f, -0.8f, -0.6f);
        world.sun.illuminance = 4.0f;
        if (!preview.setWorldEnvironment(world)) { return RHITestResult::fail("Could not set the Sun"); }
        const auto readReceiverCard = [&]() -> std::optional<float> {
            if (!preview.render(graph, kSingleBounceEdge, kSingleBounceEdge, "SurfaceCache.lighting")) {
                log = "Surface Cache receiver lighting: " + preview.lastLog();
                return std::nullopt;
            }
            using Pixel = std::array<uint16_t, 4>;
            constexpr uint32_t kReceiverFrontCard = 5;
            const auto& pixels = preview.readbackBytes();
            const uint32_t tilesPerRow = preview.width() / tileResolution;
            const uint32_t x = (kReceiverFrontCard % tilesPerRow) * tileResolution + tileResolution / 2;
            const uint32_t y = (kReceiverFrontCard / tilesPerRow) * tileResolution + tileResolution / 2;
            if (preview.readbackFormat() != Format::RGBA16Sfloat ||
                pixels.size() != size_t(preview.width()) * preview.height() * sizeof(Pixel)) {
                log = "Expected an RGBA16F Surface Cache lighting atlas";
                return std::nullopt;
            }
            const auto pixel = readSurfaceValue<Pixel>(pixels, size_t(y) * preview.width() + x);
            return halfToFloat(pixel[0]);
        };
        const auto sunOpen = readReceiverCard();
        if (!sunOpen) { return RHITestResult::fail(log); }
        if (!document.setObjectVisible(blocker, true)) {
            return RHITestResult::fail("Could not enable the Sun blocker");
        }
        const auto sunBlocked = readReceiverCard();
        if (!sunBlocked) { return RHITestResult::fail(log); }

        world.sun.enabled = false;
        if (!preview.setWorldEnvironment(world) || !document.setObjectVisible(blocker, false)) {
            return RHITestResult::fail("Could not isolate punctual card lighting");
        }
        scene::LightingSettings punctual = document.lighting();
        scene::PunctualLight light;
        light.properties.intensityUnit = scene::LightUnit::Candela;
        light.properties.intensity = 4.0;
        light.properties.range = 4.0;
        light.position = float3(0.0f, 1.2f, 0.95f);
        light.direction = float3(0.0f, -0.8f, -0.6f);
        punctual.lights = {light};
        preview.setLighting(punctual);
        const auto pointOpen = readReceiverCard();
        if (!pointOpen) { return RHITestResult::fail(log); }
        if (!document.setObjectVisible(blocker, true)) {
            return RHITestResult::fail("Could not enable the point light blocker");
        }
        const auto pointBlocked = readReceiverCard();
        if (!pointBlocked) { return RHITestResult::fail(log); }

        if (!document.setObjectVisible(blocker, false)) {
            return RHITestResult::fail("Could not expose the receiver to the spotlight");
        }
        punctual.lights[0].properties.type = "spot";
        punctual.lights[0].properties.innerConeAngle = 0.15;
        punctual.lights[0].properties.outerConeAngle = 0.35;
        preview.setLighting(punctual);
        const auto spotOnAxis = readReceiverCard();
        if (!spotOnAxis) { return RHITestResult::fail(log); }
        punctual.lights[0].direction = -light.direction;
        preview.setLighting(punctual);
        const auto spotAway = readReceiverCard();
        if (!spotAway) { return RHITestResult::fail(log); }

        const nlohmann::json evidence{
            {"receiverOpen", baseline->receiverAverage}, {"directMaximum", directColor->maxRgb},
            {"receiverComposite", composite->receiverAverage}, {"darkMaximum", dark->maxRgb},
            {"receiverBlocked", blocked->receiverAverage},
            {"sunCardOpen", *sunOpen}, {"sunCardBlocked", *sunBlocked},
            {"pointCardOpen", *pointOpen}, {"pointCardBlocked", *pointBlocked},
            {"spotCardOnAxis", *spotOnAxis}, {"spotCardAway", *spotAway},
        };
        std::ofstream evidenceFile(context.outputDirectory / "LumenSingleBounceReadback.json");
        evidenceFile << evidence.dump(2) << '\n';
        evidenceFile.close();
        if (!evidenceFile) { return RHITestResult::fail("Could not save single-bounce readback"); }

        const auto& rgb = baseline->receiverAverage;
        if (!(rgb[0] > 0.015 && rgb[0] > 1.25 * rgb[1] && rgb[1] > 1.25 * rgb[2] &&
            directColor->maxRgb <= 0.001f && dark->maxRgb <= 0.001f &&
            blocked->receiverAverage[0] < 0.8 * rgb[0])) {
            return RHITestResult::fail("Expected warm offscreen transport, source isolation and blocker occlusion: " +
                evidence.dump());
        }
        for (size_t channel = 0; channel < rgb.size(); ++channel) {
            if (std::abs(rgb[channel] - composite->receiverAverage[channel]) > 0.001) {
                return RHITestResult::fail("Deferred composite did not match the isolated indirect contribution: " +
                    evidence.dump());
            }
        }
        if (!(*sunOpen > 0.1f && *sunBlocked < 0.1f * *sunOpen)) {
            return RHITestResult::fail("The offscreen blocker failed to shadow direct Sun lighting on the receiver card: " +
                evidence.dump());
        }
        if (!(*pointOpen > 0.1f && *pointBlocked < 0.1f * *pointOpen &&
            *spotOnAxis > 0.1f && *spotAway == 0.0f)) {
            return RHITestResult::fail("Expected point shadows and spot cone attenuation on the receiver card: " +
                evidence.dump());
        }
        if (validation.failed()) {
            return RHITestResult::fail("Single-bounce Vulkan validation: " + validation.snapshot().dump());
        }
        return RHITestResult::pass("Offscreen emissive lighting reached the opaque receiver and decreased behind its blocker.");
    }

private:
    static bool savePreview(render::RenderGraphPreviewRenderer& preview, render::RenderGraph& graph,
        const std::filesystem::path& path, std::string& log)
    {
        if (!preview.render(graph, kSingleBounceEdge, kSingleBounceEdge, "FinalBlit.color")) {
            log = preview.lastLog();
            return false;
        }
        return saveRgba8Png(path, reinterpret_cast<const uint8_t*>(preview.pixels().data()),
            preview.width(), preview.height(), log);
    }
};

METALLIC_REGISTER_RHI_TEST(LumenSingleBounceTest);

} // namespace
} // namespace metallic::tests
