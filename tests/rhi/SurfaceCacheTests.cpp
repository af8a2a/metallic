#include "MeshDistanceFieldFixture.h"
#include "RHITest.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Debug/LumenProbeTraceReadback.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/BuiltinPass/LumenScreenProbePlacement.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Scene/MeshCards.h"
#include "Runtime/Scene/MeshDistanceField.h"
#include "Runtime/Scene/SceneDocument.h"
#include "harness/Fixtures.h"
#include "harness/ValidationRecorder.h"

#include <algorithm>
#include <array>
#include <bit>
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

bool loadSingleBounceFixture(RHITestContext& context, scene::SceneDocument& document,
    render::RenderGraph& graph, std::string& log)
{
    const std::filesystem::path root(PROJECT_SOURCE_DIR);
    const auto fixture = context.outputDirectory / "LumenSingleBounce.gltf";
    std::error_code fileError;
    std::filesystem::copy_file(root / "Asset/Lumen/SingleBounce.gltf", fixture,
        std::filesystem::copy_options::overwrite_existing, fileError);
    if (fileError) {
        log = "Could not copy the single-bounce fixture: " + fileError.message();
        return false;
    }
    if (!document.load(fixture)) {
        log = document.lastLoadResult().error;
        return false;
    }
    std::vector<std::string> fieldPaths, cardPaths;
    for (int32_t mesh = 0; mesh < 3; ++mesh) {
        const auto base = "LumenSingleBounce" + std::to_string(mesh);
        const auto field = context.outputDirectory / (base + ".meshdf");
        const auto cards = context.outputDirectory / (base + ".meshcards");
        if (!writeAnalyticBoxField(field, document, log, mesh) || !writeAnalyticBoxCards(cards, document, mesh, log)) {
            return false;
        }
        fieldPaths.push_back(field.generic_string());
        cardPaths.push_back(cards.generic_string());
    }
    if (!loadRenderGraphFromFile(root / "Pipelines/Samples/lumen_single_bounce.metallic_graph.json", graph, log)) {
        return false;
    }
    const auto* fieldNode = graph.findNode("GlobalDistanceFieldPages");
    const auto* surfaceNode = graph.findNode("SurfaceCache");
    if (!fieldNode || !surfaceNode) {
        log = "Missing one of the single-bounce sample producers";
        return false;
    }
    auto fieldProperties = fieldNode->properties;
    fieldProperties["distanceFieldPaths"] = std::move(fieldPaths);
    auto cardProperties = surfaceNode->properties;
    cardProperties["meshCardPaths"] = std::move(cardPaths);
    if (!graph.setNodeProperties(fieldNode->id, std::move(fieldProperties)) ||
        !graph.setNodeProperties(surfaceNode->id, std::move(cardProperties))) {
        log = "Could not bind fixture fields and cards to the single-bounce sample";
        return false;
    }
    return true;
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
        scene::SceneDocument document;
        RenderGraph graph;
        std::string log;
        if (!loadSingleBounceFixture(context, document, graph, log)) { return RHITestResult::fail(log); }

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

        const auto* surfaceNode = graph.findNode("SurfaceCache");
        const uint32_t tileResolution = surfaceNode->properties.value("tileResolution", kTileEdge);

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

constexpr uint32_t kPersistentCardCapacity = (kAtlasEdge / kTileEdge) * (kAtlasEdge / kTileEdge);
constexpr uint32_t kPersistentResidentCards = 3 * kCardsPerBox;
constexpr uint32_t kRadiosityCardBudget = 3;
constexpr std::string_view kPersistentSurface = "PersistentSurface";
constexpr std::string_view kFullSurface = "FullSurface";
constexpr std::string_view kPersistentRadiosity = "PersistentRadiosity";
constexpr std::string_view kDenseRadiosity = "DenseRadiosity";
constexpr uint32_t kCardCaptured = 1u;
constexpr uint32_t kCardRelit = 2u;
constexpr uint32_t kCardLightingChanged = 4u;

constexpr std::array kSurfaceAtlases{
    std::pair{"albedo", 4u}, std::pair{"normal", 4u}, std::pair{"emissive", 8u},
    std::pair{"depth", 2u}, std::pair{"lighting", 8u}, std::pair{"directIrradiance", 16u}};
constexpr size_t kCardRecords = kSurfaceAtlases.size();
constexpr size_t kCardScene = kCardRecords + 1;
constexpr size_t kSurfaceUpdateStats = kCardScene + 1;
constexpr size_t kSurfaceUpdateFlags = kSurfaceUpdateStats + 1;
constexpr size_t kSurfaceLighting = 4;
constexpr size_t kRadiosityUpdateStats = 2;
constexpr size_t kRadiosityCardState = 3;

using CardCapture = std::shared_ptr<const debug::DebugCapture>;

struct CardUpdateFrame {
    CardCapture surface, fullSurface, radiosity, denseRadiosity;
};

class CardUpdateInspector final : public render::IRenderDebugObserver {
public:
    bool detailedSnapshots() const override { return false; }
    void compiled(debug::DebugValue graph) override { runtime.compiled(std::move(graph)); }
    void beginExecution(render::Device& device, debug::DebugEvidenceStamp evidence,
        render::RenderSubsystemHost* subsystems) override
    {
        // Material edits can advance the debug generation just before this
        // boundary. Request the readback for that execution's generation.
        response = runtime.core().dispatch(request);
        runtime.beginExecution(device, std::move(evidence), subsystems);
    }
    void boundary(render::CommandBuffer& commands, std::string_view checkpoint, uint32_t passId,
        std::string_view pass, std::span<const render::DebugResourceBinding> resources,
        const debug::DebugValue& values) override
    {
        runtime.boundary(commands, checkpoint, passId, pass, resources, values);
    }
    void endExecution(bool success) override { runtime.endExecution(success); }

    render::RenderDebugRuntime runtime{{}, render::RenderDebugDetail::Resources};
    debug::DebugValue request, response;
};

debug::DebugValue cardAtlasCapture(std::string id)
{
    return {{"id", std::move(id)}, {"roi", {{"x", 0}, {"y", 0}, {"width", kAtlasEdge}, {"height", kAtlasEdge}}}};
}

debug::DebugValue cardBufferCapture(std::string id, uint32_t count)
{
    return {{"id", std::move(id)}, {"layout", "u32"}, {"count", count}};
}

class LumenProbeDispatchComparison : public RHITest {
protected:
    enum class Variant { ActiveProbes, SplitTracing, SplitDetail };

    LumenProbeDispatchComparison(const char* testName, Variant variant)
        : variant_(variant), candidatePass_(splitTracing() ? "SplitGather" : "ActiveGather")
    {
        name = testName;
        type = RHITestType::Rendering;
    }

public:
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::SceneDocument document;
        RenderGraph graph;
        std::string log;
        if (!loadSingleBounceFixture(context, document, graph, log)) { return RHITestResult::fail(log); }
        if (!prepareGraph(graph)) { return RHITestResult::fail("Could not wire both Gather dispatch paths"); }

        RenderView view;
        ViewCamera camera;
        camera.eye = {0.0f, 0.0f, 3.0f};
        camera.orthographic = true;
        camera.orthoHeight = 1.25f;
        camera.farPlane = 8.0f;
        if (splitTracing()) {
            // Frame the receiver, emitter and blocker together so valid history supplies screen hits.
            camera.eye = {1.1f, 0.72f, 3.0f};
            camera.center = {0.1f, 0.66f, 0.48f};
            camera.fovDegrees = 58.0f;
            camera.orthographic = false;
        }
        if (!view.setCamera(camera) || !view.setRenderResolution(kSingleBounceEdge, kSingleBounceEdge)) {
            return RHITestResult::fail("Invalid active-probe test camera");
        }
        bench::ValidationRecorder validation;
        CardUpdateInspector inspector;
        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        if (!preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink())) {
            return RHITestResult::fail("Active-probe preview setup: " + preview.lastLog());
        }
        preview.setEnvironment({.enabled = false, .visible = false});
        if (!preview.setWorldEnvironment(environment::WorldEnvironment{})) {
            return RHITestResult::fail("Could not select an emissive-only world");
        }
        preview.setLighting(document.lighting());
        preview.setDebugObserver(&inspector);
        preview.setRecordingWorkerLimit(1);
        inspector.request = {{"method", "capture.batch"}, {"params", {{"batches", debug::DebugValue::array({
            {{"pass", "Gather"}, {"resources", captureResources("Gather")}},
            {{"pass", candidatePass_}, {"resources", captureResources(candidatePass_)}},
        })}}}};

        auto emptyCamera = camera;
        emptyCamera.eye[0] += 20.0f;
        emptyCamera.center[0] += 20.0f;
        constexpr size_t kInitial = 0;
        const size_t emptyFrame = splitTracing() ? 4 : 2;
        const size_t restoredFrame = emptyFrame + 1;
        RayHandoffCoverage coverage;
        for (size_t frame = kInitial; frame <= restoredFrame; ++frame) {
            const bool empty = frame == emptyFrame;
            if (frame >= emptyFrame) {
                if (!view.setCamera(empty ? emptyCamera : camera)) { return RHITestResult::fail("Could not change camera"); }
                view.cameraCut();
            }
            if (!preview.render(graph, kSingleBounceEdge, kSingleBounceEdge, "Gather.indirectDiffuse", false)) {
                return RHITestResult::fail(preview.lastLog());
            }
            log = compareFrame(inspector, empty, frame > kInitial && frame < emptyFrame, coverage);
            if (!log.empty()) { return RHITestResult::fail("Frame " + std::to_string(frame) + ": " + log); }
        }
        if (splitTracing() && !coverage.mixedScreenAndGDF) {
            return RHITestResult::fail("Expected both accepted screen rays and compacted GDF rays with valid history");
        }
        if (splitDetail() && (!coverage.detailHit || !coverage.detailMiss)) {
            return RHITestResult::fail("Expected both accepted and rejected Mesh SDF rays");
        }
        if (validation.failed()) {
            return RHITestResult::fail("Probe dispatch Vulkan validation: " + validation.snapshot().dump());
        }
        if (splitDetail()) {
            return RHITestResult::pass("Fused and split Detail matched across Screen, Mesh SDF and compacted GDF");
        }
        if (splitTracing()) {
            return RHITestResult::pass("Fused and split Gather matched with invalid history, mixed tracing and zero rays");
        }
        return RHITestResult::pass("Dense and active Gather matched through repeated frames and zero valid probes");
    }

private:
    struct RayHandoffCoverage {
        bool mixedScreenAndGDF = false;
        bool detailHit = false;
        bool detailMiss = false;
    };

    static constexpr uint32_t kTileAxis = render::builtin_pass::lumenScreenProbeTiles(kSingleBounceEdge);
    static constexpr uint32_t kSlotsPerTile = render::builtin_pass::kLumenScreenProbeSlotsPerTile;
    static constexpr uint32_t kProbeSlots = kTileAxis * kTileAxis * kSlotsPerTile;
    static constexpr uint32_t kRaysPerSlot = 8 * 8;
    static constexpr uint32_t kProbeRays = kProbeSlots * kRaysPerSlot;
    static constexpr uint32_t kSHAtlasEdge = kTileAxis * 3;
    static constexpr std::array<uint32_t, 4> kReadbackROI{1, 1, 6, 6};
    static constexpr std::array<uint32_t, 4> kFullReadbackROI{0, 0, kTileAxis, kTileAxis};
    static constexpr size_t kReadbackStride = sizeof(render::LumenProbeTraceReadbackRecord);
    static constexpr std::array kBufferedResources{
        std::pair{"probePositions", 2 * kProbeSlots}, std::pair{"probeRadiance", 4 * kProbeRays},
        std::pair{"probeDistance", kProbeRays}, std::pair{"compositedProbeDistance", kProbeRays},
        std::pair{"filteredProbeRadiance", 4 * kProbeRays}};
    static constexpr std::array kTextures{
        std::pair{"probeSHAmbient", std::array{kSHAtlasEdge, kSHAtlasEdge, 4u}},
        std::pair{"probeSHDirectional", std::array{kSHAtlasEdge * 6, kSHAtlasEdge, 8u}},
        std::pair{"indirectDiffuse", std::array{kSingleBounceEdge, kSingleBounceEdge, 8u}},
        std::pair{"color", std::array{kSingleBounceEdge, kSingleBounceEdge, 16u}}};

    bool splitTracing() const { return variant_ != Variant::ActiveProbes; }
    bool splitDetail() const { return variant_ == Variant::SplitDetail; }

    uint32_t readbackRayCount() const
    {
        const auto& roi = splitDetail() ? kFullReadbackROI : kReadbackROI;
        return roi[2] * roi[3] * kSlotsPerTile * kRaysPerSlot;
    }

    bool prepareGraph(render::RenderGraph& graph) const
    {
        using render::makeRenderGraphFieldName;
        const auto* gather = graph.findNode("Gather");
        if (!gather) { return false; }
        auto properties = gather->properties;
        properties["finalGather"] = true;
        properties["screenTraces"] = true;
        properties["irradianceFormat"] = "SH3";
        properties["tracingOctahedronResolution"] = 8;
        properties["probeTraceReadback"] = splitDetail() ? kFullReadbackROI : kReadbackROI;
        properties["activeProbeDispatch"] = splitTracing();
        if (splitTracing()) { properties["screenGDFContinuation"] = true; }
        if (splitDetail()) {
            const auto* fields = graph.findNode("GlobalDistanceFieldPages");
            if (!fields || !graph.addEdge("VBuffer.depth", "Gather.detailDepth")) { return false; }
            properties["detailTracing"] = true;
            properties["detailDistanceFieldPaths"] = fields->properties.at("distanceFieldPaths");
        }
        if (!graph.setNodeProperties(gather->id, properties)) { return false; }
        properties["activeProbeDispatch"] = true;
        if (splitTracing()) { properties["splitProbeTracing"] = true; }
        if (!graph.addNode("LumenScreenProbeGatherPass", candidatePass_, properties) ||
            !graph.markOutput(makeRenderGraphFieldName(candidatePass_, "indirectDiffuse")) ||
            !graph.addNode("LumenScreenTraceSetupPass", "ScreenTrace", {{"historySource", "deferred"}}) ||
            !graph.addEdge("VBuffer.depth", "ScreenTrace.depth") || !graph.addEdge("Deferred.color", "ScreenTrace.color")) {
            return false;
        }
        for (const std::string_view field : {"closestHZB", "previousDepth", "previousColor", "traceView"}) {
            if (!graph.addEdge(makeRenderGraphFieldName("ScreenTrace", field), makeRenderGraphFieldName("Gather", field))) {
                return false;
            }
        }
        const auto edges = graph.edges();
        for (const auto& edge : edges) {
            if (edge.dstPass == "Gather" &&
                !graph.addEdge(makeRenderGraphFieldName(edge.srcPass, edge.srcField),
                    makeRenderGraphFieldName(candidatePass_, edge.dstField))) {
                return false;
            }
        }
        return true;
    }

    debug::DebugValue captureResources(std::string_view pass) const
    {
        using render::makeRenderGraphFieldName;
        auto resources = debug::DebugValue::array();
        for (const auto& [field, words] : kBufferedResources) {
            resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, field), words));
        }
        for (const auto& [field, layout] : kTextures) {
            resources.push_back({{"id", makeRenderGraphFieldName(pass, field)},
                {"roi", {{"x", 0}, {"y", 0}, {"width", layout[0]}, {"height", layout[1]}}}});
        }
        resources.push_back({{"id", makeRenderGraphFieldName(pass, "probeTraceReadback")},
            {"layout", render::kLumenProbeTraceReadbackLayout}, {"count", readbackRayCount()}});
        if (pass == candidatePass_) {
            resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "probeWorkState"), 2));
            resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "probeIndirectArgs"), 6));
            if (splitTracing()) {
                resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "probeRayWorkCount"), splitDetail() ? 2 : 1));
                resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "probeRayIndirectArgs"), splitDetail() ? 9 : 6));
            }
        }
        return resources;
    }

    using DispatchArgs = std::array<uint32_t, 3>;

    std::string compareRayDispatch(const std::vector<uint8_t>& rayCountBytes,
        const std::vector<uint8_t>& rayArgsBytes, uint32_t validRays, uint32_t gdfRays, uint32_t detailRays) const
    {
        constexpr uint32_t kCompactThreads = 256;
        constexpr uint32_t kTraceThreads = 64;
        const auto args = [](uint32_t rays, uint32_t groupSize) -> DispatchArgs {
            constexpr uint32_t kDispatchWidth = 1024;
            const uint32_t groupCount = (rays + groupSize - 1u) / groupSize;
            return {std::min(groupCount, kDispatchWidth), (groupCount + kDispatchWidth - 1u) / kDispatchWidth, 1u};
        };
        const size_t rayQueues = splitDetail() ? 2 : 1;
        if (rayCountBytes.size() != rayQueues * sizeof(uint32_t) || readSurfaceValue<uint32_t>(rayCountBytes, 0) != gdfRays ||
            (splitDetail() && readSurfaceValue<uint32_t>(rayCountBytes, 1) != detailRays)) {
            return "Compacted ray counts differ from the measured GDF or Mesh SDF work";
        }
        if (rayArgsBytes.size() != (1 + rayQueues) * sizeof(DispatchArgs) ||
            readSurfaceValue<DispatchArgs>(rayArgsBytes, 0) != args(validRays, kCompactThreads) ||
            readSurfaceValue<DispatchArgs>(rayArgsBytes, 1) != args(gdfRays, kTraceThreads) ||
            (splitDetail() && readSurfaceValue<DispatchArgs>(rayArgsBytes, 2) != args(detailRays, kTraceThreads))) {
            return "Indirect ray dispatches do not match the compacted ray counts";
        }
        return {};
    }

    std::string compareFrame(CardUpdateInspector& inspector, bool empty, bool historyExpected,
        RayHandoffCoverage& coverage) const
    {
        inspector.runtime.poll();
        if (inspector.response.value("status", "") != "ok") { return inspector.response.dump(); }
        auto& core = inspector.runtime.core();
        const auto& jobs = inspector.response.at("result").at("jobs");
        std::array<CardCapture, 2> captures;
        for (size_t index = 0; index < captures.size(); ++index) {
            const auto id = jobs.at(index).at("job").get<std::string>();
            captures[index] = core.completedCapture(id);
            if (!captures[index]) { return core.dispatch({{"method", "jobs.get"}, {"params", {{"job", id}}}}).dump(); }
            const size_t modeResources = index == 1 ? (splitTracing() ? 4 : 2) : 0;
            if (captures[index]->artifacts.size() != kBufferedResources.size() + kTextures.size() + 1 + modeResources) {
                return "Unexpected Gather artifact count";
            }
        }
        const auto& baseline = captures[0]->artifacts;
        const auto& candidate = captures[1]->artifacts;
        const auto compare = [&](size_t index, std::string_view field, size_t expectedBytes) -> std::string {
            if (baseline[index].bytes.size() != expectedBytes || candidate[index].bytes.size() != expectedBytes) {
                return "Unexpected readback size for " + std::string(field);
            }
            const auto mismatch = std::mismatch(baseline[index].bytes.begin(), baseline[index].bytes.end(),
                candidate[index].bytes.begin());
            if (mismatch.first != baseline[index].bytes.end()) {
                if (splitDetail() && field == "probeTraceReadback") {
                    using Record = render::LumenProbeTraceReadbackRecord;
                    // A UAV handoff rounds before the next kernel starts. The Cornell oracle
                    // uses the same tolerance for the two arithmetic endpoints.
                    constexpr float endpointTolerance = 1.0e-6f;
                    for (size_t ray = 0; ray < expectedBytes / sizeof(Record); ++ray) {
                        auto first = readSurfaceValue<Record>(baseline[index].bytes, ray);
                        const auto second = readSurfaceValue<Record>(candidate[index].bytes, ray);
                        if (!(std::abs(first.detailEnd - second.detailEnd) <= endpointTolerance &&
                                std::abs(first.gdfMin - second.gdfMin) <= endpointTolerance)) {
                            return "Detail handoff distance differs at ray " + std::to_string(ray);
                        }
                        first.detailEnd = second.detailEnd;
                        first.gdfMin = second.gdfMin;
                        if (std::memcmp(&first, &second, sizeof(Record)) != 0) {
                            return "Detail trace readback differs at ray " + std::to_string(ray);
                        }
                    }
                    return {};
                }
                return "Gather outputs differ in " + std::string(field) + " at byte " +
                    std::to_string(mismatch.first - baseline[index].bytes.begin());
            }
            return {};
        };
        size_t index = 0;
        for (const auto& [field, words] : kBufferedResources) {
            if (auto error = compare(index++, field, words * sizeof(uint32_t)); !error.empty()) { return error; }
        }
        for (const auto& [field, layout] : kTextures) {
            if (auto error = compare(index++, field, layout[0] * layout[1] * layout[2]); !error.empty()) { return error; }
        }
        const auto& traceReadback = candidate[index].bytes;
        if (auto error = compare(index++, "probeTraceReadback", readbackRayCount() * kReadbackStride); !error.empty()) { return error; }

        uint32_t valid = 0;
        uint32_t acceptedScreenRays = 0;
        constexpr float kScreenHit = 5.0f; // LumenScreenProbeGather.slang probeRadiance.w.
        for (uint32_t probe = 0; probe < kProbeSlots; ++probe) {
            const auto pixel = readSurfaceValue<std::array<uint32_t, 2>>(candidate[0].bytes, probe);
            const bool validProbe = pixel[0] != std::numeric_limits<uint32_t>::max();
            valid += validProbe;
            if (validProbe && !splitTracing()) { continue; }
            for (uint32_t ray = probe * kRaysPerSlot; ray < (probe + 1) * kRaysPerSlot; ++ray) {
                const auto radiance = readSurfaceValue<std::array<float, 4>>(candidate[1].bytes, ray);
                if (validProbe) {
                    acceptedScreenRays += radiance[3] == kScreenHit;
                } else if (radiance != std::array<float, 4>{} ||
                    readSurfaceValue<float>(candidate[2].bytes, ray) != -1.0f ||
                    readSurfaceValue<float>(candidate[3].bytes, ray) != -1.0f ||
                    readSurfaceValue<std::array<float, 4>>(candidate[4].bytes, ray) != std::array<float, 4>{}) {
                    return "Invalid probe did not retain its radiance and distance sentinels";
                }
            }
        }
        if ((empty && valid != 0) || (!empty && (valid == 0 || valid == kProbeSlots))) {
            return "Unexpected number of valid probe slots: " + std::to_string(valid);
        }
        using ProbeCounts = std::array<uint32_t, 2>;
        const ProbeCounts expectedCounts{valid, kProbeSlots - valid};
        const auto& counts = candidate[index++].bytes;
        const auto& args = candidate[index++].bytes;
        if (counts.size() != sizeof(ProbeCounts) || args.size() != 2 * sizeof(DispatchArgs) ||
            readSurfaceValue<ProbeCounts>(counts, 0) != expectedCounts) {
            return "Compaction counts do not match the placed probes";
        }
        for (size_t mode = 0; mode < expectedCounts.size(); ++mode) {
            const auto dispatch = readSurfaceValue<DispatchArgs>(args, mode);
            if (dispatch[0] * dispatch[1] != expectedCounts[mode] || dispatch[2] != 1u) {
                return "Indirect dispatch does not match the compacted probe count";
            }
        }
        if (splitTracing()) {
            if (!historyExpected && acceptedScreenRays != 0u) { return "Screen rays hit without valid history"; }
            constexpr uint32_t kReceiverValid = 1u << 0u;
            constexpr uint32_t kScreenInvoked = 1u << 1u;
            constexpr uint32_t kScreenAccepted = 1u << 2u;
            constexpr uint32_t kDetailInvoked = 1u << 3u;
            constexpr uint32_t kDetailAccepted = 1u << 4u;
            constexpr uint32_t kGDFInvoked = 1u << 5u;
            uint32_t detailRays = 0;
            uint32_t acceptedDetailRays = 0;
            for (size_t ray = 0; ray < readbackRayCount(); ++ray) {
                const auto record = readSurfaceValue<render::LumenProbeTraceReadbackRecord>(traceReadback, ray);
                if ((record.flags & kReceiverValid) == 0u) { continue; }
                const bool screenAccepted = (record.flags & kScreenAccepted) != 0u;
                const bool detailInvoked = (record.flags & kDetailInvoked) != 0u;
                const bool detailAccepted = (record.flags & kDetailAccepted) != 0u;
                const bool gdfInvoked = (record.flags & kGDFInvoked) != 0u;
                if ((record.flags & kScreenInvoked) == 0u || int(screenAccepted) + int(detailAccepted) + int(gdfInvoked) != 1 ||
                    (detailAccepted && !detailInvoked) || (detailInvoked && screenAccepted)) {
                    return "A valid ROI ray must be completed by exactly one of Screen, Mesh SDF or GDF";
                }
                if (gdfInvoked) {
                    float end = record.screenEnd;
                    if (detailInvoked) {
                        const uint32_t encoded = std::bit_cast<uint32_t>(record.detailEnd * 100.0f) & 0x7ffffffcu;
                        end = std::bit_cast<float>(encoded) * 0.01f;
                    }
                    const float expected = std::max(0.0001f, end - 0.1f); // Minimum trace distance and twice the surface bias, in meters.
                    if (std::abs(record.gdfMin - expected) > 0.000001f) {
                        return "GDF start did not continue from the preceding trace endpoint";
                    }
                }
                detailRays += detailInvoked;
                acceptedDetailRays += detailAccepted;
            }
            const uint32_t validRays = valid * kRaysPerSlot;
            const uint32_t gdfRays = validRays - acceptedScreenRays - acceptedDetailRays;
            coverage.mixedScreenAndGDF |= historyExpected && acceptedScreenRays != 0u && gdfRays != 0u;
            coverage.detailHit |= acceptedDetailRays != 0u;
            coverage.detailMiss |= detailRays > acceptedDetailRays;
            return compareRayDispatch(candidate[index].bytes, candidate[index + 1].bytes, validRays, gdfRays, detailRays);
        }
        return {};
    }

    Variant variant_;
    const char* candidatePass_;
};

class LumenActiveProbeDispatchTest final : public LumenProbeDispatchComparison {
public:
    LumenActiveProbeDispatchTest() : LumenProbeDispatchComparison("lumen_active_probe_dispatch", Variant::ActiveProbes) {}
};

class LumenSplitProbeTracingTest final : public LumenProbeDispatchComparison {
public:
    LumenSplitProbeTracingTest() : LumenProbeDispatchComparison("lumen_split_probe_tracing", Variant::SplitTracing) {}
};

class LumenSplitDetailTracingTest final : public LumenProbeDispatchComparison {
public:
    LumenSplitDetailTracingTest() : LumenProbeDispatchComparison("lumen_split_detail_tracing", Variant::SplitDetail) {}
};

METALLIC_REGISTER_RHI_TEST(LumenActiveProbeDispatchTest);
METALLIC_REGISTER_RHI_TEST(LumenSplitProbeTracingTest);
METALLIC_REGISTER_RHI_TEST(LumenSplitDetailTracingTest);

debug::DebugValue surfaceUpdateCaptures(std::string_view pass)
{
    using render::makeRenderGraphFieldName;
    auto captures = debug::DebugValue::array();
    for (const auto& atlas : kSurfaceAtlases) {
        captures.push_back(cardAtlasCapture(makeRenderGraphFieldName(pass, atlas.first)));
    }
    captures.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "cards"),
        kPersistentCardCapacity * sizeof(CapturedCard) / sizeof(uint32_t)));
    captures.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "scene"), 4));
    if (pass == kPersistentSurface) {
        captures.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "updateStatistics"), 3));
        captures.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "cardUpdates"), kPersistentCardCapacity));
    }
    return captures;
}

std::string compareSurfaceUpdate(const CardUpdateFrame& frame, const std::array<uint32_t, 3>& expectedStats,
    uint32_t expectedResident)
{
    const auto& surface = frame.surface->artifacts;
    const auto& full = frame.fullSurface->artifacts;
    const auto& radiosity = frame.radiosity->artifacts;
    if (surface.size() != kSurfaceUpdateFlags + 1 || full.size() != kCardScene + 1 ||
        radiosity.size() != kRadiosityCardState + 1 || frame.denseRadiosity->artifacts.size() != 2 ||
        surface[kSurfaceUpdateStats].bytes.size() != sizeof(expectedStats) ||
        surface[kSurfaceUpdateFlags].bytes.size() != kPersistentCardCapacity * sizeof(uint32_t) ||
        surface[kCardRecords].bytes.size() != kPersistentCardCapacity * sizeof(CapturedCard) ||
        surface[kCardScene].bytes.size() != 4 * sizeof(uint32_t) ||
        radiosity[0].bytes.size() != kAtlasPixels * 8 || radiosity[1].bytes.size() != kAtlasPixels * 8 ||
        frame.denseRadiosity->artifacts[0].bytes.size() != kAtlasPixels * 8 ||
        frame.denseRadiosity->artifacts[1].bytes.size() != kAtlasPixels * 8 ||
        radiosity[kRadiosityUpdateStats].bytes.size() != 4 * sizeof(uint32_t) ||
        radiosity[kRadiosityCardState].bytes.size() != kPersistentCardCapacity * 2 * sizeof(uint32_t)) {
        return "Unexpected Surface Cache or Radiosity readback layout";
    }
    if (readSurfaceValue<uint32_t>(surface[kCardScene].bytes, 0) != expectedResident) {
        return "Unexpected resident card count";
    }
    const auto stats = readSurfaceValue<std::array<uint32_t, 3>>(surface[kSurfaceUpdateStats].bytes, 0);
    if (stats != expectedStats) {
        return "Unexpected Surface Cache update statistics " + nlohmann::json(stats).dump() +
            ", expected " + nlohmann::json(expectedStats).dump();
    }
    for (size_t atlas = 0; atlas < kSurfaceAtlases.size(); ++atlas) {
        const auto& [name, stride] = kSurfaceAtlases[atlas];
        if (surface[atlas].bytes.size() != kAtlasPixels * stride || surface[atlas].bytes != full[atlas].bytes) {
            return std::string("Persistent atlas differs from a same-frame full rebuild: ") + name;
        }
    }
    if (surface[kCardRecords].bytes != full[kCardRecords].bytes || surface[kCardScene].bytes != full[kCardScene].bytes) {
        return "Persistent card transforms or scene metadata differ from a same-frame full rebuild";
    }
    return {};
}

uint32_t cardPixelOffset(uint32_t slot, uint32_t row)
{
    constexpr uint32_t tilesPerAxis = kAtlasEdge / kTileEdge;
    return ((slot / tilesPerAxis) * kTileEdge + row) * kAtlasEdge + (slot % tilesPerAxis) * kTileEdge;
}

std::string checkCardFlags(const CardUpdateFrame& frame, uint32_t resident, scene::SceneEntity object,
    uint32_t flagsForObject, uint32_t flagsForOthers)
{
    const auto& resources = frame.surface->artifacts;
    for (uint32_t slot = 0; slot < kPersistentCardCapacity; ++slot) {
        const auto card = readSurfaceValue<CapturedCard>(resources[kCardRecords].bytes, slot);
        const uint32_t expected = slot >= resident ? 0u :
            (card.metadata[0] == uint32_t(object) ? flagsForObject : flagsForOthers);
        const uint32_t actual = readSurfaceValue<uint32_t>(resources[kSurfaceUpdateFlags].bytes, slot);
        if (actual != expected) {
            return "Incorrect Surface Cache update flags for slot " + std::to_string(slot) + ": " +
                std::to_string(actual) + " versus " + std::to_string(expected);
        }
    }
    return {};
}

std::string checkRadiositySchedule(const CardUpdateFrame& previous, const CardUpdateFrame& current,
    std::array<bool, kPersistentCardCapacity>& refreshed, uint32_t* relitUnscheduledTexels = nullptr)
{
    using Pixel = std::array<uint16_t, 4>;
    const auto& previousRadiosity = previous.radiosity->artifacts;
    const auto& currentRadiosity = current.radiosity->artifacts;
    constexpr std::array expectedStats{kRadiosityCardBudget, 0u, kPersistentResidentCards, kRadiosityCardBudget};
    const auto stats = readSurfaceValue<std::array<uint32_t, 4>>(currentRadiosity[kRadiosityUpdateStats].bytes, 0);
    if (stats != expectedStats) {
        return "Radiosity did not respect its card budget without resets: " + nlohmann::json(stats).dump();
    }
    uint32_t updated = 0;
    for (uint32_t slot = 0; slot < kPersistentCardCapacity; ++slot) {
        using CardState = std::array<uint32_t, 2>;
        const auto before = readSurfaceValue<CardState>(previousRadiosity[kRadiosityCardState].bytes, slot);
        const auto after = readSurfaceValue<CardState>(currentRadiosity[kRadiosityCardState].bytes, slot);
        if (before != after) {
            if (slot >= kPersistentResidentCards) { return "Radiosity scheduled an unused atlas tile"; }
            if (after[1] != before[1] + 1u) { return "A scheduled Radiosity slot did not advance its phase"; }
            if (relitUnscheduledTexels != nullptr) {
                for (uint32_t row = 0; row < kTileEdge; ++row) {
                    const auto offset = cardPixelOffset(slot, row);
                    for (uint32_t column = 0; column < kTileEdge; ++column) {
                        const auto index = offset + column;
                        const float previousCount = halfToFloat(readSurfaceValue<Pixel>(previousRadiosity[0].bytes, index)[3]);
                        const float currentCount = halfToFloat(readSurfaceValue<Pixel>(currentRadiosity[0].bytes, index)[3]);
                        if (previousCount > 0.0f && currentCount != std::min(previousCount + 1.0f, 4.0f)) {
                            return "Sun relighting discarded accumulated samples on a scheduled slot";
                        }
                    }
                }
            }
            refreshed[slot] = true;
            ++updated;
            continue;
        }
        for (uint32_t row = 0; row < kTileEdge; ++row) {
            const uint32_t firstPixel = cardPixelOffset(slot, row);
            const size_t offset = size_t(firstPixel) * sizeof(Pixel);
            const size_t atlases = relitUnscheduledTexels != nullptr ? 1u : 2u;
            for (size_t atlas = 0; atlas < atlases; ++atlas) {
                if (std::memcmp(previousRadiosity[atlas].bytes.data() + offset,
                        currentRadiosity[atlas].bytes.data() + offset, kTileEdge * sizeof(Pixel)) != 0) {
                    return "An unscheduled Radiosity tile changed: " + std::to_string(slot);
                }
            }
            if (relitUnscheduledTexels != nullptr && slot < kPersistentResidentCards) {
                for (uint32_t column = 0; column < kTileEdge; ++column) {
                    const auto index = firstPixel + column;
                    const auto red = [index](const auto& artifact) {
                        return halfToFloat(readSurfaceValue<Pixel>(artifact.bytes, index)[0]);
                    };
                    const float directDelta = red(current.surface->artifacts[kSurfaceLighting]) -
                        red(previous.surface->artifacts[kSurfaceLighting]);
                    if (std::abs(directDelta) <= 0.02f) { continue; }
                    const float outgoingDelta = red(currentRadiosity[1]) - red(previousRadiosity[1]);
                    if (std::abs(outgoingDelta - directDelta) > 0.025f) {
                        return "An unscheduled indirect tile failed to publish its changed direct lighting";
                    }
                    ++*relitUnscheduledTexels;
                }
            }
        }
    }
    return updated == kRadiosityCardBudget ? std::string() :
        "Radiosity card state did not identify the expected number of updates";
}

float maxCardIndirect(const CardUpdateFrame& frame, scene::SceneEntity object, uint32_t resident)
{
    using Pixel = std::array<uint16_t, 4>;
    float maximum = 0.0f;
    for (uint32_t slot = 0; slot < resident; ++slot) {
        const auto card = readSurfaceValue<CapturedCard>(frame.surface->artifacts[kCardRecords].bytes, slot);
        if (card.metadata[0] != uint32_t(object)) { continue; }
        for (uint32_t row = 0; row < kTileEdge; ++row) {
            const uint32_t offset = cardPixelOffset(slot, row);
            for (uint32_t column = 0; column < kTileEdge; ++column) {
                const auto sample = readSurfaceValue<Pixel>(frame.radiosity->artifacts[0].bytes, offset + column);
                maximum = std::max(maximum, halfToFloat(sample[0]));
            }
        }
    }
    return maximum;
}

std::string checkDenseRadiosity(const CardUpdateFrame& frame, float highestExpectedCount, bool expectRadiantTrace)
{
    using Pixel = std::array<uint16_t, 4>;
    const auto& artifacts = frame.denseRadiosity->artifacts;
    float highestCount = 0.0f;
    bool tracedLight = false;
    for (uint32_t pixel = 0; pixel < kAtlasPixels; ++pixel) {
        const auto indirect = readSurfaceValue<Pixel>(artifacts[0].bytes, pixel);
        highestCount = std::max(highestCount, halfToFloat(indirect[3]));
        const auto trace = readSurfaceValue<Pixel>(artifacts[1].bytes, pixel);
        tracedLight |= halfToFloat(trace[0]) > 0.005f;
    }
    if (highestCount != highestExpectedCount) {
        return "Dense Radiosity expected " + std::to_string(highestExpectedCount) +
            " accumulated samples but observed " + std::to_string(highestCount);
    }
    return expectRadiantTrace && !tracedLight ? "Dense Radiosity did not trace lit cards" : std::string();
}

class LumenPersistentCardUpdateTest final : public RHITest {
public:
    LumenPersistentCardUpdateTest()
    {
        name = "lumen_persistent_card_updates";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto fixture = context.outputDirectory / "LumenCardUpdateBoxes.gltf";
        std::error_code error;
        std::filesystem::copy_file(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Lumen/SingleBounce.gltf", fixture,
            std::filesystem::copy_options::overwrite_existing, error);
        if (error) { return RHITestResult::fail("Could not copy the card update fixture: " + error.message()); }
        scene::SceneDocument document;
        if (!document.load(fixture)) { return RHITestResult::fail(document.lastLoadResult().error); }
        const auto receiver = document.objectForNode(0).entity();
        const auto emitter = document.objectForNode(1).entity();
        const auto blocker = document.objectForNode(2).entity();
        if (receiver == scene::kNullSceneEntity || emitter == scene::kNullSceneEntity || blocker == scene::kNullSceneEntity) {
            return RHITestResult::fail("Expected independently editable card fixture objects");
        }
        auto offsideBlocker = document.nodes()[2].worldMatrix;
        offsideBlocker.a[12] += 1.3f;
        if (!document.setObjectWorldMatrix(blocker, offsideBlocker)) {
            return RHITestResult::fail("Could not expose the receiver to the emitter in the card fixture");
        }

        std::string log;
        std::vector<std::string> fieldPaths, cardPaths;
        for (int32_t mesh = 0; mesh < 3; ++mesh) {
            const auto base = "LumenCardUpdate" + std::to_string(mesh);
            const auto field = context.outputDirectory / (base + ".meshdf");
            const auto cards = context.outputDirectory / (base + ".meshcards");
            if (!writeAnalyticBoxField(field, document, log, mesh) || !writeAnalyticBoxCards(cards, document, mesh, log)) {
                return RHITestResult::fail(log);
            }
            fieldPaths.push_back(field.generic_string());
            cardPaths.push_back(cards.generic_string());
        }

        RenderGraph graph;
        if (!makeGraph(graph, fieldPaths, cardPaths)) {
            return RHITestResult::fail("Could not configure paired Surface Caches and persistent Radiosity");
        }
        bench::ValidationRecorder validation;
        auto device = bench::createTestDevice(context, {.applicationName = "Lumen card update regression",
            .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true, .validationSink = validation.sink()});
        if (!device) { return RHITestResult::fail("Could not initialize the card update test device"); }
        auto& gpu = **device;
        if (!gpu.capabilities().shaderStorageImageExtendedFormats) {
            return RHITestResult::skip("Lumen card updates require the R8 GDF and R16 depth storage formats");
        }
        RenderWorld renderWorld;
        renderWorld.setScene(&document);
        renderWorld.setEnvironment({.enabled = false, .visible = false});
        environment::WorldEnvironment world;
        world.sun.enabled = false;
        if (!renderWorld.setWorldEnvironment(world)) { return RHITestResult::fail("Could not disable the fixture sun"); }

        RenderView view;
        ViewCamera camera;
        camera.eye = {0.0f, 0.0f, 3.0f};
        camera.orthographic = true;
        if (!view.setCamera(camera) || !view.setRenderResolution(kAtlasEdge, kAtlasEdge)) {
            return RHITestResult::fail("Invalid card update test camera");
        }
        HistoryResourceManager history;
        if (!history.initialize(gpu)) { return RHITestResult::fail("Could not initialize graph history"); }
        CardUpdateInspector inspector;
        RenderGraphExecutor executor;
        executor.bindRenderWorld(&renderWorld);
        executor.bindRenderView(&view);
        executor.setDebugObserver(&inspector);
        if (!executor.compile(gpu, graph, kAtlasEdge, kAtlasEdge, RenderGraphCompileOptions{.enableDebugReadback = true}, log)) {
            return RHITestResult::fail("Card update graph compile: " + log);
        }
        const RenderGraphSubmitDesc submission{.graphicsQueue = gpu.getQueue(QueueType::Graphics),
            .historyResources = &history, .recordingWorkerLimit = 1};

        const auto runFrame = [&](std::string_view stage, const std::array<uint32_t, 3>& stats,
            uint32_t resident = kPersistentResidentCards) -> std::optional<CardUpdateFrame> {
            auto frame = captureFrame(inspector, executor, submission, log);
            if (frame) { log = compareSurfaceUpdate(*frame, stats, resident); }
            if (!frame || !log.empty()) {
                log = std::string(stage) + ": " + log;
                return std::nullopt;
            }
            return frame;
        };

        auto previous = runFrame("Initial frame", {1u, kPersistentResidentCards, kPersistentResidentCards});
        if (!previous) { return RHITestResult::fail(log); }
        if (const auto message = checkDenseRadiosity(*previous, 0.0f, false); !message.empty()) {
            return RHITestResult::fail(message);
        }
        const auto initialStats = readSurfaceValue<std::array<uint32_t, 4>>(
            previous->radiosity->artifacts[kRadiosityUpdateStats].bytes, 0);
        if (initialStats[0] != kPersistentResidentCards || initialStats[1] != kPersistentResidentCards ||
            initialStats[2] != kPersistentResidentCards || initialStats[3] != kRadiosityCardBudget) {
            return RHITestResult::fail("First Radiosity frame failed to initialize all resident tiles");
        }
        if (const auto message = checkCardFlags(*previous, kPersistentResidentCards, emitter,
                kCardCaptured | kCardRelit, kCardCaptured | kCardRelit);
            !message.empty()) { return RHITestResult::fail(message); }

        std::array<bool, kPersistentCardCapacity> refreshed{};
        constexpr uint32_t framesPerSweep = kPersistentResidentCards / kRadiosityCardBudget;
        for (uint32_t i = 0; i < 2 * framesPerSweep; ++i) {
            auto current = runFrame("Stationary frame", {0u, 0u, 0u});
            if (!current) { return RHITestResult::fail(log); }
            if (const auto message = checkRadiositySchedule(*previous, *current, refreshed); !message.empty()) {
                return RHITestResult::fail(message);
            }
            previous = std::move(current);
        }
        if (!std::ranges::all_of(std::span{refreshed}.first(kPersistentResidentCards), [](bool updated) { return updated; })) {
            return RHITestResult::fail("The Radiosity priority failed to revisit every resident card across two sweeps");
        }
        if (!(maxCardIndirect(*previous, receiver, kPersistentResidentCards) > 0.005f)) {
            return RHITestResult::fail("Budgeted Radiosity never transported the off-screen emitter to the receiver");
        }

        struct SunEdit {
            const char* name;
            float3 direction;
            float illuminance;
            bool enabled;
        };
        const SunEdit sunEdits[]{
            {"Enable Sun", {0.0f, -0.8f, -0.6f}, 4.0f, true},
            {"Drag Sun first direction", {-0.4f, -0.6f, -0.6f}, 4.0f, true},
            {"Drag Sun second direction", {0.4f, -0.5f, -0.6f}, 4.0f, true},
            {"Raise Sun intensity", {0.4f, -0.5f, -0.6f}, 8.0f, true},
            {"Jump Sun intensity", {0.4f, -0.5f, -0.6f}, 48.0f, true},
            {"Disable Sun", {0.4f, -0.5f, -0.6f}, 48.0f, false},
        };
        uint32_t relitUnscheduledTexels = 0;
        for (const auto& edit : sunEdits) {
            world.sun.direction = edit.direction;
            world.sun.illuminance = edit.illuminance;
            world.sun.enabled = edit.enabled;
            if (!renderWorld.setWorldEnvironment(world)) {
                return RHITestResult::fail(std::string("Could not edit the fixture Sun: ") + edit.name);
            }
            auto current = runFrame(edit.name, {0u, 0u, kPersistentResidentCards});
            if (!current) { return RHITestResult::fail(log); }
            const auto check = [&]() {
                if (auto message = checkCardFlags(*current, kPersistentResidentCards, emitter, kCardRelit, kCardRelit);
                    !message.empty()) { return message; }
                if (auto message = checkRadiositySchedule(*previous, *current, refreshed, &relitUnscheduledTexels);
                    !message.empty()) { return message; }
                return checkDenseRadiosity(*current, 4.0f, true);
            };
            if (const auto message = check(); !message.empty()) {
                return RHITestResult::fail(std::string(edit.name) + ": " + message);
            }
            if (!(maxCardIndirect(*current, receiver, kPersistentResidentCards) > 0.005f)) {
                return RHITestResult::fail(std::string(edit.name) + ": Receiver lost indirect lighting during Sun edits");
            }
            previous = std::move(current);
        }
        if (relitUnscheduledTexels == 0u) {
            return RHITestResult::fail("Sun edits failed to exercise visible relighting on unscheduled indirect tiles");
        }

        scene::LightingSettings localLighting = renderWorld.lighting();
        scene::PunctualLight point;
        point.properties.intensityUnit = scene::LightUnit::Candela;
        point.properties.intensity = 4.0f;
        point.properties.range = 4.0f;
        point.position = {0.0f, 1.2f, 0.95f};
        localLighting.lights.push_back(std::move(point));
        if (!renderWorld.setLighting(std::move(localLighting))) {
            return RHITestResult::fail("Could not add the fixture point light");
        }
        const auto local = runFrame("Point light edit", {0u, 0u, kPersistentResidentCards});
        if (!local) { return RHITestResult::fail(log); }
        if (const auto message = checkCardFlags(*local, kPersistentResidentCards, emitter,
                kCardRelit | kCardLightingChanged, kCardRelit | kCardLightingChanged); !message.empty()) {
            return RHITestResult::fail(message);
        }
        const auto localStats = readSurfaceValue<std::array<uint32_t, 4>>(
            local->radiosity->artifacts[kRadiosityUpdateStats].bytes, 0);
        if (localStats[1] != kPersistentResidentCards) {
            return RHITestResult::fail("Point light edit failed to reset the persistent indirect history");
        }
        if (const auto message = checkDenseRadiosity(*local, 1.0f, true); !message.empty()) {
            return RHITestResult::fail("Point light reset: " + message);
        }

        auto dark = document.materials()[1];
        dark.emissiveFactor = float3(0.0f);
        if (!document.setMaterialProperties(1, dark)) { return RHITestResult::fail("Could not edit the emitter material"); }
        const auto material = runFrame("Emitter material edit", {0u, kCardsPerBox, kCardsPerBox});
        if (!material) { return RHITestResult::fail(log); }
        if (const auto message = checkCardFlags(*material, kPersistentResidentCards, emitter, kCardCaptured | kCardRelit, 0);
            !message.empty()) { return RHITestResult::fail(message); }
        const auto materialStats = readSurfaceValue<std::array<uint32_t, 4>>(
            material->radiosity->artifacts[kRadiosityUpdateStats].bytes, 0);
        if (materialStats[1] != kCardsPerBox) {
            return RHITestResult::fail("Emitter edit did not reset precisely its six Radiosity cards");
        }

        // A uniform shrink triggers the minimum card depth extent. This moves
        // new fragments farther from the reversed-Z near plane in the same tile.
        auto movedBlocker = document.nodes()[2].worldMatrix;
        for (size_t axis = 0; axis < 3; ++axis) { movedBlocker.a[axis * 5] *= 0.03f; }
        movedBlocker.a[12] += 0.1f;
        if (!document.setObjectWorldMatrix(blocker, movedBlocker)) {
            return RHITestResult::fail("Could not move and shrink the GDF blocker");
        }
        const auto geometry = runFrame("Blocker geometry change", {0u, kCardsPerBox, kPersistentResidentCards});
        if (!geometry) { return RHITestResult::fail(log); }
        bool deeperFragment = false;
        for (uint32_t slot = 0; slot < kPersistentResidentCards; ++slot) {
            const auto card = readSurfaceValue<CapturedCard>(geometry->surface->artifacts[kCardRecords].bytes, slot);
            if (card.metadata[0] != uint32_t(blocker)) { continue; }
            constexpr size_t kDepth = 3;
            const uint16_t before = readSurfaceValue<uint16_t>(material->surface->artifacts[kDepth].bytes, tileCenter(slot));
            const uint16_t after = readSurfaceValue<uint16_t>(geometry->surface->artifacts[kDepth].bytes, tileCenter(slot));
            deeperFragment |= after != UINT16_MAX && after > before + 1024;
        }
        if (!deeperFragment) { return RHITestResult::fail("The recapture fixture did not exercise a farther reversed-Z fragment"); }

        if (!document.setObjectVisible(blocker, false)) { return RHITestResult::fail("Could not hide the GDF blocker"); }
        constexpr uint32_t remaining = kPersistentResidentCards - kCardsPerBox;
        const auto hidden = runFrame("Blocker visibility change", {1u, remaining, remaining}, remaining);
        if (!hidden) { return RHITestResult::fail(log); }
        const auto hiddenStats = readSurfaceValue<std::array<uint32_t, 4>>(
            hidden->radiosity->artifacts[kRadiosityUpdateStats].bytes, 0);
        if (hiddenStats[1] != kPersistentResidentCards || hiddenStats[2] != remaining) {
            return RHITestResult::fail("The Radiosity roster change did not reset captured and retired tiles");
        }
        for (uint32_t slot = remaining; slot < kPersistentResidentCards; ++slot) {
            const auto state = readSurfaceValue<std::array<uint32_t, 2>>(hidden->radiosity->artifacts[kRadiosityCardState].bytes, slot);
            if (state != std::array<uint32_t, 2>{}) { return RHITestResult::fail("Retired Radiosity state survived"); }
            for (size_t atlas = 0; atlas < 2; ++atlas) {
                for (uint32_t row = 0; row < kTileEdge; ++row) {
                    constexpr uint32_t stride = 8;
                    const size_t offset = size_t(cardPixelOffset(slot, row)) * stride;
                    const auto bytes = std::span{hidden->radiosity->artifacts[atlas].bytes}.subspan(offset, kTileEdge * stride);
                    if (!std::ranges::all_of(bytes, [](uint8_t value) { return value == 0; })) {
                        return RHITestResult::fail("Retired Radiosity atlas tile retained stale lighting");
                    }
                }
            }
        }
        if (!executor.compile(gpu, graph, kAtlasEdge * 2, kAtlasEdge * 2, RenderGraphCompileOptions{.enableDebugReadback = true}, log)) {
            return RHITestResult::fail("Card update graph resize: " + log);
        }
        const auto resized = runFrame("Graph resize", {1u, remaining, remaining}, remaining);
        if (!resized) { return RHITestResult::fail(log); }
        const auto resizedStats = readSurfaceValue<std::array<uint32_t, 4>>(
            resized->radiosity->artifacts[kRadiosityUpdateStats].bytes, 0);
        if (resizedStats != std::array{remaining, remaining, remaining, kRadiosityCardBudget}) {
            return RHITestResult::fail("Graph resize did not reinitialize the Radiosity atlas");
        }
        if (validation.failed()) {
            return RHITestResult::fail("Persistent card Vulkan validation: " + validation.snapshot().dump());
        }
        return RHITestResult::pass("Sun edits relight current cards while indirect history advances within its budget. "
            "Dense Radiosity traces on reset, and retired persistent tiles clear.");
    }

private:
    static bool makeGraph(render::RenderGraph& graph, const std::vector<std::string>& fields,
        const std::vector<std::string>& cards)
    {
        using render::makeRenderGraphFieldName;
        const auto edge = [&](std::string_view from, std::string_view source, std::string_view to, std::string_view target) {
            return graph.addEdge(makeRenderGraphFieldName(from, source), makeRenderGraphFieldName(to, target));
        };
        constexpr std::string_view gdf = "GDF";
        if (!graph.addNode("GlobalDistanceFieldPagesPass", std::string(gdf), {{"distanceFieldPaths", fields},
                {"center", {0.0, 0.5, 0.65}}, {"extent", 2.2}, {"resolution", 84}, {"maxPhysicalPages", 1728},
                {"quantized", true}, {"buildObjectGrid", true}, {"persistentPages", true}})) { return false; }
        for (const auto name : {kFullSurface, kPersistentSurface}) {
            if (!graph.addNode("LumenSurfaceCachePass", std::string(name), {{"meshCardPaths", cards},
                    {"atlasSize", kAtlasEdge}, {"tileResolution", kTileEdge}, {"shadowedLighting", true},
                    {"exportDirectIrradiance", true}, {"persistentCards", name == kPersistentSurface}})) { return false; }
            for (const auto* input : {"pageTable", "pageAtlas", "coarseMip", "domain"}) {
                if (!edge(gdf, input, name, input)) { return false; }
            }
        }
        constexpr std::array surfaceBindings{std::pair{"cards", "cards"}, std::pair{"scene", "cardScene"},
            std::pair{"depth", "cardDepth"}, std::pair{"lighting", "cardLighting"}, std::pair{"normal", "cardNormal"},
            std::pair{"albedo", "cardAlbedo"}, std::pair{"emissive", "cardEmissive"},
            std::pair{"directIrradiance", "cardDirectLighting"}};
        for (const auto name : {kPersistentRadiosity, kDenseRadiosity}) {
            const bool persistent = name == kPersistentRadiosity;
            if (!graph.addNode("LumenRadiosityPass", std::string(name), {{"atlasSize", kAtlasEdge},
                    {"tileResolution", kTileEdge}, {"maxTraceDistance", 4.0}, {"persistentCards", persistent},
                    {"maxCardsPerFrame", kRadiosityCardBudget}})) { return false; }
            for (const auto* input : {"pageTable", "pageAtlas", "coarseMip", "domain", "objectGrid", "objectGridScene"}) {
                if (!edge(gdf, input, name, input)) { return false; }
            }
            const auto surface = persistent ? kPersistentSurface : kFullSurface;
            for (const auto& [source, target] : surfaceBindings) {
                if (!edge(surface, source, name, target)) { return false; }
            }
            if (!graph.markOutput(makeRenderGraphFieldName(name, "lighting"))) { return false; }
        }
        return graph.markOutput(makeRenderGraphFieldName(kFullSurface, "lighting")) &&
            edge(kPersistentSurface, "cardUpdates", kPersistentRadiosity, "cardUpdates");
    }

    static std::optional<CardUpdateFrame> captureFrame(CardUpdateInspector& inspector,
        render::RenderGraphExecutor& executor, const render::RenderGraphSubmitDesc& submission, std::string& log)
    {
        using render::makeRenderGraphFieldName;
        log.clear();
        const auto radiosityField = [](std::string_view name) { return makeRenderGraphFieldName(kPersistentRadiosity, name); };
        inspector.request = {{"method", "capture.batch"}, {"params", {
            {"batches", debug::DebugValue::array({
                {{"pass", kPersistentSurface}, {"resources", surfaceUpdateCaptures(kPersistentSurface)}},
                {{"pass", kFullSurface}, {"resources", surfaceUpdateCaptures(kFullSurface)}},
                {{"pass", kPersistentRadiosity}, {"resources", debug::DebugValue::array({
                    cardAtlasCapture(radiosityField("indirectIrradiance")), cardAtlasCapture(radiosityField("lighting")),
                    cardBufferCapture(radiosityField("updateStatistics"), 4),
                    cardBufferCapture(radiosityField("cardState"), 2 * kPersistentCardCapacity)})}},
                {{"pass", kDenseRadiosity}, {"resources", debug::DebugValue::array({
                    cardAtlasCapture(makeRenderGraphFieldName(kDenseRadiosity, "indirectIrradiance")),
                    cardAtlasCapture(makeRenderGraphFieldName(kDenseRadiosity, "traceRadiance"))})}},
            })},
        }}};
        if (const auto result = executor.execute(submission); !result) {
            log = std::string(render::resultToString(result));
            return std::nullopt;
        }
        if (!executor.waitForSubmittedWork()) { log = "Could not wait for the card update frame"; return std::nullopt; }
        inspector.runtime.poll();
        const auto& response = inspector.response;
        if (response.value("status", "") != "ok") { log = response.dump(); return std::nullopt; }
        auto& core = inspector.runtime.core();
        const auto& jobs = response.at("result").at("jobs");
        const auto read = [&](size_t index) {
            const auto id = jobs.at(index).at("job").get<std::string>();
            auto capture = core.completedCapture(id);
            if (!capture) { log = core.dispatch({{"method", "jobs.get"}, {"params", {{"job", id}}}}).dump(); }
            return capture;
        };
        auto surface = read(0);
        auto full = read(1);
        auto radiosity = read(2);
        auto denseRadiosity = read(3);
        if (!surface || !full || !radiosity || !denseRadiosity) { return std::nullopt; }
        return CardUpdateFrame{std::move(surface), std::move(full), std::move(radiosity), std::move(denseRadiosity)};
    }
};

METALLIC_REGISTER_RHI_TEST(LumenPersistentCardUpdateTest);

class LumenTemporalInputPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("indirectDiffuse").transferWrite().format = render::Format::RGBA16Sfloat;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        return context.commandBuffer().clearColorTexture(*context.outputTexture("indirectDiffuse").texture(),
            render::TextureLayout::TransferDestination, {0.5f, 0.25f, 0.125f, 1.0f});
    }
};

struct LumenTemporalFrame {
    CardCapture deferred, temporal, staticTemporal, screenTrace;

    std::array<float, 4> position(size_t pixel) const
    {
        return readSurfaceValue<std::array<float, 4>>(deferred->artifacts[0].bytes, pixel);
    }

    std::array<float, 4> previousPosition(size_t pixel) const
    {
        return readSurfaceValue<std::array<float, 4>>(deferred->artifacts[1].bytes, pixel);
    }

    float historyWeight(size_t pixel, bool dynamic = true) const
    {
        return readSurfaceValue<float>((dynamic ? temporal : staticTemporal)->artifacts[0].bytes, pixel);
    }

    float screenHistoryDepth(size_t pixel) const
    {
        return readSurfaceValue<float>(screenTrace->artifacts[0].bytes, pixel);
    }
};

class LumenDynamicTemporalTest final : public RHITest {
public:
    LumenDynamicTemporalTest()
    {
        name = "lumen_dynamic_temporal";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        constexpr float startX = -0.28f;
        constexpr float destinationX = 0.28f;
        const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Lumen/SingleBounce.gltf";
        const auto fixturePath = context.outputDirectory / "LumenDynamicTemporal.gltf";
        std::ifstream input(source);
        if (!input) { return RHITestResult::fail("Could not read the temporal test geometry"); }
        auto asset = nlohmann::json::parse(input, nullptr, false);
        if (asset.is_discarded()) { return RHITestResult::fail("Could not parse the temporal test geometry"); }
        auto& movingNode = asset.at("nodes").at(2);
        movingNode["scale"] = {0.25f, 2.0f, 0.4f};
        movingNode["translation"] = {startX, 0.0f, 0.8f};
        {
            std::ofstream output(fixturePath);
            output << asset.dump();
            if (!output) { return RHITestResult::fail("Could not write the moving-object fixture"); }
        }
        scene::SceneDocument document;
        if (!document.load(fixturePath)) { return RHITestResult::fail(document.lastLoadResult().error); }
        const auto movingObject = std::ranges::find_if(document.renderNodes(), [&](const auto& node) {
            return document.renderPrimitives()[node.renderPrimitiveIndex].meshIndex == 2;
        });
        if (movingObject == document.renderNodes().end()) { return RHITestResult::fail("Missing moving receiver"); }
        const auto blocker = movingObject->object;

        RenderGraph graph;
        if (!makeGraph(graph)) { return RHITestResult::fail("Could not build the dynamic Lumen temporal graph"); }
        bench::ValidationRecorder validation;
        auto device = render::createDevice({.applicationName = "Lumen dynamic temporal regression",
            .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true, .enableMeshShader = true, .enableTaskShader = true,
            .enableTaskShaderSubgroupBallot = true, .enableGeometryShader = true, .validationSink = validation.sink()});
        if (!device) { return RHITestResult::fail("Could not initialize the Lumen temporal test device"); }
        auto& gpu = **device;
        const auto& caps = gpu.capabilities();
        if (!caps.meshShader || !caps.taskShader || !caps.geometryShader) {
            return RHITestResult::skip("Lumen temporal fixture requires mesh, task and geometry shader support");
        }
        RenderWorld world;
        world.setScene(&document);
        world.setEnvironment({.enabled = false, .visible = false});
        environment::WorldEnvironment environment;
        environment.sun.enabled = true;
        environment.sun.direction = {0.0f, -0.8f, -0.6f};
        environment.sun.illuminance = 24.0f;
        if (!world.setWorldEnvironment(environment)) { return RHITestResult::fail("Could not initialize the fixture sun"); }

        RenderView view;
        ViewCamera camera;
        camera.eye = {0.0f, 0.0f, 3.0f};
        camera.orthographic = true;
        camera.orthoHeight = 1.25f;
        camera.farPlane = 8.0f;
        if (!view.setCamera(camera) || !view.setRenderResolution(kAtlasEdge, kAtlasEdge)) {
            return RHITestResult::fail("Could not initialize the temporal test camera");
        }
        HistoryResourceManager history;
        if (!history.initialize(gpu)) { return RHITestResult::fail("Could not initialize graph history"); }
        CardUpdateInspector inspector;
        RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        executor.bindRenderView(&view);
        executor.setDebugObserver(&inspector);
        std::string log;
        if (!executor.compile(gpu, graph, kAtlasEdge, kAtlasEdge, RenderGraphCompileOptions{.enableDebugReadback = true}, log)) {
            return RHITestResult::fail("Dynamic temporal graph compile: " + log);
        }
        const RenderGraphSubmitDesc submission{.graphicsQueue = gpu.getQueue(QueueType::Graphics),
            .historyResources = &history, .recordingWorkerLimit = 1};
        const auto renderFrame = [&]() { return captureFrame(inspector, executor, submission, log); };

        const auto initial = renderFrame();
        if (!initial || !historyEmpty(*initial) || !historyEmpty(*initial, false) || screenHistoryPixels(*initial) != 0u) {
            return RHITestResult::fail(initial ? "First frame unexpectedly used temporal history" : log);
        }
        auto stationary = renderFrame();
        if (!stationary) { return RHITestResult::fail(log); }
        stationary = renderFrame();
        if (!stationary) { return RHITestResult::fail(log); }

        auto transform = document.nodes()[2].worldMatrix;
        transform.a[12] += destinationX - startX;
        if (!document.setObjectWorldMatrix(blocker, transform)) { return RHITestResult::fail("Could not move the receiver"); }
        const auto moved = renderFrame();
        if (!moved) { return RHITestResult::fail(log); }
        if (!historyEmpty(*moved, false) || screenHistoryPixels(*moved) != 0u) {
            return RHITestResult::fail("A rigid motion retained the static temporal or screen trace history");
        }
        uint32_t shiftedPixels = 0, revealedPixels = 0, staticPixels = 0;
        for (size_t pixel = 0; pixel < kAtlasPixels; ++pixel) {
            const auto current = moved->position(pixel);
            if (current[3] <= 0.0f || std::abs(current[1]) > 0.09f) { continue; }
            const auto before = stationary->position(pixel);
            const auto previous = moved->previousPosition(pixel);
            const float weight = moved->historyWeight(pixel);
            if (current[2] > 0.75f && std::abs(current[0] - destinationX) < 0.06f && before[2] < 0.5f) {
                ++shiftedPixels;
                if (previous[3] <= 0.0f || std::abs(previous[0] - current[0] + destinationX - startX) > 0.002f ||
                    std::abs(previous[1] - current[1]) > 0.002f || std::abs(previous[2] - current[2]) > 0.002f ||
                    !(weight > 0.6f)) {
                    return RHITestResult::fail("Translated receiver did not reproject to its previous world-space surface");
                }
            } else if (before[2] > 0.75f && current[2] < 0.5f && std::abs(current[0] - startX) < 0.06f) {
                ++revealedPixels;
                if (weight != 0.0f) { return RHITestResult::fail("Newly revealed receiver retained foreground GI history"); }
            } else if (std::abs(current[0]) < 0.08f && before[2] < 0.5f && current[2] < 0.5f) {
                ++staticPixels;
                if (previous[3] <= 0.0f || !(weight > 0.6f)) {
                    return RHITestResult::fail("Moving one object discarded unrelated static receiver history");
                }
            }
        }
        if (std::min({shiftedPixels, revealedPixels, staticPixels}) < 80u) {
            return RHITestResult::fail("Insufficient dynamic temporal coverage for moved, revealed and static pixels: " +
                std::to_string(shiftedPixels) + ", " + std::to_string(revealedPixels) + ", " + std::to_string(staticPixels));
        }

        auto recovered = renderFrame();
        if (!recovered || !hasRetainedHistory(*recovered)) {
            return RHITestResult::fail(recovered ? "History did not recover after rigid motion" : log);
        }
        environment.sun.illuminance = 120.0f;
        if (!world.setWorldEnvironment(environment)) { return RHITestResult::fail("Could not stage a global sun change"); }
        environment.sun.illuminance = 24.0f;
        if (!world.setWorldEnvironment(environment)) { return RHITestResult::fail("Could not revert the staged sun change"); }
        const auto coalesced = renderFrame();
        if (!coalesced || !hasRetainedHistory(*coalesced)) {
            return RHITestResult::fail(coalesced ? "An unrendered sun change discarded history" : log);
        }
        struct LightEdit {
            const char* name;
            float illuminance;
            bool enabled;
            bool retainsHistory;
        };
        constexpr LightEdit edits[]{
            {"Sun direction change", 24.0f, true, true},
            {"Sun intensity doubled", 48.0f, true, true},
            {"Sun intensity exactly quadrupled", 192.0f, true, true},
            {"Sun intensity changed by more than four times", 960.0f, true, false},
            {"Sun disabled", 960.0f, false, false},
            {"Sun re-enabled", 960.0f, true, false},
        };
        environment.sun.direction = {0.0f, -0.6f, -0.8f};
        for (const auto& edit : edits) {
            environment.sun.illuminance = edit.illuminance;
            environment.sun.enabled = edit.enabled;
            if (!world.setWorldEnvironment(environment)) {
                return RHITestResult::fail(std::string("Could not apply light edit: ") + edit.name);
            }
            const auto relit = renderFrame();
            if (!relit) { return RHITestResult::fail(log); }
            if (edit.retainsHistory) {
                if (!hasRetainedHistory(*relit)) {
                    return RHITestResult::fail(std::string("History did not survive: ") + edit.name);
                }
            } else {
                if (!historyEmpty(*relit) || !historyEmpty(*relit, false) || screenHistoryPixels(*relit) != 0u) {
                    return RHITestResult::fail(std::string("Stale history survived: ") + edit.name);
                }
                recovered = renderFrame();
                if (!recovered || !hasRetainedHistory(*recovered)) {
                    return RHITestResult::fail(recovered ? "History did not recover after a global light change" : log);
                }
            }
        }
        camera.eye[0] += 0.12f;
        camera.center[0] += 0.12f;
        if (!view.setCamera(camera)) { return RHITestResult::fail("Could not move the camera for its cut"); }
        view.cameraCut();
        const auto cut = renderFrame();
        if (!cut || !historyEmpty(*cut) || !historyEmpty(*cut, false) || screenHistoryPixels(*cut) != 0u) {
            return RHITestResult::fail(cut ? "Camera cut retained stale history" : log);
        }
        if (validation.failed()) {
            return RHITestResult::fail("Dynamic temporal Vulkan validation: " + validation.snapshot().dump());
        }
        return RHITestResult::pass("Rigid motion kept valid history, revealed surfaces rejected it, and light edits followed the UE history threshold.");
    }

private:
    static bool makeGraph(render::RenderGraph& graph)
    {
        using namespace render;
        static const bool registered = registerRenderGraphPassType("LumenTemporalInputPass", "Constant diffuse test input",
            [] { return std::make_unique<LumenTemporalInputPass>(); });
        if (!registered || !graph.addNode("VisibilityBufferPass", "VBuffer", {{"visualization", "none"}, {"enableClusterRtx", false}}) ||
            !graph.addNode("VisibilityBufferDeferredPass", "Deferred", {{"materialBinning", false}, {"outputLinear", true},
                {"exportLumenReceivers", true}, {"exportLumenPreviousPosition", true}}) ||
            !graph.addNode("LumenTemporalInputPass", "Fixture") ||
            !graph.addNode("LumenTemporalGatherPass", "Temporal", {{"dynamicReprojection", true},
                {"publishScreenTraceHistory", true}}) ||
            !graph.addNode("LumenTemporalGatherPass", "StaticTemporal") ||
            !graph.addNode("LumenScreenTraceSetupPass", "ScreenTrace", {{"historySource", "composited"},
                {"historyProducer", "Temporal"}})) {
            return false;
        }
        for (const std::string_view pass : {"Temporal", "StaticTemporal"}) {
            if (!graph.addEdge("Deferred.lumenPosition", makeRenderGraphFieldName(pass, "position")) ||
                !graph.addEdge("Deferred.lumenNormal", makeRenderGraphFieldName(pass, "normal")) ||
                !graph.addEdge("Deferred.color", makeRenderGraphFieldName(pass, "directColor")) ||
                !graph.addEdge("Fixture.indirectDiffuse", makeRenderGraphFieldName(pass, "currentIndirectDiffuse")) ||
                !graph.markOutput(makeRenderGraphFieldName(pass, "historyWeight"))) {
                return false;
            }
        }
        return graph.addEdge("VBuffer.visibility", "Deferred.visibility") && graph.addEdge("VBuffer.depth", "Deferred.depth") &&
            graph.addEdge("VBuffer.rasterInfo", "Deferred.rasterInfo") && graph.addEdge("VBuffer.depth", "Temporal.sceneDepth") &&
            graph.addEdge("Deferred.lumenPreviousPosition", "Temporal.previousPosition") &&
            graph.addEdge("VBuffer.depth", "ScreenTrace.depth") && graph.markOutput("ScreenTrace.previousDepth");
    }

    static std::optional<LumenTemporalFrame> captureFrame(CardUpdateInspector& inspector, render::RenderGraphExecutor& executor,
        const render::RenderGraphSubmitDesc& submission, std::string& log)
    {
        using namespace render;
        inspector.request = {{"method", "capture.batch"}, {"params", {{"batches", debug::DebugValue::array({
            {{"pass", "Deferred"}, {"resources", debug::DebugValue::array({cardAtlasCapture("Deferred.lumenPosition"),
                cardAtlasCapture("Deferred.lumenPreviousPosition")})}},
            {{"pass", "Temporal"}, {"resources", debug::DebugValue::array({cardAtlasCapture("Temporal.historyWeight")})}},
            {{"pass", "StaticTemporal"}, {"resources", debug::DebugValue::array({cardAtlasCapture("StaticTemporal.historyWeight")})}},
            {{"pass", "ScreenTrace"}, {"resources", debug::DebugValue::array({cardAtlasCapture("ScreenTrace.previousDepth")})}},
        })}}}};
        if (const auto result = executor.execute(submission); !result) {
            log = std::string(resultToString(result));
            return std::nullopt;
        }
        if (!executor.waitForSubmittedWork()) { log = "Could not wait for the Lumen temporal frame"; return std::nullopt; }
        inspector.runtime.poll();
        if (inspector.response.value("status", "") != "ok") { log = inspector.response.dump(); return std::nullopt; }
        auto& core = inspector.runtime.core();
        const auto& jobs = inspector.response.at("result").at("jobs");
        const auto read = [&](size_t index) {
            const auto id = jobs.at(index).at("job").get<std::string>();
            auto capture = core.completedCapture(id);
            if (!capture) { log = core.dispatch({{"method", "jobs.get"}, {"params", {{"job", id}}}}).dump(); }
            return capture;
        };
        auto deferred = read(0);
        auto temporal = read(1);
        auto staticTemporal = read(2);
        auto screenTrace = read(3);
        if (!deferred || !temporal || !staticTemporal || !screenTrace) { return std::nullopt; }
        const auto singleFloatOutput = [](const CardCapture& capture) {
            return capture->artifacts.size() == 1 && capture->artifacts[0].bytes.size() == kAtlasPixels * sizeof(float);
        };
        if (deferred->artifacts.size() != 2 ||
            deferred->artifacts[0].bytes.size() != kAtlasPixels * sizeof(std::array<float, 4>) ||
            deferred->artifacts[1].bytes.size() != kAtlasPixels * sizeof(std::array<float, 4>) ||
            !singleFloatOutput(temporal) || !singleFloatOutput(staticTemporal) || !singleFloatOutput(screenTrace)) {
            log = "Unexpected Lumen dynamic temporal readback layout";
            return std::nullopt;
        }
        return LumenTemporalFrame{std::move(deferred), std::move(temporal), std::move(staticTemporal), std::move(screenTrace)};
    }

    static bool historyEmpty(const LumenTemporalFrame& frame, bool dynamic = true)
    {
        size_t visiblePixels = 0;
        for (size_t pixel = 0; pixel < kAtlasPixels; ++pixel) {
            if (frame.position(pixel)[3] <= 0.0f) { continue; }
            ++visiblePixels;
            if (frame.historyWeight(pixel, dynamic) != 0.0f) { return false; }
        }
        return visiblePixels >= 100u;
    }

    static bool hasRetainedHistory(const LumenTemporalFrame& frame)
    {
        size_t visiblePixels = 0;
        for (size_t pixel = 0; pixel < kAtlasPixels; ++pixel) {
            if (frame.position(pixel)[3] > 0.0f && frame.historyWeight(pixel) > 0.45f &&
                frame.historyWeight(pixel, false) > 0.45f && frame.screenHistoryDepth(pixel) > 0.0f) {
                ++visiblePixels;
            }
        }
        return visiblePixels >= 100u;
    }

    static size_t screenHistoryPixels(const LumenTemporalFrame& frame)
    {
        size_t validPixels = 0;
        for (size_t pixel = 0; pixel < kAtlasPixels; ++pixel) {
            if (frame.screenHistoryDepth(pixel) > 0.0f) { ++validPixels; }
        }
        return validPixels;
    }
};

METALLIC_REGISTER_RHI_TEST(LumenDynamicTemporalTest);

class LumenReflectionTest final : public RHITest {
public:
    LumenReflectionTest()
    {
        name = "lumen_software_reflections";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Lumen/SingleBounce.gltf";
        const auto fixture = context.outputDirectory / "LumenReflection.gltf";
        const auto field = context.outputDirectory / "LumenReflectionEmitter.meshdf";
        const auto cards = context.outputDirectory / "LumenReflectionEmitter.meshcards";
        std::ifstream input(source);
        if (!input) { return RHITestResult::fail("Could not read the reflection fixture"); }
        auto asset = nlohmann::json::parse(input, nullptr, false);
        if (asset.is_discarded()) { return RHITestResult::fail("Could not parse the reflection fixture"); }

        // Tilt the receiver 22.5 degrees around X. Its mirror ray travels
        // toward the emitter above the visible orthographic rectangle.
        constexpr float halfAngle = -0.19634954084936207f;
        asset["nodes"][0]["rotation"] = {std::sin(halfAngle), 0.0f, 0.0f, std::cos(halfAngle)};
        asset["nodes"][1]["translation"][2] = 1.35f;
        asset["scenes"][0]["nodes"] = {0, 1, 3};
        auto& material = asset["materials"][0];
        material["pbrMetallicRoughness"]["metallicFactor"] = 1.0f;
        material["pbrMetallicRoughness"]["roughnessFactor"] = 0.0f;
        material["extensions"]["KHR_materials_specular"]["specularFactor"] = 1.0f;
        {
            std::ofstream output(fixture);
            output << asset.dump();
            if (!output) { return RHITestResult::fail("Could not write the reflection fixture"); }
        }
        scene::SceneDocument document;
        if (!document.load(fixture)) { return RHITestResult::fail(document.lastLoadResult().error); }
        std::string log;
        // Keep only the off-screen emitter in the tracing representation so
        // this fixture tests incoming radiance independently of receiver self-hits.
        if (!writeAnalyticBoxField(field, document, log, 1) || !writeAnalyticBoxCards(cards, document, 1, log)) {
            return RHITestResult::fail(log);
        }

        RenderGraph graph;
        if (!makeGraph(graph, field, cards)) { return RHITestResult::fail("Could not build the software reflection graph"); }
        bench::ValidationRecorder validation;
        auto device = render::createDevice({.applicationName = "Lumen software reflection regression",
            .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true, .enableMeshShader = true, .enableTaskShader = true,
            .enableTaskShaderSubgroupBallot = true, .enableGeometryShader = true, .validationSink = validation.sink()});
        if (!device) { return RHITestResult::fail("Could not initialize the Lumen reflection test device"); }
        auto& gpu = **device;
        const auto& caps = gpu.capabilities();
        if (!caps.meshShader || !caps.taskShader || !caps.geometryShader || !caps.shaderStorageImageExtendedFormats) {
            return RHITestResult::skip("Lumen reflection fixture requires mesh, task and geometry shaders and extended storage image formats");
        }
        RenderWorld world;
        world.setScene(&document);
        world.setEnvironment({.enabled = false, .visible = false});
        environment::WorldEnvironment environment;
        environment.moon.enabled = false;
        environment.sun.enabled = false;
        if (!world.setWorldEnvironment(environment)) { return RHITestResult::fail("Could not disable the fixture sun"); }

        RenderView view;
        ViewCamera camera;
        camera.eye = {0.0f, 0.0f, 3.0f};
        camera.orthographic = true;
        camera.orthoHeight = 1.25f;
        camera.farPlane = 8.0f;
        if (!view.setCamera(camera) || !view.setRenderResolution(kAtlasEdge, kAtlasEdge)) {
            return RHITestResult::fail("Could not initialize the reflection test camera");
        }
        HistoryResourceManager history;
        if (!history.initialize(gpu)) { return RHITestResult::fail("Could not initialize reflection graph history"); }
        CardUpdateInspector inspector;
        RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        executor.bindRenderView(&view);
        executor.setDebugObserver(&inspector);
        if (!executor.compile(gpu, graph, kAtlasEdge, kAtlasEdge, RenderGraphCompileOptions{.enableDebugReadback = true}, log)) {
            return RHITestResult::fail("Software reflection graph compile: " + log);
        }
        const RenderGraphSubmitDesc submission{.graphicsQueue = gpu.getQueue(QueueType::Graphics),
            .historyResources = &history, .recordingWorkerLimit = 1};
        const auto runFrame = [&](float roughness, bool unlit, uint32_t minHits) {
            auto frame = captureFrame(inspector, executor, submission, log);
            if (frame) { log = verifyFrame(*frame, roughness, unlit, minHits); }
            return frame && log.empty();
        };

        if (!runFrame(0.0f, false, 144u)) { return RHITestResult::fail("Mirror: " + log); }
        auto receiver = document.materials()[0];
        receiver.roughnessFactor = 0.2f;
        if (!document.setMaterialProperties(0, receiver)) { return RHITestResult::fail("Could not set the rough receiver material"); }
        if (!runFrame(0.2f, false, 64u)) { return RHITestResult::fail("Rough reflection: " + log); }

        receiver.roughnessFactor = 0.65f;
        if (!document.setMaterialProperties(0, receiver)) { return RHITestResult::fail("Could not set the roughness cutoff fixture"); }
        if (!runFrame(0.65f, false, 0u)) { return RHITestResult::fail("Roughness cutoff: " + log); }

        receiver.roughnessFactor = 0.0f;
        receiver.unlit = true;
        if (!document.setMaterialProperties(0, receiver)) { return RHITestResult::fail("Could not set the unlit receiver material"); }
        if (!runFrame(0.0f, true, 0u)) { return RHITestResult::fail("Unlit receiver: " + log); }
        if (validation.failed()) { return RHITestResult::fail("Lumen reflection Vulkan validation: " + validation.snapshot().dump()); }
        return RHITestResult::pass("The mirror and rough receiver reflected an off-screen emissive card. Roughness cutoff, unlit surfaces and background bypassed tracing.");
    }

private:
    static constexpr float kMaxRoughness = 0.25f;
    using ReflectionFrame = std::array<CardCapture, 2>;
    using Pixel = std::array<float, 4>;

    static bool makeGraph(render::RenderGraph& graph, const std::filesystem::path& field, const std::filesystem::path& cards)
    {
        using render::makeRenderGraphFieldName;
        const auto edge = [&](std::string_view from, std::string_view source, std::string_view to, std::string_view target) {
            return graph.addEdge(makeRenderGraphFieldName(from, source), makeRenderGraphFieldName(to, target));
        };
        if (!graph.addNode("VisibilityBufferPass", "VBuffer", {{"visualization", "none"}, {"enableClusterRtx", false}}) ||
            !graph.addNode("VisibilityBufferDeferredPass", "Deferred", {{"materialBinning", false}, {"outputLinear", true},
                {"exportLumenReceivers", true}, {"exportLumenSpecular", true}}) ||
            !graph.addNode("GlobalDistanceFieldPagesPass", "GDF", {{"distanceFieldPaths", {field.generic_string()}},
                {"center", {0.0f, 0.5f, 0.75f}}, {"extent", 2.5f}, {"resolution", 112}, {"maxPhysicalPages", 1728},
                {"quantized", true}, {"buildObjectGrid", true}, {"persistentPages", true}}) ||
            !graph.addNode("LumenSurfaceCachePass", "SurfaceCache", {{"meshCardPaths", {cards.generic_string()}},
                {"atlasSize", kAtlasEdge}, {"tileResolution", kTileEdge}, {"persistentCards", true}}) ||
            !graph.addNode("LumenReflectionPass", "Reflections", {{"maxTraceDistance", 4.0f}, {"maxRoughness", kMaxRoughness}})) {
            return false;
        }
        for (const auto* name : {"visibility", "depth", "rasterInfo"}) {
            if (!edge("VBuffer", name, "Deferred", name)) { return false; }
        }
        for (const auto* name : {"pageTable", "pageAtlas", "coarseMip", "domain", "objectGrid", "objectGridScene"}) {
            if (!edge("GDF", name, "Reflections", name)) { return false; }
        }
        constexpr std::array surfaceBindings{std::pair{"cards", "cards"}, std::pair{"scene", "cardScene"},
            std::pair{"depth", "cardDepth"}, std::pair{"lighting", "cardLighting"}};
        for (const auto& [from, to] : surfaceBindings) {
            if (!edge("SurfaceCache", from, "Reflections", to)) { return false; }
        }
        return edge("Deferred", "lumenPosition", "Reflections", "position") &&
            edge("Deferred", "lumenNormal", "Reflections", "normal") &&
            edge("Deferred", "lumenSpecular", "Reflections", "specular") &&
            edge("Deferred", "color", "Reflections", "sceneColor") && graph.markOutput("Reflections.color");
    }

    static std::optional<ReflectionFrame> captureFrame(CardUpdateInspector& inspector, render::RenderGraphExecutor& executor,
        const render::RenderGraphSubmitDesc& submission, std::string& log)
    {
        inspector.request = {{"method", "capture.batch"}, {"params", {{"batches", debug::DebugValue::array({
            {{"pass", "Deferred"}, {"resources", debug::DebugValue::array({cardAtlasCapture("Deferred.lumenPosition"),
                cardAtlasCapture("Deferred.lumenSpecular"), cardAtlasCapture("Deferred.color")})}},
            {{"pass", "Reflections"}, {"resources", debug::DebugValue::array({cardAtlasCapture("Reflections.reflections"),
                cardAtlasCapture("Reflections.trace"), cardAtlasCapture("Reflections.color")})}},
        })}}}};
        if (const auto result = executor.execute(submission); !result) {
            log = std::string(render::resultToString(result));
            return std::nullopt;
        }
        if (!executor.waitForSubmittedWork()) { log = "Could not wait for the reflection frame"; return std::nullopt; }
        inspector.runtime.poll();
        if (inspector.response.value("status", "") != "ok") { log = inspector.response.dump(); return std::nullopt; }
        auto& core = inspector.runtime.core();
        const auto& jobs = inspector.response.at("result").at("jobs");
        ReflectionFrame captures;
        for (size_t index = 0; index < captures.size(); ++index) {
            const auto id = jobs.at(index).at("job").get<std::string>();
            captures[index] = core.completedCapture(id);
            if (!captures[index]) {
                log = core.dispatch({{"method", "jobs.get"}, {"params", {{"job", id}}}}).dump();
                return std::nullopt;
            }
        }
        return captures;
    }

    static Pixel readHalfPixel(const debug::DebugArtifact& artifact, size_t pixel)
    {
        const auto halfValues = readSurfaceValue<std::array<uint16_t, 4>>(artifact.bytes, pixel);
        return {halfToFloat(halfValues[0]), halfToFloat(halfValues[1]), halfToFloat(halfValues[2]), halfToFloat(halfValues[3])};
    }

    static std::string verifyFrame(const ReflectionFrame& frame, float roughness, bool unlit, uint32_t minHits)
    {
        constexpr std::array strides{std::array{16u, 8u, 16u}, std::array{8u, 16u, 16u}};
        for (size_t group = 0; group < frame.size(); ++group) {
            if (frame[group]->artifacts.size() != strides[group].size()) { return "Incomplete reflection readback"; }
            for (size_t output = 0; output < strides[group].size(); ++output) {
                if (frame[group]->artifacts[output].bytes.size() != kAtlasPixels * strides[group][output]) {
                    return "Unexpected reflection readback layout";
                }
            }
        }
        const auto& deferred = frame[0]->artifacts;
        const auto& reflections = frame[1]->artifacts;
        constexpr Pixel untraced{-1.0f, 8.0f, -1.0f, 0.0f};
        uint32_t hits = 0, background = 0;
        for (uint32_t y = 0; y < kAtlasEdge; ++y) {
            for (uint32_t x = 0; x < kAtlasEdge; ++x) {
                const size_t pixel = size_t(y) * kAtlasEdge + x;
                const auto position = readSurfaceValue<Pixel>(deferred[0].bytes, pixel);
                const auto specular = readHalfPixel(deferred[1], pixel);
                const auto baseColor = readSurfaceValue<Pixel>(deferred[2].bytes, pixel);
                const auto raw = readHalfPixel(reflections[0], pixel);
                const auto trace = readSurfaceValue<Pixel>(reflections[1].bytes, pixel);
                const auto color = readSurfaceValue<Pixel>(reflections[2].bytes, pixel);
                const bool center = x >= 52 && x < 76 && y >= 52 && y < 76;
                if (!center && position[3] <= 0.0f) { ++background; }
                if (center && !unlit && (!(position[3] > 0.0f) || !(specular[0] > 0.0f) ||
                    std::abs(specular[3] - roughness) >= 0.001f)) {
                    return "Deferred did not export the expected reflective material at the receiver";
                }
                if (center && unlit && (specular[0] != 0.0f || specular[1] != 0.0f || specular[2] != 0.0f)) {
                    return "Deferred exported a specular lobe for the unlit receiver";
                }
                const bool traced = !unlit && position[3] > 0.0f && specular[3] < kMaxRoughness &&
                    (specular[0] > 0.0f || specular[1] > 0.0f || specular[2] > 0.0f);
                if (!traced && (raw != Pixel{} || trace != untraced)) {
                    return "Invalid, unlit or roughness-filtered pixel produced a reflection ray";
                }
                if (traced) {
                    const float fade = std::clamp((kMaxRoughness - specular[3]) * 10.0f, 0.0f, 1.0f);
                    if (!(std::abs(raw[3] - fade) < 0.001f)) { return "Incorrect roughness fade"; }
                    if (center && trace[0] > 0.5f && trace[0] < 3.0f && trace[1] == 2.0f && trace[2] == 0.0f &&
                        trace[3] == 3.0f && raw[0] > 1.0f && raw[1] > 0.0f && raw[2] > 0.0f) {
                        ++hits;
                    }
                }
                for (size_t channel = 0; channel < 3; ++channel) {
                    const float expected = baseColor[channel] + std::max(raw[channel] * raw[3] * specular[channel], 0.0f);
                    if (!(std::abs(color[channel] - expected) <= 0.0001f)) { return "Reflection composite differs from scene color plus weighted radiance"; }
                }
                if (color[3] != baseColor[3]) { return "Reflection composite changed scene alpha"; }
            }
        }
        if (background < 64) { return "The reflection fixture did not contain uncovered pixels"; }
        if (hits < minHits) { return "Too few reflected off-screen emitter pixels: " + std::to_string(hits); }
        return {};
    }
};

METALLIC_REGISTER_RHI_TEST(LumenReflectionTest);

} // namespace
} // namespace metallic::tests
