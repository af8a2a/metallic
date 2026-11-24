#include "MeshDistanceFieldFixture.h"
#include "RHITest.h"
#include "Runtime/Render/Core/ColorSpace.h"
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
#include "../../Shaders/Interop/LumenCardUpdate.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <spdlog/spdlog.h>

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

std::array<float, 3> cardCenterWorldPosition(const CapturedCard& card, uint16_t encodedDepth)
{
    const float depth = float(encodedDepth) / float(UINT16_MAX);
    // An even tile's center texel is one half-pixel from the midpoint.
    std::array<float, 3> position{};
    for (size_t axis = 0; axis < position.size(); ++axis) {
        position[axis] = card.origin[axis] + card.axisX[axis] * (card.axisX[3] / kTileEdge) -
            card.axisY[axis] * (card.axisY[3] / kTileEdge) + card.axisZ[axis] * ((1.0f - 2.0f * depth) * card.axisZ[3]);
    }
    return position;
}

float centeredRectCosineIntegral(float halfWidth, float halfHeight, float distance)
{
    const float edgeX = std::hypot(halfWidth, distance);
    const float edgeY = std::hypot(halfHeight, distance);
    return 2.0f * (halfWidth / edgeX * std::atan(halfHeight / edgeX) +
        halfHeight / edgeY * std::atan(halfWidth / edgeY));
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

        const auto& cards = capture->artifacts[4].bytes;
        uint32_t front = 0;
        while (front < kCardCount && readSurfaceValue<CapturedCard>(cards, front).axisZ[2] < 0.9999f) {
            ++front;
        }
        if (front == kCardCount) { return RHITestResult::fail("Missing unrotated receiver card for Rect lighting"); }
        const size_t pixel = tileCenter(front);
        const auto position = cardCenterWorldPosition(readSurfaceValue<CapturedCard>(cards, front),
            readSurfaceValue<uint16_t>(depth, pixel));
        scene::LightingSettings lighting = document.lighting();
        scene::RectLight rect;
        rect.position = {position[0], position[1], position[2] + 1.0f};
        rect.direction = {0.0f, 0.0f, -1.0f};
        rect.width = rect.height = 2.0f;
        rect.luminance = 1.0f;
        lighting.rectLights = {rect};
        if (!preview.setLighting(lighting) || !preview.render(graph, kAtlasEdge, kAtlasEdge, "SurfaceCache.lighting")) {
            return RHITestResult::fail("Basic Resolve Rect lighting: " + preview.lastLog());
        }
        using HalfPixel = std::array<uint16_t, 4>;
        if (preview.readbackFormat() != Format::RGBA16Sfloat ||
            preview.readbackBytes().size() != kAtlasPixels * sizeof(HalfPixel)) {
            return RHITestResult::fail("Unexpected basic Resolve lighting atlas format");
        }
        const auto before = readSurfaceValue<HalfPixel>(capture->artifacts[3].bytes, pixel);
        const auto after = readSurfaceValue<HalfPixel>(preview.readbackBytes(), pixel);
        const auto albedo = readSurfaceValue<std::array<uint8_t, 4>>(capture->artifacts[0].bytes, pixel);
        const float normalizedIntegral = centeredRectCosineIntegral(1.0f, 1.0f, 1.0f) / 3.14159265358979323846f;
        for (size_t channel = 0; channel < 3; ++channel) {
            const float encoded = float(albedo[channel]) / 255.0f;
            const float actual = halfToFloat(after[channel]) - halfToFloat(before[channel]);
            const float expected = normalizedIntegral * encoded * encoded;
            if (!(std::abs(actual - expected) < 0.003f)) {
                return RHITestResult::fail(std::format("Basic Resolve Rect contribution is {}, expected {}", actual, expected));
            }
        }
        if (!graph.setNodeProperties(node->id, {{"meshCardPaths", {cardPath.generic_string()}},
                {"atlasSize", kAtlasEdge}, {"tileResolution", kTileEdge}, {"persistentCards", true},
                {"rectLightSamples", 1u}, {"maxStochasticDirectLightingCardsPerFrame", kCardCount}})) {
            return RHITestResult::fail("Could not enable stochastic lighting on the basic Surface Cache");
        }
        rect.luminance *= 0.5f;
        lighting.rectLights = {rect, rect};
        if (!preview.setLighting(lighting) || !preview.render(graph, kAtlasEdge, kAtlasEdge, "SurfaceCache.lighting")) {
            return RHITestResult::fail("Basic Resolve stochastic composition: " + preview.lastLog());
        }
        const auto composed = readSurfaceValue<HalfPixel>(preview.readbackBytes(), pixel);
        for (size_t channel = 0; channel < 3; ++channel) {
            if (std::abs(halfToFloat(composed[channel]) - halfToFloat(after[channel])) > 0.003f) {
                return RHITestResult::fail("Basic Resolve stochastic composition changed irradiance for identical Rects");
            }
        }
        if (!preview.render(graph, kAtlasEdge, kAtlasEdge, "SurfaceCache.stochasticDirectHistory")) {
            return RHITestResult::fail("Basic Resolve stochastic refresh: " + preview.lastLog());
        }
        if (preview.readbackFormat() != Format::RGBA32Sfloat ||
            readSurfaceValue<std::array<float, 4>>(preview.readbackBytes(), pixel)[3] != 2.0f) {
            return RHITestResult::fail("Basic Resolve did not accumulate a second Rect batch");
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
        const auto linearHits = hits;
        surfaceProperties["objectCardRanges"] = true;
        if (!graph.setNodeProperties(surfaceId, surfaceProperties) || !captureRays(hits)) {
            return RHITestResult::fail("Object-scoped Surface Cache tracing: " + log);
        }
        if (std::memcmp(linearHits.data(), hits.data(), sizeof(hits)) != 0) {
            return RHITestResult::fail("Object-scoped Surface Cache tracing changed a fixed-ray result");
        }
        constexpr uint32_t rangeWords = kLumenCardSceneRangeBase + 2 * kLumenCardSceneRangeStride;
        const auto rangeCapture = capturePreview.captureAfterPass(graph, "SurfaceCache", debug::DebugValue::array({
            {{"id", "SurfaceCache.scene"}, {"layout", "u32"}, {"count", rangeWords}},
        }), output, log);
        if (!rangeCapture || rangeCapture->artifacts.size() != 1 ||
            rangeCapture->artifacts[0].bytes.size() != rangeWords * sizeof(uint32_t)) {
            return RHITestResult::fail("Object-scoped Surface Cache metadata: " + log);
        }
        const auto table = readSurfaceValue<std::array<uint32_t, rangeWords>>(rangeCapture->artifacts[0].bytes, 0);
        if (table[kLumenCardSceneHeaderWords] != kLumenCardSceneRangeTag || table[kLumenCardSceneRangeCount] != 2) {
            return RHITestResult::fail("Surface Cache range table must contain both fixture objects");
        }
        std::array<bool, kCardCount> visited{};
        for (uint32_t word = kLumenCardSceneRangeBase; word < rangeWords; word += kLumenCardSceneRangeStride) {
            const uint32_t object = table[word], first = table[word + 1], end = table[word + 2];
            if (first >= end || end > kCardCount ||
                (word > kLumenCardSceneRangeBase && table[word - kLumenCardSceneRangeStride] >= object)) {
                return RHITestResult::fail("Surface Cache range table has unsorted objects or invalid bounds");
            }
            for (uint32_t slot = first; slot < end; ++slot) {
                if (visited[slot] || readSurfaceValue<CapturedCard>(cards, slot).metadata[0] != object) {
                    return RHITestResult::fail("Surface Cache range table includes an unrelated or repeated atlas slot");
                }
                visited[slot] = true;
            }
        }
        if (!std::ranges::all_of(visited, [](bool value) { return value; })) {
            return RHITestResult::fail("Surface Cache range table omitted an atlas slot");
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
constexpr size_t kSurfaceStochasticHistory = kSurfaceUpdateFlags + 1;
constexpr size_t kSurfaceLighting = 4;
constexpr size_t kSurfaceDirectIrradiance = 5;
constexpr size_t kRadiosityUpdateStats = 2;
constexpr size_t kRadiosityCardState = 3;

using CardCapture = std::shared_ptr<const debug::DebugCapture>;

struct CardUpdateFrame {
    CardCapture surface, fullSurface, radiosity, denseRadiosity, gdf;
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
    enum class Variant { ActiveProbes, SplitTracing, SplitDetail, CompactAtlas, CompactAtlasSaturated,
        SharedCacheAtlas, SharedCacheAtlasSaturated };

    LumenProbeDispatchComparison(const char* testName, Variant variant)
        : variant_(variant), candidatePass_(compactAtlas() ? "CompactGather" : splitTracing() ? "SplitGather" : "ActiveGather")
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
        auto batches = debug::DebugValue::array({
            {{"pass", "Gather"}, {"resources", captureResources("Gather")}},
            {{"pass", candidatePass_}, {"resources", captureResources(candidatePass_)}},
        });
        if (sharedPlacement()) {
            batches.push_back({{"pass", "DenseCache"}, {"resources", captureCache("DenseCache")}});
            batches.push_back({{"pass", "CompactCache"}, {"resources", captureCache("CompactCache")}});
        }
        inspector.request = {{"method", "capture.batch"}, {"params", {{"batches", std::move(batches)}}}};

        auto emptyCamera = camera;
        emptyCamera.eye[0] += 20.0f;
        emptyCamera.center[0] += 20.0f;
        auto shiftedCamera = camera;
        shiftedCamera.eye[0] += kCacheCellSize;
        shiftedCamera.center[0] += kCacheCellSize;
        constexpr size_t kInitial = 0;
        const size_t emptyFrame = splitTracing() ? 4 : 2;
        const size_t restoredFrame = emptyFrame + 1;
        RayHandoffCoverage coverage;
        for (size_t frame = kInitial; frame <= restoredFrame; ++frame) {
            const bool empty = frame == emptyFrame;
            const bool scroll = sharedPlacement() && frame == 3;
            if (scroll && !view.setCamera(shiftedCamera)) { return RHITestResult::fail("Could not scroll the cache clipmap"); }
            if (frame >= emptyFrame) {
                if (!view.setCamera(empty ? emptyCamera : camera)) { return RHITestResult::fail("Could not change camera"); }
                view.cameraCut();
            }
            if (!preview.render(graph, kSingleBounceEdge, kSingleBounceEdge, "Gather.indirectDiffuse", false)) {
                return RHITestResult::fail(preview.lastLog());
            }
            log = compareFrame(inspector, empty, frame > kInitial && frame < emptyFrame, scroll, coverage);
            if (!log.empty()) { return RHITestResult::fail("Frame " + std::to_string(frame) + ": " + log); }
        }
        if (splitTracing() && !coverage.mixedScreenAndGDF) {
            return RHITestResult::fail("Expected both accepted screen rays and compacted GDF rays with valid history");
        }
        if (splitDetail() && (!coverage.detailHit || !coverage.detailMiss)) {
            return RHITestResult::fail("Expected both accepted and rejected Mesh SDF rays");
        }
        if (compactAtlas() && (!coverage.adaptiveProbes || (saturatedAtlas() && !coverage.adaptiveOverflow))) {
            return RHITestResult::fail("Fixture did not exercise the requested adaptive admission and budget");
        }
        if (sharedPlacement() && (!coverage.cacheSampled || !coverage.cacheScrolled)) {
            return RHITestResult::fail("Fixture did not sample the shared cache and reuse cells as its clipmap scrolled");
        }
        if (comparesSharedInterpolation() && !coverage.signedEdgeTap) {
            return RHITestResult::fail("Fixture did not exercise a signed edge interpolation tap");
        }
        if (validation.failed()) {
            return RHITestResult::fail("Probe dispatch Vulkan validation: " + validation.snapshot().dump());
        }
        if (splitDetail()) {
            return RHITestResult::pass("Fused and split Detail matched across Screen, Mesh SDF and compacted GDF");
        }
        if (sharedPlacement()) {
            return RHITestResult::pass("Verified world cells from shared probes, cache sampling and scrolling reuse" +
                (comparesSharedInterpolation() ? "; equal-weight alternate taps: " + std::to_string(coverage.alternateAdaptiveTaps) : ""));
        }
        if (compactAtlas()) {
            return RHITestResult::pass("Verified packed probe membership, atlas addresses, ray dispatch and empty-frame recovery");
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
        bool adaptiveProbes = false;
        bool adaptiveOverflow = false;
        bool cacheSampled = false;
        bool cacheScrolled = false;
        bool signedEdgeTap = false;
        uint32_t alternateAdaptiveTaps = 0;
    };

    static constexpr uint32_t kTileAxis = render::builtin_pass::lumenScreenProbeTiles(kSingleBounceEdge);
    static constexpr uint32_t kSlotsPerTile = render::builtin_pass::kLumenScreenProbeSlotsPerTile;
    static constexpr uint32_t kTileCount = kTileAxis * kTileAxis;
    static constexpr uint32_t kAdaptivePerTile = kSlotsPerTile - 1;
    static constexpr uint32_t kProbeSlots = kTileCount * kSlotsPerTile;
    static constexpr uint32_t kRaysPerSlot = 8 * 8;
    static constexpr uint32_t kProbeRays = kProbeSlots * kRaysPerSlot;
    static constexpr uint32_t kSHAtlasEdge = kTileAxis * 3;
    static constexpr uint32_t kCacheGridResolution = 16;
    static constexpr uint32_t kCacheAtlasProbeAxis = 16;
    static constexpr uint32_t kCacheGridCells = kCacheGridResolution * kCacheGridResolution * kCacheGridResolution;
    static constexpr uint32_t kCacheSlots = kCacheAtlasProbeAxis * kCacheAtlasProbeAxis;
    static constexpr float kCacheWorldExtent = 4.0f;
    static constexpr float kCacheCellSize = 2.0f * kCacheWorldExtent / kCacheGridResolution;
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
    static constexpr size_t kTraceRecords = kBufferedResources.size() + kTextures.size();
    static constexpr size_t kWorkState = kTraceRecords + 1;
    static constexpr size_t kOrigins = kWorkState + 4;
    static constexpr size_t kAdaptiveCount = kOrigins + 1;

    bool splitTracing() const { return variant_ != Variant::ActiveProbes; }
    bool splitDetail() const { return variant_ == Variant::SplitDetail; }
    bool compactAtlas() const { return variant_ == Variant::CompactAtlas || saturatedAtlas() || sharedPlacement(); }
    bool saturatedAtlas() const { return variant_ == Variant::CompactAtlasSaturated || variant_ == Variant::SharedCacheAtlasSaturated; }
    bool sharedPlacement() const { return variant_ == Variant::SharedCacheAtlas || variant_ == Variant::SharedCacheAtlasSaturated; }
    bool comparesSharedInterpolation() const { return sharedPlacement() && !saturatedAtlas(); }
    uint32_t adaptiveBudget() const { return saturatedAtlas() ? 1u : kTileCount * kAdaptivePerTile; }
    uint32_t packedCapacity() const { return kTileCount + adaptiveBudget(); }
    uint32_t packedAtlasHeight() const { return (packedCapacity() + kTileAxis - 1u) / kTileAxis; }

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
        if (compactAtlas()) { properties["splitProbeTracing"] = true; }
        if (sharedPlacement()) { properties["radianceCache"] = true; }
        if (comparesSharedInterpolation()) {
            properties["probeInterpolationReadback"] = {0, 0, kSingleBounceEdge, kSingleBounceEdge};
        }
        if (splitDetail()) {
            const auto* fields = graph.findNode("GlobalDistanceFieldPages");
            if (!fields || !graph.addEdge("VBuffer.depth", "Gather.detailDepth")) { return false; }
            properties["detailTracing"] = true;
            properties["detailDistanceFieldPaths"] = fields->properties.at("distanceFieldPaths");
        }
        if (!graph.setNodeProperties(gather->id, properties)) { return false; }
        properties["activeProbeDispatch"] = true;
        if (splitTracing()) { properties["splitProbeTracing"] = true; }
        if (compactAtlas()) {
            properties["compactProbeAtlas"] = true;
            properties["adaptiveProbeAllocationFraction"] = double(adaptiveBudget()) / kTileCount;
        }
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
        return !sharedPlacement() || wireSharedCache(graph);
    }

    bool wireSharedCache(render::RenderGraph& graph) const
    {
        using render::makeRenderGraphFieldName;
        if (!graph.addNode("LumenScreenProbePlacementPass", "Placement",
                {{"adaptiveProbeAllocationFraction", double(adaptiveBudget()) / kTileCount}}) ||
            !graph.addEdge("Deferred.lumenPosition", "Placement.position") ||
            !graph.addEdge("Deferred.lumenNormal", "Placement.normal")) {
            return false;
        }
        for (const std::string_view field : {"probePositions", "probeOrigins", "probeAdaptiveCount",
                 "probeTileAdaptiveCounts", "probeTileAdaptiveIndices"}) {
            if (!graph.addEdge(makeRenderGraphFieldName("Placement", field), makeRenderGraphFieldName(candidatePass_, field))) {
                return false;
            }
        }
        const render::RenderGraphProperties cacheProperties{
            {"gridResolution", kCacheGridResolution}, {"clipmapCount", 1u}, {"probeResolution", 8u},
            {"atlasProbesPerAxis", kCacheAtlasProbeAxis}, {"traceBudget", kCacheSlots}, {"worldExtent", kCacheWorldExtent},
            {"maxTraceDistance", 5.0f}};
        auto compactProperties = cacheProperties;
        compactProperties["compactProbeAtlas"] = true;
        if (!graph.addNode("LumenRadianceCachePass", "DenseCache", cacheProperties) ||
            !graph.addNode("LumenRadianceCachePass", "CompactCache", compactProperties) ||
            !graph.addEdge("Deferred.lumenNormal", "DenseCache.normal") ||
            !graph.addEdge("DenseCache.placedProbes", "Gather.placedProbes") ||
            !graph.addEdge("Placement.probePositions", "CompactCache.probePositions") ||
            !graph.addEdge("Placement.probeAdaptiveCount", "CompactCache.probeAdaptiveCount")) {
            return false;
        }
        constexpr std::array<std::string_view, 11> inputs{"position", "pageTable", "pageAtlas", "coarseMip", "domain",
            "objectGrid", "objectGridScene", "cards", "cardScene", "cardDepth", "cardLighting"};
        const auto edges = graph.edges();
        for (const auto& edge : edges) {
            if (edge.dstPass == "Gather" && std::ranges::find(inputs, edge.dstField) != inputs.end()) {
                const auto source = makeRenderGraphFieldName(edge.srcPass, edge.srcField);
                if (!graph.addEdge(source, makeRenderGraphFieldName("DenseCache", edge.dstField)) ||
                    !graph.addEdge(source, makeRenderGraphFieldName("CompactCache", edge.dstField))) {
                    return false;
                }
            }
        }
        // In the parity case both Gather layouts sample the same physical WRC atlas.
        // DenseCache still executes and captures residency because it supplies legacy placement.
        const std::string_view baselineCache = comparesSharedInterpolation() ? "CompactCache" : "DenseCache";
        for (const std::string_view output : {"cacheIndirection", "cacheClipmaps", "cacheRadiance"}) {
            if (!graph.addEdge(makeRenderGraphFieldName(baselineCache, output), makeRenderGraphFieldName("Gather", output)) ||
                !graph.addEdge(makeRenderGraphFieldName("CompactCache", output), makeRenderGraphFieldName(candidatePass_, output))) {
                return false;
            }
        }
        return true;
    }

    static debug::DebugValue captureCache(std::string_view pass)
    {
        using render::makeRenderGraphFieldName;
        return debug::DebugValue::array({cardBufferCapture(makeRenderGraphFieldName(pass, "cacheIndirection"), kCacheGridCells),
            cardBufferCapture(makeRenderGraphFieldName(pass, "probeFrames"), 2u * kCacheSlots),
            cardBufferCapture(makeRenderGraphFieldName(pass, "cacheClipmaps"), 12u),
            cardBufferCapture(makeRenderGraphFieldName(pass, "previousClipmaps"), 12u),
            cardBufferCapture(makeRenderGraphFieldName(pass, "updateState"), 32u),
            {{"id", makeRenderGraphFieldName(pass, "position")}, {"roi", {{"x", 0}, {"y", 0},
                {"width", kSingleBounceEdge}, {"height", kSingleBounceEdge}}}}});
    }

    debug::DebugValue captureResources(std::string_view pass) const
    {
        using render::makeRenderGraphFieldName;
        auto resources = debug::DebugValue::array();
        const bool packed = compactAtlas() && pass == candidatePass_;
        for (const auto& [field, words] : kBufferedResources) {
            const uint32_t count = packed ? words / kProbeSlots * packedCapacity() : words;
            const auto name = sharedPlacement() && !packed && std::string_view(field) == "probePositions" ? "placedProbes" : field;
            resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, name), count));
        }
        for (const auto& [field, layout] : kTextures) {
            const bool packedSH = packed && std::string_view(field).starts_with("probeSH");
            resources.push_back({{"id", makeRenderGraphFieldName(pass, field)},
                {"roi", {{"x", 0}, {"y", 0}, {"width", packedSH ? layout[0] / 3u : layout[0]},
                    {"height", packedSH ? packedAtlasHeight() : layout[1]}}}});
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
            if (packed) {
                resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "probeOrigins"), packedCapacity()));
                resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "probeAdaptiveCount"), 1));
                resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "probeTileAdaptiveCounts"), kTileCount));
                resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "probeTileAdaptiveIndices"),
                    kTileCount * kAdaptivePerTile));
            }
        }
        if (comparesSharedInterpolation()) {
            resources.push_back(cardBufferCapture(makeRenderGraphFieldName(pass, "probeInterpolationReadback"),
                8u * kSingleBounceEdge * kSingleBounceEdge));
        }
        return resources;
    }

    using DispatchArgs = std::array<uint32_t, 3>;

    struct Interpolation {
        std::array<uint32_t, 4> indices;
        std::array<float, 4> weights;
    };

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

    std::string compareCacheCells(const debug::DebugCapture& dense, const debug::DebugCapture& compact,
        const std::vector<uint8_t>& positions, uint32_t admitted, bool scroll, RayHandoffCoverage& coverage) const
    {
        struct Clipmap {
            std::array<float, 4> cornerAndCellSize, traceSettings;
            std::array<uint32_t, 4> layout;
        };
        constexpr std::array<size_t, 6> sizes{kCacheGridCells * sizeof(uint32_t),
            kCacheSlots * 2 * sizeof(uint32_t), sizeof(Clipmap), sizeof(Clipmap), 32 * sizeof(uint32_t),
            size_t(kSingleBounceEdge) * kSingleBounceEdge * 4 * sizeof(float)};
        if (dense.artifacts.size() != sizes.size() || compact.artifacts.size() != sizes.size() ||
            dense.snapshot.evidence.execution != compact.snapshot.evidence.execution) {
            return "Missing same-frame WRC evidence";
        }
        for (size_t index = 0; index < sizes.size(); ++index) {
            if (dense.artifacts[index].bytes.size() != sizes[index] || compact.artifacts[index].bytes.size() != sizes[index]) {
                return "Unexpected WRC capture size";
            }
        }
        constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
        const auto clipmap = readSurfaceValue<Clipmap>(compact.artifacts[2].bytes, 0);
        if (clipmap.cornerAndCellSize[3] != kCacheCellSize || clipmap.layout[0] != kCacheGridResolution || clipmap.layout[1] != 1u) {
            return "Unexpected WRC clipmap layout";
        }
        std::array<bool, kCacheGridCells> expected{};
        for (uint32_t index = 0; index < kTileCount + admitted; ++index) {
            const auto pixel = readSurfaceValue<std::array<uint32_t, 2>>(positions, index);
            if (pixel[0] == invalid) { continue; }
            if (pixel[0] >= kSingleBounceEdge || pixel[1] >= kSingleBounceEdge) { return "Invalid shared probe pixel"; }
            const auto receiver = readSurfaceValue<std::array<float, 4>>(compact.artifacts[5].bytes,
                pixel[1] * kSingleBounceEdge + pixel[0]);
            if (!(receiver[3] > 0.0f)) { return "Shared probe does not address a receiver"; }
            std::array<int32_t, 3> corner{};
            bool inside = true;
            for (size_t axis = 0; axis < corner.size(); ++axis) {
                const float coord = (receiver[axis] - clipmap.cornerAndCellSize[axis]) / kCacheCellSize - 0.5f;
                inside &= 0.0f < coord && coord < float(kCacheGridResolution - 1u);
                corner[axis] = int32_t(std::floor(coord));
            }
            if (!inside) { continue; }
            for (uint32_t offset = 0; offset < 8u; ++offset) {
                const uint32_t x = uint32_t(corner[0]) + (offset >> 2u);
                const uint32_t y = uint32_t(corner[1]) + ((offset >> 1u) & 1u);
                const uint32_t z = uint32_t(corner[2]) + (offset & 1u);
                expected[x + (y + z * kCacheGridResolution) * kCacheGridResolution] = true;
            }
        }
        const uint32_t marked = uint32_t(std::ranges::count(expected, true));
        const auto& counters = compact.artifacts[4].bytes;
        constexpr size_t kMarked = 24, kReused = 25, kOverflow = 28;
        if (readSurfaceValue<uint32_t>(counters, kMarked) != marked || readSurfaceValue<uint32_t>(counters, kOverflow) != 0u) {
            return "WRC requests do not match the marked cells from the shared positions";
        }
        std::array<bool, kCacheSlots> usedSlots{};
        std::array<uint32_t, kCacheGridCells> lastUsed{};
        uint32_t latestUse = 0;
        for (uint32_t cell = 0; cell < kCacheGridCells; ++cell) {
            const uint32_t slot = readSurfaceValue<uint32_t>(compact.artifacts[0].bytes, cell);
            const uint32_t denseSlot = readSurfaceValue<uint32_t>(dense.artifacts[0].bytes, cell);
            if (!saturatedAtlas() && (slot == invalid) != (denseSlot == invalid)) {
                return "Unsaturated shared WRC changed the resident world-cell set";
            }
            if (slot == invalid) { continue; }
            if (slot >= kCacheSlots || usedSlots[slot]) { return "Invalid or aliased resident WRC slot"; }
            usedSlots[slot] = true;
            lastUsed[cell] = readSurfaceValue<uint32_t>(compact.artifacts[1].bytes, slot * 2u + 1u);
            latestUse = std::max(latestUse, lastUsed[cell]);
        }
        // A retained cell may still be resident without having been requested this frame.
        // Last-use stamps identify exactly the current request set whenever it is nonempty.
        if (marked != 0u) {
            for (size_t cell = 0; cell < expected.size(); ++cell) {
                if ((latestUse != 0u && lastUsed[cell] == latestUse) != expected[cell]) {
                    return "WRC marked the wrong world cell " + std::to_string(cell);
                }
            }
        }
        if (scroll) {
            const auto previous = readSurfaceValue<Clipmap>(compact.artifacts[3].bytes, 0);
            coverage.cacheScrolled |= clipmap.cornerAndCellSize[0] - previous.cornerAndCellSize[0] == kCacheCellSize &&
                readSurfaceValue<uint32_t>(counters, kReused) != 0u;
        }
        return {};
    }

    std::string compareCompactAtlas(const debug::DebugCapture& baselineCapture, const debug::DebugCapture& candidateCapture,
        bool empty, bool historyExpected, RayHandoffCoverage& coverage) const
    {
        const auto& baseline = baselineCapture.artifacts;
        const auto& candidate = candidateCapture.artifacts;
        const auto& positions = candidate[0].bytes;
        const auto& origins = candidate[kOrigins].bytes;
        const auto& adaptiveCounter = candidate[kAdaptiveCount].bytes;
        const auto& counts = candidate[kOrigins + 2].bytes;
        const auto& indices = candidate[kOrigins + 3].bytes;
        if (origins.size() != packedCapacity() * sizeof(uint32_t) || adaptiveCounter.size() != sizeof(uint32_t) ||
            counts.size() != kTileCount * sizeof(uint32_t) || indices.size() != kTileCount * kAdaptivePerTile * sizeof(uint32_t)) {
            return "Unexpected packed placement buffer size";
        }
        for (size_t i = 0; i < kBufferedResources.size(); ++i) {
            const auto words = kBufferedResources[i].second;
            if (baseline[i].bytes.size() != words * sizeof(uint32_t) ||
                candidate[i].bytes.size() != words / kProbeSlots * packedCapacity() * sizeof(uint32_t)) {
                return "Unexpected probe buffer size for " + std::string(kBufferedResources[i].first);
            }
        }
        for (size_t i = 0; i < 2; ++i) {
            const auto& [field, layout] = kTextures[i];
            if (baseline[kBufferedResources.size() + i].bytes.size() != size_t(layout[0]) * layout[1] * layout[2] ||
                candidate[kBufferedResources.size() + i].bytes.size() != size_t(layout[0] / 3u) * packedAtlasHeight() * layout[2]) {
                return "Unexpected packed irradiance atlas size for " + std::string(field);
            }
        }

        using Pixel = std::array<uint32_t, 2>;
        constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
        uint32_t expectedAdaptive = 0;
        for (uint32_t tile = 0; tile < kTileCount; ++tile) {
            for (uint32_t slot = 1; slot < kSlotsPerTile; ++slot) {
                expectedAdaptive += readSurfaceValue<Pixel>(baseline[0].bytes, tile * kSlotsPerTile + slot)[0] != invalid;
            }
        }
        const uint32_t requested = readSurfaceValue<uint32_t>(adaptiveCounter, 0);
        const uint32_t admitted = std::min(requested, adaptiveBudget());
        if (requested != expectedAdaptive) {
            return "Packed placement requested " + std::to_string(requested) + " adaptive probes against " +
                std::to_string(expectedAdaptive) + " in fixed slots";
        }
        coverage.adaptiveProbes |= admitted != 0u;
        coverage.adaptiveOverflow |= requested > admitted;

        std::array<bool, kProbeSlots> mapped{};
        std::array<bool, kProbeSlots> validOrigin{};
        uint32_t active = 0;
        uint32_t screenHits = 0;
        // Dropping probes may remove world-cache coverage at retained receivers.
        const bool compareLighting = !sharedPlacement() || !saturatedAtlas();
        for (uint32_t packed = 0; packed < kTileCount + admitted; ++packed) {
            const uint32_t origin = readSurfaceValue<uint32_t>(origins, packed);
            if (origin >= kProbeSlots || mapped[origin] ||
                (packed < kTileCount ? origin != packed * kSlotsPerTile : origin % kSlotsPerTile == 0u)) {
                return "Packed probe origin is duplicated or outside its uniform/adaptive range";
            }
            mapped[origin] = true;
            const auto pixel = readSurfaceValue<Pixel>(positions, packed);
            if (pixel != readSurfaceValue<Pixel>(baseline[0].bytes, origin) ||
                (packed >= kTileCount && pixel[0] == invalid)) {
                return "Probe receiver differs after remapping packed probe " + std::to_string(packed);
            }
            if (pixel[0] == invalid) { continue; }
            validOrigin[origin] = true;
            ++active;
            for (uint32_t lane = 0; lane < kRaysPerSlot; ++lane) {
                const auto radiance = readSurfaceValue<std::array<float, 4>>(candidate[1].bytes, packed * kRaysPerSlot + lane);
                screenHits += radiance[3] == 5.0f;
                coverage.cacheSampled |= radiance[3] == 6.0f;
            }
            if (!compareLighting) { continue; }
            for (size_t i = 1; i < kBufferedResources.size(); ++i) {
                const size_t stride = kBufferedResources[i].second / kProbeSlots * sizeof(uint32_t);
                if (std::memcmp(baseline[i].bytes.data() + origin * stride, candidate[i].bytes.data() + packed * stride, stride) != 0) {
                    return "Remapped probe differs in " + std::string(kBufferedResources[i].first) +
                        " at origin " + std::to_string(origin);
                }
            }
            const uint32_t tile = origin / kSlotsPerTile;
            const uint32_t slot = origin % kSlotsPerTile;
            const uint32_t oldX = (tile % kTileAxis) * 3u + slot % 3u;
            const uint32_t oldY = (tile / kTileAxis) * 3u + slot / 3u;
            for (size_t i = 0; i < 2; ++i) {
                const auto& [field, layout] = kTextures[i];
                const auto& oldData = baseline[kBufferedResources.size() + i].bytes;
                const auto& newData = candidate[kBufferedResources.size() + i].bytes;
                const uint32_t planes = layout[0] / kSHAtlasEdge;
                for (uint32_t plane = 0; plane < planes; ++plane) {
                    const uint32_t oldOffset = oldY * layout[0] + plane * kSHAtlasEdge + oldX;
                    const uint32_t newOffset = (packed / kTileAxis) * (layout[0] / 3u) + plane * kTileAxis + packed % kTileAxis;
                    if (std::memcmp(oldData.data() + oldOffset * layout[2], newData.data() + newOffset * layout[2], layout[2]) != 0) {
                        return "Remapped probe differs in " + std::string(field) + " at origin " + std::to_string(origin);
                    }
                }
            }
        }

        std::array<bool, kProbeSlots> listed{};
        uint32_t listedCount = 0;
        for (uint32_t tile = 0; tile < kTileCount; ++tile) {
            const uint32_t count = readSurfaceValue<uint32_t>(counts, tile);
            if (count > kAdaptivePerTile) { return "Tile has too many adaptive probe indices"; }
            for (uint32_t index = 0; index < count; ++index) {
                const uint32_t ordinal = readSurfaceValue<uint32_t>(indices, tile * kAdaptivePerTile + index);
                if (ordinal >= admitted || listed[ordinal]) { return "Tile references an invalid or duplicate adaptive probe"; }
                if (readSurfaceValue<uint32_t>(origins, kTileCount + ordinal) / kSlotsPerTile != tile) {
                    return "Adaptive probe is listed under the wrong tile";
                }
                listed[ordinal] = true;
                ++listedCount;
            }
        }
        if (listedCount != admitted || (empty ? active != 0u : active == 0u)) {
            return "Packed probe count does not match the active tile lists";
        }

        std::vector<bool> alternateTaps;
        if (comparesSharedInterpolation()) {
            if (auto log = compareInterpolation(baseline.back(), candidate.back(), validOrigin, alternateTaps, coverage); !log.empty()) {
                return log;
            }
        }
        for (size_t i = 2; i < kTextures.size(); ++i) {
            const auto& [field, layout] = kTextures[i];
            const auto& oldData = baseline[kBufferedResources.size() + i].bytes;
            const auto& newData = candidate[kBufferedResources.size() + i].bytes;
            const size_t bytes = size_t(layout[0]) * layout[1] * layout[2];
            if (oldData.size() != bytes || newData.size() != bytes) { return "Invalid integration capture size"; }
            if (saturatedAtlas() && !empty) { continue; }
            if (alternateTaps.empty() || empty) {
                if (oldData != newData) { return "Packed integration differs in " + std::string(field); }
                continue;
            }
            for (size_t pixel = 0; pixel < alternateTaps.size(); ++pixel) {
                const auto* first = oldData.data() + pixel * layout[2];
                const auto* second = newData.data() + pixel * layout[2];
                const size_t alphaBytes = layout[2] / 4u;
                if ((!alternateTaps[pixel] && std::memcmp(first, second, layout[2]) != 0) ||
                    std::memcmp(first + 3u * alphaBytes, second + 3u * alphaBytes, alphaBytes) != 0) {
                    const auto describe = [&](const auto& data, const auto& readback) {
                        std::array<float, 4> rgba;
                        if (alphaBytes == sizeof(float)) {
                            rgba = readSurfaceValue<std::array<float, 4>>(data, pixel);
                        } else {
                            const auto packed = readSurfaceValue<std::array<uint16_t, 4>>(data, pixel);
                            std::ranges::transform(packed, rgba.begin(), halfToFloat);
                        }
                        const auto taps = readSurfaceValue<Interpolation>(readback.bytes, pixel);
                        return std::format("RGBA [{:.9g}, {:.9g}, {:.9g}, {:.9g}] taps [{}, {}, {}, {}] weights [{:.9g}, {:.9g}, {:.9g}, {:.9g}]",
                            rgba[0], rgba[1], rgba[2], rgba[3], taps.indices[0], taps.indices[1], taps.indices[2], taps.indices[3],
                            taps.weights[0], taps.weights[1], taps.weights[2], taps.weights[3]);
                    };
                    return std::format("Packed integration differs in {} at pixel {} with alternate taps {}. Dense {}. Compact {}.",
                        field, pixel, bool(alternateTaps[pixel]), describe(oldData, baseline.back()), describe(newData, candidate.back()));
                }
            }
        }

        using Record = render::LumenProbeTraceReadbackRecord;
        constexpr Record invalidRecord{-1.0f, -1.0f, -1.0f, -1.0f, {}, -1.0f, invalid, invalid, 0u, 0u};
        const auto& beforeTrace = baseline[kTraceRecords].bytes;
        const auto& afterTrace = candidate[kTraceRecords].bytes;
        if (beforeTrace.size() != readbackRayCount() * sizeof(Record) || beforeTrace.size() != afterTrace.size()) {
            return "Unexpected packed trace ROI size";
        }
        for (uint32_t y = 0; y < kReadbackROI[3]; ++y) {
            for (uint32_t x = 0; x < kReadbackROI[2]; ++x) {
                const uint32_t originTile = (kReadbackROI[1] + y) * kTileAxis + kReadbackROI[0] + x;
                const uint32_t localTile = y * kReadbackROI[2] + x;
                for (uint32_t slot = 0; slot < kSlotsPerTile; ++slot) {
                    const bool valid = validOrigin[originTile * kSlotsPerTile + slot];
                    for (uint32_t lane = 0; lane < kRaysPerSlot; ++lane) {
                        const size_t address = (localTile * kSlotsPerTile + slot) * kRaysPerSlot + lane;
                        const auto actual = readSurfaceValue<Record>(afterTrace, address);
                        const auto expected = valid ? readSurfaceValue<Record>(beforeTrace, address) : invalidRecord;
                        if (!compareLighting && valid) {
                            constexpr uint32_t kReceiver = 1u << 0u, kScreen = 1u << 2u, kGDF = 1u << 5u;
                            constexpr uint32_t kCovered = 1u << 6u, kSampled = 1u << 7u;
                            if ((actual.flags & kReceiver) == 0u || ((actual.flags & kScreen) == 0u) == ((actual.flags & kGDF) == 0u) ||
                                ((actual.flags & kSampled) != 0u && (actual.flags & (kCovered | kGDF)) != (kCovered | kGDF))) {
                                return "Invalid packed cache trace handoff";
                            }
                        } else if (std::memcmp(&actual, &expected, sizeof(Record)) != 0) {
                            return "Packed trace ROI differs at logical ray " + std::to_string(address);
                        }
                    }
                }
            }
        }

        using ProbeCounts = std::array<uint32_t, 2>;
        const ProbeCounts workCount{active, 0u};
        const auto& state = candidate[kWorkState].bytes;
        const auto& args = candidate[kWorkState + 1].bytes;
        if (state.size() != sizeof(ProbeCounts) || readSurfaceValue<ProbeCounts>(state, 0) != workCount ||
            args.size() != 2u * sizeof(DispatchArgs) ||
            readSurfaceValue<DispatchArgs>(args, 0) != DispatchArgs{active, active != 0u ? 1u : 0u, 1u} ||
            readSurfaceValue<DispatchArgs>(args, 1) != DispatchArgs{0u, 0u, 1u}) {
            return "Packed probe dispatch did not match its active count and empty inactive list";
        }
        if (!historyExpected && screenHits != 0u) { return "Packed screen rays hit without valid history"; }
        const uint32_t validRays = active * kRaysPerSlot;
        const uint32_t gdfRays = validRays - screenHits;
        coverage.mixedScreenAndGDF |= historyExpected && screenHits != 0u && gdfRays != 0u;
        return compareRayDispatch(candidate[kWorkState + 2].bytes, candidate[kWorkState + 3].bytes, validRays, gdfRays, 0u);
    }

    static std::string compareInterpolation(const debug::DebugArtifact& dense, const debug::DebugArtifact& compact,
        const std::array<bool, kProbeSlots>& valid, std::vector<bool>& alternateTaps, RayHandoffCoverage& coverage)
    {
        constexpr size_t pixelCount = size_t(kSingleBounceEdge) * kSingleBounceEdge;
        if (dense.bytes.size() != pixelCount * sizeof(Interpolation) || compact.bytes.size() != dense.bytes.size()) {
            return "Invalid probe interpolation capture size";
        }
        alternateTaps.resize(pixelCount);
        for (size_t pixel = 0; pixel < pixelCount; ++pixel) {
            const auto first = readSurfaceValue<Interpolation>(dense.bytes, pixel);
            const auto second = readSurfaceValue<Interpolation>(compact.bytes, pixel);
            if (first.weights != second.weights) {
                return "Probe interpolation weights differ at pixel " + std::to_string(pixel);
            }
            for (size_t tap = 0; tap < first.indices.size(); ++tap) {
                const uint32_t a = first.indices[tap];
                const uint32_t b = second.indices[tap];
                const float weight = first.weights[tap];
                coverage.signedEdgeTap |= weight < 0.0f;
                if (weight == 0.0f || a == b) { continue; }
                // Both layouts keep the first maximum score. The atomic tile list can
                // choose a different admitted adaptive probe when two scores tie.
                if (weight < 0.0f || a >= kProbeSlots || b >= kProbeSlots || !valid[a] || !valid[b] ||
                    a / kSlotsPerTile != b / kSlotsPerTile || a % kSlotsPerTile == 0u || b % kSlotsPerTile == 0u) {
                    return "Equal-weight interpolation selected unrelated probes at pixel " + std::to_string(pixel);
                }
                alternateTaps[pixel] = true;
                ++coverage.alternateAdaptiveTaps;
            }
        }
        return {};
    }

    std::string compareFrame(CardUpdateInspector& inspector, bool empty, bool historyExpected, bool scroll,
        RayHandoffCoverage& coverage) const
    {
        inspector.runtime.poll();
        if (inspector.response.value("status", "") != "ok") { return inspector.response.dump(); }
        auto& core = inspector.runtime.core();
        const auto& jobs = inspector.response.at("result").at("jobs");
        std::array<CardCapture, 4> captures;
        const size_t captureCount = sharedPlacement() ? 4u : 2u;
        for (size_t index = 0; index < captureCount; ++index) {
            const auto id = jobs.at(index).at("job").get<std::string>();
            captures[index] = core.completedCapture(id);
            if (!captures[index]) { return core.dispatch({{"method", "jobs.get"}, {"params", {{"job", id}}}}).dump(); }
            const size_t modeResources = index == 1 ? (compactAtlas() ? 8 : splitTracing() ? 4 : 2) : 0;
            if (index < 2 && captures[index]->artifacts.size() != kTraceRecords + 1u + modeResources + comparesSharedInterpolation()) {
                return "Unexpected Gather artifact count";
            }
        }
        const auto& baseline = captures[0]->artifacts;
        const auto& candidate = captures[1]->artifacts;
        if (compactAtlas()) {
            auto log = compareCompactAtlas(*captures[0], *captures[1], empty, historyExpected, coverage);
            if (!log.empty() || !sharedPlacement()) { return log; }
            const uint32_t admitted = std::min(readSurfaceValue<uint32_t>(candidate[kAdaptiveCount].bytes, 0), adaptiveBudget());
            return compareCacheCells(*captures[2], *captures[3], candidate[0].bytes, admitted, scroll, coverage);
        }
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

class LumenCompactProbeAtlasTest final : public LumenProbeDispatchComparison {
public:
    LumenCompactProbeAtlasTest() : LumenProbeDispatchComparison("lumen_compact_probe_atlas", Variant::CompactAtlas) {}
};

class LumenCompactProbeAtlasSaturationTest final : public LumenProbeDispatchComparison {
public:
    LumenCompactProbeAtlasSaturationTest()
        : LumenProbeDispatchComparison("lumen_compact_probe_atlas_saturation", Variant::CompactAtlasSaturated) {}
};

class LumenSharedCachePlacementTest final : public LumenProbeDispatchComparison {
public:
    LumenSharedCachePlacementTest()
        : LumenProbeDispatchComparison("lumen_shared_cache_placement", Variant::SharedCacheAtlas) {}
};

class LumenSharedCachePlacementSaturationTest final : public LumenProbeDispatchComparison {
public:
    LumenSharedCachePlacementSaturationTest()
        : LumenProbeDispatchComparison("lumen_shared_cache_placement_saturation", Variant::SharedCacheAtlasSaturated) {}
};

METALLIC_REGISTER_RHI_TEST(LumenActiveProbeDispatchTest);
METALLIC_REGISTER_RHI_TEST(LumenSplitProbeTracingTest);
METALLIC_REGISTER_RHI_TEST(LumenSplitDetailTracingTest);
METALLIC_REGISTER_RHI_TEST(LumenCompactProbeAtlasTest);
METALLIC_REGISTER_RHI_TEST(LumenCompactProbeAtlasSaturationTest);
METALLIC_REGISTER_RHI_TEST(LumenSharedCachePlacementTest);
METALLIC_REGISTER_RHI_TEST(LumenSharedCachePlacementSaturationTest);

class LumenWRCSunRefreshTest : public RHITest {
public:
    LumenWRCSunRefreshTest() : LumenWRCSunRefreshTest("lumen_wrc_sun_refresh", false) {}

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::SceneDocument document;
        RenderGraph graph;
        std::string log;
        if (!loadSingleBounceFixture(context, document, graph, log)) { return RHITestResult::fail(log); }
        if (!prepareGraph(graph, progressive_)) { return RHITestResult::fail("Could not wire the compact WRC refresh fixture"); }

        RenderView view;
        ViewCamera camera;
        camera.eye = {1.1f, 0.72f, 3.0f};
        camera.center = {0.1f, 0.66f, 0.48f};
        camera.fovDegrees = 58.0f;
        camera.farPlane = 8.0f;
        if (!view.setCamera(camera) || !view.setRenderResolution(kSingleBounceEdge, kSingleBounceEdge)) {
            return RHITestResult::fail("Invalid WRC refresh camera");
        }

        bench::ValidationRecorder validation;
        CardUpdateInspector inspector;
        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        if (!preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink())) {
            return RHITestResult::fail("WRC refresh preview setup: " + preview.lastLog());
        }
        preview.setEnvironment({.enabled = false, .visible = false});
        preview.setDebugObserver(&inspector);
        preview.setRecordingWorkerLimit(1);
        auto cacheResources = debug::DebugValue::array({
            cardBufferCapture("RadianceCache.updateState", 32),
            cardBufferCapture("RadianceCache.cacheIndirection", kGridCells)});
        if (progressive_) {
            cacheResources.push_back(cardBufferCapture("RadianceCache.savedIndirection", kGridCells));
            cacheResources.push_back(cardBufferCapture("RadianceCache.probeFrames", 2u * kAtlasSlots));
            cacheResources.push_back(cardBufferCapture("RadianceCache.directionHistory", kAtlasSlots * kDirectionsPerProbe));
        }
        inspector.request = {{"method", "capture.batch"}, {"params", {{"batches", debug::DebugValue::array({
            {{"pass", "RadianceCache"}, {"resources", std::move(cacheResources)}},
            {{"pass", "GlobalDistanceFieldPages"}, {"resources", debug::DebugValue::array({
                cardBufferCapture("GlobalDistanceFieldPages.updateStatistics", 4)})}},
        })}}}};

        environment::WorldEnvironment world;
        world.source = environment::EnvironmentSource::HDRI;
        world.astronomy.mode = environment::AstronomyMode::Manual;
        world.moon.enabled = false;
        world.sun.enabled = true;
        world.sun.illuminance = 4.0f;
        world.sun.direction = {0.0f, -0.8f, -0.6f};
        if (!preview.setLighting(document.lighting()) || !preview.setWorldEnvironment(world)) {
            return RHITestResult::fail("Could not initialize WRC fixture lighting");
        }

        std::vector<uint8_t> previousIndirection;
        uint32_t firstFrameOfLighting = 0;
        bool accumulated = false;
        bool pruned = false;
        const auto check = [&](std::string_view stage, bool retain, bool globalChange, bool first = false,
                               bool lightEdited = false) -> std::string {
            if (!preview.render(graph, kSingleBounceEdge, kSingleBounceEdge, "RadianceCache.cacheRadiance", false)) {
                return "WRC refresh rendering failed: " + preview.lastLog();
            }
            inspector.runtime.poll();
            const auto& response = inspector.response;
            if (response.value("status", "") != "ok") { return response.dump(); }
            auto& core = inspector.runtime.core();
            const auto& jobs = response.at("result").at("jobs");
            const auto cache = core.completedCapture(jobs.at(0).at("job").get<std::string>());
            const auto gdf = core.completedCapture(jobs.at(1).at("job").get<std::string>());
            if (!cache || !gdf || cache->artifacts.size() != (progressive_ ? 5u : 2u) || gdf->artifacts.size() != 1u ||
                    cache->snapshot.evidence.execution != gdf->snapshot.evidence.execution ||
                    cache->snapshot.evidence.generation != gdf->snapshot.evidence.generation) {
                return std::string(stage) + ": Missing same-frame cache or GDF evidence";
            }
            using CacheStatistics = std::array<uint32_t, 32>;
            using GdfStatistics = std::array<uint32_t, 4>;
            const auto& counters = cache->artifacts[0].bytes;
            const auto& publishedIndirection = cache->artifacts[1].bytes;
            const auto& indirection = progressive_ ? cache->artifacts[2].bytes : publishedIndirection;
            if (counters.size() != sizeof(CacheStatistics) || indirection.size() != kGridCells * sizeof(uint32_t) ||
                    publishedIndirection.size() != indirection.size() ||
                    gdf->artifacts[0].bytes.size() != sizeof(GdfStatistics)) {
                return std::string(stage) + ": Unexpected cache or GDF readback size";
            }
            const auto stats = readSurfaceValue<CacheStatistics>(counters, 0);
            const auto gdfStats = readSurfaceValue<GdfStatistics>(gdf->artifacts[0].bytes, 0);
            constexpr uint32_t kPagesPerAxis = 84 / 7;
            constexpr uint32_t kPages = kPagesPerAxis * kPagesPerAxis * kPagesPerAxis;
            const GdfStatistics expectedGdf = first ? GdfStatistics{1u, kPages, 0u, 1u} : GdfStatistics{0u, 0u, kPages, 0u};
            if (gdfStats != expectedGdf) {
                return std::string(stage) + ": Light edit changed geometry page updates " + nlohmann::json(gdfStats).dump();
            }
            const uint32_t marked = stats[24], reused = stats[25], allocated = stats[26];
            if (marked == 0u || marked != reused + allocated || stats[28] != 0u || stats[29] == 0u ||
                    retain != (reused != 0u) || stats[30] != kTraceBudget * (globalChange ? 4u : 1u) ||
                    stats[31] != uint32_t(globalChange)) {
                return std::string(stage) + ": Unexpected WRC scheduling or residency " +
                    nlohmann::json(stats).dump();
            }
            if (progressive_) {
                const auto& frames = cache->artifacts[3].bytes;
                const auto& directions = cache->artifacts[4].bytes;
                if (frames.size() != 2u * kAtlasSlots * sizeof(uint32_t) ||
                    directions.size() != kAtlasSlots * kDirectionsPerProbe * sizeof(uint32_t)) {
                    return std::string(stage) + ": Missing per-probe and directional history";
                }
                // At this fixture's 8x8 resolution each probe uses exactly one tile,
                // including new probes whose requested half resolution cannot form a tile.
                if (stats[29] != stats[2] || (retain && allocated == 0u && stats[29] > kTraceBudget)) {
                    return std::string(stage) + ": Progressive resolve changed the trace cost";
                }
                uint32_t traceFrame = 0;
                for (uint32_t cell = 0; cell < kGridCells; ++cell) {
                    const uint32_t slot = readSurfaceValue<uint32_t>(indirection, cell);
                    if (slot < kAtlasSlots) {
                        traceFrame = std::max(traceFrame, readSurfaceValue<uint32_t>(frames, slot * 2u));
                    }
                }
                const bool reset = !retain || lightEdited;
                if (reset) { firstFrameOfLighting = traceFrame; }

                for (uint32_t cell = 0; cell < kGridCells; ++cell) {
                    const uint32_t slot = readSurfaceValue<uint32_t>(indirection, cell);
                    if (slot >= kAtlasSlots) { continue; }
                    const uint32_t lastTraced = readSurfaceValue<uint32_t>(frames, slot * 2u);
                    const bool fresh = lastTraced >= firstFrameOfLighting;
                    const uint32_t published = readSurfaceValue<uint32_t>(publishedIndirection, cell);
                    if (published != (fresh ? slot : std::numeric_limits<uint32_t>::max())) {
                        return std::string(stage) + ": Published indirection disagrees with slot lighting freshness";
                    }
                    if (!fresh) {
                        pruned |= retain && lightEdited;
                        continue;
                    }
                    for (uint32_t direction = 0; direction < kDirectionsPerProbe; ++direction) {
                        const uint32_t count = readSurfaceValue<uint32_t>(directions,
                            slot * kDirectionsPerProbe + direction) & 0xffu;
                        if (count == 0u || count > 8u || (reset && count != 1u)) {
                            return std::string(stage) + ": Invalid directional sample count " + std::to_string(count);
                        }
                        accumulated |= count > 1u;
                    }
                }
            }
            if (retain) {
                uint32_t overlap = 0;
                for (size_t cell = 0; cell < kGridCells; ++cell) {
                    const uint32_t before = readSurfaceValue<uint32_t>(previousIndirection, cell);
                    const uint32_t after = readSurfaceValue<uint32_t>(indirection, cell);
                    if (before < kAtlasSlots && after < kAtlasSlots) {
                        if (before != after) { return std::string(stage) + ": Resident world cell changed physical slots"; }
                        ++overlap;
                    }
                }
                if (overlap < reused) { return std::string(stage) + ": WRC reuse counters exceed retained world cells"; }
            }
            previousIndirection = indirection;
            return {};
        };
        if (auto message = check("Cold start with active Sun", false, true, true); !message.empty()) {
            return RHITestResult::fail(message);
        }
        if (progressive_) {
            for (uint32_t frame = 0; frame <= kAtlasSlots / kTraceBudget && !accumulated; ++frame) {
                if (auto message = check("Stable direction refresh", true, false); !message.empty()) {
                    return RHITestResult::fail(message);
                }
            }
            if (!accumulated) {
                return RHITestResult::fail("No resident direction accumulated a second sample under stable lighting");
            }
            world.sun.direction = {0.0f, -1.6f, -1.2f};
            if (!preview.setWorldEnvironment(world)) {
                return RHITestResult::fail("Could not edit the authored Sun direction magnitude");
            }
            accumulated = false;
            if (auto message = check("Equivalent Sun direction", true, false); !message.empty()) {
                return RHITestResult::fail(message);
            }
            if (!accumulated) {
                return RHITestResult::fail("An equivalent Sun direction discarded directional history");
            }
        }
        struct SunEdit {
            const char* label;
            float3 direction;
            float illuminance;
            bool globalChange;
        };
        const SunEdit edits[]{
            {"Drag Sun first direction", {-0.4f, -0.6f, -0.6f}, 4.0f, false},
            {"Drag Sun second direction", {0.4f, -0.5f, -0.6f}, 4.0f, false},
            {"Increase Sun intensity at the 4x boundary", {0.4f, -0.5f, -0.6f}, 16.0f, false},
            {"Jump Sun intensity beyond the 4x boundary", {0.4f, -0.5f, -0.6f}, 96.0f, true},
        };
        for (const auto& edit : edits) {
            world.sun.direction = edit.direction;
            world.sun.illuminance = edit.illuminance;
            if (!preview.setWorldEnvironment(world)) {
                return RHITestResult::fail(std::string("Could not edit the WRC fixture Sun: ") + edit.label);
            }
            if (auto message = check(edit.label, !edit.globalChange, edit.globalChange, false, true); !message.empty()) {
                return RHITestResult::fail(message);
            }
        }
        if (auto message = check("Frame after global change", true, false); !message.empty()) {
            return RHITestResult::fail(message);
        }
        auto lighting = document.lighting();
        scene::PunctualLight point;
        point.properties.intensityUnit = scene::LightUnit::Candela;
        point.properties.intensity = 4.0f;
        point.properties.range = 4.0f;
        point.position = {0.0f, 1.2f, 0.95f};
        lighting.lights.push_back(std::move(point));
        if (!preview.setLighting(std::move(lighting))) { return RHITestResult::fail("Could not add the WRC fixture point light"); }
        if (auto message = check("Point-light edit", false, false); !message.empty()) { return RHITestResult::fail(message); }
        if (validation.failed()) { return RHITestResult::fail("WRC refresh Vulkan validation: " + validation.snapshot().dump()); }
        if (progressive_) {
            if (!pruned) { return RHITestResult::fail("A budgeted light edit did not exercise stale published probes"); }
            return RHITestResult::pass("Verified directional accumulation, lighting invalidation, and budgeted probe publication");
        }
        return RHITestResult::pass("Verified Sun-only reuse, the 4x threshold and boosted budget, normal recovery, and point-light reset");
    }

protected:
    LumenWRCSunRefreshTest(const char* testName, bool progressive) : progressive_(progressive)
    {
        name = testName;
        type = RHITestType::Rendering;
    }

private:
    static constexpr uint32_t kGridResolution = 16;
    static constexpr uint32_t kGridCells = kGridResolution * kGridResolution * kGridResolution;
    static constexpr uint32_t kAtlasProbeAxis = 16;
    static constexpr uint32_t kAtlasSlots = kAtlasProbeAxis * kAtlasProbeAxis;
    static constexpr uint32_t kTraceBudget = 16;
    static constexpr uint32_t kProbeResolution = 8;
    static constexpr uint32_t kDirectionsPerProbe = kProbeResolution * kProbeResolution;
    bool progressive_;

    static bool prepareGraph(render::RenderGraph& graph, bool progressive)
    {
        using render::makeRenderGraphFieldName;
        const auto* fields = graph.findNode("GlobalDistanceFieldPages");
        auto fieldProperties = fields->properties;
        fieldProperties["persistentPages"] = true;
        if (!graph.setNodeProperties(fields->id, std::move(fieldProperties)) ||
            !graph.addNode("LumenScreenProbePlacementPass", "Placement", {}) ||
            !graph.addNode("LumenRadianceCachePass", "RadianceCache", {{"gridResolution", kGridResolution},
                {"clipmapCount", 1u}, {"probeResolution", kProbeResolution}, {"atlasProbesPerAxis", kAtlasProbeAxis},
                {"traceBudget", kTraceBudget}, {"worldExtent", 4.0f}, {"maxTraceDistance", 5.0f},
                {"compactProbeAtlas", true}, {"progressiveAccumulation", progressive}}) ||
            !graph.addEdge("Deferred.lumenPosition", "Placement.position") ||
            !graph.addEdge("Deferred.lumenNormal", "Placement.normal") ||
            !graph.addEdge("Placement.probePositions", "RadianceCache.probePositions") ||
            !graph.addEdge("Placement.probeAdaptiveCount", "RadianceCache.probeAdaptiveCount")) {
            return false;
        }
        constexpr std::array<std::string_view, 11> inputs{"position", "pageTable", "pageAtlas", "coarseMip", "domain",
            "objectGrid", "objectGridScene", "cards", "cardScene", "cardDepth", "cardLighting"};
        const auto edges = graph.edges();
        for (const auto& edge : edges) {
            if (edge.dstPass == "Gather" && std::ranges::find(inputs, edge.dstField) != inputs.end() &&
                !graph.addEdge(makeRenderGraphFieldName(edge.srcPass, edge.srcField), makeRenderGraphFieldName("RadianceCache", edge.dstField))) {
                return false;
            }
        }
        return graph.unmarkOutput("FinalBlit.color") && graph.unmarkOutput("Gather.indirectDiffuse") &&
            graph.markOutput("RadianceCache.cacheRadiance");
    }
};

METALLIC_REGISTER_RHI_TEST(LumenWRCSunRefreshTest);

class LumenWRCProgressiveRefreshTest final : public LumenWRCSunRefreshTest {
public:
    LumenWRCProgressiveRefreshTest() : LumenWRCSunRefreshTest("lumen_wrc_progressive_refresh", true) {}
};

METALLIC_REGISTER_RHI_TEST(LumenWRCProgressiveRefreshTest);

debug::DebugValue surfaceUpdateCaptures(std::string_view pass, bool captureStochasticHistory = false)
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
        if (captureStochasticHistory) {
            captures.push_back(cardAtlasCapture(makeRenderGraphFieldName(pass, "stochasticDirectHistory")));
        }
    }
    return captures;
}

std::string compareSurfaceUpdate(const CardUpdateFrame& frame, const std::array<uint32_t, 3>& expectedStats,
    uint32_t expectedResident, bool compareLighting = true, bool captureStochasticHistory = false)
{
    const auto& surface = frame.surface->artifacts;
    const auto& full = frame.fullSurface->artifacts;
    const auto& radiosity = frame.radiosity->artifacts;
    const size_t surfaceResourceCount = captureStochasticHistory ? kSurfaceStochasticHistory + 1 : kSurfaceUpdateFlags + 1;
    if (surface.size() != surfaceResourceCount || full.size() != kCardScene + 1 ||
        radiosity.size() != kRadiosityCardState + 1 || frame.denseRadiosity->artifacts.size() != 2 ||
        surface[kSurfaceUpdateStats].bytes.size() != sizeof(expectedStats) ||
        surface[kSurfaceUpdateFlags].bytes.size() != kPersistentCardCapacity * sizeof(uint32_t) ||
        surface[kCardRecords].bytes.size() != kPersistentCardCapacity * sizeof(CapturedCard) ||
        surface[kCardScene].bytes.size() != 4 * sizeof(uint32_t) ||
        radiosity[0].bytes.size() != kAtlasPixels * 8 || radiosity[1].bytes.size() != kAtlasPixels * 8 ||
        frame.denseRadiosity->artifacts[0].bytes.size() != kAtlasPixels * 8 ||
        frame.denseRadiosity->artifacts[1].bytes.size() != kAtlasPixels * 8 ||
        radiosity[kRadiosityUpdateStats].bytes.size() != 4 * sizeof(uint32_t) ||
        radiosity[kRadiosityCardState].bytes.size() != kPersistentCardCapacity * 2 * sizeof(uint32_t) ||
        (captureStochasticHistory &&
            surface[kSurfaceStochasticHistory].bytes.size() != kAtlasPixels * sizeof(std::array<float, 4>))) {
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
        const bool lighting = atlas == kSurfaceLighting || atlas == kSurfaceDirectIrradiance;
        if (surface[atlas].bytes.size() != kAtlasPixels * stride || full[atlas].bytes.size() != kAtlasPixels * stride ||
            ((compareLighting || !lighting) && surface[atlas].bytes != full[atlas].bytes)) {
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

class LumenPersistentCardUpdateTest : public RHITest {
public:
    enum class Mode { CardUpdates, RectDiffuse, RectReGIR };
    static constexpr uint32_t kRectReGIRTestSamples = 2u;
    static constexpr uint32_t kRectReGIRTestRefreshCards = 3u;

    explicit LumenPersistentCardUpdateTest(Mode mode = Mode::CardUpdates)
        : mode_(mode)
    {
        name = mode_ == Mode::RectReGIR ? "lumen_surface_cache_rect_regir" :
            mode_ == Mode::RectDiffuse ? "lumen_surface_cache_rect_diffuse" : "lumen_persistent_card_updates";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const bool testRectLighting = mode_ != Mode::CardUpdates;
        const bool testReGIR = mode_ == Mode::RectReGIR;
        const std::string fixtureName = testReGIR ? "LumenRectReGIR" :
            testRectLighting ? "LumenRectDiffuse" : "LumenCardUpdate";
        const auto fixture = context.outputDirectory / (fixtureName + "Boxes.gltf");
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
        if (testRectLighting) {
            auto material = document.materials()[1];
            material.emissiveFactor = float3(0.0f);
            if (!document.setMaterialProperties(1, material)) {
                return RHITestResult::fail("Could not isolate the analytical rectangle from emissive surfaces");
            }
        }

        std::string log;
        std::vector<std::string> fieldPaths, cardPaths;
        for (int32_t mesh = 0; mesh < 3; ++mesh) {
            const auto base = fixtureName + std::to_string(mesh);
            const auto field = context.outputDirectory / (base + ".meshdf");
            const auto cards = context.outputDirectory / (base + ".meshcards");
            if (!writeAnalyticBoxField(field, document, log, mesh) || !writeAnalyticBoxCards(cards, document, mesh, log)) {
                return RHITestResult::fail(log);
            }
            fieldPaths.push_back(field.generic_string());
            cardPaths.push_back(cards.generic_string());
        }

        RenderGraph graph;
        if (!makeGraph(graph, fieldPaths, cardPaths, testReGIR)) {
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
        if (testRectLighting) { camera.center = {0.0f, 0.0f, 4.0f}; }
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
            uint32_t resident = kPersistentResidentCards, std::optional<bool> rebuildGdf = false,
            bool compareLighting = true) -> std::optional<CardUpdateFrame> {
            auto frame = captureFrame(inspector, executor, submission, log, testReGIR);
            if (frame) { log = compareSurfaceUpdate(*frame, stats, resident, compareLighting, testReGIR); }
            if (frame && log.empty() && rebuildGdf.has_value()) {
                constexpr uint32_t pageEdge = kGdfResolution / 7;
                constexpr uint32_t pageCount = pageEdge * pageEdge * pageEdge;
                constexpr std::array rebuilt{1u, pageCount, 0u, 1u};
                constexpr std::array reused{0u, 0u, pageCount, 0u};
                const auto counters = readSurfaceValue<std::array<uint32_t, 4>>(frame->gdf->artifacts[0].bytes, 0);
                if (counters != (*rebuildGdf ? rebuilt : reused)) {
                    log = "Unexpected persistent GDF update statistics " + nlohmann::json(counters).dump();
                }
            }
            if (!frame || !log.empty()) {
                log = std::string(stage) + ": " + log;
                return std::nullopt;
            }
            return frame;
        };

        auto previous = runFrame("Initial frame", {1u, kPersistentResidentCards, kPersistentResidentCards},
            kPersistentResidentCards, true);
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

        if (testRectLighting) {
            const auto result = testReGIR ?
                checkRectReGIR(renderWorld, receiver, *previous, runFrame,
                    context.outputDirectory / "LumenRectReGIRAtlas.png", log) :
                checkRectDiffuse(renderWorld, receiver, *previous, runFrame,
                    context.outputDirectory / "LumenRectDiffuseAtlas.png", log);
            if (validation.failed()) {
                return RHITestResult::fail("Rect lighting Vulkan validation " + validation.snapshot().dump());
            }
            return result;
        }

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
            if (current->gdf->artifacts[1].bytes != previous->gdf->artifacts[1].bytes) {
                return RHITestResult::fail(std::string(edit.name) + ": Sun edit changed the coarse distance field");
            }
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
        const auto material = runFrame("Emitter material edit", {0u, kCardsPerBox, kCardsPerBox},
            kPersistentResidentCards, true);
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
        const auto geometry = runFrame("Blocker geometry change", {0u, kCardsPerBox, kPersistentResidentCards},
            kPersistentResidentCards, true);
        if (!geometry) { return RHITestResult::fail(log); }
        if (geometry->gdf->artifacts[1].bytes == material->gdf->artifacts[1].bytes) {
            return RHITestResult::fail("Moving the blocker did not change the coarse distance field");
        }
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
        const auto hidden = runFrame("Blocker visibility change", {1u, remaining, remaining}, remaining, true);
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
        const auto resized = runFrame("Graph resize", {1u, remaining, remaining}, remaining, std::nullopt);
        if (!resized) { return RHITestResult::fail(log); }
        const auto resizedStats = readSurfaceValue<std::array<uint32_t, 4>>(
            resized->radiosity->artifacts[kRadiosityUpdateStats].bytes, 0);
        if (resizedStats != std::array{remaining, remaining, remaining, kRadiosityCardBudget}) {
            return RHITestResult::fail("Graph resize did not reinitialize the Radiosity atlas");
        }

        constexpr uint32_t directBudget = 3;
        const auto* surfaceNode = graph.findNode(kPersistentSurface);
        auto surfaceProperties = surfaceNode->properties;
        surfaceProperties["maxDirectLightingCardsPerFrame"] = directBudget;
        surfaceProperties["objectCardRanges"] = true;
        if (!graph.setNodeProperties(surfaceNode->id, std::move(surfaceProperties)) ||
            !executor.compile(gpu, graph, kAtlasEdge, kAtlasEdge, RenderGraphCompileOptions{.enableDebugReadback = true}, log)) {
            return RHITestResult::fail("Could not compile the budgeted Surface Cache: " + log);
        }
        if (!runFrame("Initialize budgeted Surface Cache", {1u, remaining, remaining}, remaining, true)) {
            return RHITestResult::fail(log);
        }
        auto editedLighting = renderWorld.lighting();
        editedLighting.lights.back().properties.intensity *= 2.0f;
        if (!renderWorld.setLighting(std::move(editedLighting))) {
            return RHITestResult::fail("Could not edit the point light for budgeted card updates");
        }
        for (uint32_t first = 0; first < remaining; first += directBudget) {
            const uint32_t end = std::min(first + directBudget, remaining);
            auto frame = captureFrame(inspector, executor, submission, log);
            if (!frame) { return RHITestResult::fail(log); }
            const auto& surface = frame->surface->artifacts;
            const auto stats = readSurfaceValue<std::array<uint32_t, 3>>(surface[kSurfaceUpdateStats].bytes, 0);
            if (stats != std::array{0u, 0u, end - first}) {
                return RHITestResult::fail("The direct lighting pass exceeded its budget or abandoned pending cards");
            }
            for (uint32_t slot = 0; slot < kPersistentCardCapacity; ++slot) {
                const auto flags = readSurfaceValue<uint32_t>(surface[kSurfaceUpdateFlags].bytes, slot);
                const uint32_t expected = slot >= first && slot < end ? kCardRelit | kCardLightingChanged : 0u;
                if (flags != expected) { return RHITestResult::fail("Direct relight and indirect reset advanced on different tiles"); }
            }
            if (end == remaining) {
                if (auto message = compareSurfaceUpdate(*frame, stats, remaining); !message.empty()) {
                    return RHITestResult::fail("The budgeted atlas did not converge to a full update: " + message);
                }
            }
        }
        if (!runFrame("Idle after draining the direct relight budget", {0u, 0u, 0u}, remaining)) {
            return RHITestResult::fail(log);
        }
        if (validation.failed()) {
            return RHITestResult::fail("Persistent card Vulkan validation: " + validation.snapshot().dump());
        }
        return RHITestResult::pass("Sun edits reuse GDF pages and relight current cards while indirect history advances within its budget. "
            "Dense Radiosity traces on reset, and retired persistent tiles clear.");
    }

private:
    template<class RunFrame>
    static RHITestResult checkRectDiffuse(render::RenderWorld& world, scene::SceneEntity receiver,
        const CardUpdateFrame& initial, const RunFrame& runFrame, const std::filesystem::path& imagePath,
        std::string& log)
    {
        using FloatPixel = std::array<float, 4>;
        using HalfPixel = std::array<uint16_t, 4>;
        using BytePixel = std::array<uint8_t, 4>;
        constexpr float kPi = 3.14159265358979323846f;
        const auto& atlas = initial.surface->artifacts;
        uint32_t slot = 0;
        CapturedCard card;
        for (; slot < kPersistentResidentCards; ++slot) {
            card = readSurfaceValue<CapturedCard>(atlas[kCardRecords].bytes, slot);
            if (card.metadata[0] == uint32_t(receiver) && card.axisZ[2] > 0.9999f) { break; }
        }
        if (slot == kPersistentResidentCards) { return RHITestResult::fail("Missing receiver card facing +Z"); }
        const size_t pixelIndex = tileCenter(slot);
        const auto encodedDepth = readSurfaceValue<uint16_t>(atlas[3].bytes, pixelIndex);
        if (encodedDepth == UINT16_MAX) { return RHITestResult::fail("Offscreen receiver card center has no coverage"); }
        const auto position = cardCenterWorldPosition(card, encodedDepth);
        const auto normal = readSurfaceValue<BytePixel>(atlas[1].bytes, pixelIndex);
        const float normalX = float(normal[0]) * (2.0f / 255.0f) - 1.0f;
        const float normalY = float(normal[1]) * (2.0f / 255.0f) - 1.0f;
        const float receiverCosine = std::sqrt(1.0f - normalX * normalX - normalY * normalY);
        const auto checkEnergy = [&](const CardUpdateFrame& frame, float expected) -> std::string {
            const auto& surface = frame.surface->artifacts;
            const auto& radiosity = frame.radiosity->artifacts;
            const auto irradiance = readSurfaceValue<FloatPixel>(surface[kSurfaceDirectIrradiance].bytes, pixelIndex);
            const auto albedo = readSurfaceValue<BytePixel>(surface[0].bytes, pixelIndex);
            const auto emission = readSurfaceValue<HalfPixel>(surface[2].bytes, pixelIndex);
            const auto outgoing = readSurfaceValue<HalfPixel>(surface[kSurfaceLighting].bytes, pixelIndex);
            const auto indirect = readSurfaceValue<HalfPixel>(radiosity[0].bytes, pixelIndex);
            const auto combined = readSurfaceValue<HalfPixel>(radiosity[1].bytes, pixelIndex);
            for (size_t channel = 0; channel < 3; ++channel) {
                if (!(std::abs(irradiance[channel] - expected) <= 0.002f * std::max(1.0f, expected))) {
                    return std::format("Receiver irradiance channel {} is {}, expected {}",
                        channel, irradiance[channel], expected);
                }
                const float encoded = float(albedo[channel]) / 255.0f;
                const float response = encoded * encoded / kPi;
                const float emitted = halfToFloat(emission[channel]);
                const float direct = irradiance[channel] * response + emitted;
                const float total = direct + halfToFloat(indirect[channel]) * response;
                if (!(std::abs(halfToFloat(outgoing[channel]) - direct) < 0.002f &&
                    std::abs(halfToFloat(combined[channel]) - total) < 0.002f)) {
                    return "Rect irradiance and the card material response disagree in Surface Cache or Radiosity";
                }
            }
            return {};
        };
        if (const auto message = checkEnergy(initial, 0.0f); !message.empty()) {
            return RHITestResult::fail("Before adding a rectangle. " + message);
        }

        scene::RectLight light;
        light.position = {position[0], position[1], position[2] + 1.0f};
        light.direction = {0.0f, 0.0f, -1.0f};
        light.width = light.height = 2.0f;
        light.luminance = 1.0f;
        scene::LightingSettings lighting = world.lighting();
        struct LightEdit {
            const char* name;
            float width, range, direction, distance;
            uint32_t count;
        };
        const LightEdit edits[]{
            {"Enable Rect", 2.0f, 0.0f, -1.0f, 1.0f, 1u},
            {"Add overlapping Rect", 2.0f, 0.0f, -1.0f, 1.0f, 2u},
            {"Resize and move Rect", 1.0f, 2.0f, -1.0f, 1.5f, 1u},
            {"Exclude receiver by range", 1.0f, 1.0f, -1.0f, 1.5f, 1u},
            {"Rotate Rect away", 1.0f, 2.0f, 1.0f, 1.5f, 1u},
            {"Restore Rect face", 1.0f, 2.0f, -1.0f, 1.5f, 1u},
            {"Disable Rect", 1.0f, 2.0f, -1.0f, 1.5f, 0u},
        };
        for (const auto& edit : edits) {
            light.width = edit.width;
            light.range = edit.range;
            light.direction.z = edit.direction;
            light.position.z = position[2] + edit.distance;
            light.enabled = edit.count != 0;
            lighting.rectLights.assign(std::max(1u, edit.count), light);
            if (!world.setLighting(lighting)) {
                return RHITestResult::fail(std::string("Could not apply Rect light edit ") + edit.name);
            }
            auto frame = runFrame(edit.name, {0u, 0u, kPersistentResidentCards});
            if (!frame) { return RHITestResult::fail(log); }
            if (const auto message = checkCardFlags(*frame, kPersistentResidentCards, receiver,
                    kCardRelit | kCardLightingChanged, kCardRelit | kCardLightingChanged); !message.empty()) {
                return RHITestResult::fail(std::string(edit.name) + ". " + message);
            }
            const auto resets = readSurfaceValue<std::array<uint32_t, 4>>(
                frame->radiosity->artifacts[kRadiosityUpdateStats].bytes, 0)[1];
            if (resets != kPersistentResidentCards) {
                return RHITestResult::fail(std::string(edit.name) + " failed to reset the resident indirect history");
            }
            const float cosineIntegral = centeredRectCosineIntegral(edit.width / 2.0f,
                light.height / 2.0f, edit.distance);
            const float fade = edit.range == 0.0f ? 1.0f :
                std::max(0.0f, 1.0f - std::pow(edit.distance / edit.range, 4.0f));
            const float expected = edit.direction < 0.0f
                ? float(edit.count) * light.luminance * cosineIntegral * receiverCosine * fade : 0.0f;
            if (const auto message = checkEnergy(*frame, expected); !message.empty()) {
                return RHITestResult::fail(std::string(edit.name) + ". " + message);
            }
            if (edit.count == 2) {
                // Display irradiance on the left and outgoing Surface Cache radiance on the right.
                const auto& surface = frame->surface->artifacts;
                std::vector<uint8_t> pixels(kAtlasPixels * 2 * 4, 255);
                const auto display = [](float value) {
                    return static_cast<uint8_t>(std::lround(255.0f * std::sqrt(std::clamp(value, 0.0f, 1.0f))));
                };
                for (size_t index = 0; index < kAtlasPixels; ++index) {
                    const size_t offset = (index + index / kAtlasEdge * kAtlasEdge) * 4;
                    const auto direct = readSurfaceValue<FloatPixel>(surface[kSurfaceDirectIrradiance].bytes, index);
                    const auto outgoing = readSurfaceValue<HalfPixel>(surface[kSurfaceLighting].bytes, index);
                    for (size_t channel = 0; channel < 3; ++channel) {
                        pixels[offset + channel] = display(direct[channel] / edit.count);
                        pixels[offset + kAtlasEdge * 4 + channel] = display(halfToFloat(outgoing[channel]));
                    }
                }
                if (!saveRgba8Png(imagePath, pixels.data(), kAtlasEdge * 2, kAtlasEdge, log)) {
                    return RHITestResult::fail(log);
                }
            }
        }
        const auto settled = runFrame("Idle after disabling Rect", {0u, 0u, 0u});
        if (!settled) { return RHITestResult::fail(log); }
        if (const auto message = checkEnergy(*settled, 0.0f); !message.empty()) {
            return RHITestResult::fail(message);
        }
        return RHITestResult::pass("Offscreen rectangular emitters match analytical irradiance, material response, and persistent history edits.");
    }

    template<class RunFrame>
    static RHITestResult checkRectReGIR(render::RenderWorld& world, scene::SceneEntity receiver,
        const CardUpdateFrame& initial, const RunFrame& runFrame, const std::filesystem::path& imagePath,
        std::string& log)
    {
        using render::color::RGB;
        using Pixel = std::array<float, 4>;
        constexpr uint32_t kLightsPerColor = 48u;
        constexpr uint32_t kRefreshSweeps = 4u;
        static_assert(kPersistentResidentCards % kRectReGIRTestRefreshCards == 0u);
        constexpr uint32_t kRefreshFrames = kRefreshSweeps * kPersistentResidentCards / kRectReGIRTestRefreshCards;
        constexpr uint32_t kRefreshFlags = kCardRelit | render::kLumenCardStochasticRelit;
        constexpr uint32_t kResetFlags = kRefreshFlags | kCardLightingChanged | render::kLumenCardResetStochasticLighting;
        std::array<uint32_t, kPersistentCardCapacity> batchCounts{};
        const auto checkHistory = [&](const CardUpdateFrame& state) -> std::string {
            const auto& resources = state.surface->artifacts;
            const auto& history = resources[kSurfaceStochasticHistory].bytes;
            for (uint32_t tile = 0; tile < kPersistentCardCapacity; ++tile) {
                for (uint32_t y = 0; y < kTileEdge; ++y) {
                    const uint32_t offset = cardPixelOffset(tile, y);
                    for (uint32_t x = 0; x < kTileEdge; ++x) {
                        const auto index = offset + x;
                        const bool covered = readSurfaceValue<uint16_t>(resources[3].bytes, index) != UINT16_MAX;
                        const float expectedCount = covered ? float(batchCounts[tile]) : 0.0f;
                        const auto pixel = readSurfaceValue<Pixel>(history, index);
                        if (pixel[3] != expectedCount || (expectedCount == 0.0f && pixel != Pixel{})) {
                            return std::format("Stochastic Rect history count at tile {} is {}, expected {}",
                                tile, pixel[3], expectedCount);
                        }
                    }
                }
            }
            return {};
        };
        if (const auto message = checkHistory(initial); !message.empty()) { return RHITestResult::fail(message); }
        const auto& atlas = initial.surface->artifacts;
        uint32_t slot = 0u;
        CapturedCard card;
        for (; slot < kPersistentResidentCards; ++slot) {
            card = readSurfaceValue<CapturedCard>(atlas[kCardRecords].bytes, slot);
            if (card.metadata[0] == uint32_t(receiver) && card.axisZ[2] > 0.9999f) { break; }
        }
        if (slot == kPersistentResidentCards) { return RHITestResult::fail("Missing the ReGIR receiver card facing +Z"); }
        const size_t center = tileCenter(slot);
        const auto encodedDepth = readSurfaceValue<uint16_t>(atlas[3].bytes, center);
        if (encodedDepth == UINT16_MAX) { return RHITestResult::fail("The ReGIR receiver center has no coverage"); }
        const auto position = cardCenterWorldPosition(card, encodedDepth);
        auto lighting = world.lighting();
        for (size_t channel = 0; channel < 3; ++channel) {
            RGB primary{};
            primary[channel] = 1.0f;
            const float workingLuminance = render::color::luminance(render::color::fromLinearRec709(primary));
            scene::RectLight light;
            light.position = {position[0], position[1], position[2] + 1.0f};
            light.direction = {0.0f, 0.0f, -1.0f};
            light.width = light.height = 2.0f;
            light.color = {primary[0], primary[1], primary[2]};
            light.luminance = 0.025f / (float(kLightsPerColor) * workingLuminance);
            lighting.rectLights.insert(lighting.rectLights.end(), kLightsPerColor, light);
            // The global proposal sees the same power here, but these lights cannot reach the card.
            light.position.x += 100.0f;
            light.range = 1.0f;
            lighting.rectLights.insert(lighting.rectLights.end(), kLightsPerColor, light);
        }
        if (!world.setLighting(lighting)) { return RHITestResult::fail("Could not apply the many-Rect fixture"); }
        const auto frame = runFrame("Sample many Rects", {0u, 0u, kPersistentResidentCards},
            kPersistentResidentCards, std::nullopt, false);
        if (!frame) { return RHITestResult::fail(log); }
        if (const auto message = checkCardFlags(*frame, kPersistentResidentCards, receiver, kResetFlags, kResetFlags);
            !message.empty()) { return RHITestResult::fail(message); }
        std::fill_n(batchCounts.begin(), kPersistentResidentCards, 1u);
        if (const auto message = checkHistory(*frame); !message.empty()) { return RHITestResult::fail(message); }
        const auto& sampled = frame->surface->artifacts[kSurfaceDirectIrradiance].bytes;
        const auto& exact = frame->fullSurface->artifacts[kSurfaceDirectIrradiance].bytes;
        const auto decode = [](Pixel value) { return render::color::toLinearRec709({value[0], value[1], value[2]}); };
        std::array<double, 3> sampledEnergy{}, exactEnergy{};
        double firstSquaredError = 0.0;
        uint32_t covered = 0u;
        uint32_t lit = 0u;
        constexpr uint32_t kPreviewWidth = 3u * kTileEdge;
        std::vector<uint8_t> pixels(kPreviewWidth * kTileEdge * 4, 255u);
        const auto display = [](float value) {
            return static_cast<uint8_t>(std::lround(255.0f * std::sqrt(std::clamp(value, 0.0f, 1.0f))));
        };
        for (uint32_t y = 0; y < kTileEdge; ++y) {
            for (uint32_t x = 0; x < kTileEdge; ++x) {
                const size_t index = cardPixelOffset(slot, y) + x;
                if (readSurfaceValue<uint16_t>(atlas[3].bytes, index) == UINT16_MAX) { continue; }
                ++covered;
                const auto estimate = decode(readSurfaceValue<Pixel>(sampled, index));
                const auto reference = decode(readSurfaceValue<Pixel>(exact, index));
                const float tolerance = *std::ranges::max_element(estimate) * 0.0001f;
                if (std::ranges::count_if(estimate, [=](float value) { return value > tolerance; }) > kRectReGIRTestSamples) {
                    return RHITestResult::fail("A ReGIR texel accumulated more light colors than its draw budget");
                }
                lit += tolerance > 0.0f ? 1u : 0u;
                const size_t offset = (size_t(y) * kPreviewWidth + x) * 4;
                for (size_t channel = 0; channel < 3; ++channel) {
                    if (!(std::isfinite(estimate[channel]) && reference[channel] > 0.0f)) {
                        return RHITestResult::fail("Invalid ReGIR estimate or unlit exact receiver");
                    }
                    sampledEnergy[channel] += estimate[channel];
                    exactEnergy[channel] += reference[channel];
                    const double relativeError = double(estimate[channel]) / reference[channel] - 1.0;
                    firstSquaredError += relativeError * relativeError;
                    pixels[offset + channel] = display(estimate[channel]);
                    pixels[offset + 2u * kTileEdge * 4 + channel] = display(reference[channel]);
                }
            }
        }
        if (covered < kTileEdge * kTileEdge / 2 || lit * 10u < covered * 9u) {
            return RHITestResult::fail(std::format("ReGIR lit {} of {} covered receiver texels", lit, covered));
        }
        double maxRelativeError = 0.0;
        for (size_t channel = 0; channel < sampledEnergy.size(); ++channel) {
            maxRelativeError = std::max(maxRelativeError, std::abs(sampledEnergy[channel] / exactEnergy[channel] - 1.0));
        }
        if (maxRelativeError > 0.30) {
            return RHITestResult::fail(std::format("Sampled Rect irradiance differs from the exact sum by {:.1f}%",
                maxRelativeError * 100.0));
        }
        const auto sameTile = [](const auto& before, const auto& after, uint32_t tile, size_t stride) {
            for (uint32_t y = 0; y < kTileEdge; ++y) {
                const size_t offset = cardPixelOffset(tile, y) * stride;
                if (std::memcmp(before.bytes.data() + offset, after.bytes.data() + offset, kTileEdge * stride) != 0) {
                    return false;
                }
            }
            return true;
        };
        auto accumulated = frame;
        std::array<bool, kPersistentCardCapacity> indirectRefreshed{};
        uint32_t indirectUnscheduledRelights = 0u;
        for (uint32_t i = 0; i < kRefreshFrames; ++i) {
            auto next = runFrame("Refresh stochastic direct lighting", {0u, 0u, kRectReGIRTestRefreshCards},
                kPersistentResidentCards, std::nullopt, false);
            if (!next) { return RHITestResult::fail(log); }
            const auto& before = accumulated->surface->artifacts;
            const auto& after = next->surface->artifacts;
            uint32_t refreshed = 0u;
            for (uint32_t tile = 0; tile < kPersistentCardCapacity; ++tile) {
                const auto flags = readSurfaceValue<uint32_t>(after[kSurfaceUpdateFlags].bytes, tile);
                if (tile < kPersistentResidentCards && flags == kRefreshFlags) {
                    ++batchCounts[tile];
                    ++refreshed;
                    continue;
                }
                if (flags != 0u) { return RHITestResult::fail("Stochastic refresh requested a capture or history reset"); }
                for (size_t resource : {kSurfaceLighting, kSurfaceDirectIrradiance, kSurfaceStochasticHistory}) {
                    const size_t stride = resource == kSurfaceLighting ? sizeof(std::array<uint16_t, 4>) : sizeof(Pixel);
                    if (!sameTile(before[resource], after[resource], tile, stride)) {
                        return RHITestResult::fail("A tile outside the stochastic refresh budget changed");
                    }
                }
            }
            if (refreshed != kRectReGIRTestRefreshCards) {
                return RHITestResult::fail("Stochastic direct refresh did not respect its own card budget");
            }
            if (const auto message = checkHistory(*next); !message.empty()) { return RHITestResult::fail(message); }
            if (const auto message = checkRadiositySchedule(*accumulated, *next, indirectRefreshed,
                    &indirectUnscheduledRelights); !message.empty()) {
                return RHITestResult::fail(message);
            }
            accumulated = std::move(next);
        }
        if (!std::ranges::all_of(std::span(batchCounts).first(kPersistentResidentCards),
                [](uint32_t count) { return count == 1u + kRefreshSweeps; })) {
            return RHITestResult::fail("Stochastic direct refresh did not revisit every resident tile each sweep");
        }
        const auto& converged = accumulated->surface->artifacts[kSurfaceDirectIrradiance].bytes;
        double accumulatedSquaredError = 0.0;
        for (uint32_t y = 0; y < kTileEdge; ++y) {
            for (uint32_t x = 0; x < kTileEdge; ++x) {
                const auto index = cardPixelOffset(slot, y) + x;
                if (readSurfaceValue<uint16_t>(atlas[3].bytes, index) == UINT16_MAX) { continue; }
                const auto estimate = decode(readSurfaceValue<Pixel>(converged, index));
                const auto reference = decode(readSurfaceValue<Pixel>(exact, index));
                const size_t offset = (size_t(y) * kPreviewWidth + x + kTileEdge) * 4u;
                for (size_t channel = 0; channel < estimate.size(); ++channel) {
                    const double relativeError = double(estimate[channel]) / reference[channel] - 1.0;
                    accumulatedSquaredError += relativeError * relativeError;
                    pixels[offset + channel] = display(estimate[channel]);
                }
            }
        }
        if (!(accumulatedSquaredError < firstSquaredError * 0.65)) {
            return RHITestResult::fail("Stochastic direct history did not reduce error against the exact Rect sum");
        }
        if (!saveRgba8Png(imagePath, pixels.data(), kPreviewWidth, kTileEdge, log)) { return RHITestResult::fail(log); }

        const auto environmentBeforeSun = world.worldEnvironment();
        auto sunLighting = environmentBeforeSun;
        sunLighting.sun.direction = {0.0f, 0.0f, -1.0f};
        sunLighting.sun.illuminance = 1.0f;
        sunLighting.sun.enabled = true;
        if (!world.setWorldEnvironment(sunLighting)) { return RHITestResult::fail("Could not enable the fixture Sun"); }
        const auto sunRelit = runFrame("Relight deterministic Sun lighting", {0u, 0u, kPersistentResidentCards},
            kPersistentResidentCards, std::nullopt, false);
        if (!sunRelit) { return RHITestResult::fail(log); }
        const auto& beforeSunHistory = accumulated->surface->artifacts[kSurfaceStochasticHistory];
        const auto& afterSunHistory = sunRelit->surface->artifacts[kSurfaceStochasticHistory];
        uint32_t sunRefreshes = 0u;
        for (uint32_t tile = 0; tile < kPersistentResidentCards; ++tile) {
            const auto flags = readSurfaceValue<uint32_t>(sunRelit->surface->artifacts[kSurfaceUpdateFlags].bytes, tile);
            if (flags == kRefreshFlags) {
                ++sunRefreshes;
                ++batchCounts[tile];
            } else if (flags != kCardRelit || !sameTile(beforeSunHistory, afterSunHistory, tile, sizeof(Pixel))) {
                return RHITestResult::fail("Deterministic Sun relighting discarded accumulated Rect history");
            }
        }
        if (sunRefreshes != kRectReGIRTestRefreshCards) {
            return RHITestResult::fail("Deterministic Sun relighting suppressed the independent stochastic budget");
        }
        if (const auto message = checkHistory(*sunRelit); !message.empty()) { return RHITestResult::fail(message); }
        const auto directAfterSun = readSurfaceValue<Pixel>(sunRelit->surface->artifacts[kSurfaceDirectIrradiance].bytes, center);
        const auto exactAfterSun = readSurfaceValue<Pixel>(sunRelit->fullSurface->artifacts[kSurfaceDirectIrradiance].bytes, center);
        const auto directBeforeSun = readSurfaceValue<Pixel>(converged, center);
        const auto exactBeforeSun = readSurfaceValue<Pixel>(exact, center);
        const auto rectBeforeSun = readSurfaceValue<Pixel>(beforeSunHistory.bytes, center);
        const auto rectAfterSun = readSurfaceValue<Pixel>(afterSunHistory.bytes, center);
        for (size_t channel = 0; channel < 3; ++channel) {
            const float delta = (directAfterSun[channel] - rectAfterSun[channel]) -
                (directBeforeSun[channel] - rectBeforeSun[channel]);
            const float expected = exactAfterSun[channel] - exactBeforeSun[channel];
            if (!(expected > 0.01f && std::abs(delta - expected) < 0.0001f)) {
                return RHITestResult::fail("The deterministic Sun edit did not immediately reach the lit receiver");
            }
        }
        if (const auto message = checkRadiositySchedule(*accumulated, *sunRelit, indirectRefreshed,
                &indirectUnscheduledRelights); !message.empty()) {
            return RHITestResult::fail(message);
        }
        if (!world.setWorldEnvironment(environmentBeforeSun)) {
            return RHITestResult::fail("Could not restore the disabled fixture Sun");
        }

        for (auto& light : lighting.rectLights) { light.luminance *= 0.5f; }
        if (!world.setLighting(lighting)) { return RHITestResult::fail("Could not dim the many-Rect fixture"); }
        const auto dimmed = runFrame("Relight with a new Rect grid", {0u, 0u, kPersistentResidentCards},
            kPersistentResidentCards, std::nullopt, false);
        if (!dimmed) { return RHITestResult::fail(log); }
        if (const auto message = checkCardFlags(*dimmed, kPersistentResidentCards, receiver, kResetFlags, kResetFlags);
            !message.empty()) { return RHITestResult::fail(message); }
        std::fill_n(batchCounts.begin(), kPersistentResidentCards, 1u);
        if (const auto message = checkHistory(*dimmed); !message.empty()) { return RHITestResult::fail(message); }
        lighting.rectLights.resize(1);
        if (!world.setLighting(lighting)) { return RHITestResult::fail("Could not reduce the fixture to one Rect"); }
        const auto small = runFrame("Use the exact small-light path", {0u, 0u, kPersistentResidentCards});
        if (!small) { return RHITestResult::fail(log); }
        batchCounts.fill(0u);
        if (const auto message = checkHistory(*small); !message.empty()) { return RHITestResult::fail(message); }
        if (readSurfaceValue<Pixel>(small->surface->artifacts[kSurfaceDirectIrradiance].bytes, center)[0] <= 0.0f) {
            return RHITestResult::fail("The remaining Rect failed to light the receiver");
        }
        lighting.rectLights.front().enabled = false;
        if (!world.setLighting(lighting)) { return RHITestResult::fail("Could not disable the remaining Rect"); }
        const auto off = runFrame("Clear Rect lighting", {0u, 0u, kPersistentResidentCards});
        if (!off) { return RHITestResult::fail(log); }
        if (const auto message = checkHistory(*off); !message.empty()) { return RHITestResult::fail(message); }
        if (off->surface->artifacts[kSurfaceDirectIrradiance].bytes != atlas[kSurfaceDirectIrradiance].bytes) {
            return RHITestResult::fail("Disabled Rect left stale irradiance on the cards");
        }
        if (!runFrame("Idle after clearing Rect lighting", {0u, 0u, 0u})) { return RHITestResult::fail(log); }
        const auto summary = std::format("{} ReGIR draws light {} of {} covered texels with {} Rects. "
            "{} batches reduce relative RMS error from {:.3f} to {:.3f}.", kRectReGIRTestSamples, lit, covered,
            kLightsPerColor * 6, 1u + kRefreshSweeps, std::sqrt(firstSquaredError / (3.0 * covered)),
            std::sqrt(accumulatedSquaredError / (3.0 * covered)));
        spdlog::info("[LumenRectReGIR] {}", summary);
        return RHITestResult::pass(summary);
    }

    Mode mode_;
    static constexpr std::string_view kGdfPass = "GDF";
    static constexpr uint32_t kGdfResolution = 84;

    static bool makeGraph(render::RenderGraph& graph, const std::vector<std::string>& fields,
        const std::vector<std::string>& cards, bool testReGIR)
    {
        using render::makeRenderGraphFieldName;
        const auto edge = [&](std::string_view from, std::string_view source, std::string_view to, std::string_view target) {
            return graph.addEdge(makeRenderGraphFieldName(from, source), makeRenderGraphFieldName(to, target));
        };
        if (!graph.addNode("GlobalDistanceFieldPagesPass", std::string(kGdfPass), {{"distanceFieldPaths", fields},
                {"center", {0.0, 0.5, 0.65}}, {"extent", 2.2}, {"resolution", kGdfResolution}, {"maxPhysicalPages", 1728},
                {"quantized", true}, {"buildObjectGrid", true}, {"persistentPages", true}})) { return false; }
        for (const auto name : {kFullSurface, kPersistentSurface}) {
            render::RenderGraphProperties properties{{"meshCardPaths", cards},
                    {"atlasSize", kAtlasEdge}, {"tileResolution", kTileEdge}, {"shadowedLighting", true},
                    {"exportDirectIrradiance", true}, {"persistentCards", name == kPersistentSurface}};
            if (testReGIR) {
                const bool persistent = name == kPersistentSurface;
                properties["rectLightSamples"] = persistent ? kRectReGIRTestSamples : 0u;
                if (persistent) {
                    properties["maxStochasticDirectLightingCardsPerFrame"] = kRectReGIRTestRefreshCards;
                }
            }
            if (!graph.addNode("LumenSurfaceCachePass", std::string(name), std::move(properties))) { return false; }
            for (const auto* input : {"pageTable", "pageAtlas", "coarseMip", "domain"}) {
                if (!edge(kGdfPass, input, name, input)) { return false; }
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
                if (!edge(kGdfPass, input, name, input)) { return false; }
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
        render::RenderGraphExecutor& executor, const render::RenderGraphSubmitDesc& submission, std::string& log,
        bool captureStochasticHistory = false)
    {
        using render::makeRenderGraphFieldName;
        log.clear();
        constexpr uint32_t mipEdge = kGdfResolution / 4;
        const auto radiosityField = [](std::string_view name) { return makeRenderGraphFieldName(kPersistentRadiosity, name); };
        inspector.request = {{"method", "capture.batch"}, {"params", {
            {"batches", debug::DebugValue::array({
                {{"pass", kPersistentSurface}, {"resources", surfaceUpdateCaptures(kPersistentSurface, captureStochasticHistory)}},
                {{"pass", kFullSurface}, {"resources", surfaceUpdateCaptures(kFullSurface)}},
                {{"pass", kPersistentRadiosity}, {"resources", debug::DebugValue::array({
                    cardAtlasCapture(radiosityField("indirectIrradiance")), cardAtlasCapture(radiosityField("lighting")),
                    cardBufferCapture(radiosityField("updateStatistics"), 4),
                    cardBufferCapture(radiosityField("cardState"), 2 * kPersistentCardCapacity)})}},
                {{"pass", kDenseRadiosity}, {"resources", debug::DebugValue::array({
                    cardAtlasCapture(makeRenderGraphFieldName(kDenseRadiosity, "indirectIrradiance")),
                    cardAtlasCapture(makeRenderGraphFieldName(kDenseRadiosity, "traceRadiance"))})}},
                {{"pass", kGdfPass}, {"resources", debug::DebugValue::array({
                    cardBufferCapture(makeRenderGraphFieldName(kGdfPass, "updateStatistics"), 4),
                    {{"id", makeRenderGraphFieldName(kGdfPass, "coarseMip")},
                        {"roi", {{"x", 0}, {"y", 0}, {"width", mipEdge}, {"height", mipEdge}}},
                        {"slice", 0}, {"sliceCount", mipEdge}}})}},
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
        auto gdf = read(4);
        if (!surface || !full || !radiosity || !denseRadiosity || !gdf) { return std::nullopt; }
        if (gdf->artifacts.size() != 2 || gdf->artifacts[0].bytes.size() != 4 * sizeof(uint32_t) ||
            gdf->artifacts[1].bytes.size() != size_t(mipEdge) * mipEdge * mipEdge) {
            log = "Unexpected persistent GDF readback layout";
            return std::nullopt;
        }
        return CardUpdateFrame{std::move(surface), std::move(full), std::move(radiosity), std::move(denseRadiosity),
            std::move(gdf)};
    }
};

METALLIC_REGISTER_RHI_TEST(LumenPersistentCardUpdateTest);

class LumenRectDiffuseTest final : public LumenPersistentCardUpdateTest {
public:
    LumenRectDiffuseTest() : LumenPersistentCardUpdateTest(Mode::RectDiffuse) {}
};

METALLIC_REGISTER_RHI_TEST(LumenRectDiffuseTest);

class LumenRectReGIRTest final : public LumenPersistentCardUpdateTest {
public:
    LumenRectReGIRTest() : LumenPersistentCardUpdateTest(Mode::RectReGIR) {}
};

METALLIC_REGISTER_RHI_TEST(LumenRectReGIRTest);

class LumenHardwareSurfaceCacheTest : public RHITest {
public:
    enum class Mode { Punctual, Rect, RectReGIR };

    explicit LumenHardwareSurfaceCacheTest(Mode mode, const char* testName) : mode_(mode)
    {
        name = testName;
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        bench::ValidationRecorder validation;
        const bool hardwareRect = mode_ != Mode::Punctual;
        auto device = bench::createTestDevice(context, {.applicationName = "Lumen hardware card reference",
            .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true, .enableRayTracingAccelerationStructure = true, .enableRayQuery = true,
            .enableClusterAccelerationStructure = true, .validationSink = validation.sink()});
        if (hasError(device, Error::Unsupported)) {
            return RHITestResult::skip("Hardware card lighting requires scene acceleration structures and ray queries");
        }
        if (!device) { return RHITestResult::fail("Could not initialize the hardware card reference device"); }
        auto& gpu = **device;
        if (!gpu.capabilities().shaderStorageImageExtendedFormats) {
            return RHITestResult::skip("Hardware card lighting requires the R8 GDF and R16 depth storage formats");
        }

        const auto fixture = context.outputDirectory / "LumenHardwareLights.gltf";
        std::error_code copyError;
        std::filesystem::copy_file(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Lumen/SingleBounce.gltf", fixture,
            std::filesystem::copy_options::overwrite_existing, copyError);
        if (copyError) { return RHITestResult::fail("Could not copy the hardware shadow fixture: " + copyError.message()); }
        scene::SceneDocument document;
        if (!document.load(fixture)) { return RHITestResult::fail(document.lastLoadResult().error); }
        const auto receiver = document.objectForNode(0).entity();
        const auto blocker = document.objectForNode(2).entity();
        const auto blockerBetween = document.nodes()[2].worldMatrix;
        auto blockerAside = blockerBetween;
        blockerAside.a[12] += 1.3f;
        if (!document.setObjectVisible(document.objectForNode(1).entity(), false) ||
            !document.setObjectWorldMatrix(blocker, blockerAside)) {
            return RHITestResult::fail("Could not isolate the hardware shadow receiver");
        }

        std::string log;
        const auto field = context.outputDirectory / "LumenHardwareReceiver.meshdf";
        const auto receiverCards = context.outputDirectory / "LumenHardwareReceiver.meshcards";
        const auto blockerCards = context.outputDirectory / "LumenHardwareBlocker.meshcards";
        // Only the scene TLAS contains blocker triangles. A GDF shadow cannot pass this test.
        if (!writeAnalyticBoxField(field, document, log, 0) || !writeAnalyticBoxCards(receiverCards, document, 0, log) ||
            !writeAnalyticBoxCards(blockerCards, document, 2, log)) {
            return RHITestResult::fail(log);
        }
        RenderGraph graph;
        if (!graph.addNode("SceneAccelerationStructurePass", "SceneAS", {{"path", fixture.generic_string()},
                {"AsyncComputePreferred", false}, {"meshletNormalConeCull", false}}) ||
            !graph.addNode("GlobalDistanceFieldPagesPass", "GDF", {{"distanceFieldPaths", {field.generic_string()}},
                {"center", {0.0f, 0.5f, 0.65f}}, {"extent", 2.2f}, {"resolution", 84}, {"maxPhysicalPages", 1728},
                {"quantized", true}, {"persistentPages", true}}) ||
            !graph.addNode("LumenSurfaceCachePass", "SurfaceCache", {
                {"meshCardPaths", {receiverCards.generic_string(), blockerCards.generic_string()}},
                {"atlasSize", kAtlasEdge}, {"tileResolution", kTileEdge}, {"shadowedLighting", true},
                {"hardwarePunctualShadows", true}, {"exportDirectIrradiance", true}, {"persistentCards", true},
                {"maxDirectLightingCardsPerFrame", kCardsPerBox},
                {"hardwareRectShadows", hardwareRect}, {"rectLightSamples", mode_ == Mode::RectReGIR ? 1u : 0u},
                {"maxStochasticDirectLightingCardsPerFrame", hardwareRect ? kCardCount : 0u}}) ||
            !graph.addEdge("SceneAS.accelerationStructure", "SurfaceCache.accelerationStructure") ||
            !graph.markOutput("SurfaceCache.directIrradiance")) {
            return RHITestResult::fail("Could not configure the hardware Surface Cache graph");
        }
        for (const char* name : {"pageTable", "pageAtlas", "coarseMip", "domain"}) {
            if (!graph.addEdge(makeRenderGraphFieldName("GDF", name), makeRenderGraphFieldName("SurfaceCache", name))) {
                return RHITestResult::fail("Could not connect the Surface Cache GDF inputs");
            }
        }
        scene::PunctualLight point, spot;
        point.properties.intensityUnit = spot.properties.intensityUnit = scene::LightUnit::Candela;
        point.properties.intensity = 4.0;
        spot.properties.intensity = 8.0;
        point.properties.range = spot.properties.range = 4.0;
        point.position = float3(0.0f, 1.2f, 0.95f);
        spot.position = point.position;
        spot.properties.type = "spot";
        spot.properties.innerConeAngle = 0.35;
        spot.properties.outerConeAngle = 0.5;
        spot.direction = float3(0.0f, -0.8f, -0.6f);
        scene::LightingSettings lights;
        scene::RectLight rect;
        if (hardwareRect) {
            rect.position = point.position;
            rect.direction = spot.direction;
            rect.width = rect.height = 0.30f;
            const uint32_t rectCount = mode_ == Mode::RectReGIR ? 2u : 1u;
            rect.luminance = 30.0f / float(rectCount);
            lights.rectLights.assign(rectCount, rect);
        } else {
            lights.lights = {point, spot};
        }

        RenderWorld world;
        world.setScene(&document);
        world.setEnvironment({.enabled = false, .visible = false});
        environment::WorldEnvironment environment;
        environment.sun.enabled = environment.moon.enabled = false;
        if (!world.setWorldEnvironment(environment) || !world.setLighting(lights)) {
            return RHITestResult::fail("Could not configure the fixture lights");
        }
        RenderView view;
        ViewCamera camera;
        camera.eye = {0.0f, 0.0f, 3.0f};
        camera.orthographic = true;
        camera.orthoHeight = 1.25f;
        if (!view.setCamera(camera) || !view.setRenderResolution(kAtlasEdge, kAtlasEdge)) {
            return RHITestResult::fail("Invalid hardware shadow fixture camera");
        }

        CardUpdateInspector inspector;
        inspector.request = {{"method", "capture.batch"}, {"params", {{"batches", debug::DebugValue::array({
            {{"pass", "SurfaceCache"}, {"resources", debug::DebugValue::array({
                cardAtlasCapture("SurfaceCache.directIrradiance"), cardAtlasCapture("SurfaceCache.depth"),
                cardBufferCapture("SurfaceCache.cards", kPersistentCardCapacity * sizeof(CapturedCard) / sizeof(uint32_t)),
                cardBufferCapture("SurfaceCache.cardUpdates", kPersistentCardCapacity)})}}})}}}};
        RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        executor.bindRenderView(&view);
        executor.setDebugObserver(&inspector);
        if (!executor.compile(gpu, graph, kAtlasEdge, kAtlasEdge, RenderGraphCompileOptions{.enableDebugReadback = true}, log)) {
            return RHITestResult::fail("Hardware card graph compile: " + log);
        }
        const RenderGraphSubmitDesc submission{.graphicsQueue = gpu.getQueue(QueueType::Graphics), .recordingWorkerLimit = 1};
        const auto capture = [&](std::string_view stage) -> CardCapture {
            log = std::string(stage);
            if (const auto result = executor.execute(submission); !result) {
                log += ": " + std::string(resultToString(result));
                return {};
            }
            if (!executor.waitForSubmittedWork()) { log += ": Could not wait for the GPU frame"; return {}; }
            inspector.runtime.poll();
            const auto& response = inspector.response;
            if (response.value("status", "") != "ok") { log += ": " + response.dump(); return {}; }
            const auto id = response.at("result").at("jobs").at(0).at("job").get<std::string>();
            auto result = inspector.runtime.core().completedCapture(id);
            if (!result || result->artifacts.size() != 4u ||
                result->artifacts[0].bytes.size() != kAtlasPixels * sizeof(Pixel) ||
                result->artifacts[1].bytes.size() != kAtlasPixels * sizeof(uint16_t) ||
                result->artifacts[2].bytes.size() != kPersistentCardCapacity * sizeof(CapturedCard) ||
                result->artifacts[3].bytes.size() != kPersistentCardCapacity * sizeof(uint32_t)) {
                log += ": Missing or invalid Surface Cache readback";
                return {};
            }
            return result;
        };

        const auto open = capture("Visible local lights");
        if (!open) { return RHITestResult::fail(log); }
        uint32_t receiverSlot = 0;
        for (; receiverSlot < kPersistentCardCapacity; ++receiverSlot) {
            const auto card = readSurfaceValue<CapturedCard>(open->artifacts[2].bytes, receiverSlot);
            if (card.metadata[0] == uint32_t(receiver) && card.axisZ[2] > 0.99f) { break; }
        }
        if (receiverSlot == kPersistentCardCapacity) { return RHITestResult::fail("Missing the receiver front card"); }
        const size_t texel = tileCenter(receiverSlot);
        const auto receiverDepth = readSurfaceValue<uint16_t>(open->artifacts[1].bytes, texel);
        if (receiverDepth == UINT16_MAX) { return RHITestResult::fail("The receiver center texel is uncovered"); }
        const auto center = cardCenterWorldPosition(
            readSurfaceValue<CapturedCard>(open->artifacts[2].bytes, receiverSlot), receiverDepth);
        const float3 toLight = point.position - float3(center[0], center[1], center[2]);
        const auto centerIrradiance = [&](const CardCapture& frame) {
            return readSurfaceValue<Pixel>(frame->artifacts[0].bytes, texel);
        };
        const auto matches = [&](const CardCapture& frame, float expected, float relativeTolerance = 0.025f) {
            const auto actual = centerIrradiance(frame);
            return std::ranges::all_of(std::span(actual).first<3>(), [=](float sample) {
                return std::abs(sample - expected) <= std::max(1e-5f, expected * relativeTolerance);
            });
        };
        if (hardwareRect) {
            const auto rectCapture = [&](std::string_view stage) -> CardCapture {
                // Include the two direct-light budgets, then allow the per-card estimate to settle.
                CardCapture result;
                for (uint32_t frame = 0; frame < 20u; ++frame) {
                    result = capture(stage);
                    if (!result) { return {}; }
                }
                return result;
            };
            // Area quadrature gives an independent irradiance reference for the unblocked Rect.
            constexpr uint32_t strata = 16u;
            const float3 right{1.0f, 0.0f, 0.0f};
            const float3 up = cross(right, rect.direction);
            float expected = 0.0f;
            for (uint32_t y = 0; y < strata; ++y) {
                for (uint32_t x = 0; x < strata; ++x) {
                    const float offsetX = ((float(x) + 0.5f) / strata - 0.5f) * rect.width;
                    const float offsetY = ((float(y) + 0.5f) / strata - 0.5f) * rect.height;
                    const float3 sample = toLight + right * offsetX + up * offsetY;
                    const float distanceSquared = dot(sample, sample);
                    expected += -dot(rect.direction, sample) * sample.z / (distanceSquared * distanceSquared);
                }
            }
            expected *= rect.luminance * float(lights.rectLights.size()) * rect.width * rect.height / float(strata * strata);
            const auto unblocked = rectCapture("Open Rect");
            if (!unblocked) { return RHITestResult::fail(log); }
            if (!matches(unblocked, expected, 0.12f)) {
                return RHITestResult::fail(std::format("Open Rect expected {} lx, observed {} lx",
                    expected, centerIrradiance(unblocked)[0]));
            }

            // The x edge intersects the Rect solid angle while the center ray is inside the blocker.
            auto blockerPartial = blockerBetween;
            constexpr float blockerHalfWidth = 0.55f;
            const float centerRayX = center[0] + toLight.x * ((blockerBetween.a[13] - center[1]) / toLight.y);
            blockerPartial.a[12] = centerRayX + blockerHalfWidth - 0.015f;
            if (!document.setObjectWorldMatrix(blocker, blockerPartial)) {
                return RHITestResult::fail("Could not move the TLAS-only blocker onto the Rect edge");
            }
            const auto partial = rectCapture("Partially blocked Rect");
            if (!partial) { return RHITestResult::fail(log); }
            const float partialIrradiance = centerIrradiance(partial)[0];
            if (!(partialIrradiance > 0.1f * expected && partialIrradiance < 0.9f * expected)) {
                return RHITestResult::fail(std::format("Partial Rect occlusion must retain a fraction of {} lx, observed {} lx",
                    expected, partialIrradiance));
            }

            if (!document.setObjectWorldMatrix(blocker, blockerBetween)) {
                return RHITestResult::fail("Could not move the TLAS-only blocker over the Rect");
            }
            const auto blocked = rectCapture("Fully blocked Rect");
            if (!blocked) { return RHITestResult::fail(log); }
            if (!matches(blocked, 0.0f)) {
                return RHITestResult::fail(std::format("Fully blocked Rect contributed {} lx", centerIrradiance(blocked)[0]));
            }

            auto blockerBeyond = blockerBetween;
            const float3 beyond = rect.position + 0.5f * toLight;
            blockerBeyond.a[12] = beyond.x;
            blockerBeyond.a[13] = beyond.y;
            blockerBeyond.a[14] = beyond.z;
            if (!document.setObjectWorldMatrix(blocker, blockerBeyond)) {
                return RHITestResult::fail("Could not move the blocker beyond the Rect");
            }
            const auto restored = rectCapture("Rect with the blocker beyond the light");
            if (!restored) { return RHITestResult::fail(log); }
            if (!matches(restored, expected, 0.12f)) {
                return RHITestResult::fail(std::format("A blocker beyond the Rect changed receiver irradiance to {} lx",
                    centerIrradiance(restored)[0]));
            }

            for (auto& light : lights.rectLights) { light.direction = -rect.direction; }
            if (!world.setLighting(lights)) { return RHITestResult::fail("Could not turn the Rect away"); }
            const auto dark = rectCapture("Back of Rect");
            if (!dark) { return RHITestResult::fail(log); }
            if (!matches(dark, 0.0f)) {
                return RHITestResult::fail(std::format("A backfacing Rect contributed {} lx", centerIrradiance(dark)[0]));
            }
            if (validation.failed()) {
                return RHITestResult::fail("Hardware Rect Vulkan validation: " + validation.snapshot().dump());
            }
            const auto summary = std::format("Rect center irradiance: {:.4f} lx visible ({:.4f} lx reference), "
                "{:.4f} lx partially blocked, {:.4f} lx fully blocked, {:.4f} lx beyond light, {:.4f} lx backfacing.",
                centerIrradiance(unblocked)[0], expected, partialIrradiance, centerIrradiance(blocked)[0],
                centerIrradiance(restored)[0], centerIrradiance(dark)[0]);
            spdlog::info("[LumenHardwareRect] {}", summary);
            return RHITestResult::pass(summary);
        }

        const float distance = std::hypot(toLight.x, toLight.y, toLight.z);
        const float ratio = distance / float(point.properties.range);
        const float expectedPoint = float(point.properties.intensity) * toLight.z *
            (1.0f - ratio * ratio * ratio * ratio) / (distance * distance * distance);
        const float expectedOpen = 3.0f * expectedPoint;
        if (!matches(open, expectedOpen)) {
            return RHITestResult::fail(std::format("Point and spot expected {} lx, observed {} lx",
                expectedOpen, centerIrradiance(open)[0]));
        }

        if (!document.setObjectWorldMatrix(blocker, blockerBetween)) {
            return RHITestResult::fail("Could not move the TLAS-only blocker into the ray segment");
        }
        const auto deferred = capture("Blocker recapture uses the card budget");
        if (!deferred) { return RHITestResult::fail(log); }
        const auto& before = open->artifacts[0].bytes;
        const auto& after = deferred->artifacts[0].bytes;
        if (readSurfaceValue<uint32_t>(deferred->artifacts[3].bytes, receiverSlot) != 0u) {
            return RHITestResult::fail("An unselected receiver card was unexpectedly relit");
        }
        for (uint32_t row = 0; row < kTileEdge; ++row) {
            const size_t offset = size_t(cardPixelOffset(receiverSlot, row)) * sizeof(Pixel);
            if (std::memcmp(before.data() + offset, after.data() + offset, kTileEdge * sizeof(Pixel)) != 0) {
                return RHITestResult::fail("An unselected receiver card failed to preserve its previous irradiance");
            }
        }
        const auto occluded = capture("Relight after moving blocker");
        if (!occluded) { return RHITestResult::fail(log); }
        if (!matches(occluded, 0.0f) ||
            !(readSurfaceValue<uint32_t>(occluded->artifacts[3].bytes, receiverSlot) & kCardRelit)) {
            return RHITestResult::fail("The updated receiver did not observe the moved hardware blocker");
        }
        auto blockerBeyond = blockerBetween;
        const float3 beyond = point.position + 0.5f * toLight;
        blockerBeyond.a[12] = beyond.x;
        blockerBeyond.a[13] = beyond.y;
        blockerBeyond.a[14] = beyond.z;
        if (!document.setObjectWorldMatrix(blocker, blockerBeyond)) {
            return RHITestResult::fail("Could not move the blocker beyond the light");
        }
        if (!capture("Blocker recapture behind the light")) { return RHITestResult::fail(log); }
        const auto beyondLight = capture("Relight with the blocker beyond the light");
        if (!beyondLight) { return RHITestResult::fail(log); }
        if (centerIrradiance(beyondLight) != centerIrradiance(open)) {
            return RHITestResult::fail("A blocker beyond the finite point or spot ray changed receiver irradiance");
        }

        lights.lights[1].direction = -spot.direction;
        if (!world.setLighting(lights)) { return RHITestResult::fail("Could not turn the spot away"); }
        if (!capture("Spot edit first budget")) { return RHITestResult::fail(log); }
        const auto pointOnly = capture("Spot edit second budget");
        if (!pointOnly) { return RHITestResult::fail(log); }
        if (!matches(pointOnly, expectedPoint)) { return RHITestResult::fail("The spot outside the cone contributed irradiance"); }
        lights.lights.clear();
        if (!world.setLighting(lights)) { return RHITestResult::fail("Could not remove the fixture lights"); }
        if (!capture("Light removal first budget")) { return RHITestResult::fail(log); }
        const auto dark = capture("Light removal second budget");
        if (!dark) { return RHITestResult::fail(log); }
        if (!std::ranges::all_of(dark->artifacts[0].bytes, [](uint8_t byte) { return byte == 0u; })) {
            return RHITestResult::fail("Removing all lights left stale direct irradiance");
        }
        if (validation.failed()) {
            return RHITestResult::fail("Hardware punctual Vulkan validation: " + validation.snapshot().dump());
        }
        const auto summary = std::format("Center irradiance: {:.4f} lx visible, {:.4f} lx blocked, {:.4f} lx beyond light, "
            "{:.4f} lx point only, {:.4f} lx with zero lights.", centerIrradiance(open)[0], centerIrradiance(occluded)[0],
            centerIrradiance(beyondLight)[0], centerIrradiance(pointOnly)[0], centerIrradiance(dark)[0]);
        spdlog::info("[LumenHardwarePunctual] {}", summary);
        return RHITestResult::pass(summary);
    }

private:
    using Pixel = std::array<float, 4>;
    Mode mode_;
};

class LumenHardwarePunctualSurfaceCacheTest final : public LumenHardwareSurfaceCacheTest {
public:
    LumenHardwarePunctualSurfaceCacheTest()
        : LumenHardwareSurfaceCacheTest(Mode::Punctual, "lumen_surface_cache_hardware_punctual") {}
};

class LumenHardwareRectSurfaceCacheTest final : public LumenHardwareSurfaceCacheTest {
public:
    LumenHardwareRectSurfaceCacheTest()
        : LumenHardwareSurfaceCacheTest(Mode::Rect, "lumen_surface_cache_hardware_rect") {}
};

class LumenHardwareRectReGIRSurfaceCacheTest final : public LumenHardwareSurfaceCacheTest {
public:
    LumenHardwareRectReGIRSurfaceCacheTest()
        : LumenHardwareSurfaceCacheTest(Mode::RectReGIR, "lumen_surface_cache_hardware_rect_regir") {}
};

METALLIC_REGISTER_RHI_TEST(LumenHardwarePunctualSurfaceCacheTest);
METALLIC_REGISTER_RHI_TEST(LumenHardwareRectSurfaceCacheTest);
METALLIC_REGISTER_RHI_TEST(LumenHardwareRectReGIRSurfaceCacheTest);

constexpr uint64_t kFastLightingPulseFrame = 6;
constexpr uint64_t kFastLightingDisocclusionFrame = kFastLightingPulseFrame + 3;
constexpr float kFastLightingPulseAmount = 0.7265625f;

class LumenTemporalInputPass final : public render::UnsafePass {
public:
    static constexpr std::array<float, 4> kConstantLighting{0.5f, 0.25f, 0.125f, 1.0f};

    static bool ensureRegistered()
    {
        static const bool registered = render::registerRenderGraphPassType("LumenTemporalInputPass", "Lumen test lighting input",
            [] { return std::make_unique<LumenTemporalInputPass>(); });
        return registered;
    }

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("indirectDiffuse").transferWrite().format = render::Format::RGBA16Sfloat;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const bool fastLighting = properties().value("fastLightingSequence", false);
        const auto frame = context.frameIndex();
        const bool pulse = frame == kFastLightingPulseFrame || frame == kFastLightingDisocclusionFrame;
        const float red = fastLighting && frame >= kFastLightingPulseFrame ? 0.875f : kConstantLighting[0];
        const float alpha = fastLighting ? (pulse ? 0.078125f : 0.004f) : kConstantLighting[3];
        return context.commandBuffer().clearColorTexture(*context.outputTexture("indirectDiffuse").texture(),
            render::TextureLayout::TransferDestination, {red, kConstantLighting[1], kConstantLighting[2], alpha});
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

    float fastUpdateAmount(size_t pixel) const
    {
        return readSurfaceValue<float>(temporal->artifacts[1].bytes, pixel);
    }

    float indirectDiffuseRed(size_t pixel) const
    {
        return halfToFloat(readSurfaceValue<std::array<uint16_t, 4>>(temporal->artifacts[2].bytes, pixel)[0]);
    }

    float screenHistoryDepth(size_t pixel) const
    {
        return readSurfaceValue<float>(screenTrace->artifacts[0].bytes, pixel);
    }
};

class LumenDynamicTemporalTest : public RHITest {
public:
    explicit LumenDynamicTemporalTest(bool fastLighting = false) : fastLighting_(fastLighting)
    {
        name = fastLighting ? "lumen_fast_temporal_history" : "lumen_dynamic_temporal";
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
        if (!makeGraph(graph, fastLighting_)) { return RHITestResult::fail("Could not build the dynamic Lumen temporal graph"); }
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
        const auto renderFrame = [&]() { return captureFrame(inspector, executor, submission, fastLighting_, log); };

        const auto initial = renderFrame();
        if (!initial || !historyEmpty(*initial) || !historyEmpty(*initial, false) || screenHistoryPixels(*initial) != 0u) {
            return RHITestResult::fail(initial ? "First frame unexpectedly used temporal history" : log);
        }
        auto stationary = renderFrame();
        if (!stationary) { return RHITestResult::fail(log); }
        stationary = renderFrame();
        if (!stationary) { return RHITestResult::fail(log); }

        if (fastLighting_) {
            for (uint64_t frame = 3; frame < kFastLightingPulseFrame; ++frame) {
                stationary = renderFrame();
                if (!stationary) { return RHITestResult::fail(log); }
            }
            const auto pulse = renderFrame();
            const auto retained = renderFrame();
            const auto quiet = renderFrame();
            if (!pulse || !retained || !quiet) { return RHITestResult::fail(log); }
            if (const auto error = compareFastLightingSequence(*stationary, *pulse, *retained, *quiet); !error.empty()) {
                return RHITestResult::fail(error);
            }
        }

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
            const bool movingLightAccepted = !fastLighting_ ||
                std::abs(moved->fastUpdateAmount(pixel) - kFastLightingPulseAmount) < 0.0001f;
            if (current[2] > 0.75f && std::abs(current[0] - destinationX) < 0.06f && before[2] < 0.5f) {
                ++shiftedPixels;
                if (previous[3] <= 0.0f || std::abs(previous[0] - current[0] + destinationX - startX) > 0.002f ||
                    std::abs(previous[1] - current[1]) > 0.002f || std::abs(previous[2] - current[2]) > 0.002f ||
                    !(weight > 0.6f) || !movingLightAccepted) {
                    return RHITestResult::fail("Translated receiver did not reproject to its previous world-space surface");
                }
            } else if (before[2] > 0.75f && current[2] < 0.5f && std::abs(current[0] - startX) < 0.06f) {
                ++revealedPixels;
                if (weight != 0.0f || (fastLighting_ && moved->fastUpdateAmount(pixel) != 0.0f)) {
                    return RHITestResult::fail("Newly revealed receiver retained foreground GI history");
                }
            } else if (std::abs(current[0]) < 0.08f && before[2] < 0.5f && current[2] < 0.5f) {
                ++staticPixels;
                if (previous[3] <= 0.0f || !(weight > 0.6f) || !movingLightAccepted) {
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
        if (fastLighting_) {
            if (validation.failed()) {
                return RHITestResult::fail("Fast temporal Vulkan validation: " + validation.snapshot().dump());
            }
            return RHITestResult::pass("Diffuse motion accelerated integration, carried the packed amount for one frame, "
                "and rejected reprojected history on newly revealed receivers.");
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
    static std::string compareFastLightingSequence(const LumenTemporalFrame& stationary, const LumenTemporalFrame& pulse,
        const LumenTemporalFrame& retained, const LumenTemporalFrame& quiet)
    {
        // The pulse's alpha 0.078125 yields an amount of 0.7265625. Packing rounds the age
        // to 4/15 of the ten-frame cap and rounds the fast amount up to 11/15.
        constexpr std::array<float, 3> ages{2.734375f, 8.0f / 3.0f, 11.0f / 3.0f};
        constexpr std::array<float, 3> amounts{kFastLightingPulseAmount, 0.0f, 0.0f};
        const std::array frames{&pulse, &retained, &quiet};
        size_t checked = 0;
        for (size_t pixel = 0; pixel < kAtlasPixels; ++pixel) {
            const auto position = stationary.position(pixel);
            if (position[3] <= 0.0f || position[2] >= 0.5f ||
                std::abs(position[0]) >= 0.08f || std::abs(position[1]) >= 0.09f) {
                continue;
            }
            if (stationary.fastUpdateAmount(pixel) != 0.0f || stationary.indirectDiffuseRed(pixel) != 0.5f ||
                std::abs(stationary.historyWeight(pixel) - stationary.historyWeight(pixel, false)) > 0.0001f) {
                return std::format("Static-alpha baseline differs before motion at pixel {}", pixel);
            }
            float previousRed = stationary.indirectDiffuseRed(pixel);
            for (size_t phase = 0; phase < frames.size(); ++phase) {
                const auto& frame = *frames[phase];
                const float alpha = 1.0f / (1.0f + ages[phase]);
                const float expectedWeight = 1.0f - alpha;
                const float expectedRed = std::lerp(previousRed, 0.875f, alpha);
                const float weight = frame.historyWeight(pixel);
                const float amount = frame.fastUpdateAmount(pixel);
                const float red = frame.indirectDiffuseRed(pixel);
                // The indirect output and the carried red channel both use FP16.
                if (!(std::abs(amount - amounts[phase]) <= 0.0001f && std::abs(weight - expectedWeight) <= 0.0001f &&
                    std::abs(red - expectedRed) <= 0.0005f && weight < frame.historyWeight(pixel, false))) {
                    return std::format("Fast temporal phase {} at pixel {} expected amount {:.6f}, weight {:.6f}, red {:.6f}; "
                        "got {:.6f}, {:.6f}, {:.6f}", phase, pixel, amounts[phase], expectedWeight, expectedRed, amount, weight, red);
                }
                previousRed = red;
            }
            ++checked;
        }
        return checked >= 80u ? std::string{} : "Insufficient stationary pixels for the fast temporal lighting sequence";
    }

    static bool makeGraph(render::RenderGraph& graph, bool fastLighting)
    {
        using namespace render;
        if (!LumenTemporalInputPass::ensureRegistered() ||
            !graph.addNode("VisibilityBufferPass", "VBuffer", {{"visualization", "none"}, {"enableClusterRtx", false}}) ||
            !graph.addNode("VisibilityBufferDeferredPass", "Deferred", {{"materialBinning", false}, {"outputLinear", true},
                {"exportLumenReceivers", true}, {"exportLumenPreviousPosition", true}}) ||
            !graph.addNode("LumenTemporalInputPass", "Fixture", {{"fastLightingSequence", fastLighting}}) ||
            !graph.addNode("LumenTemporalGatherPass", "Temporal", {{"dynamicReprojection", true},
                {"publishScreenTraceHistory", true}, {"fastLightingUpdate", fastLighting}, {"maxFramesAccumulated", 10u}}) ||
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
        const render::RenderGraphSubmitDesc& submission, bool fastLighting, std::string& log)
    {
        using namespace render;
        auto temporalResources = debug::DebugValue::array({cardAtlasCapture("Temporal.historyWeight")});
        if (fastLighting) {
            temporalResources.push_back(cardAtlasCapture("Temporal.fastUpdateAmount"));
            temporalResources.push_back(cardAtlasCapture("Temporal.indirectDiffuse"));
        }
        inspector.request = {{"method", "capture.batch"}, {"params", {{"batches", debug::DebugValue::array({
            {{"pass", "Deferred"}, {"resources", debug::DebugValue::array({cardAtlasCapture("Deferred.lumenPosition"),
                cardAtlasCapture("Deferred.lumenPreviousPosition")})}},
            {{"pass", "Temporal"}, {"resources", std::move(temporalResources)}},
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
        const bool temporalLayout = fastLighting ? temporal->artifacts.size() == 3 &&
            temporal->artifacts[0].bytes.size() == kAtlasPixels * sizeof(float) &&
            temporal->artifacts[1].bytes.size() == kAtlasPixels * sizeof(float) &&
            temporal->artifacts[2].bytes.size() == kAtlasPixels * sizeof(std::array<uint16_t, 4>) : singleFloatOutput(temporal);
        if (deferred->artifacts.size() != 2 ||
            deferred->artifacts[0].bytes.size() != kAtlasPixels * sizeof(std::array<float, 4>) ||
            deferred->artifacts[1].bytes.size() != kAtlasPixels * sizeof(std::array<float, 4>) ||
            !temporalLayout || !singleFloatOutput(staticTemporal) || !singleFloatOutput(screenTrace)) {
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

    bool fastLighting_;
};

METALLIC_REGISTER_RHI_TEST(LumenDynamicTemporalTest);

class LumenFastTemporalHistoryTest final : public LumenDynamicTemporalTest {
public:
    LumenFastTemporalHistoryTest() : LumenDynamicTemporalTest(true) {}
};

METALLIC_REGISTER_RHI_TEST(LumenFastTemporalHistoryTest);

class LumenMovingGatherHistoryTest final : public RHITest {
public:
    LumenMovingGatherHistoryTest()
    {
        name = "lumen_moving_gather_history";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::SceneDocument document;
        RenderGraph graph;
        std::string log;
        if (!loadSingleBounceFixture(context, document, graph, log)) { return RHITestResult::fail(log); }
        const auto blocker = std::ranges::find_if(document.renderNodes(), [&](const auto& node) {
            return document.renderPrimitives()[node.renderPrimitiveIndex].meshIndex == 2;
        });
        if (blocker == document.renderNodes().end() || !prepareGraph(graph)) {
            return RHITestResult::fail("Could not wire or locate the moving blocker fixture");
        }
        const auto blockerObject = blocker->object;
        auto blockerTransform = document.nodes().at(2).worldMatrix;

        RenderView view;
        ViewCamera camera;
        camera.eye = {1.1f, 0.72f, 3.0f};
        camera.center = {0.1f, 0.66f, 0.48f};
        camera.fovDegrees = 58.0f;
        camera.farPlane = 8.0f;
        if (!view.setCamera(camera) || !view.setRenderResolution(kSingleBounceEdge, kSingleBounceEdge)) {
            return RHITestResult::fail("Could not initialize the moving blocker camera");
        }
        bench::ValidationRecorder validation;
        CardUpdateInspector inspector;
        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        if (!preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink())) {
            return RHITestResult::fail("Moving Gather preview setup: " + preview.lastLog());
        }
        preview.setEnvironment({.enabled = false, .visible = false});
        if (!preview.setWorldEnvironment(environment::WorldEnvironment{}) || !preview.setLighting(document.lighting())) {
            return RHITestResult::fail("Could not initialize the emissive moving blocker scene");
        }
        preview.setRecordingWorkerLimit(1);
        preview.setDebugObserver(&inspector);
        inspector.request = {{"method", "capture.batch"}, {"params", {{"batches", debug::DebugValue::array({
            {{"pass", "Gather"}, {"resources", debug::DebugValue::array({
                cardBufferCapture("Gather.probeWorkState", 2u), cardBufferCapture("Gather.probeWorkList", kProbeSlots),
                cardBufferCapture("Gather.probeTraceMoving", kTraceRays), cardBufferCapture("Gather.probeRadiance", kTraceRays * 4u)})}},
            {{"pass", "Deferred"}, {"resources", debug::DebugValue::array({
                cardAtlasCapture("Deferred.lumenPosition"), cardAtlasCapture("Deferred.lumenPreviousPosition")})}},
            {{"pass", "Temporal"}, {"resources", debug::DebugValue::array({
                cardAtlasCapture("Temporal.historyWeight"), cardAtlasCapture("Temporal.fastUpdateAmount")})}},
            {{"pass", "ScreenTrace"}, {"resources", debug::DebugValue::array({
                cardBufferCapture("ScreenTrace.traceView", kTraceViewWords)})}},
        })}}}};

        std::array<CardCapture, 4> stationary;
        uint32_t staticScreenRays = 0, staticScreenMotion = 0, staticDetailMotion = 0;
        for (uint32_t warmup = 0; warmup < 10u; ++warmup) {
            if (auto error = captureFrame(graph, preview, inspector, stationary); !error.empty()) { return RHITestResult::fail(error); }
            if (auto error = addRayCounts(*stationary[0], staticScreenRays, staticScreenMotion, staticDetailMotion); !error.empty()) {
                return RHITestResult::fail(error);
            }
        }
        const uint32_t staticMode = traceMode(stationary);
        if (staticMode != 1u || staticScreenRays == 0 || staticScreenMotion != 0 || staticDetailMotion != 0) {
            return RHITestResult::fail(std::format("Stationary fixture produced {} accepted screen rays in mode {}; "
                "{} moving screen and {} moving detail rays", staticScreenRays, staticMode, staticScreenMotion, staticDetailMotion));
        }

        uint32_t screenRays = 0, screenMotion = 0, detailMotion = 0, reactingReceivers = 0, quietReceivers = 0;
        std::array<uint32_t, 3> movingModes{};
        for (size_t step = 0; step < movingModes.size(); ++step) {
            blockerTransform.a[12] += 0.10f;
            if (!document.setObjectWorldMatrix(blockerObject, blockerTransform)) {
                return RHITestResult::fail("Could not translate the opaque blocker");
            }
            std::array<CardCapture, 4> moved;
            if (auto error = captureFrame(graph, preview, inspector, moved); !error.empty()) { return RHITestResult::fail(error); }
            movingModes[step] = traceMode(moved);
            if (auto error = addRayCounts(*moved[0], screenRays, screenMotion, detailMotion); !error.empty()) {
                return RHITestResult::fail(error);
            }
            using Position = std::array<float, 4>;
            for (size_t pixel = 0; pixel < kAtlasPixels; ++pixel) {
                const auto before = readSurfaceValue<Position>(stationary[1]->artifacts[0].bytes, pixel);
                const auto now = readSurfaceValue<Position>(moved[1]->artifacts[0].bytes, pixel);
                const auto previous = readSurfaceValue<Position>(moved[1]->artifacts[1].bytes, pixel);
                if (before[3] <= 0.0f || now[3] <= 0.0f || previous[3] <= 0.0f ||
                    !std::equal(now.begin(), now.begin() + 3, before.begin(),
                        [](float a, float b) { return std::abs(a - b) < 0.001f; }) ||
                    !std::equal(now.begin(), now.begin() + 3, previous.begin(),
                        [](float a, float b) { return std::abs(a - b) < 0.001f; })) {
                    continue;
                }
                const float previousWeight = readSurfaceValue<float>(stationary[2]->artifacts[0].bytes, pixel);
                const float weight = readSurfaceValue<float>(moved[2]->artifacts[0].bytes, pixel);
                const float amount = readSurfaceValue<float>(moved[2]->artifacts[1].bytes, pixel);
                if (previousWeight <= 0.8f) { continue; }
                reactingReceivers += amount > 0.0f && weight > 0.0f && weight < previousWeight - 0.001f;
                quietReceivers += amount == 0.0f && std::abs(weight - previousWeight) < 0.001f;
            }
        }
        if (!std::ranges::all_of(movingModes, [](uint32_t mode) { return mode == 2u; }) ||
            screenMotion == 0 || detailMotion == 0 || reactingReceivers == 0 || quietReceivers == 0) {
            return RHITestResult::fail(std::format("Screen history warmup mode {}, movement modes [{}, {}, {}]; {} stationary and "
                "{} moving-frame accepted Screen rays; {} moving Screen and {} moving Detail rays; "
                "{} static receivers accelerated and {} retained their steady weight", staticMode, movingModes[0], movingModes[1],
                movingModes[2], staticScreenRays, screenRays, screenMotion, detailMotion, reactingReceivers, quietReceivers));
        }
        if (validation.failed()) {
            return RHITestResult::fail("Moving Gather Vulkan validation: " + validation.snapshot().dump());
        }
        return RHITestResult::pass(std::format("Moving blocker supplied {} accepted moving screen and {} moving detail rays; "
            "{} static receiver samples accelerated and {} retained steady weight", screenMotion, detailMotion, reactingReceivers, quietReceivers));
    }

private:
    static constexpr uint32_t kTilesPerAxis = render::builtin_pass::lumenScreenProbeTiles(kSingleBounceEdge);
    static constexpr uint32_t kProbeSlots = kTilesPerAxis * kTilesPerAxis * 3u / 2u;
    static constexpr uint32_t kRaysPerSlot = 64u;
    static constexpr uint32_t kTraceRays = kProbeSlots * kRaysPerSlot;
    static constexpr uint32_t kTraceViewState = sizeof(render::ViewConstants) / sizeof(uint32_t);
    static constexpr uint32_t kTraceViewWords = kTraceViewState + 4u;

    static uint32_t traceMode(const std::array<CardCapture, 4>& frame)
    {
        return readSurfaceValue<uint32_t>(frame[3]->artifacts[0].bytes, kTraceViewState);
    }

    static bool prepareGraph(render::RenderGraph& graph)
    {
        using namespace render;
        const auto* deferred = graph.findNode("Deferred");
        const auto* gather = graph.findNode("Gather");
        const auto* fields = graph.findNode("GlobalDistanceFieldPages");
        if (!deferred || !gather || !fields) { return false; }
        auto deferredProperties = deferred->properties;
        deferredProperties["exportLumenPreviousPosition"] = true;
        auto gatherProperties = gather->properties;
        gatherProperties.update({{"finalGather", true}, {"irradianceFormat", "SH3"}, {"fastLightingUpdate", true},
            {"screenTraces", true}, {"activeProbeDispatch", true}, {"splitProbeTracing", true}, {"compactProbeAtlas", true},
            {"detailTracing", true}, {"detailDistanceFieldPaths", fields->properties.at("distanceFieldPaths")}});
        if (!graph.setNodeProperties(deferred->id, std::move(deferredProperties)) ||
            !graph.setNodeProperties(gather->id, std::move(gatherProperties)) ||
            !graph.addNode("LumenTemporalGatherPass", "Temporal", {{"dynamicReprojection", true}, {"fastLightingUpdate", true},
                {"maxFramesAccumulated", 6u}, {"publishScreenTraceHistory", true}}) ||
            !graph.addNode("LumenScreenTraceSetupPass", "ScreenTrace", {{"historySource", "composited"},
                {"historyProducer", "Temporal"}, {"dynamicReprojection", true}})) {
            return false;
        }
        for (const std::string_view field : {"closestHZB", "previousDepth", "previousColor", "traceView"}) {
            if (!graph.addEdge(makeRenderGraphFieldName("ScreenTrace", field), makeRenderGraphFieldName("Gather", field))) {
                return false;
            }
        }
        return graph.addEdge("VBuffer.depth", "ScreenTrace.depth") && graph.addEdge("VBuffer.depth", "Gather.detailDepth") &&
            graph.addEdge("Deferred.lumenPreviousPosition", "Gather.previousPosition") &&
            graph.addEdge("Deferred.lumenPreviousPosition", "Temporal.previousPosition") &&
            graph.addEdge("Deferred.lumenPosition", "Temporal.position") && graph.addEdge("Deferred.lumenNormal", "Temporal.normal") &&
            graph.addEdge("Deferred.color", "Temporal.directColor") && graph.addEdge("VBuffer.depth", "Temporal.sceneDepth") &&
            graph.addEdge("Gather.indirectDiffuse", "Temporal.currentIndirectDiffuse") && graph.markOutput("Temporal.color");
    }

    static std::string captureFrame(render::RenderGraph& graph, render::RenderGraphPreviewRenderer& preview,
        CardUpdateInspector& inspector, std::array<CardCapture, 4>& frame)
    {
        if (!preview.render(graph, kSingleBounceEdge, kSingleBounceEdge, "Temporal.color", false)) {
            return "Moving Gather render failed: " + preview.lastLog();
        }
        inspector.runtime.poll();
        auto& core = inspector.runtime.core();
        const auto& response = inspector.response;
        if (response.value("status", "") != "ok") { return response.dump(); }
        const auto& jobs = response.at("result").at("jobs");
        constexpr std::array counts{4u, 2u, 2u, 1u};
        for (size_t pass = 0; pass < frame.size(); ++pass) {
            const auto id = jobs.at(pass).at("job").get<std::string>();
            frame[pass] = core.completedCapture(id);
            if (!frame[pass]) { return core.dispatch({{"method", "jobs.get"}, {"params", {{"job", id}}}}).dump(); }
            if (frame[pass]->artifacts.size() != counts[pass] ||
                frame[pass]->snapshot.evidence.execution != frame[0]->snapshot.evidence.execution ||
                frame[pass]->snapshot.evidence.generation != frame[0]->snapshot.evidence.generation) {
                return "Missing same-frame Gather motion evidence";
            }
        }
        for (size_t field = 0; field < 2; ++field) {
            if (frame[1]->artifacts[field].bytes.size() != kAtlasPixels * sizeof(std::array<float, 4>) ||
                frame[2]->artifacts[field].bytes.size() != kAtlasPixels * sizeof(float)) {
                return "Unexpected screen receiver or Temporal motion readback layout";
            }
        }
        if (frame[3]->artifacts[0].bytes.size() != kTraceViewWords * sizeof(uint32_t)) {
            return "Unexpected Screen trace view readback layout";
        }
        return {};
    }

    static std::string addRayCounts(const debug::DebugCapture& frame, uint32_t& screenRays, uint32_t& screenMotion, uint32_t& detailMotion)
    {
        const auto& artifacts = frame.artifacts;
        if (artifacts[0].bytes.size() != 2u * sizeof(uint32_t) || artifacts[1].bytes.size() != kProbeSlots * sizeof(uint32_t) ||
            artifacts[2].bytes.size() != kTraceRays * sizeof(uint32_t) ||
            artifacts[3].bytes.size() != kTraceRays * sizeof(std::array<float, 4>)) {
            return "Unexpected probe motion readback layout";
        }
        const uint32_t active = readSurfaceValue<uint32_t>(artifacts[0].bytes, 0);
        if (active == 0 || active > kProbeSlots) { return "Invalid active probe count in moving blocker fixture"; }
        for (uint32_t work = 0; work < active; ++work) {
            const uint32_t slot = readSurfaceValue<uint32_t>(artifacts[1].bytes, work);
            if (slot >= kProbeSlots) { return "Invalid probe index in moving blocker fixture"; }
            for (uint32_t lane = 0; lane < kRaysPerSlot; ++lane) {
                const uint32_t ray = slot * kRaysPerSlot + lane;
                const uint32_t moving = readSurfaceValue<uint32_t>(artifacts[2].bytes, ray);
                const float source = readSurfaceValue<std::array<float, 4>>(artifacts[3].bytes, ray)[3];
                if (moving > 1u || (moving != 0u && source != 5.0f && source != 9.0f)) {
                    return "A probe ray reported motion without an accepted Screen or Detail hit";
                }
                screenRays += source == 5.0f;
                screenMotion += moving != 0u && source == 5.0f;
                detailMotion += moving != 0u && source == 9.0f;
            }
        }
        return {};
    }
};

METALLIC_REGISTER_RHI_TEST(LumenMovingGatherHistoryTest);

class LumenReflectionTest : public RHITest {
public:
    enum class Variant { Software, Classified, RoughSpecularTransition };

    explicit LumenReflectionTest(Variant variant = Variant::Software)
        : classified_(variant == Variant::Classified), roughSpecular_(variant == Variant::RoughSpecularTransition)
    {
        name = classified_ ? "lumen_classified_reflections" :
            roughSpecular_ ? "lumen_rough_specular_transition" : "lumen_software_reflections";
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
        if (!makeGraph(graph, field, cards, classified_, false, roughSpecular_)) {
            return RHITestResult::fail("Could not build the software reflection graph");
        }
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
        camera.orthoHeight = kOrthoHeight;
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
        if (roughSpecular_) {
            auto receiver = document.materials()[0];
            uint32_t mixedPixels = 0;
            constexpr std::array ramp{0.0f, 0.29f, 0.32f, 0.35f, 0.38f, 0.399f, 0.401f, 0.6f, 0.85f};
            for (float roughness : ramp) {
                if (receiver.roughnessFactor != roughness) {
                    receiver.roughnessFactor = roughness;
                    if (!document.setMaterialProperties(0, receiver)) {
                        return RHITestResult::fail(std::format("Could not set ramp roughness to {}", roughness));
                    }
                }
                view.cameraCut();
                const auto frame = captureFrame(inspector, executor, submission, false, log);
                if (!frame) { return RHITestResult::fail(log); }
                log = verifyRoughSpecularFrame(*frame, roughness, false, mixedPixels);
                if (!log.empty()) { return RHITestResult::fail(log); }
            }
            receiver.unlit = true;
            if (!document.setMaterialProperties(0, receiver)) { return RHITestResult::fail("Could not set unlit ramp material"); }
            const auto unlit = captureFrame(inspector, executor, submission, false, log);
            if (!unlit) { return RHITestResult::fail(log); }
            log = verifyRoughSpecularFrame(*unlit, receiver.roughnessFactor, true, mixedPixels);
            if (!log.empty()) { return RHITestResult::fail(log); }
            if (mixedPixels == 0u) { return RHITestResult::fail("Roughness ramp never exercised two positive specular sources"); }
            if (validation.failed()) { return RHITestResult::fail("Roughness ramp Vulkan validation: " + validation.snapshot().dump()); }
            return RHITestResult::pass("Rough and dedicated specular match the shared material-weighted blend across the roughness ramp.");
        }
        const auto runFrame = [&](float roughness, bool unlit, uint32_t minHits,
                                  ReflectionHistory expectedHistory = ReflectionHistory::Absent) {
            auto frame = captureFrame(inspector, executor, submission, classified_, log);
            if (frame) {
                log = verifyFrame(*frame, roughness, unlit, minHits, expectedHistory);
                if (log.empty() && classified_) {
                    ReflectionCoverage coverage;
                    log = verifyClassifiedFrame(*frame, false, coverage);
                }
            }
            return frame && log.empty();
        };

        if (!runFrame(0.0f, false, 144u)) { return RHITestResult::fail("Mirror: " + log); }
        auto receiver = document.materials()[0];
        receiver.roughnessFactor = kGlossyRoughness;
        if (!document.setMaterialProperties(0, receiver)) { return RHITestResult::fail("Could not set the rough receiver material"); }
        view.cameraCut();
        if (!runFrame(kGlossyRoughness, false, 64u)) { return RHITestResult::fail("Rough reflection: " + log); }
        if (!runFrame(kGlossyRoughness, false, 0u, ReflectionHistory::FirstVirtual)) {
            return RHITestResult::fail("First accepted reflection history: " + log);
        }
        if (!runFrame(kGlossyRoughness, false, 0u, ReflectionHistory::Accumulated)) {
            return RHITestResult::fail("Accumulated reflection history: " + log);
        }

        // Pan far enough to crop the receiver at the left edge, then reveal it again.
        auto panned = camera;
        panned.eye[0] += kPanDistance;
        panned.center[0] += kPanDistance;
        if (!view.setCamera(panned)) { return RHITestResult::fail("Could not pan the reflection camera"); }
        if (!runFrame(kGlossyRoughness, false, 0u, ReflectionHistory::Accumulated)) {
            return RHITestResult::fail("Reflection history after camera pan: " + log);
        }
        if (!view.setCamera(camera)) { return RHITestResult::fail("Could not restore the reflection camera"); }
        if (!runFrame(kGlossyRoughness, false, 0u, ReflectionHistory::Disoccluded)) {
            return RHITestResult::fail("Reflection history after revealing the receiver: " + log);
        }
        view.cameraCut();
        if (!runFrame(kGlossyRoughness, false, 0u)) {
            return RHITestResult::fail("Reflection history after camera cut: " + log);
        }

        // The dominant-direction factor drops below 0.5 at this roughness.
        receiver.roughnessFactor = kSurfaceHistoryRoughness;
        if (!document.setMaterialProperties(0, receiver)) {
            return RHITestResult::fail("Could not set the surface history fixture");
        }
        view.cameraCut();
        if (!runFrame(kSurfaceHistoryRoughness, false, 0u)) {
            return RHITestResult::fail("Surface reflection history seed: " + log);
        }
        if (!runFrame(kSurfaceHistoryRoughness, false, 0u, ReflectionHistory::FirstSurface)) {
            return RHITestResult::fail("Surface-depth reflection history: " + log);
        }

        receiver.roughnessFactor = 0.8f;
        if (!document.setMaterialProperties(0, receiver)) { return RHITestResult::fail("Could not set the roughness cutoff fixture"); }
        if (!runFrame(0.8f, false, 0u)) { return RHITestResult::fail("Roughness cutoff: " + log); }

        receiver.roughnessFactor = 0.0f;
        receiver.unlit = true;
        if (!document.setMaterialProperties(0, receiver)) { return RHITestResult::fail("Could not set the unlit receiver material"); }
        if (!runFrame(0.0f, true, 0u)) { return RHITestResult::fail("Unlit receiver: " + log); }
        if (classified_) {
            receiver.unlit = false;
            if (!document.setMaterialProperties(0, receiver)) { return RHITestResult::fail("Could not restore the mirror receiver"); }
            camera.eye = {0.0f, 0.3f, 3.0f};
            camera.center = {0.0f, 0.3f, 0.0f};
            camera.orthographic = false;
            camera.fovDegrees = 63.0f;
            if (!view.setCamera(camera)) { return RHITestResult::fail("Could not initialize the reflection handoff camera"); }
            view.cameraCut();
            RenderGraph handoffGraph;
            if (!makeGraph(handoffGraph, field, cards, true, true) ||
                !executor.compile(gpu, handoffGraph, kAtlasEdge, kAtlasEdge, RenderGraphCompileOptions{.enableDebugReadback = true}, log)) {
                return RHITestResult::fail("Reflection handoff graph compile: " + log);
            }
            ReflectionCoverage coverage;
            for (uint32_t frameIndex = 0; frameIndex < 4u; ++frameIndex) {
                auto frame = captureFrame(inspector, executor, submission, true, log);
                if (!frame) { return RHITestResult::fail(log); }
                log = verifyClassifiedFrame(*frame, true, coverage);
                if (!log.empty()) { return RHITestResult::fail("Screen and Mesh SDF handoff: " + log); }
            }
            if (coverage.detailRays == 0 || coverage.gdfRays == 0) {
                return RHITestResult::fail("Reflection handoff did not exercise both world-space ray queues");
            }
            spdlog::info("Reflection handoff: {} Screen hits, {} Mesh SDF hits, {} Mesh SDF rays, {} GDF rays",
                coverage.screenHits, coverage.detailHits, coverage.detailRays, coverage.gdfRays);
        }
        if (validation.failed()) { return RHITestResult::fail("Lumen reflection Vulkan validation: " + validation.snapshot().dump()); }
        return RHITestResult::pass("Resolve reused neighboring hits. Reflection history reprojected, and Bilateral filtered rough reflections before composite.");
    }

private:
    static constexpr float kMaxRoughness = 0.75f;
    static constexpr float kTransitionMaxRoughness = 0.4f;
    static constexpr float kGlossyRoughness = 0.2f;
    static constexpr float kSurfaceHistoryRoughness = 0.68f;
    static constexpr float kMaxTraceDistance = 4.0f;
    static constexpr float kOrthoHeight = 1.25f;
    static constexpr float kPanDistance = 0.12f;
    static constexpr uint32_t kReflectionTileEdge = 8u;
    static constexpr uint32_t kTilesPerAxis = kAtlasEdge / kReflectionTileEdge;
    static constexpr uint32_t kTileCount = kTilesPerAxis * kTilesPerAxis;
    static constexpr size_t kReflectionArtifactCount = 10;
    using ReflectionFrame = std::array<CardCapture, 3>;
    using Pixel = std::array<float, 4>;
    enum class ReflectionHistory { Absent, FirstVirtual, Accumulated, Disoccluded, FirstSurface };
    struct ReflectionCoverage {
        uint32_t screenHits = 0;
        uint32_t detailHits = 0;
        uint32_t detailRays = 0;
        uint32_t gdfRays = 0;
    };

    static bool makeGraph(render::RenderGraph& graph, const std::filesystem::path& field, const std::filesystem::path& cards,
        bool classified, bool traceHandoff, bool roughSpecular = false)
    {
        using render::makeRenderGraphFieldName;
        const auto edge = [&](std::string_view from, std::string_view source, std::string_view to, std::string_view target) {
            return graph.addEdge(makeRenderGraphFieldName(from, source), makeRenderGraphFieldName(to, target));
        };
        if (!graph.addNode("VisibilityBufferPass", "VBuffer", {{"visualization", "none"}, {"enableClusterRtx", false}}) ||
            !graph.addNode("VisibilityBufferDeferredPass", "Deferred", {{"materialBinning", false}, {"outputLinear", true},
                {"exportLumenReceivers", true}, {"exportLumenSpecular", true}, {"exportLumenPreviousPosition", true}}) ||
            !graph.addNode("GlobalDistanceFieldPagesPass", "GDF", {{"distanceFieldPaths", {field.generic_string()}},
                {"center", {0.0f, 0.5f, 0.75f}}, {"extent", 2.5f}, {"resolution", 112}, {"maxPhysicalPages", 1728},
                {"quantized", true}, {"buildObjectGrid", true}, {"persistentPages", true}}) ||
            !graph.addNode("LumenSurfaceCachePass", "SurfaceCache", {{"meshCardPaths", {cards.generic_string()}},
                {"atlasSize", kAtlasEdge}, {"tileResolution", kTileEdge}, {"persistentCards", true}})) {
            return false;
        }
        for (const auto* name : {"visibility", "depth", "rasterInfo"}) {
            if (!edge("VBuffer", name, "Deferred", name)) { return false; }
        }
        if (traceHandoff && (!graph.addNode("LumenScreenTraceSetupPass", "ScreenTrace") ||
            !edge("VBuffer", "depth", "ScreenTrace", "depth") || !edge("Deferred", "color", "ScreenTrace", "color"))) {
            return false;
        }
        if (roughSpecular && (!LumenTemporalInputPass::ensureRegistered() ||
            !graph.addNode("LumenTemporalInputPass", "RoughSpecular"))) {
            return false;
        }
        const auto addReflection = [&](std::string_view name, bool classify) {
            render::RenderGraphProperties properties{{"maxTraceDistance", kMaxTraceDistance}, {"maxRoughness", kMaxRoughness},
                {"dynamicReprojection", true}, {"classifiedTracing", classify}};
            if (roughSpecular) {
                properties.update({{"maxRoughness", kTransitionMaxRoughness}, {"roughSpecular", true}});
            }
            if (traceHandoff) {
                properties.update({{"screenTraces", true}, {"detailTracing", true}, {"meshSDFTraceDistance", 3.0f},
                    {"detailDistanceFieldPaths", {field.generic_string()}}});
            }
            if (!graph.addNode("LumenReflectionPass", std::string(name), std::move(properties))) { return false; }
            if (roughSpecular && !edge("RoughSpecular", "indirectDiffuse", name, "roughSpecular")) { return false; }
            for (const auto* fieldName : {"pageTable", "pageAtlas", "coarseMip", "domain", "objectGrid", "objectGridScene"}) {
                if (!edge("GDF", fieldName, name, fieldName)) { return false; }
            }
            constexpr std::array surfaceBindings{std::pair{"cards", "cards"}, std::pair{"scene", "cardScene"},
                std::pair{"depth", "cardDepth"}, std::pair{"lighting", "cardLighting"}};
            for (const auto& [from, to] : surfaceBindings) {
                if (!edge("SurfaceCache", from, name, to)) { return false; }
            }
            if (traceHandoff) {
                for (const auto* fieldName : {"closestHZB", "previousDepth", "previousColor", "traceView"}) {
                    if (!edge("ScreenTrace", fieldName, name, fieldName)) { return false; }
                }
                if (!edge("VBuffer", "depth", name, "detailDepth")) { return false; }
            }
            return edge("Deferred", "lumenPosition", name, "position") &&
                edge("Deferred", "lumenNormal", name, "normal") &&
                edge("Deferred", "lumenSpecular", name, "specular") &&
                edge("Deferred", "lumenPreviousPosition", name, "previousPosition") &&
                edge("Deferred", "color", name, "sceneColor") && graph.markOutput(makeRenderGraphFieldName(name, "color"));
        };
        return addReflection("Reflections", classified) && (!classified || addReflection("FusedReflections", false));
    }

    static std::optional<ReflectionFrame> captureFrame(CardUpdateInspector& inspector, render::RenderGraphExecutor& executor,
        const render::RenderGraphSubmitDesc& submission, bool classified, std::string& log)
    {
        auto batches = debug::DebugValue::array({
            {{"pass", "Deferred"}, {"resources", debug::DebugValue::array({cardAtlasCapture("Deferred.lumenPosition"),
                cardAtlasCapture("Deferred.lumenSpecular"), cardAtlasCapture("Deferred.color")})}},
            {{"pass", "Reflections"}, {"resources", debug::DebugValue::array({cardAtlasCapture("Reflections.reflections"),
                cardAtlasCapture("Reflections.trace"), cardAtlasCapture("Reflections.rayData"),
                cardAtlasCapture("Reflections.rayHitDistance"), cardAtlasCapture("Reflections.resolvedReflections"),
                cardAtlasCapture("Reflections.resolvedHitDistance"), cardAtlasCapture("Reflections.temporalReflections"),
                cardAtlasCapture("Reflections.temporalState"), cardAtlasCapture("Reflections.bilateralReflections"),
                cardAtlasCapture("Reflections.color")})}},
        });
        if (classified) {
            auto& schedule = batches[1]["resources"];
            schedule.push_back(cardBufferCapture("Reflections.reflectionWorkCounts", 4u));
            schedule.push_back(cardBufferCapture("Reflections.reflectionActiveTiles", kTileCount));
            schedule.push_back(cardBufferCapture("Reflections.reflectionClearTiles", kTileCount));
            schedule.push_back(cardBufferCapture("Reflections.reflectionDispatchArgs", 15u));
            schedule.push_back(cardBufferCapture("Reflections.reflectionRayList", kAtlasPixels));
            batches.push_back({{"pass", "FusedReflections"}, {"resources", debug::DebugValue::array({
                cardAtlasCapture("FusedReflections.reflections"), cardAtlasCapture("FusedReflections.trace"),
                cardAtlasCapture("FusedReflections.rayData"), cardAtlasCapture("FusedReflections.rayHitDistance")})}});
        }
        inspector.request = {{"method", "capture.batch"}, {"params", {{"batches", std::move(batches)}}}};
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
        for (size_t index = 0; index < (classified ? captures.size() : captures.size() - 1u); ++index) {
            const auto id = jobs.at(index).at("job").get<std::string>();
            captures[index] = core.completedCapture(id);
            if (!captures[index]) {
                log = core.dispatch({{"method", "jobs.get"}, {"params", {{"job", id}}}}).dump();
                return std::nullopt;
            }
        }
        return captures;
    }

    bool classified_;
    bool roughSpecular_;

    static Pixel readHalfPixel(const debug::DebugArtifact& artifact, size_t pixel)
    {
        const auto halfValues = readSurfaceValue<std::array<uint16_t, 4>>(artifact.bytes, pixel);
        return {halfToFloat(halfValues[0]), halfToFloat(halfValues[1]), halfToFloat(halfValues[2]), halfToFloat(halfValues[3])};
    }

    static std::string verifyRoughSpecularFrame(const ReflectionFrame& frame, float roughness, bool unlit, uint32_t& mixedPixels)
    {
        const auto& deferred = frame[0]->artifacts;
        const auto& reflections = frame[1]->artifacts;
        if (deferred.size() != 3 || reflections.size() < kReflectionArtifactCount ||
            deferred[0].bytes.size() != kAtlasPixels * sizeof(Pixel) ||
            deferred[1].bytes.size() != kAtlasPixels * sizeof(std::array<uint16_t, 4>) ||
            deferred[2].bytes.size() != kAtlasPixels * sizeof(Pixel) ||
            reflections[8].bytes.size() != kAtlasPixels * sizeof(std::array<uint16_t, 4>) ||
            reflections[9].bytes.size() != kAtlasPixels * sizeof(Pixel)) {
            return "Unexpected rough specular ramp readback layout";
        }
        uint32_t receivers = 0;
        for (size_t pixel = 0; pixel < kAtlasPixels; ++pixel) {
            const auto position = readSurfaceValue<Pixel>(deferred[0].bytes, pixel);
            const auto specular = readHalfPixel(deferred[1], pixel);
            const auto base = readSurfaceValue<Pixel>(deferred[2].bytes, pixel);
            const auto dedicated = readHalfPixel(reflections[8], pixel);
            const auto combined = readSurfaceValue<Pixel>(reflections[9].bytes, pixel);
            const bool receiver = position[3] > 0.0f && (specular[0] > 0.0f || specular[1] > 0.0f || specular[2] > 0.0f);
            const size_t x = pixel % kAtlasEdge;
            const size_t y = pixel / kAtlasEdge;
            if (unlit && receiver && x >= 52 && x < 76 && y >= 52 && y < 76) {
                return "Unlit ramp receiver received indirect specular";
            }
            const float alpha = std::clamp((kTransitionMaxRoughness - specular[3]) * 10.0f, 0.0f, 1.0f);
            const bool rampReceiver = receiver && std::abs(roughness - specular[3]) < 0.001f;
            receivers += rampReceiver;
            mixedPixels += rampReceiver && alpha > 0.0f && alpha < 1.0f && dedicated[3] > 0.0f && dedicated[0] > 0.0f;
            if (base[3] != combined[3]) { return "Roughness transition changed scene alpha"; }
            for (size_t channel = 0; channel < 3; ++channel) {
                float expected = base[channel];
                if (receiver) {
                    const float rough = (1.0f - alpha) * LumenTemporalInputPass::kConstantLighting[channel];
                    const float glossy = alpha > 0.0f && dedicated[3] > 0.0f ? alpha * dedicated[channel] : 0.0f;
                    expected += (rough + glossy) * specular[channel];
                }
                if (!(std::abs(combined[channel] - expected) <= std::max(0.000025f, std::abs(expected) * 0.0002f))) {
                    return std::format("Roughness {} pixel {} channel {} did not match the complementary fade, got {}, expected {}",
                        roughness, pixel, channel, combined[channel], expected);
                }
            }
        }
        return !unlit && receivers < 144u ? "Roughness ramp did not render enough receiver pixels" : std::string{};
    }

    static std::string verifyClassifiedFrame(const ReflectionFrame& frame, bool traceHandoff, ReflectionCoverage& coverage)
    {
        const auto& staged = frame[1]->artifacts;
        const auto& fused = frame[2]->artifacts;
        constexpr std::array imageStrides{8u, 16u, 8u, 2u};
        constexpr std::array bufferSizes{4u, kTileCount, kTileCount, 15u, uint32_t(kAtlasPixels)};
        if (staged.size() != kReflectionArtifactCount + bufferSizes.size() || fused.size() != imageStrides.size()) {
            return "Missing classified reflection comparison artifacts";
        }
        for (size_t image = 0; image < imageStrides.size(); ++image) {
            if (staged[image].bytes.size() != kAtlasPixels * imageStrides[image] ||
                fused[image].bytes.size() != staged[image].bytes.size()) {
                return "Unexpected classified reflection image readback layout";
            }
        }
        for (size_t buffer = 0; buffer < bufferSizes.size(); ++buffer) {
            if (staged[kReflectionArtifactCount + buffer].bytes.size() != bufferSizes[buffer] * sizeof(uint32_t)) {
                return "Unexpected classified reflection work buffer layout";
            }
        }
        const auto& counterBytes = staged[kReflectionArtifactCount].bytes;
        const auto [activeTiles, clearTiles, detailRays, gdfRays] = readSurfaceValue<std::array<uint32_t, 4>>(counterBytes, 0);
        if (activeTiles > kTileCount || clearTiles != kTileCount - activeTiles || gdfRays > kAtlasPixels ||
            detailRays > kAtlasPixels || (!traceHandoff && detailRays != 0)) {
            return "Classified reflection counters exceed capacity or lost a tile";
        }
        constexpr float kScreenHit = 5.0f;
        constexpr float kMeshSDFHit = 9.0f;
        std::array<bool, kTileCount> tileHasRay{};
        std::array<bool, kAtlasPixels> needsGDF{};
        uint32_t screenHits = 0, detailHits = 0, expectedGDFRays = 0;
        for (size_t pixel = 0; pixel < kAtlasPixels; ++pixel) {
            const auto stagedTrace = readSurfaceValue<Pixel>(staged[1].bytes, pixel);
            const auto fusedTrace = readSurfaceValue<Pixel>(fused[1].bytes, pixel);
            const auto stagedRay = readSurfaceValue<std::array<uint16_t, 4>>(staged[2].bytes, pixel);
            const auto fusedRay = readSurfaceValue<std::array<uint16_t, 4>>(fused[2].bytes, pixel);
            const auto stagedDistance = readSurfaceValue<uint16_t>(staged[3].bytes, pixel);
            const auto fusedDistance = readSurfaceValue<uint16_t>(fused[3].bytes, pixel);
            // Separate trace kernels can round a world-space hit to an adjacent R16F
            // centimeter code. Keep the hit sign and the untraced sentinel exact.
            const bool distancesMatch = stagedDistance == fusedDistance ||
                (traceHandoff && stagedRay[3] != 0 && ((stagedDistance ^ fusedDistance) & 0x8000u) == 0u &&
                    std::abs(int(stagedDistance) - int(fusedDistance)) == 1);
            if (stagedRay != fusedRay || !distancesMatch ||
                !std::equal(stagedTrace.begin() + 1, stagedTrace.end(), fusedTrace.begin() + 1) ||
                !(std::abs(stagedTrace[0] - fusedTrace[0]) <= 0.001f)) {
                return std::format("Classified reflection trace differs from fused at pixel {} (class {} vs {}, "
                    "distance {} vs {} m, packed 0x{:04x} vs 0x{:04x}, ray equal {})", pixel, stagedTrace[1], fusedTrace[1],
                    stagedTrace[0], fusedTrace[0], stagedDistance, fusedDistance, stagedRay == fusedRay);
            }
            const auto stagedLighting = readHalfPixel(staged[0], pixel);
            const auto fusedLighting = readHalfPixel(fused[0], pixel);
            if (!std::ranges::equal(stagedLighting, fusedLighting, [](float a, float b) {
                return std::abs(a - b) <= std::max(0.0001f, std::abs(b) * 0.001f);
            })) {
                return std::format("Classified reflection radiance differs from fused at pixel {} (class {}, "
                    "RGB [{}, {}, {}] vs [{}, {}, {}])", pixel, stagedTrace[1], stagedLighting[0], stagedLighting[1],
                    stagedLighting[2], fusedLighting[0], fusedLighting[1], fusedLighting[2]);
            }
            if (stagedRay[3] == 0) { continue; }
            const auto tile = (pixel / kAtlasEdge / kReflectionTileEdge) * kTilesPerAxis +
                pixel % kAtlasEdge / kReflectionTileEdge;
            tileHasRay[tile] = true;
            screenHits += stagedTrace[1] == kScreenHit;
            detailHits += stagedTrace[1] == kMeshSDFHit;
            needsGDF[pixel] = stagedTrace[1] != kScreenHit && stagedTrace[1] != kMeshSDFHit;
            expectedGDFRays += needsGDF[pixel];
        }
        if (detailHits > detailRays || detailRays > expectedGDFRays + detailHits || gdfRays != expectedGDFRays) {
            return std::format("Invalid reflection handoff counts ({} Screen hits, {} Detail hits, {} Detail rays, {} GDF rays, "
                "{} expected GDF rays)", screenHits, detailHits, detailRays, gdfRays, expectedGDFRays);
        }
        std::array<bool, kTileCount> visitedTiles{};
        const auto checkTiles = [&](size_t buffer, uint32_t count, bool active) {
            for (uint32_t index = 0; index < count; ++index) {
                const uint32_t packed = readSurfaceValue<uint32_t>(staged[buffer].bytes, index);
                const uint32_t x = packed & 0xfffu, y = packed >> 12u;
                if (x >= kTilesPerAxis || y >= kTilesPerAxis) { return false; }
                const uint32_t tile = y * kTilesPerAxis + x;
                if (visitedTiles[tile] || tileHasRay[tile] != active) { return false; }
                visitedTiles[tile] = true;
            }
            return true;
        };
        if (!checkTiles(kReflectionArtifactCount + 1u, activeTiles, true) ||
            !checkTiles(kReflectionArtifactCount + 2u, clearTiles, false)) {
            return "Active and clear reflection tile lists did not partition the view";
        }
        const auto& args = staged[kReflectionArtifactCount + 3u].bytes;
        const auto checkArgs = [&](size_t index, uint32_t expectedX) {
            const auto [x, y, z] = readSurfaceValue<std::array<uint32_t, 3>>(args, index);
            return x == expectedX && z == 1u && (y == 1u || (x == 0 && y == 0));
        };
        if (!checkArgs(0, activeTiles) || !checkArgs(1, clearTiles) ||
            !(checkArgs(2, activeTiles) || checkArgs(2, (activeTiles + 15u) / 16u)) ||
            (traceHandoff && !checkArgs(3, (detailRays + 63u) / 64u)) || !checkArgs(4, (gdfRays + 63u) / 64u)) {
            return "Indirect reflection arguments did not match their compacted work counts";
        }
        const auto& compacted = staged[kReflectionArtifactCount + 4u].bytes;
        for (uint32_t index = 0; index < gdfRays; ++index) {
            const uint32_t packed = readSurfaceValue<uint32_t>(compacted, index);
            const uint32_t x = packed & 0x3fffu, y = packed >> 14u;
            if (x >= kAtlasEdge || y >= kAtlasEdge || !needsGDF[y * kAtlasEdge + x]) {
                return "GDF reflection queue contained a duplicate ray or a ray completed by an earlier stage";
            }
            needsGDF[y * kAtlasEdge + x] = false;
        }
        coverage.screenHits += screenHits;
        coverage.detailHits += detailHits;
        coverage.detailRays += detailRays;
        coverage.gdfRays += gdfRays;
        return {};
    }

    static bool withinBilateralNeighborhood(const debug::DebugArtifact& input, uint32_t x, uint32_t y,
        const Pixel& temporal, const Pixel& bilateral)
    {
        constexpr uint32_t kRadius = 8;
        Pixel minimum = temporal;
        Pixel maximum = temporal;
        for (uint32_t tapY = y - kRadius; tapY <= y + kRadius; ++tapY) {
            for (uint32_t tapX = x - kRadius; tapX <= x + kRadius; ++tapX) {
                const Pixel tap = readHalfPixel(input, size_t(tapY) * kAtlasEdge + tapX);
                if (tap[3] <= 0.0f) { continue; }
                for (size_t channel = 0; channel < 3; ++channel) {
                    minimum[channel] = std::min(minimum[channel], tap[channel]);
                    maximum[channel] = std::max(maximum[channel], tap[channel]);
                }
            }
        }
        // The reversible luminance tonemap rescales positive sample weights and preserves their convex hull.
        for (size_t channel = 0; channel < 3; ++channel) {
            const float tolerance = std::max(0.0001f, maximum[channel] * 0.001f);
            if (bilateral[channel] < minimum[channel] - tolerance || bilateral[channel] > maximum[channel] + tolerance) {
                return false;
            }
        }
        return true;
    }

    static std::string verifyFrame(const ReflectionFrame& frame, float roughness, bool unlit, uint32_t minHits,
        ReflectionHistory expectedHistory)
    {
        const auto& deferred = frame[0]->artifacts;
        const auto& reflections = frame[1]->artifacts;
        const auto matchesLayout = [](const auto& artifacts, const auto& strides) {
            return artifacts.size() >= strides.size() && std::equal(strides.begin(), strides.end(), artifacts.begin(),
                [](size_t stride, const auto& artifact) {
                    return artifact.bytes.size() == kAtlasPixels * stride;
                });
        };
        if (!matchesLayout(deferred, std::array{16u, 8u, 16u}) ||
            !matchesLayout(reflections, std::array{8u, 16u, 8u, 2u, 8u, 2u, 8u, 4u, 8u, 16u})) {
            return "Unexpected reflection readback layout";
        }
        constexpr Pixel untraced{-1.0f, 8.0f, -1.0f, 0.0f};
        constexpr Pixel invalidResolve{65024.0f, 65024.0f, 64512.0f, 0.0f};
        constexpr uint16_t halfMaxFinite = 0x7bffu;
        constexpr uint16_t halfPositiveInfinity = 0x7c00u;
        constexpr float maxHitDistance = kMaxTraceDistance * 100.0f;
        const bool firstHistory = expectedHistory == ReflectionHistory::FirstVirtual ||
            expectedHistory == ReflectionHistory::FirstSurface;
        const float expectedMethod = expectedHistory == ReflectionHistory::FirstSurface ? 2.0f : 1.0f;
        const bool checkFreshBilateral = expectedHistory == ReflectionHistory::Absent && roughness == kGlossyRoughness;
        uint32_t hits = 0, background = 0, reconstructedMisses = 0;
        uint32_t acceptedHistoryPixels = 0, blendedPixels = 0, revealedPixels = 0, filteredPixels = 0;
        for (uint32_t y = 0; y < kAtlasEdge; ++y) {
            for (uint32_t x = 0; x < kAtlasEdge; ++x) {
                const size_t pixel = size_t(y) * kAtlasEdge + x;
                const auto position = readSurfaceValue<Pixel>(deferred[0].bytes, pixel);
                const auto specular = readHalfPixel(deferred[1], pixel);
                const auto baseColor = readSurfaceValue<Pixel>(deferred[2].bytes, pixel);
                const auto raw = readHalfPixel(reflections[0], pixel);
                const auto trace = readSurfaceValue<Pixel>(reflections[1].bytes, pixel);
                const auto ray = readHalfPixel(reflections[2], pixel);
                const auto rawDistanceBits = readSurfaceValue<uint16_t>(reflections[3].bytes, pixel);
                const auto resolved = readHalfPixel(reflections[4], pixel);
                const auto resolvedDistanceBits = readSurfaceValue<uint16_t>(reflections[5].bytes, pixel);
                const auto temporal = readHalfPixel(reflections[6], pixel);
                const auto state = readSurfaceValue<std::array<uint16_t, 2>>(reflections[7].bytes, pixel);
                const float age = halfToFloat(state[0]);
                const float method = halfToFloat(state[1]);
                const auto bilateral = readHalfPixel(reflections[8], pixel);
                const auto color = readSurfaceValue<Pixel>(reflections[9].bytes, pixel);
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
                if (!traced) {
                    if (raw != Pixel{} || trace != untraced || ray != Pixel{} || rawDistanceBits != 0) {
                        return "Invalid, unlit or roughness-filtered pixel produced a reflection ray";
                    }
                    // Resolve writes FLT_MAX, which overflows the R16F image conversion.
                    if (resolved != invalidResolve ||
                        (resolvedDistanceBits != halfMaxFinite && resolvedDistanceBits != halfPositiveInfinity)) {
                        return "Untraced pixel did not write the invalid Resolve values at " + std::to_string(pixel) +
                            " (RGB " + std::to_string(resolved[0]) + ", " + std::to_string(resolved[1]) + ", " +
                            std::to_string(resolved[2]) + ", fade " + std::to_string(resolved[3]) +
                            ", half distance bits " + std::to_string(resolvedDistanceBits) + ")";
                    }
                    if (temporal != invalidResolve || age != 0.0f || method != 0.0f) {
                        return "Untraced pixel did not write the invalid reflection history values";
                    }
                    if (bilateral != invalidResolve) { return "Bilateral changed the invalid reflection sentinel"; }
                } else {
                    const float fade = std::clamp((kMaxRoughness - specular[3]) * 10.0f, 0.0f, 1.0f);
                    if (!(std::abs(raw[3] - fade) < 0.001f) || raw[3] != resolved[3]) {
                        return "Incorrect roughness fade";
                    }
                    const float directionLengthSq = ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2];
                    if (!(std::abs(directionLengthSq - 1.0f) < 0.002f && ray[3] > 0.0f && std::isfinite(ray[3]))) {
                        return "Trace did not preserve a unit ray direction and positive reciprocal PDF";
                    }
                    const float rawDistance = halfToFloat(rawDistanceBits);
                    const float resolvedDistance = halfToFloat(resolvedDistanceBits);
                    const bool hit = trace[1] == 2.0f;
                    if (hit) {
                        if (!std::signbit(rawDistance) ||
                            !(std::abs(-rawDistance - trace[0] * 100.0f) <= 0.1251f)) {
                            return "Packed ray distance did not match the signed centimeter hit";
                        }
                    } else if (rawDistance != maxHitDistance || raw[0] != 0.0f || raw[1] != 0.0f || raw[2] != 0.0f) {
                        return "A valid miss did not retain its maximum ray distance and black radiance";
                    }
                    if (!(resolvedDistance >= 0.0f && resolvedDistance <= std::abs(rawDistance))) {
                        return "Resolve distance exceeded the center ray distance";
                    }
                    if (center && trace[0] > 0.5f && trace[0] < 3.0f && hit && trace[2] == 0.0f &&
                        trace[3] == 3.0f && raw[0] > 1.0f && raw[1] > 0.0f && raw[2] > 0.0f) {
                        ++hits;
                    }
                    if (roughness == 0.0f && resolvedDistance != std::abs(rawDistance)) {
                        return "Mirror Resolve changed the center hit distance";
                    }
                    if (!hit && resolvedDistance < maxHitDistance && resolved[0] > 0.01f) { ++reconstructedMisses; }
                    const bool validState = (age == 0.0f && method == 0.0f) ||
                        (age >= 1.0f && age <= 12.0f && (method == 1.0f || method == 2.0f));
                    if (!validState || temporal[3] != resolved[3]) { return "Invalid reflection history state or fade"; }
                    if (bilateral[3] != temporal[3]) { return "Bilateral changed the roughness fade"; }
                    if (expectedHistory == ReflectionHistory::Absent && age != 0.0f) {
                        return "Reflection history survived a first frame or camera cut";
                    }
                    // The reference's half-code R8 bias can round zero to one UNORM step.
                    constexpr float kFirstHistoryAgeLimit = 1.0f + 32.0f / 255.0f + 0.0005f;
                    if (firstHistory && age > kFirstHistoryAgeLimit) {
                        return "New reflection history advanced beyond the seed and R8 quantization";
                    }
                    if (center && method == expectedMethod) { ++acceptedHistoryPixels; }
                    constexpr float kPreviousLeftEdge = kPanDistance - kOrthoHeight * 0.5f;
                    constexpr float kEdgeMargin = kOrthoHeight / kAtlasEdge;
                    if (expectedHistory == ReflectionHistory::Disoccluded && position[0] < kPreviousLeftEdge - kEdgeMargin) {
                        ++revealedPixels;
                        if (age != 0.0f) { return "Newly revealed receiver reused off-screen reflection history"; }
                    }
                }
                bool blended = false, filtered = false;
                for (size_t channel = 0; channel < 3; ++channel) {
                    if (traced && roughness == 0.0f &&
                        !(std::abs(resolved[channel] - raw[channel]) <= std::max(0.0001f, raw[channel] * 0.001f))) {
                        return "Mirror Resolve changed the center radiance";
                    }
                    if (!(temporal[channel] >= 0.0f && std::isfinite(temporal[channel]))) {
                        return "Temporal produced invalid radiance";
                    }
                    const float temporalDifference = std::abs(temporal[channel] - resolved[channel]);
                    const float colorTolerance = std::max(0.0001f, resolved[channel] * 0.001f);
                    if (traced && age <= 1.0f && !(temporalDifference <= colorTolerance)) {
                        return "Temporal modified radiance before reflection history acquired weight";
                    }
                    blended |= age > 1.0f && temporalDifference > colorTolerance;
                    if (!(bilateral[channel] >= 0.0f && std::isfinite(bilateral[channel]))) {
                        return "Bilateral produced invalid radiance";
                    }
                    const float filterDifference = std::abs(bilateral[channel] - temporal[channel]);
                    const float filterTolerance = std::max(0.0001f, temporal[channel] * 0.001f);
                    if (traced && roughness == 0.0f && filterDifference > filterTolerance) {
                        return "Bilateral changed a mirror reflection";
                    }
                    filtered |= filterDifference > filterTolerance;
                    const float expected = baseColor[channel] +
                        std::max(bilateral[channel] * bilateral[3] * specular[channel], 0.0f);
                    if (!(std::abs(color[channel] - expected) <= 0.0001f)) {
                        return "Reflection composite differs from scene color plus weighted bilateral radiance";
                    }
                }
                if (color[3] != baseColor[3]) { return "Reflection composite changed scene alpha"; }
                if (center && blended) { ++blendedPixels; }
                if (center && traced && checkFreshBilateral) {
                    if (!withinBilateralNeighborhood(reflections[6], x, y, temporal, bilateral)) {
                        return "Bilateral radiance fell outside its eight-pixel neighborhood";
                    }
                    if (filtered) { ++filteredPixels; }
                }
            }
        }
        if (background < 64) { return "The reflection fixture did not contain uncovered pixels"; }
        if (hits < minHits) { return "Too few reflected off-screen emitter pixels: " + std::to_string(hits); }
        if (roughness == kGlossyRoughness && reconstructedMisses == 0) {
            return "Rough Resolve did not reconstruct radiance and distance for any valid miss from neighboring hits";
        }
        if (expectedHistory != ReflectionHistory::Absent && acceptedHistoryPixels < 64u) {
            return "Too few receiver pixels retained the expected reflection history method: " +
                std::to_string(acceptedHistoryPixels);
        }
        if ((expectedHistory == ReflectionHistory::Accumulated || expectedHistory == ReflectionHistory::Disoccluded) &&
            blendedPixels < 16u) {
            return "Too few receiver pixels blended with accumulated reflection history: " + std::to_string(blendedPixels);
        }
        if (expectedHistory == ReflectionHistory::Disoccluded && revealedPixels < 16u) {
            return "Fixture did not reveal enough reflective pixels after panning";
        }
        if (checkFreshBilateral && filteredPixels < 16u) {
            return "Too few freshly traced receiver pixels were filtered by Bilateral: " + std::to_string(filteredPixels);
        }
        return {};
    }
};

METALLIC_REGISTER_RHI_TEST(LumenReflectionTest);

class LumenClassifiedReflectionTest final : public LumenReflectionTest {
public:
    LumenClassifiedReflectionTest() : LumenReflectionTest(Variant::Classified) {}
};

METALLIC_REGISTER_RHI_TEST(LumenClassifiedReflectionTest);

class LumenRoughSpecularTransitionTest final : public LumenReflectionTest {
public:
    LumenRoughSpecularTransitionTest() : LumenReflectionTest(Variant::RoughSpecularTransition) {}
};

METALLIC_REGISTER_RHI_TEST(LumenRoughSpecularTransitionTest);

} // namespace
} // namespace metallic::tests
