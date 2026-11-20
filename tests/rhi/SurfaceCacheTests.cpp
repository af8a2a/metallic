#include "RHITest.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/MeshCards.h"
#include "Runtime/Scene/MeshDistanceField.h"
#include "Runtime/Scene/SceneDocument.h"
#include "harness/ValidationRecorder.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>

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

bool prepareSurfaceFixture(const std::filesystem::path& fixture, const std::filesystem::path& cardPath,
    scene::SceneDocument& document, std::string& log)
{
    const std::filesystem::path root(PROJECT_SOURCE_DIR);
    std::ifstream input(root / "Asset/MeshDistanceField/BoxInstances.gltf");
    if (!input) { log = "Cannot open the Surface Cache box fixture"; return false; }
    auto gltf = nlohmann::json::parse(input);
    gltf["extensionsUsed"] = {"KHR_materials_specular"};
    auto& materials = gltf["materials"] = nlohmann::json::array();
    for (const auto& [albedo, emissive] : std::array{std::array{0.25f, 0.125f}, std::array{0.5625f, 0.5f}}) {
        materials.push_back({
            {"pbrMetallicRoughness", {{"baseColorFactor", {albedo, albedo, albedo, 1.0f}}, {"metallicFactor", 0.0f}}},
            {"extensions", {{"KHR_materials_specular", {{"specularFactor", 0.0f}}}}},
            {"emissiveFactor", {emissive, emissive, emissive}},
        });
    }
    // Source index order is -Z, +Z, -X, +X, -Y, +Y. Give +Z its own material.
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
            {"material", face == 1 ? 1 : 0}});
        accessors.push_back(std::move(indices));
    }
    std::ofstream output(fixture, std::ios::binary);
    output << gltf;
    output.close();
    if (!output) { log = "Cannot write the Surface Cache material fixture"; return false; }
    if (!document.load(fixture)) { log = document.lastLoadResult().error; return false; }

    scene::MeshDistanceFieldGeometry geometry;
    if (!scene::getMeshDistanceFieldGeometry(document, 0, geometry, log)) { return false; }
    scene::MeshCards field;
    field.sourceFingerprint = geometry.fingerprint;
    field.sourceMeshIndex = 0;
    // The six UE 5.7.4 cube OBBs captured directly from GenerateCardRepresentationData.
    for (uint32_t direction = 0; direction < kCardsPerBox; ++direction) {
        const uint32_t axis = direction / 2;
        const float sign = direction % 2 == 0 ? -1.0f : 1.0f;
        scene::MeshCardBuildRecord card{};
        card.origin[axis] = sign * 0.45f;
        card.axisX[axis == 0 ? 1 : 0] = 1.0f;
        card.axisY[axis == 2 ? 1 : 2] = 1.0f;
        card.axisZ[axis] = sign;
        card.extent = {0.5f, 0.5f, 0.1f};
        card.axisAlignedDirectionIndex = direction;
        field.cards.push_back(card);
    }
    return scene::saveMeshCards(cardPath, field, log);
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
        RenderGraphPreviewRenderer preview;
        RenderDebugRuntime inspector({}, RenderDebugDetail::Resources);
        struct DetachDebugObserver {
            RenderGraphPreviewRenderer& preview;
            ~DetachDebugObserver() { preview.setDebugObserver(nullptr); }
        } detach{preview};
        preview.setDebugObserver(&inspector);
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
        const auto request = inspector.core().dispatch({{"method", "capture.batch"}, {"params", {
            {"pass", "SurfaceCache"}, {"checkpoint", "AfterPass"}, {"resources", std::move(resources)}}}});
        if (request.value("status", "") != "ok") { return RHITestResult::fail(request.dump()); }
        if (!preview.render(graph, kAtlasEdge, kAtlasEdge, "SurfaceCache.depth")) {
            return RHITestResult::fail("Surface Cache encoded depth readback: " + preview.lastLog());
        }
        inspector.poll();
        const auto job = request.at("result").at("job").get<std::string>();
        const auto capture = inspector.core().completedCapture(job);
        if (!capture) {
            const auto state = inspector.core().dispatch({{"method", "jobs.get"}, {"params", {{"job", job}}}});
            return RHITestResult::fail("Surface Cache capture did not complete: " + state.dump());
        }
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

} // namespace
} // namespace metallic::tests
