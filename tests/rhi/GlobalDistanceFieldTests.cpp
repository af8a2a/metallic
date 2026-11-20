#include "MeshDistanceFieldFixture.h"
#include "RHITest.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>

namespace metallic::tests {
namespace {

constexpr uint32_t kWidth = 193;
constexpr uint32_t kHeight = 129;
constexpr uint32_t kResolution = 64;
constexpr uint32_t kTexels = kResolution + 1;
constexpr float kExtent = 1.5f;
constexpr float kVoxelSize = 2.0f * kExtent / float(kResolution);
constexpr float kInfluence = 4.0f * kVoxelSize;
constexpr float kCameraDistance = 4.0f;
constexpr float kNearPlane = 0.05f;
constexpr float kOrthoHeight = 4.8f;
constexpr std::string_view kColorOutput = "FinalBlit.color";
constexpr std::string_view kTraceOutput = "GlobalDistanceFieldDebug.trace";

struct Box {
    float3 center;
    float cosine;
    float sine;
    float3 halfExtent;
};

const std::array kBoxes{
    Box{float3(-0.35f, 0.0f, 0.0f), 1.0f, 0.0f, float3(0.5f)},
    Box{float3(0.1f, 0.0f, -0.05f), std::sqrt(3.0f) * 0.5f, 0.5f, float3(0.4f)},
    Box{float3(2.0625f, 0.0f, 0.0f), 1.0f, 0.0f, float3(0.5f, 0.25f, 0.25f)},
};

float3 toBoxDirection(const Box& box, float3 world)
{
    return {box.cosine * world.x - box.sine * world.z, world.y, box.sine * world.x + box.cosine * world.z};
}

float signedBoxDistance(const Box& box, float3 world)
{
    const float3 p = toBoxDirection(box, world - box.center);
    const float3 delta(std::abs(p.x) - box.halfExtent.x, std::abs(p.y) - box.halfExtent.y,
        std::abs(p.z) - box.halfExtent.z);
    const float3 positive(std::max(delta.x, 0.0f), std::max(delta.y, 0.0f), std::max(delta.z, 0.0f));
    return std::sqrt(dot(positive, positive)) + std::min(std::max({delta.x, delta.y, delta.z}), 0.0f);
}

float boxRayEntry(const Box& box, float3 worldOrigin, float margin)
{
    const float3 p = toBoxDirection(box, worldOrigin - box.center);
    const float3 d = toBoxDirection(box, float3(0.0f, 0.0f, -1.0f));
    const std::array origin{p.x, p.y, p.z};
    const std::array direction{d.x, d.y, d.z};
    const std::array extent{box.halfExtent.x + margin, box.halfExtent.y + margin, box.halfExtent.z + margin};
    float entry = 0.0f;
    float exit = std::numeric_limits<float>::infinity();
    for (size_t axis = 0; axis < origin.size(); ++axis) {
        if (direction[axis] == 0.0f) {
            if (std::abs(origin[axis]) > extent[axis]) {
                return std::numeric_limits<float>::infinity();
            }
            continue;
        }
        const float first = (-extent[axis] - origin[axis]) / direction[axis];
        const float second = (extent[axis] - origin[axis]) / direction[axis];
        entry = std::max(entry, std::min(first, second));
        exit = std::min(exit, std::max(first, second));
    }
    return entry < exit ? entry : std::numeric_limits<float>::infinity();
}

bool writeOverlapScene(const std::filesystem::path& source, const std::filesystem::path& destination)
{
    std::ifstream input(source);
    if (!input) {
        return false;
    }
    auto fixture = nlohmann::json::parse(input);
    const std::filesystem::path positionsPath = destination.parent_path() / "GlobalDistanceFieldRectangularPositions.bin";
    constexpr std::array<float, 24> positions{-0.5f, -0.25f, -0.25f, 0.5f, -0.25f, -0.25f,
        0.5f, 0.25f, -0.25f, -0.5f, 0.25f, -0.25f, -0.5f, -0.25f, 0.25f,
        0.5f, -0.25f, 0.25f, 0.5f, 0.25f, 0.25f, -0.5f, 0.25f, 0.25f};
    std::ofstream positionOutput(positionsPath, std::ios::binary);
    positionOutput.write(reinterpret_cast<const char*>(positions.data()), sizeof(positions));
    positionOutput.close();
    if (!positionOutput) {
        return false;
    }
    fixture.at("buffers").push_back({{"byteLength", sizeof(positions)}, {"uri", positionsPath.filename().generic_string()}});
    fixture.at("bufferViews").push_back({{"buffer", 1}, {"byteLength", sizeof(positions)}, {"target", 34962}});
    fixture.at("accessors").push_back({{"bufferView", 2}, {"componentType", 5126}, {"count", 8}, {"type", "VEC3"},
        {"min", {-0.5f, -0.25f, -0.25f}}, {"max", {0.5f, 0.25f, 0.25f}}});
    auto rectangularMesh = fixture.at("meshes").at(0);
    rectangularMesh["name"] = "Rectangular box";
    rectangularMesh.at("primitives").at(0).at("attributes")["POSITION"] = 2;
    fixture.at("meshes").push_back(std::move(rectangularMesh));
    auto& nodes = fixture.at("nodes");
    nodes.at(0)["translation"] = {kBoxes[0].center.x, kBoxes[0].center.y, kBoxes[0].center.z};
    nodes.at(1)["translation"] = {kBoxes[1].center.x, kBoxes[1].center.y, kBoxes[1].center.z};
    nodes.push_back({{"mesh", 1}, {"name", "Outside clipmap"}, {"translation", {
        kBoxes[2].center.x, kBoxes[2].center.y, kBoxes[2].center.z}}});
    fixture.at("scenes").at(0).at("nodes").push_back(2);
    std::ofstream output(destination);
    output << fixture;
    output.close();
    return static_cast<bool>(output);
}

struct TraceCounts {
    uint32_t hits = 0;
    uint32_t coveredMisses = 0;
    uint32_t outside = 0;
    float maxHitError = 0.0f;
};

std::string checkTrace(const render::RenderGraphPreviewRenderer& preview, bool empty, TraceCounts& counts)
{
    const auto& bytes = preview.readbackBytes();
    if (preview.readbackFormat() != render::Format::RGBA32Sfloat ||
        bytes.size() != size_t(kWidth) * kHeight * 4 * sizeof(float)) {
        return "Expected RGBA32F GDF trace data";
    }
    constexpr float kRayMargin = 2.0f * kVoxelSize;
    for (uint32_t y = 0; y < kHeight; ++y) {
        for (uint32_t x = 0; x < kWidth; ++x) {
            const float3 origin(
                (float(x) + 0.5f - 0.5f * float(kWidth)) * kOrthoHeight / float(kHeight),
                (0.5f * float(kHeight) - (float(y) + 0.5f)) * kOrthoHeight / float(kHeight),
                kCameraDistance - kNearPlane);
            std::array<float, 4> actual;
            std::memcpy(actual.data(), bytes.data() + (size_t(y) * kWidth + x) * sizeof(actual), sizeof(actual));
            const auto describeTrace = [&] {
                return " at pixel (" + std::to_string(x) + ", " + std::to_string(y) + ") from (" +
                    std::to_string(origin.x) + ", " + std::to_string(origin.y) + ", " + std::to_string(origin.z) +
                    "), distance " + std::to_string(actual[0]) + ", steps " + std::to_string(actual[1]) +
                    ", coverage " + std::to_string(actual[2]) + ", expansion " + std::to_string(actual[3]);
            };
            if (!std::ranges::all_of(actual, [](float value) { return std::isfinite(value); }) ||
                actual[1] < 0.0f || actual[1] > 256.0f || actual[2] < 0.0f || actual[2] > 2.0f ||
                actual[3] < 0.0f || actual[3] > kVoxelSize) {
                return "GDF trace contains an invalid diagnostic value" + describeTrace();
            }

            // Sample rays away from clip boundaries and silhouettes when assigning reference states.
            const float edge = std::max(std::abs(origin.x), std::abs(origin.y));
            if (edge > kExtent) {
                if (actual[0] != -1.0f || actual[1] != 0.0f || actual[2] != 0.0f) {
                    return "GDF traced a ray outside its world-space extent" + describeTrace();
                }
                ++counts.outside;
                continue;
            }
            if (edge >= kExtent - kRayMargin) {
                continue;
            }
            float nearHit = std::numeric_limits<float>::infinity();
            bool clearHit = false;
            bool clearMiss = true;
            if (!empty) {
                for (const Box& box : kBoxes) {
                    nearHit = std::min(nearHit, boxRayEntry(box, origin, 0.0f));
                    clearHit |= std::isfinite(boxRayEntry(box, origin, -kRayMargin));
                    clearMiss &= !std::isfinite(boxRayEntry(box, origin, kRayMargin));
                }
            }
            if (clearHit) {
                if (actual[0] < 0.0f || actual[1] == 0.0f || actual[2] != 2.0f) {
                    return "GDF missed an interior ray" + describeTrace();
                }
                counts.maxHitError = std::max(counts.maxHitError, std::abs(actual[0] - nearHit));
                ++counts.hits;
            } else if (clearMiss) {
                if (actual[0] != -1.0f || actual[1] == 0.0f || actual[2] != 1.0f) {
                    return "GDF did not report a covered miss" + describeTrace();
                }
                ++counts.coveredMisses;
            }
        }
    }
    if ((!empty && counts.hits < 100) || counts.coveredMisses < 1000 || counts.outside < 1000 ||
        counts.maxHitError > 3.0f * kVoxelSize) {
        return "GDF trace coverage or distance error failed: " + std::to_string(counts.maxHitError) + " m";
    }
    return {};
}

std::string checkVolume(const debug::DebugArtifact& artifact, bool empty)
{
    const auto& info = artifact.metadata;
    if (info.value("width", 0u) != kTexels || info.value("height", 0u) != kTexels ||
        info.value("depth", 0u) != kTexels || info.value("sliceCount", 0u) != kTexels ||
        !info.value("completeCoverage", false) || artifact.layout.name != "f32" ||
        artifact.bytes.size() != size_t(kTexels) * kTexels * kTexels * sizeof(float)) {
        return "Expected a complete R32F 3D GDF lattice including the final texel on each axis: " + info.dump();
    }
    uint32_t overlapSamples = 0;
    uint32_t outsideInfluenceSamples = 0;
    uint32_t saturatedSamples = 0;
    for (uint32_t z = 0; z < kTexels; ++z) {
        for (uint32_t y = 0; y < kTexels; ++y) {
            for (uint32_t x = 0; x < kTexels; ++x) {
                const size_t index = (size_t(z) * kTexels + y) * kTexels + x;
                float actual = 0.0f;
                std::memcpy(&actual, artifact.bytes.data() + index * sizeof(actual), sizeof(actual));
                const float3 p(-kExtent + float(x) * kVoxelSize, -kExtent + float(y) * kVoxelSize,
                    -kExtent + float(z) * kVoxelSize);

                const float left = signedBoxDistance(kBoxes[0], p);
                const float right = signedBoxDistance(kBoxes[1], p);
                const float outside = signedBoxDistance(kBoxes[2], p);
                const float expected = empty ? kInfluence : std::clamp(std::min({left, right, outside}), -kInfluence, kInfluence);
                const bool saturated = expected == kInfluence;
                const bool overlap = !empty && left < -kVoxelSize && right < -kVoxelSize;
                const bool outsideInfluence = !empty && outside < kInfluence - kVoxelSize && left > kInfluence && right > kInfluence;
                // Trilinear filtering and UNORM8 source quantization round corners of
                // the mesh field. Empty air and distances from a flat boundary are tighter.
                const float tolerance = saturated ? 1.0e-6f : (outsideInfluence ? 0.025f : 0.0625f);
                if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance ||
                    (overlap && actual >= 0.0f) || (outsideInfluence && actual >= kInfluence)) {
                    return "GDF voxel (" + std::to_string(x) + ", " + std::to_string(y) + ", " +
                        std::to_string(z) + ") expected " + std::to_string(expected) + " m, read " +
                        std::to_string(actual) + " m";
                }
                overlapSamples += uint32_t(overlap);
                outsideInfluenceSamples += uint32_t(outsideInfluence);
                saturatedSamples += uint32_t(saturated);
            }
        }
    }
    if ((!empty && (overlapSamples < 100 || outsideInfluenceSamples < 100)) || saturatedSamples < 1000) {
        return "GDF oracle does not exercise overlap, clipmap-exterior influence, and saturated air";
    }
    return {};
}

class GlobalDistanceFieldTest final : public RHITest {
public:
    GlobalDistanceFieldTest()
    {
        name = "global_distance_field_dense_rendering";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const std::filesystem::path root(PROJECT_SOURCE_DIR);
        const auto fixture = context.outputDirectory / "GlobalDistanceFieldBoxes.gltf";
        if (!writeOverlapScene(root / "Asset/MeshDistanceField/BoxInstances.gltf", fixture)) {
            return RHITestResult::fail("Could not write the GDF overlap fixture");
        }
        scene::SceneDocument document;
        if (!document.load(fixture)) {
            return RHITestResult::fail(document.lastLoadResult().error);
        }
        std::string log;
        const char* bakedPath = std::getenv("METALLIC_TEST_MESH_SDF");
        const auto fieldPath = bakedPath ? std::filesystem::path(bakedPath) : context.outputDirectory / "AnalyticBox.meshdf";
        if (!bakedPath && !writeAnalyticBoxField(fieldPath, document, log)) {
            return RHITestResult::fail(log);
        }
        const auto rectangularFieldPath = context.outputDirectory / "AnalyticRectangularBox.meshdf";
        if (!writeAnalyticBoxField(rectangularFieldPath, document, log, 1)) {
            return RHITestResult::fail(log);
        }

        RenderGraph graph;
        if (!loadRenderGraphFromFile(root / "Pipelines/Samples/global_distance_field.metallic_graph.json", graph, log)) {
            return RHITestResult::fail(log);
        }
        const auto* producer = graph.findNode("GlobalDistanceField");
        if (!producer) {
            return RHITestResult::fail("Missing GlobalDistanceField producer");
        }
        const auto producerId = producer->id;
        RenderView view;
        ViewCamera camera;
        camera.eye = {0.0f, 0.0f, kCameraDistance};
        camera.nearPlane = kNearPlane;
        camera.farPlane = 10.0f;
        camera.orthographic = true;
        camera.orthoHeight = kOrthoHeight;
        if (!view.setCamera(camera) || !view.setRenderResolution(kWidth, kHeight)) {
            return RHITestResult::fail("Invalid GDF test view");
        }
        RenderDebugRuntime inspector({}, RenderDebugDetail::Resources);
        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        preview.setDebugObserver(&inspector);
        if (!preview.initialize(context.enableValidation, false, false)) {
            return RHITestResult::fail("GDF preview setup: " + preview.lastLog());
        }
        preview.setRawReadbackEnabled(true);

        TraceCounts solidCounts;
        for (bool empty : {false, true}) {
            const auto paths = empty ? nlohmann::json::array() :
                nlohmann::json::array({fieldPath.generic_string(), rectangularFieldPath.generic_string()});
            if (!graph.setNodeProperties(producerId, {{"distanceFieldPaths", paths},
                    {"center", {0.0f, 0.0f, 0.0f}}, {"extent", kExtent}, {"resolution", kResolution}}) ||
                !preview.render(graph, kWidth, kHeight, kColorOutput)) {
                return RHITestResult::fail("GDF composition preview: " + preview.lastLog());
            }
            if (!empty && !saveRgba8Png(context.outputDirectory / "GlobalDistanceField.png",
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), kWidth, kHeight, log)) {
                return RHITestResult::fail(log);
            }
            const auto captureRequest = inspector.core().dispatch({{"method", "capture.batch"}, {"params", {
                {"pass", "GlobalDistanceField"}, {"checkpoint", "AfterPass"}, {"resources", {{
                    {"id", "GlobalDistanceField.distanceField"},
                    {"roi", {{"x", 0}, {"y", 0}, {"width", kTexels}, {"height", kTexels}}},
                    {"slice", 0}, {"sliceCount", kTexels},
                }}}}}});
            if (captureRequest.value("status", "") != "ok") {
                return RHITestResult::fail("GDF 3D capture request: " + captureRequest.dump());
            }
            if (!preview.render(graph, kWidth, kHeight, kTraceOutput)) {
                return RHITestResult::fail("GDF diagnostic readback: " + preview.lastLog());
            }
            inspector.poll();
            const auto& job = captureRequest.at("result").at("job");
            const auto capture = inspector.core().completedCapture(job.get<std::string>());
            if (!capture || capture->artifacts.size() != 1) {
                const auto status = inspector.core().dispatch({{"method", "jobs.get"}, {"params", {{"job", job}}}});
                return RHITestResult::fail("GDF did not complete a 3D capture: " + status.dump());
            }
            if (const auto error = checkVolume(capture->artifacts[0], empty); !error.empty()) {
                return RHITestResult::fail(error);
            }
            TraceCounts counts;
            if (const auto error = checkTrace(preview, empty, counts); !error.empty()) {
                return RHITestResult::fail(error);
            }
            if (!empty) {
                solidCounts = counts;
            }
        }
        preview.setDebugObserver(nullptr);
        return RHITestResult::pass("GDF populated and empty volumes validated with " +
            std::to_string(solidCounts.hits) + " hits, " + std::to_string(solidCounts.coveredMisses) +
            " covered misses, " + std::to_string(solidCounts.outside) + " outside rays, max hit error " +
            std::to_string(solidCounts.maxHitError) + " m");
    }
};

METALLIC_REGISTER_RHI_TEST(GlobalDistanceFieldTest);

} // namespace
} // namespace metallic::tests
