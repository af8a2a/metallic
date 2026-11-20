#include "RHITest.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/MeshDistanceField.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>

namespace metallic::tests {
namespace {

constexpr uint32_t kWidth = 257;
constexpr uint32_t kHeight = 145;

bool writeAnalyticBoxField(const std::filesystem::path& path, const scene::Scene& document, std::string& log)
{
    using namespace scene;
    MeshDistanceFieldGeometry geometry;
    if (!getMeshDistanceFieldGeometry(document, 0, geometry, log)) {
        return false;
    }
    MeshDistanceField field;
    field.sourceMeshIndex = 0;
    field.sourceFingerprint = geometry.fingerprint;
    field.localBoundsMin.fill(-0.5f);
    field.localBoundsMax.fill(0.5f);

    // Generate an independent, exact signed box field at the brick sample positions.
    // The runtime fixture covers serialization, upload, indirection and interpolation
    // on configurations without the optional Embree cook target.
    constexpr uint32_t kUnique = kMeshDistanceFieldUniqueBrickSize;
    constexpr uint32_t kBrick = kMeshDistanceFieldBrickSize;
    constexpr std::array<uint32_t, kMeshDistanceFieldMipCount> kDimensions{3, 2, 1};
    for (size_t level = 0; level < field.mips.size(); ++level) {
        auto& mip = field.mips[level];
        const uint32_t dimension = kDimensions[level];
        const float voxel = 2.0f / float(dimension * kUnique - 2);
        const float band = 4.0f * std::sqrt(3.0f) * voxel;
        mip.indirectionDimensions.fill(dimension);
        mip.volumeToVirtualUVScale.fill(1.0f / (voxel * float(dimension * kUnique)));
        mip.volumeToVirtualUVAdd.fill(0.5f);
        mip.distanceFieldToVolumeScaleBias = {2.0f * band, -band};

        const auto toBox = [voxel](uint32_t sample) {
            return std::abs(-1.0f + (float(sample) - 1.0f) * voxel) - 1.0f;
        };
        for (uint32_t bz = 0; bz < dimension; ++bz) {
            for (uint32_t by = 0; by < dimension; ++by) {
                for (uint32_t bx = 0; bx < dimension; ++bx) {
                    mip.indirection.push_back(static_cast<uint32_t>(mip.indirection.size()));
                    for (uint32_t z = 0; z < kBrick; ++z) {
                        for (uint32_t y = 0; y < kBrick; ++y) {
                            for (uint32_t x = 0; x < kBrick; ++x) {
                                const float3 q(toBox(bx * kUnique + x), toBox(by * kUnique + y), toBox(bz * kUnique + z));
                                const float3 outside(std::max(0.0f, q.x), std::max(0.0f, q.y), std::max(0.0f, q.z));
                                const float distance = std::sqrt(dot(outside, outside)) +
                                    std::min(0.0f, std::max({q.x, q.y, q.z}));
                                const float encoded = std::clamp((distance + band) / (2.0f * band), 0.0f, 1.0f);
                                mip.bricks.push_back(static_cast<uint8_t>(std::lround(encoded * 255.0f)));
                            }
                        }
                    }
                }
            }
        }
    }
    return saveMeshDistanceField(path, field, log);
}

struct Box {
    float3 center;
    float cosine;
    float sine;
    float halfSize;
};

float3 normalizeDirection(float3 direction)
{
    return direction / std::sqrt(dot(direction, direction));
}

float intersectBox(const Box& box, float3 worldOrigin, float3 worldDirection, float halfSize)
{
    const auto toBox = [&](float3 v) {
        return std::array{box.cosine * v.x - box.sine * v.z, v.y, box.sine * v.x + box.cosine * v.z};
    };
    const auto origin = toBox(worldOrigin - box.center);
    const auto direction = toBox(worldDirection);
    float entry = 0.0f;
    float exit = std::numeric_limits<float>::infinity();
    for (size_t axis = 0; axis < origin.size(); ++axis) {
        if (direction[axis] == 0.0f) {
            if (std::abs(origin[axis]) > halfSize) {
                return -1.0f;
            }
            continue;
        }
        const float first = (-halfSize - origin[axis]) / direction[axis];
        const float second = (halfSize - origin[axis]) / direction[axis];
        entry = std::max(entry, std::min(first, second));
        exit = std::min(exit, std::max(first, second));
    }
    return entry < exit ? entry : -1.0f;
}

class MeshDistanceFieldDebugTest final : public RHITest {
public:
    MeshDistanceFieldDebugTest()
    {
        name = "mesh_distance_field_debug_rendering";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const std::filesystem::path root(PROJECT_SOURCE_DIR);
        scene::SceneDocument document;
        if (!document.load(root / "Asset/MeshDistanceField/BoxInstances.gltf")) {
            return RHITestResult::fail(document.lastLoadResult().error);
        }
        std::string log;
        RenderGraph graph;
        if (!loadRenderGraphFromFile(root / "Pipelines/Samples/mesh_distance_field_debug.metallic_graph.json", graph, log)) {
            return RHITestResult::fail(log);
        }
        const char* bakedPath = std::getenv("METALLIC_TEST_MESH_SDF");
        const auto fieldPath = bakedPath ? std::filesystem::path(bakedPath) : context.outputDirectory / "AnalyticBox.meshdf";
        if (!bakedPath && !writeAnalyticBoxField(fieldPath, document, log)) {
            return RHITestResult::fail(log);
        }
        const auto* node = graph.findNode("MeshDistanceField");
        if (!node || !graph.setNodeProperties(node->id, {{"distanceFieldPath", fieldPath.generic_string()}})) {
            return RHITestResult::fail("Missing mesh distance field debug node");
        }
        RenderView view;
        if (!view.setCameraProperties(graph.viewProperties().at("camera"))) {
            return RHITestResult::fail("Invalid mesh distance field test camera");
        }
        view.setRenderResolution(kWidth, kHeight);

        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        const auto setup = preview.initialize(context.enableValidation, false, false);
        if (!setup) {
            return RHITestResult::fail("Mesh distance field preview setup: " + preview.lastLog());
        }
        if (!preview.render(graph, kWidth, kHeight, "FinalBlit.color") ||
            !saveRgba8Png(context.outputDirectory / "MeshDistanceFieldDebug.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), kWidth, kHeight, log)) {
            return RHITestResult::fail("Mesh distance field PNG: " + preview.lastLog() + log);
        }
        preview.setRawReadbackEnabled(true);
        if (!preview.render(graph, kWidth, kHeight, "MeshDistanceField.trace")) {
            return RHITestResult::fail("Mesh distance field trace readback: " + preview.lastLog());
        }
        const auto& trace = preview.readbackBytes();
        if (preview.readbackFormat() != Format::RGBA32Sfloat || trace.size() != size_t(kWidth) * kHeight * 4 * sizeof(float)) {
            return RHITestResult::fail("Expected RGBA32F hit distance, steps, instance ID and hit flag");
        }

        // The independent oracle uses the authored two cuboids and allows a band
        // around silhouette pixels where the coarse sampled field rounds edges.
        const std::array boxes{Box{float3(-1.05f, 0.0f, 0.0f), 1.0f, 0.0f, 0.5f},
            Box{float3(0.9f, 0.0f, -0.15f), std::sqrt(3.0f) * 0.5f, 0.5f, 0.4f}};
        const auto& camera = view.camera();
        const float3 eye(camera.eye[0], camera.eye[1], camera.eye[2]);
        const float3 target(camera.center[0], camera.center[1], camera.center[2]);
        const float3 up(camera.up[0], camera.up[1], camera.up[2]);
        const float3 forward = normalizeDirection(target - eye);
        const float3 right = normalizeDirection(cross(forward, up));
        const float3 vertical = cross(right, forward);
        const float tangent = std::tan(camera.fovDegrees * std::numbers::pi_v<float> / 360.0f);
        std::array<uint32_t, 2> checkedHits{};
        uint32_t checkedMisses = 0;
        float maxError = 0.0f;
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                const float u = (2.0f * (float(x) + 0.5f) - float(kWidth)) / float(kHeight);
                const float v = 1.0f - 2.0f * (float(y) + 0.5f) / float(kHeight);
                const float3 direction = normalizeDirection(forward + (right * u + vertical * v) * tangent);
                std::array<float, 4> pixel;
                std::memcpy(pixel.data(), trace.data() + (size_t(y) * kWidth + x) * sizeof(pixel), sizeof(pixel));
                if (!std::ranges::all_of(pixel, [](float value) { return std::isfinite(value); }) ||
                    pixel[1] < 0.0f || pixel[1] > 128.0f || (pixel[3] != 0.0f && pixel[3] != 1.0f)) {
                    return RHITestResult::fail("Invalid mesh distance field trace channels");
                }
                constexpr float kSilhouetteBand = 0.026f;
                const auto hitsOutsideBand = [&](const Box& box) {
                    return intersectBox(box, eye, direction, box.halfSize + kSilhouetteBand) >= 0.0f;
                };
                if (!std::ranges::any_of(boxes, hitsOutsideBand)) {
                    if (pixel[0] != -1.0f || pixel[2] != 0.0f || pixel[3] != 0.0f) {
                        return RHITestResult::fail("Clear miss recorded a hit at " + std::to_string(x) + ", " + std::to_string(y));
                    }
                    ++checkedMisses;
                }
                for (size_t index = 0; index < boxes.size(); ++index) {
                    const Box& box = boxes[index];
                    if (intersectBox(box, eye, direction, box.halfSize - kSilhouetteBand) < 0.0f) {
                        continue;
                    }
                    const float expectedDistance = intersectBox(box, eye, direction, box.halfSize);
                    if (pixel[3] != 1.0f || pixel[2] != float(index + 1) || pixel[1] < 1.0f) {
                        return RHITestResult::fail("Missing trace for cuboid " + std::to_string(index + 1) +
                            " at " + std::to_string(x) + ", " + std::to_string(y));
                    }
                    maxError = std::max(maxError, std::abs(pixel[0] - expectedDistance));
                    ++checkedHits[index];
                }
            }
        }
        if (checkedHits[0] < 300 || checkedHits[1] < 300 || checkedMisses < 1000 || maxError > 0.035f) {
            return RHITestResult::fail("Unexpected cuboid coverage or hit error: " + std::to_string(maxError) + " m");
        }
        return RHITestResult::pass(std::to_string(checkedHits[0]) + " + " + std::to_string(checkedHits[1]) +
            " interior hits, " + std::to_string(checkedMisses) + " clear misses, max distance error " +
            std::to_string(maxError) + " m");
    }
};

METALLIC_REGISTER_RHI_TEST(MeshDistanceFieldDebugTest);

} // namespace
} // namespace metallic::tests
