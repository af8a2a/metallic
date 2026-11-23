#include "MeshDistanceFieldFixture.h"
#include "RHITest.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Scene/SceneDocument.h"
#include "harness/Fixtures.h"
#include "harness/ValidationRecorder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

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

bool prepareFieldPaths(const std::filesystem::path& outputDirectory, scene::SceneDocument& document,
    std::array<std::string, 2>& paths, std::string& log)
{
    const std::filesystem::path root(PROJECT_SOURCE_DIR);
    const auto fixture = outputDirectory / "GlobalDistanceFieldBoxes.gltf";
    if (!writeOverlapScene(root / "Asset/MeshDistanceField/BoxInstances.gltf", fixture)) {
        log = "Could not write the GDF overlap fixture";
        return false;
    }
    if (!document.load(fixture)) {
        log = document.lastLoadResult().error;
        return false;
    }
    const char* bakedPath = std::getenv("METALLIC_TEST_MESH_SDF");
    const auto fieldPath = bakedPath ? std::filesystem::path(bakedPath) : outputDirectory / "AnalyticBox.meshdf";
    if (!bakedPath && !writeAnalyticBoxField(fieldPath, document, log)) {
        return false;
    }
    const auto rectangularFieldPath = outputDirectory / "AnalyticRectangularBox.meshdf";
    if (!writeAnalyticBoxField(rectangularFieldPath, document, log, 1)) {
        return false;
    }
    paths = {fieldPath.generic_string(), rectangularFieldPath.generic_string()};
    return true;
}

bool configureGdfTestView(render::RenderView& view)
{
    render::ViewCamera camera;
    camera.eye = {0.0f, 0.0f, kCameraDistance};
    camera.nearPlane = kNearPlane;
    camera.farPlane = 10.0f;
    camera.orthographic = true;
    camera.orthoHeight = kOrthoHeight;
    return view.setCamera(camera) && view.setRenderResolution(kWidth, kHeight);
}

debug::DebugValue textureCapture(std::string_view id, uint32_t width, uint32_t height, uint32_t depth)
{
    return {{"id", id}, {"roi", {{"x", 0}, {"y", 0}, {"width", width}, {"height", height}}},
        {"slice", 0}, {"sliceCount", depth}};
}

struct GdfCapturePreview {
    GdfCapturePreview()
    {
        preview.setDebugObserver(&inspector);
    }

    ~GdfCapturePreview()
    {
        preview.setDebugObserver(nullptr);
    }

    // The inspector drains pending Vulkan captures before the preview destroys its device.
    render::RenderGraphPreviewRenderer preview;
    render::RenderDebugRuntime inspector{{}, render::RenderDebugDetail::Resources};
};

std::shared_ptr<const debug::DebugCapture> captureAfterPass(render::RenderGraphPreviewRenderer& preview,
    render::RenderDebugRuntime& inspector, render::RenderGraph& graph, std::string_view pass,
    debug::DebugValue resources, std::string_view output, std::string& log)
{
    const auto request = inspector.core().dispatch({{"method", "capture.batch"}, {"params", {
        {"pass", pass}, {"checkpoint", "AfterPass"}, {"resources", std::move(resources)},
    }}});
    if (request.value("status", "") != "ok") {
        log = std::string(pass) + " capture request: " + request.dump();
        return {};
    }
    if (!preview.render(graph, kWidth, kHeight, output)) {
        log = std::string(pass) + " diagnostic readback of " + std::string(output) + ": " + preview.lastLog();
        return {};
    }
    inspector.poll();
    const auto& job = request.at("result").at("job");
    auto capture = inspector.core().completedCapture(job.get<std::string>());
    if (!capture) {
        const auto status = inspector.core().dispatch({{"method", "jobs.get"}, {"params", {{"job", job}}}});
        log = std::string(pass) + " did not complete a capture: " + status.dump();
    }
    return capture;
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
        scene::SceneDocument document;
        std::string log;
        std::array<std::string, 2> paths;
        if (!prepareFieldPaths(context.outputDirectory, document, paths, log)) {
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
        if (!configureGdfTestView(view)) {
            return RHITestResult::fail("Invalid GDF test view");
        }
        GdfCapturePreview capturePreview;
        auto& preview = capturePreview.preview;
        auto& inspector = capturePreview.inspector;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        if (!preview.initialize(context.enableValidation, false, false)) {
            return RHITestResult::fail("GDF preview setup: " + preview.lastLog());
        }
        preview.setRawReadbackEnabled(true);

        TraceCounts solidCounts;
        for (bool empty : {false, true}) {
            const auto sourcePaths = empty ? nlohmann::json::array() : nlohmann::json(paths);
            if (!graph.setNodeProperties(producerId, {{"distanceFieldPaths", sourcePaths},
                    {"center", {0.0f, 0.0f, 0.0f}}, {"extent", kExtent}, {"resolution", kResolution}}) ||
                !preview.render(graph, kWidth, kHeight, kColorOutput)) {
                return RHITestResult::fail("GDF composition preview: " + preview.lastLog());
            }
            if (!empty && !saveRgba8Png(context.outputDirectory / "GlobalDistanceField.png",
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), kWidth, kHeight, log)) {
                return RHITestResult::fail(log);
            }
            const auto capture = captureAfterPass(preview, inspector, graph, "GlobalDistanceField",
                debug::DebugValue::array({textureCapture("GlobalDistanceField.distanceField", kTexels, kTexels, kTexels)}),
                kTraceOutput, log);
            if (!capture) { return RHITestResult::fail(log); }
            if (capture->artifacts.size() != 1) { return RHITestResult::fail("Expected one GDF 3D artifact"); }
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
        return RHITestResult::pass("GDF populated and empty volumes validated with " +
            std::to_string(solidCounts.hits) + " hits, " + std::to_string(solidCounts.coveredMisses) +
            " covered misses, " + std::to_string(solidCounts.outside) + " outside rays, max hit error " +
            std::to_string(solidCounts.maxHitError) + " m");
    }
};

METALLIC_REGISTER_RHI_TEST(GlobalDistanceFieldTest);

constexpr uint32_t kLogicalPageWidth = 7;
constexpr uint32_t kPhysicalPageWidth = 8;
constexpr uint32_t kPageTableEdge = (kResolution + kLogicalPageWidth - 1) / kLogicalPageWidth;
constexpr uint32_t kPositivePage = 0xffffffffu;
constexpr uint32_t kNegativePage = 0xfffffffeu;
constexpr uint32_t kOverflowPage = 0xfffffffdu;
constexpr std::string_view kSparsePass = "GlobalDistanceFieldPages";
constexpr std::string_view kSparseTrace = "GlobalDistanceFieldPagesDebug.trace";

struct PageCounts {
    uint32_t required = 0;
    uint32_t resident = 0;
    uint32_t positive = 0;
    uint32_t negative = 0;
    uint32_t overflow = 0;
    uint32_t sharedFaces = 0;
};

template<typename Value, typename Byte>
Value readGdfElement(const std::vector<Byte>& bytes, size_t index)
{
    Value value;
    std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
    return value;
}

size_t linearVoxelIndex(uint32_t x, uint32_t y, uint32_t z, uint32_t edge)
{
    return (size_t(z) * edge + y) * edge + x;
}

bool matchesVolumeLayout(const debug::DebugArtifact& artifact, uint32_t edge, std::string_view layout,
    uint32_t levelCount = 1)
{
    const uint32_t depth = edge * levelCount;
    return artifact.layout.name == layout && artifact.metadata.value("width", 0u) == edge &&
        artifact.metadata.value("height", 0u) == edge && artifact.metadata.value("depth", 0u) == depth &&
        artifact.metadata.value("completeCoverage", false) &&
        artifact.bytes.size() == size_t(edge) * edge * depth * (layout == "R8" ? 1u : 4u);
}

size_t atlasVoxelIndex(uint32_t page, uint32_t x, uint32_t y, uint32_t z, uint32_t atlasPageEdge)
{
    return linearVoxelIndex(page % atlasPageEdge * kPhysicalPageWidth + x,
        (page / atlasPageEdge) % atlasPageEdge * kPhysicalPageWidth + y,
        (page / (atlasPageEdge * atlasPageEdge)) * kPhysicalPageWidth + z, atlasPageEdge * kPhysicalPageWidth);
}

std::string checkSparsePages(const debug::DebugCapture& capture, const debug::DebugArtifact* denseVolume,
    uint32_t capacity, uint32_t atlasPageEdge, PageCounts& counts, bool quantized = false)
{
    const bool empty = denseVolume == nullptr;
    if (capture.artifacts.size() != (empty ? 2u : 3u) + uint32_t(quantized)) {
        return "Unexpected sparse GDF readback resource count";
    }
    const auto& table = capture.artifacts[0];
    const auto& statistics = capture.artifacts[1];
    const uint32_t atlasTexelEdge = atlasPageEdge * kPhysicalPageWidth;
    if (!matchesVolumeLayout(table, kPageTableEdge, "u32") || statistics.bytes.size() != 5 * sizeof(uint32_t) ||
        (!empty && (!matchesVolumeLayout(capture.artifacts[2], atlasTexelEdge, quantized ? "R8" : "f32") ||
            !matchesVolumeLayout(*denseVolume, kTexels, "f32")))) {
        return "Sparse GDF page table, atlas, or statistics readback layout differs from its contract";
    }

    std::vector<bool> allocated(capacity);
    const auto pageAt = [&](uint32_t x, uint32_t y, uint32_t z) {
        return readGdfElement<uint32_t>(table.bytes, linearVoxelIndex(x, y, z, kPageTableEdge));
    };
    const auto atlasSample = [&](uint32_t page, uint32_t x, uint32_t y, uint32_t z) {
        const size_t index = atlasVoxelIndex(page, x, y, z, atlasPageEdge);
        const auto& bytes = capture.artifacts[2].bytes;
        return quantized ? (float(bytes[index]) * (2.0f / 255.0f) - 1.0f) * kInfluence :
            readGdfElement<float>(bytes, index);
    };

    constexpr float kVoxelTolerance = 2.0e-5f;
    for (uint32_t pz = 0; pz < kPageTableEdge; ++pz) {
        for (uint32_t py = 0; py < kPageTableEdge; ++py) {
            for (uint32_t px = 0; px < kPageTableEdge; ++px) {
                const uint32_t page = pageAt(px, py, pz);
                const bool resident = page < capacity;
                if (resident) {
                    if (allocated[page] || empty) { return "Sparse GDF resident page was aliased or allocated for an empty field"; }
                    allocated[page] = true;
                    ++counts.resident;
                } else if (page == kPositivePage) {
                    ++counts.positive;
                } else if (page == kNegativePage) {
                    ++counts.negative;
                } else if (page == kOverflowPage) {
                    ++counts.overflow;
                } else {
                    return "Sparse GDF page table contains an invalid physical index";
                }

                bool nonconstant = false;
                for (uint32_t z = 0; z < kPhysicalPageWidth; ++z) {
                    for (uint32_t y = 0; y < kPhysicalPageWidth; ++y) {
                        for (uint32_t x = 0; x < kPhysicalPageWidth; ++x) {
                            const uint32_t dx = std::min(px * kLogicalPageWidth + x, kResolution);
                            const uint32_t dy = std::min(py * kLogicalPageWidth + y, kResolution);
                            const uint32_t dz = std::min(pz * kLogicalPageWidth + z, kResolution);
                            const float expected = empty ? kInfluence : readGdfElement<float>(denseVolume->bytes,
                                linearVoxelIndex(dx, dy, dz, kTexels));
                            nonconstant |= std::abs(expected) < kInfluence - kVoxelTolerance;
                            if (page == kOverflowPage) { continue; }
                            const float actual = resident ? atlasSample(page, x, y, z) :
                                (page == kPositivePage ? kInfluence : -kInfluence);
                            // Vulkan float-to-UNORM conversion may choose either adjacent
                            // integer. The saturated endpoints must remain exact.
                            const bool rounded = quantized && resident && std::abs(expected) < kInfluence;
                            const float tolerance = kVoxelTolerance + (rounded ? 2.0f * kInfluence / 255.0f : 0.0f);
                            if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
                                return "Sparse/dense GDF sample mismatch at (" + std::to_string(dx) + ", " +
                                    std::to_string(dy) + ", " + std::to_string(dz) + "), page " +
                                    std::to_string(page) + " expected " + std::to_string(expected) + ", read " +
                                    std::to_string(actual);
                            }
                        }
                    }
                }
                if (nonconstant && (resident || page == kOverflowPage)) { ++counts.required; }
                if (resident) {
                    counts.sharedFaces += uint32_t(px + 1 < kPageTableEdge && pageAt(px + 1, py, pz) < capacity);
                    counts.sharedFaces += uint32_t(py + 1 < kPageTableEdge && pageAt(px, py + 1, pz) < capacity);
                    counts.sharedFaces += uint32_t(pz + 1 < kPageTableEdge && pageAt(px, py, pz + 1) < capacity);
                }
            }
        }
    }
    if (counts.required != counts.resident + counts.overflow) {
        return "Sparse GDF allocated a page without an unsaturated dense sample";
    }
    const std::array expectedStatistics{counts.required, counts.resident, counts.positive, counts.negative, counts.overflow};
    for (size_t i = 0; i < expectedStatistics.size(); ++i) {
        if (readGdfElement<uint32_t>(statistics.bytes, i) != expectedStatistics[i]) {
            return "Sparse GDF GPU statistics do not match the page table";
        }
    }
    return {};
}

std::string checkSparseTrace(const render::RenderGraphPreviewRenderer& preview, const std::vector<std::byte>& dense,
    bool overflow)
{
    const auto& bytes = preview.readbackBytes();
    constexpr size_t kBytes = size_t(kWidth) * kHeight * sizeof(std::array<float, 4>);
    if (preview.readbackFormat() != render::Format::RGBA32Sfloat || bytes.size() != kBytes || dense.size() != kBytes) {
        return "Sparse GDF trace must publish RGBA32F at the camera resolution";
    }
    uint32_t unsupportedRays = 0;
    uint32_t unresolvedHits = 0;
    for (size_t i = 0; i < size_t(kWidth) * kHeight; ++i) {
        const auto actual = readGdfElement<std::array<float, 4>>(bytes, i);
        const auto expected = readGdfElement<std::array<float, 4>>(dense, i);
        if (!std::ranges::all_of(actual, [](float value) { return std::isfinite(value); }) ||
            actual[1] < 0.0f || actual[1] > 256.0f) {
            return "Sparse GDF trace produced an invalid diagnostic value";
        }
        if (overflow && actual[2] == 3.0f) {
            if (actual[0] != -1.0f || actual[3] != 0.0f || expected[2] == 0.0f) {
                return "Sparse GDF reported a hit, expansion or out-of-domain ray as budget unavailable";
            }
            ++unsupportedRays;
            unresolvedHits += uint32_t(expected[2] == 2.0f);
            continue;
        }
        if (actual[2] != expected[2] || std::abs(actual[0] - expected[0]) > 0.002f ||
            std::abs(actual[3] - expected[3]) > 1.0e-5f) {
            return "Sparse/dense GDF trace mismatch at pixel " + std::to_string(i % kWidth) + ", " +
                std::to_string(i / kWidth) + " expected status " + std::to_string(expected[2]) +
                " and distance " + std::to_string(expected[0]) + ", read status " + std::to_string(actual[2]) +
                " and distance " + std::to_string(actual[0]);
        }
    }
    return overflow && (unsupportedRays == 0 || unresolvedHits == 0) ?
        "Sparse GDF trace never reported unavailable coverage on a dense hit" : "";
}

class GlobalDistanceFieldPagesTest final : public RHITest {
public:
    GlobalDistanceFieldPagesTest()
    {
        name = "global_distance_field_sparse_pages";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        constexpr uint32_t kFullCapacity = 216;
        constexpr uint32_t kFullAtlasPageEdge = 6;
        const std::filesystem::path root(PROJECT_SOURCE_DIR);
        const auto fixtureDirectory = context.outputDirectory / "SparseGlobalDistanceField";
        std::error_code directoryError;
        std::filesystem::create_directories(fixtureDirectory, directoryError);
        if (directoryError) { return RHITestResult::fail(directoryError.message()); }

        scene::SceneDocument document;
        std::array<std::string, 2> paths;
        std::string log;
        if (!prepareFieldPaths(fixtureDirectory, document, paths, log)) { return RHITestResult::fail(log); }
        RenderGraph denseGraph, sparseGraph;
        if (!loadRenderGraphFromFile(root / "Pipelines/Samples/global_distance_field.metallic_graph.json", denseGraph, log) ||
            !loadRenderGraphFromFile(root / "Pipelines/Samples/global_distance_field_pages.metallic_graph.json", sparseGraph, log)) {
            return RHITestResult::fail(log);
        }
        const auto* denseProducer = denseGraph.findNode("GlobalDistanceField");
        const auto* sparseProducer = sparseGraph.findNode(kSparsePass);
        if (!denseProducer || !sparseProducer) { return RHITestResult::fail("Missing dense or sparse GDF producer"); }
        const auto sparseId = sparseProducer->id;
        const nlohmann::json properties{{"distanceFieldPaths", paths}, {"center", {0.0f, 0.0f, 0.0f}},
            {"extent", kExtent}, {"resolution", kResolution}};

        RenderView view;
        if (!configureGdfTestView(view)) { return RHITestResult::fail("Invalid sparse GDF test view"); }
        GdfCapturePreview capturePreview;
        auto& preview = capturePreview.preview;
        auto& inspector = capturePreview.inspector;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        if (!preview.initialize(context.enableValidation, false, false)) {
            return RHITestResult::fail("Sparse GDF preview setup: " + preview.lastLog());
        }
        preview.setRawReadbackEnabled(true);
        if (!denseGraph.setNodeProperties(denseProducer->id, properties) ||
            !preview.render(denseGraph, kWidth, kHeight, kColorOutput)) {
            return RHITestResult::fail("Sparse GDF dense reference setup: " + preview.lastLog());
        }
        const auto dense = captureAfterPass(preview, inspector, denseGraph, "GlobalDistanceField",
            debug::DebugValue::array({textureCapture("GlobalDistanceField.distanceField", kTexels, kTexels, kTexels)}),
            kTraceOutput, log);
        if (!dense) { return RHITestResult::fail(log); }
        if (dense->artifacts.size() != 1) { return RHITestResult::fail("Expected one dense reference artifact"); }
        const auto denseTrace = preview.readbackBytes();

        uint32_t populatedPages = 0;
        const auto runSparse = [&](uint32_t capacity, uint32_t atlasPageEdge, bool empty) -> std::string {
            auto sparseProperties = properties;
            sparseProperties["maxPhysicalPages"] = capacity;
            if (empty) { sparseProperties["distanceFieldPaths"] = nlohmann::json::array(); }
            if (!sparseGraph.setNodeProperties(sparseId, std::move(sparseProperties)) ||
                !preview.render(sparseGraph, kWidth, kHeight, kColorOutput)) {
                return "Sparse GDF composition preview: " + preview.lastLog();
            }
            if (capacity == kFullCapacity && !saveRgba8Png(context.outputDirectory / "GlobalDistanceFieldPages.png",
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), kWidth, kHeight, log)) {
                return log;
            }

            auto resources = debug::DebugValue::array({
                textureCapture("GlobalDistanceFieldPages.pageTable", kPageTableEdge, kPageTableEdge, kPageTableEdge),
                {{"id", "GlobalDistanceFieldPages.statistics"}, {"layout", "u32"}, {"count", 5}},
            });
            if (!empty) {
                const uint32_t atlasEdge = atlasPageEdge * kPhysicalPageWidth;
                resources.push_back(textureCapture("GlobalDistanceFieldPages.pageAtlas", atlasEdge, atlasEdge, atlasEdge));
            }
            const auto pages = captureAfterPass(preview, inspector, sparseGraph, kSparsePass,
                std::move(resources), kSparseTrace, log);
            if (!pages) { return log; }
            PageCounts counts;
            if (const auto error = checkSparsePages(*pages, empty ? nullptr : &dense->artifacts[0], capacity,
                    atlasPageEdge, counts); !error.empty()) {
                return error;
            }
            if (empty) {
                if (counts.positive != kPageTableEdge * kPageTableEdge * kPageTableEdge) {
                    return "Empty sparse GDF did not mark every logical page as constant air";
                }
                TraceCounts traceCounts;
                return checkTrace(preview, true, traceCounts);
            }
            if (counts.resident == 0 || counts.positive == 0 || counts.negative == 0) {
                return "Populated sparse GDF did not exercise resident, constant-air and constant-interior pages";
            }
            const bool overflow = capacity != kFullCapacity;
            if (overflow) {
                if (counts.required != populatedPages || counts.resident != capacity || counts.overflow == 0) {
                    return "Sparse GDF budget did not publish unavailable pages at capacity";
                }
            } else {
                if (counts.overflow != 0 || counts.sharedFaces == 0) {
                    return "Full sparse page capacity unexpectedly overflowed or did not cover a shared face";
                }
                populatedPages = counts.required;
            }
            return checkSparseTrace(preview, denseTrace, overflow);
        };
        if (const auto error = runSparse(kFullCapacity, kFullAtlasPageEdge, false); !error.empty()) {
            return RHITestResult::fail(error);
        }
        if (const auto error = runSparse(1, 1, false); !error.empty()) { return RHITestResult::fail(error); }
        if (const auto error = runSparse(1, 1, true); !error.empty()) { return RHITestResult::fail(error); }
        return RHITestResult::pass("Sparse and dense GPU fields matched both individual samples and traces with " +
            std::to_string(populatedPages) + " required physical pages. Constant and unavailable coverage validated.");
    }
};

METALLIC_REGISTER_RHI_TEST(GlobalDistanceFieldPagesTest);

constexpr uint32_t kCoarseFactor = 4;
constexpr uint32_t kCoarseEdge = (kResolution + kCoarseFactor - 1) / kCoarseFactor;
constexpr float kCoarseTransition = 0.625f;
constexpr float kFineQuantum = 2.0f * kInfluence / 255.0f;

float encodedPageLimit(uint32_t page)
{
    return page == kNegativePage ? 0.0f : 1.0f;
}

std::string checkCoarseMip(const debug::DebugCapture& capture, uint32_t atlasPageEdge, bool empty,
    uint32_t resolution = kResolution, uint32_t levelCount = 1)
{
    const uint32_t coarseEdge = resolution / kCoarseFactor;
    const uint32_t coarseDepth = coarseEdge * levelCount;
    const uint32_t pageTableEdge = (resolution + kLogicalPageWidth - 1) / kLogicalPageWidth;
    const auto& coarse = capture.artifacts.back();
    if (!matchesVolumeLayout(coarse, coarseEdge, "R8", levelCount)) { return "Expected a factor-four R8 GDF coarse mip"; }
    const auto& table = capture.artifacts[0].bytes;
    const auto index = [&](uint32_t x, uint32_t y, uint32_t z) { return linearVoxelIndex(x, y, z, coarseEdge); };
    std::vector<float> previous(coarse.bytes.size(), 1.0f);
    if (!empty) {
        const auto& fine = capture.artifacts[2].bytes;
        for (uint32_t z = 0; z < coarseDepth; ++z) {
            const uint32_t level = z / coarseEdge;
            for (uint32_t y = 0; y < coarseEdge; ++y) {
                for (uint32_t x = 0; x < coarseEdge; ++x) {
                    // These resolutions are multiples of four, so each coarse center is a fine texel.
                    const uint32_t fx = x * kCoarseFactor + kCoarseFactor / 2;
                    const uint32_t fy = y * kCoarseFactor + kCoarseFactor / 2;
                    const uint32_t fz = (z % coarseEdge) * kCoarseFactor + kCoarseFactor / 2;
                    const uint32_t page = readGdfElement<uint32_t>(table, linearVoxelIndex(
                        fx / kLogicalPageWidth, fy / kLogicalPageWidth,
                        fz / kLogicalPageWidth + level * pageTableEdge, pageTableEdge));
                    const float encoded = page >= kOverflowPage ? encodedPageLimit(page) :
                        float(fine[atlasVoxelIndex(page, fx % kLogicalPageWidth, fy % kLogicalPageWidth,
                            fz % kLogicalPageWidth, atlasPageEdge)]) / 255.0f;
                    if (encoded < 1.0f) {
                        previous[index(x, y, z)] = encoded / float(kCoarseFactor) + 0.375f;
                    }
                }
            }
        }
    }

    // UE 5.7.4 GlobalDistanceFieldMip.usf, including the tie handling in SortedV and
    // quantization on all five writes. The initial neighbors come from the fine pages.
    std::vector<float> next(previous.size());
    std::vector<int> expected(previous.size());
    constexpr float step = 0.125f;
    const uint32_t last = coarseEdge - 1;
    for (uint32_t pass = 0; pass < 5; ++pass) {
        for (uint32_t z = 0; z < coarseDepth; ++z) {
            const uint32_t localZ = z % coarseEdge;
            for (uint32_t y = 0; y < coarseEdge; ++y) {
                for (uint32_t x = 0; x < coarseEdge; ++x) {
                    const std::array neighbors{
                        std::min(previous[index(x == 0 ? 0 : x - 1, y, z)], previous[index(std::min(x + 1, last), y, z)]),
                        std::min(previous[index(x, y == 0 ? 0 : y - 1, z)], previous[index(x, std::min(y + 1, last), z)]),
                        std::min(previous[index(x, y, localZ == 0 ? z : z - 1)],
                            previous[index(x, y, localZ == last ? z : z + 1)]),
                    };
                    const float a = std::min({neighbors[0], neighbors[1], neighbors[2]});
                    const float c = std::max({neighbors[0], neighbors[1], neighbors[2]});
                    const float b = neighbors[0] != a && neighbors[0] != c ? neighbors[0] :
                        neighbors[1] != a && neighbors[1] != c ? neighbors[1] : neighbors[2];
                    const float sum2 = a + b;
                    const float sum3 = a + b + c;
                    const float squares2 = a * a + b * b;
                    const float squares3 = a * a + b * b + c * c;
                    const float eikonal2 = 0.5f * (sum2 + std::sqrt(sum2 * sum2 - 2.0f * (squares2 - step * step)));
                    const float eikonal3 = (1.0f / 3.0f) * (sum3 + std::sqrt(sum3 * sum3 - 3.0f * (squares3 - step * step)));
                    const size_t flatIndex = index(x, y, z);
                    float center = std::min(previous[flatIndex], a + step);
                    center = std::min(center, eikonal2);
                    center = std::min(center, eikonal3);
                    expected[flatIndex] = int(std::floor(std::clamp(center, 0.0f, 1.0f) * 255.0f + 0.5f));
                    next[flatIndex] = float(expected[flatIndex]) / 255.0f;
                }
            }
        }
        previous.swap(next);
    }

    uint32_t positive = 0;
    uint32_t negative = 0;
    for (size_t i = 0; i < coarse.bytes.size(); ++i) {
        const int actual = coarse.bytes[i];
        if (std::abs(actual - expected[i]) > (empty ? 0 : 1)) {
            return "R8 GDF coarse mip differs from UE propagation at texel " + std::to_string(i) +
                ", expected " + std::to_string(expected[i]) + ", read " + std::to_string(actual);
        }
        positive += uint32_t(actual > 127 && actual < 255);
        negative += uint32_t(actual < 127);
    }
    if (!empty && (positive == 0 || negative == 0)) {
        return "Coarse mip fixture failed to exercise both propagated distance signs";
    }
    return {};
}

float sampleR8(const std::vector<uint8_t>& values, uint32_t edge, const std::array<float, 3>& texel)
{
    std::array<uint32_t, 3> a{}, b{};
    std::array<float, 3> weight{};
    for (size_t axis = 0; axis < texel.size(); ++axis) {
        const float p = std::clamp(texel[axis], 0.0f, float(edge - 1));
        a[axis] = static_cast<uint32_t>(p);
        b[axis] = std::min(a[axis] + 1, edge - 1);
        weight[axis] = p - float(a[axis]);
    }
    const auto sampleXY = [&](uint32_t z) {
        const auto read = [&](uint32_t x, uint32_t y) { return float(values[linearVoxelIndex(x, y, z, edge)]) / 255.0f; };
        return std::lerp(std::lerp(read(a[0], a[1]), read(b[0], a[1]), weight[0]),
            std::lerp(read(a[0], b[1]), read(b[0], b[1]), weight[0]), weight[1]);
    };
    return std::lerp(sampleXY(a[2]), sampleXY(b[2]), weight[2]);
}

struct R8TraceWitness {
    float hitDistance = -1.0f;
    bool coarseOverflowHit = false;
    bool unavailable = false;
    bool sampledOverflow = false;
};

R8TraceWitness traceR8Witness(const debug::DebugCapture& capture, uint32_t atlasPageEdge, float x, float y)
{
    constexpr float halfVoxel = 0.5f * kVoxelSize;
    constexpr float innerExtent = kExtent - halfVoxel;
    if (std::max(std::abs(x), std::abs(y)) >= innerExtent) { return {}; }
    const auto& table = capture.artifacts[0].bytes;
    const auto& fine = capture.artifacts[2].bytes;
    const auto& coarse = capture.artifacts[3].bytes;
    constexpr float originZ = kCameraDistance - kNearPlane;
    float t = originZ - innerExtent;
    bool sampledOverflow = false;
    for (uint32_t step = 0; step < 256 && t <= originZ + innerExtent; ++step) {
        const std::array world{x, y, originZ - t};
        std::array<float, 3> coarseTexel{}, fineGrid{};
        std::array<uint32_t, 3> virtualPage{};
        for (size_t axis = 0; axis < world.size(); ++axis) {
            coarseTexel[axis] = (world[axis] + kExtent) / (2.0f * kExtent) * kCoarseEdge - 0.5f;
            fineGrid[axis] = (world[axis] + kExtent) / kVoxelSize;
            virtualPage[axis] = std::min(uint32_t(fineGrid[axis] / kLogicalPageWidth), kPageTableEdge - 1);
        }
        const float encoded = sampleR8(coarse, kCoarseEdge, coarseTexel);
        const uint32_t page = readGdfElement<uint32_t>(table,
            linearVoxelIndex(virtualPage[0], virtualPage[1], virtualPage[2], kPageTableEdge));
        if (page == kOverflowPage) {
            if (encoded >= 1.0f) { return {-1.0f, false, true, sampledOverflow}; }
            sampledOverflow = true;
        }
        float distance = (2.0f * encoded - 1.0f) * (kCoarseFactor * kInfluence);
        if (page < kOverflowPage && encoded < kCoarseTransition) {
            const std::array physicalPage{page % atlasPageEdge, (page / atlasPageEdge) % atlasPageEdge,
                page / (atlasPageEdge * atlasPageEdge)};
            std::array<float, 3> atlasTexel;
            for (size_t axis = 0; axis < world.size(); ++axis) {
                atlasTexel[axis] = float(physicalPage[axis] * kPhysicalPageWidth) + fineGrid[axis] -
                    float(virtualPage[axis] * kLogicalPageWidth);
            }
            distance = (2.0f * sampleR8(fine, atlasPageEdge * kPhysicalPageWidth, atlasTexel) - 1.0f) * kInfluence;
        } else if (page == kNegativePage && encoded < kCoarseTransition) {
            distance = -kInfluence;
        }
        if (distance < halfVoxel) {
            return {t + distance - halfVoxel, page == kOverflowPage, false, sampledOverflow};
        }
        t += std::max(distance, halfVoxel);
    }
    return {-1.0f, false, false, sampledOverflow};
}

std::string checkR8Trace(const render::RenderGraphPreviewRenderer& preview, const std::vector<std::byte>& denseTrace,
    const debug::DebugCapture& capture, uint32_t atlasPageEdge, bool overflow, bool noResidentPages)
{
    const auto& bytes = preview.readbackBytes();
    using Pixel = std::array<float, 4>;
    constexpr size_t byteCount = size_t(kWidth) * kHeight * sizeof(Pixel);
    if (preview.readbackFormat() != render::Format::RGBA32Sfloat || bytes.size() != byteCount ||
        denseTrace.size() != byteCount) {
        return "Expected matching RGBA32F GDF trace readbacks";
    }
    uint32_t refinedHits = 0;
    uint32_t fewerAirSteps = 0;
    uint32_t resolvedOverflowRays = 0;
    uint32_t coarseOverflowHits = 0;
    uint32_t unavailable = 0;
    for (size_t i = 0; i < size_t(kWidth) * kHeight; ++i) {
        const auto actual = readGdfElement<Pixel>(bytes, i);
        const auto expected = readGdfElement<Pixel>(denseTrace, i);
        if (!std::ranges::all_of(actual, [](float value) { return std::isfinite(value); }) ||
            (expected[2] == 0.0f) != (actual[2] == 0.0f) ||
            actual[1] < 0.0f || actual[1] > 256.0f || actual[2] < 0.0f || actual[2] > 3.0f ||
            (actual[2] != 2.0f && (actual[0] != -1.0f || actual[3] != 0.0f))) {
            return "Invalid R8 GDF trace at pixel " + std::to_string(i);
        }
        if (!overflow) {
            if (actual[2] == 3.0f) { return "Fully covered R8 GDF reported unavailable"; }
            refinedHits += uint32_t(actual[2] == 2.0f && expected[2] == 2.0f &&
                std::abs(actual[0] - expected[0]) <= 2.0f * kFineQuantum);
            fewerAirSteps += uint32_t(actual[2] == 1.0f && expected[2] == 1.0f && actual[1] + 1 < expected[1]);
            continue;
        }
        const float x = (float(i % kWidth) + 0.5f - 0.5f * float(kWidth)) * kOrthoHeight / float(kHeight);
        const float y = (0.5f * float(kHeight) - float(i / kWidth) - 0.5f) * kOrthoHeight / float(kHeight);
        const auto witness = traceR8Witness(capture, atlasPageEdge, x, y);
        unavailable += uint32_t(actual[2] == 3.0f && witness.unavailable);
        const bool matchesWitness = witness.hitDistance >= 0.0f ?
            actual[2] == 2.0f && std::abs(actual[0] - witness.hitDistance) <= 0.01f :
            actual[2] == 1.0f;
        resolvedOverflowRays += uint32_t(witness.sampledOverflow && !witness.unavailable && matchesWitness);
        coarseOverflowHits += uint32_t(actual[2] == 2.0f && witness.coarseOverflowHit &&
            std::abs(actual[0] - witness.hitDistance) <= 0.01f);
    }
    if (!overflow && (refinedHits < 100 || fewerAirSteps < 100)) {
        return "R8 trace did not exercise both fine-precision hits and long coarse steps in clear air";
    }
    if (overflow && resolvedOverflowRays == 0) { return "The coarse mip did not resolve any overflow rays"; }
    if (noResidentPages && (coarseOverflowHits == 0 || unavailable == 0)) {
        return "Zero page budget did not exercise both a coarse fallback hit and unavailable coverage. Hits " +
            std::to_string(coarseOverflowHits) + ", unavailable " + std::to_string(unavailable);
    }
    return {};
}

class GlobalDistanceFieldR8Test final : public RHITest {
public:
    GlobalDistanceFieldR8Test()
    {
        name = "global_distance_field_r8_coarse_mip";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        constexpr uint32_t kFullCapacity = 216;
        constexpr uint32_t kFullAtlasPageEdge = 6;
        const std::filesystem::path root(PROJECT_SOURCE_DIR);
        const auto fixtureDirectory = context.outputDirectory / "R8GlobalDistanceField";
        std::error_code directoryError;
        std::filesystem::create_directories(fixtureDirectory, directoryError);
        if (directoryError) { return RHITestResult::fail(directoryError.message()); }

        scene::SceneDocument document;
        std::array<std::string, 2> paths;
        std::string log;
        if (!prepareFieldPaths(fixtureDirectory, document, paths, log)) { return RHITestResult::fail(log); }
        RenderGraph denseGraph, quantizedGraph;
        if (!loadRenderGraphFromFile(root / "Pipelines/Samples/global_distance_field.metallic_graph.json", denseGraph, log) ||
            !loadRenderGraphFromFile(root / "Pipelines/Samples/global_distance_field_r8.metallic_graph.json", quantizedGraph, log)) {
            return RHITestResult::fail(log);
        }
        const auto* denseProducer = denseGraph.findNode("GlobalDistanceField");
        const auto* quantizedProducer = quantizedGraph.findNode(kSparsePass);
        if (!denseProducer || !quantizedProducer) { return RHITestResult::fail("Missing dense or R8 GDF producer"); }
        const auto quantizedId = quantizedProducer->id;
        const nlohmann::json properties{{"distanceFieldPaths", paths}, {"center", {0.0f, 0.0f, 0.0f}},
            {"extent", kExtent}, {"resolution", kResolution}};
        RenderView view;
        if (!configureGdfTestView(view)) { return RHITestResult::fail("Invalid R8 GDF test view"); }
        bench::ValidationRecorder validation;
        GdfCapturePreview capturePreview;
        auto& preview = capturePreview.preview;
        auto& inspector = capturePreview.inspector;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        if (!preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink())) {
            return RHITestResult::fail("R8 GDF preview setup: " + preview.lastLog());
        }
        preview.setRawReadbackEnabled(true);
        if (!denseGraph.setNodeProperties(denseProducer->id, properties) ||
            !preview.render(denseGraph, kWidth, kHeight, kColorOutput)) {
            return RHITestResult::fail("R8 GDF dense reference setup: " + preview.lastLog());
        }
        const auto dense = captureAfterPass(preview, inspector, denseGraph, "GlobalDistanceField",
            debug::DebugValue::array({textureCapture("GlobalDistanceField.distanceField", kTexels, kTexels, kTexels)}),
            kTraceOutput, log);
        if (!dense) { return RHITestResult::fail(log); }
        if (dense->artifacts.size() != 1) { return RHITestResult::fail("Expected one dense reference artifact"); }
        const auto denseTrace = preview.readbackBytes();

        bool r8StorageUnsupported = false;
        const auto testR8 = [&](uint32_t capacity, uint32_t atlasPageEdge, bool empty) -> std::string {
            auto quantizedProperties = properties;
            quantizedProperties["quantized"] = true;
            quantizedProperties["maxPhysicalPages"] = capacity;
            if (empty) { quantizedProperties["distanceFieldPaths"] = nlohmann::json::array(); }
            if (!quantizedGraph.setNodeProperties(quantizedId, std::move(quantizedProperties))) {
                return "Failed to set the R8 GDF properties";
            }
            const auto rendered = preview.render(quantizedGraph, kWidth, kHeight, kColorOutput);
            if (!rendered) {
                r8StorageUnsupported = rendered.error() == Error::Unsupported &&
                    preview.lastLog().find("requires R8 storage image support") != std::string::npos;
                return "R8 GDF composition preview: " + preview.lastLog();
            }
            if (capacity == kFullCapacity && !saveRgba8Png(context.outputDirectory / "GlobalDistanceFieldR8.png",
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), kWidth, kHeight, log)) {
                return log;
            }
            auto resources = debug::DebugValue::array({
                textureCapture("GlobalDistanceFieldPages.pageTable", kPageTableEdge, kPageTableEdge, kPageTableEdge),
                {{"id", "GlobalDistanceFieldPages.statistics"}, {"layout", "u32"}, {"count", 5}},
            });
            if (!empty) {
                const uint32_t atlasEdge = atlasPageEdge * kPhysicalPageWidth;
                resources.push_back(textureCapture("GlobalDistanceFieldPages.pageAtlas", atlasEdge, atlasEdge, atlasEdge));
            }
            resources.push_back(textureCapture("GlobalDistanceFieldPages.coarseMip", kCoarseEdge, kCoarseEdge, kCoarseEdge));
            const auto capture = captureAfterPass(preview, inspector, quantizedGraph, kSparsePass,
                std::move(resources), kSparseTrace, log);
            if (!capture) { return log; }
            PageCounts counts;
            if (const auto error = checkSparsePages(*capture, empty ? nullptr : &dense->artifacts[0],
                    capacity, atlasPageEdge, counts, true); !error.empty()) {
                return error;
            }
            if (const auto error = checkCoarseMip(*capture, atlasPageEdge, empty); !error.empty()) { return error; }
            const bool overflow = !empty && capacity < kFullCapacity;
            if ((counts.overflow > 0) != overflow) { return "R8 GDF page budget did not reach the requested test state"; }
            if (!overflow) {
                TraceCounts traceCounts;
                if (const auto error = checkTrace(preview, empty, traceCounts); !error.empty()) { return error; }
            }
            return empty ? std::string() : checkR8Trace(preview, denseTrace, *capture, atlasPageEdge, overflow, capacity == 0);
        };
        if (const auto error = testR8(kFullCapacity, kFullAtlasPageEdge, false); !error.empty()) {
            return r8StorageUnsupported ? RHITestResult::skip(error) : RHITestResult::fail(error);
        }
        if (const auto error = testR8(64, 4, false); !error.empty()) { return RHITestResult::fail(error); }
        // The physical allocation order is unspecified. With zero resident pages,
        // the negative constant pages still provide a deterministic coarse seed.
        if (const auto error = testR8(0, 1, false); !error.empty()) { return RHITestResult::fail(error); }
        if (const auto error = testR8(1, 1, true); !error.empty()) { return RHITestResult::fail(error); }
        if (validation.failed()) { return RHITestResult::fail("R8 GDF Vulkan validation: " + validation.snapshot().dump()); }
        return RHITestResult::pass("R8 atlas, five-step UE coarse propagation, fine refinement and overflow traces validated.");
    }
};

METALLIC_REGISTER_RHI_TEST(GlobalDistanceFieldR8Test);

constexpr uint32_t kClipmapCount = 3;
constexpr uint32_t kClipmapResolution = 56;
constexpr uint32_t kClipmapPageEdge = kClipmapResolution / kLogicalPageWidth;
constexpr uint32_t kClipmapMipEdge = kClipmapResolution / kCoarseFactor;
constexpr uint32_t kClipmapCapacity = 512;
constexpr uint32_t kClipmapAtlasPages = 8;
constexpr uint32_t kClipmapAtlasEdge = kClipmapAtlasPages * kPhysicalPageWidth;
constexpr float kClipmapOrthoHeight = 8.4f;

std::string checkClipmapResources(const debug::DebugCapture& capture, const render::ViewCamera& camera,
    uint32_t firstPopulatedLevel = 0)
{
    if (capture.artifacts.size() != 5) { return "Expected five clipmap capture resources"; }
    const auto& table = capture.artifacts[0];
    const auto& statistics = capture.artifacts[1];
    const auto& atlas = capture.artifacts[2];
    const auto& domains = capture.artifacts[3];
    if (!matchesVolumeLayout(table, kClipmapPageEdge, "u32", kClipmapCount) ||
        statistics.bytes.size() != 5 * sizeof(uint32_t) ||
        !matchesVolumeLayout(atlas, kClipmapAtlasEdge, "R8") ||
        domains.bytes.size() != kClipmapCount * 8 * sizeof(float)) {
        return "Clipmap capture dimensions or formats differ from the packed GPU layout";
    }

    std::array<bool, kClipmapCapacity> allocated{};
    PageCounts counts;
    for (uint32_t level = 0; level < kClipmapCount; ++level) {
        const float extent = std::ldexp(kExtent, int(level));
        const float voxelSize = 2.0f * extent / float(kClipmapResolution);
        const float pageSize = voxelSize * float(kLogicalPageWidth);
        const auto domain = readGdfElement<std::array<float, 8>>(domains.bytes, level);
        for (size_t axis = 0; axis < camera.eye.size(); ++axis) {
            const float center = std::floor(camera.eye[axis] / pageSize + 0.5f) * pageSize;
            if (std::abs(domain[axis] - (center - extent)) > 1.0e-6f ||
                std::abs(domain[4 + axis] - (center + extent)) > 1.0e-6f) {
                return "Clipmap level " + std::to_string(level) + " did not follow the page-aligned camera origin";
            }
        }
        if (std::abs(domain[3] - voxelSize) > 1.0e-6f || std::abs(domain[7] - 4.0f * voxelSize) > 1.0e-6f) {
            return "Incorrect clipmap level voxel size or influence radius";
        }
        constexpr size_t pagesPerLevel = size_t(kClipmapPageEdge) * kClipmapPageEdge * kClipmapPageEdge;
        uint32_t levelResident = 0;
        for (size_t i = 0; i < pagesPerLevel; ++i) {
            const uint32_t page = readGdfElement<uint32_t>(table.bytes, size_t(level) * pagesPerLevel + i);
            if (level < firstPopulatedLevel && page != kPositivePage) {
                return "A clipmap that moved past the scene contains a stale virtual page";
            }
            if (page < kClipmapCapacity) {
                if (allocated[page]) { return "Clipmap pages alias a shared physical atlas allocation"; }
                allocated[page] = true;
                ++levelResident;
            } else if (page == kPositivePage) {
                ++counts.positive;
            } else if (page == kNegativePage) {
                ++counts.negative;
            } else {
                return "Clipmap page table contains an overflow or invalid ID with a full page budget";
            }
        }
        if (level >= firstPopulatedLevel && levelResident == 0) {
            return "Missing resident pages at clipmap level " + std::to_string(level);
        }
        counts.resident += levelResident;
    }
    const std::array expectedStatistics{counts.resident, counts.resident, counts.positive, counts.negative, 0u};
    for (size_t i = 0; i < expectedStatistics.size(); ++i) {
        if (readGdfElement<uint32_t>(statistics.bytes, i) != expectedStatistics[i]) {
            return "Clipmap statistics do not match the packed page table";
        }
    }
    return checkCoarseMip(capture, kClipmapAtlasPages, false, kClipmapResolution, kClipmapCount);
}

std::string checkClipmapRay(const render::RenderGraphPreviewRenderer& preview, const render::ViewCamera& camera,
    float worldX, float worldY, uint32_t classification, int hitLevel = -1)
{
    const auto& bytes = preview.readbackBytes();
    if (preview.readbackFormat() != render::Format::RGBA32Sfloat ||
        bytes.size() != size_t(kWidth) * kHeight * 4 * sizeof(float)) {
        return "Expected RGBA32F clipmap trace data";
    }
    const float pixelsPerMeter = float(kHeight) / camera.orthoHeight;
    const auto x = uint32_t(float(kWidth) * 0.5f + (worldX - camera.eye[0]) * pixelsPerMeter);
    const auto y = uint32_t(float(kHeight) * 0.5f - (worldY - camera.eye[1]) * pixelsPerMeter);
    const auto trace = readGdfElement<std::array<float, 4>>(bytes, size_t(y) * kWidth + x);
    const bool hit = hitLevel >= 0;
    const float expectedDistance = hit ? camera.eye[2] - camera.nearPlane - kBoxes[0].halfExtent.z : -1.0f;
    const float tolerance = hit ? 3.0f * std::ldexp(2.0f * kExtent / float(kClipmapResolution), hitLevel) : 0.0f;
    if (!std::ranges::all_of(trace, [](float value) { return std::isfinite(value); }) ||
        std::abs(trace[0] - expectedDistance) > tolerance || trace[2] != float(classification) ||
        trace[3] != float(hitLevel) || trace[1] < 0.0f || trace[1] > 256.0f * kClipmapCount ||
        (classification == 0u && trace[1] != 0.0f) ||
        ((classification == 1u || hit || classification == 4u) && trace[1] == 0.0f)) {
        return "Clipmap ray at (" + std::to_string(worldX) + ", " + std::to_string(worldY) +
            ") expected class " + std::to_string(classification) + ", level " + std::to_string(hitLevel) +
            ", distance " + std::to_string(expectedDistance) + ", read " + std::to_string(trace[2]) +
            ", " + std::to_string(trace[3]) + ", " + std::to_string(trace[0]) + ", steps " + std::to_string(trace[1]);
    }
    return {};
}

class GlobalDistanceFieldClipmapTest final : public RHITest {
public:
    GlobalDistanceFieldClipmapTest()
    {
        name = "global_distance_field_camera_clipmaps";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto fixtureDirectory = context.outputDirectory / "GlobalDistanceFieldClipmaps";
        std::error_code directoryError;
        std::filesystem::create_directories(fixtureDirectory, directoryError);
        if (directoryError) { return RHITestResult::fail(directoryError.message()); }
        scene::SceneDocument document;
        std::array<std::string, 2> paths;
        std::string log;
        if (!prepareFieldPaths(fixtureDirectory, document, paths, log)) { return RHITestResult::fail(log); }
        const std::filesystem::path root(PROJECT_SOURCE_DIR);
        RenderGraph graph;
        if (!loadRenderGraphFromFile(root / "Pipelines/Samples/global_distance_field_clipmaps.metallic_graph.json", graph, log)) {
            return RHITestResult::fail(log);
        }
        const auto* producer = graph.findNode(kSparsePass);
        if (!producer) { return RHITestResult::fail("Missing clipmap GDF producer"); }
        const auto producerId = producer->id;
        nlohmann::json properties{{"distanceFieldPaths", paths}, {"extent", kExtent},
            {"resolution", kClipmapResolution - 1}, {"quantized", true},
            {"clipmapCount", kClipmapCount}, {"maxPhysicalPages", kClipmapCapacity}};
        if (!graph.setNodeProperties(producerId, properties)) { return RHITestResult::fail("Invalid clipmap graph"); }

        ViewCamera camera;
        camera.nearPlane = kNearPlane;
        camera.farPlane = 10.0f;
        camera.orthographic = true;
        camera.orthoHeight = kClipmapOrthoHeight;
        RenderView view;
        const auto setEye = [&](std::array<float, 3> eye) {
            camera.eye = eye;
            camera.center = {eye[0], eye[1], eye[2] - 1.0f};
            return view.setCamera(camera);
        };
        if (!view.setRenderResolution(kWidth, kHeight) || !setEye({0.125f, -0.125f, 1.4f})) {
            return RHITestResult::fail("Invalid clipmap test camera");
        }

        bench::ValidationRecorder validation;
        GdfCapturePreview capturePreview;
        auto& preview = capturePreview.preview;
        auto& inspector = capturePreview.inspector;
        preview.bindRuntimeScene(&document);
        preview.bindRenderView(&view);
        if (!preview.initialize(context.enableValidation, false, false, context.enableValidation, validation.sink())) {
            return RHITestResult::fail("Clipmap GDF preview setup: " + preview.lastLog());
        }
        preview.setRawReadbackEnabled(true);
        if (const auto rendered = preview.render(graph, kWidth, kHeight, kColorOutput); !rendered) {
            const auto error = "Clipmap GDF composition: " + preview.lastLog();
            if (rendered.error() == Error::Unsupported &&
                preview.lastLog().find("requires R8 storage image support") != std::string::npos) {
                return RHITestResult::skip(error);
            }
            return RHITestResult::fail(error);
        }
        if (!saveRgba8Png(context.outputDirectory / "GlobalDistanceFieldClipmaps.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), kWidth, kHeight, log)) {
            return RHITestResult::fail(log);
        }

        const auto captureClipmaps = [&] {
            return captureAfterPass(preview, inspector, graph, kSparsePass, debug::DebugValue::array({
                textureCapture("GlobalDistanceFieldPages.pageTable", kClipmapPageEdge, kClipmapPageEdge,
                    kClipmapPageEdge * kClipmapCount),
                {{"id", "GlobalDistanceFieldPages.statistics"}, {"layout", "u32"}, {"count", 5}},
                textureCapture("GlobalDistanceFieldPages.pageAtlas", kClipmapAtlasEdge, kClipmapAtlasEdge, kClipmapAtlasEdge),
                {{"id", "GlobalDistanceFieldPages.domain"}, {"layout", "f32"}, {"count", kClipmapCount * 8}},
                textureCapture("GlobalDistanceFieldPages.coarseMip", kClipmapMipEdge, kClipmapMipEdge,
                    kClipmapMipEdge * kClipmapCount),
            }), kSparseTrace, log);
        };
        const auto initial = captureClipmaps();
        if (!initial) { return RHITestResult::fail(log); }
        if (const auto error = checkClipmapResources(*initial, camera); !error.empty()) { return RHITestResult::fail(error); }
        if (const auto error = checkClipmapRay(preview, camera, kBoxes[0].center.x, 0.0f, 2, 0); !error.empty()) {
            return RHITestResult::fail(error);
        }

        if (!setEye({0.15f, -0.15f, 1.45f})) { return RHITestResult::fail("Invalid sub-page camera movement"); }
        const auto subPage = captureClipmaps();
        if (!subPage) { return RHITestResult::fail(log); }
        if (const auto error = checkClipmapResources(*subPage, camera); !error.empty()) { return RHITestResult::fail(error); }
        if (initial->artifacts[3].bytes != subPage->artifacts[3].bytes ||
            initial->artifacts[4].bytes != subPage->artifacts[4].bytes) {
            return RHITestResult::fail("Sub-page camera movement changed the clipmap lattice or coarse distances");
        }
        if (const auto error = checkClipmapRay(preview, camera, kBoxes[0].center.x, 0.0f, 2, 0); !error.empty()) {
            return RHITestResult::fail(error);
        }

        if (!setEye({0.25f, -0.25f, 2.1875f})) { return RHITestResult::fail("Invalid cross-page camera movement"); }
        const auto moved = captureClipmaps();
        if (!moved) { return RHITestResult::fail(log); }
        if (const auto error = checkClipmapResources(*moved, camera, 1); !error.empty()) { return RHITestResult::fail(error); }
        if (const auto error = checkClipmapRay(preview, camera, kBoxes[0].center.x, 0.0f, 2, 1); !error.empty()) {
            return RHITestResult::fail(error);
        }
        if (const auto error = checkClipmapRay(preview, camera, 0.0f, 1.5f, 4); !error.empty()) {
            return RHITestResult::fail(error);
        }
        if (const auto error = checkClipmapRay(preview, camera, 6.25f, 0.0f, 0); !error.empty()) {
            return RHITestResult::fail(error);
        }

        camera.farPlane = 3.0f;
        if (!view.setCamera(camera) || !preview.render(graph, kWidth, kHeight, kSparseTrace)) {
            return RHITestResult::fail("Clipmap full-coverage ray: " + preview.lastLog());
        }
        if (const auto error = checkClipmapRay(preview, camera, 0.0f, 1.5f, 1); !error.empty()) {
            return RHITestResult::fail(error);
        }
        camera.farPlane = 10.0f;
        if (!setEye({0.25f, -0.25f, 4.8f}) || !preview.render(graph, kWidth, kHeight, kSparseTrace)) {
            return RHITestResult::fail("Clipmap far-level trace: " + preview.lastLog());
        }
        if (const auto error = checkClipmapRay(preview, camera, kBoxes[0].center.x, 0.0f, 2, 2); !error.empty()) {
            return RHITestResult::fail(error);
        }

        camera.nearPlane = 1.8f;
        properties["maxPhysicalPages"] = 0;
        if (!setEye({0.25f, -0.25f, 2.1875f}) || !graph.setNodeProperties(producerId, properties) ||
            !preview.render(graph, kWidth, kHeight, kSparseTrace)) {
            return RHITestResult::fail("Clipmap overflow trace: " + preview.lastLog());
        }
        if (const auto error = checkClipmapRay(preview, camera, kBoxes[2].center.x, 0.0f, 3); !error.empty()) {
            return RHITestResult::fail(error);
        }
        if (validation.failed()) { return RHITestResult::fail("Clipmap GDF Vulkan validation: " + validation.snapshot().dump()); }
        return RHITestResult::pass("Camera page snapping, packed mip isolation, three hit levels and four miss states validated.");
    }
};

METALLIC_REGISTER_RHI_TEST(GlobalDistanceFieldClipmapTest);

constexpr uint32_t kPersistentClipmapCount = 3;
constexpr uint32_t kPersistentResolution = 28;
constexpr float kPersistentExtent = 1.75f;
constexpr uint32_t kPersistentPageEdge = kPersistentResolution / kLogicalPageWidth;
constexpr uint32_t kPersistentMipEdge = kPersistentResolution / kCoarseFactor;
constexpr uint32_t kPersistentAtlasPages = 6;
constexpr uint32_t kPersistentAtlasEdge = kPersistentAtlasPages * kPhysicalPageWidth;
constexpr uint32_t kPersistentCapacity = kPersistentAtlasPages * kPersistentAtlasPages * kPersistentAtlasPages;
constexpr uint32_t kPersistentPageSlab = kPersistentPageEdge * kPersistentPageEdge;
constexpr uint32_t kPersistentVirtualPages = kPersistentClipmapCount * kPersistentPageSlab * kPersistentPageEdge;
constexpr size_t kObjectGridWordsPerPage = 64 * 4;
constexpr std::string_view kPersistentPass = "PersistentGDF";
constexpr std::string_view kFullReferencePass = "FullGDF";

debug::DebugValue persistentCaptureResources(std::string_view pass, bool persistent)
{
    const auto field = [&](std::string_view name) { return render::makeRenderGraphFieldName(pass, name); };
    auto resources = debug::DebugValue::array({
        textureCapture(field("pageTable"), kPersistentPageEdge, kPersistentPageEdge,
            kPersistentPageEdge * kPersistentClipmapCount),
        textureCapture(field("pageAtlas"), kPersistentAtlasEdge, kPersistentAtlasEdge, kPersistentAtlasEdge),
        textureCapture(field("coarseMip"), kPersistentMipEdge, kPersistentMipEdge,
            kPersistentMipEdge * kPersistentClipmapCount),
        {{"id", field("domain")}, {"layout", "f32"}, {"count", kPersistentClipmapCount * 8}},
        {{"id", field("statistics")}, {"layout", "u32"}, {"count", 5}},
        {{"id", field("objectGrid")}, {"layout", "u32"}, {"count", kPersistentCapacity * kObjectGridWordsPerPage}},
    });
    if (persistent) {
        resources.push_back({{"id", field("updateStatistics")}, {"layout", "u32"}, {"count", 4}});
        resources.push_back({{"id", field("freeCount")}, {"layout", "u32"}, {"count", 1}});
    }
    return resources;
}

std::string checkPersistentClipmaps(const debug::DebugCapture& persistent, const debug::DebugCapture& reference,
    const std::array<uint32_t, 4>& expectedUpdate)
{
    if (persistent.artifacts.size() != 8 || reference.artifacts.size() != 6 ||
        persistent.artifacts[6].bytes.size() != sizeof(expectedUpdate) ||
        persistent.artifacts[7].bytes.size() != sizeof(uint32_t)) {
        return "Expected distance fields, object grids, update counters and free page count";
    }
    const auto update = readGdfElement<std::array<uint32_t, 4>>(persistent.artifacts[6].bytes, 0);
    if (update != expectedUpdate) {
        return "Unexpected GDF update counters [full, updated, preserved, coarse levels]: " +
            nlohmann::json(update).dump() + ", expected " + nlohmann::json(expectedUpdate).dump();
    }
    for (const auto* capture : {&persistent, &reference}) {
        const auto& fields = capture->artifacts;
        if (!matchesVolumeLayout(fields[0], kPersistentPageEdge, "u32", kPersistentClipmapCount) ||
            !matchesVolumeLayout(fields[1], kPersistentAtlasEdge, "R8") ||
            !matchesVolumeLayout(fields[2], kPersistentMipEdge, "R8", kPersistentClipmapCount) ||
            fields[3].bytes.size() != kPersistentClipmapCount * 8 * sizeof(float) ||
            fields[4].bytes.size() != 5 * sizeof(uint32_t) ||
            fields[5].bytes.size() != kPersistentCapacity * kObjectGridWordsPerPage * sizeof(uint32_t)) {
            return "Unexpected persistent clipmap capture layout";
        }
    }
    const auto& actual = persistent.artifacts;
    const auto& full = reference.artifacts;
    if (actual[2].bytes != full[2].bytes || actual[3].bytes != full[3].bytes || actual[4].bytes != full[4].bytes) {
        return "Persistent GDF coarse mip, domains, or page statistics differ from a full rebuild";
    }
    const uint32_t residentCount = readGdfElement<uint32_t>(actual[4].bytes, 1);
    const uint32_t freeCount = readGdfElement<uint32_t>(actual[7].bytes, 0);
    if (residentCount > kPersistentCapacity || freeCount != kPersistentCapacity - residentCount) {
        return "The persistent page pool lost or duplicated a physical slot";
    }
    std::array<bool, kPersistentCapacity> physicalPages{};
    for (uint32_t virtualPage = 0; virtualPage < kPersistentVirtualPages; ++virtualPage) {
        const uint32_t page = readGdfElement<uint32_t>(actual[0].bytes, virtualPage);
        const uint32_t fullPage = readGdfElement<uint32_t>(full[0].bytes, virtualPage);
        if (page == kPositivePage || page == kNegativePage) {
            if (page != fullPage) { return "Persistent GDF constant page differs at " + std::to_string(virtualPage); }
            continue;
        }
        if (page >= kPersistentCapacity || fullPage >= kPersistentCapacity || physicalPages[page]) {
            return "Persistent GDF physical page missing or aliased at " + std::to_string(virtualPage);
        }
        physicalPages[page] = true;
        for (uint32_t z = 0; z < kPhysicalPageWidth; ++z) {
            for (uint32_t y = 0; y < kPhysicalPageWidth; ++y) {
                const size_t offset = atlasVoxelIndex(page, 0, y, z, kPersistentAtlasPages);
                const size_t fullOffset = atlasVoxelIndex(fullPage, 0, y, z, kPersistentAtlasPages);
                if (std::memcmp(actual[1].bytes.data() + offset, full[1].bytes.data() + fullOffset, kPhysicalPageWidth)) {
                    return "Persistent GDF R8 texels differ at virtual page " + std::to_string(virtualPage);
                }
            }
        }
        constexpr size_t objectGridBytesPerPage = kObjectGridWordsPerPage * sizeof(uint32_t);
        if (std::memcmp(actual[5].bytes.data() + page * objectGridBytesPerPage,
                full[5].bytes.data() + fullPage * objectGridBytesPerPage, objectGridBytesPerPage)) {
            return "Persistent GDF object cells differ at virtual page " + std::to_string(virtualPage);
        }
    }
    return {};
}

std::string checkPersistentScrollPageIds(const debug::DebugCapture& before, const debug::DebugCapture& after)
{
    const auto& previousTable = before.artifacts[0].bytes;
    const auto& currentTable = after.artifacts[0].bytes;
    uint32_t preservedResidentPages = 0;
    for (uint32_t z = 0; z < kPersistentPageEdge * kPersistentClipmapCount; ++z) {
        const uint32_t scrollX = z < kPersistentPageEdge ? 1u : 0u;
        for (uint32_t y = 0; y < kPersistentPageEdge; ++y) {
            for (uint32_t x = 0; x < kPersistentPageEdge - scrollX; ++x) {
                const uint32_t previous = readGdfElement<uint32_t>(previousTable,
                    linearVoxelIndex(x + scrollX, y, z, kPersistentPageEdge));
                const uint32_t current = readGdfElement<uint32_t>(currentTable,
                    linearVoxelIndex(x, y, z, kPersistentPageEdge));
                if (previous != current) { return "A retained virtual page changed its physical ID or sentinel"; }
                preservedResidentPages += uint32_t(scrollX != 0 && current < kPersistentCapacity);
            }
        }
    }
    return preservedResidentPages == 0 ? "The scroll fixture did not retain a resident fine page" : std::string();
}

class GlobalDistanceFieldPersistentPagesTest final : public RHITest {
public:
    GlobalDistanceFieldPersistentPagesTest()
    {
        name = "global_distance_field_persistent_pages";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto fixtureDirectory = context.outputDirectory / "PersistentGlobalDistanceField";
        std::error_code directoryError;
        std::filesystem::create_directories(fixtureDirectory, directoryError);
        if (directoryError) { return RHITestResult::fail(directoryError.message()); }
        scene::SceneDocument document;
        std::array<std::string, 2> paths;
        std::string log;
        if (!prepareFieldPaths(fixtureDirectory, document, paths, log)) { return RHITestResult::fail(log); }

        bench::ValidationRecorder validation;
        auto device = bench::createTestDevice(context, {.applicationName = "Persistent GDF pages",
            .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true, .validationSink = validation.sink()});
        if (!device) { return RHITestResult::fail("Could not initialize the persistent GDF device"); }
        auto& gpu = **device;
        if (!gpu.capabilities().shaderStorageImageExtendedFormats) {
            return RHITestResult::skip("Persistent GDF requires R8 storage image support");
        }

        RenderGraph graph;
        nlohmann::json properties{{"distanceFieldPaths", paths}, {"extent", kPersistentExtent},
            {"resolution", kPersistentResolution}, {"quantized", true}, {"clipmapCount", kPersistentClipmapCount},
            {"maxPhysicalPages", kPersistentCapacity}, {"buildObjectGrid", true}};
        if (!graph.addNode("GlobalDistanceFieldPagesPass", std::string(kFullReferencePass), properties) ||
            !graph.markOutput(makeRenderGraphFieldName(kFullReferencePass, "coarseMip"))) {
            return RHITestResult::fail("Could not configure the full-rebuild GDF reference");
        }
        properties["persistentPages"] = true;
        if (!graph.addNode("GlobalDistanceFieldPagesPass", std::string(kPersistentPass), std::move(properties)) ||
            !graph.markOutput(makeRenderGraphFieldName(kPersistentPass, "coarseMip"))) {
            return RHITestResult::fail("Could not configure the persistent GDF candidate");
        }

        ViewCamera camera;
        camera.nearPlane = kNearPlane;
        camera.farPlane = 12.0f;
        camera.orthographic = true;
        camera.orthoHeight = kClipmapOrthoHeight;
        RenderView view;
        const auto moveCamera = [&](float x, float y = 0.0f) {
            camera.eye = {x, y, 1.4f};
            camera.center = {x, y, 0.4f};
            return view.setCamera(camera);
        };
        if (!view.setRenderResolution(kWidth, kHeight) || !moveCamera(0.0f)) {
            return RHITestResult::fail("Invalid persistent GDF test view");
        }
        HistoryResourceManager history;
        if (!history.initialize(gpu)) { return RHITestResult::fail("Could not initialize graph history"); }
        RenderDebugRuntime inspector{{}, RenderDebugDetail::Resources};
        RenderGraphExecutor executor;
        executor.bindRuntimeScene(&document);
        executor.bindRenderView(&view);
        executor.setDebugObserver(&inspector);
        if (!executor.compile(gpu, graph, kWidth, kHeight, RenderGraphCompileOptions{.enableDebugReadback = true}, log)) {
            return RHITestResult::fail("Persistent GDF compile: " + log);
        }

        auto* queue = gpu.getQueue(QueueType::Graphics);
        {
            // Cancel the first recording before its initial image transitions
            // or page state have reached the queue.
            auto pool = gpu.createCommandPool(*queue);
            if (!pool) { return RHITestResult::fail("Could not allocate a GDF cancellation command pool"); }
            auto commands = (*pool)->createCommandBuffer();
            RenderFrameContext frame;
            if (!commands || !frame.begin(0) || !(*commands)->begin(frame.submissionContext()) ||
                !executor.execute(**commands, &history) || !(*commands)->end()) {
                return RHITestResult::fail("Could not record the GDF frame to cancel");
            }
            frame.cancel();
            if (!executor.waitForSubmittedWork()) { return RHITestResult::fail("Could not retire the cancelled GDF frame"); }
        }
        const RenderGraphSubmitDesc submission{.graphicsQueue = queue,
            .historyResources = &history, .recordingWorkerLimit = 1};
        const auto runFrame = [&](const std::array<uint32_t, 4>& expected) -> std::shared_ptr<const debug::DebugCapture> {
            const auto request = inspector.core().dispatch({{"method", "capture.batch"}, {"params", {
                {"batches", debug::DebugValue::array({
                    {{"pass", kPersistentPass}, {"resources", persistentCaptureResources(kPersistentPass, true)}},
                    {{"pass", kFullReferencePass}, {"resources", persistentCaptureResources(kFullReferencePass, false)}},
                })},
            }}});
            if (request.value("status", "") != "ok") { log = "Persistent GDF capture: " + request.dump(); return {}; }
            if (auto result = executor.execute(submission); !result) {
                log = "Persistent GDF execute: " + std::string(resultToString(result));
                return {};
            }
            if (!executor.waitForSubmittedWork()) { log = "Persistent GDF completion failed"; return {}; }
            inspector.poll();
            const auto& jobs = request.at("result").at("jobs");
            const auto readCapture = [&](size_t i) {
                const auto id = jobs.at(i).at("job").get<std::string>();
                auto capture = inspector.core().completedCapture(id);
                if (!capture) {
                    const auto status = inspector.core().dispatch({{"method", "jobs.get"}, {"params", {{"job", id}}}});
                    log = "Persistent GDF readback failed: " + status.dump();
                }
                return capture;
            };
            auto persistent = readCapture(0);
            const auto reference = readCapture(1);
            if (!persistent || !reference) { return {}; }
            log = checkPersistentClipmaps(*persistent, *reference, expected);
            return log.empty() ? persistent : nullptr;
        };

        constexpr std::array full{1u, kPersistentVirtualPages, 0u, kPersistentClipmapCount};
        constexpr std::array unchanged{0u, 0u, kPersistentVirtualPages, 0u};
        const auto initial = runFrame(full);
        if (!initial) { return RHITestResult::fail("Initial frame: " + log); }
        const auto stationary = runFrame(unchanged);
        if (!stationary) { return RHITestResult::fail("Stationary frame: " + log); }
        if (!moveCamera(0.1f)) { return RHITestResult::fail("Could not move the camera by a fraction of a page"); }
        const auto subPage = runFrame(unchanged);
        if (!subPage) { return RHITestResult::fail("Sub-page frame: " + log); }
        if (initial->artifacts[0].bytes != stationary->artifacts[0].bytes ||
            initial->artifacts[0].bytes != subPage->artifacts[0].bytes) {
            return RHITestResult::fail("The stationary or sub-page frame changed page IDs");
        }

        if (!moveCamera(0.5f)) { return RHITestResult::fail("Could not scroll the finest clipmap by one page"); }
        const auto scrolled = runFrame({0u, kPersistentPageSlab, kPersistentVirtualPages - kPersistentPageSlab, 1u});
        if (!scrolled) { return RHITestResult::fail("One-page scroll: " + log); }
        if (const auto error = checkPersistentScrollPageIds(*subPage, *scrolled); !error.empty()) {
            return RHITestResult::fail(error);
        }

        if (!moveCamera(0.0f, 0.5f)) { return RHITestResult::fail("Could not scroll the finest clipmap diagonally"); }
        constexpr uint32_t diagonalPages = 2 * kPersistentPageSlab - kPersistentPageEdge;
        const auto diagonal = runFrame({0u, diagonalPages, kPersistentVirtualPages - diagonalPages, 1u});
        if (!diagonal) { return RHITestResult::fail("Diagonal scroll: " + log); }

        float4x4 movedBox = document.nodes()[0].worldMatrix;
        movedBox.a13 += 1.0f;
        const auto box = document.objectForNode(0).entity();
        if (!document.setObjectWorldMatrix(box, movedBox)) {
            return RHITestResult::fail("Could not move the GDF fixture's first box");
        }
        const auto moved = runFrame(full);
        if (!moved) { return RHITestResult::fail("Moved-instance frame: " + log); }
        if (moved->artifacts[2].bytes == diagonal->artifacts[2].bytes) {
            return RHITestResult::fail("Moving the instance did not change the rebuilt coarse distance field");
        }
        if (!document.setObjectVisible(box, false)) { return RHITestResult::fail("Could not hide a GDF fixture box"); }
        const auto hidden = runFrame(full);
        if (!hidden) { return RHITestResult::fail("Hidden-instance frame: " + log); }
        if (hidden->artifacts[2].bytes == moved->artifacts[2].bytes) {
            return RHITestResult::fail("Hiding the instance did not change the rebuilt coarse distance field");
        }

        if (!moveCamera(20.0f)) { return RHITestResult::fail("Could not teleport the clipmaps beyond their overlap"); }
        const auto teleported = runFrame(full);
        if (!teleported) { return RHITestResult::fail("Teleported frame: " + log); }
        if (readGdfElement<uint32_t>(teleported->artifacts[4].bytes, 2) != kPersistentVirtualPages) {
            return RHITestResult::fail("The teleported clipmaps retained stale geometry in an empty area");
        }
        if (validation.failed()) { return RHITestResult::fail("Persistent GDF Vulkan validation: " + validation.snapshot().dump()); }
        return RHITestResult::pass("Verified page reuse, physical pool conservation, scrolling and full refresh against a same-frame rebuild.");
    }
};

METALLIC_REGISTER_RHI_TEST(GlobalDistanceFieldPersistentPagesTest);

struct GdfContinuationResources {
    render::ShaderSampledImage pageTable, coarseMip;
    render::ShaderSampler sampler;
    render::ShaderBuffer domains, stepLimitDomains, output;
};

struct GdfContinuationDomain {
    std::array<float, 4> minimumAndVoxelSize, maximumAndInfluenceRadius;
};

struct GdfContinuationResult {
    float distance;
    uint32_t classification, hitClipmap, steps;
};

static_assert(sizeof(GdfContinuationResources) == 24);
static_assert(sizeof(GdfContinuationDomain) == 32);
static_assert(sizeof(GdfContinuationResult) == 16);

class GlobalDistanceFieldContinuationTest final : public RHITest {
public:
    GlobalDistanceFieldContinuationTest()
    {
        name = "global_distance_field_clipmap_continuation";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto require = [](bool condition, std::string_view message) {
            if (!condition) { throw std::runtime_error(std::string(message)); }
        };
        try {
            bench::ValidationRecorder validation;
            auto device = bench::createTestDevice(context, {.applicationName = "GDF clipmap continuation",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = validation.sink()});
            require(bool(device), "Could not initialize the GDF continuation test device");
            auto& gpu = **device;
            std::string log;
            auto shader = compileSlangShaderToSpirv({.moduleName = "GlobalDistanceFieldContinuationProbe",
                .entryPointName = "main", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, log);
            require(bool(shader), log);
            using Resources = GdfContinuationResources;
            const ComputeResourceBindingDesc bindings[]{
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, pageTable), .kind = ComputeResourceBindingKind::SampledImage},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, coarseMip), .kind = ComputeResourceBindingKind::SampledImage},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, sampler), .kind = ComputeResourceBindingKind::Sampler},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, domains)},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, stepLimitDomains)},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, output)},
            };
            ComputeProgram program;
            require(bool(program.initialize(gpu, {.spirv = shader->spirv, .bindings = bindings, .requiresRayQuery = false,
                .resourceParameterSize = sizeof(Resources)}, log)), log);

            constexpr uint32_t kPageSide = 3, kMipSide = 8;
            constexpr uint32_t kPositivePage = 0xffffffffu, kOverflowPage = 0xfffffffdu;
            constexpr std::array domains{
                GdfContinuationDomain{{-8.0f, -8.0f, -8.0f, 1.0f}, {8.0f, 8.0f, 8.0f, 4.0f}},
                GdfContinuationDomain{{-16.0f, -16.0f, -16.0f, 2.0f}, {16.0f, 16.0f, 16.0f, 8.0f}},
            };
            auto stepLimitDomains = domains;
            stepLimitDomains[0].minimumAndVoxelSize[3] = 0.03125f;
            stepLimitDomains[0].maximumAndInfluenceRadius[3] = 0.125f;
            constexpr uint32_t kDomainCount = static_cast<uint32_t>(domains.size());
            std::array<uint32_t, kPageSide * kPageSide * kPageSide * kDomainCount> pages;
            pages.fill(kPositivePage);
            for (uint32_t z = 0; z < kPageSide; ++z) {
                for (uint32_t x = 0; x < kPageSide; ++x) {
                    pages[z * kPageSide * kPageSide + x] = kOverflowPage;
                }
            }
            std::array<float, kMipSide * kMipSide * kMipSide * kDomainCount> mip;
            mip.fill(1.0f);
            for (uint32_t z = 0; z < kMipSide; ++z) {
                // These rows decode to the respective minimum step exactly. In
                // the coarse rows a zero-distance texel marks the far surface.
                const uint32_t fine = z * kMipSide * kMipSide + 6u * kMipSide;
                std::fill_n(mip.begin() + fine, kMipSide, 0.515625f);
                for (uint32_t y : {2u, 5u}) {
                    const uint32_t coarse = (z + kMipSide) * kMipSide * kMipSide + y * kMipSide;
                    std::fill_n(mip.begin() + coarse, kMipSide - 1u, 0.515625f);
                    mip[coarse + kMipSide - 1u] = 0.5f;
                }
            }
            const auto upload = [&](std::span<const std::byte> data, BufferUsageBits usage) {
                auto buffer = gpu.createBuffer({.size = data.size_bytes(), .usage = usage, .memoryLocation = MemoryLocation::HostUpload});
                require(bool(buffer), "Could not allocate the GDF test upload");
                void* mapped = (*buffer)->map();
                require(mapped != nullptr, "Could not map the GDF test upload");
                std::memcpy(mapped, data.data(), data.size_bytes());
                (*buffer)->flush();
                (*buffer)->unmap();
                return std::move(*buffer);
            };
            auto pageUpload = upload(std::as_bytes(std::span(pages)), BufferUsageBits::TransferSource);
            auto mipUpload = upload(std::as_bytes(std::span(mip)), BufferUsageBits::TransferSource);
            auto domainBuffer = upload(std::as_bytes(std::span(domains)), BufferUsageBits::Storage);
            auto stepLimitDomainBuffer = upload(std::as_bytes(std::span(stepLimitDomains)), BufferUsageBits::Storage);
            std::array<GdfContinuationResult, 6> actual{};
            auto output = gpu.createBuffer({.size = sizeof(actual), .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostReadback});
            require(bool(output), "Could not allocate the GDF test readback");
            const auto createVolume = [&](Format format, uint32_t side) {
                return gpu.createTexture({.type = TextureType::Texture3D,
                    .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                    .format = format, .width = side, .height = side, .depth = side * kDomainCount});
            };
            auto pageTable = createVolume(Format::R32Uint, kPageSide);
            auto coarseMip = createVolume(Format::R32Sfloat, kMipSide);
            require(bool(pageTable) && bool(coarseMip), "Could not allocate the test page table and coarse mip");
            auto pageTableView = gpu.createTextureView(**pageTable, {});
            auto coarseMipView = gpu.createTextureView(**coarseMip, {});
            require(bool(pageTableView) && bool(coarseMipView), "Could not create the GDF test views");
            bench::GPUCommands commands(*gpu.getQueue(QueueType::Graphics));
            require(bool(commands.initialize(gpu)), "Could not begin GDF test commands");
            const auto recordUpload = [&](Texture& texture, Buffer& source) {
                const TextureBarrierDesc before{.texture = &texture, .oldLayout = TextureLayout::Undefined,
                    .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
                require(bool(commands.commands->synchronize({.textures = {&before, 1}})), "GDF texture upload barrier failed");
                const auto& desc = texture.desc();
                require(bool(source.slice().and_then([&](const auto& slice) {
                    return commands.commands->copyBufferToTexture({.texture = &texture, .buffer = slice,
                        .width = desc.width, .height = desc.height, .depth = desc.depth});
                })), "GDF texture upload failed");
                const TextureBarrierDesc after{.texture = &texture, .oldLayout = TextureLayout::TransferDestination,
                    .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
                require(bool(commands.commands->synchronize({.textures = {&after, 1}})), "GDF texture sampling barrier failed");
            };
            recordUpload(**pageTable, *pageUpload);
            recordUpload(**coarseMip, *mipUpload);
            const SamplerDesc sampler{.minFilter = SamplerFilter::Nearest, .magFilter = SamplerFilter::Nearest};
            const ComputeDispatchBinding dispatch[]{
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, pageTable), .textureView = pageTableView->get()},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, coarseMip), .textureView = coarseMipView->get()},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, sampler), .sampler = &sampler},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, domains), .buffer = domainBuffer.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, stepLimitDomains), .buffer = stepLimitDomainBuffer.get()},
                {.binding = METALLIC_RESOURCE_MEMBER(Resources, output), .buffer = output->get()},
            };
            require(bool(program.dispatch({.commandBuffer = commands.commands.get(), .bindings = dispatch})),
                "GDF continuation dispatch failed");
            const BufferBarrierDesc readbackBarrier{.buffer = output->get(),
                .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                .after = {PipelineStageBits::Host, AccessBits::HostRead}};
            require(bool(commands.commands->synchronize({.buffers = {&readbackBarrier, 1}})), "GDF readback barrier failed");
            require(bool(commands.submitAndWait()), "GDF continuation execution failed");
            (*output)->invalidate();
            const void* mapped = (*output)->map();
            require(mapped != nullptr, "Could not read the GDF test results");
            std::memcpy(actual.data(), mapped, sizeof(actual));
            (*output)->unmap();
            require(!validation.failed(), "GDF continuation Vulkan validation: " + validation.snapshot().dump());

            constexpr uint32_t kNoHit = 0xffffffffu;
            constexpr std::array expected{
                GdfContinuationResult{11.5f, 2u, 1u, 6u},
                GdfContinuationResult{-1.0f, 3u, kNoHit, 1u},
                GdfContinuationResult{-1.0f, 1u, kNoHit, 2u},
                GdfContinuationResult{-1.0f, 8u, kNoHit, 0u},
                GdfContinuationResult{11.984375f, 2u, 1u, 262u},
                GdfContinuationResult{-1.0f, 7u, kNoHit, 256u},
            };
            constexpr std::array labels{"missing then hit", "missing then miss", "covered miss", "empty interval",
                "step limit then hit", "step limit without later coverage"};
            for (size_t index = 0; index < expected.size(); ++index) {
                const auto& ref = expected[index];
                const auto& value = actual[index];
                if (!std::isfinite(value.distance) || std::abs(value.distance - ref.distance) > 1.0e-4f ||
                    value.classification != ref.classification || value.hitClipmap != ref.hitClipmap || value.steps != ref.steps) {
                    return RHITestResult::fail(std::string(labels[index]) + ": got class " + std::to_string(value.classification) +
                        ", clipmap " + std::to_string(value.hitClipmap) + ", distance " + std::to_string(value.distance) +
                        ", steps " + std::to_string(value.steps));
                }
            }
            return RHITestResult::pass("Missing pages and step limits resume in the next clipmap and preserve unresolved states.");
        } catch (const std::exception& error) {
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(GlobalDistanceFieldContinuationTest);

} // namespace
} // namespace metallic::tests
