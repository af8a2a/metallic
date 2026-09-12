#include "RhiTest.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/MeshletLod.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {

using namespace render;

class StreamMeshletLodSceneTest final : public RhiTest {
public:
    StreamMeshletLodSceneTest()
    {
        type = RhiTestType::Rendering;
        name = "meshlet_lod_stream_scene_runtime_cut";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        using debug::DebugValue;
        constexpr uint32_t kWidth = 193;
        constexpr uint32_t kHeight = 157;
        const auto assetPath = std::filesystem::absolute(context.outputDirectory / "StreamLodBunny.meshstream.bin");
        const std::filesystem::path sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
        std::string log;
        if (!scene::buildMeshletStreamAssetOffline({.sourcePath = sourcePath, .outputPath = assetPath}, log)) {
            return RhiTestResult::fail("Bunny streamasset build: " + log);
        }
        scene::MeshletStreamAsset asset;
        if (!asset.open(assetPath, log)) { return RhiTestResult::fail(log); }
        uint32_t capacity = 0;
        uint32_t lodStateWordCount = asset.instanceCount() * 4u;
        for (const auto& instance : asset.instances()) {
            const uint32_t groupCount = asset.primitives()[instance.primitiveIndex].groupCount;
            capacity += groupCount;
            lodStateWordCount += 4u + 3u * groupCount;
        }
        if (asset.pageCount() > 64 || asset.terminalGroups().empty()) {
            return RhiTestResult::fail("Bunny fixture exceeds the fully resident validation budget or has no roots");
        }

        RenderDebugRuntime debug;
        RenderGraphPreviewRenderer preview;
        const auto initialized = preview.initialize(context.enableValidation, false, false);
        if (hasError(initialized, Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders"); }
        if (!initialized) { return RhiTestResult::fail(preview.lastLog()); }
        preview.setDebugObserver(&debug);
        RenderGraph graph;
        graph.addNode("VisibilityBufferPass", "VBuffer", {
            {"path", sourcePath.generic_string()}, {"streamAssetPath", assetPath.generic_string()},
            {"enableMeshletStreaming", true}, {"maxResidentPages", 64}, {"maxLockedFallbackPages", 64},
            {"maxPageUploadsPerFrame", 64}, {"maxActiveGroups", capacity},
            {"maxGpuPageRequests", 256}, {"maxGpuPageUnloadRequests", 256},
            {"maxTraversalWorkers", 64}, {"maxTraversalWorkItems", 4096},
            {"visualization", "triangle"}, {"dlssJitter", false},
            {"camera", {{"eye", {-.0168404f, .110154f, .22f}},
                {"center", {-.0168404f, .110154f, -.00153695f}},
                {"znear", .001f}, {"zfar", 10.f}, {"orthoHeight", .24f}, {"fovDegrees", 60.f}}}});
        graph.markOutput("VBuffer.visibility");
        graph.markOutput("VBuffer.color");
        graph.markOutput("VBuffer.depth");
        const auto node = graph.findNode("VBuffer")->id;
        const auto render = [&](const char* output = "VBuffer.visibility") {
            const auto result = preview.render(graph, kWidth, kHeight, output);
            debug.poll();
            if (!result) { throw std::runtime_error(preview.lastLog()); }
        };
        const auto call = [&](const char* method, DebugValue params) {
            auto response = debug.core().dispatch({{"id", "stream-lod-scene-test"}, {"method", method}, {"params", params}});
            if (response.value("status", "") != "ok") { throw std::runtime_error(response.dump()); }
            return response.at("result");
        };
        const auto capture = [&]() -> std::string {
            return call("capture.batch", {{"pass", "VBuffer"}, {"checkpoint", "AfterTraversal"},
                {"resources", {{{"id", "streaming.VBuffer.pageTable"}, {"count", asset.pageCount()}},
                    {{"id", "streaming.VBuffer.activeHeader"}, {"count", 1}},
                    {{"id", "streaming.VBuffer.lodState"}, {"count", lodStateWordCount}},
                    {{"id", "streaming.VBuffer.activeGroups"}, {"count", capacity}}}}}).at("job");
        };
        const auto read = [&](const std::string& job, const std::string& id) {
            auto rows = DebugValue::array();
            uint64_t total = 0;
            do {
                auto page = call("eval", {{"job", job}, {"expression", "buffers[\"" + id + "\"]"},
                    {"offset", rows.size()}, {"count", 4096}}).at("value");
                total = page.at("total");
                for (auto& row : page.at("items")) { rows.push_back(std::move(row)); }
            } while (rows.size() < total);
            return rows;
        };
        struct TraversalStats {
            uint64_t allActiveGroups = 0;
            uint64_t visitedBvhNodes = 0;
            uint64_t testedGroups = 0;
        };
        const auto traversalStats = [&](const DebugValue& words) {
            if (words.size() != lodStateWordCount) {
                throw std::runtime_error("Incomplete stream BVH state capture");
            }
            TraversalStats result;
            uint32_t base = asset.instanceCount() * 4u;
            for (const auto& instance : asset.instances()) {
                const uint32_t groupCount = asset.primitives()[instance.primitiveIndex].groupCount;
                // The sparse header follows the unchanged active/mask pairs.
                const uint32_t header = base + groupCount * 2u;
                const uint32_t active = words.at(header).get<uint32_t>();
                const uint32_t visited = words.at(header + 1u).get<uint32_t>();
                const uint32_t tested = words.at(header + 2u).get<uint32_t>();
                if (active > tested || tested > groupCount || (active != 0 && visited == 0)) {
                    throw std::runtime_error("Invalid stream BVH traversal counters");
                }
                result.allActiveGroups += active;
                result.visitedBvhNodes += visited;
                result.testedGroups += tested;
                base += 4u + 3u * groupCount;
            }
            return result;
        };

        struct ExpectedGroup {
            uint32_t instance, primitive, page, mask;
            bool operator==(const ExpectedGroup&) const = default;
        };
        struct ReferenceCut {
            std::vector<ExpectedGroup> groups;
            size_t selectedCount = 0;
            DebugValue levels = DebugValue::object();
        };
        const auto referenceCut = [&](const MeshletLodView& view, uint32_t manual, const DebugValue* pageTable) {
            ReferenceCut result;
            for (uint32_t instanceIndex = 0; instanceIndex < asset.instanceCount(); ++instanceIndex) {
                const auto& sourceInstance = asset.instances()[instanceIndex];
                const auto& primitive = asset.primitives()[sourceInstance.primitiveIndex];
                std::vector<MeshletLodGroupRecord> groups;
                std::vector<MeshletLodGroupRange> ranges;
                std::vector<uint32_t> refined;
                std::vector<uint8_t> drawable;
                for (uint32_t local = 0; local < primitive.groupCount; ++local) {
                    const auto& source = asset.groups()[primitive.groupOffset + local];
                    MeshletLodGroupRecord group;
                    std::copy_n(source.boundsCenterRadius, 4, group.sphere.begin());
                    group.error = source.maxQuadricError;
                    group.level = source.lodLevel;
                    group.flags = source.flags;
                    groups.push_back(group);
                    ranges.push_back({static_cast<uint32_t>(refined.size()), source.clusterCount});
                    for (uint32_t cluster = 0; cluster < source.clusterCount; ++cluster) {
                        const uint32_t child = asset.refinedGroups()[source.clusterRefinedOffset + cluster];
                        refined.push_back(child == UINT32_MAX ? child : child - primitive.groupOffset);
                    }
                    const uint32_t state = pageTable == nullptr ? 2u :
                        pageTable->at(source.pageIndex).at("deviceOffsetAndState").get<uint32_t>() & 7u;
                    drawable.push_back(state == 2 || state == 3 ? 1 : 0);
                }
                GPUSceneGpuInstanceRecord instance;
                std::copy_n(sourceInstance.worldMatrix, 16, instance.worldMatrix.begin());
                instance.identity[3] = sourceInstance.visible != 0 ? GPUSceneGpuInstanceVisible : 0;
                const auto cut = selectStreamMeshletLodReference(groups, ranges, refined, drawable, instance, view, manual);
                if (!cut.valid) { throw std::runtime_error("Scene CPU cut is invalid: " + cut.reason); }
                std::vector<uint8_t> selected(refined.size(), 0);
                for (uint32_t cluster : cut.selectedClusters) { selected.at(cluster) = 1; }
                result.selectedCount += cut.selectedClusters.size();
                for (uint32_t local = 0; local < primitive.groupCount; ++local) {
                    uint32_t mask = 0;
                    for (uint32_t cluster = 0; cluster < ranges[local].clusterCount; ++cluster) {
                        if (selected[ranges[local].clusterOffset + cluster]) { mask |= 1u << cluster; }
                    }
                    if (mask == 0) { continue; }
                    const auto& source = asset.groups()[primitive.groupOffset + local];
                    result.groups.push_back({instanceIndex, sourceInstance.primitiveIndex, source.pageIndex, mask});
                    const std::string level = std::to_string(source.lodLevel);
                    result.levels[level] = result.levels.value(level, 0u) + 1u;
                }
            }
            return result;
        };
        const auto matchesCut = [](const DebugValue& header, const DebugValue& rows, const std::vector<ExpectedGroup>& cut) {
            if (header.at("overflowCount") != 0 || header.at("activeGroupCount") != cut.size() || rows.size() < cut.size()) {
                return false;
            }
            for (size_t index = 0; index < cut.size(); ++index) {
                const auto& row = rows.at(index);
                const auto& group = cut[index];
                if (row.at("instanceIndex") != group.instance || row.at("primitiveIndex") != group.primitive ||
                    row.at("pageIndex") != group.page || row.at("clusterSelectionMask") != group.mask) {
                    return false;
                }
            }
            return true;
        };

        try {
            render();
            auto* gpuScene = preview.subsystemHost()->get<GPUSceneSubsystem>();
            if (!gpuScene) { return RhiTestResult::fail("Missing GPUScene"); }
            const bool independent = preview.subsystemHost()->device()->capabilities().independentComputeQueue;
            DebugValue report{{"asset", sourcePath.generic_string()}, {"pages", asset.pageCount()},
                {"groups", asset.groupCount()}, {"terminalGroups", asset.terminalGroups().size()},
                {"independentComputeQueue", independent}, {"cases", DebugValue::array()}};
            std::array<size_t, 2> finestCount{}, coarsestCount{};
            std::array<std::array<uint64_t, 2>, 2> finestGroupTests{}, coarsestGroupTests{};
            for (uint32_t test = 0; test < 8; ++test) {
                const bool ortho = test >= 4;
                const uint32_t configuration = test % 4;
                const uint32_t manual = configuration < 2 ? UINT32_MAX : configuration == 2 ? 0u : 31u;
                const float error = configuration == 0 ? .05f : configuration == 1 ? 16.f : 1.5f;
                const bool reversed = (test % 2) != 0;
                graph.setNodeRuntimeProperty(node, "autoLod", manual == UINT32_MAX);
                graph.setNodeRuntimeProperty(node, "lodLevel", manual == UINT32_MAX ? 0u : manual);
                graph.setNodeRuntimeProperty(node, "lodPixelError", error);
                graph.setNodeRuntimeProperty(node, "lodBias", 0.f);
                graph.setNodeRuntimeProperty(node, "camera.projection", ortho ? "orthographic" : "perspective");
                graph.setNodeRuntimeProperty(node, "camera.reversedZ", reversed);
                MeshletLodView view;
                view.eye = {-.0168404f, .110154f, .22f, .001f};
                view.forward = {0, 0, -1, ortho ? 1.f : 0.f};
                view.projection = {float(kHeight), std::tan(3.14159265358979323846f / 6.f), .24f, error};
                const auto fullTarget = referenceCut(view, manual, nullptr);
                if (fullTarget.groups.empty()) { return RhiTestResult::fail("Fully resident Bunny cut is empty"); }
                std::vector<uint32_t> hardwareVisibility;
                std::vector<uint32_t> hardwareDepth;
                for (uint32_t producer = 0; producer < 2; ++producer) {
                    const bool hybrid = producer != 0;
                    graph.setNodeRuntimeProperty(node, "hybridRaster", hybrid);
                    graph.setNodeRuntimeProperty(node, "clusterPrebin", hybrid);
                    graph.setNodeRuntimeProperty(node, "asyncSoftwareRaster", hybrid);
                    graph.setNodeRuntimeProperty(node, "softwareRasterMaxPixels", 8.f);
                    DebugValue header, pageTable, actual;
                    TraversalStats traversal;
                    uint32_t warmupFrames = 0;
                    bool settled = false;
                    for (uint32_t batch = 0; batch < 28; ++batch) {
                        for (uint32_t frame = 0; frame < 8; ++frame) { render(); }
                        const auto job = capture();
                        render();
                        warmupFrames += 9;
                        header = read(job, "streaming.VBuffer.activeHeader").at(0);
                        pageTable = read(job, "streaming.VBuffer.pageTable");
                        actual = read(job, "streaming.VBuffer.activeGroups");
                        if (matchesCut(header, actual, fullTarget.groups)) {
                            traversal = traversalStats(read(job, "streaming.VBuffer.lodState"));
                            settled = true;
                            break;
                        }
                    }
                    if (!settled) {
                        uint32_t actualClusters = 0;
                        const uint32_t count = std::min(header.at("activeGroupCount").get<uint32_t>(),
                            static_cast<uint32_t>(actual.size()));
                        for (uint32_t index = 0; index < count; ++index) {
                            actualClusters += std::popcount(actual.at(index).at("clusterSelectionMask").get<uint32_t>());
                        }
                        return RhiTestResult::fail("Streaming did not converge to the fully resident target after " +
                            std::to_string(warmupFrames) + " frames in case " + std::to_string(test) +
                            ": expected " + std::to_string(fullTarget.selectedCount) + " clusters, got " + std::to_string(actualClusters));
                    }
                    const uint32_t activeCount = header.at("activeGroupCount");
                    if (header.at("overflowCount") != 0 || activeCount == 0 || activeCount > actual.size()) {
                        return RhiTestResult::fail("Invalid stream active header in case " + std::to_string(test));
                    }
                    if (traversal.allActiveGroups < activeCount) {
                        return RhiTestResult::fail("Stream BVH active list omitted emitted groups");
                    }

                    const auto reference = referenceCut(view, manual, &pageTable);
                    const auto& expected = reference.groups;
                    const size_t selectedCount = reference.selectedCount;
                    const auto& levels = reference.levels;
                    if (expected != fullTarget.groups) {
                        return RhiTestResult::fail("GPU matched the full cut but residency-constrained CPU selection did not");
                    }
                    if (expected.size() != activeCount) {
                        return RhiTestResult::fail("Stream CPU/GPU group count mismatch in case " + std::to_string(test) +
                            ": CPU " + std::to_string(expected.size()) + ", GPU " + std::to_string(activeCount));
                    }
                    for (uint32_t index = 0; index < activeCount; ++index) {
                        const auto& row = actual.at(index);
                        const auto& group = expected[index];
                        if (row.at("instanceIndex") != group.instance || row.at("primitiveIndex") != group.primitive ||
                            row.at("pageIndex") != group.page || row.at("clusterSelectionMask") != group.mask) {
                            return RhiTestResult::fail("Stream CPU/GPU cluster mask mismatch in case " + std::to_string(test) +
                                " group " + std::to_string(index));
                        }
                    }

                    // Stream IDs retain sparse active-group slots; every pixel must refer
                    // to a selected cluster in this exact frame's active list.
                    const uint32_t recordBase = static_cast<uint32_t>(
                        gpuScene->globalBufferViews().meshletDraws.size / sizeof(VisibleClusterRecord));
                    const uint32_t clusterStride = header.at("maxActiveGroupClusters");
                    if (clusterStride == 0) { return RhiTestResult::fail("Zero stream cluster slot stride"); }
                    size_t covered = 0;
                    for (uint32_t id : preview.pixels()) {
                        if (id == 0) { continue; }
                        const uint32_t record = (id >> kVisibilityTriangleBits) - 1u;
                        if (record < recordBase) { return RhiTestResult::fail("Resident Bunny geometry leaked into stream visibility"); }
                        const uint32_t local = record - recordBase;
                        const uint32_t group = local / clusterStride;
                        const uint32_t cluster = local % clusterStride;
                        if (group >= activeCount || cluster >= 32 || (expected[group].mask & (1u << cluster)) == 0) {
                            return RhiTestResult::fail("Stream visibility referenced a cluster outside the captured cut");
                        }
                        ++covered;
                    }
                    if (covered < 100) { return RhiTestResult::fail("Stream LOD lost Bunny visibility coverage"); }
                    const uint32_t branches = preview.executionStats().asyncComputeBranches;
                    if ((hybrid && independent) ? branches < 2 : branches != 0) {
                        return RhiTestResult::fail("Stream raster producer did not use the expected queue topology");
                    }
                    const auto visibility = preview.pixels();
                    size_t roundingTies = 0;
                    if (!hybrid) {
                        hardwareVisibility = visibility;
                        render("VBuffer.depth");
                        hardwareDepth = preview.pixels();
                    } else if (visibility != hardwareVisibility) {
                        render("VBuffer.depth");
                        const auto& depth = preview.pixels();
                        for (size_t pixel = 0; pixel < visibility.size(); ++pixel) {
                            if (visibility[pixel] == hardwareVisibility[pixel]) { continue; }
                            const uint32_t delta = std::max(depth[pixel], hardwareDepth[pixel]) -
                                std::min(depth[pixel], hardwareDepth[pixel]);
                            // Same qualified depth-tie rule as the resident hybrid regression:
                            // coverage and cluster identity stay exact; only triangle ties may differ.
                            if (visibility[pixel] == 0 || hardwareVisibility[pixel] == 0 ||
                                (visibility[pixel] >> kVisibilityTriangleBits) !=
                                    (hardwareVisibility[pixel] >> kVisibilityTriangleBits) || delta > 8) {
                                return RhiTestResult::fail("Settled stream HW/async visibility mismatch in case " +
                                    std::to_string(test) + " pixel " + std::to_string(pixel) +
                                    " depth ULP " + std::to_string(delta));
                            }
                            ++roundingTies;
                        }
                    }
                    report["cases"].push_back({{"case", test}, {"orthographic", ortho}, {"reversedZ", reversed},
                        {"manualLevel", manual}, {"targetPixels", error}, {"hybrid", hybrid},
                        {"activeGroups", activeCount}, {"selectedClusters", selectedCount},
                        {"bvhVisitedNodes", traversal.visitedBvhNodes},
                        {"bvhTestedGroups", traversal.testedGroups},
                        {"bvhAllActiveGroups", traversal.allActiveGroups}, {"groupsWithoutBvh", capacity},
                        {"fullResidentTargetClusters", fullTarget.selectedCount}, {"warmupFrames", warmupFrames},
                        {"triangleDepthTies", roundingTies},
                        {"coveredPixels", covered}, {"asyncBranches", branches}, {"groupLevels", levels}});
                    if (manual == 0) {
                        finestCount[ortho] = selectedCount;
                        finestGroupTests[ortho][producer] = traversal.testedGroups;
                    }
                    if (manual == 31) {
                        coarsestCount[ortho] = selectedCount;
                        coarsestGroupTests[ortho][producer] = traversal.testedGroups;
                    }
                    if (hybrid) {
                        render("VBuffer.color");
                        if (!saveRgba8Png(context.outputDirectory / ("StreamLodBunny-" + std::to_string(test) + ".png"),
                                reinterpret_cast<const uint8_t*>(preview.pixels().data()), kWidth, kHeight, log)) {
                            return RhiTestResult::fail(log);
                        }
                    }
                }
                if (configuration == 0) {
                    graph.setNodeRuntimeProperty(node, "freezeCullingCamera", true);
                    render();
                    const auto frozenJob = capture();
                    render();
                    if (!matchesCut(read(frozenJob, "streaming.VBuffer.activeHeader").at(0),
                            read(frozenJob, "streaming.VBuffer.activeGroups"), fullTarget.groups)) {
                        return RhiTestResult::fail("Freezing the camera changed the settled stream cut");
                    }
                    const auto frozenVisibility = preview.pixels();
                    graph.setNodeRuntimeProperty(node, "camera.eye", {.0231596f, .110154f, 3.2f});
                    graph.setNodeRuntimeProperty(node, "camera.center", {.0231596f, .110154f, -.00153695f});
                    graph.setNodeRuntimeProperty(node, "camera.orthoHeight", 4.f);
                    for (uint32_t frame = 0; frame < 3; ++frame) { render(); }
                    const auto movedJob = capture();
                    render();
                    if (!matchesCut(read(movedJob, "streaming.VBuffer.activeHeader").at(0),
                            read(movedJob, "streaming.VBuffer.activeGroups"), fullTarget.groups)) {
                        return RhiTestResult::fail("Moving the render camera changed a frozen stream LOD cut");
                    }
                    if (preview.pixels() == frozenVisibility ||
                        std::none_of(preview.pixels().begin(), preview.pixels().end(), [](uint32_t id) { return id != 0; })) {
                        return RhiTestResult::fail("Frozen stream cut did not render the moved camera");
                    }
                    report["frozenCamera"].push_back({{"orthographic", ortho}, {"clusters", fullTarget.selectedCount},
                        {"sameCut", true}, {"changedPixels", true}});
                    graph.setNodeRuntimeProperty(node, "freezeCullingCamera", false);
                    graph.setNodeRuntimeProperty(node, "camera.eye", {-.0168404f, .110154f, .22f});
                    graph.setNodeRuntimeProperty(node, "camera.center", {-.0168404f, .110154f, -.00153695f});
                    graph.setNodeRuntimeProperty(node, "camera.orthoHeight", .24f);
                    render();
                }
            }
            for (uint32_t projection = 0; projection < 2; ++projection) {
                if (coarsestCount[projection] == 0 || coarsestCount[projection] >= finestCount[projection]) {
                    return RhiTestResult::fail("Stream manual coarse LOD did not reduce the Bunny cluster count");
                }
                for (uint32_t producer = 0; producer < 2; ++producer) {
                    if (coarsestGroupTests[projection][producer] >= capacity ||
                        coarsestGroupTests[projection][producer] >= finestGroupTests[projection][producer]) {
                        return RhiTestResult::fail("Stream BVH did not prune group tests for the coarse Bunny cut: coarse " +
                            std::to_string(coarsestGroupTests[projection][producer]) + ", fine " +
                            std::to_string(finestGroupTests[projection][producer]) + ", total " + std::to_string(capacity));
                    }
                }
            }
            std::ofstream reportFile(context.outputDirectory / "StreamMeshletLodSceneReport.json");
            reportFile << report.dump(2);
            if (!reportFile) { return RhiTestResult::fail("Could not write stream LOD report"); }
            return RhiTestResult::pass("16 converged Bunny stream cuts match fully resident CPU targets and HW/async visibility");
        } catch (const std::exception& error) {
            return RhiTestResult::fail(error.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLodSceneTest);

} // namespace
} // namespace metallic::tests
