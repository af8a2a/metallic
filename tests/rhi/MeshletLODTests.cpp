#include "Runtime/Render/Core/ResourceRegistry.h"
#include "RHITest.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Subsystem/GPUSceneSubsystem.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace metallic::tests {
namespace {

using namespace render;

struct LODFixture {
    scene::RenderPrimitive primitive;
    std::vector<MeshletLODGroupRecord> groups;
    std::vector<GPUSceneGPUMeshletRecord> clusters;
    std::vector<GPUSceneGPUInstanceRecord> instances;
    std::vector<VisibleClusterRecord> candidates;
};

LODFixture makeLodFixture()
{
    LODFixture f;
    f.primitive.positions = {float3(0, 0, 0), float3(1, 0, 0), float3(0, 1, 0)};
    f.primitive.meshletLodVertices = {0, 1, 2};
    f.primitive.meshletLodTriangles = {0, 1, 2};
    const uint32_t counts[] = {2, 2, 1, 3, 1};
    const uint32_t levels[] = {0, 0, 0, 1, 2};
    const float errors[] = {.1f, .5f, .02f, 2.f, 2.5f};
    for (uint32_t g = 0; g < 5; ++g) {
        scene::MeshletLODGroup group;
        group.clusterOffset = static_cast<uint32_t>(f.clusters.size());
        group.clusterCount = counts[g];
        group.lodLevel = levels[g];
        group.maxQuadricError = errors[g];
        group.boundingSphereCenter = float3(g == 0 ? -2.f : g == 1 ? 2.f : 0.f, 0, 0);
        group.boundingSphereRadius = g < 3 ? 1.f : 5.f;
        f.primitive.meshletLodGroups.push_back(group);
        for (uint32_t c = 0; c < counts[g]; ++c) {
            const uint32_t refined = g == 3 ? (c < 2 ? 0u : 1u) : g == 4 ? 3u : UINT32_MAX;
            scene::MeshletCluster cluster;
            cluster.vertexCount = 3;
            cluster.triangleCount = 1;
            cluster.lodLevel = levels[g];
            cluster.lodGroupIndex = g;
            cluster.refinedGroupIndex = static_cast<int32_t>(refined);
            f.primitive.meshletLodClusters.push_back(cluster);
            GPUSceneGPUMeshletRecord gpu;
            gpu.lod = {levels[g], g, refined, 0};
            f.clusters.push_back(gpu);
        }
    }
    GPUSceneGPUInstanceRecord instance;
    instance.worldMatrix = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    instance.identity[3] = GPUSceneGPUInstanceVisible;
    f.instances.push_back(instance);
    for (uint32_t c = 0; c < f.clusters.size(); ++c) { f.candidates.push_back({c, 0, 7, 0}); }
    return f;
}

class MeshletLODReferenceTest final : public RHITest {
public:
    MeshletLODReferenceTest() { type = RHITestType::Validation; name = "meshlet_lod_reference_cut_and_metadata"; }
    RHITestResult run(RHITestContext&) override
    {
        auto f = makeLodFixture();
        std::string reason;
        if (!buildMeshletLodMetadata(f.primitive, f.groups, reason)) { return RHITestResult::fail(reason); }
        MeshletLODView view;
        view.forward[3] = 1;
        view.projection = {1000, .5f, 100, 1.5f};
        const auto check = [&](std::initializer_list<uint32_t> expected, uint32_t manual = UINT32_MAX) {
            auto result = selectMeshletLodReference(f.groups, f.clusters, f.instances, f.candidates, view, 17, manual);
            std::vector<uint32_t> ids;
            for (const auto& selected : result) {
                if (selected.instanceIndex != 0 || selected.geometryIndex != 7 || selected.recordIndex != selected.clusterIndex + 17) { return false; }
                ids.push_back(selected.clusterIndex);
            }
            return ids == std::vector<uint32_t>(expected);
        };
        if (!check({2, 3, 4, 5, 6}) || !check({0, 1, 2, 3, 4}, 0) ||
            !check({4, 5, 6, 7}, 1) || !check({4, 8}, 31)) {
            return RHITestResult::fail("mixed cut, shared generating group or early terminal coverage is incorrect");
        }
        view.projection[3] = .5f;
        if (!check({0, 1, 2, 3, 4})) { return RHITestResult::fail("smaller pixel error did not refine"); }
        view.projection[3] = 8;
        if (!check({4, 5, 6, 7})) { return RHITestResult::fail("larger pixel error did not coarsen"); }
        view.projection[3] = 30;
        if (!check({4, 8})) { return RHITestResult::fail("root cut omitted an early terminal"); }
        view.forward[3] = 0;
        view.eye = {0, 0, 100, .1f};
        auto& instance = f.instances.front();
        float base = meshletLodPixelError(f.groups[0], instance, view);
        instance.worldMatrix[4] = 3;
        float sheared = meshletLodPixelError(f.groups[0], instance, view);
        if (!(sheared > base)) { return RHITestResult::fail("shear is not included in projected error"); }
        view.eye[2] = .1f;
        if (meshletLodPixelError(f.groups[0], instance, view) != FLT_MAX) { return RHITestResult::fail("near-plane intersection did not request detail"); }
        f.primitive.meshletLodClusters.back().refinedGroupIndex = 4;
        if (buildMeshletLodMetadata(f.primitive, f.groups, reason) || !f.groups.empty()) {
            return RHITestResult::fail("cyclic metadata was accepted");
        }
        return RHITestResult::pass("mixed-depth roots, many-to-one refinement, pixel threshold, manual cuts, shear and invalid DAG");
    }
};
METALLIC_REGISTER_RHI_TEST(MeshletLODReferenceTest);

class MeshletLODDisplayPixelThresholdTest final : public RHITest {
public:
    MeshletLODDisplayPixelThresholdTest()
    {
        type = RHITestType::Validation;
        name = "meshlet_lod_display_pixel_threshold";
    }

    RHITestResult run(RHITestContext&) override
    {
        auto fixture = makeLodFixture();
        std::string reason;
        if (!buildMeshletLodMetadata(fixture.primitive, fixture.groups, reason)) {
            return RHITestResult::fail(reason);
        }
        std::vector<MeshletLODGroupRange> ranges;
        std::vector<uint32_t> refined;
        for (const auto& group : fixture.primitive.meshletLodGroups) {
            ranges.push_back({group.clusterOffset, group.clusterCount});
        }
        for (const auto& cluster : fixture.clusters) { refined.push_back(cluster.lod[2]); }
        const std::vector<uint8_t> drawable(fixture.groups.size(), 1);
        constexpr uint32_t kDisplayHeight = 1080;
        constexpr float kDisplayPixelError = 1.5f;
        const auto referenceCut = [&](const MeshletLODView& view) {
            return selectMeshletLodReference(fixture.groups, fixture.clusters, fixture.instances,
                fixture.candidates, view);
        };
        const auto streamCut = [&](const MeshletLODView& view) {
            return selectStreamMeshletLodReference(fixture.groups, ranges, refined, drawable,
                fixture.instances.front(), view);
        };
        for (bool orthographic : {false, true}) {
            MeshletLODView native;
            native.eye = {0, 0, 50, .1f};
            native.forward[3] = orthographic ? 1.f : 0.f;
            native.projection = {float(kDisplayHeight), .577350269f, 60.f, kDisplayPixelError};
            const auto expectedReference = referenceCut(native);
            const auto expectedStream = streamCut(native);
            std::vector<uint32_t> expectedClusters;
            for (const auto& selected : expectedReference) { expectedClusters.push_back(selected.clusterIndex); }
            if (expectedReference.empty() || !expectedStream.valid || expectedStream.selectedClusters != expectedClusters) {
                return RHITestResult::fail("CPU reference and fully drawable streaming fixture disagree");
            }
            bool legacyCutChanged = false;
            for (uint32_t renderHeight : {1080u, 720u, 540u, 2160u}) {
                auto scaled = native;
                scaled.projection[0] = float(renderHeight);
                scaled.projection[3] = meshletLodRenderPixelThreshold(kDisplayPixelError, renderHeight, kDisplayHeight);
                const auto actualStream = streamCut(scaled);
                if (referenceCut(scaled) != expectedReference || !actualStream.valid ||
                    actualStream.selectedClusters != expectedStream.selectedClusters ||
                    actualStream.activeGroups != expectedStream.activeGroups ||
                    actualStream.requestedGroups != expectedStream.requestedGroups) {
                    return RHITestResult::fail("Display-pixel cut changed with render height " + std::to_string(renderHeight));
                }
                // Ensure this fixture detects the old internal-pixel policy.
                scaled.projection[3] = kDisplayPixelError;
                legacyCutChanged |= referenceCut(scaled) != expectedReference &&
                    streamCut(scaled).selectedClusters != expectedStream.selectedClusters;
            }
            if (!legacyCutChanged) { return RHITestResult::fail("Resolution regression fixture never crosses a LOD boundary"); }
        }
        struct ThresholdCase {
            uint32_t renderHeight, displayHeight;
            float expected;
        };
        const ThresholdCase thresholds[] = {{720, 1080, 1.f}, {540, 1080, .75f},
            {2160, 1080, 3.f}, {720, 0, 1.5f}, {0, 0, 1.5f}, {0, 1080, 1.5f / 1080.f}};
        for (const auto& test : thresholds) {
            const float actual = meshletLodRenderPixelThreshold(1.5f, test.renderHeight, test.displayHeight);
            // Multiplying a rounded ratio can differ by one ULP from dividing
            // the pixel error directly, notably for a zero render extent.
            const float tolerance = 4.f * std::numeric_limits<float>::epsilon() * test.expected;
            if (!std::isfinite(actual) || std::abs(actual - test.expected) > tolerance) {
                return RHITestResult::fail("Threshold conversion failed for render/display height " +
                    std::to_string(test.renderHeight) + "/" + std::to_string(test.displayHeight) +
                    ": expected " + std::to_string(test.expected) + ", actual " + std::to_string(actual));
            }
        }
        return RHITestResult::pass("Perspective/orthographic streaming cuts remain stable across native, Quality, Performance and supersampling; missing sizes are safe");
    }
};
METALLIC_REGISTER_RHI_TEST(MeshletLODDisplayPixelThresholdTest);

} // namespace
} // namespace metallic::tests
