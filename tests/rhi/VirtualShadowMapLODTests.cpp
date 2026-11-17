#include "RHITest.h"
#include "harness/ValidationRecorder.h"
#include "Runtime/Debug/DebugHash.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Profiling/CPUProfile.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Shadow/VirtualShadowMapCacheDiagnostics.h"
#include "Runtime/Render/Shadow/VirtualShadowMapLOD.h"
#include "Runtime/Render/Shadow/VirtualShadowMaps.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;
constexpr uint32_t kWidth = 193, kHeight = 157, kPixels = kWidth * kHeight;

void check(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

template<class T> void check(const Result<T>& result, const std::string& message)
{
    check(bool(result), message + (result ? "" : ": " + std::string(resultToString(result))));
}

struct LODCommands
{
    RenderFrameContext frame;
    QueueSubmissionTracker tracker;
    std::unique_ptr<CommandPool> pool;
    std::unique_ptr<CommandBuffer> commands;
    uint64_t index = 0;

    explicit LODCommands(Device& device)
    {
        auto& queue = *device.getQueue(QueueType::Graphics);
        check(tracker.initialize(device, queue), "LOD submission tracker");
        check(device.createCommandPool(queue).transform([&](auto p) { pool = std::move(p); }), "LOD pool");
        check(pool->createCommandBuffer().transform([&](auto c) { commands = std::move(c); }), "LOD commands");
    }
    ~LODCommands()
    {
        frame.cancel();
        if (frame.completion().isSubmitted()) { (void)frame.wait(); }
        (void)pool->reset();
        (void)frame.reset();
    }
    void begin()
    {
        check(frame.begin(index++), "LOD frame begin");
        check(pool->reset(), "LOD pool reset");
        check(commands->begin(frame.submissionContext()), "LOD command begin");
    }
    void submit(Semaphore* gate = nullptr)
    {
        check(commands->end(), "LOD command end");
        CommandBuffer* list[]{commands.get()};
        const SemaphoreSubmitDesc wait{.semaphore = gate, .value = 1};
        check(tracker.submit({.waitSemaphores = {gate ? &wait : nullptr, gate ? 1u : 0u},
            .commandBuffers = list}, frame), "LOD submit");
    }
    void finish(bool cancel = false)
    {
        if (cancel) {
            check(commands->end(), "LOD cancelled command end");
            frame.cancel();
            return;
        }
        submit();
        check(frame.wait(), "LOD wait");
    }
};

void compareLODSelection(const VirtualShadowMapLODSelection& actual, const VirtualShadowMapLODSelection& reference)
{
    check(actual.cut.valid && reference.cut.valid, "Adaptive or flat shadow cut failed validation");
    check(actual.cut.selectedClusters == reference.cut.selectedClusters, "Adaptive selected cluster IDs differ from flat reference");
    check(actual.cut.requestedGroups == reference.cut.requestedGroups, "Adaptive missing-page demand differs from flat reference");
    check(actual.cut.activeGroups == reference.cut.activeGroups, "Adaptive active groups differ from flat reference");
}

// The test oracle evaluates every group and delegates topology to the independent
// stream reference selector. Production only exposes hierarchical traversal.
VirtualShadowMapLODSelection selectReferenceShadowLOD(const VirtualShadowMapLODHierarchy& hierarchy,
    std::span<const uint8_t> drawableGroups, const GPUSceneGPUInstanceRecord& instance,
    std::span<const VirtualShadowMapLODProjection> projections, float pixelThreshold,
    float maxWorldError = std::numeric_limits<float>::max())
{
    VirtualShadowMapLODSelection result;
    auto requirements = buildVirtualShadowMapLodRequirements(hierarchy.groups, instance, projections, pixelThreshold, maxWorldError);
    if (pixelThreshold == 0) { std::ranges::fill(requirements.needsFine, uint8_t(1)); }
    for (uint32_t owner = 0; owner < hierarchy.ranges.size(); ++owner) {
        const auto& range = hierarchy.ranges[owner];
        for (uint32_t cluster = range.clusterOffset; cluster < range.clusterOffset + range.clusterCount; ++cluster) {
            const uint32_t child = hierarchy.refinedGroups[cluster];
            if (child != kMeshletLODInvalidGroup) { requirements.needsFine[owner] |= requirements.needsFine[child]; }
        }
    }
    result.cut = selectStreamMeshletLodReference(hierarchy.groups, hierarchy.ranges, hierarchy.refinedGroups,
        drawableGroups, instance, MeshletLODView{}, UINT32_MAX, UINT32_MAX, {}, {}, requirements.needsFine);
    return result;
}

void verifyMetricAndCoverage()
{
    GPUSceneGPUInstanceRecord instance;
    instance.worldMatrix = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    instance.identity[3] = GPUSceneGPUInstanceVisible;
    MeshletLODGroupRecord group{.sphere = {0,0,5,.2f}, .error = .01f};
    VirtualShadowMapLODProjection projection{.rowX = {.5f,0,0,0}, .rowY = {0,.5f,0,0}, .resolution = 256};
    const auto error = [&] { return virtualShadowMapLodPixelError(group, instance, {&projection, 1}); };
    check(std::abs(error() - .64f) < 1e-5f, "Orthographic texel metric does not match analytic error");
    instance.worldMatrix[4] = 2;
    check(error() >= .64f * 2.4142f, "Shear metric underestimates largest affine singular value");
    instance.worldMatrix[4] = 0;
    projection.rowX = {1,0,0,0}; projection.rowY = {0,1,0,0}; projection.rowW = {0,0,1,0}; projection.nearPlane = .1f;
    const float nearError = error(); group.sphere[2] = 50;
    check(error() < nearError * .11f && error() > 0, "Perspective distance did not reduce shadow texel error");
    group.sphere[2] = .2f;
    check(error() == std::numeric_limits<float>::max(), "Near-plane crossing must refine conservatively");
    group.sphere[2] = -5;
    check(error() == 0, "Entirely behind-light group requested refinement");
    group.sphere = {100,0,5,.2f};
    check(error() == 0, "Entirely outside-face group requested refinement");
    group.flags = kMeshletLODTerminalGroup; group.error = std::numeric_limits<float>::max();
    const auto terminal = buildVirtualShadowMapLodRequirements({&group, 1}, instance, {&projection, 1}, 1);
    check(terminal.needsFine == std::vector<uint8_t>{1} && terminal.maxProjectedError == 0,
        "Terminal stop sentinel polluted the projected error or omitted fallback");

    // Shared descendants require every parent before refinement. Coverage atoms
    // exercise the shadow override rather than borrowing the raster camera cut.
    std::vector<MeshletLODGroupRecord> groups;
    for (uint32_t i = 0; i < 6; ++i) {
        groups.push_back({.sphere = {0,0,5,1}, .error = .1f, .level = i < 3 ? 0u : i < 5 ? 1u : 2u,
            .flags = i == 2 || i == 5 ? kMeshletLODTerminalGroup : 0u});
    }
    const std::array<MeshletLODGroupRange, 6> ranges{{{0,2},{2,2},{4,1},{5,2},{7,1},{8,2}}};
    const std::array<uint32_t, 10> refined{UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX,0,1,0,3,4};
    const std::array<uint32_t, 10> coverage{1,2,4,8,16,1,12,2,13,2};
    const std::array<uint8_t, 6> wanted{1,1,0,1,1,0};
    std::vector<MeshletLODBVHNode> bvh;
    std::string reason;
    check(buildMeshletLodBvh(groups, bvh, reason), "Shadow reference BVH: " + reason);
    MeshletLODView raster;
    raster.eye = {10000,10000,10000,.1f}; raster.projection[3] = 1e9f;
    for (uint32_t mask = 0; mask < 16; ++mask) {
        std::array<uint8_t, 6> drawable{uint8_t(mask & 1),uint8_t((mask >> 1) & 1),1,
            uint8_t((mask >> 2) & 1),uint8_t((mask >> 3) & 1),1};
        const auto cut = selectStreamMeshletLodReference(groups, ranges, refined, drawable, instance, raster,
            UINT32_MAX, UINT32_MAX, {}, bvh, wanted);
        check(cut.valid, "Shadow override produced an invalid DAG cut: " + cut.reason);
        uint32_t covered = 0;
        for (uint32_t cluster : cut.selectedClusters) {
            check((covered & coverage[cluster]) == 0, "Shadow override duplicated surface coverage");
            covered |= coverage[cluster];
        }
        check(covered == 31, "Shadow override omitted surface coverage");
        if (mask == 15) { check(cut.selectedClusters == std::vector<uint32_t>{0,1,2,3,4}, "Raster BVH pruned shadow refinement"); }
    }
    VirtualShadowMapLODHierarchy hierarchy;
    check(buildVirtualShadowMapLODHierarchy(groups, ranges, refined, hierarchy, reason), "Adaptive DAG metadata: " + reason);
    const std::array<VirtualShadowMapLODProjection, 3> views{{
        {.rowX = {.5f,0,0,0}, .rowY = {0,.5f,0,0}, .resolution = 256},
        {.rowX = {1,0,0,-4}, .rowY = {0,1,0,0}, .rowW = {0,0,1,0}, .resolution = 256, .nearPlane = .1f},
        {.rowX = {1,0,0,4}, .rowY = {0,1,0,0}, .rowW = {0,0,-1,8}, .resolution = 512, .nearPlane = .1f}}};
    for (uint32_t mask = 0; mask < 16; ++mask) {
        const std::array<uint8_t, 6> drawable{uint8_t(mask & 1),uint8_t((mask >> 1) & 1),1,
            uint8_t((mask >> 2) & 1),uint8_t((mask >> 3) & 1),1};
        for (uint32_t transform = 0; transform < 8; ++transform) {
            instance.worldMatrix = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
            instance.worldMatrix[0] = transform & 1 ? -2.f : 1.f;
            instance.worldMatrix[4] = transform & 2 ? 1.5f : 0.f;
            instance.worldMatrix[14] = transform & 4 ? -4.9f : 0.f;
            for (const float threshold : {0.f, .01f, .1f, 1.f, 4.f, 1e9f}) {
                for (float budget : {std::numeric_limits<float>::max(), 0.f, .05f}) {
                    const auto flat = selectReferenceShadowLOD(hierarchy, drawable, instance, views, threshold, budget);
                    const auto adaptive = selectVirtualShadowMapLOD(hierarchy, drawable, instance, views, threshold, budget);
                    compareLODSelection(adaptive, flat);
                    uint32_t covered = 0;
                    for (uint32_t cluster : adaptive.cut.selectedClusters) {
                        check((covered & coverage[cluster]) == 0, "Adaptive DAG duplicated surface coverage");
                        covered |= coverage[cluster];
                    }
                    check(covered == 31, "Adaptive DAG lost a terminal branch");
                }
            }
        }
    }
    // Large world translations cancel against light projection offsets. The
    // hierarchy must remain conservative before side-plane rejection as well
    // as at the projected-error threshold.
    const std::array<uint8_t, 6> resident{1,1,1,1,1,1};
    for (const float translation : {-0x1p48f, -0x1p24f, 0x1p24f, 0x1p48f}) {
        instance.worldMatrix = {1,0,0,0, 1.5f,1,0,0, 0,0,1,0, translation,0,0,1};
        for (bool perspective : {false, true}) {
            VirtualShadowMapLODProjection translated{.rowX = {1,0,0,-translation},
                .rowY = {0,1,0,0}, .rowW = perspective ? std::array<float,4>{0,0,1,0} :
                    std::array<float,4>{0,0,0,1}, .resolution = 256, .nearPlane = perspective ? .1f : 0.f};
            for (float threshold : {.01f, 1.f, 16.f, 1e9f}) {
                compareLODSelection(selectVirtualShadowMapLOD(hierarchy, resident, instance, {&translated,1}, threshold),
                    selectReferenceShadowLOD(hierarchy, resident, instance, {&translated,1}, threshold));
            }
        }
    }

}

uint32_t verifyDepthGuardMetricAndCoverage()
{
    GPUSceneGPUInstanceRecord instance;
    instance.worldMatrix = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    instance.identity[3] = GPUSceneGPUInstanceVisible;
    VirtualShadowMapLODProjection projection{.rowX = {.5f,0,0,0}, .rowY = {0,.5f,0,0}, .resolution = 256};
    std::array<MeshletLODGroupRecord, 2> groups{{
        {.sphere = {0,0,5,.2f}, .error = .01f},
        {.sphere = {0,0,5,.2f}, .error = std::numeric_limits<float>::max(), .level = 1, .flags = kMeshletLODTerminalGroup}}};
    const std::array<MeshletLODGroupRange, 2> ranges{{{0,2},{2,1}}};
    const std::array<uint32_t, 3> refined{UINT32_MAX,UINT32_MAX,0};
    const std::array<uint8_t, 2> resident{1,1}, missingFine{0,1};
    uint32_t cases = 0;
    const auto verify = [&](float budget, std::span<const uint8_t> drawable, bool expectFine, bool expectRequest) {
        VirtualShadowMapLODHierarchy hierarchy;
        std::string reason;
        check(buildVirtualShadowMapLODHierarchy(groups, ranges, refined, hierarchy, reason), "Depth guard metadata: " + reason);
        // Deliberately allow the XY error. Only depth safety, near-plane
        // saturation, or residency may decide this fixture's refinement.
        const auto requirements = buildVirtualShadowMapLodRequirements(groups, instance, {&projection,1}, 1e9f, budget);
        const auto flat = selectReferenceShadowLOD(hierarchy, drawable, instance, {&projection,1}, 1e9f, budget);
        const auto adaptive = selectVirtualShadowMapLOD(hierarchy, drawable, instance, {&projection,1}, 1e9f, budget);
        compareLODSelection(adaptive, flat);
        check(requirements.needsFine == std::vector<uint8_t>{uint8_t(expectFine || expectRequest),1},
            "Depth guard requirement differs from the expected world-error boundary");
        check(adaptive.cut.selectedClusters == (expectFine ? std::vector<uint32_t>{0,1} : std::vector<uint32_t>{2}),
            "Depth guard chose the wrong complete cut");
        check(adaptive.cut.requestedGroups == (expectRequest ? std::vector<uint32_t>{0} : std::vector<uint32_t>{}),
            "Depth guard lost a missing fine-page request or requested an available group");
        ++cases;
    };
    verify(std::numeric_limits<float>::max(), resident, false, false);
    verify(.01f, resident, false, false);
    verify(std::nextafter(.01f, 0.f), resident, true, false);
    verify(0, resident, true, false);
    verify(0, missingFine, false, true);
    verify(0, resident, true, false); // Normal publication replaces the ancestor without holes.
    instance.worldMatrix[10] = -4;
    verify(.02f, resident, true, false); // Reflection and nonuniform scale affect depth error.
    instance.worldMatrix[10] = 1; instance.worldMatrix[4] = 2;
    verify(.02f, resident, true, false); // Shear must use a conservative affine norm.
    instance.worldMatrix[4] = 0;
    groups[0].error = 0;
    verify(0, resident, false, false); // Exact replacements do not consume the depth budget.
    groups[0].error = .01f; groups[0].sphere[0] = 100;
    verify(0, resident, false, false); // A view outside the group contributes no demand.
    groups[0].sphere[0] = 0;
    projection.rowX = {1,0,0,0}; projection.rowY = {0,1,0,0};
    projection.rowW = {0,0,1,0}; projection.nearPlane = .1f;
    groups[0].sphere[2] = .2f;
    verify(std::numeric_limits<float>::max(), resident, true, false);
    groups[0].sphere[2] = -5;
    verify(0, resident, false, false);
    return cases;
}

std::filesystem::path writeBunnyFixture(const std::filesystem::path& directory)
{
    std::filesystem::create_directories(directory);
    const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny";
    std::ifstream input(source / "scene.gltf");
    Json document; input >> document;
    std::filesystem::copy_file(source / "scene.bin", directory / "scene.bin", std::filesystem::copy_options::overwrite_existing);
    // Preserve the genuine cooked multi-level Bunny, enlarged to world meters.
    document["nodes"][0]["matrix"] = {20,0,0,0, 0,20,0,0, 0,0,20,0, .3368,-2.203,1.3,1};
    const float positions[]{-6,-5,0, 7,-5,0, 7,5,0, -6,5,0};
    const uint32_t indices[]{0,1,2,0,2,3};
    std::ofstream binary(directory / "Plane.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(positions), sizeof(positions));
    binary.write(reinterpret_cast<const char*>(indices), sizeof(indices));
    check(binary.good(), "Write shadow LOD receiver plane"); binary.close();
    const auto buffer = document["buffers"].size(), view = document["bufferViews"].size(), accessor = document["accessors"].size();
    document["buffers"].push_back({{"uri","Plane.bin"},{"byteLength",sizeof(positions) + sizeof(indices)}});
    document["bufferViews"].push_back({{"buffer",buffer},{"byteOffset",0},{"byteLength",sizeof(positions)}});
    document["bufferViews"].push_back({{"buffer",buffer},{"byteOffset",sizeof(positions)},{"byteLength",sizeof(indices)}});
    document["accessors"].push_back({{"bufferView",view},{"componentType",5126},{"count",4},{"type","VEC3"},{"min",{-6,-5,0}},{"max",{7,5,0}}});
    document["accessors"].push_back({{"bufferView",view + 1},{"componentType",5125},{"count",6},{"type","SCALAR"}});
    const auto mesh = document["meshes"].size(), node = document["nodes"].size();
    document["meshes"].push_back({{"primitives",Json::array({{{"attributes",{{"POSITION",accessor}}},{"indices",accessor + 1}}})}});
    document["nodes"].push_back({{"mesh",mesh},{"name","Shadow receiver"}});
    document["scenes"][0]["nodes"].push_back(node);
    const auto path = directory / "scene.gltf";
    std::ofstream(path) << document.dump(2);
    return path;
}

struct LODCapture
{
    std::vector<uint8_t> visibility;
    std::vector<uint32_t> atlas;
    std::vector<std::array<uint32_t, 4>> pages;
    std::array<uint32_t, 32> statistics{}, diagnostics{};
    uint32_t atlasWidth = 0, physicalPagesX = 0;
    Json selection;
};

uint32_t fullyResidentFinestClusterCount(const scene::MeshletStreamAsset& asset)
{
    uint32_t count = 0;
    for (const auto& sourceInstance : asset.instances()) {
        const auto& primitive = asset.primitives()[sourceInstance.primitiveIndex];
        std::vector<MeshletLODGroupRecord> groups;
        std::vector<MeshletLODGroupRange> ranges;
        std::vector<uint32_t> refined;
        for (uint32_t local = 0; local < primitive.groupCount; ++local) {
            const auto& source = asset.groups()[primitive.groupOffset + local];
            groups.push_back({.level = source.lodLevel, .flags = source.flags});
            ranges.push_back({uint32_t(refined.size()), source.clusterCount});
            for (uint32_t cluster = 0; cluster < source.clusterCount; ++cluster) {
                const uint32_t child = asset.refinedGroups()[source.clusterRefinedOffset + cluster];
                refined.push_back(child == UINT32_MAX ? child : child - primitive.groupOffset);
            }
        }
        const std::vector<uint8_t> available(groups.size(), 1);
        GPUSceneGPUInstanceRecord instance;
        instance.worldMatrix = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        instance.identity[3] = GPUSceneGPUInstanceVisible;
        const auto cut = selectStreamMeshletLodReference(groups, ranges, refined, available, instance,
            MeshletLODView{}, UINT32_MAX, UINT32_MAX, {}, {}, available);
        check(cut.valid, "Fully resident finest CPU reference is invalid: " + cut.reason);
        count += uint32_t(cut.selectedClusters.size());
    }
    return count;
}

Json verifyHierarchy(const scene::MeshletStreamAsset& asset)
{
    const auto& primitive = asset.primitives().front();
    std::vector<MeshletLODGroupRecord> groups;
    std::vector<MeshletLODGroupRange> ranges;
    std::vector<uint32_t> refined;
    for (uint32_t local = 0; local < primitive.groupCount; ++local) {
        const auto& source = asset.groups()[primitive.groupOffset + local];
        MeshletLODGroupRecord group;
        std::copy_n(source.boundsCenterRadius, 4, group.sphere.begin());
        group.error = source.maxQuadricError; group.level = source.lodLevel; group.flags = source.flags;
        groups.push_back(group); ranges.push_back({uint32_t(refined.size()), source.clusterCount});
        for (uint32_t cluster = 0; cluster < source.clusterCount; ++cluster) {
            const uint32_t child = asset.refinedGroups()[source.clusterRefinedOffset + cluster];
            refined.push_back(child == UINT32_MAX ? child : child - primitive.groupOffset);
        }
    }
    std::string reason;
    VirtualShadowMapLODHierarchy hierarchy;
    check(buildVirtualShadowMapLODHierarchy(groups, ranges, refined, hierarchy, reason), "Bunny hierarchy: " + reason);
    GPUSceneGPUInstanceRecord instance;
    instance.worldMatrix = {20,0,0,0, 0,20,0,0, 0,0,20,0, .3368f,-2.203f,1.3f,1};
    instance.identity[3] = GPUSceneGPUInstanceVisible;
    std::vector<uint8_t> drawable(groups.size(), 1);
    struct Workload {
        std::string name;
        std::vector<VirtualShadowMapLODProjection> views;
        float threshold = 1;
    };
    const VirtualShadowMapLODProjection ortho{.rowX = {.25f,0,0,0}, .rowY = {0,.25f,0,0}, .resolution = 256};
    const VirtualShadowMapLODProjection near{.rowX = {.1724767f,0,0,0}, .rowY = {0,.1724767f,0,0},
        .rowW = {0,0,-1,5}, .resolution = 256, .nearPlane = .001f};
    auto far = near; far.rowW[3] = 40;
    auto opposite = near; opposite.rowW = {0,0,1,5}; opposite.rowX[3] = -1;
    const std::array<Workload, 5> workloads{{{"Ortho1Texel",{ortho}}, {"SpotNear",{near}},
        {"SpotFar",{far}}, {"MultipleLights",{ortho,near,opposite}}, {"Finest",{ortho},0}}};
    Json report{{"scope","CPU selector only; prebuilt metadata, resident snapshot, no geometry upload or GPU work"},
        {"equivalenceCases",0},{"workloads",Json::array()}};
    uint32_t cases = 0, guardedCases = 0;
    uint64_t randomState = 0x72ac4513u;
    const auto random = [&] {
        randomState = randomState * 6364136223846793005ull + 1442695040888963407ull;
        return float(uint32_t(randomState >> 32)) / float(UINT32_MAX);
    };
    // Deterministic changes of transform, face overlap and residency exercise
    // the pruning bound against the unchanged per-group metric and flat DAG.
    for (uint32_t trial = 0; trial < 256; ++trial) {
        auto transformed = instance;
        transformed.worldMatrix[0] *= trial & 1 ? -1.f : 1.f;
        transformed.worldMatrix[4] = (random() - .5f) * 40;
        transformed.worldMatrix[8] = (random() - .5f) * 12;
        transformed.worldMatrix[12] += (random() - .5f) * 10;
        transformed.worldMatrix[14] += (random() - .5f) * 15;
        for (uint32_t group = 0; group < groups.size(); ++group) {
            drawable[group] = (groups[group].flags & kMeshletLODTerminalGroup) != 0 || random() > .35f;
        }
        const auto& work = workloads[trial % workloads.size()];
        const float threshold = trial % 16 == 0 ? 0.f : std::exp2(random() * 12 - 6);
        const auto flat = selectReferenceShadowLOD(hierarchy, drawable, transformed, work.views, threshold);
        const auto adaptive = selectVirtualShadowMapLOD(hierarchy, drawable, transformed, work.views, threshold);
        compareLODSelection(adaptive, flat); ++cases;
        for (float budget : {0.f, .0005f, .001f, .01f}) {
            compareLODSelection(selectVirtualShadowMapLOD(hierarchy, drawable, transformed, work.views, threshold, budget),
                selectReferenceShadowLOD(hierarchy, drawable, transformed, work.views, threshold, budget));
            ++guardedCases;
        }
    }
    report["equivalenceCases"] = cases;
    report["depthGuardEquivalenceCases"] = guardedCases;
    std::ranges::fill(drawable, uint8_t(1));
    for (const auto& work : workloads) {
        const auto reference = selectReferenceShadowLOD(hierarchy, drawable, instance, work.views, work.threshold);
        const auto selected = selectVirtualShadowMapLOD(hierarchy, drawable, instance, work.views, work.threshold);
        compareLODSelection(selected, reference);
        report["workloads"].push_back({{"name",work.name},{"selectedClusters",selected.cut.selectedClusters.size()},
            {"metricGroups",selected.metricGroups},{"hierarchyNodes",selected.hierarchyNodes},
            {"prunedGroups",selected.prunedGroups},{"topologyGroups",selected.topologyGroups}});
    }
    check(report["workloads"][2]["prunedGroups"].get<uint32_t>() > 0,
        "Far-light Bunny hierarchy did not prune any metric groups");
    return report;
}

Json compareSameCut(const LODCapture& actual, const LODCapture& reference, uint32_t pageSize)
{
    check(actual.visibility == reference.visibility, "Same-cut cached/forced visibility differs");
    check(actual.pages.size() == reference.pages.size(), "Same-cut page table sizes differ");
    Json diagnostics{{"comparedDepthTexels",0},{"unadmittedValidityDifferences",Json::array()}};
    uint64_t comparedDepthTexels = 0;
    for (size_t page = 0; page < actual.pages.size(); ++page) {
        const auto& a = actual.pages[page]; const auto& b = reference.pages[page];
        const std::string detail = " at virtual page " + std::to_string(page) + ": actual=" + Json(a).dump() + ", reference=" + Json(b).dump();
        check((a[1] & 3u) == (b[1] & 3u), "Same-cut page demand/admission differs" + detail);
        if ((a[1] & 2u) == 0) {
            // Reset preserves clean resident pages even when Mark no longer
            // requests them. ForceRedraw dirties those pages; Args redraws only
            // admitted pages. Projection never queries unadmitted mappings.
            if ((a[3] & 1u) != (b[3] & 1u)) {
                diagnostics["unadmittedValidityDifferences"].push_back({{"page",page},{"actual",a},{"reference",b}});
            }
            continue;
        }
        check((a[3] & 1u) == (b[3] & 1u), "Same-cut admitted page validity differs" + detail);
        if ((a[3] & 1u) == 0) { continue; }
        check(a[0] < actual.atlas.size() / (pageSize * pageSize) && b[0] < reference.atlas.size() / (pageSize * pageSize),
            "Same-cut admitted valid page has no physical depth mapping" + detail);
        for (uint32_t y = 0; y < pageSize; ++y) { for (uint32_t x = 0; x < pageSize; ++x) {
            const auto texel = [&](const LODCapture& capture, uint32_t physical) {
                return size_t(physical / capture.physicalPagesX * pageSize + y) * capture.atlasWidth + physical % capture.physicalPagesX * pageSize + x;
            };
            const uint32_t actualBits = actual.atlas[texel(actual, a[0])];
            const uint32_t referenceBits = reference.atlas[texel(reference, b[0])];
            check(actualBits == referenceBits,
                "Same-cut cached/forced D32 differs" + detail + ", texel=" + std::to_string(x) + "," + std::to_string(y) +
                ", actualBits=" + std::to_string(actualBits) + ", referenceBits=" + std::to_string(referenceBits) +
                ", ULP=" + std::to_string(std::max(actualBits, referenceBits) - std::min(actualBits, referenceBits)));
            ++comparedDepthTexels;
        }}
    }
    diagnostics["comparedDepthTexels"] = comparedDepthTexels;
    return diagnostics;
}

Json saveQuality(const std::filesystem::path& directory, const std::string& name,
    const LODCapture& capture, const LODCapture& reference)
{
    std::vector<uint8_t> image(kPixels * 4), difference(kPixels * 4);
    uint64_t absoluteError = 0, changed = 0, shadowed = 0;
    uint32_t maximum = 0;
    for (uint32_t i = 0; i < kPixels; ++i) {
        const auto delta = uint32_t(std::abs(int(capture.visibility[i]) - int(reference.visibility[i])));
        absoluteError += delta; maximum = std::max(maximum, delta); changed += delta != 0; shadowed += capture.visibility[i] < 255;
        for (uint32_t c = 0; c < 3; ++c) { image[i * 4 + c] = capture.visibility[i]; difference[i * 4 + c] = uint8_t(delta); }
        image[i * 4 + 3] = difference[i * 4 + 3] = 255;
    }
    std::string message;
    check(saveRgba8Png(directory / (name + ".png"), image.data(), kWidth, kHeight, message), "Write LOD visibility: " + message);
    check(saveRgba8Png(directory / (name + "-difference.png"), difference.data(), kWidth, kHeight, message), "Write LOD difference: " + message);
    return {{"changedPixels",changed},{"meanAbsoluteVisibilityError",double(absoluteError) / (kPixels * 255.)},
        {"maxVisibilityError",maximum},{"shadowedPixels",shadowed},{"visibilitySHA256",debug::debugSha256(capture.visibility)}};
}


LODCapture readCapture(Device& device, LODCommands& gpu, const VirtualShadowMapResult& result,
    const VirtualShadowMapSettings& settings, bool cancel)
{
    const uint32_t faceCount = 1;
    const std::array results{result};
    std::vector<LODCapture> captures(faceCount);
    std::vector<std::unique_ptr<Buffer>> readbacks;
    const uint64_t tableBytes = uint64_t(settings.clipmapCount * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis + 1) * 16u;
    for (uint32_t face = 0; face < faceCount; ++face) {
        const auto& shadow = results[face];
        auto& capture = captures[face];
        const auto atlasDesc = shadow.atlas->desc();
        capture.atlasWidth = atlasDesc.width; capture.physicalPagesX = shadow.physicalPagesX;
        capture.atlas.resize(size_t(atlasDesc.width) * atlasDesc.height); capture.pages.resize(tableBytes / 16u);
        capture.visibility.resize(kPixels);
        const uint64_t atlasBytes = capture.atlas.size() * sizeof(uint32_t);
        const uint64_t statsOffset = atlasBytes + tableBytes, diagnosticsOffset = statsOffset + sizeof(capture.statistics);
        const uint64_t visibilityOffset = diagnosticsOffset + sizeof(capture.diagnostics);
        auto readback = device.createBuffer({.size = visibilityOffset + kPixels, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback});
        check(readback, "LOD readback");
        const TextureBarrierDesc textures[]{
            {.texture = shadow.atlas, .oldLayout = TextureLayout::ShaderRead, .newLayout = TextureLayout::TransferSource,
                .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::DepthStencilWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}},
            {.texture = shadow.texture, .oldLayout = TextureLayout::ShaderRead, .newLayout = TextureLayout::TransferSource,
                .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}}};
        const BufferBarrierDesc buffers[]{
            {.buffer = shadow.pageTable, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}},
            {.buffer = shadow.statistics, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}},
            {.buffer = shadow.cacheDiagnostics, .before = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}, .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}}};
        check(gpu.commands->synchronize({.textures = textures, .buffers = buffers}), "LOD readback barriers");
        check((*readback)->slice({0, atlasBytes}).and_then([&](const auto& slice) {
            return gpu.commands->copyTextureToBuffer({.texture = shadow.atlas, .buffer = slice, .width = atlasDesc.width, .height = atlasDesc.height});
        }), "LOD atlas copy");
        check((*readback)->slice({visibilityOffset, kPixels}).and_then([&](const auto& slice) {
            return gpu.commands->copyTextureToBuffer({.texture = shadow.texture, .buffer = slice, .width = kWidth, .height = kHeight});
        }), "LOD visibility copy");
        for (const auto& [source, offset, bytes] : {std::tuple{shadow.pageTable, atlasBytes, tableBytes},
            std::tuple{shadow.statistics, statsOffset, uint64_t(sizeof(capture.statistics))},
            std::tuple{shadow.cacheDiagnostics, diagnosticsOffset, uint64_t(sizeof(capture.diagnostics))}}) {
            auto from = source->slice({0, bytes}), to = (*readback)->slice({offset, bytes});
            check(from, "LOD buffer source"); check(to, "LOD buffer destination");
            check(gpu.commands->copyBuffer(*from, *to), "LOD buffer copy");
        }
        const TextureBarrierDesc restore[]{
            {.texture = shadow.atlas, .oldLayout = TextureLayout::TransferSource, .newLayout = TextureLayout::ShaderRead,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}},
            {.texture = shadow.texture, .oldLayout = TextureLayout::TransferSource, .newLayout = TextureLayout::ShaderRead,
                .before = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}}};
        check(gpu.commands->synchronize({.textures = restore}), "Restore LOD atlas");
        readbacks.push_back(std::move(*readback));
    }
    gpu.finish(cancel);
    if (cancel) { return captures[0]; }
    for (uint32_t face = 0; face < faceCount; ++face) {
        auto& capture = captures[face];
        const uint64_t atlasBytes = capture.atlas.size() * sizeof(uint32_t);
        const uint64_t statsOffset = atlasBytes + tableBytes, diagnosticsOffset = statsOffset + sizeof(capture.statistics);
        const uint64_t visibilityOffset = diagnosticsOffset + sizeof(capture.diagnostics);
        readbacks[face]->invalidate(); const auto* bytes = static_cast<const std::byte*>(readbacks[face]->map());
        check(bytes != nullptr, "Map LOD capture");
        std::memcpy(capture.atlas.data(), bytes, atlasBytes); std::memcpy(capture.pages.data(), bytes + atlasBytes, tableBytes);
        std::memcpy(capture.statistics.data(), bytes + statsOffset, sizeof(capture.statistics));
        std::memcpy(capture.diagnostics.data(), bytes + diagnosticsOffset, sizeof(capture.diagnostics));
        std::memcpy(capture.visibility.data(), bytes + visibilityOffset, kPixels); readbacks[face]->unmap();
    }
    return captures[0];
}

class VirtualShadowMapLODTest final : public RHITest
{
public:
    VirtualShadowMapLODTest() { name = "virtual_shadow_map_shadow_lod"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapLOD");
        std::filesystem::create_directories(directory);
        Json report{{"passed",false},{"states",Json::array()},{"streamWarmup",Json::array()},
            {"scope","Correctness and workload counts; no GPU timing or performance claim"},
            {"quality","Changed-cut hard-shadow differences are measured, not assumed bit-identical or bounded by geometry texel error"}};
        bench::ValidationRecorder validation;
        const auto save = [&] { std::ofstream(directory / "VirtualShadowMapLOD.json") << report.dump(2); };
        try {
            verifyMetricAndCoverage(); report["metricAndDAG"] = "passed"; report["largeTranslationCases"] = 32;
            report["depthGuardDAGCases"] = 1536;
            report["depthGuardMetricCases"] = verifyDepthGuardMetricAndCoverage(); save();
            const auto path = writeBunnyFixture(directory / "Fixture");
            const auto streamPath = directory / "Bunny.meshstream.bin";
            std::string log;
            check(scene::buildMeshletStreamAssetOffline({.sourcePath = path, .outputPath = streamPath,
                .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log), "Cook shadow LOD Bunny: " + log);
            scene::MeshletStreamAsset asset;
            check(asset.open(streamPath, log), "Open shadow LOD Bunny: " + log);
            check(asset.lodLevels().size() > 1 && asset.groups().size() > asset.terminalGroups().size(), "Bunny did not produce genuine multi-level geometry");
            const uint32_t expectedFinestClusters = fullyResidentFinestClusterCount(asset);
            report["asset"] = {{"pages",asset.pageCount()},{"groups",asset.groups().size()},{"levels",asset.lodLevels().size()}};
            report["asset"]["fullyResidentFinestClusters"] = expectedFinestClusters;
            report["hierarchyCPU"] = verifyHierarchy(asset); save();
            auto deviceResult = createDevice({.applicationName = "VSM shadow LOD",
                .enableValidation = context.enableValidation, .enableSynchronizationValidation = context.enableValidation,
                .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false,
                .enableRayTracingPipeline = false, .enableClusterAccelerationStructure = false,
                .validationSink = validation.sink()});
            if (hasError(deviceResult, Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and bindless"); }
            check(deviceResult, "Shadow LOD device"); auto& device = **deviceResult;
            LODCommands gpu(device);
            std::unique_ptr<Streamer> streamer;
            check(createStreamer(device, {.constantBufferSize = 1u << 20}).transform([&](auto s) { streamer = std::move(s); }), "LOD streamer");
            std::unique_ptr<Texture> depth; std::unique_ptr<TextureView> depthView;
            check(device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::D32Sfloat, .width = kWidth, .height = kHeight})
                .transform([&](auto t) { depth = std::move(t); }), "LOD receiver depth");
            check(device.createTextureView(*depth, {.format = Format::D32Sfloat})
                .transform([&](auto v) { depthView = std::move(v); }), "LOD receiver view");
            auto upload = device.createBuffer({.size = kPixels * sizeof(float), .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload});
            check(upload, "LOD receiver upload");
            auto* values = static_cast<float*>((*upload)->map()); check(values != nullptr, "Map LOD receiver");
            std::fill(values, values + kPixels, (20.f - 8.f) / (20.f - .01f)); (*upload)->flush(); (*upload)->unmap();
            gpu.begin();
            const TextureBarrierDesc begin{.texture = depth.get(), .oldLayout = TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
            check(gpu.commands->synchronize({.textures = {&begin, 1}}), "LOD depth upload barrier");
            check((*upload)->slice().and_then([&](const auto& slice) {
                return gpu.commands->copyBufferToTexture({.texture = depth.get(), .buffer = slice, .width = kWidth, .height = kHeight});
            }), "LOD receiver copy");
            const TextureBarrierDesc ready{.texture = depth.get(), .oldLayout = TextureLayout::TransferDestination,
                .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
            check(gpu.commands->synchronize({.textures = {&ready, 1}}), "LOD receiver ready"); gpu.finish();

            for (bool streamed : {false, true}) {
                const std::string label = streamed ? "Stream" : "Resident";
                scene::SceneDocument scene;
                check(streamed ? scene.loadStreamMetadata(path) : scene.load(path), "Load LOD scene");
                std::shared_ptr<MeshletStreamRuntime> runtime;
                if (streamed) {
                    runtime = std::make_shared<MeshletStreamRuntime>();
                    check(runtime->initialize(device, {.sourcePath = path, .streamAssetPath = streamPath,
                        .maxResidentBytes = 64ull << 20, .maxResidentPages = 128, .maxLockedFallbackPages = 128,
                        .maxGpuPageRequests = 1024, .maxGpuPageUnloadRequests = 1024, .maxActiveGroups = 4096,
                        .maxRasterCandidates = 4096, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
                        .pageLoadConcurrency = 0, .queuedFrameCount = 1}, log), "LOD runtime: " + log);
                    check(runtime->syncRuntimeScene(scene, log), "LOD stream mapping: " + log);
                    for (uint32_t attempt = 0; attempt < 32 && !runtime->sceneReady(); ++attempt) {
                        gpu.begin(); check(streamer->beginFrame(gpu.frame), "LOD initial frame");
                        check(runtime->cmdLoadInitialResources(*gpu.commands, *streamer), "LOD initial load");
                        gpu.finish(); streamer->endFrame(); runtime->prepareMaintenance();
                    }
                    check(runtime->sceneReady(), "LOD terminal pages never became ready");
                }
                RenderView camera;
                const auto setCamera = [&](bool offscreen) {
                    const float x = offscreen ? 4.5f : 1.5f;
                    check(camera.setCamera({.eye = {x,0,8}, .center = {x,0,0}, .nearPlane = .01f,
                        .farPlane = 20, .orthoHeight = offscreen ? 3.f : 8.f, .orthographic = true}), "LOD camera");
                };
                setCamera(false);
                VirtualShadowMapLight light; light.direction = {2,0,-1};
                VirtualShadowMapSettings settings;
                settings.pageSize = 32; settings.virtualPagesPerAxis = 8; settings.clipmapCount = 3;
                settings.firstClipmapExtent = 4; settings.physicalPageCount = 256;
                settings.cacheEnabled = true; settings.collectGPUWork = true; settings.maxWorkItems = 1u << 20;
                VirtualShadowMapCasterSettings casterSettings;
                VirtualShadowMapCasterCache casterCache;
                std::array<VirtualShadowMaps, 4> maps;
                const auto selectionJSON = [](const VirtualShadowMapCasterResources& selected) {
                    const auto& d = selected.shadowLOD;
                    Json identities = Json::array();
                    for (const auto& state : selected.states) {
                        identities.push_back({state.id,state.contentKey,state.lodKey,state.triangleFirst,state.triangleCount});
                    }
                    return Json{{"selectedClusters",d.selectedClusters},{"triangleSlots",d.selectedTriangleSlots},
                        {"requestedPages",d.requestedPages},{"blockedPages",d.blockedPages},{"fallbackCasters",d.fallbackCasters},
                        {"protectedCasters",d.protectedCasters},{"testedGroups",d.testedGroups},{"casterIdentities",identities},
                        {"selectedLevels",d.selectedLevels},{"selectedClusterIDs",d.selectedClusterIDs},
                        {"metadataBuilds",d.metadataBuilds},{"metadataCacheHits",d.metadataCacheHits},{"reusedResidentCut",d.reusedResidentCut},
                        {"metricGroups",d.metricGroups},{"hierarchyNodes",d.hierarchyNodes},
                        {"prunedGroups",d.prunedGroups},{"topologyGroups",d.topologyGroups}};
                };
                const auto prepare = [&](float threshold, bool freeze, bool pump, float maxWorldError, auto&& consume) {
                    gpu.begin();
                    if (runtime) {
                        check(runtime->syncRuntimeScene(scene, log), "LOD synchronize stream: " + log);
                        if (pump) {
                            check(streamer->beginFrame(gpu.frame), "LOD stream frame");
                            MeshletStreamFrameDesc frame;
                            frame.width = kWidth; frame.height = kHeight;
                            // Raster snapshot freeze retains source mappings and
                            // blocks shadow requests without requiring RT BLAS.
                            // freezeSourceResidency also rebuilds RT representations.
                            frame.freezeRasterSnapshot = freeze;
                            // Publish through the normal stream frame while the primary
                            // view looks away. No visibility preparation or draw follows.
                            frame.camera.eye = {100,100,100}; frame.camera.center = {100,100,101};
                            check(runtime->cmdBeginFrame(*gpu.commands, *streamer, frame), "LOD stream publication");
                            check(streamer->copyStreamedData(*gpu.commands), "LOD copy stream uploads");
                            check(runtime->cmdPreTraversal(*gpu.commands, frame), "LOD publish page table");
                            check(runtime->cmdPostTraversal(*gpu.commands), "LOD post traversal");
                            check(runtime->cmdEndFrame(*gpu.commands), "LOD end stream frame");
                        }
                        check(runtime->cmdPrepareShadowGeometry(*gpu.commands), "LOD shadow geometry access");
                    }
                    auto stream = runtime ? runtime->shadowGeometryResources() : MeshletStreamShadowGeometryView{};
                    stream.owner = runtime;
                    auto base = casterCache.prepare(device, *gpu.commands, scene, runtime ? &stream : nullptr, log, casterSettings);
                    check(base, "Prepare LOD base: " + log);
                    const auto view = camera.constants(gpu.index, kWidth, kHeight, kWidth, kHeight);
                    const auto projections = VirtualShadowMaps::makeShadowLODProjections(base->bounds, view, light, settings, log);
                    check(projections, "LOD light projections: " + log);
                    auto selected = casterCache.prepareShadowLOD(device, *gpu.commands, scene,
                        runtime ? &stream : nullptr, *base, casterSettings, *projections, threshold, log, maxWorldError);
                    check(selected, "Select shadow LOD: " + log);
                    if (!streamed && threshold == 1e9f) {
                        // A wider error budget recomputes the same terminal cut. This
                        // must preserve the uploaded geometry even across a cache miss.
                        auto same = casterCache.prepareShadowLOD(device, *gpu.commands, scene, nullptr, *base, casterSettings,
                            *projections, 2e9f, log, maxWorldError);
                        check(same && selected->owner != base->owner && same->owner == selected->owner &&
                            !same->shadowLOD.reusedResidentCut && same->shadowLOD.selectedClusterIDs == selected->shadowLOD.selectedClusterIDs,
                            "Unchanged resident replacement cut repacked its geometry after a budget change");
                        // Restore the recorded budget so the next comparison arm
                        // can exercise an identical-input cache hit.
                        auto restored = casterCache.prepareShadowLOD(device, *gpu.commands, scene, nullptr, *base, casterSettings,
                            *projections, threshold, log, maxWorldError);
                        check(restored && restored->owner == selected->owner &&
                            restored->shadowLOD.selectedClusterIDs == selected->shadowLOD.selectedClusterIDs,
                            "Restoring the resident probe budget changed its geometry");
                    }
                    auto result = consume(*base, *selected, view);
                    if (runtime && pump) { streamer->endFrame(); runtime->prepareMaintenance(); }
                    return result;
                };
                const auto probe = [&](float threshold, bool freeze) {
                    return prepare(threshold, freeze, true, std::numeric_limits<float>::max(), [&](const auto&, const auto& selected, const auto&) {
                        const auto result = selectionJSON(selected); gpu.finish(); return result;
                    });
                };
                if (runtime) {
                    const auto frozen = probe(0, true);
                    report["streamWarmup"].push_back({{"phase","FrozenOffscreen"},
                        {"freezeMode","rasterSnapshot"},{"selection",frozen}}); save();
                    check(frozen.at("requestedPages") == 0, "Frozen shadow traversal enqueued streaming requests");
                    check(frozen.at("blockedPages").get<uint32_t>() > 0, "Freeze did not exercise missing fine pages");
                    const uint32_t frozenClusters = frozen.at("selectedClusters");
                    bool converged = false; uint64_t requested = 0;
                    for (uint32_t frame = 0; frame < 40; ++frame) {
                        const auto selected = probe(0, false);
                        requested += selected.at("requestedPages").get<uint32_t>();
                        report["streamWarmup"].push_back({{"phase","OffscreenRefine"},{"frame",frame},{"selection",selected}}); save();
                        if (selected.at("selectedClusters").get<uint32_t>() > frozenClusters &&
                            selected.at("selectedClusters").get<uint32_t>() == expectedFinestClusters &&
                            selected.at("fallbackCasters") == 0 && selected.at("requestedPages") == 0) { converged = true; break; }
                    }
                    check(requested > 0 && converged, "Offscreen shadow-only streaming failed to request and refine");
                }
                // The same fully available Bunny must respond to light distance,
                // independently of the receiver camera and primary stream view.
                const auto directional = light;
                light.type = VirtualShadowMapLightType::Spot; light.direction = {0,0,-1};
                light.position = {0,0,5}; light.outerConeAngle = 1.4f; light.range = 100;
                const auto nearSelection = probe(1, false);
                light.position[2] = 40;
                const auto farSelection = probe(1, false);
                report["lightDistance"][label] = {{"near",nearSelection},{"far",farSelection}}; save();
                check(nearSelection.at("selectedClusters").get<uint32_t>() > farSelection.at("selectedClusters").get<uint32_t>(),
                    "Light near/far distance did not change actual Bunny cluster selection");
                light = directional;
                const auto record = [&](uint32_t arm, float threshold, bool cancel = false, bool baseReference = false,
                    float maxWorldError = std::numeric_limits<float>::max()) {
                    return prepare(threshold, false, false, maxWorldError, [&](const auto& base, const auto& selected, const auto& view) {
                        auto options = settings; options.cacheLayering = arm == 0; options.forceRedraw = arm >= 2;
                        auto result = maps[arm].record(device, *gpu.commands, *depthView, view, light,
                            baseReference ? base : selected, scene.materials(), {}, options, log);
                        check(result, "Record shadow LOD: " + log);
                        auto capture = readCapture(device, gpu, *result, options, cancel);
                        capture.selection = selectionJSON(selected);
                        return capture;
                    });
                };
                const auto compare = [&](const std::string& state, float threshold, const LODCapture* reference,
                    float maxWorldError = std::numeric_limits<float>::max()) {
                    report["activeState"] = label + "-" + state; validation.phase(label + "-" + state);
                    const auto layered = record(0, threshold, false, false, maxWorldError), single = record(1, threshold, false, false, maxWorldError),
                        forced = record(2, threshold, false, false, maxWorldError);
                    const auto snapshot = [](const LODCapture& capture) {
                        return Json{{"pages",capture.pages},{"statistics",capture.statistics},{"diagnostics",capture.diagnostics},
                            {"selection",capture.selection},{"visibilitySHA256",debug::debugSha256(capture.visibility)}};
                    };
                    report["captureComparisons"].push_back({{"name",label + "-" + state},
                        {"layered",snapshot(layered)},{"single",snapshot(single)},{"forced",snapshot(forced)}}); save();
                    for (const auto* other : {&single, &forced}) {
                        for (const char* key : {"selectedClusters","triangleSlots","selectedLevels","selectedClusterIDs",
                            "requestedPages","blockedPages","fallbackCasters","protectedCasters","casterIdentities"}) {
                            check(layered.selection.at(key) == other->selection.at(key), std::string("A/B shadow cut differs: ") + key);
                        }
                    }
                    if (streamed) {
                        check(single.selection.at("metadataBuilds") == 0 && single.selection.at("metadataCacheHits").get<uint32_t>() > 0,
                            "Repeated stream selection did not reuse immutable metadata");
                    } else {
                        check(single.selection.at("reusedResidentCut").get<bool>() && single.selection.at("testedGroups") == 0 &&
                            single.selection.at("hierarchyNodes") == 0 && single.selection.at("metadataBuilds") == 0,
                            "Identical resident projections did not reuse the published cut");
                    }
                    report["captureComparisons"].back()["layeredVsSingle"] = compareSameCut(layered, single, settings.pageSize);
                    report["captureComparisons"].back()["layeredVsForced"] = compareSameCut(layered, forced, settings.pageSize);
                    save();
                    check(layered.statistics[5] == 0 && forced.statistics[5] == 0, "LOD correctness fixture overflowed build work");
                    check(layered.diagnostics[kVSMCacheDynamicCasters] == 0 && layered.diagnostics[kVSMCachePromotedCasters] == 0 &&
                        layered.diagnostics[kVSMCacheChangedCasters] == 0,
                        "LOD-only change promoted an authored static caster into dynamic cache");
                    auto quality = saveQuality(directory, label + "-" + state, layered, reference ? *reference : layered);
                    report["states"].push_back({{"name",label + "-" + state},{"threshold",threshold},{"maxWorldError",maxWorldError},
                        {"selection",layered.selection},{"layeredDiagnostics",layered.diagnostics},
                        {"layeredStatistics",layered.statistics},{"forcedStatistics",forced.statistics},{"quality",quality}}); save();
                    check(quality.at("shadowedPixels").get<uint64_t>() > 0, "LOD fixture did not produce a visible shadow");
                    return layered;
                };
                const auto finest = compare("Finest", 0, nullptr);
                if (!streamed) { compareSameCut(finest, record(3, 0, false, true), settings.pageSize); }
                const auto warm = compare("FinestWarm", 0, &finest);
                check(warm.diagnostics[kVSMCacheStaticDirtyFinePages] == 0 && warm.diagnostics[kVSMCacheStaticDirtyCoarsePages] == 0,
                    "Unchanged LOD cut redrew static depth");
                const auto terminal = compare("TerminalBaseline", 1e9f, &finest);
                check(terminal.diagnostics[kVSMCacheLODChangedCasters] > 0,
                    "Finest-to-terminal cut change did not report independent LOD invalidation");
                check(terminal.selection.at("selectedClusters").get<uint32_t>() < finest.selection.at("selectedClusters").get<uint32_t>(),
                    "Real Bunny did not reduce selected clusters for the coarse shadow cut");
                if (streamed) { compareSameCut(terminal, record(3, 1e9f, false, true), settings.pageSize); }
                const auto texelBudget = compare("TexelBudget", 1, &finest);
                const auto guarded = compare("DepthGuard", 1, &finest, settings.shadowBias * .5f);
                check(guarded.selection.at("triangleSlots").get<uint32_t>() >= texelBudget.selection.at("triangleSlots").get<uint32_t>(),
                    "Depth guard reduced rather than preserved or refined the Bunny geometry");
                const auto guardedWarm = compare("DepthGuardWarm", 1, &finest, settings.shadowBias * .5f);
                check(guardedWarm.diagnostics[kVSMCacheStaticDirtyFinePages] == 0 && guardedWarm.diagnostics[kVSMCacheStaticDirtyCoarsePages] == 0,
                    "Unchanged depth-guarded cut redrew static depth");
                (void)record(0, 1e9f, true);
                (void)compare("CancelledCutRollback", 1, &finest);
                (void)compare("ThresholdBackToFinest", 0, &finest);
                setCamera(true);
                const auto offscreenFinest = record(3, 0);
                (void)compare("OffscreenCaster", 1, &offscreenFinest);
            }
            report["validation"] = validation.snapshot();
            check(!validation.failed(), "Shadow LOD Vulkan validation error");
            report["passed"] = true; save();
            return RHITestResult::pass("Bunny shadow LOD: reference cut and D32/visibility checks, cached metadata, independent offscreen streaming, metric/DAG guards and cache/cancellation");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save();
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapLODTest);

// These records are read from the production uploads and selected GPU buffers.
// The oracle below evaluates every leaf and uses the independent stream DAG
// reference; it never calls the GPU selector's hierarchical traversal.
struct GPUViewLODProjection
{
    std::array<float, 4> rowX, rowY, rowZ, rowW, metric;
    std::array<uint32_t, 4> page;
    std::array<float, 4> originNear;
};
static_assert(sizeof(GPUViewLODProjection) == 112);

template<class T> std::vector<T> readLODHostBuffer(Buffer& buffer, uint32_t count)
{
    std::vector<T> result(count);
    if (count == 0) { return result; }
    const void* mapped = buffer.map(); check(mapped != nullptr, "Map immutable LOD metadata");
    std::memcpy(result.data(), mapped, result.size() * sizeof(T)); buffer.unmap();
    return result;
}

struct GPUViewLODMetadata
{
    std::vector<VirtualShadowMapLODGroupGPU> groups;
    std::vector<VirtualShadowMapLODInstanceGPU> instances;
    std::vector<VirtualShadowMapLODClusterGPU> clusters;
    std::vector<uint32_t> triangleClusters;
};

struct GPUViewLODFace
{
    LODCapture image;
    std::vector<uint32_t> selected, changed;
    std::vector<std::array<uint32_t, 32>> lod;
    std::vector<GPUViewLODProjection> projections;
};

struct GPUViewLODCapture
{
    std::vector<GPUViewLODFace> faces;
    GPUViewLODMetadata metadata;
    std::vector<std::array<uint32_t, 2>> pageTable;
    std::vector<uint32_t> pageRequests;
};

Json saveGPUViewLODDepthComparison(const std::filesystem::path& directory, uint32_t face,
    const LODCapture& actual, const LODCapture& reference, uint32_t pageSize, uint32_t pagesPerAxis)
{
    Json result{{"face",face},{"differentTexels",0},{"comparedTexels",0},{"relocatedPages",0},
        {"maxULP",0},{"maxAbsoluteDepthError",0},{"firstDifferences",Json::array()}};
    const uint32_t width = pagesPerAxis * pageSize;
    const uint32_t height = uint32_t((actual.pages.size() + pagesPerAxis - 1) / pagesPerAxis) * pageSize;
    std::array<std::vector<uint8_t>, 3> images;
    for (auto& image : images) { image.resize(uint64_t(width) * height * 4); }
    uint64_t different = 0, compared = 0;
    uint32_t relocated = 0, maxULP = 0;
    float maxError = 0;
    for (uint32_t page = 0; page < actual.pages.size(); ++page) {
        const auto& a = actual.pages[page]; const auto& b = reference.pages[page];
        if ((a[1] & 2u) == 0 || (b[1] & 2u) == 0 || (a[3] & 1u) == 0 || (b[3] & 1u) == 0) { continue; }
        relocated += a[0] != b[0];
        for (uint32_t y = 0; y < pageSize; ++y) { for (uint32_t x = 0; x < pageSize; ++x) {
            const auto bits = [&](const LODCapture& capture, uint32_t physical) {
                return capture.atlas[size_t(physical / capture.physicalPagesX * pageSize + y) * capture.atlasWidth +
                    physical % capture.physicalPagesX * pageSize + x];
            };
            const uint32_t aBits = bits(actual, a[0]), bBits = bits(reference, b[0]);
            const float aDepth = std::bit_cast<float>(aBits), bDepth = std::bit_cast<float>(bBits);
            const uint32_t ulp = std::max(aBits, bBits) - std::min(aBits, bBits); // D32 values are nonnegative.
            ++compared; different += aBits != bBits;
            maxULP = std::max(maxULP, ulp); maxError = std::max(maxError, std::abs(aDepth - bDepth));
            if (aBits != bBits && result["firstDifferences"].size() < 32) {
                result["firstDifferences"].push_back({{"virtualPage",page},{"texel",{x,y}},
                    {"actualPage",a},{"referencePage",b},{"actualBits",aBits},{"referenceBits",bBits},
                    {"actualDepth",aDepth},{"referenceDepth",bDepth},{"ULP",ulp}});
            }
            const uint64_t pixel = (uint64_t(page / pagesPerAxis * pageSize + y) * width + page % pagesPerAxis * pageSize + x) * 4;
            const uint8_t values[]{uint8_t(std::clamp(aDepth, 0.f, 1.f) * 255.f),
                uint8_t(std::clamp(bDepth, 0.f, 1.f) * 255.f), uint8_t(aBits != bBits ? 255 : 0)};
            for (uint32_t image = 0; image < images.size(); ++image) {
                for (uint32_t channel = 0; channel < 3; ++channel) { images[image][pixel + channel] = values[image]; }
                images[image][pixel + 3] = 255;
            }
        }}
    }
    result["differentTexels"] = different; result["comparedTexels"] = compared; result["relocatedPages"] = relocated;
    result["maxULP"] = maxULP; result["maxAbsoluteDepthError"] = maxError;
    const std::string name = "ZeroVsDisabled-Face" + std::to_string(face);
    if (different != 0) {
        constexpr const char* suffixes[]{"-actual-depth", "-reference-depth", "-depth-difference-mask"};
        for (uint32_t image = 0; image < images.size(); ++image) {
            std::string log;
            check(saveRgba8Png(directory / (name + suffixes[image] + ".png"), images[image].data(), width, height, log),
                "Write per-view depth evidence: " + log);
        }
        result["visibility"] = saveQuality(directory, name + "-visibility", actual, reference);
    }
    std::ofstream(directory / (name + ".json")) << result.dump(2);
    return result;
}

void verifyGPUViewLOD(const GPUViewLODCapture& capture, float threshold, float worldError,
    uint32_t pageSize, bool streamed, bool requestsEnabled)
{
    const auto& metadata = capture.metadata;
    uint64_t slots = 0;
    for (uint32_t cluster = 0; cluster < metadata.clusters.size(); ++cluster) {
        const auto& record = metadata.clusters[cluster];
        check(record.triangleFirst == slots, "Candidate upload has a gap or overlaps another cluster");
        for (uint32_t triangle = 0; triangle < record.triangleCount; ++triangle) {
            check(metadata.triangleClusters[record.triangleFirst + triangle] == cluster,
                "Build triangle-to-cluster mapping disagrees with the uploaded candidate");
        }
        slots += record.triangleCount;
    }
    check(slots == metadata.triangleClusters.size(), "Candidate slots omit geometry from the triangle map");
    std::vector<uint32_t> expectedRequests(capture.pageRequests.size());
    for (const auto& face : capture.faces) {
        check(face.selected.size() == metadata.clusters.size() * face.projections.size(), "GPU selected mask dimensions");
        for (uint32_t view = 0; view < face.projections.size(); ++view) {
            const auto& projection = face.projections[view];
            const VirtualShadowMapLODProjection metric{projection.rowX, projection.rowY, projection.rowW,
                projection.page[1] * pageSize, projection.originNear[3]};
            std::vector<uint32_t> expected(metadata.clusters.size());
            for (const auto& instance : metadata.instances) {
                const bool resident = (instance.flags & kVirtualShadowMapLODInstanceResident) != 0;
                bool base = (instance.flags & kVirtualShadowMapLODInstanceProtected) != 0 || instance.groupCount == 0 ||
                    (resident && threshold == 0);
                if (!base) {
                    VirtualShadowMapLODHierarchy hierarchy;
                    std::vector<uint8_t> drawable(instance.groupCount, 1);
                    const uint32_t hierarchyFirst = instance.clusterOffset + instance.baseClusterCount;
                    for (uint32_t local = 0; local < instance.groupCount; ++local) {
                        const auto& group = metadata.groups[instance.groupOffset + local];
                        hierarchy.groups.push_back({.sphere = group.sphere, .error = group.error,
                            .level = group.level, .flags = group.flags});
                        hierarchy.ranges.push_back({group.clusterOffset - hierarchyFirst, group.clusterCount});
                        if (streamed) {
                            const uint32_t state = capture.pageTable[group.pageIndex][0] & 7u;
                            drawable[local] = state == 2u || state == 3u;
                        }
                    }
                    for (uint32_t cluster = hierarchyFirst; cluster < instance.clusterOffset + instance.clusterCount; ++cluster) {
                        hierarchy.refinedGroups.push_back(metadata.clusters[cluster].refinedGroup);
                    }
                    GPUSceneGPUInstanceRecord transform;
                    std::copy(instance.world0.begin(), instance.world0.end(), transform.worldMatrix.begin());
                    std::copy(instance.world1.begin(), instance.world1.end(), transform.worldMatrix.begin() + 4);
                    std::copy(instance.world2.begin(), instance.world2.end(), transform.worldMatrix.begin() + 8);
                    std::copy(instance.world3.begin(), instance.world3.end(), transform.worldMatrix.begin() + 12);
                    transform.identity[3] = GPUSceneGPUInstanceVisible;
                    const auto reference = selectReferenceShadowLOD(hierarchy, drawable, transform, {&metric, 1}, threshold, worldError);
                    check(reference.cut.valid, "Independent per-view DAG produced an incomplete cut: " + reference.cut.reason);
                    base = resident && std::ranges::all_of(reference.cut.selectedClusters, [&](uint32_t cluster) {
                        return metadata.clusters[hierarchyFirst + cluster].level == 0;
                    });
                    if (!base) {
                        for (uint32_t cluster : reference.cut.selectedClusters) { expected[hierarchyFirst + cluster] = 1; }
                    }
                    if (streamed && requestsEnabled) {
                        for (uint32_t group : reference.cut.requestedGroups) {
                            expectedRequests[metadata.groups[instance.groupOffset + group].pageIndex] = 1;
                        }
                        for (uint32_t cluster : reference.cut.selectedClusters) {
                            const auto& record = metadata.clusters[hierarchyFirst + cluster];
                            expectedRequests[metadata.groups[instance.groupOffset + record.group].pageIndex] = 1;
                        }
                    }
                }
                if (base) {
                    for (uint32_t local = 0; local < instance.baseClusterCount; ++local) { expected[instance.clusterOffset + local] = 1; }
                }
            }
            const auto actual = std::span(face.selected).subspan(view * metadata.clusters.size(), metadata.clusters.size());
            check(std::ranges::equal(actual, expected), "GPU per-view selected IDs differ from independent flat DAG oracle at view " + std::to_string(view));
            uint32_t selected = 0, triangles = 0;
            std::array<uint32_t, 16> levels{};
            for (uint32_t cluster = 0; cluster < expected.size(); ++cluster) {
                if (expected[cluster] == 0) { continue; }
                ++selected; triangles += metadata.clusters[cluster].triangleCount;
                ++levels[std::min(metadata.clusters[cluster].level, 15u)];
            }
            const auto& stats = face.lod[view];
            check(stats[0] == 3 && stats[1] == 1 && stats[2] == 1 && stats[4] == selected && stats[5] == triangles,
                "GPU per-view statistics disagree with selected cluster records");
            check(std::equal(levels.begin(), levels.end(), stats.begin() + 16), "GPU LOD level histogram disagrees with selection");
        }
    }
    // Feedback includes selected pages and may also touch resident wanted
    // ancestors. Every independently required missing/selected page must survive
    // the union over lights, point faces, mips and the coarse view.
    for (uint32_t page = 0; page < expectedRequests.size(); ++page) {
        check(!expectedRequests[page] || capture.pageRequests[page] != 0, "Per-view GPU feedback lost a required stream page");
        check(requestsEnabled || capture.pageRequests[page] == 0, "Frozen GPU LOD published page requests");
    }
}

class VirtualShadowMapGPUShadowLODTest final : public RHITest
{
public:
    VirtualShadowMapGPUShadowLODTest() { name = "virtual_shadow_map_gpu_shadow_lod"; type = RHITestType::Rendering; }

    RHITestResult run(RHITestContext& context) override
    {
        const auto directory = std::filesystem::absolute(context.outputDirectory / "VirtualShadowMapGPUShadowLOD");
        std::filesystem::create_directories(directory);
        Json report{{"passed",false},{"frames",Json::array()},{"streamWarmup",Json::array()}};
        bench::ValidationRecorder validation;
        const auto save = [&] { std::ofstream(directory / "VirtualShadowMapGPUShadowLOD.json") << report.dump(2); };
        try {
            const auto path = writeBunnyFixture(directory / "Fixture");
            const auto streamPath = directory / "Bunny.meshstream.bin";
            std::string log;
            check(scene::buildMeshletStreamAssetOffline({.sourcePath = path, .outputPath = streamPath,
                .compressionMode = scene::MeshletStreamPayloadCompression::Reference}, log), "Cook GPU LOD Bunny: " + log);
            auto created = createDevice({.applicationName = "VSM per-view GPU LOD", .enableValidation = context.enableValidation,
                .enableSynchronizationValidation = context.enableValidation, .enableBindlessDescriptorHeap = true, .enableMeshShader = true,
                .enableRayTracingAccelerationStructure = false, .enableRayQuery = false, .enableRayTracingPipeline = false,
                .enableClusterAccelerationStructure = false, .validationSink = validation.sink()});
            if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and bindless"); }
            check(created, "GPU LOD device"); auto& device = **created;
            LODCommands gpu(device);
            std::unique_ptr<Streamer> streamer;
            check(createStreamer(device, {.constantBufferSize = 1u << 20}).transform([&](auto value) { streamer = std::move(value); }), "GPU LOD streamer");
            auto depth = device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
                .format = Format::D32Sfloat, .width = kWidth, .height = kHeight});
            check(depth, "GPU LOD depth");
            auto depthView = device.createTextureView(**depth, {.format = Format::D32Sfloat}); check(depthView, "GPU LOD depth view");
            auto depthUpload = device.createBuffer({.size = kPixels * sizeof(float), .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload}); check(depthUpload, "GPU LOD depth upload");
            auto* depths = static_cast<float*>((*depthUpload)->map()); check(depths != nullptr, "Map GPU LOD depth");
            std::fill_n(depths, kPixels, (20.f - 8.f) / (20.f - .01f)); (*depthUpload)->flush(); (*depthUpload)->unmap();
            gpu.begin();
            const TextureBarrierDesc uploadBegin{.texture = depth->get(), .oldLayout = TextureLayout::Undefined,
                .newLayout = TextureLayout::TransferDestination, .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite}};
            check(gpu.commands->synchronize({.textures = {&uploadBegin, 1}}), "GPU LOD depth begin");
            check((*depthUpload)->slice().and_then([&](const auto& slice) {
                return gpu.commands->copyBufferToTexture({.texture = depth->get(), .buffer = slice, .width = kWidth, .height = kHeight});
            }), "GPU LOD depth copy");
            const TextureBarrierDesc depthReady{.texture = depth->get(), .oldLayout = TextureLayout::TransferDestination,
                .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
            check(gpu.commands->synchronize({.textures = {&depthReady, 1}}), "GPU LOD depth ready"); gpu.finish();
            RenderView camera;
            check(camera.setCamera({.eye = {1.5f,0,8}, .center = {1.5f,0,0}, .nearPlane = .01f,
                .farPlane = 20, .orthoHeight = 8, .orthographic = true}), "GPU LOD camera");
            VirtualShadowMapSettings settings;
            settings.pageSize = 32; settings.virtualPagesPerAxis = 8; settings.clipmapCount = 3;
            settings.firstClipmapExtent = 2; settings.physicalPageCount = 512; settings.maxWorkItems = 1u << 20;
            settings.cacheEnabled = true; settings.cacheLayering = true; settings.collectGPUWork = true;
            settings.shadowLOD = true; settings.shadowBias = .25f;
            std::vector<VirtualShadowMapLight> lights(9);
            lights[0].direction = {2,0,-1};
            lights[1].type = lights[2].type = VirtualShadowMapLightType::Spot;
            lights[1].position = {0,0,5}; lights[1].direction = {0,0,-1}; lights[1].outerConeAngle = 1.4f; lights[1].range = 100;
            lights[2] = lights[1]; lights[2].position[2] = 40;
            for (uint32_t face = 0; face < 6; ++face) {
                lights[face + 3].type = VirtualShadowMapLightType::PointFace;
                lights[face + 3].position = {1,0,4}; lights[face + 3].range = 100; lights[face + 3].face = face;
            }
            for (bool streamed : {false, true}) {
                scene::SceneDocument scene;
                check(streamed ? scene.loadStreamMetadata(path) : scene.load(path), "Load GPU LOD scene");
                std::shared_ptr<MeshletStreamRuntime> runtime;
                if (streamed) {
                    runtime = std::make_shared<MeshletStreamRuntime>();
                    runtime->setDebugReadbackEnabled(true);
                    check(runtime->initialize(device, {.sourcePath = path, .streamAssetPath = streamPath,
                        .maxResidentBytes = 64ull << 20, .maxResidentPages = 128, .maxLockedFallbackPages = 128,
                        .maxGpuPageRequests = 1024, .maxGpuPageUnloadRequests = 1024, .maxActiveGroups = 4096,
                        .maxRasterCandidates = 4096, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
                        .pageLoadConcurrency = 0, .queuedFrameCount = 1}, log), "GPU LOD runtime: " + log);
                    check(runtime->syncRuntimeScene(scene, log), "GPU LOD stream scene: " + log);
                    for (uint32_t attempt = 0; attempt < 32 && !runtime->sceneReady(); ++attempt) {
                        gpu.begin(); check(streamer->beginFrame(gpu.frame), "GPU LOD initial frame");
                        check(runtime->cmdLoadInitialResources(*gpu.commands, *streamer), "GPU LOD initial resources");
                        gpu.finish(); streamer->endFrame(); runtime->prepareMaintenance();
                    }
                    check(runtime->sceneReady(), "GPU LOD terminal pages never became ready");
                }
                VirtualShadowMapCasterCache casterCache;
                std::array<std::array<VirtualShadowMaps, 9>, 4> maps;
                std::array<VirtualShadowMapPhysicalPool, 4> pools;
                const auto residencyQueueCounts = [&] {
                    const auto snapshot = runtime->debugSnapshot(false);
                    Json counts;
                    for (const char* key : {"residentPageCount", "pendingPageCount", "queuedUploadCount",
                        "pendingPageLoadCount", "activePageLoadCount", "pendingPatchCount", "usedResidentBytes",
                        "frameGpuRequestCount", "totalCompletedPageLoadCount", "totalCompletedUploadCount", "totalUploadBytes"}) {
                        counts[key] = snapshot.at("stats").at(key);
                    }
                    counts["loadedPages"] = snapshot.at("throughput").at("loadedPages");
                    return counts;
                };
                const auto record = [&](uint32_t arm, float threshold, bool cancel = false, bool freeze = false,
                                        bool disabled = false, bool forceAll = false) {
                    report["activeRecord"] = {{"streamed",streamed},{"arm",arm},{"threshold",threshold},
                        {"cancel",cancel},{"freeze",freeze},{"disabled",disabled},{"forceAll",forceAll},
                        {"stage","Begin"},{"sourceReadyBefore",runtime ? runtime->sceneReady() : true}};
                    gpu.begin();
                    if (runtime) {
                        report["activeRecord"]["stage"] = "PublishStream";
                        check(runtime->syncRuntimeScene(scene, log), "GPU LOD sync stream");
                        check(streamer->beginFrame(gpu.frame), "GPU LOD stream frame");
                        MeshletStreamFrameDesc frame;
                        frame.width = kWidth; frame.height = kHeight; frame.freezeRasterSnapshot = freeze;
                        frame.camera.eye = {100,100,100}; frame.camera.center = {100,100,101};
                        check(runtime->cmdBeginFrame(*gpu.commands, *streamer, frame), "GPU LOD stream begin");
                        check(streamer->copyStreamedData(*gpu.commands), "GPU LOD stream uploads");
                        check(runtime->cmdPreTraversal(*gpu.commands, frame), "GPU LOD stream publish");
                        check(runtime->cmdPostTraversal(*gpu.commands), "GPU LOD stream post traversal");
                        check(runtime->cmdEndFrame(*gpu.commands), "GPU LOD stream end");
                        if (cancel) {
                            // Cancelling stream page-table patches invalidates
                            // the source publication and requires a rebuild.
                            // Commit that publication first, then cancel only
                            // VSM selection/feedback against the valid source.
                            gpu.finish(); streamer->endFrame(); runtime->prepareMaintenance();
                            check(runtime->sceneReady(), "Cancelled-shadow fixture did not commit a ready stream source");
                            report["activeRecord"]["sourcePublicationCommitted"] = true;
                            report["activeRecord"]["residencyBeforeShadow"] = residencyQueueCounts();
                            gpu.begin();
                        }
                        check(runtime->cmdPrepareShadowGeometry(*gpu.commands), "GPU LOD shadow geometry access");
                    }
                    report["activeRecord"]["stage"] = "PrepareShadowCandidates";
                    auto stream = runtime ? runtime->shadowGeometryResources() : MeshletStreamShadowGeometryView{};
                    stream.owner = runtime;
                    const VirtualShadowMapCasterSettings casterSettings{.instanceBuild = settings.instanceBuild};
                    auto base = casterCache.prepare(device, *gpu.commands, scene, runtime ? &stream : nullptr, log, casterSettings);
                    check(base, "GPU LOD base: " + log);
                    auto selected = casterCache.prepareGPUShadowLOD(device, *gpu.commands, scene, runtime ? &stream : nullptr, *base, casterSettings, log);
                    check(selected, "GPU LOD candidate publication: " + log);
                    check(!settings.instanceBuild || selected->instanceTable != UINT32_MAX, "GPU LOD instance Build omitted its instance table");
                    auto& hierarchy = *selected->gpuLOD;
                    if (runtime) {
                        report["activeRecord"]["requestsEnabled"] = hierarchy.context.requestsEnabled;
                        check(!cancel || hierarchy.context.requestsEnabled != 0,
                            "Cancelled shadow feedback test had requests disabled and exercised no feedback transaction");
                    }
                    GPUViewLODCapture capture;
                    capture.metadata.groups = readLODHostBuffer<VirtualShadowMapLODGroupGPU>(*hierarchy.metadataBuffers[0], hierarchy.context.groupStateCount);
                    capture.metadata.instances = readLODHostBuffer<VirtualShadowMapLODInstanceGPU>(*hierarchy.metadataBuffers[4], hierarchy.context.instanceCount);
                    capture.metadata.clusters = readLODHostBuffer<VirtualShadowMapLODClusterGPU>(*hierarchy.metadataBuffers[5], hierarchy.context.clusterCount);
                    capture.metadata.triangleClusters = readLODHostBuffer<uint32_t>(*hierarchy.metadataBuffers[6], selected->triangleCapacity);
                    auto output = device.createTexture({.usage = TextureUsageBits::Storage | TextureUsageBits::TransferSource,
                        .format = Format::R8Unorm, .width = kWidth, .height = kHeight, .layerCount = uint32_t(lights.size())});
                    check(output, "GPU LOD projection output");
                    const TextureBarrierDesc outputReady{.texture = output->get(), .oldLayout = TextureLayout::Undefined,
                        .newLayout = TextureLayout::General, .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                        .range = {0, 1, 0, uint32_t(lights.size())}};
                    check(gpu.commands->synchronize({.textures = {&outputReady, 1}}), "GPU LOD output ready");
                    std::vector<VirtualShadowMapFaceRequest> requests;
                    for (uint32_t face = 0; face < lights.size(); ++face) {
                        requests.push_back({.shadow = &maps[arm][face], .light = lights[face], .output = {.shadow = output->get(), .layer = face}});
                    }
                    auto frameSettings = settings;
                    frameSettings.shadowLODErrorTexels = threshold; frameSettings.shadowLOD = !disabled;
                    frameSettings.batchBuildRaster = arm != 1; frameSettings.forceRedraw = arm == 2 || forceAll;
                    const auto view = camera.constants(gpu.index, kWidth, kHeight, kWidth, kHeight);
                    report["activeRecord"]["stage"] = "RecordShadowViews";
                    auto results = VirtualShadowMaps::recordShared(pools[arm], device, *gpu.commands, **depthView, view, requests,
                        disabled ? *base : *selected, scene.materials(), {}, frameSettings, settings.physicalPageCount, true, log);
                    check(results, "Record GPU LOD: " + log);
                    std::vector<VirtualShadowMapProjectionGroup> groups(lights.size());
                    for (uint32_t face = 0; face < lights.size(); ++face) {
                        groups[face].faces[0] = &maps[arm][face]; groups[face].results[0] = (*results)[face]; groups[face].faceCount = 1;
                    }
                    check(VirtualShadowMaps::projectShared(*gpu.commands, groups, log), "Project GPU LOD: " + log);
                    struct Readback { std::unique_ptr<Buffer> buffer; void* destination; uint64_t bytes; };
                    std::vector<Readback> readbacks;
                    const auto copyBuffer = [&](Buffer& source, void* destination, uint64_t bytes) {
                        auto buffer = device.createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                            .memoryLocation = MemoryLocation::HostReadback}); check(buffer, "GPU LOD buffer readback");
                        const BufferBarrierDesc ready{.buffer = &source,
                            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                        check(gpu.commands->synchronize({.buffers = {&ready, 1}}), "GPU LOD buffer visibility");
                        auto from = source.slice({0, bytes}), to = (*buffer)->slice(); check(from, "GPU LOD source"); check(to, "GPU LOD destination");
                        check(gpu.commands->copyBuffer(*from, *to), "GPU LOD copy buffer");
                        const BufferBarrierDesc restore{.buffer = &source,
                            .before = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                            .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
                        check(gpu.commands->synchronize({.buffers = {&restore, 1}}), "GPU LOD restore buffer visibility");
                        readbacks.push_back({std::move(*buffer), destination, bytes});
                    };
                    capture.faces.resize(lights.size());
                    for (uint32_t face = 0; face < lights.size(); ++face) {
                        auto& result = (*results)[face]; auto& image = capture.faces[face].image; auto& item = capture.faces[face];
                        const uint32_t views = result.clipmapCount + 1;
                        image.pages.resize(result.clipmapCount * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis + 1);
                        copyBuffer(*result.pageTable, image.pages.data(), image.pages.size() * sizeof(image.pages[0]));
                        copyBuffer(*result.statistics, image.statistics.data(), sizeof(image.statistics));
                        copyBuffer(*result.cacheDiagnostics, image.diagnostics.data(), sizeof(image.diagnostics));
                        if (!disabled) {
                            check(result.lodClusterCount == hierarchy.context.clusterCount && result.lodInstanceCount == hierarchy.context.instanceCount,
                                "VSM result LOD layout differs from its candidate publication");
                            item.selected.resize(uint64_t(views) * result.lodClusterCount); item.changed.resize(uint64_t(views) * result.lodInstanceCount);
                            item.lod.resize(views); item.projections.resize(views);
                            copyBuffer(*result.lodSelection, item.selected.data(), item.selected.size() * sizeof(uint32_t));
                            copyBuffer(*result.lodChanges, item.changed.data(), item.changed.size() * sizeof(uint32_t));
                            copyBuffer(*result.lodStatistics, item.lod.data(), item.lod.size() * sizeof(item.lod[0]));
                            copyBuffer(*result.projections, item.projections.data(), item.projections.size() * sizeof(item.projections[0]));
                        }
                    }
                    if (runtime) {
                        capture.pageTable.resize(hierarchy.pageCount); capture.pageRequests.resize(hierarchy.pageCount);
                        copyBuffer(*stream.pageTableBuffer, capture.pageTable.data(), capture.pageTable.size() * sizeof(capture.pageTable[0]));
                        copyBuffer(*hierarchy.pageRequests, capture.pageRequests.data(), capture.pageRequests.size() * sizeof(uint32_t));
                    }
                    const auto atlasDesc = results->front().atlas->desc();
                    std::vector<uint32_t> atlas(uint64_t(atlasDesc.width) * atlasDesc.height * atlasDesc.layerCount);
                    std::vector<uint8_t> visibility(uint64_t(kPixels) * lights.size());
                    const auto copyTexture = [&](Texture& texture, TextureLayout oldLayout, void* destination, uint64_t bytes) {
                        auto buffer = device.createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                            .memoryLocation = MemoryLocation::HostReadback}); check(buffer, "GPU LOD texture readback");
                        const auto desc = texture.desc();
                        const TextureBarrierDesc ready{.texture = &texture, .oldLayout = oldLayout, .newLayout = TextureLayout::TransferSource,
                            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .range = {0, 1, 0, desc.layerCount}};
                        check(gpu.commands->synchronize({.textures = {&ready, 1}}), "GPU LOD texture visibility");
                        check((*buffer)->slice().and_then([&](const auto& slice) {
                            return gpu.commands->copyTextureToBuffer({.texture = &texture, .buffer = slice,
                                .width = desc.width, .height = desc.height, .layerCount = desc.layerCount});
                        }), "GPU LOD copy texture");
                        const TextureBarrierDesc restore{.texture = &texture, .oldLayout = TextureLayout::TransferSource, .newLayout = oldLayout,
                            .before = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
                            .range = {0, 1, 0, desc.layerCount}};
                        check(gpu.commands->synchronize({.textures = {&restore, 1}}), "GPU LOD restore texture");
                        readbacks.push_back({std::move(*buffer), destination, bytes});
                    };
                    copyTexture(*results->front().atlas, TextureLayout::ShaderRead, atlas.data(), atlas.size() * sizeof(uint32_t));
                    copyTexture(**output, TextureLayout::General, visibility.data(), visibility.size());
                    report["activeRecord"]["stage"] = cancel ? "CancelShadowRecording" : "SubmitShadowRecording";
                    gpu.finish(cancel);
                    if (!cancel) {
                        for (auto& readback : readbacks) {
                            readback.buffer->invalidate(); const void* data = readback.buffer->map(); check(data != nullptr, "Map GPU LOD result");
                            std::memcpy(readback.destination, data, readback.bytes); readback.buffer->unmap();
                        }
                        for (uint32_t face = 0; face < lights.size(); ++face) {
                            auto& image = capture.faces[face].image;
                            image.atlas = atlas; image.atlasWidth = atlasDesc.width; image.physicalPagesX = results->front().physicalPagesX;
                            image.visibility.assign(visibility.begin() + uint64_t(face) * kPixels, visibility.begin() + uint64_t(face + 1) * kPixels);
                        }
                        report["activeRecord"]["stage"] = "VerifyGPUSelection";
                        if (!disabled) { verifyGPUViewLOD(capture, threshold, settings.shadowBias * .5f, settings.pageSize, streamed, !freeze); }
                    }
                    if (runtime) {
                        if (!cancel) { streamer->endFrame(); }
                        runtime->prepareMaintenance();
                        report["activeRecord"]["sourceReadyAfter"] = runtime->sceneReady();
                        if (cancel) {
                            check(runtime->sceneReady(), "Cancelling shadow-only GPU feedback invalidated its committed stream source");
                            report["activeRecord"]["residencyAfterCancellation"] = residencyQueueCounts();
                            check(report["activeRecord"]["residencyBeforeShadow"] == report["activeRecord"]["residencyAfterCancellation"],
                                "Cancelled GPU feedback admitted requests or changed CPU residency/load queues");
                            report["streamCancellations"].push_back(report["activeRecord"]);
                        }
                    }
                    report["activeRecord"]["stage"] = "Complete";
                    return capture;
                };
                if (runtime) {
                    auto frozen = record(0, 0, false, true);
                    uint64_t blocked = 0;
                    for (const auto& face : frozen.faces) { for (const auto& stats : face.lod) { blocked += stats[7]; } }
                    check(blocked > 0, "Frozen GPU hierarchy did not retain a complete fallback for missing Bunny detail");
                    // A cancelled refinement must not enqueue IO. A committed
                    // source publication may advance lastRequestFrame (word 1),
                    // while the next frozen frame must retain residency and
                    // device offsets (word 0).
                    (void)record(0, 0, true, false);
                    // End-of-frame maintenance can already have run when the
                    // shadow tail is cancelled. Advance the stream alone, with
                    // no new valid shadow sample, then prove that maintenance
                    // actually visits the cancelled feedback ticket.
                    const auto queuesBeforeConsumer = residencyQueueCounts();
                    report["activeRecord"] = {{"streamed",true},{"stage","ConsumeCancelledShadowFeedback"}};
                    gpu.begin(); check(streamer->beginFrame(gpu.frame), "Cancelled feedback audit frame");
                    MeshletStreamFrameDesc auditFrame;
                    auditFrame.width = kWidth; auditFrame.height = kHeight;
                    auditFrame.camera.eye = {100,100,100}; auditFrame.camera.center = {100,100,101};
                    check(runtime->cmdBeginFrame(*gpu.commands, *streamer, auditFrame), "Cancelled feedback audit begin");
                    check(streamer->copyStreamedData(*gpu.commands), "Cancelled feedback audit uploads");
                    check(runtime->cmdPreTraversal(*gpu.commands, auditFrame), "Cancelled feedback audit publication");
                    check(runtime->cmdPostTraversal(*gpu.commands), "Cancelled feedback audit post traversal");
                    check(runtime->cmdEndFrame(*gpu.commands), "Cancelled feedback audit end");
                    gpu.finish(); streamer->endFrame();
                    CPUProfileRecorder consumerProfile;
                    runtime->prepareMaintenance(&consumerProfile);
                    const auto queuesAfterConsumer = residencyQueueCounts();
                    Json sections = Json::array();
                    bool consumedShadowQueue = false;
                    for (const auto& section : consumerProfile.sections) {
                        sections.push_back(section.name);
                        consumedShadowQueue |= section.name == "Shadow page feedback";
                    }
                    report["cancelledFeedbackConsumer"] = {{"before",queuesBeforeConsumer},{"after",queuesAfterConsumer},
                        {"profileSections",sections},{"enteredShadowFeedbackConsumer",consumedShadowQueue}};
                    save();
                    check(consumedShadowQueue, "Cancelled shadow feedback audit did not execute its consumer");
                    check(queuesBeforeConsumer == queuesAfterConsumer,
                        "Consuming cancelled shadow feedback admitted requests or changed residency/load queues");
                    const auto cancelled = record(0, 0, false, true);
                    check(cancelled.pageTable.size() == frozen.pageTable.size(), "Cancelled GPU feedback changed page-table dimensions");
                    uint32_t residencyChanges = 0, timestampChanges = 0;
                    report["cancelledStreamPageDifferences"] = Json::array();
                    for (uint32_t page = 0; page < frozen.pageTable.size(); ++page) {
                        const auto& before = frozen.pageTable[page]; const auto& after = cancelled.pageTable[page];
                        residencyChanges += before[0] != after[0]; timestampChanges += before[1] != after[1];
                        if (before != after) {
                            report["cancelledStreamPageDifferences"].push_back({{"page",page},{"before",before},{"after",after},
                                {"deviceOffsetAndStateChanged",before[0] != after[0]},{"lastRequestFrameChanged",before[1] != after[1]}});
                        }
                    }
                    report["cancelledStreamResidencyChanges"] = residencyChanges;
                    report["cancelledStreamRequestTimestampChanges"] = timestampChanges;
                    save();
                    check(residencyChanges == 0, "Cancelled GPU feedback changed stream residency or physical mapping");
                    bool converged = false; uint64_t requested = 0;
                    for (uint32_t attempt = 0; attempt < 48 && !converged; ++attempt) {
                        auto warm = record(0, 0);
                        uint64_t missing = 0;
                        for (const auto& face : warm.faces) { for (const auto& stats : face.lod) { missing += stats[6]; } }
                        requested += missing; converged = missing == 0;
                        report["streamWarmup"].push_back({{"frame",attempt},{"missingGroupReferences",missing}}); save();
                    }
                    check(converged && requested > 0, "Unfrozen offscreen GPU feedback did not request and converge to resident detail");
                }
                const auto compare = [&](const char* phase, float threshold, bool forceAll = false) {
                    validation.phase(std::string(streamed ? "Stream-" : "Resident-") + phase);
                    auto actual = record(0, threshold, false, false, false, forceAll);
                    const auto serial = record(1, threshold, false, false, false, forceAll);
                    const auto force = record(2, threshold);
                    Json frame{{"phase",phase},{"streamed",streamed},{"views",Json::array()},
                        {"compactBuild",settings.compactBuild},{"instanceBuild",settings.instanceBuild},
                        {"dirtyPageHierarchy",settings.dirtyPageHierarchy},{"pageWindowRaster",settings.pageWindowRaster},
                        {"clusterPageRaster",settings.clusterPageRaster},{"forceAll",forceAll}};
                    for (uint32_t face = 0; face < actual.faces.size(); ++face) {
                        check(actual.faces[face].selected == serial.faces[face].selected && actual.faces[face].selected == force.faces[face].selected,
                            "Batched, serial and forced VSM changed per-view LOD masks");
                        try {
                            compareSameCut(actual.faces[face].image, serial.faces[face].image, settings.pageSize);
                            compareSameCut(actual.faces[face].image, force.faces[face].image, settings.pageSize);
                        } catch (const std::exception& error) {
                            throw std::runtime_error(std::string(phase) + " face " + std::to_string(face) + ": " + error.what());
                        }
                        const auto& cache = actual.faces[face].image.diagnostics;
                        check(cache[kVSMCacheDynamicCasters] == 0 && cache[kVSMCachePromotedCasters] == 0 && cache[kVSMCacheChangedCasters] == 0,
                            "LOD-only selection classified a static caster as moving or authored content changed");
                        frame["views"].push_back({{"face",face},{"lod",actual.faces[face].lod},{"cache",cache}});
                    }
                    report["frames"].push_back(std::move(frame)); save(); return actual;
                };
                const auto finest = compare("Finest", 0);
                if (!streamed) {
                    // Compare both geometry paths from the same initial pool
                    // state. Reusing an arm invalidated by the LOD mode switch
                    // relocates physical tiles and changes raster interpolation.
                    const auto disabled = record(3, 0, false, false, true);
                    report["zeroDisabledDepth"] = Json::array();
                    for (uint32_t face = 0; face < finest.faces.size(); ++face) {
                        report["zeroDisabledDepth"].push_back(saveGPUViewLODDepthComparison(directory, face,
                            finest.faces[face].image, disabled.faces[face].image, settings.pageSize, settings.virtualPagesPerAxis));
                    }
                    save();
                    for (uint32_t face = 0; face < finest.faces.size(); ++face) {
                        try { compareSameCut(finest.faces[face].image, disabled.faces[face].image, settings.pageSize); }
                        catch (const std::exception& error) {
                            throw std::runtime_error("ZeroVsDisabled face " + std::to_string(face) + ": " + error.what());
                        }
                    }
                }
                const auto warm = compare("FinestWarm", 0);
                for (const auto& face : warm.faces) {
                    check(std::ranges::all_of(face.changed, [](uint32_t value) { return value == 0; }), "Unchanged per-view GPU masks invalidated static geometry");
                    check(face.image.diagnostics[kVSMCacheStaticDirtyFinePages] == 0 && face.image.diagnostics[kVSMCacheStaticDirtyCoarsePages] == 0,
                        "Unchanged GPU LOD masks redrew static depth");
                }
                const auto adaptive = compare("Adaptive", 1);
                bool differentLevels = false, differentLights = false, differentFaces = false;
                const uint32_t clusters = uint32_t(adaptive.metadata.clusters.size());
                const auto mask = [&](uint32_t face, uint32_t view) { return std::span(adaptive.faces[face].selected).subspan(uint64_t(view) * clusters, clusters); };
                for (uint32_t view = 1; view < adaptive.faces[0].projections.size(); ++view) { differentLevels |= !std::ranges::equal(mask(0, 0), mask(0, view)); }
                differentLights = !std::ranges::equal(mask(1, 0), mask(2, 0));
                for (uint32_t face = 4; face < lights.size(); ++face) { differentFaces |= !std::ranges::equal(mask(3, 0), mask(face, 0)); }
                check(differentLevels && differentLights && differentFaces, "Bunny did not exercise distinct clipmap, spot-light and point-face GPU cuts");
                uint64_t lodChanges = 0, dirty = 0;
                for (const auto& face : adaptive.faces) {
                    lodChanges += face.image.diagnostics[kVSMCacheLODChangedCasters];
                    dirty += face.image.diagnostics[kVSMCacheStaticDirtyFinePages] + face.image.diagnostics[kVSMCacheStaticDirtyCoarsePages];
                }
                check(lodChanges > 0 && dirty > 0, "LOD-only GPU mask changes failed to invalidate cached static depth");
                (void)record(0, 0, true);
                const auto rollback = compare("CancelledFinestRollback", 1);
                for (uint32_t face = 0; face < rollback.faces.size(); ++face) {
                    check(rollback.faces[face].selected == adaptive.faces[face].selected &&
                        std::ranges::all_of(rollback.faces[face].changed, [](uint32_t value) { return value == 0; }),
                        "Cancelled GPU cut replaced committed per-view selection");
                }
                (void)compare("FinestRetry", 0);
                const auto quality = saveQuality(directory, streamed ? "StreamAdaptive" : "ResidentAdaptive", adaptive.faces[0].image, finest.faces[0].image);
                report[streamed ? "streamQuality" : "residentQuality"] = quality;
                for (uint32_t mode = 0; mode < 3; ++mode) {
                    settings.compactBuild = mode < 2;
                    settings.dirtyPageHierarchy = mode < 2;
                    settings.instanceBuild = mode == 1;
                    settings.clusterPageRaster = mode == 1;
                    settings.pageWindowRaster = mode == 2;
                    constexpr const char* names[]{"CompactHierarchy", "InstancePackedHierarchy", "PageWindow"};
                    if (mode == 1 && !device.capabilities().computeSubgroupBallotArithmetic) {
                        report["unavailableModes"].push_back({{"streamed",streamed},{"mode",names[mode]},
                            {"reason","Packed raster requires compute subgroup ballot arithmetic"}});
                        continue;
                    }
                    // Mode flags deliberately preserve cache compatibility.
                    // Force all three arms once so every face executes the new
                    // Build/Raster variant before testing its warm-cache path.
                    const auto forced = compare((std::string(names[mode]) + "Forced").c_str(), 1, true);
                    check(forced.faces[0].image.statistics[16] > 0 && forced.faces[0].image.statistics[4] > 0,
                        std::string(names[mode]) + ": forced directional mode produced no dirty fine raster work");
                    for (uint32_t face = 0; face < forced.faces.size(); ++face) {
                        check(forced.faces[face].image.statistics[5] == 0, std::string(names[mode]) + ": fine work capacity overflow");
                        compareSameCut(forced.faces[face].image, adaptive.faces[face].image, settings.pageSize);
                    }
                    if (mode == 2) {
                        const uint64_t windowBytes = 16u + uint64_t(settings.maxWorkItems) * 16u +
                            uint64_t(std::min(uint32_t(forced.metadata.triangleClusters.size()), settings.maxWorkItems)) * 64u;
                        check(std::has_single_bit(settings.virtualPagesPerAxis) && windowBytes <= (128ull << 20),
                            "Page-window case fell outside the production activation contract");
                    }
                    const auto modeWarm = compare((std::string(names[mode]) + "Warm").c_str(), 1);
                    for (uint32_t face = 0; face < modeWarm.faces.size(); ++face) {
                        check(modeWarm.faces[face].selected == forced.faces[face].selected &&
                            std::ranges::all_of(modeWarm.faces[face].changed, [](uint32_t value) { return value == 0; }),
                            std::string(names[mode]) + ": unchanged mode invalidated per-view LOD selection");
                        check(modeWarm.faces[face].image.diagnostics[kVSMCacheStaticDirtyFinePages] == 0 &&
                            modeWarm.faces[face].image.diagnostics[kVSMCacheStaticDirtyCoarsePages] == 0,
                            std::string(names[mode]) + ": unchanged mode redrew static fine/coarse depth");
                    }
                }
                settings.compactBuild = settings.dirtyPageHierarchy = settings.instanceBuild = false;
                settings.clusterPageRaster = settings.pageWindowRaster = false;
            }
            report["validation"] = validation.snapshot(); check(!validation.failed(), "GPU LOD Vulkan validation error");
            report["passed"] = true; save();
            return RHITestResult::pass("Per-view GPU Bunny cuts match flat DAG oracle across clipmaps/lights/faces, batch/serial/forced depth, cache cancellation and offscreen stream convergence");
        } catch (const std::exception& error) {
            report["error"] = error.what(); report["validation"] = validation.snapshot(); save(); return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapGPUShadowLODTest);

} // namespace
} // namespace metallic::tests
