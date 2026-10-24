#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "RHITest.h"
#include "Runtime/Render/MeshletLOD.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace metallic::tests {
namespace {

using namespace render;

struct StreamLODFixture {
    std::vector<MeshletLODGroupRecord> groups;
    std::vector<MeshletLODGroupRange> ranges{{0, 2}, {2, 2}, {4, 1}, {5, 2}, {7, 1}, {8, 2}};
    std::vector<uint32_t> refined{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
        UINT32_MAX, 0, 1, 0, 3, 4};
    // Every original surface atom must be covered exactly once by every cut.
    std::array<uint32_t, 10> coverage{1, 2, 4, 8, 16, 1, 12, 2, 13, 2};
    std::vector<uint8_t> drawable = std::vector<uint8_t>(6, 1);
    GPUSceneGPUInstanceRecord instance;
    MeshletLODView view;

    explicit StreamLODFixture(bool large = false, uint32_t leafCount = 256)
    {
        for (uint32_t group = 0; group < ranges.size(); ++group) {
            groups.push_back({.sphere = {0, 0, 0, 1},
                .error = group < 3 ? .5f : group < 5 ? 2.f : 4.f,
                .level = group < 3 ? 0u : group < 5 ? 1u : 2u,
                .flags = group == 2 || group == 5 ? kMeshletLODTerminalGroup : 0u});
        }
        instance.worldMatrix = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        instance.identity[3] = GPUSceneGPUInstanceVisible;
        view.eye = {0, 0, 10, .1f};
        view.forward = {0, 0, -1, 1};
        view.projection = {1000, .577350269f, 100, 1.5f};
        if (large) {
            groups.clear();
            ranges.clear();
            refined.clear();
            uint32_t previousOffset = 0;
            for (uint32_t level = 0, count = leafCount; count != 0; ++level, count /= 2) {
                const uint32_t offset = static_cast<uint32_t>(groups.size());
                for (uint32_t group = 0; group < count; ++group) {
                    groups.push_back({.sphere = {0, 0, 0, 1}, .error = .002f * std::exp2(float(level)),
                        .level = level, .flags = count == 1 ? kMeshletLODTerminalGroup : 0u});
                    ranges.push_back({static_cast<uint32_t>(refined.size()), level == 0 ? 1u : 2u});
                    if (level == 0) {
                        refined.push_back(UINT32_MAX);
                    } else {
                        refined.push_back(previousOffset + group * 2);
                        refined.push_back(previousOffset + group * 2 + 1);
                    }
                }
                previousOffset = offset;
            }
            drawable.assign(groups.size(), 1);
            view.projection[3] = .005f;
        }
    }

    StreamMeshletLODReference select(uint32_t manual = UINT32_MAX, uint32_t capacity = UINT32_MAX) const
    {
        return selectStreamMeshletLodReference(groups, ranges, refined, drawable, instance, view, manual, capacity);
    }

    bool coversExactlyOnce(const StreamMeshletLODReference& cut) const
    {
        if (!cut.valid) { return false; }
        uint32_t covered = 0;
        for (uint32_t cluster : cut.selectedClusters) {
            if (cluster >= coverage.size() || (covered & coverage[cluster]) != 0) { return false; }
            covered |= coverage[cluster];
        }
        return covered == 31;
    }
};

class StreamMeshletLODReferenceTest final : public RHITest {
public:
    StreamMeshletLODReferenceTest() { type = RHITestType::Validation; name = "meshlet_lod_stream_reference_frontier"; }
    RHITestResult run(RHITestContext&) override
    {
        StreamLODFixture fixture;
        auto cut = fixture.select();
        if (!fixture.coversExactlyOnce(cut) || cut.selectedClusters != std::vector<uint32_t>{0, 1, 2, 3, 4}) {
            return RHITestResult::fail("fully resident stream cut did not refine");
        }
        fixture.drawable[3] = 0;
        cut = fixture.select();
        if (!fixture.coversExactlyOnce(cut) || cut.selectedClusters != std::vector<uint32_t>{4, 7, 8} ||
            cut.requestedGroups != std::vector<uint32_t>{3}) {
            return RHITestResult::fail("missing ancestor exposed a descendant with another resident parent");
        }
        for (uint32_t residency = 0; residency < 16; ++residency) {
            for (uint32_t group : {0u, 1u, 3u, 4u}) {
                const uint32_t bit = group < 2 ? group : group - 1;
                fixture.drawable[group] = (residency >> bit) & 1;
            }
            for (uint32_t manual : {UINT32_MAX, 0u, 1u, 2u, 31u}) {
                cut = fixture.select(manual);
                if (!fixture.coversExactlyOnce(cut)) {
                    return RHITestResult::fail("residency/LOD combination overlaps or omits a surface atom");
                }
            }
        }
        std::fill(fixture.drawable.begin(), fixture.drawable.end(), uint8_t{1});
        // Request all desired levels in one feedback round while only the
        // terminal cut is drawable. Arrival order must not expose orphans.
        for (uint32_t group : {0u, 1u, 3u, 4u}) { fixture.drawable[group] = 0; }
        cut = fixture.select(0);
        if (!fixture.coversExactlyOnce(cut) || cut.requestedGroups != std::vector<uint32_t>{0, 1, 3, 4} ||
            cut.selectedClusters != std::vector<uint32_t>{4, 8, 9}) {
            return RHITestResult::fail("detail requests waited for ancestors or changed the safe terminal cut");
        }
        for (uint32_t group : {0u, 1u}) { fixture.drawable[group] = 1; }
        cut = fixture.select(0);
        if (!fixture.coversExactlyOnce(cut) || cut.selectedClusters != std::vector<uint32_t>{4, 8, 9}) {
            return RHITestResult::fail("early arriving detail escaped the safe cut");
        }
        std::fill(fixture.drawable.begin(), fixture.drawable.end(), uint8_t{1});
        cut = fixture.select(UINT32_MAX, 2);
        if (!fixture.coversExactlyOnce(cut) || !cut.capacityExceeded || !cut.capacityFallback ||
            cut.selectedClusters != std::vector<uint32_t>{4, 8, 9}) {
            return RHITestResult::fail("capacity did not atomically fall back to the complete terminal cut");
        }
        cut = fixture.select(UINT32_MAX, 1);
        if (cut.valid || !cut.capacityExceeded || !cut.selectedClusters.empty()) {
            return RHITestResult::fail("an undersized terminal capacity exposed a partial cut");
        }
        fixture.drawable[5] = 0;
        cut = fixture.select();
        if (cut.valid || !cut.selectedClusters.empty() || cut.requestedGroups != std::vector<uint32_t>{5}) {
            return RHITestResult::fail("missing terminal page exposed orphan descendants");
        }
        fixture.drawable[5] = 1;
        fixture.drawable[3] = 0;
        const std::array<uint8_t, 6> available{1, 1, 1, 1, 1, 1};
        cut = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
            fixture.drawable, fixture.instance, fixture.view, UINT32_MAX, UINT32_MAX, available);
        if (!fixture.coversExactlyOnce(cut) || !cut.requestedGroups.empty() ||
            cut.selectedClusters != std::vector<uint32_t>{4, 7, 8}) {
            return RHITestResult::fail("pending upload replaced the fallback or generated a duplicate request");
        }
        fixture.drawable[3] = 1;
        fixture.view.projection[3] = 100;
        if (fixture.select().selectedClusters != std::vector<uint32_t>{4, 8, 9}) {
            return RHITestResult::fail("orthographic pixel error did not select cross-level terminal groups");
        }
        fixture.view.forward[3] = 0;
        fixture.view.eye[2] = .1f;
        if (fixture.select().selectedClusters != std::vector<uint32_t>{0, 1, 2, 3, 4}) {
            return RHITestResult::fail("near-plane error did not refine");
        }
        fixture.instance.identity[3] = 0;
        cut = fixture.select();
        if (!cut.valid || !cut.selectedClusters.empty() || !cut.requestedGroups.empty()) {
            return RHITestResult::fail("hidden instance produced geometry or page requests");
        }
        fixture.refined[8] = 5;
        if (fixture.select().valid) { return RHITestResult::fail("cyclic stream topology accepted"); }
        return RHITestResult::pass("80 residency/manual cuts preserve exact coverage; multi-parent gating, mixed-depth terminals, complete capacity fallback and near-plane projection");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLODReferenceTest);

// Keep transitions consecutive: State is initialized only once for each GPU
// algorithm, so pruning must clear yesterday's finer active bits and masks.
void configureStreamLodCase(StreamLODFixture& fixture, bool large, uint32_t test,
    uint32_t& manual, uint32_t& capacity, std::vector<uint8_t>& available, uint32_t leafCount = 256)
{
    fixture = StreamLODFixture{large, leafCount};
    manual = UINT32_MAX;
    capacity = static_cast<uint32_t>(fixture.groups.size());
    if (!large) {
        if (test < 16) {
            for (uint32_t group : {0u, 1u, 3u, 4u}) {
                const uint32_t bit = group < 2 ? group : group - 1;
                fixture.drawable[group] = (test >> bit) & 1;
            }
        } else if (test < 32) {
            fixture.view.forward[3] = test % 2 ? 1.f : 0.f;
            fixture.view.eye[2] = test % 4 ? 30.f : .1f;
            fixture.view.projection[0] = test % 3 ? 1000.f : 500.f;
            fixture.view.projection[3] = test % 5 ? 1.5f : 100.f;
            fixture.instance.worldMatrix[4] = test % 3 ? 2.f : 0.f;
            fixture.instance.worldMatrix[0] = test % 2 ? -2.f : 1.f;
            if (test >= 28) { manual = test == 31 ? 31 : test - 28; }
        } else if (test == 32 || test == 33) {
            capacity = test == 32 ? 2 : 1;
        } else if (test == 34) {
            fixture.drawable[5] = 0;
        } else if (test == 35) {
            fixture.instance.identity[3] = 0;
        } else {
            fixture.drawable[3] = 0;
        }
        available = fixture.drawable;
        if (test == 36) { available[3] = 1; }
        return;
    }
    const uint32_t root = static_cast<uint32_t>(fixture.groups.size() - 1);
    switch (test) {
    case 0: case 2: case 4: case 6: case 8: case 10: case 25: case 27: manual = 0; break;
    case 1: case 24: manual = 31; break;
    case 3: fixture.instance.identity[3] = 0; break;
    case 5: case 7: fixture.drawable[root - 1] = 0; break;
    case 9: fixture.drawable[root] = 0; break;
    case 11: case 18: fixture.view.projection[3] = 16.f * float(leafCount) / 256.f; break;
    case 12: fixture.view.projection[3] = .005f; break;
    case 13:
        fixture.view.forward[3] = 0;
        fixture.instance.worldMatrix[12] = 300.f;
        fixture.view.projection[3] = 1.5f;
        break;
    case 14:
        fixture.view.forward[3] = 0;
        fixture.view.eye[2] = .1f;
        fixture.view.projection[3] = 16.f;
        break;
    case 15:
        fixture.instance.worldMatrix[4] = 3.f;
        fixture.instance.worldMatrix[9] = -2.f;
        fixture.view.projection[3] = .5f;
        break;
    case 16:
        fixture.instance.worldMatrix[0] = -4.f;
        fixture.instance.worldMatrix[5] = .5f;
        fixture.view.projection[3] = .5f;
        break;
    case 17: fixture.view.projection[3] = .5f; break;
    case 19:
        fixture.view.forward[3] = 0;
        fixture.view.projection[0] = 2160;
        fixture.view.projection[3] = .04f;
        break;
    case 20: manual = 3; break;
    case 21: manual = 6; break;
    case 22: capacity = 1; break;
    case 23: capacity = 0; break;
    case 26:
        fixture.drawable[root - 1] = 0;
        fixture.drawable[root - 3] = 0;
        break;
    default:
        fixture.view.forward[3] = test % 2 ? 1.f : 0.f;
        fixture.view.eye[2] = test % 3 ? 50.f : 2.f;
        fixture.instance.worldMatrix[0] = test % 2 ? -3.f : 2.f;
        fixture.instance.worldMatrix[4] = float(test % 3);
        fixture.view.projection[3] = .25f * float(test - 27);
        break;
    }
    available = fixture.drawable;
    if (test == 7) { available[root - 1] = 1; }
}

bool sameStreamCut(const StreamMeshletLODReference& lhs, const StreamMeshletLODReference& rhs)
{
    return lhs.valid == rhs.valid && lhs.capacityExceeded == rhs.capacityExceeded &&
        lhs.capacityFallback == rhs.capacityFallback && lhs.selectedClusters == rhs.selectedClusters &&
        lhs.requestedGroups == rhs.requestedGroups && lhs.activeGroups == rhs.activeGroups;
}

// The metric sphere of a replacement need not cover its refinement descendants.
// Keep one descendant on screen while its parent metric (and most of the tree)
// is behind the eye. An ordinary metric-sphere frustum test would drop that
// parent's raster-preserving demand.
StreamLODFixture rtDemandFixture(uint32_t leafCount)
{
    StreamLODFixture fixture(true, leafCount);
    for (auto& group : fixture.groups) { group.sphere[2] = 10000.f; }
    fixture.groups[0].sphere[2] = 0.f;
    fixture.view.forward[3] = 0.f;
    fixture.view.projection[3] = 1.5f;
    return fixture;
}

// Independent cut oracle: consume the brute-force leaf demand decisions, then
// apply dependency/residency gating in descending DAG order. This intentionally
// does not visit BVH nodes or substitute altered LOD errors for boolean demand.
StreamMeshletLODReference cutFromDemand(const StreamLODFixture& fixture,
    std::span<const uint8_t> wanted, std::span<const uint8_t> available)
{
    StreamMeshletLODReference result;
    const uint32_t count = static_cast<uint32_t>(fixture.groups.size());
    result.activeGroups.resize(count);
    std::vector<std::vector<uint32_t>> parents(count);
    for (uint32_t parent = 0; parent < count; ++parent) {
        const auto& range = fixture.ranges[parent];
        for (uint32_t cluster = 0; cluster < range.clusterCount; ++cluster) {
            const uint32_t child = fixture.refined[range.clusterOffset + cluster];
            if (child != UINT32_MAX) { parents[child].push_back(parent); }
        }
    }
    for (uint32_t reverse = count; reverse != 0; --reverse) {
        const uint32_t group = reverse - 1;
        if (!wanted[group]) { continue; }
        if (!fixture.drawable[group]) {
            if (!available[group]) { result.requestedGroups.push_back(group); }
            continue;
        }
        result.activeGroups[group] = std::all_of(parents[group].begin(), parents[group].end(),
            [&](uint32_t parent) { return result.activeGroups[parent] != 0; });
    }
    for (uint32_t group = 0; group < count; ++group) {
        if (!result.activeGroups[group]) { continue; }
        const auto& range = fixture.ranges[group];
        for (uint32_t cluster = 0; cluster < range.clusterCount; ++cluster) {
            const uint32_t index = range.clusterOffset + cluster;
            const uint32_t child = fixture.refined[index];
            if (child == UINT32_MAX || !result.activeGroups[child]) { result.selectedClusters.push_back(index); }
        }
    }
    std::sort(result.requestedGroups.begin(), result.requestedGroups.end());
    return result;
}

class StreamMeshletLODDemandBoundsTest final : public RHITest {
public:
    StreamMeshletLODDemandBoundsTest() { type = RHITestType::Validation; name = "meshlet_lod_stream_demand_bounds"; }
    RHITestResult run(RHITestContext&) override
    {
        const auto fixture = rtDemandFixture(256);
        std::vector<MeshletLODRefinementBounds> bounds;
        std::string reason;
        if (!buildMeshletLodRefinementBounds(fixture.groups, fixture.ranges, fixture.refined, bounds, reason)) {
            return RHITestResult::fail(reason);
        }
        uint32_t checked = 0;
        for (bool tiled : {false, true}) {
            const auto build = [&](std::span<const MeshletLODRefinementBounds> input, std::vector<MeshletLODBVHNode>& nodes) {
                return tiled ? buildMeshletLodTiles(fixture.groups, nodes, reason, input) :
                    buildMeshletLodBvh(fixture.groups, nodes, reason, input);
            };
            std::vector<MeshletLODBVHNode> nodes;
            if (!build(bounds, nodes)) { return RHITestResult::fail(reason); }
            for (uint32_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
                const auto& node = nodes[nodeIndex];
                for (uint32_t descendant = nodeIndex; descendant < node.escapeIndex; ++descendant) {
                    const auto& leaf = nodes[descendant];
                    for (uint32_t index = 0; index < leaf.groupCount; ++index) {
                        const auto& child = bounds[leaf.groupOffset + index];
                        for (uint32_t axis = 0; axis < 3; ++axis) {
                            if (node.demandBounds.min[axis] > child.min[axis] || node.demandBounds.max[axis] < child.max[axis]) {
                                return RHITestResult::fail("Node demand bounds omit a refinement descendant");
                            }
                        }
                        ++checked;
                    }
                }
            }
            std::vector<MeshletLODBVHNode> unknown;
            if (!build({}, unknown)) { return RHITestResult::fail(reason); }
            for (const auto& node : unknown) {
                if (!meshletLodBoundsVisible(node.demandBounds, fixture.instance, fixture.view, {0, 1, 0}, 1.f, 1000000.f)) {
                    return RHITestResult::fail("Omitted demand metadata was treated as known offscreen");
                }
            }
            auto partial = bounds;
            partial[0] = {};
            if (!build(partial, unknown)) { return RHITestResult::fail(reason); }
            for (uint32_t node = 0; node < unknown.size(); ++node) {
                const bool containsUnknown = std::any_of(unknown.begin() + node, unknown.begin() + unknown[node].escapeIndex,
                    [](const auto& leaf) { return leaf.groupCount != 0 && leaf.groupOffset == 0; });
                if (containsUnknown && !meshletLodBoundsVisible(unknown[node].demandBounds,
                    fixture.instance, fixture.view, {0, 1, 0}, 1.f, 1000000.f)) {
                    return RHITestResult::fail("Unknown child demand bounds were hidden by a finite sibling union");
                }
            }
            if (build(std::span<const MeshletLODRefinementBounds>(bounds).first(bounds.size() - 1), unknown)) {
                return RHITestResult::fail("Mismatched node demand metadata count was accepted");
            }
        }
        if (!meshletLodBoundsVisible(bounds[256], fixture.instance, fixture.view, {0, 1, 0}, 1.f, 1000000.f)) {
            return RHITestResult::fail("Directed fixture lost the visible refinement descendant");
        }
        return RHITestResult::pass(std::to_string(checked) + " BVH/tile ancestor bounds contain refinement descendants; omitted and partial metadata remain conservative");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLODDemandBoundsTest);

class StreamMeshletLODHistoryTokenTest final : public RHITest {
public:
    StreamMeshletLODHistoryTokenTest() { type = RHITestType::Validation; name = "meshlet_lod_stream_history_token"; }
    RHITestResult run(RHITestContext&) override
    {
        MeshletStreamLODHistoryState history;
        history.frameToken = UINT32_MAX - 1u;
        history.initialized = true;
        if (history.advanceFrameToken() != UINT32_MAX || !history.initialized ||
            history.advanceFrameToken() != 1u || history.initialized || history.generation == 0u) {
            return RHITestResult::fail("RT history wrap reused zero or failed to invalidate initialization");
        }
        const uint64_t wrappedGeneration = history.generation;
        // No traversal is submitted while frozen: later tokens must not make
        // the required wrap clear disappear before recording resumes.
        if (history.advanceFrameToken() != 2u || history.advanceFrameToken() != 3u ||
            history.initialized || history.generation != wrappedGeneration) {
            return RHITestResult::fail("Paused RT traversal forgot the pending wrap clear");
        }
        const uint64_t cancelled = history.beginInitialization();
        history.finishInitialization(cancelled, false);
        if (history.initialized) { return RHITestResult::fail("Cancelled history clear published initialization"); }
        const uint64_t retry = history.beginInitialization();
        history.finishInitialization(cancelled, true);
        if (history.initialized) { return RHITestResult::fail("An older accepted clear initialized the retry generation"); }
        history.finishInitialization(retry, true);
        history.finishInitialization(cancelled, false);
        if (!history.initialized) { return RHITestResult::fail("An older cancellation undid an accepted retry"); }
        history.invalidate();
        history.finishInitialization(retry, true);
        if (history.initialized) { return RHITestResult::fail("A stale submission undid a newer history invalidation"); }
        return RHITestResult::pass("RT tokens skip zero at wrap and retain invalidation until the matching GPU clear is accepted");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLODHistoryTokenTest);

class StreamMeshletLODBVHReferenceTest final : public RHITest {
public:
    StreamMeshletLODBVHReferenceTest() { type = RHITestType::Validation; name = "meshlet_lod_stream_bvh_reference"; }
    RHITestResult run(RHITestContext&) override
    {
        uint32_t comparisons = 0;
        for (bool large : {false, true}) {
            StreamLODFixture fixture{large};
            std::vector<MeshletLODBVHNode> nodes;
            std::string reason;
            if (!buildMeshletLodBvh(fixture.groups, nodes, reason) || nodes.empty()) {
                return RHITestResult::fail("stream BVH build: " + reason);
            }
            for (uint32_t test = 0; test < (large ? 36u : 37u); ++test) {
                uint32_t manual, capacity;
                std::vector<uint8_t> available;
                configureStreamLodCase(fixture, large, test, manual, capacity, available);
                const auto brute = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
                    fixture.drawable, fixture.instance, fixture.view, manual, capacity, available);
                const auto accelerated = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
                    fixture.drawable, fixture.instance, fixture.view, manual, capacity, available, nodes);
                if (!sameStreamCut(brute, accelerated)) {
                    return RHITestResult::fail("BVH/reference cut mismatch: fixture " + std::to_string(large) +
                        ", case " + std::to_string(test) + ", " + accelerated.reason);
                }
                if (large && (test == 1 || test == 11 || test == 18 || test == 24) &&
                    (accelerated.testedGroups >= fixture.groups.size() / 8 || accelerated.visitedBvhNodes == 0 ||
                        accelerated.selectedClusters.size() != 2)) {
                    return RHITestResult::fail("coarse BVH cut did not prune at least 7/8 of 511 group tests");
                }
                ++comparisons;
            }
            fixture = StreamLODFixture{large};
            const auto invalidRejected = [&](const std::vector<MeshletLODBVHNode>& malformed) {
                return !selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
                    fixture.drawable, fixture.instance, fixture.view, UINT32_MAX, UINT32_MAX, {}, malformed).valid;
            };
            auto malformed = nodes;
            malformed[0].escapeIndex = 0;
            if (!invalidRejected(malformed)) { return RHITestResult::fail("BVH accepted non-advancing escape"); }
            malformed = nodes;
            malformed[0].maxError = 0;
            if (!invalidRejected(malformed)) { return RHITestResult::fail("BVH accepted nonconservative error"); }
            const auto leaf = std::find_if(nodes.begin(), nodes.end(), [](const auto& node) { return node.groupCount != 0; });
            if (leaf == nodes.end()) { return RHITestResult::fail("BVH contains no group leaves"); }
            malformed = nodes;
            malformed[static_cast<size_t>(leaf - nodes.begin())].groupOffset = static_cast<uint32_t>(fixture.groups.size());
            if (!invalidRejected(malformed)) { return RHITestResult::fail("BVH accepted out-of-range leaf groups"); }
        }
        StreamLODFixture spread{true, 64};
        uint32_t randomState = 0x3ca961e5;
        const auto randomCoordinate = [&]() {
            randomState ^= randomState << 13;
            randomState ^= randomState >> 17;
            randomState ^= randomState << 5;
            return (float(randomState & 0xffffu) / 65535.f - .5f) * 100.f;
        };
        for (uint32_t group = 0; group < spread.groups.size(); ++group) {
            auto& sphere = spread.groups[group].sphere;
            const auto& range = spread.ranges[group];
            if (spread.groups[group].level == 0) {
                sphere = {randomCoordinate(), randomCoordinate(), randomCoordinate(),
                    .1f + std::abs(randomCoordinate()) * .1f};
            } else {
                // Parent groups enclose their children, but unrelated groups
                // have distinct centers: BVH aggregate centers must be safe.
                const auto& left = spread.groups[spread.refined[range.clusterOffset]].sphere;
                const auto& right = spread.groups[spread.refined[range.clusterOffset + 1]].sphere;
                float radiusSquared = 0;
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    const float low = std::min(left[axis] - left[3], right[axis] - right[3]);
                    const float high = std::max(left[axis] + left[3], right[axis] + right[3]);
                    sphere[axis] = (low + high) * .5f;
                    radiusSquared += (high - low) * (high - low) * .25f;
                }
                sphere[3] = std::sqrt(radiusSquared) + .001f;
            }
        }
        std::vector<MeshletLODBVHNode> spreadNodes;
        std::string reason;
        if (!buildMeshletLodBvh(spread.groups, spreadNodes, reason)) {
            return RHITestResult::fail("spatial stream BVH build: " + reason);
        }
        for (uint32_t viewCase = 0; viewCase < 9; ++viewCase) {
            auto fixture = spread;
            fixture.view.forward[3] = viewCase == 1 ? 1.f : 0.f;
            fixture.view.eye = {0, 0, 300, .1f};
            if (viewCase == 2) { fixture.view.eye = {300, -100, 200, .1f}; }
            if (viewCase == 3) { fixture.view.eye = {0, 0, .1f, .1f}; }
            if (viewCase == 4) {
                fixture.instance.worldMatrix[4] = 3.f;
                fixture.instance.worldMatrix[9] = -2.f;
            }
            if (viewCase == 5) {
                fixture.instance.worldMatrix[0] = -4.f;
                fixture.instance.worldMatrix[5] = .5f;
            }
            if (viewCase == 6 || viewCase == 7) {
                const float translation = viewCase == 6 ? 1048576.f : 134217728.f;
                fixture.instance.worldMatrix[12] = translation;
                fixture.instance.worldMatrix[13] = -translation;
                fixture.instance.worldMatrix[14] = translation;
                fixture.view.eye = {translation + 32.f, -translation - 64.f, translation + 320.f, .1f};
            }
            if (viewCase == 8) {
                fixture.instance.worldMatrix[0] = 500.f;
                fixture.instance.worldMatrix[5] = 500.f;
                fixture.instance.worldMatrix[10] = 500.f;
                fixture.view.eye[2] = 150000.f;
            }
            for (uint32_t probe = 0; probe < fixture.groups.size(); probe += 7) {
                const float error = meshletLodPixelError(fixture.groups[probe], fixture.instance, fixture.view);
                if (!std::isfinite(error) || error >= std::numeric_limits<float>::max() * .25f || error <= 1e-5f) { continue; }
                for (float factor : {.9998f, 1.f, 1.0002f}) {
                    fixture.view.projection[3] = error * factor;
                    fixture.drawable = spread.drawable;
                    const uint32_t missing = (probe * 17 + viewCase) % uint32_t(fixture.groups.size() - 1);
                    fixture.drawable[missing] = 0;
                    auto available = fixture.drawable;
                    if (viewCase % 2 != 0) { available[missing] = 1; }
                    const auto brute = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
                        fixture.drawable, fixture.instance, fixture.view, UINT32_MAX, UINT32_MAX, available);
                    const auto accelerated = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
                        fixture.drawable, fixture.instance, fixture.view, UINT32_MAX, UINT32_MAX, available, spreadNodes);
                    if (!sameStreamCut(brute, accelerated)) {
                        return RHITestResult::fail("spatial BVH bound changed the cut: view " + std::to_string(viewCase) +
                            ", group " + std::to_string(probe) + ", threshold " + std::to_string(fixture.view.projection[3]));
                    }
                    ++comparisons;
                }
            }
        }
        // A transformed aggregate sphere must account for float cancellation:
        // x * .05 - 5000000 rounds individual group centers to 0 and 1, while
        // their aggregate center rounds to .5. Its unexpanded radius can miss
        // the near plane and incorrectly prune the first leaf's FLT_MAX error.
        StreamLODFixture cancellation{true, 64};
        for (uint32_t group = 0; group < cancellation.groups.size(); ++group) {
            cancellation.groups[group].sphere = cancellation.groups[group].level == 0 ?
                std::array<float, 4>{group == 0 ? 100000000.f : 100000016.f, 0, 0, 1} :
                std::array<float, 4>{100000008.f, 0, 0, 100};
        }
        cancellation.instance.worldMatrix[0] = .05f;
        cancellation.instance.worldMatrix[5] = .05f;
        cancellation.instance.worldMatrix[10] = .05f;
        cancellation.instance.worldMatrix[12] = -5000000.f;
        cancellation.view.eye = {0, 0, 0, .01f};
        cancellation.view.forward = {1, 0, 0, 0};
        cancellation.view.projection[3] = 100;
        std::vector<MeshletLODBVHNode> cancellationNodes;
        if (!buildMeshletLodBvh(cancellation.groups, cancellationNodes, reason)) {
            return RHITestResult::fail("cancellation BVH build: " + reason);
        }
        const auto bruteCancellation = selectStreamMeshletLodReference(cancellation.groups, cancellation.ranges,
            cancellation.refined, cancellation.drawable, cancellation.instance, cancellation.view);
        const auto bvhCancellation = selectStreamMeshletLodReference(cancellation.groups, cancellation.ranges,
            cancellation.refined, cancellation.drawable, cancellation.instance, cancellation.view,
            UINT32_MAX, UINT32_MAX, {}, cancellationNodes);
        if (!bruteCancellation.valid || bruteCancellation.activeGroups[0] == 0 ||
            !sameStreamCut(bruteCancellation, bvhCancellation)) {
            return RHITestResult::fail("BVH bound lost near-plane group after large-coordinate float cancellation");
        }
        ++comparisons;
        return RHITestResult::pass(std::to_string(comparisons) +
            " BVH/brute cuts agree across residency, shared parents, projection, transforms, translated/spread spheres, capacity and manual levels; coarse 511-group cuts prune >87% of group tests");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLODBVHReferenceTest);

#define STREAM_LOD_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { return RHITestResult::fail(std::string(#expr) + ": " + toString(checked)); } } while (false)

class StreamMeshletLODGPUTest final : public RHITest {
public:
    StreamMeshletLODGPUTest() { type = RHITestType::Rendering; name = "meshlet_lod_stream_gpu_matches_reference"; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto small = runFixture(context, false);
        if (!small.passed) { return small; }
        const auto large = runFixture(context, true);
        if (!large.passed) { return large; }
        // Many sibling tiles per level exercise deferred device-memory
        // publication, with the same CPU oracle and residency changes.
        const auto tiled = runFixture(context, true, 4096);
        if (!tiled.passed) { return tiled; }
        return RHITestResult::pass("872 linear/BVH/cooperative/view-driven/prefetch/split/distributed/switching GPU-reference cuts across shared-parent, 511- and 8191-group fixtures");
    }
    RHITestResult runFixture(RHITestContext& context, bool large, uint32_t leafCount = 256,
        bool rtDemand = false, bool rtDemandC2 = false, bool rtDemandC3 = false, bool packedHistory = false,
        bool sparseHistoryComparison = false)
    {
        StreamLODFixture fixture = rtDemand ? rtDemandFixture(leafCount) : StreamLODFixture{large, leafCount};
        const uint32_t kGroupCount = static_cast<uint32_t>(fixture.groups.size());
        const uint32_t kRequestCapacity = kGroupCount + 2;
        const uint32_t kPriorityOffset = 16 + kRequestCapacity * 2;
        const uint32_t kPriorityTable = kPriorityOffset + kRequestCapacity;
        const uint32_t kRequestWords = kPriorityTable + kGroupCount;
        std::vector<MeshletStreamGPUGroup> groups(kGroupCount);
        std::vector<uint32_t> topology = fixture.refined;
        for (uint32_t group = 0; group < kGroupCount; ++group) {
            auto& output = groups[group];
            output.pageIndex = group;
            output.lodLevel = fixture.groups[group].level;
            output.clusterCount = fixture.ranges[group].clusterCount;
            std::copy(fixture.groups[group].sphere.begin(), fixture.groups[group].sphere.end(), output.boundsCenterRadius);
            output.maxQuadricError = fixture.groups[group].error;
            output.clusterRefinedOffset = fixture.ranges[group].clusterOffset;
            output.flags = fixture.groups[group].flags;
            output.parentOffset = static_cast<uint32_t>(topology.size());
            for (uint32_t parent = group + 1; parent < kGroupCount; ++parent) {
                const auto& range = fixture.ranges[parent];
                if (std::find(fixture.refined.begin() + range.clusterOffset,
                    fixture.refined.begin() + range.clusterOffset + range.clusterCount, group) !=
                    fixture.refined.begin() + range.clusterOffset + range.clusterCount) {
                    topology.push_back(parent);
                    ++output.parentCount;
                }
            }
        }
        const uint32_t instanceOffsetsOffset = static_cast<uint32_t>(topology.size());
        topology.push_back(4);
        MeshletStreamGPUPrimitive primitive;
        primitive.groupCount = kGroupCount;
        primitive.pageCount = kGroupCount;
        primitive.lodLevelCount = fixture.groups.back().level + 1;
        std::vector<MeshletLODBVHNode> nodes;
        std::string reason;
        std::vector<MeshletLODRefinementBounds> refinementBounds;
        if (!buildMeshletLodRefinementBounds(fixture.groups, fixture.ranges, fixture.refined, refinementBounds, reason)) {
            return RHITestResult::fail(reason);
        }
        if (!buildMeshletLodBvh(fixture.groups, nodes, reason, refinementBounds)) {
            return RHITestResult::fail("stream GPU fixture BVH: " + reason);
        }
        primitive.lodBvhOffset = static_cast<uint32_t>(topology.size());
        const size_t bvhWords = nodes.size() * sizeof(MeshletLODBVHNode) / sizeof(uint32_t);
        topology.resize(topology.size() + bvhWords);
        std::memcpy(topology.data() + primitive.lodBvhOffset, nodes.data(), bvhWords * sizeof(uint32_t));
        std::vector<MeshletLODBVHNode> tiles;
        if (!buildMeshletLodTiles(fixture.groups, tiles, reason, refinementBounds)) { return RHITestResult::fail(reason); }
        primitive.lodTileOffset = static_cast<uint32_t>(topology.size());
        topology.push_back(static_cast<uint32_t>(tiles.size()));
        const size_t tileWords = tiles.size() * sizeof(MeshletLODBVHNode) / sizeof(uint32_t);
        topology.resize(topology.size() + tileWords);
        std::memcpy(topology.data() + primitive.lodTileOffset + 1, tiles.data(), tileWords * sizeof(uint32_t));
        const auto tileParents = buildMeshletLodTileParents(tiles);
        topology.insert(topology.end(), tileParents.begin(), tileParents.end());
        const uint32_t demandInstanceOffsets = static_cast<uint32_t>(topology.size());
        topology.push_back(0);
        topology.push_back(kGroupCount);
        const auto demandRoots = buildMeshletLodDemandRoots(tiles);
        topology.push_back(0);
        topology.push_back(static_cast<uint32_t>(demandRoots.size()));
        const uint32_t demandTaskOffset = static_cast<uint32_t>(topology.size());
        for (uint32_t root : demandRoots) { topology.push_back(root); }
        std::vector<uint32_t> invalidDemandTopology;
        if (rtDemandC2) {
            invalidDemandTopology = topology;
            constexpr uint32_t kNodeWords = sizeof(MeshletLODBVHNode) / sizeof(uint32_t);
            constexpr uint32_t kBoundsWord = offsetof(MeshletLODBVHNode, demandBounds) / sizeof(uint32_t);
            const MeshletLODRefinementBounds unknown;
            for (const auto [offset, count] : {std::pair{primitive.lodBvhOffset, nodes.size()},
                    std::pair{primitive.lodTileOffset + 1u, tiles.size()}}) {
                for (size_t node = 0; node < count; ++node) {
                    std::memcpy(invalidDemandTopology.data() + offset + node * kNodeWords + kBoundsWord,
                        &unknown, sizeof(unknown));
                }
            }
        }
        enum BufferIndex { Instance, Primitive, Groups, Params, Topology, State, PageTable,
            Requests, ActiveGroups, Header, Arguments, Dummy, Demand, DemandTasks, BufferCount };
        const uint32_t strides[] = {sizeof(MeshletStreamGPUInstance), sizeof(primitive), sizeof(MeshletStreamGPUGroup),
            sizeof(MeshletStreamGPUParams), 4, 4, sizeof(StreamPageTableEntry), 4,
            sizeof(MeshletStreamGPUActiveGroup), sizeof(MeshletStreamGPUActiveHeader), sizeof(MeshletStreamGPUDrawIndirect), 4, 4, sizeof(MeshletStreamGPUTraversalWorkItem)};
        // Retain the ordinary pair/sparse layout; diagnostic history is an
        // optional two words per instance-group and one visible-frame stamp.
        const uint32_t rtHistoryWords = 3u + kGroupCount * (packedHistory ? 1u : 2u);
        const uint64_t logicalStateBytes = (rtDemand ? 4u + kGroupCount * 3u + 4u + rtHistoryWords :
            4u + kGroupCount * 5u + 5u) * sizeof(uint32_t);
        // Cross the maximum X dispatch dimension and leave a partial last row.
        const uint64_t allocatedStateBytes = large && !rtDemand ? (65535ull * 64 + 67) * sizeof(uint32_t) : logicalStateBytes;
        const uint64_t sizes[] = {strides[Instance], sizeof(primitive), groups.size() * sizeof(groups[0]), strides[Params],
            topology.size() * 4, allocatedStateBytes, kGroupCount * sizeof(StreamPageTableEntry),
            kRequestWords * 4, kGroupCount * sizeof(MeshletStreamGPUActiveGroup), strides[Header],
            kMeshletStreamDrawIndirectCommandCount * strides[Arguments], 64,
            (kMeshletStreamDemandStatsWords + (kGroupCount + tiles.size() + 31) / 32) * 4, demandRoots.size() * sizeof(MeshletStreamGPUTraversalWorkItem)};
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream LOD frontier regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless compute"); }
        STREAM_LOD_REQUIRE(created);
        std::unique_ptr<BindlessHeap> heap;
        STREAM_LOD_REQUIRE(device->createBindlessHeap({.maxBuffers = BufferCount}).transform([&](auto rhiValue) { heap = std::move(rhiValue); }));
        std::array<std::unique_ptr<Buffer>, BufferCount> buffers;
        std::array<BindlessHandle, BufferCount> handles;
        for (uint32_t index = 0; index < BufferCount; ++index) {
            STREAM_LOD_REQUIRE(device->createBuffer({.size = sizes[index], .structureStride = strides[index],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto rhiValue) { buffers[index] = std::move(rhiValue); }));
            STREAM_LOD_REQUIRE(heap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { handles[index] = std::move(rhiValue); }));
            STREAM_LOD_REQUIRE((*buffers[index]).slice().and_then([&](const auto& bufferSlice) { return heap->writeStorageBuffer(handles[index], bufferSlice); }));
        }
        const auto upload = [&](BufferIndex index, const void* source, size_t size) {
            void* mapped = buffers[index]->map();
            if (mapped == nullptr) { return false; }
            std::memset(mapped, index == State && large && !rtDemand ? 0xa5 : 0, sizes[index]);
            if (source != nullptr) { std::memcpy(mapped, source, size); }
            buffers[index]->flush();
            buffers[index]->unmap();
            return true;
        };
        for (uint32_t index = 0; index < BufferCount; ++index) {
            if (!upload(static_cast<BufferIndex>(index), nullptr, 0)) { return RHITestResult::fail("stream test buffer map"); }
        }
        if (!upload(Primitive, &primitive, sizeof(primitive)) || !upload(Groups, groups.data(), sizes[Groups]) ||
            !upload(Topology, topology.data(), sizes[Topology])) { return RHITestResult::fail("stream topology map"); }
        ShaderCompileResult compiled;
        const auto compile = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamActiveBuildEntryPoint, .searchPath = kMeshletStreamShaderSearchPath}, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); });
        if (!compile) { return RHITestResult::fail("stream frontier shader compile: " + compiled.diagnostics); }
        std::unique_ptr<ShaderModule> shader;
        STREAM_LOD_REQUIRE(device->createShaderModule({.spirv = compiled.spirv}).transform([&](auto rhiValue) { shader = std::move(rhiValue); }));
        std::unique_ptr<ComputePipeline> pipeline;
        STREAM_LOD_REQUIRE(device->createComputePipeline({
            .computeShader = {shader.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
        }).transform([&](auto rhiValue) { pipeline = std::move(rhiValue); }));
        ShaderCompileResult cooperativeCompiled;
        const auto cooperativeCompile = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamCooperativeBuildEntryPoint, .searchPath = kMeshletStreamShaderSearchPath}, cooperativeCompiled.diagnostics).transform([&](auto value) { cooperativeCompiled = std::move(value); });
        if (!cooperativeCompile) { return RHITestResult::fail(cooperativeCompiled.diagnostics); }
        std::unique_ptr<ShaderModule> cooperativeShader;
        STREAM_LOD_REQUIRE(device->createShaderModule({
            .spirv = cooperativeCompiled.spirv,
        }).transform([&](auto rhiValue) { cooperativeShader = std::move(rhiValue); }));
        std::unique_ptr<ComputePipeline> cooperativePipeline;
        STREAM_LOD_REQUIRE(device->createComputePipeline({
            .computeShader = {cooperativeShader.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
        }).transform([&](auto rhiValue) { cooperativePipeline = std::move(rhiValue); }));
        ShaderCompileResult traversalCompiled;
        STREAM_LOD_REQUIRE(compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamTraversalEntryPoint, .searchPath = kMeshletStreamShaderSearchPath}, traversalCompiled.diagnostics).transform([&](auto value) { traversalCompiled = std::move(value); }));
        std::unique_ptr<ShaderModule> traversalShader;
        STREAM_LOD_REQUIRE(device->createShaderModule({
            .spirv = traversalCompiled.spirv,
        }).transform([&](auto rhiValue) { traversalShader = std::move(rhiValue); }));
        std::unique_ptr<ComputePipeline> traversalPipeline;
        STREAM_LOD_REQUIRE(device->createComputePipeline({
            .computeShader = {traversalShader.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
        }).transform([&](auto rhiValue) { traversalPipeline = std::move(rhiValue); }));
        ShaderCompileResult demandCompiled;
        const auto demandCompile = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamDemandEntryPoint, .searchPath = kMeshletStreamShaderSearchPath}, demandCompiled.diagnostics).transform([&](auto value) { demandCompiled = std::move(value); });
        if (!demandCompile) { return RHITestResult::fail(demandCompiled.diagnostics); }
        std::unique_ptr<ShaderModule> demandShader;
        STREAM_LOD_REQUIRE(device->createShaderModule({
            .spirv = demandCompiled.spirv,
        }).transform([&](auto rhiValue) { demandShader = std::move(rhiValue); }));
        std::unique_ptr<ComputePipeline> demandPipeline;
        STREAM_LOD_REQUIRE(device->createComputePipeline({
            .computeShader = {demandShader.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
        }).transform([&](auto rhiValue) { demandPipeline = std::move(rhiValue); }));
        uint32_t positivePriorities = 0;
        uint32_t viewDemandReductions = 0;
        uint32_t speculativeRequests = 0;
        uint32_t predictedRequests = 0;
        MeshletStreamUserPush push;
        push.instanceBuffer = handles[Instance].shaderIndex;
        push.primitiveBuffer = handles[Primitive].shaderIndex;
        push.groupBuffer = handles[Groups].shaderIndex;
        push.paramsBuffer = handles[Params].shaderIndex;
        push.pageTableBuffer = handles[PageTable].shaderIndex;
        push.requestBuffer = handles[Requests].shaderIndex;
        push.activeGroupBuffer = handles[ActiveGroups].shaderIndex;
        push.activeHeaderBuffer = handles[Header].shaderIndex;
        push.drawIndirectBuffer = handles[Arguments].shaderIndex;
        push.pageBuffer = handles[Dummy].shaderIndex;
        push.nodeBuffer = handles[Dummy].shaderIndex;
        push.lodLevelBuffer = handles[Dummy].shaderIndex;
        push.traversalHeaderBuffer = handles[Dummy].shaderIndex;
        push.traversalWorkBuffer = handles[DemandTasks].shaderIndex;
        std::unique_ptr<Buffer> readback;
        constexpr uint32_t outputs[] = {Header, ActiveGroups, Requests, Arguments, State, Demand};
        const uint64_t outputSizes[] = {sizes[Header], sizes[ActiveGroups], sizes[Requests], sizes[Arguments], logicalStateBytes, sizes[Demand]};
        std::array<uint64_t, std::size(outputs)> offsets{};
        uint64_t outputBytes = 0;
        for (uint32_t index = 0; index < std::size(outputs); ++index) { offsets[index] = outputBytes; outputBytes += outputSizes[index]; }
        const uint64_t tailOffset = outputBytes;
        if (large && !rtDemand) { outputBytes += 3 * sizeof(uint32_t); }
        const uint64_t prefetchHistoryOffset = outputBytes;
        if (rtDemandC3) { outputBytes += rtHistoryWords * sizeof(uint32_t); }
        STREAM_LOD_REQUIRE(device->createBuffer({.size = outputBytes, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }));
        Queue* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        STREAM_LOD_REQUIRE(device->createCommandPool(*queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        STREAM_LOD_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
        STREAM_LOD_REQUIRE(device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }));
        const uint32_t caseCount = large ? 36u : 37u;
        const uint32_t ordinaryFrames = caseCount * 8;
        MeshletStreamGPUParams priorParams;
        std::vector<uint8_t> previousWanted(kGroupCount), expectedWanted(kGroupCount);
        std::array<uint64_t, 2> rtGroupTests{};
        std::array<std::array<uint64_t, 2>, 2> rtUnitScaleTests{};
        const uint32_t rtSequenceFrames = rtDemandC3 ? 36u : rtDemandC2 ? 28u : 16u;
        const uint32_t rtModes = rtDemandC3 && !sparseHistoryComparison ? 3u : 2u;
        const uint32_t frameCount = rtDemand ? rtSequenceFrames * rtModes * 2u : ordinaryFrames + (large ? 0u : 80u);
        std::vector<uint32_t> eagerGroupTests(rtSequenceFrames * 2u);
        std::vector<uint32_t> previousHistoryWords(rtHistoryWords);
        uint64_t baselineFalseWrites = 0, sparseFalsePreserved = 0, sparseExpiredWanted = 0;
        uint32_t previousHistoryToken = 0u, previousHistoryPolicy = 0u;
        for (uint32_t frame = 0; frame < frameCount; ++frame) {
            const uint32_t rtStep = frame % rtSequenceFrames;
            const uint32_t rtMode = (frame / rtSequenceFrames) % rtModes;
            const bool rtBaseline = (frame / rtSequenceFrames) % 2u == 0;
            float rtOffscreenScale = 2.f;
            float rtHysteresis = .2f;
            uint32_t rtPolicyRevision = 1u;
            const bool telemetrySequence = frame >= ordinaryFrames;
            const uint32_t telemetryStep = telemetrySequence ? (frame - ordinaryFrames) % 20u : 0u;
            const uint32_t telemetryPath = telemetrySequence ? (frame - ordinaryFrames) / 20u : 0u;
            const uint32_t test = telemetrySequence ? 0u : frame % caseCount;
            const bool useBvh = rtDemand || (telemetrySequence ? telemetryPath != 0 : frame >= caseCount);
            const bool cooperative = rtDemand || (telemetrySequence ? telemetryPath >= 2 : frame >= caseCount * 2);
            const bool viewDriven = rtDemand || telemetrySequence || frame >= caseCount * 3;
            const bool prefetch = rtDemandC3 ? rtStep == 28u : !rtDemand && !telemetrySequence && frame >= caseCount * 4;
            const bool switching = !rtDemand && !telemetrySequence && frame >= caseCount * 7;
            const bool distributed = rtDemand ? frame >= rtSequenceFrames * rtModes : telemetrySequence ? telemetryPath == 3 : frame >= caseCount * 6 && (!switching || test % 2 == 0);
            const bool split = rtDemand ? distributed : telemetrySequence ? distributed : switching ? distributed : frame >= caseCount * 5;
            const float viewAspect = !rtDemand && viewDriven && test % 7 == 1 ? .25f : 1.f;
            primitive.lodBvhNodeCount = useBvh ? static_cast<uint32_t>(nodes.size()) : 0;
            uint32_t manual, capacity;
            std::vector<uint8_t> available;
            configureStreamLodCase(fixture, large, test, manual, capacity, available, leafCount);
            // Keep a genuinely speculative leaf below the demand threshold;
            // missing ancestors no longer turn desired leaves into forecasts.
            if (prefetch && !large && (test == 1 || test == 9)) { fixture.view.projection[3] = 5.1f; }
            if (prefetch && large && test == 17) {
                fixture.view.projection[3] = .021f;
                fixture.drawable[128] = 0; available[128] = 0;
            }
            if (viewDriven && !telemetrySequence && !rtDemand) {
                if (prefetch && large && test == 7) { fixture.drawable[128] = 0; available[128] = 0; }
                if (test % 6 == 0) { fixture.instance.worldMatrix[12] = 500.f; }
                if (test % 6 == 2) { fixture.instance.worldMatrix[14] = 1000.f; }
                if (test % 6 == 4) { fixture.instance.worldMatrix[14] = -2000000.f; }
            }
            if (telemetrySequence) {
                fixture = StreamLODFixture{};
                available.assign(kGroupCount, 1);
                manual = UINT32_MAX;
                capacity = kGroupCount;
                if (telemetryStep == 0 || telemetryStep == 10) { fixture.view.projection[3] = 100.f; }
                if (telemetryStep == 3 || telemetryStep == 7 || telemetryStep == 12) {
                    fixture.drawable[0] = 0; available[0] = 0;
                }
                if (telemetryStep == 5) { fixture.drawable[3] = 0; available[3] = 1; } // PendingUpload parent.
                if (telemetryStep == 8) { fixture.instance.identity[3] = 0; }
                // Instance remains visible, but its refinement bounds enter the
                // frustum on step 11. This must not be called an SSE crossing.
                if (telemetryStep == 10) { fixture.view.eye[0] = 1000.f; }
                if (telemetryStep == 13) { capacity = 2; }
                if (telemetryStep == 14) { manual = 0; }
            }
            const bool forecastExclusive = switching && !large && test == 9;
            if (forecastExclusive) {
                fixture.view.eye[2] = 30.f;
                fixture.view.forward[3] = 0.f;
                fixture.view.projection[3] = 18.f;
                std::fill(fixture.drawable.begin(), fixture.drawable.end(), uint8_t{1});
                fixture.drawable[1] = 0;
                available = fixture.drawable;
            }
            if (rtDemand) {
                fixture = rtDemandFixture(leafCount);
                available.assign(kGroupCount, 1);
                manual = UINT32_MAX;
                capacity = kGroupCount;
                bool resetHistory = rtStep == 0;
                if (!rtDemandC2) {
                    if (rtStep == 2) { fixture.view.forward = {0, 0, 1, 0}; }
                    if (rtStep == 3) { fixture.view.forward = {1, 0, 0, 0}; }
                    if (rtStep == 5) {
                        fixture.instance.worldMatrix[0] = 2.f;
                        fixture.instance.worldMatrix[4] = .75f;
                        fixture.instance.worldMatrix[5] = 3.f;
                    }
                    if (rtStep == 6) { fixture.view.eye[2] = .05f; }
                    if (rtStep == 7) { fixture.view.eye[2] = 0.f; }
                    if (rtStep == 8) { fixture.view.forward[3] = 1.f; }
                    if (rtStep == 9) {
                        fixture.instance.worldMatrix[12] = 1000000.f;
                        fixture.instance.worldMatrix[14] = -1000000.f;
                        fixture.view.eye[0] = 1000000.f;
                        fixture.view.eye[2] = -999990.f;
                    }
                    if (rtStep == 10 || rtStep == 11) {
                        fixture.drawable[leafCount] = 0;
                        available[leafCount] = rtStep == 11; // Pending upload is not a duplicate request.
                    }
                    if (rtStep == 12) { manual = 0; }
                    if (rtStep == 14) { fixture.view.projection[3] = .1f; }
                    resetHistory = resetHistory || rtStep == 13;
                } else {
                    constexpr float kRatios[]{.75f, .9f, 1.05f, .9f, .75f};
                    constexpr float kScales[]{1.f, 2.f, 16.f};
                    if (rtStep < 15) {
                        rtOffscreenScale = kScales[rtStep / 5u];
                        rtPolicyRevision = rtStep / 5u + 1u;
                        resetHistory = rtStep % 5u == 0;
                    } else {
                        rtOffscreenScale = rtStep == 15 || rtStep == 16 || rtStep == 20 ? 16.f : 2.f;
                        rtPolicyRevision = std::min(rtStep - 11u, 10u);
                        resetHistory = rtStep != 23;
                    }
                    if (rtStep == 17) { fixture.view.forward = {0, 0, 1, 0}; }
                    if (rtStep == 18) { fixture.view.eye[2] = 0.f; }
                    if (rtStep == 19) { fixture.view.forward[3] = 1.f; }
                    if (rtStep == 20) {
                        fixture.instance.worldMatrix[0] = 2.f;
                        fixture.instance.worldMatrix[4] = .75f;
                        fixture.instance.worldMatrix[5] = 3.f;
                    }
                    const float rayError = meshletLodRayPixelError(fixture.groups[leafCount - 1u], fixture.instance, fixture.view);
                    float ratio = rtStep < 15 ? kRatios[rtStep % 5u] : rtStep >= 22 ? .9f : .75f;
                    if (rtStep >= 24 && rtStep < 28) {
                        // A policy change resets retained demand. Seed again
                        // under h=.9, then straddle its remaining .1 threshold.
                        constexpr float kEdgeRatios[]{.9f, 1.05f, .11f, .09f};
                        ratio = kEdgeRatios[rtStep - 24u];
                        rtOffscreenScale = 16.f;
                        rtHysteresis = rtStep == 24 ? 0.f : .9f;
                        rtPolicyRevision = rtStep == 24 ? 11u : 12u;
                        resetHistory = rtStep <= 25;
                    }
                    fixture.view.projection[3] = rayError / (rtOffscreenScale * ratio);
                    if (rtStep == 15) {
                        // This parent metric is behind the eye but its demand
                        // subtree contains the visible leaf: C2 must not scale
                        // its threshold as though that whole subtree were offscreen.
                        fixture.view.projection[3] = meshletLodRayPixelError(fixture.groups[leafCount],
                            fixture.instance, fixture.view) * .5f;
                    }
                    if (rtStep == 16) { fixture.view.projection[3] = 1.5f; } // Unknown node bounds must descend.
                    if (rtStep == 21) { manual = 0; }
                    if (rtDemandC3 && rtStep >= 28 && rtStep < 32) {
                        rtOffscreenScale = 2.f;
                        rtHysteresis = .2f;
                        rtPolicyRevision = 13u;
                        resetHistory = rtStep == 28 || rtStep == 30;
                        fixture.view.projection[3] = rayError / (rtOffscreenScale * (rtStep == 30 ? .9f : 1.05f));
                        if (rtStep == 28) {
                            // A false on-screen leaf is requested only by the
                            // lookahead's lower threshold. That request must
                            // leave both header and per-leaf history untouched.
                            const float visibleError = std::max(meshletLodRayPixelError(fixture.groups[0], fixture.instance, fixture.view),
                                meshletLodPixelError(fixture.groups[0], fixture.instance, fixture.view));
                            fixture.view.projection[3] = visibleError / .95f;
                            fixture.drawable[0] = 0;
                            available[0] = 0;
                        }
                    }
                    if (rtDemandC3 && rtStep >= 32) {
                        // Identity scale and equal power-of-two projection
                        // factors preserve the authored error bits exactly on
                        // both CPU and GPU. No approximate boundary allowance:
                        // seed true, E==T retains, E==T*(1-h) rejects, then E==T
                        // must not resurrect the previous false/pruned demand.
                        fixture.view.forward[3] = 1.f;
                        fixture.view.projection[0] = 1024.f;
                        fixture.view.projection[2] = 1024.f;
                        rtOffscreenScale = 1.f;
                        rtHysteresis = .5f;
                        rtPolicyRevision = 14u;
                        resetHistory = rtStep == 32;
                        const float exactError = fixture.groups[leafCount - 1u].error;
                        constexpr float kThresholdMultipliers[]{.5f, 1.f, 2.f, 1.f};
                        fixture.view.projection[3] = exactError * kThresholdMultipliers[rtStep - 32u];
                        if (meshletLodRayPixelError(fixture.groups[leafCount - 1u], fixture.instance, fixture.view) != exactError) {
                            return RHITestResult::fail("C3 equality fixture did not preserve the exact error bits");
                        }
                    }
                }
                if (resetHistory) { std::fill(previousWanted.begin(), previousWanted.end(), uint8_t{0}); }
                for (uint32_t group = 0; group < kGroupCount; ++group) {
                    const auto& metric = fixture.groups[group];
                    if ((metric.flags & kMeshletLODTerminalGroup) != 0) { expectedWanted[group] = 1; continue; }
                    if (manual != UINT32_MAX) { expectedWanted[group] = metric.level >= manual; continue; }
                    const bool visible = meshletLodBoundsVisible(refinementBounds[group], fixture.instance,
                        fixture.view, {0, 1, 0}, viewAspect, 1000000.f);
                    float error = meshletLodRayPixelError(metric, fixture.instance, fixture.view);
                    if (visible) { error = std::max(error, meshletLodPixelError(metric, fixture.instance, fixture.view)); }
                    const bool retained = previousWanted[group] != 0;
                    expectedWanted[group] = meshletLodRayNeedsFine(error,
                        fixture.view.projection[3] * (visible ? 1.f : rtOffscreenScale), rtHysteresis, retained);
                }
                if (rtDemandC2 && rtStep < 15 && bool(expectedWanted[leafCount - 1u]) !=
                    (rtStep % 5u == 2u || rtStep % 5u == 3u)) {
                    return RHITestResult::fail("C2 fixture failed to exercise the offscreen hysteresis roundtrip");
                }
                if (rtDemandC2 && rtStep >= 24 && rtStep < 28 && bool(expectedWanted[leafCount - 1u]) != (rtStep == 25 || rtStep == 26)) {
                    return RHITestResult::fail("C2 fixture failed to straddle the zero/maximum hysteresis boundaries");
                }
                if (rtDemandC3 && rtStep >= 32 && bool(expectedWanted[leafCount - 1u]) != (rtStep < 34)) {
                    return RHITestResult::fail("C3 exact upper/lower equality fixture changed the strict leaf predicate");
                }
                if (rtStep == 0 && !expectedWanted[leafCount]) {
                    return RHITestResult::fail("RT fixture failed to expose a visible refinement descendant of a behind-eye metric sphere");
                }
            }
            auto demandMetrics = fixture.groups;
            for (uint32_t group = 0; group < kGroupCount; ++group) {
                groups[group].refinementBounds = viewDriven ? refinementBounds[group] : MeshletLODRefinementBounds{};
                if (!rtDemand && viewDriven && manual == UINT32_MAX && (demandMetrics[group].flags & kMeshletLODTerminalGroup) == 0 &&
                    !meshletLodBoundsVisible(refinementBounds[group], fixture.instance, fixture.view, {0, 1, 0}, viewAspect, 1000000.f)) {
                    demandMetrics[group].error = 0;
                }
            }
            const auto expected = rtDemand ? cutFromDemand(fixture, expectedWanted, available) :
                selectStreamMeshletLodReference(demandMetrics, fixture.ranges, fixture.refined,
                    fixture.drawable, fixture.instance, fixture.view, manual, capacity, available);
            if (viewDriven && !rtDemand) {
                const auto legacy = fixture.select(manual, capacity);
                viewDemandReductions += expected.selectedClusters.size() < legacy.selectedClusters.size();
            }
            const auto expectedState = rtDemand ? expected : selectStreamMeshletLodReference(demandMetrics, fixture.ranges, fixture.refined,
                fixture.drawable, fixture.instance, fixture.view, manual, UINT32_MAX, available,
                useBvh ? std::span<const MeshletLODBVHNode>(nodes) : std::span<const MeshletLODBVHNode>{});
            const std::string caseLabel = (large ? std::to_string(kGroupCount) + " groups, " : "shared parents, ") +
                (rtDemand ? "RT demand, frame " : viewDriven ? "view demand, case " : cooperative ? "cooperative, case " : useBvh ? "BVH, case " : "linear, case ") + std::to_string(rtDemand ? frame : test);
            MeshletStreamGPUInstance instance;
            instance.visible = fixture.instance.identity[3] != 0;
            instance.gpuSceneInstanceIndex = 17;
            std::copy_n(fixture.instance.worldMatrix.data(), 4, instance.world0);
            std::copy_n(fixture.instance.worldMatrix.data() + 4, 4, instance.world1);
            std::copy_n(fixture.instance.worldMatrix.data() + 8, 4, instance.world2);
            std::copy_n(fixture.instance.worldMatrix.data() + 12, 4, instance.world3);
            std::copy_n(instance.world3, 3, instance.boundsCenterRadius);
            instance.boundsCenterRadius[3] = 1000000.f; // Group bounds exercise the forecast frustum below.
            MeshletStreamGPUParams params;
            std::copy_n(fixture.view.eye.data(), 3, params.eye);
            for (uint32_t axis = 0; axis < 3; ++axis) { params.center[axis] = params.eye[axis] + fixture.view.forward[axis]; }
            params.upProjection[1] = 1;
            params.upProjection[3] = fixture.view.forward[3];
            params.viewport[2] = fixture.view.projection[0];
            params.viewport[0] = viewAspect;
            params.viewport[1] = params.viewport[2] * viewAspect;
            params.viewport[3] = 2 * std::atan(fixture.view.projection[1]);
            params.clipOrtho[0] = fixture.view.eye[3];
            params.clipOrtho[1] = 1000000.f;
            params.clipOrtho[2] = fixture.view.projection[2];
            params.lodPixelError = fixture.view.projection[3];
            params.lodTopologyBuffer = handles[Topology].shaderIndex;
            params.lodStateBuffer = handles[State].shaderIndex;
            params.lodInstanceOffsetsOffset = instanceOffsetsOffset;
            params.sceneInstanceCount = 1;
            params.demandBuffer = distributed ? handles[Demand].shaderIndex : UINT32_MAX;
            params.demandStatsBuffer = distributed || switching || telemetrySequence ? handles[Demand].shaderIndex : UINT32_MAX;
            params.demandTaskOffset = demandTaskOffset;
            params.demandInstanceOffsetsOffset = demandInstanceOffsets;
            params.demandTaskCount = static_cast<uint32_t>(demandRoots.size());
            params.splitFrontier = split;
            params.scenePrimitiveCount = 1;
            params.sceneGroupCount = kGroupCount;
            params.scenePageCount = kGroupCount;
            params.maxActiveGroupClusters = 2;
            params.activeGroupCount = capacity;
            params.drawTaskCount = capacity * 4;
            params.frameIndex = frame + 1;
            if (telemetrySequence && telemetryStep >= 18) { params.frameIndex = 0x04000000u + telemetryStep - 18u; }
            params.selectedLodLevel = manual;
            params.enableGpuLodSelection = manual == UINT32_MAX;
            if (rtDemand) {
                // C1 remains isolated from C2. C2 compares the same C1 path
                // with only its node offscreen-threshold scaling toggled.
                params.rtLodFlags = rtDemandC2 ? (rtBaseline ? 11u : 3u) : (rtBaseline ? 15u : 11u);
                if (rtDemandC3) {
                    params.rtLodFlags = 3u | (packedHistory ? 16u : 0u) |
                        (rtMode >= 1u ? 32u : 0u) | (rtMode >= 2u ? 64u : 0u);
                }
                if (sparseHistoryComparison) {
                    // Isolate write suppression: both arms retain the same
                    // two-word allocation, late reads and visibility-first metric.
                    params.rtLodFlags = 3u | 32u | (rtMode == 1u ? 128u : 0u);
                }
                params.rtOffscreenErrorScale = rtOffscreenScale;
                params.rtLodHysteresis = rtHysteresis;
                params.rtLodPolicyRevision = rtPolicyRevision;
                params.rtLodFrameToken = frame + 1u;
                if (rtDemandC3) {
                    // Tokens and editor frame indices are deliberately not
                    // adjacent; only the last submitted token owns history.
                    params.rtLodFrameToken = frame * 7u + 1u;
                    params.frameIndex = frame * 11u + 1u;
                    if (rtStep == 29) { params.rtLodFrameToken = UINT32_MAX; params.frameIndex = UINT32_MAX; }
                    if (rtStep == 30) { params.rtLodFrameToken = 1u; params.frameIndex = 0u; }
                    if (rtStep == 31) { params.rtLodFrameToken = 5u; params.frameIndex = 7u; }
                }
            }
            params.enableLodTransitionTelemetry = telemetrySequence && telemetryStep != 16;
            std::copy_n(priorParams.eye, 4, params.previousEye);
            std::copy_n(priorParams.center, 4, params.previousCenter);
            std::copy_n(priorParams.upProjection, 4, params.previousUpProjection);
            std::copy_n(priorParams.viewport, 4, params.previousViewport);
            std::copy_n(priorParams.clipOrtho, 4, params.previousClipOrtho);
            params.previousLodPixelError = priorParams.lodPixelError;
            priorParams = params;
            params.maxGpuPageRequests = kRequestCapacity;
            params.prefetchParams[0] = 1.125f;
            params.prefetchParams[1] = .85f;
            params.prefetchParams[2] = prefetch ? 1.f : 0.f;
            // Exercise a translated/turned forecast independently of the real
            // selection view. The cut oracle below must still match that view.
            if (switching && test % 2 != 0 && test != 1) {
                params.prefetchParams[3] = 1.f;
                for (uint32_t axis = 0; axis < 3; ++axis) {
                    params.prefetchEye[axis] = params.eye[axis] + fixture.view.forward[axis] * (forecastExclusive ? 3.f : 2.f);
                }
                const float yaw = forecastExclusive ? 0.f : .15f;
                const float x = fixture.view.forward[0], z = fixture.view.forward[2];
                params.prefetchCenter[0] = params.prefetchEye[0] + x * std::cos(yaw) + z * std::sin(yaw);
                params.prefetchCenter[1] = params.prefetchEye[1] + fixture.view.forward[1];
                params.prefetchCenter[2] = params.prefetchEye[2] - x * std::sin(yaw) + z * std::cos(yaw);
                params.prefetchUp[1] = 1.f;
            }
            std::vector<StreamPageTableEntry> pages(kGroupCount);
            for (uint32_t group = 0; group < kGroupCount; ++group) {
                pages[group].deviceOffsetAndState = packStreamPageTableEntry(
                    fixture.drawable[group] ? group * 512 : kInvalidStreamDeviceOffsetBytes,
                    fixture.drawable[group] ? MeshletStreamPageResidencyState::Resident : MeshletStreamPageResidencyState::Unloaded);
            }
            for (uint32_t group = 0; group < kGroupCount; ++group) {
                if (available[group] != 0 && fixture.drawable[group] == 0) {
                    pages[group].deviceOffsetAndState = packStreamPageTableEntry(group * 512,
                        MeshletStreamPageResidencyState::PendingUpload);
                }
            }
            std::vector<uint32_t> requests(kRequestWords);
            requests[0] = kRequestCapacity;
            requests[1] = kRequestCapacity;
            requests[4] = params.frameIndex;
            requests[11] = cooperative ? kPriorityOffset : 0u;
            requests[12] = cooperative ? kPriorityTable : 0u;
            requests[13] = prefetch && test % 5 != 0 ? std::max(kRequestCapacity / 4u, 1u) : 0u;
            if (rtDemandC3 && prefetch) { requests[13] = 8u; }
            if (prefetch && test % 3 == 1) {
                requests[0] = std::max(static_cast<uint32_t>(expected.requestedGroups.size()), 1u);
            }
            std::fill(requests.begin() + kPriorityTable, requests.end(), 0x7fc00000u);
            if (rtDemandC2 && !upload(Topology,
                rtStep == 16 ? invalidDemandTopology.data() : topology.data(), sizes[Topology])) {
                return RHITestResult::fail("C2 node demand bounds upload failed");
            }
            if (!upload(Instance, &instance, sizeof(instance)) || !upload(Params, &params, sizeof(params)) ||
                !upload(Primitive, &primitive, sizeof(primitive)) ||
                !upload(Groups, groups.data(), sizes[Groups]) ||
                !upload(PageTable, pages.data(), sizes[PageTable]) || !upload(Requests, requests.data(), sizes[Requests])) {
                return RHITestResult::fail("stream frame input map");
            }
            if (frame != 0) { STREAM_LOD_REQUIRE(fence->reset()); STREAM_LOD_REQUIRE(pool->reset()); }
            STREAM_LOD_REQUIRE(commands->begin());
            std::array<BufferBarrierDesc, BufferCount> barriers{};
            for (uint32_t index = 0; index < BufferCount; ++index) {
                barriers[index] = {
                    .buffer = buffers[index].get(),
                    .before = metallic::render::resourceSyncScope(frame == 0 ? ResourceState::Undefined : ResourceState::General, metallic::render::PipelineStageBits::AllCommands),
                    .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                };
            }
            if (auto commandResult = commands->synchronize({.buffers = {barriers.data(), BufferCount}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            if (auto commandResult = commands->bindBindlessHeap(*heap); !commandResult) { return RHITestResult::fail(std::string("bindBindlessHeap failed: ") + render::resultToString(commandResult)); }
            if (cooperative) {
                if (auto commandResult = commands->bindExecution((traversalPipeline)->execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
                push.traversalPhase = 2u;
                if (auto commandResult = commands->pushBindlessData(&push, sizeof(push)); !commandResult) { return RHITestResult::fail(std::string("pushBindlessData failed: ") + render::resultToString(commandResult)); }
                if (auto commandResult = commands->dispatch((kGroupCount + 63u) / 64u, 1, 1); !commandResult) { return RHITestResult::fail(std::string("dispatch failed: ") + render::resultToString(commandResult)); }
                if (auto commandResult = commands->synchronize({.buffers = {barriers.data(), BufferCount}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            }
            if (auto commandResult = commands->bindExecution((pipeline)->execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
            const bool clearRTState = rtDemand && (rtStep == 0 || (rtDemandC3 && rtStep == 30));
            if ((rtDemand ? clearRTState : frame == caseCount) || (telemetrySequence && (telemetryStep == 0 || telemetryStep == 18))) {
                // The compatibility linear path predates sparse state. Run
                // the production initializer once when changing algorithms.
                push.activeBuildPhase = 8;
                if (auto commandResult = commands->pushBindlessData(&push, sizeof(push)); !commandResult) { return RHITestResult::fail(std::string("pushBindlessData failed: ") + render::resultToString(commandResult)); }
                const uint32_t initGroups = static_cast<uint32_t>((sizes[State] / 4 + 63) / 64);
                if (auto commandResult = commands->dispatch(std::min(initGroups, 65535u), (initGroups + 65534) / 65535, 1); !commandResult) { return RHITestResult::fail(std::string("dispatch failed: ") + render::resultToString(commandResult)); }
                if (auto commandResult = commands->synchronize({.buffers = {barriers.data(), BufferCount}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            }
            bool capturedPrefetchHistory = false;
            for (uint32_t phase : {0u, 12u, 10u, 13u, 5u, 11u, 6u, 7u, 2u, 9u, 9u}) {
                if (phase == 9u && !prefetch) { continue; }
                if ((phase == 10u || phase == 11u) && !split) { continue; }
                if ((phase == 12u || phase == 13u) && !distributed) { continue; }
                if (rtDemandC3 && phase == 9u && !capturedPrefetchHistory) {
                    BufferBarrierDesc historyBarrier{.buffer = buffers[State].get(),
                        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                    STREAM_LOD_REQUIRE(commands->synchronize({.buffers = {&historyBarrier, 1}}));
                    const uint64_t historyByteOffset = (4ull + kGroupCount * 3ull + 4ull) * sizeof(uint32_t);
                    auto source = buffers[State]->slice({historyByteOffset, rtHistoryWords * sizeof(uint32_t)});
                    auto destination = readback->slice({prefetchHistoryOffset, rtHistoryWords * sizeof(uint32_t)});
                    if (!source || !destination) { return RHITestResult::fail("C3 prefetch history snapshot slice failed"); }
                    STREAM_LOD_REQUIRE(commands->copyBuffer(*source, *destination));
                    std::swap(historyBarrier.before, historyBarrier.after);
                    STREAM_LOD_REQUIRE(commands->synchronize({.buffers = {&historyBarrier, 1}}));
                    capturedPrefetchHistory = true;
                }
                if (auto commandResult = commands->bindExecution((phase == 12u || phase == 13u ? *demandPipeline :
                    cooperative && (phase == 5u || phase == 7u || phase == 9u || phase == 10u || phase == 11u) ? *cooperativePipeline : *pipeline).execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
                push.activeBuildPhase = phase;
                if (auto commandResult = commands->pushBindlessData(&push, sizeof(push)); !commandResult) { return RHITestResult::fail(std::string("pushBindlessData failed: ") + render::resultToString(commandResult)); }
                if (auto commandResult = commands->dispatch(phase == 12u ? static_cast<uint32_t>((sizes[Demand] / 4 + 63) / 64) : phase == 13u ? 4u : 1u, 1, 1); !commandResult) { return RHITestResult::fail(std::string("dispatch failed: ") + render::resultToString(commandResult)); }
                for (auto& barrier : barriers) { barrier.before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}; }
                if (auto commandResult = commands->synchronize({.buffers = {barriers.data(), BufferCount}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            }
            if (cooperative) {
                if (auto commandResult = commands->bindExecution((traversalPipeline)->execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
                push.traversalPhase = 3u;
                if (auto commandResult = commands->pushBindlessData(&push, sizeof(push)); !commandResult) { return RHITestResult::fail(std::string("pushBindlessData failed: ") + render::resultToString(commandResult)); }
                if (auto commandResult = commands->dispatch((kRequestCapacity + 63u) / 64u, 1, 1); !commandResult) { return RHITestResult::fail(std::string("dispatch failed: ") + render::resultToString(commandResult)); }
                if (auto commandResult = commands->synchronize({.buffers = {barriers.data(), BufferCount}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            }
            for (uint32_t index = 0; index < std::size(outputs); ++index) {
                BufferBarrierDesc barrier{
                    .buffer = buffers[outputs[index]].get(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                };
                if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                {
                    auto sourceSlice = (buffers[outputs[index]].get())->slice({0, outputSizes[index]});
                    if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                    auto destinationSlice = readback.get()->slice({offsets[index], outputSizes[index]});
                    if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                    if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
                }
                if (large && !rtDemand && outputs[index] == State) {
                    {
                        auto sourceSlice = (buffers[State].get())->slice({sizes[State] - 3 * sizeof(uint32_t), 3 * sizeof(uint32_t)});
                        if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                        auto destinationSlice = readback.get()->slice({tailOffset, 3 * sizeof(uint32_t)});
                        if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                        if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
                    }
                }
                std::swap(barrier.before, barrier.after);
                if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            }
            STREAM_LOD_REQUIRE(commands->end());
            CommandBuffer* submitted[] = {commands.get()};
            STREAM_LOD_REQUIRE(queue->submit({.commandBuffers = {submitted, 1}, .signalFence = fence.get()}));
            STREAM_LOD_REQUIRE(fence->wait());
            readback->invalidate();
            const auto* mapped = static_cast<const uint8_t*>(readback->map());
            if (mapped == nullptr) { return RHITestResult::fail("stream result map"); }
            MeshletStreamGPUActiveHeader header;
            std::memcpy(&header, mapped, sizeof(header));
            std::vector<MeshletStreamGPUActiveGroup> active(kGroupCount);
            std::memcpy(active.data(), mapped + offsets[1], sizes[ActiveGroups]);
            std::memcpy(requests.data(), mapped + offsets[2], sizes[Requests]);
            MeshletStreamGPUDrawIndirect arguments;
            std::memcpy(&arguments, mapped + offsets[3], sizeof(arguments));
            MeshletStreamGPUDrawIndirect tessellationArguments;
            std::memcpy(&tessellationArguments, mapped + offsets[3] + sizeof(arguments), sizeof(tessellationArguments));
            std::vector<uint32_t> state(logicalStateBytes / sizeof(uint32_t));
            std::memcpy(state.data(), mapped + offsets[4], logicalStateBytes);
            std::array<uint32_t, kMeshletStreamDemandStatsWords> demandStats{};
            std::memcpy(demandStats.data(), mapped + offsets[5], sizeof(demandStats));
            if (rtDemand) {
                const uint32_t history = 4u + kGroupCount * 3u + 4u;
                const uint32_t expectedPreviousToken = !clearRTState && manual == UINT32_MAX &&
                    previousHistoryPolicy == params.rtLodPolicyRevision ? previousHistoryToken : 0u;
                if (state[history] != expectedPreviousToken || state[history + 1u] != params.rtLodFrameToken ||
                    state[history + 2u] != params.rtLodPolicyRevision) {
                    readback->unmap();
                    return RHITestResult::fail("Submitted RT history token/policy mismatch: " + caseLabel);
                }
                previousHistoryToken = params.rtLodFrameToken;
                previousHistoryPolicy = params.rtLodPolicyRevision;
                if (manual == UINT32_MAX) {
                    for (uint32_t group = 0; group < kGroupCount; ++group) {
                        if ((fixture.groups[group].flags & kMeshletLODTerminalGroup) != 0) { continue; }
                        const uint32_t address = history + 3u + group * (packedHistory ? 1u : 2u);
                        const bool actualWanted = state[address] == params.rtLodFrameToken && (packedHistory || state[address + 1u] != 0u);
                        if (actualWanted != bool(expectedWanted[group])) {
                            readback->unmap();
                            return RHITestResult::fail("Brute-force leaf / GPU RT demand mismatch at group " +
                                std::to_string(group) + ", " + caseLabel);
                        }
                        if (sparseHistoryComparison && !expectedWanted[group]) {
                            if (rtMode == 0u) {
                                baselineFalseWrites += state[address] == params.rtLodFrameToken && state[address + 1u] == 0u;
                            } else {
                                // A false/pruned leaf must expire by header token,
                                // without overwriting either legacy-layout word.
                                const uint32_t oldToken = clearRTState ? 0u : previousHistoryWords[address - history];
                                const uint32_t oldWanted = clearRTState ? 0u : previousHistoryWords[address + 1u - history];
                                if (state[address] != oldToken || state[address + 1u] != oldWanted) {
                                    readback->unmap();
                                    return RHITestResult::fail("Sparse false demand wrote legacy history words at group " +
                                        std::to_string(group) + ", " + caseLabel);
                                }
                                ++sparseFalsePreserved;
                                sparseExpiredWanted += oldToken != 0u && oldWanted == 1u;
                            }
                        }
                    }
                }
                if (sparseHistoryComparison) {
                    std::copy_n(state.begin() + history, rtHistoryWords, previousHistoryWords.begin());
                }
                if (rtDemandC3 && prefetch && (!capturedPrefetchHistory ||
                    std::memcmp(mapped + prefetchHistoryOffset, state.data() + history, rtHistoryWords * sizeof(uint32_t)) != 0)) {
                    readback->unmap();
                    return RHITestResult::fail("RT prefetch mutated demand history despite updateHistory=false");
                }
                previousWanted = expectedWanted;
                const uint32_t testedGroups = distributed ? demandStats[3] : state[4u + kGroupCount * 2u + 2u];
                rtGroupTests[rtBaseline ? 0u : 1u] += testedGroups;
                if (rtDemandC3) {
                    const uint32_t index = (distributed ? rtSequenceFrames : 0u) + rtStep;
                    if (rtMode == 0) { eagerGroupTests[index] = testedGroups; }
                    else if (eagerGroupTests[index] != testedGroups) {
                        readback->unmap();
                        return RHITestResult::fail("RT leaf candidate changed traversal work instead of only leaf demand evaluation: " + caseLabel);
                    }
                }
                if (rtDemandC2 && rtStep < 5) { rtUnitScaleTests[distributed ? 1u : 0u][rtBaseline ? 0u : 1u] += testedGroups; }
                if (rtDemandC2 && rtStep == 16 && testedGroups != kGroupCount) {
                    readback->unmap();
                    return RHITestResult::fail("C2 pruned unknown node demand bounds instead of descending conservatively");
                }
            }
            if (telemetrySequence) {
                const uint32_t expectedCatchup = telemetryStep == 4 ? 1u : telemetryStep == 6 ? 2u : 0u;
                const uint32_t expectedThreshold = telemetryStep == 1 ? 2u : 0u;
                const uint32_t expectedOwnBlocked = telemetryStep == 3 || telemetryStep == 5 ||
                    telemetryStep == 7 || telemetryStep == 12 ? 1u : 0u;
                const uint32_t expectedDependencyBlocked = telemetryStep == 5 ? 2u : 0u;
                if (demandStats[20] != expectedOwnBlocked || demandStats[21] != expectedDependencyBlocked ||
                    demandStats[23] != expectedCatchup || demandStats[24] != expectedCatchup * 2u ||
                    demandStats[25] != expectedThreshold || demandStats[26] != expectedThreshold * 2u) {
                    return RHITestResult::fail("LOD transition telemetry mismatch: path " + std::to_string(telemetryPath) +
                        ", step " + std::to_string(telemetryStep) + ", own/dependency/catchup/threshold " +
                        std::to_string(demandStats[20]) + "/" + std::to_string(demandStats[21]) + "/" +
                        std::to_string(demandStats[23]) + "/" + std::to_string(demandStats[25]));
                }
                if ((telemetryStep == 9 || telemetryStep == 11 || telemetryStep == 15 ||
                        telemetryStep == 17 || telemetryStep == 19) && demandStats[27] != 2u) {
                    return RHITestResult::fail("First/reentry/manual/frozen/wrap selection was not left unclassified");
                }
                if (telemetryStep == 6 && demandStats[22] != 3u) {
                    return RHITestResult::fail("Missing parent did not preserve descendant catch-up history");
                }
                if (telemetryStep == 13 && (demandStats[22] != 1u || demandStats[23] != 0u)) {
                    return RHITestResult::fail("Capacity fallback was misreported as selected catch-up");
                }
            }
            if ((distributed || switching) && (demandStats[17] != uint32_t(distributed) ||
                    demandStats[18] != (distributed ? demandStats[3] : state[4 + kGroupCount * 2 + 2]))) {
                return RHITestResult::fail("Demand policy feedback did not match the current traversal");
            }
            if (distributed) {
                uint32_t tasks = 0;
                for (uint32_t bin = 8; bin < 16; ++bin) { tasks += demandStats[bin]; }
                if (demandStats[1] > (instance.visible ? demandRoots.size() : 0u) || tasks != demandStats[1] ||
                    demandStats[16] != demandStats[1] ||
                    demandStats[4] > 8 || demandStats[3] > demandStats[7]) {
                    return RHITestResult::fail("Distributed task accounting, bound or wave occupancy mismatch");
                }
            }
            std::array<uint32_t, 3> tail{};
            if (large && !rtDemand) { std::memcpy(tail.data(), mapped + tailOffset, sizeof(tail)); }
            readback->unmap();
            if (large && !rtDemand && !std::all_of(tail.begin(), tail.end(),
                [&](uint32_t word) { return word == (useBvh ? 0u : 0xa5a5a5a5u); })) {
                return RHITestResult::fail("2D LOD state initialization did not clear the partial last row: " + caseLabel);
            }
            if (header.activeGroupCount > capacity || header.activeGroupCount > kGroupCount) {
                return RHITestResult::fail("stream frontier overflow: " + caseLabel);
            }
            std::vector<uint32_t> actual;
            for (uint32_t index = 0; index < header.activeGroupCount; ++index) {
                const auto& group = active[index];
                if (group.pageIndex >= kGroupCount || group.gpuSceneInstanceIndex != 17 || group.instanceIndex != 0) {
                    return RHITestResult::fail("stream selected group identity");
                }
                for (uint32_t cluster = 0; cluster < group.clusterCount; ++cluster) {
                    if ((group.clusterSelectionMask & (1u << cluster)) != 0) {
                        actual.push_back(fixture.ranges[group.pageIndex].clusterOffset + cluster);
                    }
                }
            }
            if (requests[2] > requests[0] || requests[5] != 0 || requests[7] != 0) {
                return RHITestResult::fail("stream frontier request overflow or invalid page");
            }
            std::vector<uint32_t> requested(requests.begin() + 16, requests.begin() + 16 + requests[2]);
            if (rtDemandC3 && prefetch && std::find(requested.begin(), requested.end(), kStreamPrefetchPageTag) == requested.end()) {
                return RHITestResult::fail("C3 read-only history fixture did not issue its forecast-only leaf request");
            }
            if (forecastExclusive && std::find(requested.begin(), requested.end(), 1u | kStreamPrefetchPageTag) == requested.end()) {
                return RHITestResult::fail("Predicted approach did not request the forecast-exclusive page");
            }
            uint32_t forecastCount = 0;
            std::vector<uint8_t> requestedOnce(kGroupCount);
            for (uint32_t encoded : requested) {
                const bool tagged = (encoded & kStreamPrefetchPageTag) != 0;
                const uint32_t page = encoded & ~kStreamPrefetchPageTag;
                if (page >= kGroupCount || requestedOnce[page]++) { return RHITestResult::fail("Invalid or duplicate forecast page"); }
                if (!tagged && forecastCount != 0) { return RHITestResult::fail("Forecast displaced the demand prefix"); }
                if (!tagged) { continue; }
                ++forecastCount;
                auto forecastView = fixture.view;
                if (params.prefetchParams[3] != 0.f) {
                    for (uint32_t axis = 0; axis < 3; ++axis) {
                        forecastView.eye[axis] = params.prefetchEye[axis];
                        forecastView.forward[axis] = params.prefetchCenter[axis] - params.prefetchEye[axis];
                    }
                }
                auto forecastMetric = forecastView;
                forecastMetric.projection[3] *= .85f;
                if (!meshletLodNeedsFine(fixture.groups[page], fixture.instance, forecastMetric, UINT32_MAX)) {
                    return RHITestResult::fail("Forecast ignored predicted-camera screen error: " + caseLabel);
                }
                forecastView.projection[1] *= 1.125f;
                forecastView.projection[2] *= 1.125f;
                if (!prefetch || manual != UINT32_MAX || available[page] ||
                    (fixture.groups[page].flags & kMeshletLODTerminalGroup) != 0 ||
                    !meshletLodBoundsVisible(refinementBounds[page], fixture.instance, forecastView,
                        {0, 1, 0}, viewAspect, 1000000.f)) {
                    return RHITestResult::fail("Forecast ignored availability, view or manual LOD: " + caseLabel);
                }
            }
            if (forecastCount > requests[13]) { return RHITestResult::fail("Forecast exceeded its quota"); }
            if (prefetch && !large && test == 1 && requests[15] == 0) {
                return RHITestResult::fail("Saturated demand buffer did not drop speculative requests");
            }
            speculativeRequests += forecastCount;
            if (params.prefetchParams[3] != 0.f) { predictedRequests += forecastCount; }
            if (cooperative) {
                for (uint32_t index = 0; index < requests[2]; ++index) {
                    const uint32_t bits = requests[kPriorityOffset + index];
                    float value;
                    std::memcpy(&value, &bits, sizeof(value));
                    if (!std::isfinite(value) || value < 0.f || value > 1e9f ||
                        bits != requests[kPriorityTable + (requested[index] & ~kStreamPrefetchPageTag)]) {
                        return RHITestResult::fail("GPU page priority clear/aggregate/gather mismatch");
                    }
                    positivePriorities += value > 0.f;
                }
                for (uint32_t page = 0; page < kGroupCount; ++page) {
                    if (!prefetch && std::find(requested.begin(), requested.end(), page) == requested.end() &&
                        requests[kPriorityTable + page] != 0u) {
                        return RHITestResult::fail("Unused page retained a stale screen benefit");
                    }
                }
            }
            std::erase_if(requested, [](uint32_t page) { return (page & kStreamPrefetchPageTag) != 0; });
            std::sort(requested.begin(), requested.end());
            if (actual != expected.selectedClusters || requested != expected.requestedGroups ||
                (expected.valid && bool(header.padding0) != expected.capacityFallback) ||
                arguments.groupCountX != header.activeGroupCount * 4 || arguments.groupCountY != 1 || arguments.groupCountZ != 1) {
                return RHITestResult::fail("CPU/GPU stream cut, requests, capacity fallback or indirect arguments mismatch in case " +
                    caseLabel + ": expected " + std::to_string(expected.selectedClusters.size()) +
                    " clusters, actual " + std::to_string(actual.size()));
            }
            const uint32_t tessellationTasks = header.activeGroupCount * 16;
            if (tessellationArguments.groupCountX != std::min(tessellationTasks, 65535u) ||
                tessellationArguments.groupCountY != std::max(1u, (tessellationTasks + 65534u) / 65535u) ||
                tessellationArguments.groupCountZ != 1) {
                return RHITestResult::fail("Recursive tessellation indirect arguments mismatch: " + caseLabel);
            }
            const uint32_t stats = 4 + kGroupCount * 2;
            if (useBvh && !cooperative && (state[stats] > kGroupCount || state[stats + 1] != expectedState.visitedBvhNodes ||
                state[stats + 2] != expectedState.testedGroups)) {
                return RHITestResult::fail("GPU/CPU BVH traversal statistics mismatch: " + caseLabel +
                    ", visited " + std::to_string(state[stats + 1]) + "/" + std::to_string(expectedState.visitedBvhNodes) +
                    ", tested " + std::to_string(state[stats + 2]) + "/" + std::to_string(expectedState.testedGroups));
            }
            if (useBvh && large && !rtDemand && (test == 1 || test == 11 || test == 18 || test == 24) &&
                (state[stats + 2] >= kGroupCount / 8 || actual.size() != 2)) {
                return RHITestResult::fail("GPU BVH failed to prune coarse branches: " + caseLabel);
            }
            if (useBvh && expectedState.valid) {
                uint32_t activeCount = 0;
                std::vector<uint8_t> seen(kGroupCount);
                for (uint32_t index = 0; index < state[stats]; ++index) {
                    const uint32_t group = state[stats + 4 + index];
                    if (group >= kGroupCount || seen[group] != 0 || expectedState.activeGroups[group] == 0) {
                        return RHITestResult::fail("GPU sparse frontier has duplicate/stale groups: " + caseLabel);
                    }
                    seen[group] = 1;
                }
                for (uint32_t group = 0; group < kGroupCount; ++group) {
                    activeCount += expectedState.activeGroups[group];
                    if (state[4 + group * 2] != expectedState.activeGroups[group] ||
                        (expectedState.activeGroups[group] == 0 && state[4 + group * 2 + 1] != 0)) {
                        return RHITestResult::fail("GPU retained stale group state across frames: " + caseLabel);
                    }
                }
                if (state[stats] != activeCount) {
                    return RHITestResult::fail("GPU sparse frontier count differs from active bits: " + caseLabel);
                }
            }
        }
        if (rtDemand) {
            if (sparseHistoryComparison) {
                if (baselineFalseWrites == 0u || sparseFalsePreserved == 0u || sparseExpiredWanted == 0u) {
                    return RHITestResult::fail("Sparse history fixture did not observe baseline false stores and expired wanted tokens");
                }
                return RHITestResult::pass(std::to_string(frameCount) +
                    " dual-word late-history frames match per-leaf demand, cuts, requests and traversal counts; " +
                    std::to_string(baselineFalseWrites) + " baseline false stores, " +
                    std::to_string(sparseFalsePreserved) + " sparse false records untouched, " +
                    std::to_string(sparseExpiredWanted) + " old wanted tokens expire; exact hysteresis boundaries, token gaps, wrap and readonly prefetch preserved");
            }
            if (rtDemandC3) {
                return RHITestResult::pass("216 eager/late/ray-first ordered/distributed frames match brute-force leaf demand, cut and requests with " +
                    std::string(packedHistory ? "packed" : "legacy") + " history; gaps, wrap and read-only prefetch preserved");
            }
            if (rtDemandC2 && (rtUnitScaleTests[0][0] != rtUnitScaleTests[0][1] ||
                rtUnitScaleTests[1][0] != rtUnitScaleTests[1][1])) {
                return RHITestResult::fail("C2 changed traversal work at unit offscreen scale");
            }
            if (rtGroupTests[1] >= rtGroupTests[0]) {
                return RHITestResult::fail(std::string(rtDemandC2 ? "C2" : "C1") + " did not reduce behind-eye leaf tests versus the baseline: " +
                    std::to_string(rtGroupTests[1]) + "/" + std::to_string(rtGroupTests[0]));
            }
            return RHITestResult::pass(std::to_string(frameCount) + " consecutive " + (rtDemandC2 ? "C1/C2" : "baseline/C1") +
                " ordered/distributed frames match brute-force leaf demand, cuts and requests; leaf tests " +
                std::to_string(rtGroupTests[0]) + " -> " + std::to_string(rtGroupTests[1]));
        }
        if (positivePriorities == 0) { return RHITestResult::fail("Visible requests never generated screen benefit"); }
        if (speculativeRequests == 0) { return RHITestResult::fail("Lookahead never requested a missing descendant"); }
        if (predictedRequests == 0) { return RHITestResult::fail("Predicted camera never requested a missing descendant"); }
        if (viewDemandReductions == 0) { return RHITestResult::fail("View demand never pruned an offscreen refinement"); }
        return RHITestResult::pass("GPU-reference cuts, requests, sparse state and fallback agree");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLODGPUTest);

class StreamMeshletLODRTDemandC1Test final : public RHITest {
public:
    StreamMeshletLODRTDemandC1Test() { type = RHITestType::Rendering; name = "meshlet_lod_stream_rt_demand_c1"; }
    RHITestResult run(RHITestContext& context) override
    {
        return StreamMeshletLODGPUTest{}.runFixture(context, true, 256, true);
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLODRTDemandC1Test);

class StreamMeshletLODRTDemandC2Test final : public RHITest {
public:
    StreamMeshletLODRTDemandC2Test() { type = RHITestType::Rendering; name = "meshlet_lod_stream_rt_demand_c2"; }
    RHITestResult run(RHITestContext& context) override
    {
        return StreamMeshletLODGPUTest{}.runFixture(context, true, 256, true, true);
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLODRTDemandC2Test);

class StreamMeshletLODRTDemandC3Test final : public RHITest {
public:
    StreamMeshletLODRTDemandC3Test() { type = RHITestType::Rendering; name = "meshlet_lod_stream_rt_demand_c3"; }
    RHITestResult run(RHITestContext& context) override
    {
        for (bool packed : {false, true}) {
            const auto result = StreamMeshletLODGPUTest{}.runFixture(context, true, 256, true, true, true, packed);
            if (!result.passed) { return result; }
        }
        return RHITestResult::pass("432 eager/late/ray-first frames match the per-leaf oracle in both history layouts");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLODRTDemandC3Test);

class StreamMeshletLODRTSparseHistoryTest final : public RHITest {
public:
    StreamMeshletLODRTSparseHistoryTest() { type = RHITestType::Rendering; name = "meshlet_lod_stream_rt_sparse_history"; }
    RHITestResult run(RHITestContext& context) override
    {
        return StreamMeshletLODGPUTest{}.runFixture(context, true, 256, true, true, true, false, true);
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLODRTSparseHistoryTest);


class StreamLODTileHierarchyTest final : public RHITest {
public:
    StreamLODTileHierarchyTest() { type = RHITestType::Validation; name = "meshlet_lod_stream_tile_hierarchy"; }
    RHITestResult run(RHITestContext&) override
    {
        StreamLODFixture fixture(true, 8192);
        std::vector<MeshletLODBVHNode> nodes;
        std::string reason;
        if (!buildMeshletLodTiles(fixture.groups, nodes, reason)) { return RHITestResult::fail(reason); }
        uint32_t expectedEnd = static_cast<uint32_t>(fixture.groups.size()), leafCount = 0;
        for (uint32_t i = 0; i < nodes.size(); ++i) {
            const auto& node = nodes[i];
            if (node.escapeIndex <= i || node.escapeIndex > nodes.size()) { return RHITestResult::fail("Invalid tile escape"); }
            if (node.groupCount == 0) { continue; }
            ++leafCount;
            if (node.groupCount > 64 || node.groupOffset + node.groupCount != expectedEnd || node.escapeIndex != i + 1) {
                return RHITestResult::fail("Tile leaves lost parent-before-child order");
            }
            expectedEnd = node.groupOffset;
            for (uint32_t j = 0; j < node.groupCount; ++j) {
                if (fixture.groups[node.groupOffset + j].level != node.maxLevel) {
                    return RHITestResult::fail("Dependent LOD levels share a tile");
                }
            }
        }
        if (expectedEnd != 0 || leafCount < 100) { return RHITestResult::fail("Incomplete hierarchy fixture"); }
        for (uint32_t limit : {1u, 2u, 8u, 32u}) {
            const auto roots = buildMeshletLodDemandRoots(nodes, limit);
            std::vector<uint8_t> covered(nodes.size());
            for (uint32_t root : roots) {
                if (nodes[root].escapeIndex - root > limit) { return RHITestResult::fail("Unbounded demand task"); }
                for (uint32_t i = root; i < nodes[root].escapeIndex; ++i) {
                    if (nodes[i].groupCount != 0 && ++covered[i] != 1) { return RHITestResult::fail("Overlapping demand tasks"); }
                }
            }
            for (uint32_t i = 0; i < nodes.size(); ++i) {
                if (nodes[i].groupCount != 0 && covered[i] != 1) { return RHITestResult::fail("Missing demand task leaf"); }
            }
        }
        uint32_t bestVisited = UINT32_MAX;
        for (uint32_t test = 0; test < 32; ++test) {
            fixture.view.projection[3] = std::exp2(float(test) - 12.f);
            fixture.view.forward[3] = test % 2;
            const auto needs = [&](const MeshletLODBVHNode& node) {
                return meshletLodNeedsFine({.sphere = node.sphere, .error = node.maxError,
                    .level = node.maxLevel, .flags = node.flags}, fixture.instance, fixture.view);
            };
            std::vector<uint32_t> expected, actual;
            for (uint32_t i = 0; i < nodes.size(); ++i) {
                if (nodes[i].groupCount != 0 && needs(nodes[i])) { expected.push_back(i); }
            }
            uint32_t visited = 0;
            for (uint32_t i = 0; i < nodes.size();) {
                ++visited;
                if (!needs(nodes[i])) { i = nodes[i].escapeIndex; }
                else { if (nodes[i].groupCount != 0) { actual.push_back(i); } ++i; }
            }
            if (actual != expected) { return RHITestResult::fail("Hierarchy pruned a demanded tile"); }
            bestVisited = std::min(bestVisited, visited);
        }
        if (bestVisited >= leafCount / 4) { return RHITestResult::fail("Coarse traversal still scans the tile list"); }
        return RHITestResult::pass("16383 groups: leaf order and demand identical; coarse traversal visits " +
            std::to_string(bestVisited) + " hierarchy nodes versus " + std::to_string(leafCount) + " flat tiles");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamLODTileHierarchyTest);

class StreamLODCapacityTest final : public RHITest {
public:
    StreamLODCapacityTest() { type = RHITestType::Rendering; name = "meshlet_lod_stream_per_instance_budget"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream capacity regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless compute"); }
        STREAM_LOD_REQUIRE(created);
        constexpr uint32_t count = 257;
        std::unique_ptr<BindlessHeap> heap;
        STREAM_LOD_REQUIRE(device->createBindlessHeap({.maxBuffers = 3}).transform([&](auto rhiValue) { heap = std::move(rhiValue); }));
        std::array<std::unique_ptr<Buffer>, 3> buffers;
        std::array<BindlessHandle, 3> handles;
        const uint64_t sizes[] = {sizeof(MeshletStreamGPUParams), count * 16u, sizeof(MeshletStreamGPUActiveHeader)};
        const uint32_t strides[] = {sizeof(MeshletStreamGPUParams), 4u, sizeof(MeshletStreamGPUActiveHeader)};
        for (uint32_t i = 0; i < 3; ++i) {
            STREAM_LOD_REQUIRE(device->createBuffer({.size = sizes[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto rhiValue) { buffers[i] = std::move(rhiValue); }));
            STREAM_LOD_REQUIRE(heap->allocate(metallic::render::BindlessHandleKind::Buffer).transform([&](auto rhiValue) { handles[i] = std::move(rhiValue); }));
            STREAM_LOD_REQUIRE((*buffers[i]).slice().and_then([&](const auto& bufferSlice) { return heap->writeStorageBuffer(handles[i], bufferSlice); }));
        }
        ShaderCompileResult compiled;
        STREAM_LOD_REQUIRE(compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamActiveBuildEntryPoint, .searchPath = kMeshletStreamShaderSearchPath}, compiled.diagnostics).transform([&](auto value) { compiled = std::move(value); }));
        std::unique_ptr<ShaderModule> shader;
        STREAM_LOD_REQUIRE(device->createShaderModule({.spirv = compiled.spirv}).transform([&](auto rhiValue) { shader = std::move(rhiValue); }));
        std::unique_ptr<ComputePipeline> pipeline;
        STREAM_LOD_REQUIRE(device->createComputePipeline({
            .computeShader = {shader.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush),
        }).transform([&](auto rhiValue) { pipeline = std::move(rhiValue); }));
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        STREAM_LOD_REQUIRE(device->createCommandPool(*queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        STREAM_LOD_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
        STREAM_LOD_REQUIRE(device->createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }));
        const auto write = [&](uint32_t index, const void* source) {
            auto* data = buffers[index]->map();
            if (!data) { return false; }
            std::memcpy(data, source, sizes[index]);
            buffers[index]->flush(); buffers[index]->unmap(); return true;
        };
        uint32_t mixedCuts = 0;
        for (uint32_t test = 0; test < 24; ++test) {
            std::vector<uint32_t> state(count * 4);
            uint64_t selectedTotal = 0, baselineTotal = 0;
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t roots = i % 9 == 0 ? 0u : 1u + i % 3;
                uint32_t selected = roots == 0 ? 0u : roots + i % 11;
                if (i % 13 == 0 && roots != 0) { selected = 1; } // A detail cut can be cheaper.
                if (test >= 16 && i % 53 == 1) { selected = UINT32_MAX; }
                state[i * 4] = selected; state[i * 4 + 2] = roots; state[i * 4 + 3] = roots != 0;
                selectedTotal += selected;
                baselineTotal += std::min(roots, selected);
            }
            const uint32_t capacities[] = {0u, 1u, uint32_t(baselineTotal - 1), uint32_t(baselineTotal),
                uint32_t(baselineTotal + 17), uint32_t(baselineTotal + 400), 4096u, 65536u};
            MeshletStreamGPUParams params;
            params.sceneInstanceCount = count;
            params.lodTopologyBuffer = handles[1].shaderIndex;
            params.lodStateBuffer = handles[1].shaderIndex;
            params.activeGroupCount = capacities[test % 8];
            MeshletStreamGPUActiveHeader header{};
            if (!write(0, &params) || !write(1, state.data()) || !write(2, &header)) { return RHITestResult::fail("Capacity input map"); }
            if (test != 0) { STREAM_LOD_REQUIRE(fence->reset()); STREAM_LOD_REQUIRE(pool->reset()); }
            STREAM_LOD_REQUIRE(commands->begin());
            commands->hostWriteBarrier();
            std::array<BufferBarrierDesc, 3> barriers;
            for (uint32_t i = 0; i < 3; ++i) {
                barriers[i] = {
                    .buffer = buffers[i].get(),
                    .before = metallic::render::resourceSyncScope(test == 0 ? ResourceState::Undefined : ResourceState::General, metallic::render::PipelineStageBits::AllCommands),
                    .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                };
            }
            if (auto commandResult = commands->synchronize({.buffers = {barriers.data(), 3}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            if (auto commandResult = commands->bindBindlessHeap(*heap); !commandResult) { return RHITestResult::fail(std::string("bindBindlessHeap failed: ") + render::resultToString(commandResult)); } if (auto commandResult = commands->bindExecution((pipeline)->execution()); !commandResult) { return RHITestResult::fail(std::string("bindExecution failed: ") + render::resultToString(commandResult)); }
            MeshletStreamUserPush push{};
            push.paramsBuffer = handles[0].shaderIndex; push.activeHeaderBuffer = handles[2].shaderIndex;
            push.activeBuildPhase = kMeshletStreamActiveBuildPrefixPhase;
            if (auto commandResult = commands->pushBindlessData(&push, sizeof(push)); !commandResult) { return RHITestResult::fail(std::string("pushBindlessData failed: ") + render::resultToString(commandResult)); } if (auto commandResult = commands->dispatch(1, 1, 1); !commandResult) { return RHITestResult::fail(std::string("dispatch failed: ") + render::resultToString(commandResult)); }
            STREAM_LOD_REQUIRE(commands->end());
            CommandBuffer* list[] = {commands.get()};
            STREAM_LOD_REQUIRE(queue->submit({.commandBuffers = {list, 1}, .signalFence = fence.get()}));
            STREAM_LOD_REQUIRE(fence->wait());
            buffers[2]->invalidate();
            auto* headerData = buffers[2]->map();
            if (!headerData) { return RHITestResult::fail("Capacity header map"); }
            std::memcpy(&header, headerData, sizeof(header)); buffers[2]->unmap();
            buffers[1]->invalidate();
            auto* output = static_cast<const uint32_t*>(buffers[1]->map());
            if (!output) { return RHITestResult::fail("Capacity state map"); }
            const bool overflow = selectedTotal > params.activeGroupCount;
            const bool invalid = baselineTotal > params.activeGroupCount;
            uint64_t extraPrefix = 0;
            uint32_t outputCount = 0, fallbackCount = 0;
            for (uint32_t i = 0; i < count && !invalid; ++i) {
                const auto selected = state[i * 4], roots = state[i * 4 + 2];
                const uint64_t extra = selected - std::min(selected, roots);
                const bool fallback = overflow && extra != 0 && extraPrefix + extra > params.activeGroupCount - baselineTotal;
                extraPrefix += extra;
                if (output[i * 4 + 1] != outputCount || bool(output[i * 4 + 3] & 2u) != fallback) {
                    buffers[1]->unmap(); return RHITestResult::fail("Mixed-cut prefix/decision mismatch");
                }
                outputCount += fallback ? roots : selected;
                fallbackCount += fallback;
            }
            buffers[1]->unmap();
            if (header.activeGroupCount != (invalid ? 0u : outputCount) ||
                header.overflowCount != (invalid ? 2u : overflow ? 1u : 0u) ||
                header.padding2 != (invalid ? 0u : fallbackCount) || header.activeGroupCount > params.activeGroupCount) {
                return RHITestResult::fail("Capacity did not preserve complete bounded output");
            }
            mixedCuts += !invalid && fallbackCount != 0 && outputCount > baselineTotal;
        }
        if (mixedCuts == 0) { return RHITestResult::fail("All instances fell back together"); }
        return RHITestResult::pass("24 multi-instance capacity cases; mixed cuts, insufficient roots, exact limits, cheaper detail and uint saturation");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamLODCapacityTest);

#undef STREAM_LOD_REQUIRE

} // namespace
} // namespace metallic::tests
