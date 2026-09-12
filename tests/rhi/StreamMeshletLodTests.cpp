#include "RhiTest.h"
#include "Runtime/Render/MeshletLod.h"
#include "Runtime/Render/MeshletStreamRuntime.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/SlangCompiler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace metallic::tests {
namespace {

using namespace render;

struct StreamLodFixture {
    std::vector<MeshletLodGroupRecord> groups;
    std::vector<MeshletLodGroupRange> ranges{{0, 2}, {2, 2}, {4, 1}, {5, 2}, {7, 1}, {8, 2}};
    std::vector<uint32_t> refined{UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
        UINT32_MAX, 0, 1, 0, 3, 4};
    // Every original surface atom must be covered exactly once by every cut.
    std::array<uint32_t, 10> coverage{1, 2, 4, 8, 16, 1, 12, 2, 13, 2};
    std::vector<uint8_t> drawable = std::vector<uint8_t>(6, 1);
    GPUSceneGpuInstanceRecord instance;
    MeshletLodView view;

    explicit StreamLodFixture(bool large = false, uint32_t leafCount = 256)
    {
        for (uint32_t group = 0; group < ranges.size(); ++group) {
            groups.push_back({.sphere = {0, 0, 0, 1},
                .error = group < 3 ? .5f : group < 5 ? 2.f : 4.f,
                .level = group < 3 ? 0u : group < 5 ? 1u : 2u,
                .flags = group == 2 || group == 5 ? kMeshletLodTerminalGroup : 0u});
        }
        instance.worldMatrix = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        instance.identity[3] = GPUSceneGpuInstanceVisible;
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
                        .level = level, .flags = count == 1 ? kMeshletLodTerminalGroup : 0u});
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

    StreamMeshletLodReference select(uint32_t manual = UINT32_MAX, uint32_t capacity = UINT32_MAX) const
    {
        return selectStreamMeshletLodReference(groups, ranges, refined, drawable, instance, view, manual, capacity);
    }

    bool coversExactlyOnce(const StreamMeshletLodReference& cut) const
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

class StreamMeshletLodReferenceTest final : public RhiTest {
public:
    StreamMeshletLodReferenceTest() { type = RhiTestType::Validation; name = "meshlet_lod_stream_reference_frontier"; }
    RhiTestResult run(RhiTestContext&) override
    {
        StreamLodFixture fixture;
        auto cut = fixture.select();
        if (!fixture.coversExactlyOnce(cut) || cut.selectedClusters != std::vector<uint32_t>{0, 1, 2, 3, 4}) {
            return RhiTestResult::fail("fully resident stream cut did not refine");
        }
        fixture.drawable[3] = 0;
        cut = fixture.select();
        if (!fixture.coversExactlyOnce(cut) || cut.selectedClusters != std::vector<uint32_t>{4, 7, 8} ||
            cut.requestedGroups != std::vector<uint32_t>{3}) {
            return RhiTestResult::fail("missing ancestor exposed a descendant with another resident parent");
        }
        for (uint32_t residency = 0; residency < 16; ++residency) {
            for (uint32_t group : {0u, 1u, 3u, 4u}) {
                const uint32_t bit = group < 2 ? group : group - 1;
                fixture.drawable[group] = (residency >> bit) & 1;
            }
            for (uint32_t manual : {UINT32_MAX, 0u, 1u, 2u, 31u}) {
                cut = fixture.select(manual);
                if (!fixture.coversExactlyOnce(cut)) {
                    return RhiTestResult::fail("residency/LOD combination overlaps or omits a surface atom");
                }
            }
        }
        std::fill(fixture.drawable.begin(), fixture.drawable.end(), uint8_t{1});
        cut = fixture.select(UINT32_MAX, 2);
        if (!fixture.coversExactlyOnce(cut) || !cut.capacityExceeded || !cut.capacityFallback ||
            cut.selectedClusters != std::vector<uint32_t>{4, 8, 9}) {
            return RhiTestResult::fail("capacity did not atomically fall back to the complete terminal cut");
        }
        cut = fixture.select(UINT32_MAX, 1);
        if (cut.valid || !cut.capacityExceeded || !cut.selectedClusters.empty()) {
            return RhiTestResult::fail("an undersized terminal capacity exposed a partial cut");
        }
        fixture.drawable[5] = 0;
        cut = fixture.select();
        if (cut.valid || !cut.selectedClusters.empty() || cut.requestedGroups != std::vector<uint32_t>{5}) {
            return RhiTestResult::fail("missing terminal page exposed orphan descendants");
        }
        fixture.drawable[5] = 1;
        fixture.drawable[3] = 0;
        const std::array<uint8_t, 6> available{1, 1, 1, 1, 1, 1};
        cut = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
            fixture.drawable, fixture.instance, fixture.view, UINT32_MAX, UINT32_MAX, available);
        if (!fixture.coversExactlyOnce(cut) || !cut.requestedGroups.empty() ||
            cut.selectedClusters != std::vector<uint32_t>{4, 7, 8}) {
            return RhiTestResult::fail("pending upload replaced the fallback or generated a duplicate request");
        }
        fixture.drawable[3] = 1;
        fixture.view.projection[3] = 100;
        if (fixture.select().selectedClusters != std::vector<uint32_t>{4, 8, 9}) {
            return RhiTestResult::fail("orthographic pixel error did not select cross-level terminal groups");
        }
        fixture.view.forward[3] = 0;
        fixture.view.eye[2] = .1f;
        if (fixture.select().selectedClusters != std::vector<uint32_t>{0, 1, 2, 3, 4}) {
            return RhiTestResult::fail("near-plane error did not refine");
        }
        fixture.instance.identity[3] = 0;
        cut = fixture.select();
        if (!cut.valid || !cut.selectedClusters.empty() || !cut.requestedGroups.empty()) {
            return RhiTestResult::fail("hidden instance produced geometry or page requests");
        }
        fixture.refined[8] = 5;
        if (fixture.select().valid) { return RhiTestResult::fail("cyclic stream topology accepted"); }
        return RhiTestResult::pass("80 residency/manual cuts preserve exact coverage; multi-parent gating, mixed-depth terminals, complete capacity fallback and near-plane projection");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLodReferenceTest);

// Keep transitions consecutive: State is initialized only once for each GPU
// algorithm, so pruning must clear yesterday's finer active bits and masks.
void configureStreamLodCase(StreamLodFixture& fixture, bool large, uint32_t test,
    uint32_t& manual, uint32_t& capacity, std::vector<uint8_t>& available)
{
    fixture = StreamLodFixture{large};
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
    case 11: case 18: fixture.view.projection[3] = 16.f; break;
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

bool sameStreamCut(const StreamMeshletLodReference& lhs, const StreamMeshletLodReference& rhs)
{
    return lhs.valid == rhs.valid && lhs.capacityExceeded == rhs.capacityExceeded &&
        lhs.capacityFallback == rhs.capacityFallback && lhs.selectedClusters == rhs.selectedClusters &&
        lhs.requestedGroups == rhs.requestedGroups && lhs.activeGroups == rhs.activeGroups;
}

class StreamMeshletLodBvhReferenceTest final : public RhiTest {
public:
    StreamMeshletLodBvhReferenceTest() { type = RhiTestType::Validation; name = "meshlet_lod_stream_bvh_reference"; }
    RhiTestResult run(RhiTestContext&) override
    {
        uint32_t comparisons = 0;
        for (bool large : {false, true}) {
            StreamLodFixture fixture{large};
            std::vector<MeshletLodBvhNode> nodes;
            std::string reason;
            if (!buildMeshletLodBvh(fixture.groups, nodes, reason) || nodes.empty()) {
                return RhiTestResult::fail("stream BVH build: " + reason);
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
                    return RhiTestResult::fail("BVH/reference cut mismatch: fixture " + std::to_string(large) +
                        ", case " + std::to_string(test) + ", " + accelerated.reason);
                }
                if (large && (test == 1 || test == 11 || test == 18 || test == 24) &&
                    (accelerated.testedGroups >= fixture.groups.size() / 8 || accelerated.visitedBvhNodes == 0 ||
                        accelerated.selectedClusters.size() != 2)) {
                    return RhiTestResult::fail("coarse BVH cut did not prune at least 7/8 of 511 group tests");
                }
                ++comparisons;
            }
            fixture = StreamLodFixture{large};
            const auto invalidRejected = [&](const std::vector<MeshletLodBvhNode>& malformed) {
                return !selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
                    fixture.drawable, fixture.instance, fixture.view, UINT32_MAX, UINT32_MAX, {}, malformed).valid;
            };
            auto malformed = nodes;
            malformed[0].escapeIndex = 0;
            if (!invalidRejected(malformed)) { return RhiTestResult::fail("BVH accepted non-advancing escape"); }
            malformed = nodes;
            malformed[0].maxError = 0;
            if (!invalidRejected(malformed)) { return RhiTestResult::fail("BVH accepted nonconservative error"); }
            const auto leaf = std::find_if(nodes.begin(), nodes.end(), [](const auto& node) { return node.groupCount != 0; });
            if (leaf == nodes.end()) { return RhiTestResult::fail("BVH contains no group leaves"); }
            malformed = nodes;
            malformed[static_cast<size_t>(leaf - nodes.begin())].groupOffset = static_cast<uint32_t>(fixture.groups.size());
            if (!invalidRejected(malformed)) { return RhiTestResult::fail("BVH accepted out-of-range leaf groups"); }
        }
        StreamLodFixture spread{true, 64};
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
        std::vector<MeshletLodBvhNode> spreadNodes;
        std::string reason;
        if (!buildMeshletLodBvh(spread.groups, spreadNodes, reason)) {
            return RhiTestResult::fail("spatial stream BVH build: " + reason);
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
                        return RhiTestResult::fail("spatial BVH bound changed the cut: view " + std::to_string(viewCase) +
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
        StreamLodFixture cancellation{true, 64};
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
        std::vector<MeshletLodBvhNode> cancellationNodes;
        if (!buildMeshletLodBvh(cancellation.groups, cancellationNodes, reason)) {
            return RhiTestResult::fail("cancellation BVH build: " + reason);
        }
        const auto bruteCancellation = selectStreamMeshletLodReference(cancellation.groups, cancellation.ranges,
            cancellation.refined, cancellation.drawable, cancellation.instance, cancellation.view);
        const auto bvhCancellation = selectStreamMeshletLodReference(cancellation.groups, cancellation.ranges,
            cancellation.refined, cancellation.drawable, cancellation.instance, cancellation.view,
            UINT32_MAX, UINT32_MAX, {}, cancellationNodes);
        if (!bruteCancellation.valid || bruteCancellation.activeGroups[0] == 0 ||
            !sameStreamCut(bruteCancellation, bvhCancellation)) {
            return RhiTestResult::fail("BVH bound lost near-plane group after large-coordinate float cancellation");
        }
        ++comparisons;
        return RhiTestResult::pass(std::to_string(comparisons) +
            " BVH/brute cuts agree across residency, shared parents, projection, transforms, translated/spread spheres, capacity and manual levels; coarse 511-group cuts prune >87% of group tests");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLodBvhReferenceTest);

#define STREAM_LOD_REQUIRE(expr) do { const auto checked = (expr); if (!checked) { return RhiTestResult::fail(std::string(#expr) + ": " + toString(checked)); } } while (false)

class StreamMeshletLodGpuTest final : public RhiTest {
public:
    StreamMeshletLodGpuTest() { type = RhiTestType::Rendering; name = "meshlet_lod_stream_gpu_matches_reference"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        const auto small = runFixture(context, false);
        if (!small.passed) { return small; }
        return runFixture(context, true);
    }
private:
    RhiTestResult runFixture(RhiTestContext& context, bool large)
    {
        StreamLodFixture fixture{large};
        const uint32_t kGroupCount = static_cast<uint32_t>(fixture.groups.size());
        const uint32_t kRequestCapacity = kGroupCount + 2;
        const uint32_t kRequestWords = 16 + kRequestCapacity * 2;
        std::vector<MeshletStreamGpuGroup> groups(kGroupCount);
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
        MeshletStreamGpuPrimitive primitive;
        primitive.groupCount = kGroupCount;
        primitive.pageCount = kGroupCount;
        primitive.lodLevelCount = large ? 9 : 3;
        std::vector<MeshletLodBvhNode> nodes;
        std::string reason;
        if (!buildMeshletLodBvh(fixture.groups, nodes, reason)) {
            return RhiTestResult::fail("stream GPU fixture BVH: " + reason);
        }
        primitive.lodBvhOffset = static_cast<uint32_t>(topology.size());
        const size_t bvhWords = nodes.size() * sizeof(MeshletLodBvhNode) / sizeof(uint32_t);
        topology.resize(topology.size() + bvhWords);
        std::memcpy(topology.data() + primitive.lodBvhOffset, nodes.data(), bvhWords * sizeof(uint32_t));
        enum BufferIndex { Instance, Primitive, Groups, Params, Topology, State, PageTable,
            Requests, ActiveGroups, Header, Arguments, Dummy, BufferCount };
        const uint32_t strides[] = {sizeof(MeshletStreamGpuInstance), sizeof(primitive), sizeof(MeshletStreamGpuGroup),
            sizeof(MeshletStreamGpuParams), 4, 4, sizeof(StreamPageTableEntry), 4,
            sizeof(MeshletStreamGpuActiveGroup), sizeof(MeshletStreamGpuActiveHeader), sizeof(MeshletStreamGpuDrawIndirect), 4};
        const uint64_t logicalStateBytes = (4 + kGroupCount * 3 + 4) * sizeof(uint32_t);
        // Cross the maximum X dispatch dimension and leave a partial last row.
        const uint64_t allocatedStateBytes = large ? (65535ull * 64 + 67) * sizeof(uint32_t) : logicalStateBytes;
        const uint64_t sizes[] = {strides[Instance], sizeof(primitive), groups.size() * sizeof(groups[0]), strides[Params],
            topology.size() * 4, allocatedStateBytes, kGroupCount * sizeof(StreamPageTableEntry),
            kRequestWords * 4, kGroupCount * sizeof(MeshletStreamGpuActiveGroup), strides[Header], strides[Arguments], 64};
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream LOD frontier regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (hasError(created, Error::Unsupported)) { return RhiTestResult::skip("Requires bindless compute"); }
        STREAM_LOD_REQUIRE(created);
        std::unique_ptr<BindlessHeap> heap;
        STREAM_LOD_REQUIRE(device->createBindlessHeap({.maxBuffers = BufferCount}, heap));
        std::array<std::unique_ptr<Buffer>, BufferCount> buffers;
        std::array<BindlessHandle, BufferCount> handles;
        for (uint32_t index = 0; index < BufferCount; ++index) {
            STREAM_LOD_REQUIRE(device->createBuffer({.size = sizes[index], .structureStride = strides[index],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload}, buffers[index]));
            STREAM_LOD_REQUIRE(heap->allocateBuffer(handles[index]));
            STREAM_LOD_REQUIRE(heap->writeStorageBuffer(handles[index], *buffers[index]));
        }
        const auto upload = [&](BufferIndex index, const void* source, size_t size) {
            void* mapped = buffers[index]->map();
            if (mapped == nullptr) { return false; }
            std::memset(mapped, index == State && large ? 0xa5 : 0, sizes[index]);
            if (source != nullptr) { std::memcpy(mapped, source, size); }
            buffers[index]->flush();
            buffers[index]->unmap();
            return true;
        };
        for (uint32_t index = 0; index < BufferCount; ++index) {
            if (!upload(static_cast<BufferIndex>(index), nullptr, 0)) { return RhiTestResult::fail("stream test buffer map"); }
        }
        if (!upload(Primitive, &primitive, sizeof(primitive)) || !upload(Groups, groups.data(), sizes[Groups]) ||
            !upload(Topology, topology.data(), sizes[Topology])) { return RhiTestResult::fail("stream topology map"); }
        ShaderCompileResult compiled;
        const auto compile = compileSlangShaderToSpirv({.moduleName = kMeshletStreamShaderModuleName,
            .entryPointName = kMeshletStreamActiveBuildEntryPoint, .searchPath = kMeshletStreamShaderSearchPath}, compiled);
        if (!compile) { return RhiTestResult::fail("stream frontier shader compile: " + compiled.diagnostics); }
        std::unique_ptr<ShaderModule> shader;
        STREAM_LOD_REQUIRE(device->createShaderModule({.code = compiled.spirv.data(), .byteSize = compiled.spirv.size() * 4}, shader));
        std::unique_ptr<ComputePipeline> pipeline;
        STREAM_LOD_REQUIRE(device->createComputePipeline({.computeShader = shader.get(), .computeEntryPoint = "main",
            .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(MeshletStreamUserPush)}, pipeline));
        MeshletStreamUserPush push;
        push.instanceBuffer = handles[Instance].index;
        push.primitiveBuffer = handles[Primitive].index;
        push.groupBuffer = handles[Groups].index;
        push.paramsBuffer = handles[Params].index;
        push.pageTableBuffer = handles[PageTable].index;
        push.requestBuffer = handles[Requests].index;
        push.activeGroupBuffer = handles[ActiveGroups].index;
        push.activeHeaderBuffer = handles[Header].index;
        push.drawIndirectBuffer = handles[Arguments].index;
        push.pageBuffer = handles[Dummy].index;
        push.nodeBuffer = handles[Dummy].index;
        push.lodLevelBuffer = handles[Dummy].index;
        push.traversalHeaderBuffer = handles[Dummy].index;
        push.traversalWorkBuffer = handles[Dummy].index;
        std::unique_ptr<Buffer> readback;
        constexpr uint32_t outputs[] = {Header, ActiveGroups, Requests, Arguments, State};
        const uint64_t outputSizes[] = {sizes[Header], sizes[ActiveGroups], sizes[Requests], sizes[Arguments], logicalStateBytes};
        std::array<uint64_t, std::size(outputs)> offsets{};
        uint64_t outputBytes = 0;
        for (uint32_t index = 0; index < std::size(outputs); ++index) { offsets[index] = outputBytes; outputBytes += outputSizes[index]; }
        const uint64_t tailOffset = outputBytes;
        if (large) { outputBytes += 3 * sizeof(uint32_t); }
        STREAM_LOD_REQUIRE(device->createBuffer({.size = outputBytes, .usage = BufferUsageBits::TransferDestination,
            .memoryLocation = MemoryLocation::HostReadback}, readback));
        Queue* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        STREAM_LOD_REQUIRE(device->createCommandPool(*queue, pool));
        STREAM_LOD_REQUIRE(pool->createCommandBuffer(commands));
        STREAM_LOD_REQUIRE(device->createFence(false, fence));
        const uint32_t caseCount = large ? 36u : 37u;
        for (uint32_t frame = 0; frame < caseCount * 2; ++frame) {
            const uint32_t test = frame % caseCount;
            const bool useBvh = frame >= caseCount;
            primitive.lodBvhNodeCount = useBvh ? static_cast<uint32_t>(nodes.size()) : 0;
            uint32_t manual, capacity;
            std::vector<uint8_t> available;
            configureStreamLodCase(fixture, large, test, manual, capacity, available);
            const auto expected = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
                fixture.drawable, fixture.instance, fixture.view, manual, capacity, available);
            const auto expectedState = selectStreamMeshletLodReference(fixture.groups, fixture.ranges, fixture.refined,
                fixture.drawable, fixture.instance, fixture.view, manual, UINT32_MAX, available,
                useBvh ? std::span<const MeshletLodBvhNode>(nodes) : std::span<const MeshletLodBvhNode>{});
            const std::string caseLabel = std::string(large ? "511 groups, " : "shared parents, ") +
                (useBvh ? "BVH, case " : "linear, case ") + std::to_string(test);
            MeshletStreamGpuInstance instance;
            instance.visible = fixture.instance.identity[3] != 0;
            instance.gpuSceneInstanceIndex = 17;
            std::copy_n(fixture.instance.worldMatrix.data(), 4, instance.world0);
            std::copy_n(fixture.instance.worldMatrix.data() + 4, 4, instance.world1);
            std::copy_n(fixture.instance.worldMatrix.data() + 8, 4, instance.world2);
            std::copy_n(fixture.instance.worldMatrix.data() + 12, 4, instance.world3);
            MeshletStreamGpuParams params;
            std::copy_n(fixture.view.eye.data(), 3, params.eye);
            for (uint32_t axis = 0; axis < 3; ++axis) { params.center[axis] = params.eye[axis] + fixture.view.forward[axis]; }
            params.upProjection[1] = 1;
            params.upProjection[3] = fixture.view.forward[3];
            params.viewport[2] = fixture.view.projection[0];
            params.viewport[3] = 2 * std::atan(fixture.view.projection[1]);
            params.clipOrtho[0] = fixture.view.eye[3];
            params.clipOrtho[2] = fixture.view.projection[2];
            params.lodPixelError = fixture.view.projection[3];
            params.lodTopologyBuffer = handles[Topology].index;
            params.lodStateBuffer = handles[State].index;
            params.lodInstanceOffsetsOffset = instanceOffsetsOffset;
            params.sceneInstanceCount = 1;
            params.scenePrimitiveCount = 1;
            params.sceneGroupCount = kGroupCount;
            params.scenePageCount = kGroupCount;
            params.maxActiveGroupClusters = 2;
            params.activeGroupCount = capacity;
            params.drawTaskCount = capacity * 4;
            params.frameIndex = frame + 1;
            params.selectedLodLevel = manual;
            params.enableGpuLodSelection = manual == UINT32_MAX;
            params.maxGpuPageRequests = kRequestCapacity;
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
            if (!upload(Instance, &instance, sizeof(instance)) || !upload(Params, &params, sizeof(params)) ||
                !upload(Primitive, &primitive, sizeof(primitive)) ||
                !upload(PageTable, pages.data(), sizes[PageTable]) || !upload(Requests, requests.data(), sizes[Requests])) {
                return RhiTestResult::fail("stream frame input map");
            }
            if (frame != 0) { STREAM_LOD_REQUIRE(fence->reset()); STREAM_LOD_REQUIRE(pool->reset()); }
            STREAM_LOD_REQUIRE(commands->begin());
            std::array<BufferBarrierDesc, BufferCount> barriers{};
            for (uint32_t index = 0; index < BufferCount; ++index) {
                barriers[index] = {.buffer = buffers[index].get(),
                    .before = frame == 0 ? ResourceState::Undefined : ResourceState::General, .after = ResourceState::General};
            }
            commands->barrier({.buffers = barriers.data(), .bufferCount = BufferCount});
            commands->bindBindlessHeap(*heap);
            commands->bindComputePipeline(*pipeline);
            if (frame == caseCount) {
                // The compatibility linear path predates sparse state. Run
                // the production initializer once when changing algorithms.
                push.activeBuildPhase = 8;
                commands->pushBindlessData(&push, sizeof(push));
                const uint32_t initGroups = static_cast<uint32_t>((sizes[State] / 4 + 63) / 64);
                commands->dispatch(std::min(initGroups, 65535u), (initGroups + 65534) / 65535, 1);
                commands->barrier({.buffers = barriers.data(), .bufferCount = BufferCount});
            }
            for (uint32_t phase : {0u, 5u, 6u, 7u, 2u}) {
                push.activeBuildPhase = phase;
                commands->pushBindlessData(&push, sizeof(push));
                commands->dispatch(1, 1, 1);
                for (auto& barrier : barriers) { barrier.before = ResourceState::General; }
                commands->barrier({.buffers = barriers.data(), .bufferCount = BufferCount});
            }
            for (uint32_t index = 0; index < std::size(outputs); ++index) {
                BufferBarrierDesc barrier{.buffer = buffers[outputs[index]].get(),
                    .before = ResourceState::General, .after = ResourceState::TransferSource};
                commands->barrier({.buffers = &barrier, .bufferCount = 1});
                commands->copyBuffer({.source = buffers[outputs[index]].get(), .destination = readback.get(),
                    .destinationOffset = offsets[index], .size = outputSizes[index]});
                if (large && outputs[index] == State) {
                    commands->copyBuffer({.source = buffers[State].get(), .destination = readback.get(),
                        .sourceOffset = sizes[State] - 3 * sizeof(uint32_t), .destinationOffset = tailOffset,
                        .size = 3 * sizeof(uint32_t)});
                }
                std::swap(barrier.before, barrier.after);
                commands->barrier({.buffers = &barrier, .bufferCount = 1});
            }
            STREAM_LOD_REQUIRE(commands->end());
            CommandBuffer* submitted[] = {commands.get()};
            STREAM_LOD_REQUIRE(queue->submit({.commandBuffers = submitted, .commandBufferCount = 1, .signalFence = fence.get()}));
            STREAM_LOD_REQUIRE(fence->wait());
            readback->invalidate();
            const auto* mapped = static_cast<const uint8_t*>(readback->map());
            if (mapped == nullptr) { return RhiTestResult::fail("stream result map"); }
            MeshletStreamGpuActiveHeader header;
            std::memcpy(&header, mapped, sizeof(header));
            std::vector<MeshletStreamGpuActiveGroup> active(kGroupCount);
            std::memcpy(active.data(), mapped + offsets[1], sizes[ActiveGroups]);
            std::memcpy(requests.data(), mapped + offsets[2], sizes[Requests]);
            MeshletStreamGpuDrawIndirect arguments;
            std::memcpy(&arguments, mapped + offsets[3], sizeof(arguments));
            std::vector<uint32_t> state(logicalStateBytes / sizeof(uint32_t));
            std::memcpy(state.data(), mapped + offsets[4], logicalStateBytes);
            std::array<uint32_t, 3> tail{};
            if (large) { std::memcpy(tail.data(), mapped + tailOffset, sizeof(tail)); }
            readback->unmap();
            if (large && !std::all_of(tail.begin(), tail.end(),
                [&](uint32_t word) { return word == (useBvh ? 0u : 0xa5a5a5a5u); })) {
                return RhiTestResult::fail("2D LOD state initialization did not clear the partial last row: " + caseLabel);
            }
            if (header.activeGroupCount > capacity || header.activeGroupCount > kGroupCount) {
                return RhiTestResult::fail("stream frontier overflow: " + caseLabel);
            }
            std::vector<uint32_t> actual;
            for (uint32_t index = 0; index < header.activeGroupCount; ++index) {
                const auto& group = active[index];
                if (group.pageIndex >= kGroupCount || group.gpuSceneInstanceIndex != 17 || group.instanceIndex != 0) {
                    return RhiTestResult::fail("stream selected group identity");
                }
                for (uint32_t cluster = 0; cluster < group.clusterCount; ++cluster) {
                    if ((group.clusterSelectionMask & (1u << cluster)) != 0) {
                        actual.push_back(fixture.ranges[group.pageIndex].clusterOffset + cluster);
                    }
                }
            }
            if (requests[2] > kRequestCapacity || requests[5] != 0 || requests[7] != 0) {
                return RhiTestResult::fail("stream frontier request overflow or invalid page");
            }
            std::vector<uint32_t> requested(requests.begin() + 16, requests.begin() + 16 + requests[2]);
            std::sort(requested.begin(), requested.end());
            if (actual != expected.selectedClusters || requested != expected.requestedGroups ||
                (expected.valid && bool(header.padding0) != expected.capacityFallback) ||
                arguments.groupCountX != header.activeGroupCount * 4 || arguments.groupCountY != 1 || arguments.groupCountZ != 1) {
                return RhiTestResult::fail("CPU/GPU stream cut, requests, capacity fallback or indirect arguments mismatch in case " +
                    caseLabel + ": expected " + std::to_string(expected.selectedClusters.size()) +
                    " clusters, actual " + std::to_string(actual.size()));
            }
            const uint32_t stats = 4 + kGroupCount * 2;
            if (useBvh && (state[stats] > kGroupCount || state[stats + 1] != expectedState.visitedBvhNodes ||
                state[stats + 2] != expectedState.testedGroups)) {
                return RhiTestResult::fail("GPU/CPU BVH traversal statistics mismatch: " + caseLabel +
                    ", visited " + std::to_string(state[stats + 1]) + "/" + std::to_string(expectedState.visitedBvhNodes) +
                    ", tested " + std::to_string(state[stats + 2]) + "/" + std::to_string(expectedState.testedGroups));
            }
            if (useBvh && large && (test == 1 || test == 11 || test == 18 || test == 24) &&
                (state[stats + 2] >= kGroupCount / 8 || actual.size() != 2)) {
                return RhiTestResult::fail("GPU BVH failed to prune coarse branches: " + caseLabel);
            }
            if (useBvh && expectedState.valid) {
                uint32_t activeCount = 0;
                std::vector<uint8_t> seen(kGroupCount);
                for (uint32_t index = 0; index < state[stats]; ++index) {
                    const uint32_t group = state[stats + 4 + index];
                    if (group >= kGroupCount || seen[group] != 0 || expectedState.activeGroups[group] == 0) {
                        return RhiTestResult::fail("GPU sparse frontier has duplicate/stale groups: " + caseLabel);
                    }
                    seen[group] = 1;
                }
                for (uint32_t group = 0; group < kGroupCount; ++group) {
                    activeCount += expectedState.activeGroups[group];
                    if (state[4 + group * 2] != expectedState.activeGroups[group] ||
                        (expectedState.activeGroups[group] == 0 && state[4 + group * 2 + 1] != 0)) {
                        return RhiTestResult::fail("GPU retained stale group state across frames: " + caseLabel);
                    }
                }
                if (state[stats] != activeCount) {
                    return RhiTestResult::fail("GPU sparse frontier count differs from active bits: " + caseLabel);
                }
            }
        }
        return RhiTestResult::pass("146 linear/BVH GPU-reference cuts, including 511-group hierarchy pruning, sparse state reuse, fine/coarse and hidden/shown transitions, missing/reloaded pages and capacity fallback");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamMeshletLodGpuTest);

#undef STREAM_LOD_REQUIRE

} // namespace
} // namespace metallic::tests
