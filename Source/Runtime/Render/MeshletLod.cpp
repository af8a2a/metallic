#include "Runtime/Render/MeshletLod.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace metallic::render {

namespace {

bool validLodSphere(const std::array<float, 4>& sphere, bool allowInfiniteRadius)
{
    return std::isfinite(sphere[0]) && std::isfinite(sphere[1]) &&
        std::isfinite(sphere[2]) && sphere[3] >= 0 &&
        (std::isfinite(sphere[3]) || (allowInfiniteRadius && sphere[3] == INFINITY));
}

bool lodSphereContains(const std::array<float, 4>& outer, const std::array<float, 4>& inner)
{
    if (outer[3] == INFINITY) { return true; }
    const double distance = std::hypot(double(outer[0]) - inner[0],
        double(outer[1]) - inner[1], double(outer[2]) - inner[2]);
    return distance + inner[3] <= double(outer[3]);
}

bool validateMeshletLodBvh(std::span<const MeshletLodGroupRecord> groups,
    std::span<const MeshletLodBvhNode> nodes)
{
    if (groups.empty() || nodes.empty() || nodes.size() > UINT32_MAX ||
        nodes.front().escapeIndex != nodes.size()) { return false; }
    struct Interior {
        uint32_t index;
        uint32_t children;
    };
    std::vector<Interior> stack;
    size_t expectedGroupEnd = groups.size();
    // Validate arbitrary input iteratively; malformed chains cannot overflow a
    // recursive CPU stack. Escapes and leaf coverage enforce a single tree.
    for (uint32_t index = 0; index < nodes.size(); ++index) {
        while (!stack.empty() && nodes[stack.back().index].escapeIndex == index) {
            const auto& parent = stack.back();
            if (parent.children < 2 || nodes[parent.index].groupOffset != expectedGroupEnd) { return false; }
            stack.pop_back();
        }
        if (index != 0 && stack.empty()) { return false; }
        const auto& node = nodes[index];
        if (node.escapeIndex <= index || node.escapeIndex > nodes.size() ||
            !validLodSphere(node.sphere, true) || std::isnan(node.maxError) || node.maxError < 0) {
            return false;
        }
        if (!stack.empty()) {
            auto& entry = stack.back();
            const auto& parent = nodes[entry.index];
            if (++entry.children > 4 || node.escapeIndex > parent.escapeIndex ||
                node.maxError > parent.maxError || node.maxLevel > parent.maxLevel ||
                (node.flags & parent.flags) != node.flags || !lodSphereContains(parent.sphere, node.sphere)) {
                return false;
            }
        }
        if (node.groupCount == 0) {
            if (node.escapeIndex - index < 3) { return false; }
            stack.push_back({index, 0});
            continue;
        }
        if (node.groupCount > 4 || node.escapeIndex != index + 1 ||
            uint64_t(node.groupOffset) + node.groupCount != expectedGroupEnd) { return false; }
        expectedGroupEnd = node.groupOffset;
        for (uint32_t offset = 0; offset < node.groupCount; ++offset) {
            const auto& group = groups[node.groupOffset + offset];
            if (!validLodSphere(group.sphere, false) || !std::isfinite(group.error) || group.error < 0 ||
                node.maxError < group.error || node.maxLevel < group.level ||
                (node.flags & group.flags) != group.flags || !lodSphereContains(node.sphere, group.sphere)) {
                return false;
            }
        }
    }
    while (!stack.empty()) {
        const auto& entry = stack.back();
        if (nodes[entry.index].escapeIndex != nodes.size() || entry.children < 2 ||
            nodes[entry.index].groupOffset != expectedGroupEnd) { return false; }
        stack.pop_back();
    }
    return expectedGroupEnd == 0;
}

} // namespace

bool buildMeshletLodBvh(std::span<const MeshletLodGroupRecord> groups,
    std::vector<MeshletLodBvhNode>& nodes, std::string& reason)
{
    nodes.clear();
    reason.clear();
    if (groups.size() > UINT32_MAX) {
        reason = "too many meshlet LOD groups for BVH indices";
        return false;
    }
    for (const auto& group : groups) {
        if (!validLodSphere(group.sphere, false) || !std::isfinite(group.error) || group.error < 0) {
            reason = "invalid meshlet LOD BVH group metric";
            return false;
        }
    }
    const auto aggregate = [](uint32_t count, const auto& getRecord) {
        MeshletLodBvhNode result;
        std::array<double, 3> lower{INFINITY, INFINITY, INFINITY};
        std::array<double, 3> upper{-INFINITY, -INFINITY, -INFINITY};
        bool infiniteRadius = false;
        for (uint32_t index = 0; index < count; ++index) {
            const auto record = getRecord(index);
            result.maxError = std::max(result.maxError, record.maxError);
            result.maxLevel = std::max(result.maxLevel, record.maxLevel);
            result.flags |= record.flags;
            infiniteRadius = infiniteRadius || !std::isfinite(record.sphere[3]);
            for (uint32_t axis = 0; axis < 3; ++axis) {
                lower[axis] = std::min(lower[axis], double(record.sphere[axis]) - record.sphere[3]);
                upper[axis] = std::max(upper[axis], double(record.sphere[axis]) + record.sphere[3]);
            }
        }
        if (infiniteRadius) {
            result.sphere = getRecord(0).sphere;
            result.sphere[3] = INFINITY;
            return result;
        }
        for (uint32_t axis = 0; axis < 3; ++axis) {
            result.sphere[axis] = static_cast<float>((lower[axis] + upper[axis]) * 0.5);
        }
        double radius = 0;
        for (uint32_t index = 0; index < count; ++index) {
            const auto sphere = getRecord(index).sphere;
            radius = std::max(radius, std::hypot(double(result.sphere[0]) - sphere[0],
                double(result.sphere[1]) - sphere[1], double(result.sphere[2]) - sphere[2]) + sphere[3]);
        }
        // Recompute about the rounded center, then round the radius outward.
        // An unrepresentably large union is conservatively never pruned.
        result.sphere[3] = radius > std::numeric_limits<float>::max() ? INFINITY :
            (radius > 0 ? std::nextafter(static_cast<float>(radius), INFINITY) : 0);
        return result;
    };
    const auto build = [&](const auto& self, uint32_t offset, uint32_t count) -> uint32_t {
        const uint32_t index = static_cast<uint32_t>(nodes.size());
        nodes.emplace_back();
        MeshletLodBvhNode node;
        if (count <= 4) {
            node = aggregate(count, [&](uint32_t child) {
                const auto& group = groups[offset + child];
                return MeshletLodBvhNode{.sphere = group.sphere, .maxError = group.error,
                    .maxLevel = group.level, .flags = group.flags};
            });
            node.groupCount = count;
        } else {
            std::array<uint32_t, 4> children{};
            const uint32_t childCount = std::min(4u, (count - 1) / 4 + 1);
            uint32_t end = offset + count;
            for (uint32_t child = 0; child < childCount; ++child) {
                const uint32_t childSize = count / childCount + (child < count % childCount ? 1u : 0u);
                end -= childSize;
                children[child] = self(self, end, childSize);
            }
            node = aggregate(childCount, [&](uint32_t child) { return nodes[children[child]]; });
        }
        node.groupOffset = offset;
        node.escapeIndex = static_cast<uint32_t>(nodes.size());
        nodes[index] = node;
        return index;
    };
    if (!groups.empty()) { build(build, 0, static_cast<uint32_t>(groups.size())); }
    return true;
}

bool buildMeshletLodMetadata(const scene::RenderPrimitive& primitive,
    std::vector<MeshletLodGroupRecord>& groups, std::string& reason)
{
    groups.clear();
    reason.clear();
    const auto fail = [&](const char* message) {
        groups.clear();
        reason = message;
        return false;
    };
    if (primitive.meshletLodGroups.empty() || primitive.meshletLodClusters.empty()) {
        return fail("no cluster LOD hierarchy");
    }
    if (!primitive.meshletLodLevels.empty()) {
        size_t clusterEnd = 0, groupEnd = 0;
        for (uint32_t levelIndex = 0; levelIndex < primitive.meshletLodLevels.size(); ++levelIndex) {
            const auto& level = primitive.meshletLodLevels[levelIndex];
            if (level.clusterOffset != clusterEnd || level.groupOffset != groupEnd ||
                level.clusterCount > primitive.meshletLodClusters.size() - clusterEnd ||
                level.groupCount > primitive.meshletLodGroups.size() - groupEnd) {
                return fail("cluster LOD levels overlap or omit payload");
            }
            clusterEnd += level.clusterCount;
            groupEnd += level.groupCount;
            for (uint32_t g = level.groupOffset; g < groupEnd; ++g) {
                const auto& group = primitive.meshletLodGroups[g];
                if (group.lodLevel != levelIndex || group.clusterOffset < level.clusterOffset ||
                    uint64_t(group.clusterOffset) + group.clusterCount > clusterEnd) {
                    return fail("cluster LOD group is outside its level");
                }
            }
        }
        if (clusterEnd != primitive.meshletLodClusters.size() || groupEnd != primitive.meshletLodGroups.size()) {
            return fail("cluster LOD levels do not cover the hierarchy");
        }
    }
    std::vector<bool> referenced(primitive.meshletLodGroups.size(), false);
    size_t expectedCluster = 0;
    for (uint32_t index = 0; index < primitive.meshletLodGroups.size(); ++index) {
        const auto& group = primitive.meshletLodGroups[index];
        if (group.clusterOffset != expectedCluster || group.clusterCount == 0 ||
            group.clusterOffset > primitive.meshletLodClusters.size() ||
            group.clusterCount > primitive.meshletLodClusters.size() - group.clusterOffset ||
            !std::isfinite(group.maxQuadricError) || group.maxQuadricError < 0 ||
            !std::isfinite(group.boundingSphereRadius) || group.boundingSphereRadius < 0 ||
            !std::isfinite(group.boundingSphereCenter.x) ||
            !std::isfinite(group.boundingSphereCenter.y) || !std::isfinite(group.boundingSphereCenter.z)) {
            return fail("invalid cluster LOD group range or metric");
        }
        groups.push_back({.sphere = {group.boundingSphereCenter.x, group.boundingSphereCenter.y,
            group.boundingSphereCenter.z, group.boundingSphereRadius},
            .error = group.maxQuadricError, .level = group.lodLevel});
        expectedCluster += group.clusterCount;
        for (uint32_t child = 0; child < group.clusterCount; ++child) {
            const auto& cluster = primitive.meshletLodClusters[group.clusterOffset + child];
            if (cluster.lodGroupIndex != static_cast<int32_t>(index) || cluster.lodLevel != group.lodLevel ||
                cluster.vertexCount == 0 || cluster.vertexCount > 128 ||
                cluster.triangleCount == 0 || cluster.triangleCount > 128 ||
                cluster.vertexOffset > primitive.meshletLodVertices.size() ||
                cluster.vertexCount > primitive.meshletLodVertices.size() - cluster.vertexOffset ||
                cluster.triangleOffset > primitive.meshletLodTriangles.size() ||
                cluster.triangleCount * 3u > primitive.meshletLodTriangles.size() - cluster.triangleOffset) {
                return fail("invalid cluster LOD payload");
            }
            for (uint32_t v = 0; v < cluster.vertexCount; ++v) {
                if (primitive.meshletLodVertices[cluster.vertexOffset + v] >= primitive.positions.size()) {
                    return fail("invalid cluster LOD vertex");
                }
            }
            for (uint32_t t = 0; t < cluster.triangleCount * 3u; ++t) {
                if (primitive.meshletLodTriangles[cluster.triangleOffset + t] >= cluster.vertexCount) {
                    return fail("invalid cluster LOD triangle");
                }
            }
            if (cluster.refinedGroupIndex != scene::kInvalidSceneIndex) {
                const uint32_t refined = static_cast<uint32_t>(cluster.refinedGroupIndex);
                if (refined >= index || primitive.meshletLodGroups[refined].lodLevel >= group.lodLevel ||
                    primitive.meshletLodGroups[refined].maxQuadricError > group.maxQuadricError) {
                    return fail("invalid or non-monotonic cluster LOD refinement");
                }
                referenced[refined] = true;
            }
        }
    }
    if (expectedCluster != primitive.meshletLodClusters.size()) {
        return fail("cluster LOD groups do not cover the payload");
    }
    // Roots can terminate at different depths. Derive them from DAG edges,
    // including small assets whose importer did not use the FLT_MAX sentinel.
    for (size_t i = 0; i < groups.size(); ++i) {
        if (!referenced[i]) { groups[i].flags |= kMeshletLodTerminalGroup; }
    }
    return true;
}

namespace {

float meshletLodPixelErrorImpl(const MeshletLodGroupRecord& group,
    const GPUSceneGpuInstanceRecord& instance, const MeshletLodView& view, float worldBoundsPadding)
{
    const auto& m = instance.worldMatrix;
    float gram[3][3]{};
    float scaleSquared = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t k = 0; k < 3; ++k) { gram[i][j] += m[i * 4 + k] * m[j * 4 + k]; }
        }
        scaleSquared = std::max(scaleSquared, std::abs(gram[i][0]) + std::abs(gram[i][1]) + std::abs(gram[i][2]));
    }
    const float scale = std::sqrt(scaleSquared);
    const float error = group.error * scale;
    if (error <= 0) { return 0; }
    if (view.forward[3] != 0) { return error * view.projection[0] / std::max(view.projection[2], 1e-6f); }
    float center[3]{};
    float z = 0, distanceSquared = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        center[i] = m[i] * group.sphere[0] + m[4 + i] * group.sphere[1] +
            m[8 + i] * group.sphere[2] + m[12 + i] - view.eye[i];
        z += center[i] * view.forward[i];
        distanceSquared += center[i] * center[i];
    }
    const float radius = group.sphere[3] * scale + error + worldBoundsPadding;
    const float minZ = z - radius;
    if (minZ <= view.eye[3]) { return std::numeric_limits<float>::max(); }
    const float radial = std::sqrt(std::max(distanceSquared - z * z, 0.0f)) + radius;
    const float slope = radial / minZ;
    const float focal = view.projection[0] / std::max(2 * view.projection[1], 1e-6f);
    return error * focal / minZ * std::sqrt(1 + slope * slope);
}

float meshletLodBvhWorldBoundsPadding(const MeshletLodBvhNode& node,
    const GPUSceneGpuInstanceRecord& instance, const MeshletLodView& view)
{
    const auto& m = instance.worldMatrix;
    float magnitude = 0;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        float coordinateMagnitude = 0;
        for (uint32_t component = 0; component < 3; ++component) {
            coordinateMagnitude += std::abs(m[component * 4 + axis]) *
                (std::abs(node.sphere[component]) + node.sphere[3]);
        }
        coordinateMagnitude += std::abs(m[12 + axis]);
        coordinateMagnitude += std::abs(view.eye[axis]);
        magnitude += coordinateMagnitude;
    }
    // A transformed aggregate and its groups can round in opposite directions,
    // especially when a large object/world origin cancels the camera position.
    // Inflate only BVH bounds; the authored per-group selection stays exact.
    return magnitude * (32.0f * std::numeric_limits<float>::epsilon());
}

} // namespace

float meshletLodPixelError(const MeshletLodGroupRecord& group,
    const GPUSceneGpuInstanceRecord& instance, const MeshletLodView& view)
{
    return meshletLodPixelErrorImpl(group, instance, view, 0);
}

bool meshletLodNeedsFine(const MeshletLodGroupRecord& group,
    const GPUSceneGpuInstanceRecord& instance, const MeshletLodView& view, uint32_t manualLevel)
{
    return (group.flags & kMeshletLodTerminalGroup) != 0 ||
        (manualLevel != UINT32_MAX ? group.level >= manualLevel :
            meshletLodPixelError(group, instance, view) > view.projection[3]);
}

StreamMeshletLodReference selectStreamMeshletLodReference(
    std::span<const MeshletLodGroupRecord> groups,
    std::span<const MeshletLodGroupRange> ranges,
    std::span<const uint32_t> refinedGroups,
    std::span<const uint8_t> drawableGroups,
    const GPUSceneGpuInstanceRecord& instance, const MeshletLodView& view,
    uint32_t manualLevel, uint32_t capacity, std::span<const uint8_t> availableGroups,
    std::span<const MeshletLodBvhNode> bvh)
{
    StreamMeshletLodReference result;
    const auto fail = [&](const char* reason) {
        result.valid = false;
        result.reason = reason;
        result.selectedClusters.clear();
        std::fill(result.activeGroups.begin(), result.activeGroups.end(), uint8_t{0});
        return result;
    };
    if (groups.size() > UINT32_MAX || ranges.size() != groups.size() || drawableGroups.size() != groups.size() ||
        (!availableGroups.empty() && availableGroups.size() != groups.size())) {
        return fail("stream LOD metadata sizes disagree");
    }
    result.activeGroups.resize(groups.size(), 0);
    std::vector<std::vector<uint32_t>> parents(groups.size());
    size_t clusterEnd = 0;
    for (uint32_t owner = 0; owner < groups.size(); ++owner) {
        const auto& range = ranges[owner];
        if (range.clusterOffset != clusterEnd || range.clusterCount == 0 ||
            range.clusterCount > refinedGroups.size() - clusterEnd) {
            return fail("stream LOD group ranges overlap or omit clusters");
        }
        clusterEnd += range.clusterCount;
        for (uint32_t cluster = range.clusterOffset; cluster < clusterEnd; ++cluster) {
            const uint32_t refined = refinedGroups[cluster];
            if (refined == kMeshletLodInvalidGroup) { continue; }
            if (refined >= owner || groups[refined].level >= groups[owner].level ||
                groups[refined].error > groups[owner].error) {
                return fail("stream LOD refinement is not a monotonic DAG");
            }
            auto& owners = parents[refined];
            if (owners.empty() || owners.back() != owner) { owners.push_back(owner); }
        }
    }
    if (clusterEnd != refinedGroups.size()) { return fail("stream LOD group ranges omit clusters"); }
    bool rootsDrawable = true;
    for (uint32_t group = 0; group < groups.size(); ++group) {
        if (((groups[group].flags & kMeshletLodTerminalGroup) != 0) != parents[group].empty()) {
            return fail("stream LOD terminal flags disagree with topology");
        }
        rootsDrawable = rootsDrawable && (!parents[group].empty() || drawableGroups[group] != 0);
    }
    if (!bvh.empty() && !validateMeshletLodBvh(groups, bvh)) { return fail("invalid stream LOD BVH"); }
    if ((instance.identity[3] & GPUSceneGpuInstanceVisible) == 0) { return result; }

    const auto visitGroup = [&](uint32_t group) {
        ++result.testedGroups;
        if (!meshletLodNeedsFine(groups[group], instance, view, manualLevel) ||
            !std::all_of(parents[group].begin(), parents[group].end(),
                [&](uint32_t parent) { return result.activeGroups[parent] != 0; })) {
            return;
        }
        if (drawableGroups[group] == 0) {
            if (availableGroups.empty() || availableGroups[group] == 0) {
                result.requestedGroups.push_back(group);
            }
        } else {
            result.activeGroups[group] = 1;
        }
    };
    if (bvh.empty()) {
        for (size_t reverse = groups.size(); reverse != 0; --reverse) {
            visitGroup(static_cast<uint32_t>(reverse - 1));
        }
    } else {
        uint32_t index = 0;
        while (index < bvh.size()) {
            const auto& node = bvh[index];
            ++result.visitedBvhNodes;
            bool mayNeedFine = (node.flags & kMeshletLodTerminalGroup) != 0;
            if (!mayNeedFine) {
                if (manualLevel != UINT32_MAX) {
                    mayNeedFine = node.maxLevel >= manualLevel;
                } else {
                    const MeshletLodGroupRecord bound{.sphere = node.sphere, .error = node.maxError};
                    const float upperError = meshletLodPixelErrorImpl(bound, instance, view,
                        meshletLodBvhWorldBoundsPadding(node, instance, view));
                    const float target = view.projection[3];
                    // The same margin is used by the GPU. Float rounding near
                    // the cut boundary must only cause extra traversal.
                    mayNeedFine = !(upperError < target - std::max(std::abs(target) * 1e-4f, 1e-5f));
                }
            }
            if (!mayNeedFine) {
                index = node.escapeIndex;
                continue;
            }
            for (uint32_t reverse = node.groupCount; reverse != 0; --reverse) {
                visitGroup(node.groupOffset + reverse - 1);
            }
            ++index;
        }
    }
    std::sort(result.requestedGroups.begin(), result.requestedGroups.end());
    if (!rootsDrawable) { return fail("stream LOD terminal pages are not all drawable"); }
    uint32_t selectedGroupCount = 0;
    for (uint32_t owner = 0; owner < groups.size(); ++owner) {
        if (result.activeGroups[owner] == 0) { continue; }
        const auto& range = ranges[owner];
        const size_t before = result.selectedClusters.size();
        for (uint32_t cluster = range.clusterOffset; cluster < range.clusterOffset + range.clusterCount; ++cluster) {
            const uint32_t refined = refinedGroups[cluster];
            if (refined == kMeshletLodInvalidGroup || result.activeGroups[refined] == 0) {
                result.selectedClusters.push_back(cluster);
            }
        }
        selectedGroupCount += result.selectedClusters.size() != before;
    }
    if (selectedGroupCount > capacity) {
        result.capacityExceeded = true;
        result.selectedClusters.clear();
        std::fill(result.activeGroups.begin(), result.activeGroups.end(), uint8_t{0});
        uint32_t terminalGroupCount = 0;
        for (uint32_t group = 0; group < groups.size(); ++group) {
            if (!parents[group].empty()) { continue; }
            ++terminalGroupCount;
            result.activeGroups[group] = 1;
            const auto& range = ranges[group];
            for (uint32_t cluster = range.clusterOffset; cluster < range.clusterOffset + range.clusterCount; ++cluster) {
                result.selectedClusters.push_back(cluster);
            }
        }
        if (terminalGroupCount > capacity) {
            return fail("stream LOD capacity cannot hold the complete terminal cut");
        }
        result.capacityFallback = true;
    }
    return result;
}

std::vector<MeshletLodSelection> selectMeshletLodReference(
    std::span<const MeshletLodGroupRecord> groups,
    std::span<const GPUSceneGpuMeshletRecord> clusters,
    std::span<const GPUSceneGpuInstanceRecord> instances,
    std::span<const VisibleClusterRecord> candidates, const MeshletLodView& view,
    uint32_t recordOffset, uint32_t manualLevel)
{
    std::vector<MeshletLodSelection> result;
    for (uint32_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        if (candidate.clusterIndex >= clusters.size() || candidate.instanceIndex >= instances.size()) { continue; }
        const auto& instance = instances[candidate.instanceIndex];
        if ((instance.identity[3] & GPUSceneGpuInstanceVisible) == 0) { continue; }
        const auto& cluster = clusters[candidate.clusterIndex];
        const auto needsFine = [&](uint32_t groupIndex) {
            return meshletLodNeedsFine(groups[groupIndex], instance, view, manualLevel);
        };
        const uint32_t owner = cluster.lod[1], refined = cluster.lod[2];
        if (owner != kMeshletLodInvalidGroup &&
            (owner >= groups.size() || !needsFine(owner) ||
                (refined != kMeshletLodInvalidGroup && (refined >= groups.size() || needsFine(refined))))) { continue; }
        result.push_back({candidate.instanceIndex, candidate.clusterIndex, recordOffset + i, candidate.dataIndex});
    }
    return result;
}

} // namespace metallic::render
