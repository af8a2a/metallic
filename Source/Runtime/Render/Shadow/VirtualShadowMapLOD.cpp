#include "Runtime/Render/Shadow/VirtualShadowMapLOD.h"
#include "Runtime/Render/Subsystem/GPUScene.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace metallic::render {
namespace {

using Vector = std::array<double, 3>;

double dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vector spatial(const std::array<float, 4>& row)
{
    return {row[0], row[1], row[2]};
}

double worldScale(const GPUSceneGPUInstanceRecord& instance)
{
    const auto& m = instance.worldMatrix;
    double scaleSquared = 0;
    // Absolute Gram row sums bound the largest singular value, including
    // nonuniform scales composed with rotations and shear.
    for (uint32_t i = 0; i < 3; ++i) {
        double rowSum = 0;
        for (uint32_t j = 0; j < 3; ++j) {
            double gram = 0;
            for (uint32_t k = 0; k < 3; ++k) { gram += double(m[i * 4 + k]) * m[j * 4 + k]; }
            rowSum += std::abs(gram);
        }
        scaleSquared = std::max(scaleSquared, rowSum);
    }
    return std::sqrt(scaleSquared);
}

double operatorNorm(const Vector& x, const Vector& y)
{
    const double xx = dot(x, x), xy = dot(x, y), yy = dot(y, y);
    return std::sqrt(.5 * (xx + yy + std::hypot(xx - yy, 2 * xy)));
}

double projectedError(const Vector& center, double radius, double error,
    const VirtualShadowMapLODProjection& projection)
{
    const Vector x = spatial(projection.rowX), y = spatial(projection.rowY), w = spatial(projection.rowW);
    const double cx = dot(x, center) + projection.rowX[3];
    const double cy = dot(y, center) + projection.rowY[3];
    const double cw = dot(w, center) + projection.rowW[3];
    const double wLength = std::sqrt(dot(w, w));
    const double minW = cw - radius * wLength;
    if (wLength != 0) {
        if (cw + radius * wLength < projection.nearPlane) { return 0; }
        if (minW <= projection.nearPlane) { return std::numeric_limits<float>::max(); }
    }
    for (uint32_t axis = 0; axis < 2; ++axis) {
        const auto& row = axis == 0 ? x : y;
        const double coordinate = axis == 0 ? cx : cy;
        // Homogeneous side planes keep orthographic clipmaps and perspective
        // faces independent of the primary camera's visible cut.
        for (double sign : {-1., 1.}) {
            const Vector plane{w[0] + sign * row[0], w[1] + sign * row[1], w[2] + sign * row[2]};
            if (cw + sign * coordinate + radius * std::sqrt(dot(plane, plane)) < 0) { return 0; }
        }
    }
    Vector bx{}, by{};
    for (uint32_t i = 0; i < 3; ++i) {
        bx[i] = x[i] - cx / cw * w[i];
        by[i] = y[i] - cy / cw * w[i];
    }
    // B is the numerator Jacobian at the sphere center. Across the error-
    // expanded sphere, ||J|| <= ||B||/minW * (1 + radius*|w|/minW).
    // For orthographic projections this is the exact linear operator norm.
    const double jacobian = operatorNorm(bx, by) / minW * (1 + radius * wLength / minW);
    return error * .5 * projection.resolution * jacobian;
}

// A merged sphere cannot be passed directly to projectedError: its center
// changes B, so that metric is not monotone with sphere containment. Bound each
// term of every descendant's existing metric instead. The error-expanded node
// contains every error-expanded group sphere after the same affine scale bound.
double hierarchyPixelError(const MeshletLODBVHNode& node,
    const GPUSceneGPUInstanceRecord& instance, double scale,
    std::span<const VirtualShadowMapLODProjection> projections)
{
    if (node.maxError == 0) { return 0; }
    if (!std::isfinite(node.sphere[3])) { return std::numeric_limits<float>::max(); }
    const auto& m = instance.worldMatrix;
    Vector center{};
    double transformMagnitude = 0;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        center[axis] = double(m[axis]) * node.sphere[0] + double(m[4 + axis]) * node.sphere[1] +
            double(m[8 + axis]) * node.sphere[2] + m[12 + axis];
        double magnitude = std::abs(double(m[12 + axis]));
        for (uint32_t component = 0; component < 3; ++component) {
            magnitude += std::abs(double(m[component * 4 + axis])) *
                (std::abs(double(node.sphere[component])) + node.sphere[3]);
        }
        transformMagnitude = std::max(transformMagnitude, magnitude);
    }
    const double error = node.maxError * scale;
    // Include arithmetic uncertainty when large translations cancel affine
    // products. Otherwise side-plane rejection could precede the error margin.
    const double radius = (double(node.sphere[3]) + node.maxError) * scale +
        64 * std::numeric_limits<double>::epsilon() * transformMagnitude;
    double maximum = 0;
    for (const auto& projection : projections) {
        const auto x = spatial(projection.rowX), y = spatial(projection.rowY), w = spatial(projection.rowW);
        const double cx = dot(x, center) + projection.rowX[3];
        const double cy = dot(y, center) + projection.rowY[3];
        const double cw = dot(w, center) + projection.rowW[3];
        const double wLength = std::sqrt(dot(w, w));
        const double minW = cw - radius * wLength;
        if (wLength != 0) {
            if (cw + radius * wLength < projection.nearPlane) { continue; }
            // This precedes side culling, matching the group metric's near-plane
            // saturation even for an off-axis group which crosses the near plane.
            if (minW <= projection.nearPlane) { return std::numeric_limits<float>::max(); }
        }
        bool outside = false;
        for (uint32_t axis = 0; axis < 2; ++axis) {
            const auto& row = axis == 0 ? x : y;
            const double coordinate = axis == 0 ? cx : cy;
            for (double sign : {-1., 1.}) {
                const Vector plane{w[0] + sign * row[0], w[1] + sign * row[1], w[2] + sign * row[2]};
                outside |= cw + sign * coordinate + radius * std::sqrt(dot(plane, plane)) < 0;
            }
        }
        if (outside) { continue; }
        const double aNorm = operatorNorm(x, y);
        // For any descendant g, Wg >= minW, rg <= radius, eg <= error.
        // ||clipXY(center_g)|| <= ||clipXY(center)|| + radius*||A||.
        // Thus ||Bg|| <= ||A|| + maxClipXY*||w||/minW. Substitution in
        // projectedError bounds its full Jacobian expression, not just the
        // actual geometric derivative. Orthographic w=0 reduces to exact ||A||.
        const double maxClipXY = std::hypot(cx, cy) + radius * aNorm;
        const double bNorm = aNorm + maxClipXY * wLength / minW;
        const double bound = error * .5 * projection.resolution * bNorm / minW *
            (1 + radius * wLength / minW);
        maximum = std::max(maximum, bound);
    }
    return maximum;
}

} // namespace

float virtualShadowMapLodPixelError(const MeshletLODGroupRecord& group,
    const GPUSceneGPUInstanceRecord& instance,
    std::span<const VirtualShadowMapLODProjection> projections)
{
    if ((group.flags & kMeshletLODTerminalGroup) != 0 || group.error == 0) { return 0; }
    const auto& m = instance.worldMatrix;
    const double scale = worldScale(instance);
    const double error = group.error * scale;
    const double radius = (double(group.sphere[3]) + group.error) * scale;
    Vector center{};
    for (uint32_t i = 0; i < 3; ++i) {
        center[i] = double(m[i]) * group.sphere[0] + double(m[4 + i]) * group.sphere[1] +
            double(m[8 + i]) * group.sphere[2] + m[12 + i];
    }
    double maximum = 0;
    for (const auto& projection : projections) {
        maximum = std::max(maximum, projectedError(center, radius, error, projection));
    }
    return static_cast<float>(std::min(maximum, double(std::numeric_limits<float>::max())));
}

VirtualShadowMapLODRequirements buildVirtualShadowMapLodRequirements(
    std::span<const MeshletLODGroupRecord> groups,
    const GPUSceneGPUInstanceRecord& instance,
    std::span<const VirtualShadowMapLODProjection> projections, float pixelThreshold)
{
    VirtualShadowMapLODRequirements result;
    result.needsFine.reserve(groups.size());
    for (const auto& group : groups) {
        const float error = virtualShadowMapLodPixelError(group, instance, projections);
        result.maxProjectedError = std::max(result.maxProjectedError, error);
        result.needsFine.push_back((group.flags & kMeshletLODTerminalGroup) != 0 || error > pixelThreshold);
    }
    return result;
}

bool buildVirtualShadowMapLODHierarchy(std::span<const MeshletLODGroupRecord> groups,
    std::span<const MeshletLODGroupRange> ranges, std::span<const uint32_t> refinedGroups,
    VirtualShadowMapLODHierarchy& hierarchy, std::string& reason)
{
    if (groups.size() > UINT32_MAX || ranges.size() != groups.size() || refinedGroups.size() > UINT32_MAX) {
        reason = "shadow LOD metadata size exceeds or disagrees with the index range";
        return false;
    }
    // The existing flat selector owns the shared DAG validation contract. Run
    // it once at publication, with every group drawable, before caching links.
    GPUSceneGPUInstanceRecord instance;
    instance.identity[3] = GPUSceneGPUInstanceVisible;
    const std::vector<uint8_t> all(groups.size(), 1);
    const auto validated = selectStreamMeshletLodReference(groups, ranges, refinedGroups,
        all, instance, MeshletLODView{}, UINT32_MAX, UINT32_MAX, {}, {}, all);
    if (!validated.valid) { reason = validated.reason; return false; }
    VirtualShadowMapLODHierarchy result;
    result.groups.assign(groups.begin(), groups.end());
    result.ranges.assign(ranges.begin(), ranges.end());
    result.refinedGroups.assign(refinedGroups.begin(), refinedGroups.end());
    result.parents.resize(groups.size());
    result.clusterGroups.resize(refinedGroups.size());
    std::vector<MeshletLODGroupRecord> metricGroups;
    for (uint32_t owner = 0; owner < groups.size(); ++owner) {
        const auto& group = groups[owner];
        if (!std::isfinite(group.error) || group.error < 0 || group.sphere[3] < 0 ||
            !std::all_of(group.sphere.begin(), group.sphere.end(), [](float value) { return std::isfinite(value); })) {
            reason = "invalid shadow LOD group metric";
            return false;
        }
        if ((group.flags & kMeshletLODTerminalGroup) != 0) {
            result.terminalGroups.push_back(owner);
        } else {
            result.metricGroupIDs.push_back(owner);
            metricGroups.push_back(group);
        }
        const auto& range = ranges[owner];
        for (uint32_t cluster = range.clusterOffset; cluster < range.clusterOffset + range.clusterCount; ++cluster) {
            result.clusterGroups[cluster] = owner;
            const uint32_t child = refinedGroups[cluster];
            if (child == kMeshletLODInvalidGroup) { continue; }
            auto& parents = result.parents[child];
            if (parents.empty() || parents.back() != owner) { parents.push_back(owner); }
        }
    }
    if (!buildMeshletLodBvh(metricGroups, result.bvh, reason)) { return false; }
    hierarchy = std::move(result);
    reason.clear();
    return true;
}

VirtualShadowMapLODSelection selectVirtualShadowMapLOD(const VirtualShadowMapLODHierarchy& hierarchy,
    std::span<const uint8_t> drawableGroups, const GPUSceneGPUInstanceRecord& instance,
    std::span<const VirtualShadowMapLODProjection> projections, float pixelThreshold, bool hierarchical)
{
    VirtualShadowMapLODSelection result;
    const auto& groups = hierarchy.groups;
    if (drawableGroups.size() != groups.size()) {
        result.cut.valid = false;
        result.cut.reason = "shadow LOD drawable states disagree with cached topology";
        return result;
    }
    if (!hierarchical) {
        auto requirements = buildVirtualShadowMapLodRequirements(groups, instance, projections, pixelThreshold);
        result.metricGroups = uint32_t(hierarchy.metricGroupIDs.size());
        if (pixelThreshold == 0) { std::fill(requirements.needsFine.begin(), requirements.needsFine.end(), uint8_t(1)); }
        // Keep the original full closure and selector as the A/B oracle.
        for (uint32_t owner = 0; owner < hierarchy.ranges.size(); ++owner) {
            const auto& range = hierarchy.ranges[owner];
            for (uint32_t cluster = range.clusterOffset; cluster < range.clusterOffset + range.clusterCount; ++cluster) {
                const uint32_t child = hierarchy.refinedGroups[cluster];
                if (child != kMeshletLODInvalidGroup) {
                    requirements.needsFine[owner] |= requirements.needsFine[child];
                    ++result.topologyEdges;
                }
            }
        }
        result.cut = selectStreamMeshletLodReference(groups, hierarchy.ranges, hierarchy.refinedGroups,
            drawableGroups, instance, MeshletLODView{}, UINT32_MAX, UINT32_MAX, {}, {}, requirements.needsFine);
        result.topologyGroups = result.cut.testedGroups;
        for (uint32_t group = 0; group < groups.size(); ++group) {
            if (result.cut.activeGroups[group] != 0) { result.clusterTests += hierarchy.ranges[group].clusterCount; }
        }
        return result;
    }
    auto& cut = result.cut;
    cut.activeGroups.resize(groups.size(), 0);
    if ((instance.identity[3] & GPUSceneGPUInstanceVisible) == 0) { return result; }
    std::vector<uint8_t> wanted(groups.size(), 0);
    std::vector<uint32_t> demand;
    const auto add = [&](uint32_t group) {
        if (wanted[group] == 0) { wanted[group] = 1; demand.push_back(group); }
    };
    for (uint32_t group : hierarchy.terminalGroups) { add(group); }
    if (pixelThreshold == 0) {
        for (uint32_t group : hierarchy.metricGroupIDs) { add(group); }
    } else {
        const double scale = worldScale(instance);
        // The strict margin allows float rounding of the existing group metric
        // to cause extra work, never an omitted demand near the budget boundary.
        const double pruneThreshold = pixelThreshold - std::max(std::abs(pixelThreshold) * 1e-4f, 1e-5f);
        uint32_t index = 0;
        while (index < hierarchy.bvh.size()) {
            const auto& node = hierarchy.bvh[index];
            ++result.hierarchyNodes;
            if (hierarchyPixelError(node, instance, scale, projections) < pruneThreshold) {
                index = node.escapeIndex;
                continue;
            }
            for (uint32_t local = 0; local < node.groupCount; ++local) {
                const uint32_t group = hierarchy.metricGroupIDs[node.groupOffset + local];
                ++result.metricGroups;
                if (virtualShadowMapLodPixelError(groups[group], instance, projections) > pixelThreshold) { add(group); }
            }
            ++index;
        }
        result.prunedGroups = uint32_t(hierarchy.metricGroupIDs.size()) - result.metricGroups;
    }
    // Sparse closure follows immutable parent links. A child requested by one
    // view forces every owner, even if their own metric spheres miss that view.
    for (size_t cursor = 0; cursor < demand.size(); ++cursor) {
        for (uint32_t parent : hierarchy.parents[demand[cursor]]) { ++result.topologyEdges; add(parent); }
    }
    std::sort(demand.begin(), demand.end(), std::greater<uint32_t>{});
    for (uint32_t group : demand) {
        ++result.topologyGroups;
        if (drawableGroups[group] == 0) {
            cut.requestedGroups.push_back(group);
        } else if (std::all_of(hierarchy.parents[group].begin(), hierarchy.parents[group].end(),
            [&](uint32_t parent) { return cut.activeGroups[parent] != 0; })) {
            cut.activeGroups[group] = 1;
        }
    }
    cut.testedGroups = result.topologyGroups;
    cut.visitedBvhNodes = result.hierarchyNodes;
    std::sort(cut.requestedGroups.begin(), cut.requestedGroups.end());
    for (uint32_t group : hierarchy.terminalGroups) {
        if (drawableGroups[group] == 0) {
            cut.valid = false;
            cut.reason = "stream LOD terminal pages are not all drawable";
            std::fill(cut.activeGroups.begin(), cut.activeGroups.end(), uint8_t(0));
            return result;
        }
    }
    // Ascending group order reproduces the flat selector's stable cluster IDs.
    for (auto group = demand.rbegin(); group != demand.rend(); ++group) {
        if (cut.activeGroups[*group] == 0) { continue; }
        const auto& range = hierarchy.ranges[*group];
        result.clusterTests += range.clusterCount;
        for (uint32_t cluster = range.clusterOffset; cluster < range.clusterOffset + range.clusterCount; ++cluster) {
            const uint32_t child = hierarchy.refinedGroups[cluster];
            if (child == kMeshletLODInvalidGroup || cut.activeGroups[child] == 0) { cut.selectedClusters.push_back(cluster); }
        }
    }
    return result;
}

} // namespace metallic::render
