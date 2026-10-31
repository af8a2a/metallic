#include "Runtime/Render/Shadow/VirtualShadowMapLOD.h"
#include "Runtime/Render/Subsystem/GPUScene.h"

#include <algorithm>
#include <cmath>
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

} // namespace metallic::render
