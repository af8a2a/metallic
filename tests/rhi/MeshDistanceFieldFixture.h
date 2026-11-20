#pragma once

#include "Runtime/Scene/MeshDistanceField.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace metallic::tests {

inline bool writeAnalyticBoxField(const std::filesystem::path& path, const scene::Scene& document, std::string& log,
    int32_t meshIndex = 0)
{
    using namespace scene;
    MeshDistanceFieldGeometry geometry;
    if (!getMeshDistanceFieldGeometry(document, meshIndex, geometry, log)) {
        return false;
    }
    MeshDistanceField field;
    field.sourceMeshIndex = meshIndex;
    field.sourceFingerprint = geometry.fingerprint;
    field.localBoundsMin = geometry.positions.front();
    field.localBoundsMax = geometry.positions.front();
    for (const auto& position : geometry.positions) {
        for (size_t axis = 0; axis < 3; ++axis) {
            field.localBoundsMin[axis] = std::min(field.localBoundsMin[axis], position[axis]);
            field.localBoundsMax[axis] = std::max(field.localBoundsMax[axis], position[axis]);
        }
    }
    std::array<float, 3> normalizedExtent{};
    for (size_t axis = 0; axis < 3; ++axis) {
        normalizedExtent[axis] = (field.localBoundsMax[axis] - field.localBoundsMin[axis]) * 0.5f;
    }
    const float maxExtent = *std::max_element(normalizedExtent.begin(), normalizedExtent.end());
    for (float& extent : normalizedExtent) { extent /= maxExtent; }

    // Generate an independent, exact signed box field at the brick sample positions.
    // The runtime fixture covers serialization, upload, indirection and interpolation
    // on configurations without the optional Embree cook target.
    constexpr uint32_t kUnique = kMeshDistanceFieldUniqueBrickSize;
    constexpr uint32_t kBrick = kMeshDistanceFieldBrickSize;
    constexpr std::array<uint32_t, kMeshDistanceFieldMipCount> kDimensions{3, 2, 1};
    const float extentLength = std::sqrt(normalizedExtent[0] * normalizedExtent[0] +
        normalizedExtent[1] * normalizedExtent[1] + normalizedExtent[2] * normalizedExtent[2]);
    for (size_t level = 0; level < field.mips.size(); ++level) {
        auto& mip = field.mips[level];
        const uint32_t dimension = kDimensions[level];
        const float voxel = 2.0f / float(dimension * kUnique - 2);
        const float band = 4.0f * extentLength * voxel;
        mip.indirectionDimensions.fill(dimension);
        for (size_t axis = 0; axis < 3; ++axis) {
            mip.volumeToVirtualUVScale[axis] = 1.0f / (normalizedExtent[axis] * voxel * float(dimension * kUnique));
        }
        mip.volumeToVirtualUVAdd.fill(0.5f);
        mip.distanceFieldToVolumeScaleBias = {2.0f * band, -band};

        const auto toBox = [&](uint32_t sample, size_t axis) {
            const float extent = normalizedExtent[axis];
            return std::abs(-extent + (float(sample) - 1.0f) * (voxel * extent)) - extent;
        };
        for (uint32_t bz = 0; bz < dimension; ++bz) {
            for (uint32_t by = 0; by < dimension; ++by) {
                for (uint32_t bx = 0; bx < dimension; ++bx) {
                    mip.indirection.push_back(static_cast<uint32_t>(mip.indirection.size()));
                    for (uint32_t z = 0; z < kBrick; ++z) {
                        for (uint32_t y = 0; y < kBrick; ++y) {
                            for (uint32_t x = 0; x < kBrick; ++x) {
                                const float3 q(toBox(bx * kUnique + x, 0), toBox(by * kUnique + y, 1),
                                    toBox(bz * kUnique + z, 2));
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

} // namespace metallic::tests
