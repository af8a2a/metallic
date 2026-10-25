#pragma once

#include <array>
#include <optional>
#include <span>

namespace metallic::render {

using StreamCLASTemplateBounds = std::array<float, 6>;

// Bounds order is minXYZ, maxXYZ; the matrix is column-major, as in the
// streamed GPU instance. allLodBounds must contain every source vertex that
// can be selected from this primitive, independently of current residency.
// Models the current FP32 tessellation domain and world/local evaluation.
// nullopt means this input has no finite envelope established by the helper;
// the caller must use an unrestricted template without changing geometry.
// A final GPU containment check remains required before using a bounded pool.
std::optional<StreamCLASTemplateBounds> computeStreamCLASTemplateBounds(
    std::span<const float, 6> allLodBounds,
    std::span<const float, 16> worldMatrix,
    float displacementMagnitude,
    float displacementCenter) noexcept;

} // namespace metallic::render
