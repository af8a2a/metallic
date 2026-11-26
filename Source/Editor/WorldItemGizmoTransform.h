#pragma once

#include "Editor/WorldItemBounds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>
#include <variant>

namespace metallic::editor {

enum class WorldItemGizmoOperation { Translate, Rotate, Scale };

struct WorldItemGizmoCapabilities {
    bool rotate = false;
    bool scale = false;
    bool scaleDepth = false;
    bool rotateRoll = true;
};

struct FogGizmoTransform {
    float3 position;
    std::array<float, 3> rotationDegrees;
    float radius;
};

struct PunctualLightGizmoTransform {
    float3 position;
    float3 direction;
    float iesRollDegrees;
    double range;
    bool spot;
    bool usesIES;
};

struct RectLightGizmoTransform {
    float3 position;
    float3 direction;
    float rollDegrees;
    float width;
    float height;
    float range;
};

using WorldItemGizmoTransform = std::variant<FogGizmoTransform, PunctualLightGizmoTransform, RectLightGizmoTransform>;

inline WorldItemGizmoTransform captureWorldItemGizmoTransform(const environment::LocalFogVolume& volume)
{
    const auto& p = volume.centerMeters;
    return FogGizmoTransform{{p[0], p[1], p[2]}, volume.rotationDegrees, volume.radiusMeters};
}

inline WorldItemGizmoTransform captureWorldItemGizmoTransform(const scene::PunctualLight& light)
{
    return PunctualLightGizmoTransform{light.position, light.direction, light.iesRollDegrees,
        light.properties.range, light.properties.type == "spot", bool(light.iesProfile)};
}

inline WorldItemGizmoTransform captureWorldItemGizmoTransform(const scene::RectLight& light)
{
    return RectLightGizmoTransform{light.position, light.direction, light.rollDegrees,
        light.width, light.height, light.range};
}

inline WorldItemGizmoCapabilities worldItemGizmoCapabilities(const WorldItemGizmoTransform& transform)
{
    return std::visit([](const auto& value) -> WorldItemGizmoCapabilities {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, PunctualLightGizmoTransform>) {
            // A punctual light with no attenuation cutoff has no finite range to scale.
            const bool finiteRange = value.range > 0.0;
            return {.rotate = value.spot || value.usesIES, .scale = finiteRange, .scaleDepth = finiteRange,
                .rotateRoll = value.usesIES};
        } else if constexpr (std::is_same_v<T, RectLightGizmoTransform>) {
            return {.rotate = true, .scale = true, .scaleDepth = value.range > 0.0f};
        } else {
            return {.rotate = true, .scale = true, .scaleDepth = true};
        }
    }, transform);
}

namespace world_item_gizmo_detail {

constexpr float kPoseEpsilon = 1e-6f;
constexpr double kToRadians = std::numbers::pi / 180.0;
constexpr double kToDegrees = 180.0 / std::numbers::pi;

inline float3 xyz(const float4& value)
{
    return {value.x, value.y, value.z};
}

inline bool close(const float3& a, const float3& b)
{
    return std::abs(a.x - b.x) <= kPoseEpsilon && std::abs(a.y - b.y) <= kPoseEpsilon &&
        std::abs(a.z - b.z) <= kPoseEpsilon;
}

inline bool sameAngle(float a, float b)
{
    return std::abs(std::remainder(double(a) - b, 360.0)) <= kPoseEpsilon;
}

inline bool sameSize(double a, double b)
{
    return std::abs(a - b) <= std::max(a, b) * kPoseEpsilon;
}

inline float4x4 fogOrientation(const std::array<float, 3>& degrees)
{
    constexpr float toRadians = std::numbers::pi_v<float> / 180.0f;
    float4x4 x, y, z;
    x.SetupByRotationX(degrees[0] * toRadians);
    y.SetupByRotationY(degrees[1] * toRadians);
    z.SetupByRotationZ(degrees[2] * toRadians);
    return z * y * x;
}

inline float4x4 lightOrientation(const float3& direction, float rollDegrees)
{
    const auto frame = world_item_bounds_detail::lightFrame(direction);
    const double radians = double(rollDegrees) * kToRadians;
    const float c = static_cast<float>(std::cos(radians));
    const float s = static_cast<float>(std::sin(radians));
    // Local -Z is the light's forward axis. The roll follows the LTC and IES frames.
    return {float4(frame.right * c - frame.up * s, 0.0f), float4(frame.up * c + frame.right * s, 0.0f),
        float4(-frame.forward, 0.0f), float4(0.0f, 0.0f, 0.0f, 1.0f)};
}

inline void editRotation(FogGizmoTransform& value, const std::array<float3, 3>& axes)
{
    const auto start = fogOrientation(value.rotationDegrees);
    if (close(xyz(start[0]), axes[0]) && close(xyz(start[1]), axes[1]) && close(xyz(start[2]), axes[2])) {
        return;
    }
    // Invert Rz * Ry * Rx. At a pitch singularity, use a zero Z rotation.
    const float cosPitch = std::hypot(axes[0].x, axes[0].y);
    const float x = cosPitch > kPoseEpsilon ? std::atan2(axes[1].z, axes[2].z)
        : std::atan2(-axes[2].y, axes[1].y);
    const float y = std::atan2(-axes[0].z, cosPitch);
    const float z = cosPitch > kPoseEpsilon ? std::atan2(axes[0].y, axes[0].x) : 0.0f;
    value.rotationDegrees = {float(x * kToDegrees), float(y * kToDegrees), float(z * kToDegrees)};
}

inline void editLightRotation(float3& direction, float* rollDegrees, const std::array<float3, 3>& axes)
{
    const auto forward = -axes[2];
    const auto start = world_item_bounds_detail::lightFrame(direction);
    const bool directionChanged = !close(start.forward, forward);
    if (directionChanged) { direction = forward; }
    if (rollDegrees) {
        const auto frame = directionChanged ? world_item_bounds_detail::lightFrame(direction) : start;
        const float roll = float(std::atan2(-dot(axes[0], frame.up), dot(axes[0], frame.right)) * kToDegrees);
        if (!sameAngle(*rollDegrees, roll)) { *rollDegrees = roll; }
    }
}

inline void editRotation(PunctualLightGizmoTransform& value, const std::array<float3, 3>& axes)
{
    editLightRotation(value.direction, value.usesIES ? &value.iesRollDegrees : nullptr, axes);
}

inline void editRotation(RectLightGizmoTransform& value, const std::array<float3, 3>& axes)
{
    editLightRotation(value.direction, &value.rollDegrees, axes);
}

template<typename T>
inline bool scaleDimension(T& dimension, float factor)
{
    const double scaled = double(dimension) * factor;
    if (!(scaled > 0.0 && scaled <= std::numeric_limits<T>::max())) { return false; }
    const auto edited = static_cast<T>(scaled);
    if (edited <= 0) { return false; }
    dimension = edited;
    return true;
}

inline float uniformScale(const std::array<float, 3>& factors)
{
    // One axis handle or the center handle can resize a spherical range.
    return *std::ranges::max_element(factors, {}, [](float factor) { return std::abs(factor - 1.0f); });
}

inline bool editScale(FogGizmoTransform& value, const std::array<float, 3>& factors)
{
    return scaleDimension(value.radius, uniformScale(factors));
}

inline bool editScale(PunctualLightGizmoTransform& value, const std::array<float, 3>& factors)
{
    return scaleDimension(value.range, uniformScale(factors));
}

inline bool editScale(RectLightGizmoTransform& value, const std::array<float, 3>& factors)
{
    return scaleDimension(value.width, factors[0]) && scaleDimension(value.height, factors[1]) &&
        (value.range == 0.0f || scaleDimension(value.range, factors[2]));
}

} // namespace world_item_gizmo_detail

inline float4x4 worldItemGizmoMatrix(const WorldItemGizmoTransform& transform)
{
    using namespace world_item_gizmo_detail;
    return std::visit([](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        float4x4 matrix = float4x4::Identity();
        if constexpr (std::is_same_v<T, FogGizmoTransform>) {
            matrix = fogOrientation(value.rotationDegrees);
        } else if constexpr (std::is_same_v<T, PunctualLightGizmoTransform>) {
            if (value.spot || value.usesIES) {
                matrix = lightOrientation(value.direction, value.usesIES ? value.iesRollDegrees : 0.0f);
            }
        } else {
            matrix = lightOrientation(value.direction, value.rollDegrees);
        }
        matrix[3] = float4(value.position, 1.0f);
        return matrix;
    }, transform);
}

// ImGuizmo outputs absolute scale factors for the drag and incremental rotations and translations.
inline std::optional<WorldItemGizmoTransform> editWorldItemGizmoTransform(WorldItemGizmoTransform transform,
    const float4x4& matrix, WorldItemGizmoOperation operation)
{
    using namespace world_item_gizmo_detail;
    if (operation == WorldItemGizmoOperation::Translate) {
        const auto position = xyz(matrix[3]);
        if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
            return std::nullopt;
        }
        std::visit([&](auto& value) { value.position = position; }, transform);
        return transform;
    }

    const auto capabilities = worldItemGizmoCapabilities(transform);
    if (operation == WorldItemGizmoOperation::Rotate ? !capabilities.rotate : !capabilities.scale) {
        return std::nullopt;
    }
    std::array axes{xyz(matrix[0]), xyz(matrix[1]), xyz(matrix[2])};
    std::array<float, 3> factors;
    for (size_t axis = 0; axis < axes.size(); ++axis) {
        const auto& v = axes[axis];
        const float factor = std::hypot(v.x, v.y, v.z);
        if (!std::isfinite(factor) || factor <= 0.0f) { return std::nullopt; }
        factors[axis] = factor;
    }
    if (operation == WorldItemGizmoOperation::Scale) {
        if (!std::visit([&](auto& value) { return editScale(value, factors); }, transform)) { return std::nullopt; }
    } else {
        for (size_t axis = 0; axis < axes.size(); ++axis) { axes[axis] = axes[axis] / factors[axis]; }
        std::visit([&](auto& value) { editRotation(value, axes); }, transform);
    }
    return transform;
}

inline bool worldItemGizmoTransformsEqual(const WorldItemGizmoTransform& lhs, const WorldItemGizmoTransform& rhs,
    WorldItemGizmoOperation operation)
{
    using namespace world_item_gizmo_detail;
    return std::visit([&](const auto& a) {
        using T = std::decay_t<decltype(a)>;
        const auto* b = std::get_if<T>(&rhs);
        if (b == nullptr) { return false; }
        if (operation == WorldItemGizmoOperation::Translate) { return close(a.position, b->position); }
        const bool rotation = operation == WorldItemGizmoOperation::Rotate;
        if constexpr (std::is_same_v<T, FogGizmoTransform>) {
            const auto& x = a.rotationDegrees;
            const auto& y = b->rotationDegrees;
            return rotation ? sameAngle(x[0], y[0]) && sameAngle(x[1], y[1]) && sameAngle(x[2], y[2])
                : sameSize(a.radius, b->radius);
        } else if constexpr (std::is_same_v<T, PunctualLightGizmoTransform>) {
            return rotation ? close(a.direction, b->direction) && sameAngle(a.iesRollDegrees, b->iesRollDegrees)
                : sameSize(a.range, b->range);
        } else {
            return rotation ? close(a.direction, b->direction) && sameAngle(a.rollDegrees, b->rollDegrees)
                : sameSize(a.width, b->width) && sameSize(a.height, b->height) && sameSize(a.range, b->range);
        }
    }, lhs);
}

inline void applyWorldItemGizmoTransform(environment::LocalFogVolume& volume, const WorldItemGizmoTransform& transform,
    WorldItemGizmoOperation operation)
{
    const auto& value = std::get<FogGizmoTransform>(transform);
    switch (operation) {
    case WorldItemGizmoOperation::Translate:
        volume.centerMeters = {value.position.x, value.position.y, value.position.z};
        break;
    case WorldItemGizmoOperation::Rotate: volume.rotationDegrees = value.rotationDegrees; break;
    case WorldItemGizmoOperation::Scale: volume.radiusMeters = value.radius; break;
    }
}

inline void applyWorldItemGizmoTransform(scene::PunctualLight& light, const WorldItemGizmoTransform& transform,
    WorldItemGizmoOperation operation)
{
    const auto& value = std::get<PunctualLightGizmoTransform>(transform);
    switch (operation) {
    case WorldItemGizmoOperation::Translate: light.position = value.position; break;
    case WorldItemGizmoOperation::Rotate:
        light.direction = value.direction;
        if (value.usesIES && light.iesProfile) { light.iesRollDegrees = value.iesRollDegrees; }
        break;
    case WorldItemGizmoOperation::Scale: light.properties.range = value.range; break;
    }
}

inline void applyWorldItemGizmoTransform(scene::RectLight& light, const WorldItemGizmoTransform& transform,
    WorldItemGizmoOperation operation)
{
    const auto& value = std::get<RectLightGizmoTransform>(transform);
    switch (operation) {
    case WorldItemGizmoOperation::Translate: light.position = value.position; break;
    case WorldItemGizmoOperation::Rotate:
        light.direction = value.direction;
        light.rollDegrees = value.rollDegrees;
        break;
    case WorldItemGizmoOperation::Scale:
        light.width = value.width;
        light.height = value.height;
        light.range = value.range;
        break;
    }
}

} // namespace metallic::editor
