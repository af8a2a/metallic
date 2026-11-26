#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>

#include "Runtime/Scene/SceneLighting.h"
#include "Runtime/Environment/VolumetricFog.h"

namespace metallic {

struct WorldItemBoundsEdge {
    float3 start;
    float3 end;
};

namespace world_item_bounds_detail {

constexpr float kPi = std::numbers::pi_v<float>;
constexpr int kArcSegments = 32;

struct LightFrame {
    float3 forward;
    float3 right;
    float3 up;
};

inline LightFrame lightFrame(const float3& direction)
{
    // Match the frame used by buildRectLightRecords for the LTC winding.
    const double length = std::sqrt(double(direction.x) * direction.x +
        double(direction.y) * direction.y + double(direction.z) * direction.z);
    const float3 forward{float(direction.x / length), float(direction.y / length), float(direction.z / length)};
    const float3 referenceUp = std::abs(forward.y) < 0.999f ? float3(0, 1, 0) : float3(0, 0, -1);
    const float3 right = normalize(cross(forward, referenceUp));
    return {forward, right, cross(right, forward)};
}

inline void appendArc(std::vector<WorldItemBoundsEdge>& edges, const float3& center,
    const float3& axisX, const float3& axisY, float startAngle, float endAngle)
{
    const auto pointAt = [&](float angle) {
        return center + axisX * std::cos(angle) + axisY * std::sin(angle);
    };
    auto start = pointAt(startAngle);
    for (int i = 1; i <= kArcSegments; ++i) {
        const auto end = pointAt(std::lerp(startAngle, endAngle, float(i) / float(kArcSegments)));
        edges.push_back({start, end});
        start = end;
    }
}

inline void appendSphere(std::vector<WorldItemBoundsEdge>& edges, const float3& center, float radius)
{
    const float3 x(radius, 0.0f, 0.0f);
    const float3 y(0.0f, radius, 0.0f);
    const float3 z(0.0f, 0.0f, radius);
    appendArc(edges, center, x, y, 0.0f, 2.0f * kPi);
    appendArc(edges, center, x, z, 0.0f, 2.0f * kPi);
    appendArc(edges, center, y, z, 0.0f, 2.0f * kPi);
}

inline void appendDirectionMarker(std::vector<WorldItemBoundsEdge>& edges, const float3& center,
    const float3& forward, const float3& right, float length)
{
    const auto tip = center + forward * length;
    const auto base = center + forward * (length * 0.7f);
    const auto halfWidth = right * (length * 0.12f);
    edges.push_back({center, tip});
    edges.push_back({base - halfWidth, tip});
    edges.push_back({base + halfWidth, tip});
}

} // namespace world_item_bounds_detail

inline void appendWorldItemBounds(std::vector<WorldItemBoundsEdge>& edges, const environment::LocalFogVolume& volume)
{
    const auto& center = volume.centerMeters;
    world_item_bounds_detail::appendSphere(edges, float3(center[0], center[1], center[2]), volume.radiusMeters);
}

inline void appendWorldItemBounds(std::vector<WorldItemBoundsEdge>& edges, const scene::PunctualLight& light)
{
    using namespace world_item_bounds_detail;
    const double authoredRange = light.properties.range;
    const float range = authoredRange <= std::numeric_limits<float>::max() ? float(authoredRange) : 0.0f;
    if (light.properties.type == "point") {
        if (range > 0.0f) {
            appendSphere(edges, light.position, range);
        } else {
            // An unbounded light only has a placement marker.
            for (const auto& offset : {float3(0.15f, 0.0f, 0.0f), float3(0.0f, 0.15f, 0.0f), float3(0.0f, 0.0f, 0.15f)}) {
                edges.push_back({light.position - offset, light.position + offset});
            }
        }
        return;
    }

    const auto frame = lightFrame(light.direction);
    if (range == 0.0f) {
        appendDirectionMarker(edges, light.position, frame.forward, frame.right, 0.5f);
        return;
    }
    // Match the intersection of the outer spot cone and radial attenuation cutoff.
    const double angle = light.properties.outerConeAngle;
    const auto rimCenter = light.position + frame.forward * (range * float(std::cos(angle)));
    const float rimRadius = range * float(std::sin(angle));
    const auto rimRight = frame.right * rimRadius;
    const auto rimUp = frame.up * rimRadius;
    appendArc(edges, rimCenter, rimRight, rimUp, 0.0f, 2.0f * kPi);
    for (const auto& rim : {rimRight, rimUp}) {
        edges.push_back({light.position, rimCenter + rim});
        edges.push_back({light.position, rimCenter - rim});
    }
    const float opening = float(angle);
    const auto axialRange = frame.forward * range;
    appendArc(edges, light.position, axialRange, frame.right * range, -opening, opening);
    appendArc(edges, light.position, axialRange, frame.up * range, -opening, opening);
}

inline void appendWorldItemBounds(std::vector<WorldItemBoundsEdge>& edges, const scene::RectLight& light)
{
    using namespace world_item_bounds_detail;
    const auto frame = lightFrame(light.direction);
    const double roll = double(light.rollDegrees) * 0.017453292519943295;
    const float c = static_cast<float>(std::cos(roll));
    const float s = static_cast<float>(std::sin(roll));
    const auto right = frame.right * c - frame.up * s;
    const auto up = frame.up * c + frame.right * s;
    const auto halfWidth = right * (light.width * 0.5f);
    const auto halfHeight = up * (light.height * 0.5f);
    const std::array corners{
        light.position - halfWidth - halfHeight, light.position + halfWidth - halfHeight,
        light.position + halfWidth + halfHeight, light.position - halfWidth + halfHeight,
    };
    for (size_t i = 0; i < corners.size(); ++i) {
        edges.push_back({corners[i], corners[(i + 1) % corners.size()]});
    }
    if (light.range > 0.0f) {
        // RectLightLTC limits attenuation by radius from the center and the emitter-facing half-space.
        const auto forwardRange = frame.forward * light.range;
        appendArc(edges, light.position, right * light.range, up * light.range, 0.0f, 2.0f * kPi);
        appendArc(edges, light.position, right * light.range, forwardRange, 0.0f, kPi);
        appendArc(edges, light.position, up * light.range, forwardRange, 0.0f, kPi);
    } else {
        appendDirectionMarker(edges, light.position, frame.forward, right, 0.5f);
    }
}

} // namespace metallic
