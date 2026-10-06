#pragma once

#include "ml.h"

#include <array>
#include <cmath>
#include <cstdint>

namespace metallic::environment {

enum class CelestialLightIndex : uint32_t {
    Sun = 0,
    Moon = 1,
};

inline constexpr uint32_t kCelestialLightCount = 2;

struct CelestialLight {
    // Emitted-light direction in world space, matching legacy directional lights.
    // Shading points toward the source with -normalize(direction).
    float3 direction{0.0f, -1.0f, 0.0f};
    // Linear Rec.709 multiplier; illuminance is incident lux perpendicular to
    // the emitted direction. Color conversion occurs at the renderer boundary.
    float3 color{1.0f, 1.0f, 1.0f};
    float illuminance = 0.0f;
    float angularRadius = 0.00465f; // Radians. Delta-light approximation for now.
    bool enabled = false;

    bool operator==(const CelestialLight& other) const
    {
        return direction.x == other.direction.x && direction.y == other.direction.y &&
            direction.z == other.direction.z && color.x == other.color.x &&
            color.y == other.color.y && color.z == other.color.z &&
            illuminance == other.illuminance && angularRadius == other.angularRadius &&
            enabled == other.enabled;
    }
};

inline bool validCelestialLight(const CelestialLight& light)
{
    const auto& d = light.direction;
    const auto& c = light.color;
    return std::isfinite(d.x) && std::isfinite(d.y) && std::isfinite(d.z) &&
        double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z >= 1e-12 &&
        std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z) &&
        c.x >= 0.0f && c.y >= 0.0f && c.z >= 0.0f &&
        std::isfinite(light.illuminance) && light.illuminance >= 0.0f && light.illuminance <= 1e12f &&
        std::isfinite(light.angularRadius) && light.angularRadius >= 0.0f &&
        light.angularRadius <= 1.57079632679489661923f;
}

// Detached, renderer-facing state. A caller owns its copy; editing the world
// cannot change a snapshot already submitted to rendering.
struct EnvironmentSnapshot {
    std::array<CelestialLight, kCelestialLightCount> celestial;
    uint64_t celestialRevision = 1;
    uint64_t atmosphereRevision = 0;
    uint64_t weatherRevision = 0;
    uint64_t lightingRevision = 1;
};

struct WorldEnvironment {
    // A disabled default preserves scenes that previously had no distant light.
    CelestialLight sun;
    CelestialLight moon;

    bool operator==(const WorldEnvironment&) const = default;

    EnvironmentSnapshot snapshot(uint64_t celestialRevision = 1, uint64_t lightingRevision = 1) const
    {
        return {.celestial = {sun, moon}, .celestialRevision = celestialRevision,
            .lightingRevision = lightingRevision};
    }
};

inline bool validWorldEnvironment(const WorldEnvironment& world)
{
    return validCelestialLight(world.sun) && validCelestialLight(world.moon);
}

} // namespace metallic::environment
