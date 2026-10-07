#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>

#include "Runtime/Environment/Astronomy.h"
#include "Runtime/Environment/Atmosphere.h"
#include "Runtime/Environment/Weather.h"

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
    float angularRadius = 0.00465f; // Radians; physical PT disk, realtime directional approximation.
    bool enabled = false;
    // Spectral irradiance at the top of the atmosphere, sampled at 680, 550
    // and 440 nm, in W/m^2/nm. Used by the physical provider; HDRI keeps lux.
    float3 topOfAtmosphereIrradiance{1.474f, 1.8504f, 1.91198f};

    bool operator==(const CelestialLight& other) const
    {
        return direction.x == other.direction.x && direction.y == other.direction.y &&
            direction.z == other.direction.z && color.x == other.color.x &&
            color.y == other.color.y && color.z == other.color.z &&
            illuminance == other.illuminance && angularRadius == other.angularRadius &&
            enabled == other.enabled &&
            topOfAtmosphereIrradiance.x == other.topOfAtmosphereIrradiance.x &&
            topOfAtmosphereIrradiance.y == other.topOfAtmosphereIrradiance.y &&
            topOfAtmosphereIrradiance.z == other.topOfAtmosphereIrradiance.z;
    }
};

inline bool validCelestialLight(const CelestialLight& light)
{
    const auto& d = light.direction;
    const auto& c = light.color;
    const auto& toa = light.topOfAtmosphereIrradiance;
    return std::isfinite(d.x) && std::isfinite(d.y) && std::isfinite(d.z) &&
        double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z >= 1e-12 &&
        std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z) &&
        c.x >= 0.0f && c.y >= 0.0f && c.z >= 0.0f &&
        std::isfinite(light.illuminance) && light.illuminance >= 0.0f && light.illuminance <= 1e12f &&
        std::isfinite(light.angularRadius) && light.angularRadius >= 0.0f &&
        light.angularRadius <= 1.57079632679489661923f &&
        std::isfinite(toa.x) && std::isfinite(toa.y) && std::isfinite(toa.z) &&
        toa.x >= 0.0f && toa.y >= 0.0f && toa.z >= 0.0f &&
        toa.x <= 100.0f && toa.y <= 100.0f && toa.z <= 100.0f;
}

// Detached, renderer-facing state. A caller owns its copy; editing the world
// cannot change a snapshot already submitted to rendering.
struct EnvironmentSnapshot {
    std::array<CelestialLight, kCelestialLightCount> celestial;
    std::array<CelestialLight, kCelestialLightCount> authoredCelestial;
    EnvironmentSource source = EnvironmentSource::HDRI;
    AtmosphereState atmosphere;
    AtmosphereState authoredAtmosphere;
    EnvironmentTimeState time;
    AstronomyState astronomy;
    WeatherState weather;
    AstronomyEvaluation evaluatedAstronomy;
    double elapsedSeconds = 0.0;
    uint64_t celestialRevision = 1;
    uint64_t atmosphereRevision = 0;
    uint64_t weatherRevision = 0;
    uint64_t astronomyRevision = 0;
    uint64_t lightingRevision = 1;
};

struct WorldEnvironment {
    // A disabled default preserves scenes that previously had no distant light.
    CelestialLight sun;
    // Manual full-Moon-scale TOA preset. Automatic modes derive the evaluated
    // lunar spectrum from the Sun, lunar albedo and phase without changing it.
    CelestialLight moon{.topOfAtmosphereIrradiance = float3(2.948e-6f, 3.7008e-6f, 3.82396e-6f)};
    EnvironmentSource source = EnvironmentSource::HDRI;
    AtmosphereState atmosphere;
    EnvironmentTimeState time;
    AstronomyState astronomy;
    WeatherState weather;

    bool operator==(const WorldEnvironment&) const = default;

    EnvironmentSnapshot snapshot(uint64_t celestialRevision = 1, uint64_t lightingRevision = 1,
        uint64_t atmosphereRevision = 0, uint64_t weatherRevision = 0, uint64_t astronomyRevision = 0,
        double elapsedSeconds = 0.0, std::optional<double> runtimeJulianDateUTC = std::nullopt) const
    {
        auto result = evaluateWorldEnvironment(*this, elapsedSeconds, runtimeJulianDateUTC);
        result.celestialRevision = celestialRevision;
        result.atmosphereRevision = atmosphereRevision;
        result.weatherRevision = weatherRevision;
        result.astronomyRevision = astronomyRevision;
        result.lightingRevision = lightingRevision;
        return result;
    }
};

inline bool validWorldEnvironment(const WorldEnvironment& world)
{
    const bool activePhysicalClouds = world.source == EnvironmentSource::PhysicalAtmosphere &&
        world.weather.cloudEnabled && world.weather.cloudCoverage > 0.0f &&
        world.weather.cloudDensity > 0.0f && world.weather.cloudExtinctionPerKm > 0.0f;
    const double atmosphereHeightKm = double(world.atmosphere.topRadiusKm) - world.atmosphere.bottomRadiusKm;
    return validCelestialLight(world.sun) && validCelestialLight(world.moon) &&
        (world.source == EnvironmentSource::HDRI || world.source == EnvironmentSource::PhysicalAtmosphere) &&
        validAtmosphereState(world.atmosphere) && validEnvironmentTimeState(world.time) &&
        validAstronomyState(world.astronomy) && validWeatherState(world.weather) &&
        (!activePhysicalClouds || double(world.weather.cloudTopAltitudeKm) <= atmosphereHeightKm);
}

} // namespace metallic::environment
