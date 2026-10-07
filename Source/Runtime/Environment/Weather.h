#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "Runtime/Environment/Atmosphere.h"

namespace metallic::environment {

struct WeatherState {
    bool cloudEnabled = false;
    float cloudCoverage = 0.0f;
    float cloudDensity = 1.0f; // Dimensionless cloud extinction multiplier.
    float cloudBaseAltitudeKm = 1.5f;
    float cloudTopAltitudeKm = 4.0f;
    float cloudExtinctionPerKm = 8.0f;
    float aerosolDensity = 1.0f; // Multiplier of authored Mie density.
    float humidity = 0.0f; // Relative humidity, used by the growth approximation.
    float precipitation = 0.0f; // Normalized precipitation state.
    float windSpeed = 0.0f; // Metres per real second, independent of clock pause.
    float2 windDirection{1.0f, 0.0f}; // Horizontal ENU X/Z; normalized on evaluation.
    uint32_t noiseSeed = 0;

    bool operator==(const WeatherState& other) const
    {
        return cloudEnabled == other.cloudEnabled && cloudCoverage == other.cloudCoverage &&
            cloudDensity == other.cloudDensity && cloudBaseAltitudeKm == other.cloudBaseAltitudeKm &&
            cloudTopAltitudeKm == other.cloudTopAltitudeKm && cloudExtinctionPerKm == other.cloudExtinctionPerKm &&
            aerosolDensity == other.aerosolDensity && humidity == other.humidity &&
            precipitation == other.precipitation && windSpeed == other.windSpeed &&
            windDirection.x == other.windDirection.x && windDirection.y == other.windDirection.y &&
            noiseSeed == other.noiseSeed;
    }
};

inline bool validWeatherState(const WeatherState& weather)
{
    const auto range = [](float value, float low, float high) {
        return std::isfinite(value) && value >= low && value <= high;
    };
    return range(weather.cloudCoverage, 0.0f, 1.0f) && range(weather.cloudDensity, 0.0f, 100.0f) &&
        range(weather.cloudBaseAltitudeKm, 0.0f, 1e5f) && range(weather.cloudTopAltitudeKm, 0.0f, 1e5f) &&
        weather.cloudTopAltitudeKm > weather.cloudBaseAltitudeKm &&
        range(weather.cloudExtinctionPerKm, 0.0f, 1000.0f) && range(weather.aerosolDensity, 0.0f, 100.0f) &&
        range(weather.humidity, 0.0f, 1.0f) && range(weather.precipitation, 0.0f, 1.0f) &&
        range(weather.windSpeed, 0.0f, 1000.0f) && std::isfinite(weather.windDirection.x) &&
        std::isfinite(weather.windDirection.y) &&
        double(weather.windDirection.x) * weather.windDirection.x +
            double(weather.windDirection.y) * weather.windDirection.y >= 1e-12;
}

inline float2 normalizedWeatherWindDirection(const WeatherState& weather)
{
    const double length = std::hypot(double(weather.windDirection.x), double(weather.windDirection.y));
    return length >= 1e-6 && std::isfinite(length)
        ? float2(float(weather.windDirection.x / length), float(weather.windDirection.y / length))
        : float2(1.0f, 0.0f);
}

inline AtmosphereState evaluateWeatherAtmosphere(const AtmosphereState& authored, const WeatherState& weather)
{
    AtmosphereState result = authored;
    // Bounded, phenomenological hygroscopic growth; this is not a forecast or
    // a measured aerosol composition. Default dry weather preserves B exactly.
    float scale = weather.aerosolDensity * (1.0f + 3.0f * weather.humidity * weather.humidity);
    // Preserve spectral ratios and single-scattering albedo at the validated
    // coefficient ceiling rather than independently clipping scattering.
    const float maximumExtinction = std::max({authored.mieExtinction.x, authored.mieExtinction.y,
        authored.mieExtinction.z});
    if (maximumExtinction > 0.0f) {
        scale = std::min(scale, 1000.0f / maximumExtinction);
    }
    result.mieScattering = authored.mieScattering * scale;
    result.mieExtinction = authored.mieExtinction * scale;
    return result;
}

} // namespace metallic::environment
