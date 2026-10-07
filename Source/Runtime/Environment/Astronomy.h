#pragma once

#include <cmath>
#include <cstdint>
#include <cwchar>
#include <optional>

#ifdef _MSC_VER
// MathLib emulates several intrinsic names with macros. Parse the native
// declarations first so independent header consumers do not rewrite them.
#include <intrin.h>
#endif

#include "ml.h"

namespace metallic::environment {

enum class AstronomyMode : uint32_t {
    Manual = 0,
    SimplifiedDayCycle = 1,
    Astronomical = 2,
};

// UTC is sufficient for this visual ephemeris: leap seconds, nutation,
// aberration and TT-UTC corrections are deliberately outside its precision.
inline constexpr double kMinimumJulianDateUTC = 2415020.5; // 1900-01-01.
inline constexpr double kMaximumJulianDateUTC = 2488069.5; // 2100-01-01.

struct EnvironmentTimeState {
    double julianDateUTC = 2451545.0; // 2000-01-01 12:00 UTC (J2000).
    double timeScale = 1.0; // Simulated seconds per elapsed real second; signed.
    bool paused = true;

    bool operator==(const EnvironmentTimeState&) const = default;
};

struct AstronomyState {
    AstronomyMode mode = AstronomyMode::Manual;
    double latitudeDegrees = 0.0; // North positive, geodetic latitude.
    double longitudeDegrees = 0.0; // East positive.
    float moonAlbedo = 0.12f; // Lambert diffuse reflectance, spectrally neutral.

    bool operator==(const AstronomyState&) const = default;
};

struct AstronomyEvaluation {
    bool automatic = false; // False retains the authored uniform manual disk.
    double julianDateUTC = 2451545.0;
    // Local ENU world frame: X east, Y up, Z north. Emitted-light directions.
    float3 sunDirection{0.0f, -1.0f, 0.0f};
    float3 moonDirection{0.0f, -1.0f, 0.0f};
    double sunDistanceAU = 1.0;
    double moonDistanceKm = 384400.0; // Observer-to-Moon, includes parallax.
    float sunAngularRadius = 0.00465047f;
    float moonAngularRadius = 0.004519f;
    float moonPhaseAngleRadians = 0.0f; // 0 full, pi new.
    float moonIlluminatedFraction = 1.0f;
    float moonLambertPhase = 1.0f; // Integrated flux relative to a full disk.
    float3 moonToSunDirection{0.0f, 1.0f, 0.0f}; // Unit ENU vector from Moon.
};

inline bool validEnvironmentTimeState(const EnvironmentTimeState& time)
{
    return std::isfinite(time.julianDateUTC) && time.julianDateUTC >= kMinimumJulianDateUTC &&
        time.julianDateUTC <= kMaximumJulianDateUTC && std::isfinite(time.timeScale) &&
        std::abs(time.timeScale) <= 1e9;
}

inline bool validAstronomyState(const AstronomyState& astronomy)
{
    return (astronomy.mode == AstronomyMode::Manual || astronomy.mode == AstronomyMode::SimplifiedDayCycle ||
        astronomy.mode == AstronomyMode::Astronomical) && std::isfinite(astronomy.latitudeDegrees) &&
        std::abs(astronomy.latitudeDegrees) <= 90.0 && std::isfinite(astronomy.longitudeDegrees) &&
        std::abs(astronomy.longitudeDegrees) <= 180.0 && std::isfinite(astronomy.moonAlbedo) &&
        astronomy.moonAlbedo >= 0.0f && astronomy.moonAlbedo <= 1.0f;
}

// Pure evaluation; neither playback nor weather writes back to authoring state.
double effectiveJulianDateUTC(const EnvironmentTimeState& time, double elapsedSeconds = 0.0);
AstronomyEvaluation evaluateAstronomy(const AstronomyState& astronomy, double julianDateUTC);
float lambertMoonPhase(float phaseAngleRadians);
float3 reflectedMoonIrradiance(float3 solarIrradiance, const AstronomyEvaluation& evaluation, float moonAlbedo);

struct WorldEnvironment;
struct EnvironmentSnapshot;
EnvironmentSnapshot evaluateWorldEnvironment(const WorldEnvironment& authored, double elapsedSeconds = 0.0,
    std::optional<double> runtimeJulianDateUTC = std::nullopt);

} // namespace metallic::environment
