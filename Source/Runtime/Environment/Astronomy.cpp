#include <algorithm>
#include <cmath>
#include <numbers>

#include "Runtime/Environment/Astronomy.h"
#include "Runtime/Environment/WorldEnvironment.h"

namespace metallic::environment {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;
constexpr double kRadiansPerDegree = kPi / 180.0;
constexpr double kAstronomicalUnitKm = 149597870.7;
constexpr double kEarthEquatorialRadiusKm = 6378.137;
constexpr double kMoonRadiusKm = 1737.4;
constexpr double kSunRadiusKm = 695700.0;

struct Vector {
    double x, y, z;

    Vector operator-(const Vector& other) const
    {
        return {x - other.x, y - other.y, z - other.z};
    }
    Vector operator*(double scale) const
    {
        return {x * scale, y * scale, z * scale};
    }
};

double dot(Vector lhs, Vector rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

double length(Vector vector)
{
    return std::sqrt(dot(vector, vector));
}

Vector normalized(Vector vector)
{
    return vector * (1.0 / length(vector));
}

float3 asFloat(Vector vector)
{
    return float3(float(vector.x), float(vector.y), float(vector.z));
}

double radians(double degrees)
{
    return std::remainder(degrees, 360.0) * kRadiansPerDegree;
}

double sinDegrees(double degrees)
{
    return std::sin(radians(degrees));
}

double cosDegrees(double degrees)
{
    return std::cos(radians(degrees));
}

double eccentricAnomaly(double meanAnomalyDegrees, double eccentricity)
{
    const double mean = radians(meanAnomalyDegrees);
    double eccentric = mean + eccentricity * std::sin(mean) * (1.0 + eccentricity * std::cos(mean));
    for (uint32_t iteration = 0; iteration < 8; ++iteration) {
        const double correction = (eccentric - eccentricity * std::sin(eccentric) - mean) /
            (1.0 - eccentricity * std::cos(eccentric));
        eccentric -= correction;
        if (std::abs(correction) < 1e-12) { break; }
    }
    return eccentric;
}

Vector equatorial(Vector ecliptic, double obliquity)
{
    return {ecliptic.x, ecliptic.y * std::cos(obliquity) - ecliptic.z * std::sin(obliquity),
        ecliptic.y * std::sin(obliquity) + ecliptic.z * std::cos(obliquity)};
}

Vector toENU(Vector equatorialVector, double latitude, double sidereal)
{
    const double meridian = equatorialVector.x * std::cos(sidereal) + equatorialVector.y * std::sin(sidereal);
    return {-equatorialVector.x * std::sin(sidereal) + equatorialVector.y * std::cos(sidereal),
        meridian * std::cos(latitude) + equatorialVector.z * std::sin(latitude),
        -meridian * std::sin(latitude) + equatorialVector.z * std::cos(latitude)};
}

Vector horizontalDirection(double declination, double hourAngle, double latitude)
{
    return {-std::cos(declination) * std::sin(hourAngle),
        std::sin(latitude) * std::sin(declination) + std::cos(latitude) * std::cos(declination) * std::cos(hourAngle),
        std::cos(latitude) * std::sin(declination) - std::sin(latitude) * std::cos(declination) * std::cos(hourAngle)};
}

void evaluateDirections(AstronomyEvaluation& result, Vector sunPosition, Vector moonPosition, Vector observer,
    double latitude, double sidereal)
{
    const Vector observerToSun = sunPosition - observer;
    const Vector observerToMoon = moonPosition - observer;
    const Vector moonToSun = normalized(sunPosition - moonPosition);
    const Vector moonToObserver = normalized(observer - moonPosition);
    result.sunDistanceAU = length(observerToSun) / kAstronomicalUnitKm;
    result.moonDistanceKm = length(observerToMoon);
    result.sunDirection = asFloat(toENU(normalized(observerToSun), latitude, sidereal) * -1.0);
    result.moonDirection = asFloat(toENU(normalized(observerToMoon), latitude, sidereal) * -1.0);
    result.moonToSunDirection = asFloat(toENU(moonToSun, latitude, sidereal));
    result.sunAngularRadius = float(std::asin(kSunRadiusKm / length(observerToSun)));
    result.moonAngularRadius = float(std::asin(kMoonRadiusKm / result.moonDistanceKm));
    const double cosinePhase = std::clamp(dot(moonToSun, moonToObserver), -1.0, 1.0);
    result.moonPhaseAngleRadians = float(std::acos(cosinePhase));
    result.moonIlluminatedFraction = float(0.5 * (1.0 + cosinePhase));
    result.moonLambertPhase = lambertMoonPhase(result.moonPhaseAngleRadians);
}

AstronomyEvaluation astronomicalEvaluation(const AstronomyState& state, double julianDate)
{
    // Original evaluation of Schlyter's low precision orbital elements and
    // lunar perturbations, equinox of date. All 12 longitude, 5 latitude and
    // 2 distance terms are included; no reference implementation was copied.
    // https://stjarnhimlen.se/comp/ppcomp.html (sections 4, 5, 7, 9, 12, 13).
    const double day = julianDate - 2451543.5;
    const double obliquity = radians(23.4393 - 3.563e-7 * day);
    const double sunPerihelion = 282.9404 + 4.70935e-5 * day;
    const double sunMean = 356.0470 + 0.9856002585 * day;
    const double sunEccentricity = 0.016709 - 1.151e-9 * day;
    const double sunEccentric = eccentricAnomaly(sunMean, sunEccentricity);
    const double sunX = std::cos(sunEccentric) - sunEccentricity;
    const double sunY = std::sqrt(1.0 - sunEccentricity * sunEccentricity) * std::sin(sunEccentric);
    const double sunLongitude = std::atan2(sunY, sunX) + radians(sunPerihelion);
    const double sunDistance = std::hypot(sunX, sunY);
    const Vector sunPosition = equatorial({sunDistance * std::cos(sunLongitude),
        sunDistance * std::sin(sunLongitude), 0.0}, obliquity) * kAstronomicalUnitKm;

    const double node = 125.1228 - 0.0529538083 * day;
    const double perihelion = 318.0634 + 0.1643573223 * day;
    const double moonMean = 115.3654 + 13.0649929509 * day;
    const double moonEccentric = eccentricAnomaly(moonMean, 0.054900);
    const double moonX = 60.2666 * (std::cos(moonEccentric) - 0.054900);
    const double moonY = 60.2666 * std::sqrt(1.0 - 0.054900 * 0.054900) * std::sin(moonEccentric);
    const double anomalyAndPerihelion = std::atan2(moonY, moonX) + radians(perihelion);
    double moonDistance = std::hypot(moonX, moonY);
    const Vector unperturbed = {moonDistance * (cosDegrees(node) * std::cos(anomalyAndPerihelion) -
        sinDegrees(node) * std::sin(anomalyAndPerihelion) * cosDegrees(5.1454)),
        moonDistance * (sinDegrees(node) * std::cos(anomalyAndPerihelion) +
        cosDegrees(node) * std::sin(anomalyAndPerihelion) * cosDegrees(5.1454)),
        moonDistance * std::sin(anomalyAndPerihelion) * sinDegrees(5.1454)};
    double moonLongitude = std::atan2(unperturbed.y, unperturbed.x) / kRadiansPerDegree;
    double moonLatitude = std::atan2(unperturbed.z, std::hypot(unperturbed.x, unperturbed.y)) / kRadiansPerDegree;
    const double elongation = moonMean + perihelion + node - sunMean - sunPerihelion;
    const double latitudeArgument = moonMean + perihelion;
    moonLongitude += -1.274 * sinDegrees(moonMean - 2.0 * elongation) + 0.658 * sinDegrees(2.0 * elongation)
        - 0.186 * sinDegrees(sunMean) - 0.059 * sinDegrees(2.0 * moonMean - 2.0 * elongation)
        - 0.057 * sinDegrees(moonMean - 2.0 * elongation + sunMean) + 0.053 * sinDegrees(moonMean + 2.0 * elongation)
        + 0.046 * sinDegrees(2.0 * elongation - sunMean) + 0.041 * sinDegrees(moonMean - sunMean)
        - 0.035 * sinDegrees(elongation) - 0.031 * sinDegrees(moonMean + sunMean)
        - 0.015 * sinDegrees(2.0 * latitudeArgument - 2.0 * elongation) + 0.011 * sinDegrees(moonMean - 4.0 * elongation);
    moonLatitude += -0.173 * sinDegrees(latitudeArgument - 2.0 * elongation)
        - 0.055 * sinDegrees(moonMean - latitudeArgument - 2.0 * elongation)
        - 0.046 * sinDegrees(moonMean + latitudeArgument - 2.0 * elongation)
        + 0.033 * sinDegrees(latitudeArgument + 2.0 * elongation)
        + 0.017 * sinDegrees(2.0 * moonMean + latitudeArgument);
    moonDistance += -0.58 * cosDegrees(moonMean - 2.0 * elongation) - 0.46 * cosDegrees(2.0 * elongation);
    const Vector moonPosition = equatorial({moonDistance * cosDegrees(moonLongitude) * cosDegrees(moonLatitude),
        moonDistance * sinDegrees(moonLongitude) * cosDegrees(moonLatitude), moonDistance * sinDegrees(moonLatitude)},
        obliquity) * kEarthEquatorialRadiusKm;

    const double latitude = radians(state.latitudeDegrees);
    // GMST in equinox of date; longitude east is added to Greenwich rotation.
    const double century = (julianDate - 2451545.0) / 36525.0;
    const double sidereal = radians(280.46061837 + 360.98564736629 * (julianDate - 2451545.0) +
        0.000387933 * century * century - century * century * century / 38710000.0 + state.longitudeDegrees);
    // Cartesian subtraction avoids singular topocentric RA/declination
    // formulas at the equator or poles. Account for Earth's flattening.
    constexpr double kFlattening = 1.0 / 298.257223563;
    const double eccentricitySquared = kFlattening * (2.0 - kFlattening);
    const double primeVertical = kEarthEquatorialRadiusKm /
        std::sqrt(1.0 - eccentricitySquared * std::sin(latitude) * std::sin(latitude));
    const Vector observer{primeVertical * std::cos(latitude) * std::cos(sidereal),
        primeVertical * std::cos(latitude) * std::sin(sidereal),
        primeVertical * (1.0 - eccentricitySquared) * std::sin(latitude)};
    AstronomyEvaluation result;
    result.automatic = true;
    result.julianDateUTC = julianDate;
    evaluateDirections(result, sunPosition, moonPosition, observer, latitude, sidereal);
    return result;
}

AstronomyEvaluation simplifiedEvaluation(const AstronomyState& state, double julianDate)
{
    // Uniform local mean-solar day with zero declination (equinox), no seasonal
    // equation of time. The lunar orbit is coplanar and has a uniform synodic
    // period; use Astronomical for actual seasonal and lunar orbital motion.
    const double localDays = julianDate + 0.5 + state.longitudeDegrees / 360.0;
    const double solarHourAngle = 2.0 * kPi * (localDays - std::floor(localDays) - 0.5);
    const double lunarElongation = std::remainder((julianDate - 2451550.1) / 29.530588853, 1.0) * 2.0 * kPi;
    const Vector sunPosition = horizontalDirection(0.0, solarHourAngle, radians(state.latitudeDegrees)) * kAstronomicalUnitKm;
    const Vector moonPosition = horizontalDirection(0.0, solarHourAngle - lunarElongation,
        radians(state.latitudeDegrees)) * 384400.0;
    AstronomyEvaluation result;
    result.automatic = true;
    result.julianDateUTC = julianDate;
    // Positions here are already ENU, so an identity equatorial-to-ENU basis
    // is unnecessary: compute phase geometry in the same frame directly.
    result.sunDirection = asFloat(normalized(sunPosition) * -1.0);
    result.moonDirection = asFloat(normalized(moonPosition) * -1.0);
    result.moonToSunDirection = asFloat(normalized(sunPosition - moonPosition));
    const double cosinePhase = std::clamp(dot(normalized(sunPosition - moonPosition), normalized(moonPosition) * -1.0), -1.0, 1.0);
    result.moonPhaseAngleRadians = float(std::acos(cosinePhase));
    result.moonIlluminatedFraction = float(0.5 * (1.0 + cosinePhase));
    result.moonLambertPhase = lambertMoonPhase(result.moonPhaseAngleRadians);
    return result;
}

float spectrumIlluminance(float3 spectrum)
{
    // Same 680/550/440 nm reconstruction and CIE Y quadrature as
    // Shaders/Modules/Atmosphere.slang::spectralToWorking().
    return float(683.0 * (18.3286150698 * spectrum.x + 76.9932864076 * spectrum.y + 11.6242660516 * spectrum.z));
}

} // namespace

double effectiveJulianDateUTC(const EnvironmentTimeState& time, double elapsedSeconds)
{
    const double elapsed = std::isfinite(elapsedSeconds) ? elapsedSeconds : 0.0;
    const double julianDate = time.julianDateUTC + (time.paused ? 0.0 : elapsed * time.timeScale / 86400.0);
    return std::clamp(julianDate, kMinimumJulianDateUTC, kMaximumJulianDateUTC);
}

float lambertMoonPhase(float phaseAngleRadians)
{
    const double angle = std::clamp(double(phaseAngleRadians), 0.0, kPi);
    if (angle <= 0.0) { return 1.0f; }
    if (angle >= kPi) { return 0.0f; }
    // Lambert sphere phase function, e.g. Seager/Whitney/Sasselov 2000 eq.3:
    // https://lweb.cfa.harvard.edu/~sasselov/exopl/p2.pdf
    return float(std::clamp((std::sin(angle) + (kPi - angle) * std::cos(angle)) / kPi, 0.0, 1.0));
}

AstronomyEvaluation evaluateAstronomy(const AstronomyState& astronomy, double julianDateUTC)
{
    const double julianDate = std::clamp(julianDateUTC, kMinimumJulianDateUTC, kMaximumJulianDateUTC);
    if (astronomy.mode == AstronomyMode::Astronomical) { return astronomicalEvaluation(astronomy, julianDate); }
    if (astronomy.mode == AstronomyMode::SimplifiedDayCycle) { return simplifiedEvaluation(astronomy, julianDate); }
    AstronomyEvaluation result;
    result.julianDateUTC = julianDate;
    return result;
}

float3 reflectedMoonIrradiance(float3 solarIrradiance, const AstronomyEvaluation& evaluation, float moonAlbedo)
{
    const double apparentRadius = kMoonRadiusKm / evaluation.moonDistanceKm;
    const double reflectedScale = moonAlbedo * (2.0 / 3.0) * apparentRadius * apparentRadius * evaluation.moonLambertPhase;
    return solarIrradiance * float(reflectedScale);
}

EnvironmentSnapshot evaluateWorldEnvironment(const WorldEnvironment& authored, double elapsedSeconds,
    std::optional<double> runtimeJulianDateUTC)
{
    EnvironmentSnapshot result;
    result.celestial = {authored.sun, authored.moon};
    result.authoredCelestial = result.celestial;
    result.source = authored.source;
    result.authoredAtmosphere = authored.atmosphere;
    result.atmosphere = evaluateWeatherAtmosphere(authored.atmosphere, authored.weather);
    result.time = authored.time;
    result.astronomy = authored.astronomy;
    result.weather = authored.weather;
    result.elapsedSeconds = std::isfinite(elapsedSeconds) ? elapsedSeconds : 0.0;
    const double julianDate = runtimeJulianDateUTC && std::isfinite(*runtimeJulianDateUTC)
        ? std::clamp(*runtimeJulianDateUTC, kMinimumJulianDateUTC, kMaximumJulianDateUTC)
        : effectiveJulianDateUTC(authored.time, result.elapsedSeconds);
    result.evaluatedAstronomy = evaluateAstronomy(authored.astronomy, julianDate);
    if (result.evaluatedAstronomy.automatic) {
        auto& sun = result.celestial[size_t(CelestialLightIndex::Sun)];
        auto& moon = result.celestial[size_t(CelestialLightIndex::Moon)];
        sun.direction = result.evaluatedAstronomy.sunDirection;
        sun.angularRadius = result.evaluatedAstronomy.sunAngularRadius;
        moon.direction = result.evaluatedAstronomy.moonDirection;
        moon.angularRadius = result.evaluatedAstronomy.moonAngularRadius;
        moon.topOfAtmosphereIrradiance = reflectedMoonIrradiance(sun.topOfAtmosphereIrradiance,
            result.evaluatedAstronomy, authored.astronomy.moonAlbedo);
        // HDRI retains the Sun's authored lux, while automatic lunar lighting
        // uses reflected photometric irradiance instead of an authored tint.
        moon.illuminance = spectrumIlluminance(moon.topOfAtmosphereIrradiance);
        moon.color = float3(1.0f, 1.0f, 1.0f);
    }
    return result;
}

} // namespace metallic::environment
