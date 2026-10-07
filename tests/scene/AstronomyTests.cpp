#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>

#include "Runtime/Environment/Astronomy.h"
#include "Runtime/Environment/Weather.h"
#include "Runtime/Environment/WorldEnvironment.h"

namespace metallic::environment {
namespace {

constexpr double kPi = std::numbers::pi_v<double>;

double dateUTC(int year, unsigned month, unsigned day, double hour = 0.0)
{
    const auto date = std::chrono::sys_days(std::chrono::year(year) / month / day);
    return 2440587.5 + double(date.time_since_epoch().count()) + hour / 24.0;
}

double vectorLength(float3 vector)
{
    return std::sqrt(double(vector.x) * vector.x + double(vector.y) * vector.y + double(vector.z) * vector.z);
}

double angularDifferenceDegrees(float3 lhs, float3 rhs)
{
    const double cosine = (double(lhs.x) * rhs.x + double(lhs.y) * rhs.y + double(lhs.z) * rhs.z) /
        (vectorLength(lhs) * vectorLength(rhs));
    return std::acos(std::clamp(cosine, -1.0, 1.0)) * 180.0 / kPi;
}

float3 directionFromAzimuthAltitude(double azimuthDegrees, double altitudeDegrees)
{
    const double azimuth = azimuthDegrees * kPi / 180.0;
    const double altitude = altitudeDegrees * kPi / 180.0;
    return float3(float(std::sin(azimuth) * std::cos(altitude)), float(std::sin(altitude)),
        float(std::cos(azimuth) * std::cos(altitude)));
}

float3 directionFromHourAngleDeclination(double hourAngleDegrees, double declinationDegrees, double latitudeDegrees)
{
    const double hourAngle = hourAngleDegrees * kPi / 180.0;
    const double declination = declinationDegrees * kPi / 180.0;
    const double latitude = latitudeDegrees * kPi / 180.0;
    return float3(float(-std::cos(declination) * std::sin(hourAngle)),
        float(std::sin(latitude) * std::sin(declination) + std::cos(latitude) * std::cos(declination) * std::cos(hourAngle)),
        float(std::cos(latitude) * std::sin(declination) - std::sin(latitude) * std::cos(declination) * std::cos(hourAngle)));
}

void expectSpectrumEqual(float3 actual, float3 expected)
{
    EXPECT_FLOAT_EQ(actual.x, expected.x);
    EXPECT_FLOAT_EQ(actual.y, expected.y);
    EXPECT_FLOAT_EQ(actual.z, expected.z);
}

TEST(Astronomy, PublishedSolarAndTopocentricLunarReference)
{
    // Schlyter's independently published worked example, 1990-04-19 00 UTC,
    // 60 N / 15 E: Sun az 15.6767, alt -17.9570; lunar topocentric
    // RA 310.0017, Dec -19.8790 at local sidereal angle 221.8388 degrees.
    // https://stjarnhimlen.se/comp/tutorial.html (sections 6 and 9).
    AstronomyState state{.mode = AstronomyMode::Astronomical, .latitudeDegrees = 60.0, .longitudeDegrees = 15.0};
    const auto result = evaluateAstronomy(state, dateUTC(1990, 4, 19));
    EXPECT_TRUE(result.automatic);
    EXPECT_LT(angularDifferenceDegrees(-result.sunDirection, directionFromAzimuthAltitude(15.6767, -17.9570)), 0.04);
    EXPECT_LT(angularDifferenceDegrees(-result.moonDirection,
        directionFromHourAngleDeclination(221.8388 - 310.0017, -19.8790, 60.0)), 0.06);
    EXPECT_NEAR(result.sunDistanceAU, 1.004323, 0.00006);
    EXPECT_GT(result.moonDistanceKm, 380000.0);
    EXPECT_LT(result.moonDistanceKm, 400000.0);
}

TEST(Astronomy, USNOIlluminatedFractionAcrossLunarCycle)
{
    // USNO 2024 fraction table at 00 UTC (rounded to 0.01), including new,
    // quarter, full and crescent phases. Observer parallax changes apparent
    // illuminated fraction slightly, so tolerance includes that difference.
    // https://aa.usno.navy.mil/calculated/moon/fraction?submit=Get+Data&task=00&tz=0.00&tz_label=false&tz_sign=-1&year=2024
    const AstronomyState state{.mode = AstronomyMode::Astronomical};
    struct Reference { unsigned month, day; float fraction; };
    const Reference references[] = {{1, 1, 0.78f}, {1, 8, 0.15f}, {1, 18, 0.48f}, {1, 26, 1.00f},
        {4, 2, 0.52f}, {4, 9, 0.00f}, {4, 16, 0.52f}, {4, 24, 1.00f}};
    for (const auto& reference : references) {
        SCOPED_TRACE(reference.month * 100 + reference.day);
        const auto result = evaluateAstronomy(state, dateUTC(2024, reference.month, reference.day));
        EXPECT_NEAR(result.moonIlluminatedFraction, reference.fraction, 0.015f);
        EXPECT_GE(result.moonLambertPhase, 0.0f);
        EXPECT_LE(result.moonLambertPhase, 1.0f);
        const double cosinePhase = double(result.moonDirection.x) * result.moonToSunDirection.x +
            double(result.moonDirection.y) * result.moonToSunDirection.y +
            double(result.moonDirection.z) * result.moonToSunDirection.z;
        EXPECT_NEAR(cosinePhase, std::cos(result.moonPhaseAngleRadians), 2e-7);
    }
}

TEST(Astronomy, SimplifiedENUHasEastSunriseAndWestSunset)
{
    const AstronomyState state{.mode = AstronomyMode::SimplifiedDayCycle};
    const auto sunrise = evaluateAstronomy(state, dateUTC(2024, 3, 20, 6));
    const auto noon = evaluateAstronomy(state, dateUTC(2024, 3, 20, 12));
    const auto sunset = evaluateAstronomy(state, dateUTC(2024, 3, 20, 18));
    const auto midnight = evaluateAstronomy(state, dateUTC(2024, 3, 20));
    EXPECT_LT(angularDifferenceDegrees(-sunrise.sunDirection, float3(1, 0, 0)), 0.001);
    EXPECT_LT(angularDifferenceDegrees(-noon.sunDirection, float3(0, 1, 0)), 0.001);
    EXPECT_LT(angularDifferenceDegrees(-sunset.sunDirection, float3(-1, 0, 0)), 0.001);
    EXPECT_LT(angularDifferenceDegrees(-midnight.sunDirection, float3(0, -1, 0)), 0.001);
    auto east = state;
    east.longitudeDegrees = 90;
    const auto eastNoon = evaluateAstronomy(east, dateUTC(2024, 3, 20, 6));
    EXPECT_LT(angularDifferenceDegrees(eastNoon.sunDirection, noon.sunDirection), 0.001);
}

TEST(Astronomy, SeasonalPolarDayNightAndMidnightContinuity)
{
    for (const double latitude : {-90.0, 0.0, 90.0}) {
        AstronomyState state{.mode = AstronomyMode::Astronomical, .latitudeDegrees = latitude};
        const auto summer = evaluateAstronomy(state, dateUTC(2024, 6, 21, 12));
        const auto winter = evaluateAstronomy(state, dateUTC(2024, 12, 21, 12));
        for (const auto& result : {summer, winter}) {
            EXPECT_NEAR(vectorLength(result.sunDirection), 1.0, 1e-7);
            EXPECT_NEAR(vectorLength(result.moonDirection), 1.0, 1e-7);
            EXPECT_NEAR(vectorLength(result.moonToSunDirection), 1.0, 1e-7);
        }
        if (latitude == 90.0) {
            EXPECT_LT(summer.sunDirection.y, -0.35f);
            EXPECT_GT(winter.sunDirection.y, 0.35f);
        } else if (latitude == -90.0) {
            EXPECT_GT(summer.sunDirection.y, 0.35f);
            EXPECT_LT(winter.sunDirection.y, -0.35f);
        }
    }
    const AstronomyState state{.mode = AstronomyMode::Astronomical, .latitudeDegrees = 51.5, .longitudeDegrees = -0.12};
    const double midnight = dateUTC(2024, 1, 1);
    const auto before = evaluateAstronomy(state, midnight - 1.0 / 86400.0);
    const auto after = evaluateAstronomy(state, midnight + 1.0 / 86400.0);
    EXPECT_LT(angularDifferenceDegrees(before.sunDirection, after.sunDirection), 0.01);
    EXPECT_LT(angularDifferenceDegrees(before.moonDirection, after.moonDirection), 0.01);
}

TEST(Astronomy, ManualPreservesAuthoredLightsAndPlaybackDoesNotMutate)
{
    WorldEnvironment world;
    world.sun.enabled = true;
    world.sun.direction = float3(1, -2, 3);
    world.sun.illuminance = 7777;
    world.sun.angularRadius = 0.03f;
    world.moon.enabled = true;
    world.moon.color = float3(0.8f, 0.9f, 1.2f);
    world.moon.angularRadius = 0.02f;
    const auto authorCopy = world;
    const auto manual = evaluateWorldEnvironment(world, 100.0);
    EXPECT_EQ(manual.celestial[0], world.sun);
    EXPECT_EQ(manual.celestial[1], world.moon);
    EXPECT_FALSE(manual.evaluatedAstronomy.automatic);
    EXPECT_EQ(world, authorCopy);
    EXPECT_EQ(manual.authoredCelestial, manual.celestial);
    EXPECT_EQ(manual.authoredAtmosphere, world.atmosphere);
    EXPECT_DOUBLE_EQ(manual.elapsedSeconds, 100.0);
    EXPECT_DOUBLE_EQ(manual.evaluatedAstronomy.julianDateUTC, world.time.julianDateUTC);
    world.astronomy.mode = AstronomyMode::Astronomical;
    world.time.paused = false;
    world.time.timeScale = 24.0;
    const auto automatic = evaluateWorldEnvironment(world, 3600.0);
    EXPECT_DOUBLE_EQ(automatic.evaluatedAstronomy.julianDateUTC, world.time.julianDateUTC + 1.0);
    EXPECT_DOUBLE_EQ(world.time.julianDateUTC, authorCopy.time.julianDateUTC);
    EXPECT_EQ(automatic.celestial[0].enabled, world.sun.enabled);
    EXPECT_EQ(automatic.celestial[1].enabled, world.moon.enabled);
    expectSpectrumEqual(automatic.celestial[0].topOfAtmosphereIrradiance, world.sun.topOfAtmosphereIrradiance);
    const auto sought = evaluateWorldEnvironment(world, 1800.0, dateUTC(2024, 4, 24));
    EXPECT_DOUBLE_EQ(sought.evaluatedAstronomy.julianDateUTC, dateUTC(2024, 4, 24));
    EXPECT_DOUBLE_EQ(sought.elapsedSeconds, 1800.0);
    world.time.timeScale = -24.0;
    EXPECT_DOUBLE_EQ(effectiveJulianDateUTC(world.time, 3600.0), world.time.julianDateUTC - 1.0);
}

TEST(Astronomy, MoonRadiometryDependsOnSolarSpectrumPhaseDistanceAndAlbedo)
{
    AstronomyEvaluation evaluation;
    const float3 solar(1.0f, 2.0f, 3.0f);
    const auto full = reflectedMoonIrradiance(solar, evaluation, 0.12f);
    const double expectedScale = 0.12 * (2.0 / 3.0) * std::pow(1737.4 / 384400.0, 2.0);
    EXPECT_NEAR(full.x, expectedScale, 1e-13);
    EXPECT_NEAR(full.y, 2.0 * full.x, 1e-12);
    EXPECT_NEAR(full.z, 3.0 * full.x, 1e-12);
    evaluation.moonDistanceKm *= 2.0;
    const auto distant = reflectedMoonIrradiance(solar, evaluation, 0.12f);
    EXPECT_FLOAT_EQ(distant.x, full.x * 0.25f);
    evaluation.moonDistanceKm *= 0.5;
    evaluation.moonLambertPhase = lambertMoonPhase(float(kPi * 0.5));
    EXPECT_NEAR(evaluation.moonLambertPhase, 1.0 / kPi, 1e-7);
    EXPECT_NEAR(reflectedMoonIrradiance(solar, evaluation, 0.12f).x / full.x, 1.0 / kPi, 1e-7);
    evaluation.moonLambertPhase = lambertMoonPhase(float(kPi));
    EXPECT_FLOAT_EQ(reflectedMoonIrradiance(solar, evaluation, 0.12f).x, 0.0f);
    EXPECT_FLOAT_EQ(reflectedMoonIrradiance(solar, evaluation, 0.0f).y, 0.0f);
    EXPECT_FLOAT_EQ(lambertMoonPhase(0.0f), 1.0f);
}

TEST(Weather, DefaultPreservesDryAtmosphereAndGrowthOnlyChangesMie)
{
    WorldEnvironment world;
    const auto dry = evaluateWorldEnvironment(world);
    EXPECT_EQ(dry.atmosphere, world.atmosphere);
    world.weather.aerosolDensity = 2.0f;
    world.weather.humidity = 0.5f;
    const auto authorCopy = world;
    const auto wet = evaluateWorldEnvironment(world);
    expectSpectrumEqual(wet.atmosphere.mieScattering, world.atmosphere.mieScattering * 3.5f);
    expectSpectrumEqual(wet.atmosphere.mieExtinction, world.atmosphere.mieExtinction * 3.5f);
    expectSpectrumEqual(wet.atmosphere.rayleighScattering, world.atmosphere.rayleighScattering);
    expectSpectrumEqual(wet.atmosphere.ozoneAbsorption, world.atmosphere.ozoneAbsorption);
    EXPECT_EQ(wet.celestial[0], world.sun);
    EXPECT_EQ(wet.celestial[1], world.moon);
    EXPECT_EQ(world, authorCopy);
    world.weather.aerosolDensity = 0.0f;
    const auto vacuumAerosols = evaluateWorldEnvironment(world);
    expectSpectrumEqual(vacuumAerosols.atmosphere.mieExtinction, float3(0.0f));
    expectSpectrumEqual(vacuumAerosols.atmosphere.mieScattering, float3(0.0f));
}

TEST(Weather, CoefficientLimitPreservesSpectralRatiosAndWindIsNormalized)
{
    AtmosphereState authored;
    authored.mieExtinction = float3(100.0f, 200.0f, 1000.0f);
    authored.mieScattering = authored.mieExtinction * 0.9f;
    WeatherState weather;
    weather.aerosolDensity = 100.0f;
    weather.humidity = 1.0f;
    weather.windDirection = float2(3.0f, 4.0f);
    const auto effective = evaluateWeatherAtmosphere(authored, weather);
    EXPECT_TRUE(validAtmosphereState(effective));
    expectSpectrumEqual(effective.mieExtinction, authored.mieExtinction);
    expectSpectrumEqual(effective.mieScattering, authored.mieScattering);
    const auto wind = normalizedWeatherWindDirection(weather);
    EXPECT_FLOAT_EQ(wind.x, 0.6f);
    EXPECT_FLOAT_EQ(wind.y, 0.8f);
}

TEST(Astronomy, RejectsInvalidAuthorStateAndStopsAtSupportedDateBounds)
{
    EnvironmentTimeState time;
    time.paused = false;
    EXPECT_DOUBLE_EQ(effectiveJulianDateUTC(time, 1e12), kMaximumJulianDateUTC);
    EXPECT_DOUBLE_EQ(effectiveJulianDateUTC(time, -1e12), kMinimumJulianDateUTC);
    EXPECT_FALSE(validEnvironmentTimeState({.julianDateUTC = std::numeric_limits<double>::quiet_NaN()}));
    AstronomyState astronomy;
    astronomy.latitudeDegrees = 91.0;
    EXPECT_FALSE(validAstronomyState(astronomy));
    astronomy.latitudeDegrees = 90.0;
    EXPECT_TRUE(validAstronomyState(astronomy));
    astronomy.moonAlbedo = -0.01f;
    EXPECT_FALSE(validAstronomyState(astronomy));
    WeatherState weather;
    weather.humidity = 1.1f;
    EXPECT_FALSE(validWeatherState(weather));
    weather.humidity = 0.0f;
    weather.cloudTopAltitudeKm = weather.cloudBaseAltitudeKm;
    EXPECT_FALSE(validWeatherState(weather));
    weather.cloudTopAltitudeKm = 4.0f;
    weather.windDirection = float2(0.0f);
    EXPECT_FALSE(validWeatherState(weather));
}

} // namespace
} // namespace metallic::environment
