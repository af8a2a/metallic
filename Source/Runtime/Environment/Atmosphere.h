#pragma once

#include "ml.h"

#include <array>
#include <cmath>
#include <cstdint>

namespace metallic::environment {

enum class EnvironmentSource : uint32_t {
    HDRI = 0,
    PhysicalAtmosphere = 1,
};

// Spectral-like samples at 680, 550 and 440 nm, in that order. These are
// propagation coefficients, not Rec.709 or ACEScg color components.
struct AtmosphereState {
    std::array<double, 3> planetCenter{0.0, -6360000.0, 0.0}; // World metres.
    float bottomRadiusKm = 6360.0f;
    float topRadiusKm = 6460.0f;
    float3 rayleighScattering{0.005802f, 0.013558f, 0.033100f}; // km^-1.
    float rayleighScaleHeightKm = 8.0f;
    float3 mieScattering{0.003996f, 0.003996f, 0.003996f}; // km^-1.
    float3 mieExtinction{0.004440f, 0.004440f, 0.004440f}; // km^-1.
    float mieScaleHeightKm = 1.2f;
    float mieAnisotropy = 0.8f;
    float3 ozoneAbsorption{0.000650f, 0.001881f, 0.000085f}; // km^-1.
    float ozoneCenterAltitudeKm = 25.0f;
    float ozoneWidthKm = 15.0f; // Triangular profile half width.
    float3 groundAlbedo{0.3f, 0.3f, 0.3f};
    float maxAerialDistanceKm = 1000.0f;

    bool operator==(const AtmosphereState& other) const
    {
        const auto equal = [](const float3& lhs, const float3& rhs) {
            return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
        };
        return planetCenter == other.planetCenter && bottomRadiusKm == other.bottomRadiusKm &&
            topRadiusKm == other.topRadiusKm && equal(rayleighScattering, other.rayleighScattering) &&
            rayleighScaleHeightKm == other.rayleighScaleHeightKm &&
            equal(mieScattering, other.mieScattering) && equal(mieExtinction, other.mieExtinction) &&
            mieScaleHeightKm == other.mieScaleHeightKm && mieAnisotropy == other.mieAnisotropy &&
            equal(ozoneAbsorption, other.ozoneAbsorption) &&
            ozoneCenterAltitudeKm == other.ozoneCenterAltitudeKm && ozoneWidthKm == other.ozoneWidthKm &&
            equal(groundAlbedo, other.groundAlbedo) && maxAerialDistanceKm == other.maxAerialDistanceKm;
    }
};

inline bool validAtmosphereState(const AtmosphereState& atmosphere)
{
    const auto finiteRange = [](float value, float minimum, float maximum) {
        return std::isfinite(value) && value >= minimum && value <= maximum;
    };
    const auto spectrumRange = [&](const float3& value, float maximum) {
        return finiteRange(value.x, 0.0f, maximum) && finiteRange(value.y, 0.0f, maximum) &&
            finiteRange(value.z, 0.0f, maximum);
    };
    for (const double value : atmosphere.planetCenter) {
        if (!std::isfinite(value) || std::abs(value) > 1e15) { return false; }
    }
    return finiteRange(atmosphere.bottomRadiusKm, 1.0f, 1e8f) &&
        finiteRange(atmosphere.topRadiusKm, 1.0f, 1e8f) &&
        atmosphere.topRadiusKm > atmosphere.bottomRadiusKm &&
        spectrumRange(atmosphere.rayleighScattering, 1000.0f) &&
        finiteRange(atmosphere.rayleighScaleHeightKm, 0.001f, 1e5f) &&
        spectrumRange(atmosphere.mieScattering, 1000.0f) &&
        spectrumRange(atmosphere.mieExtinction, 1000.0f) &&
        atmosphere.mieScattering.x <= atmosphere.mieExtinction.x &&
        atmosphere.mieScattering.y <= atmosphere.mieExtinction.y &&
        atmosphere.mieScattering.z <= atmosphere.mieExtinction.z &&
        finiteRange(atmosphere.mieScaleHeightKm, 0.001f, 1e5f) &&
        finiteRange(atmosphere.mieAnisotropy, -0.99f, 0.99f) &&
        spectrumRange(atmosphere.ozoneAbsorption, 1000.0f) &&
        finiteRange(atmosphere.ozoneCenterAltitudeKm, 0.0f, 1e5f) &&
        finiteRange(atmosphere.ozoneWidthKm, 0.001f, 1e5f) &&
        spectrumRange(atmosphere.groundAlbedo, 1.0f) &&
        finiteRange(atmosphere.maxAerialDistanceKm, 0.001f, 1e6f);
}

} // namespace metallic::environment
