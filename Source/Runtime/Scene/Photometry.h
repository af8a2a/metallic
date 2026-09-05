#pragma once

#include <cmath>
#include <initializer_list>
#include <limits>
#include <string_view>

namespace metallic::scene {

// SI means lux for a directional light, candela for a point/spot light.
// All scene-space distances are metres. RGB colors are linear multipliers.
enum class LightUnit { SI, Lux, Candela, Lumens, EV100 };

inline const char* lightUnitName(LightUnit unit)
{
    switch (unit) {
    case LightUnit::SI: return "si";
    case LightUnit::Lux: return "lux";
    case LightUnit::Candela: return "candela";
    case LightUnit::Lumens: return "lumens";
    case LightUnit::EV100: return "ev100";
    }
    return "invalid";
}

inline bool parseLightUnit(std::string_view name, LightUnit& unit)
{
    for (LightUnit candidate : {LightUnit::SI, LightUnit::Lux, LightUnit::Candela,
             LightUnit::Lumens, LightUnit::EV100}) {
        if (name == lightUnitName(candidate)) {
            unit = candidate;
            return true;
        }
    }
    return false;
}

inline bool validLightUnit(std::string_view type, LightUnit unit)
{
    if (type != "directional" && type != "point" && type != "spot") {
        return false;
    }
    return unit == LightUnit::SI || unit == LightUnit::EV100 ||
        (type == "directional" ? unit == LightUnit::Lux :
            (unit == LightUnit::Candela || unit == LightUnit::Lumens));
}

// Integral of the squared cosine-space spot falloff used by KHR_lights_punctual.
inline double lightSolidAngle(std::string_view type, double inner, double outer)
{
    constexpr double kTwoPi = 6.28318530717958647692;
    return type == "spot"
        ? kTwoPi * (1.0 - std::cos(inner) + (std::cos(inner) - std::cos(outer)) / 3.0)
        : 2.0 * kTwoPi;
}

inline double lightIntensitySI(std::string_view type, LightUnit unit, double value,
    double inner = 0.0, double outer = 0.7853981633974483)
{
    if (!validLightUnit(type, unit) || !std::isfinite(value)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (unit == LightUnit::EV100) {
        // Incident-light meter C=250 at ISO 100: lux=2.5*2^EV100.
        // Local lights follow UE LocalLightComponent / RenderUtils: cd=2^EV.
        return (type == "directional" ? 2.5 : 1.0) * std::exp2(value);
    }
    return unit == LightUnit::Lumens ? value / lightSolidAngle(type, inner, outer) : value;
}

inline double lightIntensityFromSI(std::string_view type, LightUnit unit, double value,
    double inner = 0.0, double outer = 0.7853981633974483)
{
    if (unit == LightUnit::EV100) {
        return std::log2(value / (type == "directional" ? 2.5 : 1.0));
    }
    return unit == LightUnit::Lumens ? value * lightSolidAngle(type, inner, outer) : value;
}

} // namespace metallic::scene
