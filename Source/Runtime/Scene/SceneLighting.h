#pragma once

#include "Runtime/Scene/SceneComponents.h"

namespace metallic::scene {

struct PunctualLight {
    std::string name = "Light";
    LightProperties properties{.type = "point", .intensity = 1000.0};
    float3 position{0.0f, 2.0f, 0.0f};
    // Direction of emitted light, in world space; point lights ignore it.
    float3 direction{0.0f, -1.0f, 0.0f};
    bool enabled = true;
};

struct LightingSettings {
    std::vector<PunctualLight> lights;
    // Manual sensor saturation exposure: Lmax=2^EV100 cd/m^2, as in UE
    // RenderUtils::EV100ToLuminance(1, EV100). Applied after light transport.
    float exposureEV100 = 0.0f;
};

inline bool validLightingSettings(const LightingSettings& settings)
{
    if (!std::isfinite(settings.exposureEV100) ||
        settings.exposureEV100 < -32.0f || settings.exposureEV100 > 32.0f) {
        return false;
    }
    for (const PunctualLight& light : settings.lights) {
        const float3& p = light.position;
        const float3& d = light.direction;
        if (!validLightProperties(light.properties) ||
            !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z) ||
            (light.properties.type != "point" &&
                (double(d.x) * d.x + double(d.y) * d.y + double(d.z) * d.z) < 1e-12)) {
            return false;
        }
    }
    return true;
}

} // namespace metallic::scene
