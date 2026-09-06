#pragma once

#include "Runtime/Scene/SceneComponents.h"

#include <optional>

namespace metallic::scene {

struct ImportedLightBinding {
    // Persistent identity is per source node, not per shared glTF light definition.
    std::string sourceId;
    int32_t sourceNodeIndex = -1;
    float3 localPosition{0.0f, 0.0f, 0.0f};
    float3 localDirection{0.0f, 0.0f, -1.0f};
    // Resolved for the current load only. Never serialize EnTT handles or lifetimes.
    uint64_t sceneIdentity = 0;
    SceneEntity object = kNullSceneEntity;
};

struct PunctualLight {
    std::string name = "Light";
    LightProperties properties{.type = "point", .intensity = 1000.0};
    float3 position{0.0f, 2.0f, 0.0f};
    // Direction of emitted light, in world space; point lights ignore it.
    float3 direction{0.0f, -1.0f, 0.0f};
    bool enabled = true;
    // Imported lights remain native editable lights, with poses relative to their
    // source node. position/direction are resolved world-space editor snapshots.
    std::optional<ImportedLightBinding> imported;
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
        if (light.imported) {
            const auto& source = *light.imported;
            const auto& localP = source.localPosition;
            const auto& localD = source.localDirection;
            if (source.sourceId.empty() || source.sourceNodeIndex < 0 ||
                !std::isfinite(localP.x) || !std::isfinite(localP.y) || !std::isfinite(localP.z) ||
                !std::isfinite(localD.x) || !std::isfinite(localD.y) || !std::isfinite(localD.z) ||
                (light.properties.type != "point" &&
                    double(localD.x) * localD.x + double(localD.y) * localD.y + double(localD.z) * localD.z < 1e-12)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace metallic::scene
