#pragma once

#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"
#include "Runtime/Render/ScreenSpaceShadows.h"

namespace metallic::render::builtin_pass {

inline bool rayTracedShadowsEnabled(const RenderGraphProperties& properties)
{
    // Accept saved prototype graphs without retaining a screen-space tracing path.
    return boolProperty(&properties, "rayTracedShadows", boolProperty(&properties, "screenSpaceShadows", true));
}

inline std::vector<RenderGraphRuntimeSetting> screenSpaceShadowRuntimeSettings(const RenderGraphProperties& properties)
{
    return {
        runtimeBoolSetting("rayTracedShadows", "Ray-traced Shadows", rayTracedShadowsEnabled(properties), true),
        runtimeBoolSetting("sigmaDenoise", "SIGMA Shadow Denoising", true, true),
        runtimeBoolSetting("shadowDebug", "Show Shadow Visibility", false, true),
        runtimeIntSetting("shadowLightIndex", "Shadow Light (-1 Auto)", -1, -1, 65535, true),
        runtimeFloatSetting("shadowRayLength", "Shadow Ray Length (m)", 100000.0f, 0.01f, 1000000.0f, true),
        runtimeFloatSetting("shadowBias", "Shadow Normal Bias (m)", 0.01f, 0.0f, 1.0f, true),
        runtimeFloatSetting("shadowAngularRadius", "Shadow Angular Radius (deg)", 0.266f, 0.0f, 10.0f, true),
        runtimeFloatSetting("shadowLightRadius", "Local Shadow Light Radius (m)", 0.05f, 0.0f, 10.0f, true),
        runtimeIntSetting("sigmaHistoryLength", "SIGMA History", 5, 0, 7, true),
    };
}

inline ScreenSpaceShadowSettings screenSpaceShadowSettings(const RenderGraphProperties& properties)
{
    ScreenSpaceShadowSettings settings;
    settings.enabled = rayTracedShadowsEnabled(properties) && !boolProperty(&properties, "debugDisableShadows", false);
    settings.denoise = boolProperty(&properties, "sigmaDenoise", true);
    settings.debug = boolProperty(&properties, "shadowDebug", false);
    settings.lightIndex = std::clamp(properties.value("shadowLightIndex", -1), -1, 65535);
    settings.historyLength = static_cast<uint32_t>(std::clamp(properties.value("sigmaHistoryLength", 5), 0, 7));
    settings.maxDistance = std::clamp(properties.value("shadowRayLength", 100000.0f), 0.01f, 1000000.0f);
    settings.normalBias = std::clamp(properties.value("shadowBias", 0.01f), 0.0f, 1.0f);
    settings.angularRadiusDegrees = std::clamp(properties.value("shadowAngularRadius", 0.266f), 0.0f, 10.0f);
    settings.lightRadius = std::clamp(properties.value("shadowLightRadius", 0.05f), 0.0f, 10.0f);
    return settings;
}

} // namespace metallic::render::builtin_pass
