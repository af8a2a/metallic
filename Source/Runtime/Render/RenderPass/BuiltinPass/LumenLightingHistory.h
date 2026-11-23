#pragma once

#include <algorithm>

namespace metallic::render::builtin_pass {

inline bool lumenGlobalSunLightingChanged(float previousMax, float currentMax)
{
    // UE 5.7.4 LumenSceneRendering.cpp, UpdateGlobalLightingState.
    const float ratio = std::max(previousMax, 0.00001f) / std::max(currentMax, 0.00001f);
    return ratio > 4.0f || ratio < 0.25f;
}

} // namespace metallic::render::builtin_pass
