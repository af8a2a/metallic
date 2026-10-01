#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <cmath>

namespace metallic::render {

// Texture encoding is explicit: float storage alone does not imply scene radiance.
enum class DisplayColorEncoding : uint8_t {
    sRGB,
    ExposedLinear,
    scRGB,
    SceneLinear, // Rec.709 primaries, D65, unbounded scene-referred radiance.
};

struct DisplayOutputParameters {
    DisplayOutputMode mode = DisplayOutputMode::SDR;
    float paperWhiteNits = 203.0f;
    float peakNits = 1000.0f;
    float exposureEV = 0.0f;
    bool calibrationPattern = false;

    bool valid() const
    {
        return (mode == DisplayOutputMode::SDR_sRGB || isHDROutput(mode)) &&
            std::isfinite(paperWhiteNits) && paperWhiteNits >= 80.0f && paperWhiteNits <= 10000.0f &&
            std::isfinite(peakNits) && peakNits >= paperWhiteNits && peakNits <= 10000.0f &&
            std::isfinite(exposureEV) && exposureEV >= -20.0f && exposureEV <= 20.0f;
    }

    bool operator==(const DisplayOutputParameters&) const = default;
};

} // namespace metallic::render
