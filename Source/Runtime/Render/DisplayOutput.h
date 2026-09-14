#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <cmath>

namespace metallic::render {

// Texture encoding is explicit: float storage alone does not imply scene radiance.
enum class DisplayColorEncoding : uint8_t {
    Srgb,
    ExposedLinear,
    ScRgb,
};

struct DisplayOutputParameters {
    DisplayOutputMode mode = DisplayOutputMode::Sdr;
    float paperWhiteNits = 203.0f;
    float peakNits = 1000.0f;
    float exposureEV = 0.0f;
    bool calibrationPattern = false;

    bool valid() const
    {
        return (mode == DisplayOutputMode::Sdr || mode == DisplayOutputMode::HdrScRgb) &&
            std::isfinite(paperWhiteNits) && paperWhiteNits >= 80.0f && paperWhiteNits <= 10000.0f &&
            std::isfinite(peakNits) && peakNits >= paperWhiteNits && peakNits <= 10000.0f &&
            std::isfinite(exposureEV) && exposureEV >= -20.0f && exposureEV <= 20.0f;
    }

    bool operator==(const DisplayOutputParameters&) const = default;
};

} // namespace metallic::render
