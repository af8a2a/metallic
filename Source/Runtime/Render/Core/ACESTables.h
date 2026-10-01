#pragma once

#include <array>

namespace metallic::render {
struct ACESTables {
    std::array<float, 360> reach;
    std::array<float, 362 * 3> gamut;
    std::array<float, 362> gamma;
};
ACESTables makeACESTables(float peakNits);
} // namespace metallic::render
