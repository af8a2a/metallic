#pragma once

#include "MaterialClosureIR.h"
#include <optional>

namespace metallic::render {

// Logical second-level schedule. Program slots still select material evaluation;
// family slots select a compatible closure layout and lighting implementation.
// Slot zero in both tables is background and is excluded from the ratio.
struct MaterialClosureClassification
{
    uint32_t programCount = 0;
    uint32_t closureFamilyCount = 0;
    std::vector<std::optional<MaterialClosureFamily>> families;
    std::vector<uint32_t> programFamilyBins;
    std::vector<uint32_t> familyProgramOffsets; // size families.size()+1
    std::vector<uint32_t> programOrder; // Stable grouping; original bin IDs are preserved.

    static MaterialClosureClassification create(std::span<const std::optional<MaterialClosureFamily>> programFamilies);
};

} // namespace metallic::render
