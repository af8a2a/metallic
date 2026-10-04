#include "MaterialClosureClassification.h"

#include <algorithm>
#include <stdexcept>

namespace metallic::render {

MaterialClosureClassification MaterialClosureClassification::create(
    std::span<const std::optional<MaterialClosureFamily>> programFamilies)
{
    if (programFamilies.empty() || programFamilies[0]) {
        throw std::invalid_argument("Closure classification requires background at program slot zero");
    }
    MaterialClosureClassification result;
    result.families.push_back(std::nullopt);
    for (uint32_t program = 0; program < programFamilies.size(); ++program) {
        const auto family = programFamilies[program];
        if (program != 0 && (!family || (*family != MaterialClosureFamily::SingleSlabClosure &&
            *family != MaterialClosureFamily::DualSlabClosure && *family != MaterialClosureFamily::OpenPBRCompositeClosure))) {
            throw std::invalid_argument("Surface program has no supported Closure Family");
        }
        auto found = std::find(result.families.begin(), result.families.end(), family);
        const auto bin = static_cast<uint32_t>(found - result.families.begin());
        if (found == result.families.end()) { result.families.push_back(family); }
        result.programFamilyBins.push_back(bin);
    }
    result.programCount = static_cast<uint32_t>(programFamilies.size() - 1);
    result.closureFamilyCount = static_cast<uint32_t>(result.families.size() - 1);
    for (uint32_t family = 0; family < result.families.size(); ++family) {
        result.familyProgramOffsets.push_back(static_cast<uint32_t>(result.programOrder.size()));
        for (uint32_t program = 0; program < programFamilies.size(); ++program) {
            if (result.programFamilyBins[program] == family) { result.programOrder.push_back(program); }
        }
    }
    result.familyProgramOffsets.push_back(static_cast<uint32_t>(result.programOrder.size()));
    return result;
}

} // namespace metallic::render
