#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <json.hpp>
#include "Runtime/Material/MaterialValueIR.h"

namespace metallic::render {

// Postorder, read-only Coverage slice. Every operand precedes its consumer.
// This bounded backend is independent of generated Surface Slang programs.
struct MaterialCoverageInstruction
{
    std::array<uint32_t, 4> operation{};
    std::array<float, 4> constant{};
};
static_assert(sizeof(MaterialCoverageInstruction) == 32);
struct MaterialCoverageSlice
{
    std::vector<MaterialCoverageInstruction> instructions;
    uint32_t parameterMask = 0;
    bool usesBaseAlpha = false;
};

// Throws on malformed/unsupported expressions; the owning generation publishes
// only after both Surface and Coverage compilation and upload have succeeded.
MaterialCoverageSlice compileMaterialCoverageSlice(const nlohmann::json& expression);
MaterialCoverageSlice compileMaterialCoverageSlice(const MaterialValueIR& coverage);

} // namespace metallic::render
