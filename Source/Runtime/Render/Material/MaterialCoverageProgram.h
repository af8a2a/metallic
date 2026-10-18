#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <json.hpp>
#include "Runtime/Material/MaterialValueIR.h"

namespace metallic::render {

// Postorder, read-only Coverage slice. Every operand precedes its consumer.
// Retained for validation oracles; production uploads only instance parameters.
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

// Emit a statically compiled function with explicit inputs. The caller owns the
// module/namespace and dispatch; no Surface resources or runtime opcodes remain.
std::string emitMaterialCoverageProgram(const MaterialValueIR& coverage, uint32_t programId);

} // namespace metallic::render
