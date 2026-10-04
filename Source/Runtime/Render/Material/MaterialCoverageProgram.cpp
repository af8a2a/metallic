#include "Runtime/Render/Material/MaterialCoverageProgram.h"
#include <stdexcept>

namespace metallic::render {
MaterialCoverageSlice compileMaterialCoverageSlice(const MaterialValueIR& coverage)
{
    if (coverage.outputs().size() != 1 || !coverage.outputs().contains("coverage") || coverage.nodes().size() > 64) {
        throw std::runtime_error("Coverage requires one output and at most 64 live IR instructions");
    }
    MaterialCoverageSlice slice;
    slice.parameterMask = coverage.usage().parameterMask;
    for (const auto& node : coverage.nodes()) {
        if (node.op > MaterialValueOp::Select) {
            throw std::runtime_error("Coverage IR permits UV, shared parameters and alpha at finest resident mip; Surface inputs and arbitrary texture footprints are unavailable");
        }
        MaterialCoverageInstruction instruction;
        instruction.operation = {static_cast<uint32_t>(node.op), node.operands[0], node.operands[1], node.operands[2]};
        instruction.constant = node.constant;
        if (node.op == MaterialValueOp::Parameter) { instruction.operation[1] = node.index; }
        if (node.op == MaterialValueOp::Swizzle) { instruction.constant[0] = float(node.index); }
        slice.usesBaseAlpha |= node.op == MaterialValueOp::Alpha;
        slice.instructions.push_back(instruction);
    }
    return slice;
}

MaterialCoverageSlice compileMaterialCoverageSlice(const nlohmann::json& expression)
{
    return compileMaterialCoverageSlice(MaterialValueIR::lower({{"version", 1}, {"coverage", expression}}));
}
} // namespace metallic::render
