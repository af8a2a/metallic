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

std::string emitMaterialCoverageProgram(const MaterialValueIR& coverage, uint32_t programId)
{
    // Keep the same supported language and bounded authored IR as the oracle.
    (void)compileMaterialCoverageSlice(coverage);
    std::string source = "float materialCoverage" + std::to_string(programId) +
        "(ByteAddressBuffer inputs, uint parameterOffset, float2 uv, float baseAlpha)\n{\n";
    for (size_t index = 0; index < coverage.nodes().size(); ++index) {
        const auto& node = coverage.nodes()[index];
        const auto operand = [&](uint32_t i) { return "v" + std::to_string(node.operands[i]); };
        const auto a = operand(0), b = operand(1), c = operand(2);
        std::string value;
        switch (node.op) {
        case MaterialValueOp::Constant:
            value = "float4(";
            for (uint32_t i = 0; i < 4; ++i) { value += (i ? "," : "") + nlohmann::json(node.constant[i]).dump(); }
            value += ")"; break;
        case MaterialValueOp::Parameter:
            value = "asfloat(inputs.Load4(parameterOffset + " + std::to_string(16u + node.index * 16u) + "u))"; break;
        case MaterialValueOp::UV: value = "float4(uv,0.0,0.0)"; break;
        case MaterialValueOp::Alpha: value = "float4(baseAlpha)"; break;
        case MaterialValueOp::Add: value = a + "+" + b; break;
        case MaterialValueOp::Multiply: value = a + "*" + b; break;
        case MaterialValueOp::Dot: value = "float4(dot(" + a + "," + b + "))"; break;
        case MaterialValueOp::Lerp: value = "lerp(" + a + "," + b + ",saturate(" + c + "))"; break;
        case MaterialValueOp::Sin: value = "sin(" + a + ")"; break;
        case MaterialValueOp::Fract: value = "frac(" + a + ")"; break;
        case MaterialValueOp::Abs: value = "abs(" + a + ")"; break;
        case MaterialValueOp::Saturate: value = "saturate(" + a + ")"; break;
        case MaterialValueOp::Clamp: value = "min(max(" + a + "," + b + ")," + c + ")"; break;
        case MaterialValueOp::Normalize:
        case MaterialValueOp::NormalMap: {
            const auto normal = "n" + std::to_string(index);
            source += "    float3 " + normal + " = " + a + ".xyz" +
                (node.op == MaterialValueOp::NormalMap ? " * 2.0 - 1.0" : "") + ";\n";
            value = "float4(dot(" + normal + "," + normal + ") > 1e-20 ? normalize(" + normal + ") : float3(0,0,1),0)";
            break;
        }
        case MaterialValueOp::UVTransform:
            value = "float4(dot(" + a + ".xy," + b + ".xy)+" + b + ".z,dot(" + a + ".xy," + c + ".xy)+" + c + ".z,0,0)"; break;
        case MaterialValueOp::Swizzle:
            value = a + ".";
            for (uint32_t i = 0; i < 4; ++i) { value += "xyzw"[(node.index >> (2*i)) & 3]; }
            break;
        case MaterialValueOp::Select: value = "(" + a + ".x>0.0?" + b + ":" + c + ")"; break;
        default: throw std::runtime_error("Unsupported Coverage operation");
        }
        // Unlike Surface, the Coverage interpreter clamps every node, including
        // UV/alpha inputs. Preserve that contract before downstream operations.
        source += "    float4 v" + std::to_string(index) + " = clamp(" + value + ",-1e6,1e6);\n";
    }
    source += "    return v" + std::to_string(coverage.outputs().at("coverage")) + ".x;\n}\n";
    return source;
}
} // namespace metallic::render
