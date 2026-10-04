#include "Runtime/Render/Material/MaterialCoverageProgram.h"
#include <cmath>
#include <map>
#include <stdexcept>

namespace metallic::render {
namespace {
void require(bool value, const char* message)
{
    if (!value) { throw std::runtime_error(message); }
}

struct Compiler
{
    MaterialCoverageSlice slice;
    std::map<std::string, uint32_t> expressions;
    uint32_t visited = 0;

    uint32_t emit(const nlohmann::json& expression, uint32_t depth = 0)
    {
        require(depth <= 24 && ++visited <= 256, "Coverage expression exceeds depth/node budget (24/256)");
        const auto key = expression.dump();
        if (const auto found = expressions.find(key); found != expressions.end()) { return found->second; }
        MaterialCoverageInstruction instruction;
        const auto scalar = [](const auto& value) {
            require(value.is_number(), "Coverage constants must be numbers");
            const float number = value.template get<float>();
            require(std::isfinite(number) && std::abs(number) <= 1e6f, "Coverage constants exceed finite range");
            return number;
        };
        if (expression.is_number()) { instruction.constant.fill(scalar(expression)); }
        else if (expression.is_array()) {
            require(expression.size() == 4, "Coverage vector requires four components");
            for (uint32_t i = 0; i < 4; ++i) { instruction.constant[i] = scalar(expression[i]); }
        } else {
            require(expression.is_object() && expression.contains("op") && expression["op"].is_string(), "Invalid Coverage expression");
            const auto op = expression["op"].get<std::string>();
            if (op == "parameter") {
                require(expression.size() == 2 && expression.contains("index") && expression["index"].is_number_integer(), "Coverage parameter requires index");
                const auto index = expression["index"].get<int64_t>();
                require(index >= 0 && index < 4, "Coverage parameter index must be 0..3");
                instruction.operation = {1, static_cast<uint32_t>(index), 0, 0};
                slice.parameterMask |= 1u << index;
            } else if (op == "uv" || op == "alpha") {
                require(expression.size() == 1, "Coverage input takes no arguments");
                instruction.operation[0] = op == "uv" ? 2 : 3;
                slice.usesBaseAlpha |= op == "alpha";
            } else {
                const std::map<std::string, std::pair<uint32_t, uint32_t>> operations{
                    {"add", {4, 2}}, {"mul", {5, 2}}, {"dot", {6, 2}}, {"mix", {7, 3}},
                    {"sin", {8, 1}}, {"fract", {9, 1}}, {"abs", {10, 1}}, {"saturate", {11, 1}}};
                const auto found = operations.find(op);
                require(found != operations.end(), "Unsupported Coverage op; only UV, alpha, shared parameters and read-only arithmetic are allowed");
                const auto [opcode, arity] = found->second;
                require(expression.size() == 2 && expression.contains("args") && expression["args"].is_array() &&
                    expression["args"].size() == arity, "Wrong Coverage arguments");
                instruction.operation[0] = opcode;
                for (uint32_t i = 0; i < arity; ++i) { instruction.operation[i + 1] = emit(expression["args"][i], depth + 1); }
            }
        }
        require(slice.instructions.size() < 64, "Coverage slice exceeds 64 unique instructions");
        const auto index = static_cast<uint32_t>(slice.instructions.size());
        slice.instructions.push_back(instruction);
        expressions.emplace(key, index);
        return index;
    }
};
} // namespace

MaterialCoverageSlice compileMaterialCoverageSlice(const nlohmann::json& expression)
{
    Compiler compiler;
    compiler.emit(expression);
    return std::move(compiler.slice);
}
} // namespace metallic::render
