#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace metallic::tools {

// Structural counts from the final module sent to the driver. These are not
// compiler cost estimates or executable instruction counts.
struct SPIRVStatistics {
    uint32_t idBound = 0;
    uint64_t instructions = 0, functions = 0, functionCalls = 0;
    uint64_t basicBlocks = 0, loopMerges = 0, selectionMerges = 0;
    uint64_t branches = 0, conditionalBranches = 0, switches = 0, phiNodes = 0;
    uint64_t maxFunctionInstructions = 0, maxFunctionBasicBlocks = 0;
    std::vector<uint64_t> largestFunctionInstructions;
};

inline SPIRVStatistics inspectSPIRV(std::span<const uint32_t> words)
{
    if (words.size() < 5 || words[0] != 0x07230203 || !words[3] || words[4]) {
        throw std::runtime_error("Invalid SPIR-V header for structural statistics");
    }
    SPIRVStatistics stats{.idBound = words[3]};
    bool inFunction = false;
    uint64_t functionInstructions = 0, functionBlocks = 0;
    for (size_t offset = 5; offset < words.size();) {
        const uint32_t count = words[offset] >> 16;
        const uint32_t opcode = words[offset] & 65535;
        if (!count || count > words.size() - offset) {
            throw std::runtime_error("Invalid SPIR-V instruction for structural statistics");
        }
        ++stats.instructions;
        if (opcode == 54) { // OpFunction
            if (inFunction) { throw std::runtime_error("Nested SPIR-V function"); }
            inFunction = true;
            functionInstructions = functionBlocks = 0;
            ++stats.functions;
        }
        if (inFunction) { ++functionInstructions; }
        switch (opcode) {
        case 57: ++stats.functionCalls; break; // OpFunctionCall
        case 245: ++stats.phiNodes; break;
        case 246: ++stats.loopMerges; break;
        case 247: ++stats.selectionMerges; break;
        case 248: ++stats.basicBlocks; if (inFunction) { ++functionBlocks; } break;
        case 249: ++stats.branches; break;
        case 250: ++stats.conditionalBranches; break;
        case 251: ++stats.switches; break;
        case 56: // OpFunctionEnd
            if (!inFunction) { throw std::runtime_error("Unexpected SPIR-V function end"); }
            stats.maxFunctionInstructions = std::max(stats.maxFunctionInstructions, functionInstructions);
            stats.maxFunctionBasicBlocks = std::max(stats.maxFunctionBasicBlocks, functionBlocks);
            stats.largestFunctionInstructions.push_back(functionInstructions);
            inFunction = false;
            break;
        default: break;
        }
        offset += count;
    }
    if (inFunction) { throw std::runtime_error("Unterminated SPIR-V function"); }
    std::sort(stats.largestFunctionInstructions.begin(), stats.largestFunctionInstructions.end(), std::greater<>());
    if (stats.largestFunctionInstructions.size() > 10) { stats.largestFunctionInstructions.resize(10); }
    return stats;
}

} // namespace metallic::tools
