#pragma once

#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/ComputeProgram.h"

namespace metallic::render {

struct DebugProbePush {
    uint32_t byteOffset, stride, fieldOffset, count;
    uint32_t scalarType, operation, predicate, lowerBits, upperBits, groupCount;
    uint32_t bitOffset, bitMask, scale;
};
static_assert(sizeof(DebugProbePush) == 52 && offsetof(DebugProbePush, groupCount) == 36);

struct PreparedDebugProbe {
    const DebugResourceBinding* source = nullptr; // Callback-local borrow only.
    DebugProbePush push{};
    uint64_t scanBytes = 0;
    debug::DebugValue metadata;
};

debug::DebugResult<std::vector<PreparedDebugProbe>> prepareDebugProbes(
    const debug::DebugValue& specification, std::span<const DebugResourceBinding> resources,
    const std::unordered_map<std::string, debug::DebugTypeDesc>& layouts,
    const debug::DebugEvidenceStamp& evidence, uint64_t scanBudget);
Result initializeDebugProbe(Device& device, ComputeProgram& program, std::string& log);
Result recordDebugProbe(CommandBuffer& commands, ComputeProgram& program,
    const PreparedDebugProbe& probe, Buffer& output, Buffer& readback);

} // namespace metallic::render
