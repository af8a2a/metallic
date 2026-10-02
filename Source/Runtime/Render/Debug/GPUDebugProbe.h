#pragma once

#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/DebugProbeParameters.h"

namespace metallic::render {

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
Result<> initializeDebugProbe(Device& device, ComputeKernel& program, std::string& log);
Result<> recordDebugProbe(Device& device, CommandBuffer& commands, ComputeKernel& program,
    const PreparedDebugProbe& probe, Buffer& output, Buffer& readback);

} // namespace metallic::render
