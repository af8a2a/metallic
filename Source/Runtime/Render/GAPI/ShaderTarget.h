#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace metallic::render {

// Deterministic, device-independent finalization of compiler output. Cache keys
// MUST include both id and revision. Only successful output may be persisted;
// cache hits already contain finalized code and must not run this pass again.
// Device limits, descriptor sizes and feature negotiation belong to module
// creation and must never enter this process-wide compiler cache contract.
struct ShaderTargetPostprocess {
    const char* id;
    uint32_t revision;
    bool (*finalizeSpirv)(std::vector<uint32_t>& code, std::string& diagnostics);
};

// Immutable Vulkan SPIR-V target policy. Bump its revision whenever emitted
// normalized code or accepted-input rules change, independently of Slang.
const ShaderTargetPostprocess& vulkanSpirvTarget();

} // namespace metallic::render
