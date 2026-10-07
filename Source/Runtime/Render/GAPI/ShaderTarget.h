#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace metallic::render {

// Deterministic, device-independent finalization of compiler output. Cache keys
// MUST include both id and revision. Only successful output may be persisted;
// cache hits already contain finalized code and must not run this pass again.
// Device-specific compiler inputs belong to the request/cache identity, never
// to this finalization pass. Feature negotiation remains a device operation.
struct ShaderTargetPostprocess {
    const char* id;
    uint32_t revision;
    bool (*finalizeSpirv)(std::vector<uint32_t>& code, std::string& diagnostics);
};

// TO-REMOVE(VVL payload-size): remove the literal-stride compiler policy when native
// task/mesh payload validation accepts device-independent OpConstantSizeOfEXT
// and unified stride expressions. See Documentation/NativeDescriptorHeapStrideWorkaround.md.
struct DescriptorHeapShaderStrides {
    uint32_t resource = 0;
    uint32_t sampler = 0;

    bool operator==(const DescriptorHeapShaderStrides&) const = default;
};

// Instance-only query for native warmup before a logical device exists. The
// renderer uses a single GPU; shader module creation rejects an incompatible ABI.
// This query does not replace the global Vulkan dispatch state.
bool queryVulkanDescriptorHeapShaderStrides(
    DescriptorHeapShaderStrides& strides, std::string& diagnostics);

// Immutable Vulkan SPIR-V target policy. Bump its revision whenever emitted
// normalized code or accepted-input rules change, independently of Slang.
const ShaderTargetPostprocess& vulkanSpirvTarget();

} // namespace metallic::render
