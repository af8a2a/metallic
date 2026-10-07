#pragma once

#include "Runtime/Render/GAPI/ShaderTarget.h"
#include "SpirvWalker.h"

#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace metallic::render::vulkan {

// TO-REMOVE(VVL payload-size): read-only check of the temporary literal-stride ABI.
// The caller must recompile for a different stride pair; module creation never
// repairs its code.
inline bool validateNativeDescriptorHeapStrides(std::span<const uint32_t> code,
    DescriptorHeapShaderStrides expected, std::string& diagnostics)
{
    const SpirvWalker walker(code);
    const auto fail = [&](const char* reason) {
        diagnostics = std::string("Native descriptor heap shader ABI: ") + reason +
            ". Recompile this shader with resource/sampler strides " +
            std::to_string(expected.resource) + "/" + std::to_string(expected.sampler) +
            " (-spirv-resource-heap-stride " + std::to_string(expected.resource) +
            " -spirv-sampler-heap-stride " + std::to_string(expected.sampler) + ").";
        return false;
    };
    if (!walker.valid()) { return fail(walker.error()); }
    constexpr uint32_t kCapability = 17, kDescriptorHeap = 5128;
    bool native = false;
    for (const auto& [offset, count, op] : walker.instructions()) {
        if (op == kCapability && count >= 2 && code[offset + 1] == kDescriptorHeap) {
            if (count != 2) { return fail("invalid DescriptorHeapEXT capability"); }
            native = true;
        }
    }
    if (!native) { diagnostics.clear(); return true; }
    if (!expected.resource || !expected.sampler) { return fail("device descriptor strides are unavailable"); }

    constexpr uint32_t kTypeImage = 25, kTypeSampler = 26, kTypeArray = 28, kTypeRuntimeArray = 29;
    constexpr uint32_t kTypeBuffer = 5115, kConstantSizeOf = 5129;
    constexpr uint32_t kDecorate = 71, kDecorateId = 332;
    constexpr uint32_t kArrayStride = 6, kArrayStrideId = 5124;
    std::unordered_map<uint32_t, uint32_t> opaqueStrides, arrayElements, literalStrides;
    std::unordered_set<uint32_t> idStrides;
    for (const auto& [offset, count, op] : walker.instructions()) {
        if (op == kConstantSizeOf) { return fail("opaque descriptor-size expressions are unsupported"); }
        if (op == kTypeImage || op == kTypeSampler || op == kTypeBuffer) {
            if (count < (op == kTypeImage ? 9u : op == kTypeBuffer ? 3u : 2u)) {
                return fail("invalid opaque descriptor type");
            }
            opaqueStrides[code[offset + 1]] = op == kTypeSampler ? expected.sampler : expected.resource;
        } else if (op == kTypeRuntimeArray || op == kTypeArray) {
            if (count != (op == kTypeArray ? 4u : 3u)) { return fail("invalid descriptor array"); }
            arrayElements[code[offset + 1]] = code[offset + 2];
        } else if (op == kDecorate || op == kDecorateId) {
            if (count < 3) { return fail("invalid descriptor decoration"); }
            const uint32_t decoration = code[offset + 2];
            if (decoration != kArrayStride && decoration != kArrayStrideId) { continue; }
            if (count != 4) { return fail("invalid ArrayStride decoration"); }
            if (op == kDecorateId || decoration == kArrayStrideId) {
                idStrides.insert(code[offset + 1]);
                continue;
            }
            const auto [position, inserted] = literalStrides.emplace(code[offset + 1], code[offset + 3]);
            if (!inserted && position->second != code[offset + 3]) {
                return fail("conflicting ArrayStride decorations");
            }
        }
    }
    for (const auto& [array, element] : arrayElements) {
        const auto opaque = opaqueStrides.find(element);
        if (opaque == opaqueStrides.end()) { continue; }
        if (idStrides.contains(array)) { return fail("descriptor ArrayStride must be a literal"); }
        const auto literal = literalStrides.find(array);
        if (literal == literalStrides.end() || literal->second != opaque->second) {
            return fail("literal descriptor ArrayStride does not match the current device");
        }
    }
    diagnostics.clear();
    return true;
}

} // namespace metallic::render::vulkan
