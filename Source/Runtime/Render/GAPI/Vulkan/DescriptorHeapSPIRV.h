#pragma once

#include "SpirvWalker.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace metallic::render::vulkan {

// Resolve device-dependent opaque descriptor sizes before module creation.
// Slang's unified stride is Select(imageSize > bufferSize, imageSize, bufferSize).
// The local validation stack miscalculates unrelated task payload arrays when that
// expression still depends on OpConstantSizeOfEXT. Literal aligned sizes preserve the
// same heap ABI and let the normal specialization evaluator handle the Select.
// Disk compiler output stays device-independent; hash the final device binary.
inline bool specializeDescriptorHeapSizes(std::span<const uint32_t> code, std::vector<uint32_t>& output,
    uint32_t imageSize, uint32_t bufferSize, uint32_t samplerSize)
{
    const SpirvWalker walker(code);
    if (!walker.valid()) { return false; }
    constexpr uint32_t kTypeInt = 21, kTypeImage = 25, kTypeSampler = 26;
    constexpr uint32_t kTypeBuffer = 5115, kConstantSizeOf = 5129, kConstant = 43;
    std::unordered_map<uint32_t, uint32_t> sizes, widths;
    for (const auto& [offset, count, op] : walker.instructions()) {
        if (op == kTypeInt) {
            if (count != 4) { return false; }
            widths[code[offset + 1]] = code[offset + 2];
        } else if (op == kTypeImage || op == kTypeBuffer || op == kTypeSampler) {
            if (count < (op == kTypeImage ? 9u : op == kTypeBuffer ? 3u : 2u)) { return false; }
            sizes[code[offset + 1]] = op == kTypeImage ? imageSize : op == kTypeBuffer ? bufferSize : samplerSize;
        } else if (op == kConstantSizeOf && count != 4) { return false; }
    }
    std::vector<uint32_t> result(code.begin(), code.begin() + SpirvWalker::kHeaderWords);
    for (const auto& [offset, count, op] : walker.instructions()) {
        const auto size = op == kConstantSizeOf ? sizes.find(code[offset + 3]) : sizes.end();
        if (size != sizes.end()) {
            const auto width = widths.find(code[offset + 1]);
            if (!size->second || width == widths.end() || (width->second != 32 && width->second != 64)) {
                return false;
            }
            result.push_back(((width->second == 32 ? 4u : 5u) << 16) | kConstant);
            result.push_back(code[offset + 1]);
            result.push_back(code[offset + 2]);
            result.push_back(size->second);
            if (width->second == 64) { result.push_back(0); }
        } else {
            result.insert(result.end(), code.begin() + offset, code.begin() + offset + count);
        }
    }
    output = std::move(result);
    return true;
}

} // namespace metallic::render::vulkan
