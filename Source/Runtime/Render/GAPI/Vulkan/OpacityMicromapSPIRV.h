#pragma once

#include "SpirvWalker.h"

#include <algorithm>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>
#include <utility>
#include <cstdint>

namespace metallic::render::vulkan {

// Declare the device's OMM shader support: the legacy EXT capability, or the
// KHR execution mode (revision 4) that older Slang releases cannot emit.
// Patch after compiler/cache lookup and hash/register the final device binary.
inline bool enableOpacityMicromapSpirv(
    std::span<const uint32_t> code, std::vector<uint32_t>& output, bool useExt = false)
{
    const SpirvWalker walker(code);
    if (!walker.valid()) { return false; }
    // Stage edits so an aliased output cannot invalidate the source during traversal.
    std::vector<uint32_t> result(code.begin(), code.end());
    constexpr uint32_t kRayQuery = 4472;
    const uint32_t kOpacityCapability = useExt ? 5381 : 6032;
    const std::string_view extension = useExt ? "SPV_EXT_opacity_micromap" : "SPV_KHR_opacity_micromap";
    constexpr uint32_t kOpacityMode = 6031;
    bool rayQuery = false;
    bool hasCapability = false;
    bool hasExtension = false;
    uint32_t boolType = 0;
    size_t capabilitiesEnd = SpirvWalker::kHeaderWords, extensionsEnd = SpirvWalker::kHeaderWords,
        modesEnd = SpirvWalker::kHeaderWords, functionsBegin = code.size();
    std::vector<uint32_t> entryPoints, configuredEntries;
    for (const auto& [offset, count, op] : walker.instructions()) {
        if (op == 17 && count == 2) { // OpCapability
            rayQuery |= code[offset + 1] == kRayQuery;
            hasCapability |= code[offset + 1] == kOpacityCapability;
            capabilitiesEnd = offset + count;
        } else if (op == 10) { // OpExtension
            hasExtension |= (count - 1) * 4 >= extension.size() + 1 &&
                std::memcmp(code.data() + offset + 1, extension.data(), extension.size() + 1) == 0;
            extensionsEnd = offset + count;
        } else if (op == 15 && count >= 4) { // OpEntryPoint
            entryPoints.push_back(code[offset + 2]);
            modesEnd = offset + count;
        } else if (op == 16 || op == 331) { // OpExecutionMode / OpExecutionModeId
            modesEnd = offset + count;
            if (count >= 4 && code[offset + 2] == kOpacityMode) {
                configuredEntries.push_back(code[offset + 1]);
            }
        } else if (op == 20 && count == 2) { // OpTypeBool
            boolType = code[offset + 1];
        } else if (op == 54) { // OpFunction
            functionsBegin = std::min(functionsBegin, offset);
        }
    }
    if (!rayQuery) {
        output = std::move(result);
        return true;
    }
    if (useExt) {
        // The legacy compiler path recognizes OMM traversal through the EXT
        // capability. Do not emit the execution mode that requires the KHR feature.
        if (!hasExtension) {
            const size_t words = (extension.size() + 1 + 3) / 4;
            std::vector<uint32_t> instruction(words + 1, 0);
            instruction[0] = (uint32_t(words + 1) << 16) | 10u;
            std::memcpy(instruction.data() + 1, extension.data(), extension.size());
            result.insert(result.begin() + std::max(extensionsEnd, capabilitiesEnd), instruction.begin(), instruction.end());
        }
        if (!hasCapability) {
            result.insert(result.begin() + capabilitiesEnd, {(2u << 16) | 17u, kOpacityCapability});
        }
        output = std::move(result);
        return true;
    }
    std::erase_if(entryPoints, [&](uint32_t entry) {
        return std::find(configuredEntries.begin(), configuredEntries.end(), entry) != configuredEntries.end();
    });
    if (entryPoints.empty()) {
        output = std::move(result);
        return true;
    }
    if (functionsBegin == code.size() || code[3] > 0xfffffffdU) {
        return false;
    }
    uint32_t nextId = code[3];
    const bool needsBool = boolType == 0;
    if (needsBool) {
        boolType = nextId++;
    }
    const uint32_t enabledId = nextId++;
    extensionsEnd = std::max(extensionsEnd, capabilitiesEnd);
    result.clear();
    result.insert(result.end(), code.begin(), code.begin() + SpirvWalker::kHeaderWords);
    result[3] = nextId;
    for (const auto& [offset, count, op] : walker.instructions()) {
        if (offset == capabilitiesEnd && !hasCapability) {
            result.insert(result.end(), {(2u << 16) | 17u, kOpacityCapability});
        }
        if (offset == extensionsEnd && !hasExtension) {
            constexpr char kExtension[] = "SPV_KHR_opacity_micromap";
            constexpr size_t kWords = (sizeof(kExtension) + 3) / 4;
            result.push_back((uint32_t(kWords + 1) << 16) | 10u);
            const size_t begin = result.size();
            result.resize(begin + kWords, 0);
            std::memcpy(result.data() + begin, kExtension, sizeof(kExtension));
        }
        if (offset == modesEnd) {
            for (uint32_t entry : entryPoints) {
                result.insert(result.end(), {(4u << 16) | 331u, entry, kOpacityMode, enabledId});
            }
        }
        if (offset == functionsBegin) {
            if (needsBool) {
                result.insert(result.end(), {(2u << 16) | 20u, boolType});
            }
            result.insert(result.end(), {(3u << 16) | 41u, boolType, enabledId}); // OpConstantTrue
        }
        result.insert(result.end(), code.begin() + offset, code.begin() + offset + count);
    }
    output = std::move(result);
    return true;
}

} // namespace metallic::render::vulkan
