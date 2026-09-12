#pragma once

#include <algorithm>
#include <cstring>
#include <span>
#include <string_view>
#include <vector>
#include <cstdint>

namespace metallic::render::vulkan {

// Declare the device's OMM shader support: the legacy EXT capability, or the
// KHR execution mode (revision 4) that older Slang releases cannot emit.
// Patch after compiler/cache lookup and hash/register the final device binary.
inline bool enableOpacityMicromapSpirv(
    std::span<const uint32_t> code, std::vector<uint32_t>& output, bool useExt = false)
{
    output.assign(code.begin(), code.end());
    if (code.size() < 5 || code[0] != 0x07230203u) {
        return false;
    }
    constexpr uint32_t kRayQuery = 4472;
    const uint32_t kOpacityCapability = useExt ? 5381 : 6032;
    const std::string_view extension = useExt ? "SPV_EXT_opacity_micromap" : "SPV_KHR_opacity_micromap";
    constexpr uint32_t kOpacityMode = 6031;
    bool rayQuery = false;
    bool hasCapability = false;
    bool hasExtension = false;
    uint32_t boolType = 0;
    size_t capabilitiesEnd = 5, extensionsEnd = 5, modesEnd = 5, functionsBegin = code.size();
    std::vector<uint32_t> entryPoints, configuredEntries;
    for (size_t offset = 5; offset < code.size();) {
        const uint32_t count = code[offset] >> 16;
        const uint32_t op = code[offset] & 0xffffu;
        if (count == 0 || count > code.size() - offset) {
            return false;
        }
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
        offset += count;
    }
    if (!rayQuery) {
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
            output.insert(output.begin() + std::max(extensionsEnd, capabilitiesEnd), instruction.begin(), instruction.end());
        }
        if (!hasCapability) {
            output.insert(output.begin() + capabilitiesEnd, {(2u << 16) | 17u, kOpacityCapability});
        }
        return true;
    }
    std::erase_if(entryPoints, [&](uint32_t entry) {
        return std::find(configuredEntries.begin(), configuredEntries.end(), entry) != configuredEntries.end();
    });
    if (entryPoints.empty()) {
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
    output.clear();
    output.insert(output.end(), code.begin(), code.begin() + 5);
    output[3] = nextId;
    for (size_t offset = 5; offset < code.size();) {
        if (offset == capabilitiesEnd && !hasCapability) {
            output.insert(output.end(), {(2u << 16) | 17u, kOpacityCapability});
        }
        if (offset == extensionsEnd && !hasExtension) {
            constexpr char kExtension[] = "SPV_KHR_opacity_micromap";
            constexpr size_t kWords = (sizeof(kExtension) + 3) / 4;
            output.push_back((uint32_t(kWords + 1) << 16) | 10u);
            const size_t begin = output.size();
            output.resize(begin + kWords, 0);
            std::memcpy(output.data() + begin, kExtension, sizeof(kExtension));
        }
        if (offset == modesEnd) {
            for (uint32_t entry : entryPoints) {
                output.insert(output.end(), {(4u << 16) | 331u, entry, kOpacityMode, enabledId});
            }
        }
        if (offset == functionsBegin) {
            if (needsBool) {
                output.insert(output.end(), {(2u << 16) | 20u, boolType});
            }
            output.insert(output.end(), {(3u << 16) | 41u, boolType, enabledId}); // OpConstantTrue
        }
        const uint32_t count = code[offset] >> 16;
        output.insert(output.end(), code.begin() + offset, code.begin() + offset + count);
        offset += count;
    }
    return true;
}

} // namespace metallic::render::vulkan
