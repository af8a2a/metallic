#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace metallic::render::vulkan {

// Decode framing once for the multiple passes used by SPIR-V transformations.
// Opcode-specific operands, ID bounds and capabilities belong to each transform.
// The source must remain unchanged while its instruction offsets are in use.
class SpirvWalker {
public:
    struct Instruction {
        size_t offset;
        uint32_t wordCount;
        uint32_t opcode;
    };
    static constexpr size_t kHeaderWords = 5;

    explicit SpirvWalker(std::span<const uint32_t> code)
    {
        if (code.size() < kHeaderWords || code[0] != 0x07230203u) {
            error_ = "invalid SPIR-V header";
            return;
        }
        for (size_t offset = kHeaderWords; offset < code.size();) {
            const uint32_t count = code[offset] >> 16;
            if (count == 0 || count > code.size() - offset) {
                error_ = "truncated instruction";
                instructions_.clear();
                return;
            }
            instructions_.push_back({offset, count, code[offset] & 0xffffu});
            offset += count;
        }
    }

    bool valid() const { return error_ == nullptr; }
    const char* error() const { return error_; }
    std::span<const Instruction> instructions() const { return instructions_; }

private:
    std::vector<Instruction> instructions_;
    const char* error_ = nullptr;
};

} // namespace metallic::render::vulkan
