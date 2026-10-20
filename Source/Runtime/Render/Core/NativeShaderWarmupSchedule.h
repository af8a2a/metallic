#pragma once

#include <cstddef>
#include <cstdint>

namespace metallic::render::detail {

struct NativeShaderWarmupPriority {
    uint64_t historicalNanoseconds = 0;
    size_t spirvWords = 0;
};

inline bool nativeShaderWarmupPriorityBefore(
    const NativeShaderWarmupPriority& left, const NativeShaderWarmupPriority& right)
{
    const bool leftKnown = left.historicalNanoseconds != 0;
    const bool rightKnown = right.historicalNanoseconds != 0;
    if (leftKnown != rightKnown) { return leftKnown; }
    if (left.historicalNanoseconds != right.historicalNanoseconds) {
        return left.historicalNanoseconds > right.historicalNanoseconds;
    }
    return left.spirvWords > right.spirvWords;
}

} // namespace metallic::render::detail
