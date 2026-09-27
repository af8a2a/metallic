#pragma once
#include "Evidence.h"

namespace metallic::tests::bench {
Json generateBufferSequence(uint64_t seed, uint32_t iteration, bool injected = false);
void validateBufferSequence(const Json& sequence);
int shrinkBufferSequence(const std::filesystem::path& executable, const std::filesystem::path& original,
    const Json& input, const std::filesystem::path& output);
} // namespace metallic::tests::bench
