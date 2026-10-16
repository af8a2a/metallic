#pragma once

#include <cstdint>

namespace metallic::render {

// Pipeline ABI shared by the DLSS alpha export owner and native warmup.
struct StreamlineDLSSAlphaUserPush {
    uint32_t outputImage = 0;
};

} // namespace metallic::render
