#pragma once
#include "Runtime/Render/GAPI/RHI.h"

namespace metallic::render {
inline constexpr uint32_t compressedBlockBytes(Format format)
{
    switch (format) {
    case Format::BC4Unorm:
        return 8;
    case Format::BC5Unorm:
    case Format::BC7Unorm:
    case Format::BC7sRGB:
        return 16;
    default:
        return 0;
    }
}
inline constexpr uint64_t bcMipBytes(Format format, uint32_t width, uint32_t height)
{
    return ((uint64_t(width) + 3) / 4) * ((uint64_t(height) + 3) / 4) * compressedBlockBytes(format);
}
} // namespace metallic::render
