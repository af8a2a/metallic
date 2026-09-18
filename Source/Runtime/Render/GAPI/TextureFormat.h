#pragma once
#include "Runtime/Render/GAPI/Rhi.h"

namespace metallic::render {
inline constexpr uint32_t compressedBlockBytes(Format format)
{
    switch (format) {
    case Format::Bc4Unorm:
        return 8;
    case Format::Bc5Unorm:
    case Format::Bc7Unorm:
    case Format::Bc7Srgb:
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
