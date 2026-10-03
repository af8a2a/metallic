#pragma once
#include "Runtime/Render/GAPI/RHI.h"
#include <limits>

namespace metallic::render {

struct FormatInfo {
    uint32_t blockExtent = 0;
    uint32_t bytesPerBlock = 0;
};

inline constexpr FormatInfo formatInfo(Format format)
{
    switch (format) {
    case Format::R8Unorm:
    case Format::R8Snorm:
    case Format::R8Uint:
    case Format::R8Sint:
        return {1, 1};
    case Format::RG8Unorm:
    case Format::RG8Snorm:
    case Format::RG8Uint:
    case Format::RG8Sint:
    case Format::BGRA4Unorm:
    case Format::R16Unorm:
    case Format::R16Snorm:
    case Format::R16Uint:
    case Format::R16Sint:
    case Format::R16Sfloat:
        return {1, 2};
    case Format::BGRA8Unorm:
    case Format::BGRA8sRGB:
    case Format::RGBA8Unorm:
    case Format::RGBA8Snorm:
    case Format::RGBA8sRGB:
    case Format::RGBA8Uint:
    case Format::RGBA8Sint:
    case Format::RG16Unorm:
    case Format::RG16Snorm:
    case Format::RG16Uint:
    case Format::RG16Sint:
    case Format::RG16Sfloat:
    case Format::R32Uint:
    case Format::R32Sint:
    case Format::R32Sfloat:
    case Format::A2B10G10R10UnormPack32:
    case Format::A2R10G10B10UintPack32:
    case Format::B10G11R11UfloatPack32:
    case Format::E5B9G9R9UfloatPack32:
    case Format::D32Sfloat:
        return {1, 4};
    case Format::RGBA16Unorm:
    case Format::RGBA16Snorm:
    case Format::RGBA16Uint:
    case Format::RGBA16Sint:
    case Format::RGBA16Sfloat:
    case Format::RG32Uint:
    case Format::RG32Sint:
    case Format::RG32Sfloat:
        return {1, 8};
    case Format::RGB32Uint:
    case Format::RGB32Sint:
    case Format::RGB32Sfloat:
        return {1, 12};
    case Format::RGBA32Uint:
    case Format::RGBA32Sint:
    case Format::RGBA32Sfloat:
        return {1, 16};
    case Format::BC4Unorm: return {4, 8};
    case Format::BC5Unorm:
    case Format::BC7Unorm:
    case Format::BC7sRGB: return {4, 16};
    case Format::Unknown:
        break;
    }
    return {};
}

inline constexpr uint32_t compressedBlockBytes(Format format)
{
    const auto info = formatInfo(format);
    return info.blockExtent > 1 ? info.bytesPerBlock : 0;
}

inline constexpr uint64_t bcMipBytes(Format format, uint32_t width, uint32_t height)
{
    return ((uint64_t(width) + 3) / 4) * ((uint64_t(height) + 3) / 4) * compressedBlockBytes(format);
}

struct TextureCopyFootprint {
    uint64_t rowBytes = 0;
    uint64_t rows = 0;
    uint64_t rowPitch = 0;
    uint64_t slicePitch = 0;
    uint64_t requiredBytes = 0; // Excludes padding after the final row.
};

// Byte pitches also support CPU source data with arbitrary padding. Native
// buffer/image copies additionally require pitches representable in texels.
inline Result<TextureCopyFootprint> textureCopyFootprint(Format format,
    uint32_t width, uint32_t height, uint64_t slices = 1,
    uint64_t rowPitch = 0, uint64_t slicePitch = 0)
{
    const auto info = formatInfo(format);
    if (!info.bytesPerBlock || !width || !height || !slices) { return makeError(Error::InvalidArgument); }
    const uint64_t rowBytes = ((uint64_t(width) + info.blockExtent - 1) / info.blockExtent) * info.bytesPerBlock;
    const uint64_t rows = (uint64_t(height) + info.blockExtent - 1) / info.blockExtent;
    if (!rowPitch) { rowPitch = rowBytes; }
    constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
    if (rowPitch < rowBytes || rowPitch > kMax / rows) { return makeError(Error::InvalidArgument); }
    const uint64_t tightSlice = rowPitch * rows;
    if (!slicePitch) { slicePitch = tightSlice; }
    if (slicePitch < tightSlice) { return makeError(Error::InvalidArgument); }
    const uint64_t lastSliceBytes = rowPitch * (rows - 1) + rowBytes;
    if (slices - 1 > (kMax - lastSliceBytes) / slicePitch) { return makeError(Error::InvalidArgument); }
    return TextureCopyFootprint{rowBytes, rows, rowPitch, slicePitch, (slices - 1) * slicePitch + lastSliceBytes};
}

} // namespace metallic::render
