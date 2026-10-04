#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string_view>

namespace metallic::render {

enum class SceneWorkingColorSpace : uint8_t { LinearRec709, ACEScg };
enum class ColorPrimaries : uint8_t { Rec709, ACESAP1, ACESAP0, Rec2020 };
enum class WhitePoint : uint8_t { D65, ACESD60 };
enum class TransferFunction : uint8_t { Linear, sRGB };
enum class TextureSemantic : uint8_t { Color, Data };

struct ColorSpaceDesc {
    ColorPrimaries primaries = ColorPrimaries::Rec709;
    WhitePoint whitePoint = WhitePoint::D65;
    TransferFunction transfer = TransferFunction::Linear;
    bool operator==(const ColorSpaceDesc&) const = default;
};

struct TextureColorMetadata {
    TextureSemantic semantic = TextureSemantic::Data;
    ColorSpaceDesc source{};
    bool operator==(const TextureColorMetadata&) const = default;
};

inline constexpr ColorSpaceDesc kLinearRec709{};
inline constexpr ColorSpaceDesc ksRGB{ColorPrimaries::Rec709, WhitePoint::D65, TransferFunction::sRGB};
inline constexpr ColorSpaceDesc kACEScg{ColorPrimaries::ACESAP1, WhitePoint::ACESD60, TransferFunction::Linear};
inline constexpr TextureColorMetadata ksRGBColorTexture{TextureSemantic::Color, ksRGB};

// Immutable process-wide selection: set before any asset upload or shader warmup.
// Changing basis requires restarting; all shader cache keys include this value.
inline SceneWorkingColorSpace sceneWorkingColorSpace()
{
    static const auto space = [] {
        const char* value = std::getenv("METALLIC_WORKING_COLOR_SPACE");
        if (value == nullptr || std::string_view(value) == "acescg") { return SceneWorkingColorSpace::ACEScg; }
        if (std::string_view(value) == "rec709") { return SceneWorkingColorSpace::LinearRec709; }
        throw std::invalid_argument("METALLIC_WORKING_COLOR_SPACE must be acescg or rec709");
    }();
    return space;
}

inline bool parseColorSpace(std::string_view name, ColorSpaceDesc& space)
{
    if (name == "lin_rec709") { space = kLinearRec709; }
    else if (name == "srgb") { space = ksRGB; }
    else if (name == "acescg" || name == "lin_ap1_scene") { space = kACEScg; }
    else if (name == "lin_ap0") { space = {ColorPrimaries::ACESAP0, WhitePoint::ACESD60, TransferFunction::Linear}; }
    else if (name == "lin_rec2020") { space = {ColorPrimaries::Rec2020, WhitePoint::D65, TransferFunction::Linear}; }
    else { return false; }
    return true;
}
inline std::string_view colorSpaceName(ColorSpaceDesc space)
{
    if (space == ksRGB) { return "srgb"; }
    if (space == kACEScg) { return "acescg"; }
    if (space == kLinearRec709) { return "lin_rec709"; }
    if (space == ColorSpaceDesc{ColorPrimaries::ACESAP0, WhitePoint::ACESD60, TransferFunction::Linear}) { return "lin_ap0"; }
    if (space == ColorSpaceDesc{ColorPrimaries::Rec2020, WhitePoint::D65, TransferFunction::Linear}) { return "lin_rec2020"; }
    throw std::invalid_argument("Unsupported source color space");
}
inline uint32_t textureColorFlags(TextureColorMetadata metadata)
{
    if (metadata.semantic == TextureSemantic::Data) { return 5u; }
    // GPU flags encode only these supported primary/white/transfer combinations.
    (void)colorSpaceName(metadata.source);
    return (metadata.source.transfer == TransferFunction::Linear ? 1u : 0u) |
        (static_cast<uint32_t>(metadata.source.primaries) << 3u);
}

namespace color {
using RGB = std::array<float, 3>;
using Matrix = std::array<double, 9>;
#define METALLIC_COLOR_MATRIX(name, ...) inline constexpr Matrix name{__VA_ARGS__};
#include "ColorSpaceMatrices.h"
#undef METALLIC_COLOR_MATRIX

inline RGB transform(const Matrix& m, RGB c)
{
    return {float(m[0]*c[0]+m[1]*c[1]+m[2]*c[2]),
        float(m[3]*c[0]+m[4]*c[1]+m[5]*c[2]),
        float(m[6]*c[0]+m[7]*c[1]+m[8]*c[2])};
}
inline float decodeSRGB(float c)
{
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
inline RGB rec709ToACEScg(RGB c)
{
    return transform(kXYZToAP1, transform(kD65ToD60, transform(kRec709ToXYZ, c)));
}
inline RGB ACEScgToRec709(RGB c)
{
    return transform(kXYZToRec709, transform(kD60ToD65, transform(kAP1ToXYZ, c)));
}
inline RGB fromLinearRec709(RGB c, SceneWorkingColorSpace space = sceneWorkingColorSpace())
{
    return space == SceneWorkingColorSpace::ACEScg ? rec709ToACEScg(c) : c;
}
inline RGB toLinearRec709(RGB c, SceneWorkingColorSpace space = sceneWorkingColorSpace())
{
    return space == SceneWorkingColorSpace::ACEScg ? ACEScgToRec709(c) : c;
}
inline RGB fromSource(RGB c, ColorSpaceDesc source, SceneWorkingColorSpace space = sceneWorkingColorSpace())
{
    if (source.transfer == TransferFunction::sRGB) {
        for (float& v : c) { v = decodeSRGB(v); }
    }
    const auto target = space == SceneWorkingColorSpace::ACEScg ? kACEScg : kLinearRec709;
    if (source.primaries == target.primaries && source.whitePoint == target.whitePoint) { return c; }
    const Matrix& input = source.primaries == ColorPrimaries::ACESAP1 ? kAP1ToXYZ :
        source.primaries == ColorPrimaries::ACESAP0 ? kAP0ToXYZ :
        source.primaries == ColorPrimaries::Rec2020 ? kRec2020ToXYZ : kRec709ToXYZ;
    RGB xyz = transform(input, c);
    if (source.whitePoint != target.whitePoint) {
        xyz = transform(target.whitePoint == WhitePoint::ACESD60 ? kD65ToD60 : kD60ToD65, xyz);
    }
    return transform(space == SceneWorkingColorSpace::ACEScg ? kXYZToAP1 : kXYZToRec709, xyz);
}
inline float luminance(RGB c, SceneWorkingColorSpace space = sceneWorkingColorSpace())
{
    // Retain historical rounded weights in compatibility mode.
    return space == SceneWorkingColorSpace::ACEScg
        ? float(kAP1ToXYZ[3]*c[0]+kAP1ToXYZ[4]*c[1]+kAP1ToXYZ[5]*c[2])
        : 0.2126f*c[0]+0.7152f*c[1]+0.0722f*c[2];
}
} // namespace color
} // namespace metallic::render
