#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace metallic::render {

class Buffer;

inline constexpr uint32_t kTextureDemandEnabled = 0x80000000u;
inline constexpr uint32_t kTextureResidentFirstMipMask = 0x7fffffffu;

// Positive dimensions: nonnegative terms are exact power-of-two LODs. Other
// dimensions store negative area so the shader retains its original GPU log2.
inline constexpr float makeTextureLodTerm(uint32_t width, uint32_t height)
{
    if (std::has_single_bit(width) && std::has_single_bit(height)) {
        return 0.5f * float((std::bit_width(width) - 1) + (std::bit_width(height) - 1));
    }
    return -(float(width) * float(height));
}

// Immutable for a submitted frame and matched to its sampled-image generation.
struct TextureSamplingMetadata
{
    float samplingFloor = 0.0f;
    // Encoded by makeTextureLodTerm; decode negative values with GPU log2.
    float residentLodTerm = 0.0f;
    uint32_t residentWidth = 1;
    uint32_t residentHeight = 1;
    uint32_t residentMipCount = 1;
    uint32_t residentFirstMipAndFlags = 0;
};
static_assert(sizeof(TextureSamplingMetadata) == 24);
static_assert(offsetof(TextureSamplingMetadata, residentFirstMipAndFlags) == 20);
static_assert(std::is_trivially_copyable_v<TextureSamplingMetadata>);

struct TextureFeedbackResources
{
    Buffer* samplingMetadata = nullptr;
    Buffer* demandMip = nullptr;
    Buffer* demandHits = nullptr;
    uint32_t textureCount = 0;
};

} // namespace metallic::render
