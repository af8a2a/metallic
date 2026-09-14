#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <span>
#include <volk.h>

namespace metallic::render::vulkan {

// WSI advertises pairs. Never combine a format from one entry with another
// entry's color space, or silently fall back to an unknown HDR encoding.
inline bool selectSurfaceFormat(std::span<const VkSurfaceFormatKHR> available,
    VkFormat requestedSdrFormat, DisplayOutputMode requestedMode, bool allowSdrFallback,
    VkSurfaceFormatKHR& selected, DisplayOutputMode& actualMode)
{
    auto findPair = [&](VkFormat format, VkColorSpaceKHR colorSpace) {
        for (const auto& candidate : available) {
            if (candidate.colorSpace == colorSpace &&
                (candidate.format == format || candidate.format == VK_FORMAT_UNDEFINED)) {
                selected = {format, colorSpace};
                return true;
            }
        }
        return false;
    };
    if (requestedMode == DisplayOutputMode::HdrScRgb) {
        if (findPair(VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)) {
            actualMode = DisplayOutputMode::HdrScRgb;
            return true;
        }
        if (!allowSdrFallback) { return false; }
    } else if (requestedMode != DisplayOutputMode::Sdr) {
        return false;
    }
    const VkFormat sdrFormats[] = {requestedSdrFormat, VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB};
    for (VkFormat format : sdrFormats) {
        if (format != VK_FORMAT_B8G8R8A8_UNORM && format != VK_FORMAT_R8G8B8A8_UNORM &&
            format != VK_FORMAT_B8G8R8A8_SRGB && format != VK_FORMAT_R8G8B8A8_SRGB) { continue; }
        if (findPair(format, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)) {
            actualMode = DisplayOutputMode::Sdr;
            return true;
        }
    }
    return false;
}

} // namespace metallic::render::vulkan
