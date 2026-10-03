#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include <volk.h>

namespace metallic::render::vulkan {

// Pure translation, also usable without a live device.
inline Result<> mapVkResult(VkResult result)
{
    switch (result) {
    case VK_SUCCESS:
        return {};
    case VK_ERROR_OUT_OF_HOST_MEMORY:
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return makeError(Error::OutOfMemory);
    case VK_ERROR_DEVICE_LOST:
        return makeError(Error::DeviceLost);
    case VK_ERROR_OUT_OF_DATE_KHR:
    case VK_ERROR_SURFACE_LOST_KHR:
        return makeError(Error::OutOfDate);
    case VK_ERROR_EXTENSION_NOT_PRESENT:
    case VK_ERROR_FEATURE_NOT_PRESENT:
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
    case VK_ERROR_INCOMPATIBLE_DRIVER:
    case VK_ERROR_LAYER_NOT_PRESENT:
        return makeError(Error::Unsupported);
    default:
        return makeError(Error::Failure);
    }
}

// Translation plus the shared device-lost diagnostic notification.
Result<> resultFromVk(VkResult result);

} // namespace metallic::render::vulkan
