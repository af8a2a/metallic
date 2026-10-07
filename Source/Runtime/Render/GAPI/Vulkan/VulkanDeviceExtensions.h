#pragma once

#include "Runtime/Render/GAPI/RHI.h"

namespace metallic::render::vulkan {

class ShaderPrintf;

// A value stored in DeviceDesc::backendExtensions. No Vulkan SDK headers are
// required by callers configuring these options.
struct VulkanDeviceExtensions {
    // Unsupported devices retain optimal image layouts.
    bool preferUnifiedImageLayouts = true;
    // Implies validation and excludes NvPerf. The capture must outlive Device.
    ShaderPrintf* shaderPrintf = nullptr;
};

// Configuration helpers preserve value semantics when a DeviceDesc is copied.
// An unrelated payload is a caller error (std::bad_any_cast); createDevice itself
// rejects it with Error::InvalidArgument before any backend initialization.
inline const VulkanDeviceExtensions& deviceExtensions(const DeviceDesc& desc)
{
    static const VulkanDeviceExtensions defaults;
    return desc.backendExtensions.has_value()
        ? std::any_cast<const VulkanDeviceExtensions&>(desc.backendExtensions) : defaults;
}

inline VulkanDeviceExtensions& deviceExtensions(DeviceDesc& desc)
{
    if (!desc.backendExtensions.has_value()) { desc.backendExtensions.emplace<VulkanDeviceExtensions>(); }
    return std::any_cast<VulkanDeviceExtensions&>(desc.backendExtensions);
}

// Enabled backend/integration capabilities, separate from portable RHI features.
struct VulkanDeviceCapabilities {
    bool unifiedImageLayouts = false;
    bool pushDescriptor = false;
    bool streamline = false;
    bool streamlineDlssSr = false;
    bool streamlineDlssRr = false;
    bool aftermath = false;
};

[[nodiscard]] VulkanDeviceCapabilities deviceCapabilities(const Device& device);

} // namespace metallic::render::vulkan
