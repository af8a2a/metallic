#pragma once

#include "VulkanNative.h"

namespace metallic::render::vulkan {

VkFormat nativeFormat(Format format);
Format resourceFormat(VkFormat format);

// Backend adapters only: native queue calls share synchronization, diagnostics
// and isolation with RHI submission. No command-buffer ownership is transferred.
VkResult submitInterop(Queue& queue, std::span<const VkSubmitInfo2> submits, VkFence fence);
VkResult presentInterop(Queue& queue, const VkPresentInfoKHR& present);
VkResult waitInterop(Queue& queue);
VkResult waitInterop(Device& device);
// A second wrapper owns the same native view/allocation after the caller releases it.
Result<std::unique_ptr<TextureView>> retainInteropView(TextureView& view);

} // namespace metallic::render::vulkan
