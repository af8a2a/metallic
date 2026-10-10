#pragma once
#include "Runtime/Render/GAPI/Vulkan/VulkanImGuiBackend.h"

namespace metallic::render {
vulkan::ImGuiShaderServices imGuiDisplayShaders();
// Registers and retains the composition texture in the shared DR registry.
Result<> encodeEditorHDR10(vulkan::VulkanImGuiBackend& backend, Device& device,
    CommandBuffer& commands, TextureView& source, uint32_t width, uint32_t height);
} // namespace metallic::render
