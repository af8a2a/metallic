#pragma once
#include <volk.h>
#include <string>
#include <vector>
namespace metallic::render::vulkan {
bool nvPerfInstanceExtensions(std::vector<const char*>& extensions, uint32_t apiVersion, std::string& error);
bool nvPerfDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice,
    std::vector<const char*>& extensions, std::string& error);
} // namespace metallic::render::vulkan
