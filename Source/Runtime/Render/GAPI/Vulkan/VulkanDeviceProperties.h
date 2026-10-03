#pragma once

#include <volk.h>

namespace metallic::render::vulkan {

namespace negotiation { struct VulkanDeviceFeatureSelection; }

// Immutable after device creation. All pNext pointers are cleared before publication.
struct VulkanDeviceProperties {
    VkPhysicalDeviceProperties core{};
    VkPhysicalDeviceDriverProperties driver{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceMaintenance4Properties maintenance4{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructure{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceDescriptorHeapPropertiesEXT descriptorHeap{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT};
    VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT generatedCommands{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT};
    VkPhysicalDeviceOpacityMicromapPropertiesKHR opacityMicromap{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_KHR};
#ifdef VK_NV_cluster_acceleration_structure
    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV cluster{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
#endif
};

VulkanDeviceProperties queryDeviceProperties(VkPhysicalDevice physicalDevice,
    const negotiation::VulkanDeviceFeatureSelection& features, PFN_vkGetPhysicalDeviceProperties2 query);

} // namespace metallic::render::vulkan
