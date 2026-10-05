#include "VulkanDeviceProperties.h"
#include "VulkanDeviceFeatures.h"

#include <limits>

namespace metallic::render::vulkan {

VulkanDeviceProperties queryDeviceProperties(VkPhysicalDevice physicalDevice,
    const negotiation::VulkanDeviceFeatureSelection& features, PFN_vkGetPhysicalDeviceProperties2 query)
{
    VulkanDeviceProperties result;
    VkPhysicalDeviceProperties2 root{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    auto append = [&root](auto& properties) {
        properties.pNext = root.pNext;
        root.pNext = &properties;
    };
    append(result.driver);
    append(result.maintenance4);
    if (features.shaderObject) { append(result.shaderObject); }
    if (features.rayTracingAccelerationStructure) { append(result.accelerationStructure); }
    if (features.bindlessDescriptorHeap) { append(result.descriptorHeap); }
    if (features.deviceGeneratedCommands) { append(result.generatedCommands); }
#ifdef VK_NV_cluster_acceleration_structure
    if (features.clusterAccelerationStructure) { append(result.cluster); }
#endif
    VkPhysicalDeviceOpacityMicromapPropertiesEXT extMicromap{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT};
    if (features.opacityMicromap) {
        if (features.opacityMicromapExt) { append(extMicromap); }
        else { append(result.opacityMicromap); }
    }
    query(physicalDevice, &root);
    result.core = root.properties;
    if (features.opacityMicromap && features.opacityMicromapExt) {
        result.opacityMicromap.maxOpacity2StateSubdivisionLevel = extMicromap.maxOpacity2StateSubdivisionLevel;
        result.opacityMicromap.maxOpacity4StateSubdivisionLevel = extMicromap.maxOpacity4StateSubdivisionLevel;
        // EXT has no equivalent property; our triangle indices/counts are uint32.
        result.opacityMicromap.maxMicromapTriangles = std::numeric_limits<uint32_t>::max();
    }
    // Do not retain links to this stack frame (including across copies or moves).
    for (auto* node = static_cast<VkBaseOutStructure*>(root.pNext); node != nullptr;) {
        auto* next = node->pNext;
        node->pNext = nullptr;
        node = next;
    }
    return result;
}

} // namespace metallic::render::vulkan
