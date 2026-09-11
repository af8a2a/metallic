#pragma once

#include <volk.h>

// KHR uses acceleration structures, not VkMicromapEXT. Keep the small set of
// KHR declarations here until the minimum Vulkan SDK includes this extension.
// Values and layouts follow Khronos Vulkan-Headers (Apache-2.0 OR MIT).
#ifndef VK_KHR_opacity_micromap
#define VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME "VK_KHR_opacity_micromap"
constexpr auto VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_KHR = static_cast<VkStructureType>(1000623000);
constexpr auto VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_KHR = static_cast<VkStructureType>(1000623001);
constexpr auto VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MICROMAP_DATA_KHR = static_cast<VkStructureType>(1000623002);
constexpr auto VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_KHR = static_cast<VkStructureType>(1000623003);
constexpr auto VK_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_KHR = static_cast<VkAccelerationStructureTypeKHR>(1000623000);
constexpr auto VK_GEOMETRY_TYPE_MICROMAP_KHR = static_cast<VkGeometryTypeKHR>(1000623000);
using VkOpacityMicromapFormatKHR = VkOpacityMicromapFormatEXT;
constexpr auto VK_OPACITY_MICROMAP_FORMAT_2_STATE_KHR = VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT;
constexpr auto VK_OPACITY_MICROMAP_FORMAT_4_STATE_KHR = VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT;
using VkMicromapUsageKHR = VkMicromapUsageEXT;
struct VkPhysicalDeviceOpacityMicromapFeaturesKHR {
    VkStructureType sType;
    void* pNext;
    VkBool32 micromap;
};
struct VkPhysicalDeviceOpacityMicromapPropertiesKHR {
    VkStructureType sType;
    void* pNext;
    uint32_t maxOpacity2StateSubdivisionLevel;
    uint32_t maxOpacity4StateSubdivisionLevel;
    uint32_t maxOpacityLossy4StateSubdivisionLevel;
    uint64_t maxMicromapTriangles;
};
struct VkAccelerationStructureGeometryMicromapDataKHR {
    VkStructureType sType;
    const void* pNext;
    uint32_t usageCountsCount;
    const VkMicromapUsageKHR* pUsageCounts;
    const VkMicromapUsageKHR* const* ppUsageCounts;
    VkDeviceAddress data;
    VkDeviceAddress triangleArray;
    VkDeviceSize triangleArrayStride;
};
struct VkAccelerationStructureTrianglesOpacityMicromapKHR {
    VkStructureType sType;
    void* pNext;
    VkIndexType indexType;
    VkDeviceAddress indexBuffer;
    VkDeviceSize indexStride;
    uint32_t baseTriangle;
    VkAccelerationStructureKHR micromap;
};
#endif
