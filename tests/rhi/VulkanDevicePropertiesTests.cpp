#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceProperties.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceFeatures.h"

#include <gtest/gtest.h>
#include <set>
#include <limits>

namespace metallic::render::vulkan {
namespace {

thread_local uint32_t queryCount = 0;
thread_local std::set<VkStructureType> queriedTypes;

VKAPI_ATTR void VKAPI_CALL fakeQuery(VkPhysicalDevice, VkPhysicalDeviceProperties2* root)
{
    ++queryCount;
    queriedTypes.clear();
    root->properties.deviceID = queryCount;
    for (auto* node = static_cast<VkBaseOutStructure*>(root->pNext); node; node = node->pNext) {
        if (!queriedTypes.insert(node->sType).second) { FAIL() << "Duplicate or cyclic property chain"; }
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES) {
            reinterpret_cast<VkPhysicalDeviceMaintenance4Properties*>(node)->maxBufferSize = 4096;
        }
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT) {
            auto& properties = *reinterpret_cast<VkPhysicalDeviceShaderObjectPropertiesEXT*>(node);
            properties.shaderBinaryVersion = 42;
            properties.shaderBinaryUUID[0] = 17;
        }
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT) {
            auto& properties = *reinterpret_cast<VkPhysicalDeviceOpacityMicromapPropertiesEXT*>(node);
            properties.maxOpacity2StateSubdivisionLevel = 8;
            properties.maxOpacity4StateSubdivisionLevel = 6;
        }
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_KHR) {
            reinterpret_cast<VkPhysicalDeviceOpacityMicromapPropertiesKHR*>(node)->maxMicromapTriangles = 123;
        }
    }
}

void expectDetached(const VulkanDeviceProperties& properties)
{
    EXPECT_EQ(properties.driver.pNext, nullptr);
    EXPECT_EQ(properties.maintenance4.pNext, nullptr);
    EXPECT_EQ(properties.shaderObject.pNext, nullptr);
    EXPECT_EQ(properties.accelerationStructure.pNext, nullptr);
    EXPECT_EQ(properties.descriptorHeap.pNext, nullptr);
    EXPECT_EQ(properties.generatedCommands.pNext, nullptr);
    EXPECT_EQ(properties.opacityMicromap.pNext, nullptr);
#ifdef VK_NV_cluster_acceleration_structure
    EXPECT_EQ(properties.cluster.pNext, nullptr);
#endif
}

TEST(VulkanDeviceProperties, CoreOnlyUsesOneQueryAndSnapshotsRemainIndependent)
{
    queryCount = 0;
    const auto first = queryDeviceProperties(VK_NULL_HANDLE, {}, fakeQuery);
    EXPECT_EQ(queryCount, 1u);
    const std::set<VkStructureType> expected{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    EXPECT_EQ(queriedTypes, expected);
    EXPECT_EQ(first.maintenance4.maxBufferSize, 4096u);
    EXPECT_EQ(first.opacityMicromap.maxMicromapTriangles, 0u);
    const auto second = queryDeviceProperties(VK_NULL_HANDLE, {}, fakeQuery);
    EXPECT_EQ(first.core.deviceID, 1u);
    EXPECT_EQ(second.core.deviceID, 2u);
    expectDetached(first);
    expectDetached(second);
}

TEST(VulkanDeviceProperties, EnabledExtensionsShareOneQueryWithExclusiveMicromapRoute)
{
    for (bool useExt : {false, true}) {
        negotiation::VulkanDeviceFeatureSelection features{};
        features.shaderObject = true;
        features.rayTracingAccelerationStructure = true;
        features.bindlessDescriptorHeap = true;
        features.deviceGeneratedCommands = true;
        features.clusterAccelerationStructure = true;
        features.opacityMicromap = true;
        features.opacityMicromapExt = useExt;
        queryCount = 0;
        const auto snapshot = queryDeviceProperties(VK_NULL_HANDLE, features, fakeQuery);
        EXPECT_EQ(queryCount, 1u);
        std::set<VkStructureType> expected{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT,
            useExt ? VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT
                   : VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_KHR};
#ifdef VK_NV_cluster_acceleration_structure
        expected.insert(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV);
#endif
        EXPECT_EQ(queriedTypes, expected);
        EXPECT_EQ(snapshot.shaderObject.shaderBinaryVersion, 42u);
        EXPECT_EQ(snapshot.shaderObject.shaderBinaryUUID[0], 17u);
        EXPECT_EQ(snapshot.opacityMicromap.maxMicromapTriangles,
            useExt ? std::numeric_limits<uint32_t>::max() : 123u);
        if (useExt) {
            EXPECT_EQ(snapshot.opacityMicromap.maxOpacity2StateSubdivisionLevel, 8u);
            EXPECT_EQ(snapshot.opacityMicromap.maxOpacity4StateSubdivisionLevel, 6u);
        }
        auto copy = snapshot;
        auto moved = std::move(copy);
        expectDetached(snapshot);
        expectDetached(moved);
    }
}

} // namespace
} // namespace metallic::render::vulkan
