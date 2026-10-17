#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceProperties.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceFeatures.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanRayTracing.h"

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
#ifdef VK_EXT_ray_tracing_invocation_reorder
        if (node->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT) {
            auto& properties = *reinterpret_cast<VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT*>(node);
            properties.rayTracingInvocationReorderReorderingHint = VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT;
            properties.maxShaderBindingTableRecordIndex = 65535;
        }
#endif
    }
}

void expectDetached(const VulkanDeviceProperties& properties)
{
    EXPECT_EQ(properties.driver.pNext, nullptr);
    EXPECT_EQ(properties.maintenance4.pNext, nullptr);
    EXPECT_EQ(properties.shaderObject.pNext, nullptr);
    EXPECT_EQ(properties.accelerationStructure.pNext, nullptr);
    EXPECT_EQ(properties.rayTracingPipeline.pNext, nullptr);
#ifdef VK_EXT_ray_tracing_invocation_reorder
    EXPECT_EQ(properties.rayTracingInvocationReorder.pNext, nullptr);
#endif
    EXPECT_EQ(properties.descriptorHeap.pNext, nullptr);
    EXPECT_EQ(properties.generatedCommands.pNext, nullptr);
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
    const auto second = queryDeviceProperties(VK_NULL_HANDLE, {}, fakeQuery);
    EXPECT_EQ(first.core.deviceID, 1u);
    EXPECT_EQ(second.core.deviceID, 2u);
    expectDetached(first);
    expectDetached(second);
}

TEST(VulkanDeviceProperties, EnabledExtensionsShareOneQuery)
{
    {
        negotiation::VulkanDeviceFeatureSelection features{};
        features.shaderObject = true;
        features.rayTracingAccelerationStructure = true;
        features.rayTracingPipeline = true;
        features.rayTracingInvocationReorder = true;
        features.bindlessDescriptorHeap = true;
        features.deviceGeneratedCommands = true;
        features.clusterAccelerationStructure = true;
        queryCount = 0;
        const auto snapshot = queryDeviceProperties(VK_NULL_HANDLE, features, fakeQuery);
        EXPECT_EQ(queryCount, 1u);
        std::set<VkStructureType> expected{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_PROPERTIES_EXT,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT};
#ifdef VK_NV_cluster_acceleration_structure
        expected.insert(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV);
#endif
#ifdef VK_EXT_ray_tracing_invocation_reorder
        expected.insert(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT);
        EXPECT_EQ(snapshot.rayTracingInvocationReorder.rayTracingInvocationReorderReorderingHint,
            VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_EXT);
        EXPECT_EQ(snapshot.rayTracingInvocationReorder.maxShaderBindingTableRecordIndex, 65535u);
#endif
        EXPECT_EQ(queriedTypes, expected);
        EXPECT_EQ(snapshot.shaderObject.shaderBinaryVersion, 42u);
        EXPECT_EQ(snapshot.shaderObject.shaderBinaryUUID[0], 17u);
        auto copy = snapshot;
        auto moved = std::move(copy);
        expectDetached(snapshot);
        expectDetached(moved);
    }
}

TEST(VulkanRayTracingSBT, AlignsAllThreeRegionsAndBoundsPadding)
{
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{};
    properties.shaderGroupHandleSize = 24;
    properties.shaderGroupHandleAlignment = 32;
    properties.shaderGroupBaseAlignment = 64;
    properties.maxShaderGroupStride = 4096;
    const auto layout = fixedRayTracingSBTLayout(properties);
    ASSERT_TRUE(layout);
    EXPECT_EQ(layout->stride, 64u);
    for (uint64_t address = 4096; address < 4160; ++address) {
        const auto offset = layout->offset(address);
        ASSERT_TRUE(offset);
        EXPECT_LE(*offset + 3 * layout->stride, layout->allocationSize);
        for (uint64_t group = 0; group != 3; ++group) {
            EXPECT_EQ((address + *offset + group * layout->stride) % properties.shaderGroupBaseAlignment, 0u);
        }
    }
    EXPECT_FALSE(layout->offset(0));
    EXPECT_FALSE(layout->offset(std::numeric_limits<uint64_t>::max() - 16));
}

TEST(VulkanRayTracingSBT, RejectsInvalidPropertiesAndExcessiveStride)
{
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{};
    EXPECT_FALSE(fixedRayTracingSBTLayout(properties));
    properties.shaderGroupHandleSize = 32;
    properties.shaderGroupHandleAlignment = 32;
    properties.shaderGroupBaseAlignment = 64;
    properties.maxShaderGroupStride = 32;
    EXPECT_FALSE(fixedRayTracingSBTLayout(properties));
    properties.maxShaderGroupStride = 4096;
    properties.shaderGroupBaseAlignment = 63;
    EXPECT_FALSE(fixedRayTracingSBTLayout(properties));
    properties.shaderGroupBaseAlignment = 64;
    properties.shaderGroupHandleSize = std::numeric_limits<uint32_t>::max();
    EXPECT_FALSE(fixedRayTracingSBTLayout(properties));
}

} // namespace
} // namespace metallic::render::vulkan
