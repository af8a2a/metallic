#include "Runtime/Render/GAPI/Vulkan/VulkanValidation.h"
#include <gtest/gtest.h>

namespace metallic::tests {
using namespace render;

TEST(ValidationTranslation, SeverityUsesIndependentValues)
{
    EXPECT_EQ(vulkan::validationSeverity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT), ValidationSeverity::Verbose);
    EXPECT_EQ(vulkan::validationSeverity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT), ValidationSeverity::Info);
    EXPECT_EQ(vulkan::validationSeverity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT), ValidationSeverity::Warning);
    EXPECT_EQ(vulkan::validationSeverity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT), ValidationSeverity::Error);
    EXPECT_NE(static_cast<uint32_t>(ValidationSeverity::Error), uint32_t(VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT));
    EXPECT_EQ(vulkan::validationSeverity(static_cast<VkDebugUtilsMessageSeverityFlagBitsEXT>(0)), ValidationSeverity::Unknown);
    EXPECT_EQ(vulkan::validationSeverity(static_cast<VkDebugUtilsMessageSeverityFlagBitsEXT>(0x40000000)), ValidationSeverity::Unknown);
}

TEST(ValidationTranslation, CategoriesPreserveCombinationsAndUnknownBits)
{
    const auto categories = vulkan::validationCategory(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT | 0x80000000u);
    EXPECT_TRUE(hasFlag(categories, ValidationCategory::General));
    EXPECT_TRUE(hasFlag(categories, ValidationCategory::Validation));
    EXPECT_TRUE(hasFlag(categories, ValidationCategory::Performance));
    EXPECT_TRUE(hasFlag(categories, ValidationCategory::ResourceBinding));
    EXPECT_TRUE(hasFlag(categories, ValidationCategory::Unknown));
    EXPECT_EQ(vulkan::validationCategory(0), ValidationCategory::None);
    EXPECT_FALSE(hasFlag(ValidationCategory::General, ValidationCategory::Validation | ValidationCategory::Performance));
    EXPECT_TRUE(hasFlag(ValidationCategory::Performance, ValidationCategory::Validation | ValidationCategory::Performance));
    EXPECT_FALSE(hasFlag(categories, ValidationCategory::None));
}

TEST(ValidationTranslation, ObjectsUseResourceKindsAndUnknownFallback)
{
    EXPECT_EQ(vulkan::validationObjectType(VK_OBJECT_TYPE_BUFFER), ValidationObjectType::Buffer);
    EXPECT_EQ(vulkan::validationObjectType(VK_OBJECT_TYPE_IMAGE), ValidationObjectType::Texture);
    EXPECT_EQ(vulkan::validationObjectType(VK_OBJECT_TYPE_IMAGE_VIEW), ValidationObjectType::TextureView);
    EXPECT_EQ(vulkan::validationObjectType(VK_OBJECT_TYPE_PHYSICAL_DEVICE), ValidationObjectType::Adapter);
    EXPECT_EQ(vulkan::validationObjectType(VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR), ValidationObjectType::AccelerationStructure);
    EXPECT_EQ(vulkan::validationObjectType(VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV), ValidationObjectType::AccelerationStructure);
    EXPECT_EQ(vulkan::validationObjectType(VK_OBJECT_TYPE_MICROMAP_EXT), ValidationObjectType::Micromap);
    EXPECT_EQ(vulkan::validationObjectType(VK_OBJECT_TYPE_DESCRIPTOR_POOL), ValidationObjectType::DescriptorHeap);
    EXPECT_EQ(vulkan::validationObjectType(VK_OBJECT_TYPE_UNKNOWN), ValidationObjectType::Unknown);
    EXPECT_EQ(vulkan::validationObjectType(static_cast<VkObjectType>(0x7ffffffe)), ValidationObjectType::Unknown);
}
} // namespace metallic::tests
