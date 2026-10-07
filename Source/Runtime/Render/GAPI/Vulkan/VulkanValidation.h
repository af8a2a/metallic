#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include <volk.h>

namespace metallic::render::vulkan {

constexpr ValidationSeverity validationSeverity(VkDebugUtilsMessageSeverityFlagBitsEXT severity)
{
    switch (severity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: return ValidationSeverity::Verbose;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: return ValidationSeverity::Info;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: return ValidationSeverity::Warning;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: return ValidationSeverity::Error;
    default: return ValidationSeverity::Unknown;
    }
}

constexpr ValidationCategory validationCategory(VkDebugUtilsMessageTypeFlagsEXT flags)
{
    ValidationCategory result = ValidationCategory::None;
    if (flags & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) { result = result | ValidationCategory::General; }
    if (flags & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) { result = result | ValidationCategory::Validation; }
    if (flags & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) { result = result | ValidationCategory::Performance; }
    if (flags & VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT) { result = result | ValidationCategory::ResourceBinding; }
    constexpr VkDebugUtilsMessageTypeFlagsEXT known = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT;
    if (flags & ~known) { result = result | ValidationCategory::Unknown; }
    return result;
}

// Backend-only object kinds, including future extension objects, stay Unknown.
// The opaque handle/name remain available for correlation in diagnostics.
constexpr ValidationObjectType validationObjectType(VkObjectType type)
{
    switch (type) {
    case VK_OBJECT_TYPE_INSTANCE: return ValidationObjectType::Instance;
    case VK_OBJECT_TYPE_PHYSICAL_DEVICE: return ValidationObjectType::Adapter;
    case VK_OBJECT_TYPE_DEVICE: return ValidationObjectType::Device;
    case VK_OBJECT_TYPE_QUEUE: return ValidationObjectType::Queue;
    case VK_OBJECT_TYPE_COMMAND_BUFFER: return ValidationObjectType::CommandBuffer;
    case VK_OBJECT_TYPE_SEMAPHORE: return ValidationObjectType::Semaphore;
    case VK_OBJECT_TYPE_FENCE: return ValidationObjectType::Fence;
    case VK_OBJECT_TYPE_DEVICE_MEMORY: return ValidationObjectType::DeviceMemory;
    case VK_OBJECT_TYPE_BUFFER: return ValidationObjectType::Buffer;
    case VK_OBJECT_TYPE_BUFFER_VIEW: return ValidationObjectType::BufferView;
    case VK_OBJECT_TYPE_IMAGE: return ValidationObjectType::Texture;
    case VK_OBJECT_TYPE_IMAGE_VIEW: return ValidationObjectType::TextureView;
    case VK_OBJECT_TYPE_SAMPLER: return ValidationObjectType::Sampler;
    case VK_OBJECT_TYPE_SHADER_MODULE: return ValidationObjectType::ShaderModule;
    case VK_OBJECT_TYPE_PIPELINE: return ValidationObjectType::Pipeline;
    case VK_OBJECT_TYPE_PIPELINE_CACHE: return ValidationObjectType::PipelineCache;
    case VK_OBJECT_TYPE_PIPELINE_LAYOUT: return ValidationObjectType::PipelineLayout;
    case VK_OBJECT_TYPE_DESCRIPTOR_POOL: return ValidationObjectType::DescriptorHeap;
    case VK_OBJECT_TYPE_QUERY_POOL: return ValidationObjectType::QueryPool;
    case VK_OBJECT_TYPE_COMMAND_POOL: return ValidationObjectType::CommandPool;
    case VK_OBJECT_TYPE_SURFACE_KHR: return ValidationObjectType::Surface;
    case VK_OBJECT_TYPE_SWAPCHAIN_KHR: return ValidationObjectType::Swapchain;
    case VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR: return ValidationObjectType::AccelerationStructure;
    case VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV: return ValidationObjectType::AccelerationStructure;
    case VK_OBJECT_TYPE_MICROMAP_EXT: return ValidationObjectType::Micromap;
    default: return ValidationObjectType::Unknown;
    }
}

} // namespace metallic::render::vulkan
