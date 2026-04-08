#include "vulkan_backend.h"
#include "rhi_resource_utils.h"
#include "slang_compiler.h"
#include "vulkan_resource_handles.h"
#include "vulkan_pipeline_cache.h"
#include "vulkan_diagnostics.h"
#include "vulkan_transient_allocator.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

#ifndef VK_API_VERSION_1_4
#define VK_API_VERSION_1_4 VK_MAKE_API_VERSION(0, 1, 4, 0)
#endif

#ifndef VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME
#define VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME "VK_EXT_external_memory_host"
#endif

#ifndef VK_EXT_DEVICE_FAULT_EXTENSION_NAME
#define VK_EXT_DEVICE_FAULT_EXTENSION_NAME "VK_EXT_device_fault"
#endif

#ifndef VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME
#define VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME "VK_NV_device_diagnostic_checkpoints"
#endif

#ifndef VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME
#define VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME "VK_EXT_subgroup_size_control"
#endif

#ifndef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES \
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT
using VkPhysicalDeviceSubgroupSizeControlFeatures =
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT;
#endif

#ifndef VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES \
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT
using VkPhysicalDeviceSubgroupSizeControlProperties =
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT;
#endif

#ifndef VK_SUBGROUP_FEATURE_ROTATE_BIT
#define VK_SUBGROUP_FEATURE_ROTATE_BIT VK_SUBGROUP_FEATURE_ROTATE_BIT_KHR
#endif

#ifndef VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT
#define VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT_KHR
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <optional>
#include <set>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

void vulkanLoadMeshShaderFunctions(VkDevice device);

namespace {

constexpr uint32_t kMaxFramesInFlight = 2;
constexpr uint32_t kPushConstantSize = 256;
const char* kValidationLayerName = "VK_LAYER_KHRONOS_validation";
const char* kRenderDocLayerName = "VK_LAYER_RENDERDOC_Capture";

template <typename VkHandle>
uint64_t vkObjectHandle(VkHandle handle) {
    if constexpr (std::is_pointer_v<VkHandle>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugUtilsCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT /*messageTypes*/,
    const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
    void* /*userData*/) {
    std::string message =
        (callbackData && callbackData->pMessage) ? callbackData->pMessage : "Unknown Vulkan message";
    if (callbackData) {
        if (callbackData->cmdBufLabelCount > 0 && callbackData->pCmdBufLabels) {
            const auto& label = callbackData->pCmdBufLabels[callbackData->cmdBufLabelCount - 1];
            if (label.pLabelName && label.pLabelName[0] != '\0') {
                message += " [cmdLabel=";
                message += label.pLabelName;
                message += "]";
            }
        }

        if (callbackData->objectCount > 0 && callbackData->pObjects) {
            constexpr uint32_t kMaxReportedObjects = 3;
            const uint32_t objectCount = std::min(callbackData->objectCount, kMaxReportedObjects);
            for (uint32_t i = 0; i < objectCount; ++i) {
                const auto& object = callbackData->pObjects[i];
                message += " [obj";
                message += std::to_string(i);
                message += "=";
                if (object.pObjectName && object.pObjectName[0] != '\0') {
                    message += object.pObjectName;
                } else {
                    message += std::to_string(static_cast<uint32_t>(object.objectType));
                    message += ":0x";
                    char handleBuffer[17]{};
                    std::snprintf(handleBuffer, sizeof(handleBuffer), "%llx",
                                  static_cast<unsigned long long>(object.objectHandle));
                    message += handleBuffer;
                }
                message += "]";
            }
            if (callbackData->objectCount > objectCount) {
                message += " [objects=";
                message += std::to_string(callbackData->objectCount);
                message += "]";
            }
        }
    }
    if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        spdlog::error("VulkanValidation: {}", message);
    } else if ((messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        spdlog::warn("VulkanValidation: {}", message);
    } else {
        spdlog::info("VulkanValidation: {}", message);
    }
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT makeDebugMessengerCreateInfo() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = vulkanDebugUtilsCallback;
    return createInfo;
}

VkFormat toVkFormat(RhiFormat format) {
    switch (format) {
    case RhiFormat::R8Unorm: return VK_FORMAT_R8_UNORM;
    case RhiFormat::R16Float: return VK_FORMAT_R16_SFLOAT;
    case RhiFormat::R32Float: return VK_FORMAT_R32_SFLOAT;
    case RhiFormat::R32Uint: return VK_FORMAT_R32_UINT;
    case RhiFormat::RG8Unorm: return VK_FORMAT_R8G8_UNORM;
    case RhiFormat::RG16Float: return VK_FORMAT_R16G16_SFLOAT;
    case RhiFormat::RG32Float: return VK_FORMAT_R32G32_SFLOAT;
    case RhiFormat::BGRA8Unorm: return VK_FORMAT_B8G8R8A8_UNORM;
    case RhiFormat::RGBA8Unorm: return VK_FORMAT_R8G8B8A8_UNORM;
    case RhiFormat::RGBA8Srgb: return VK_FORMAT_R8G8B8A8_SRGB;
    case RhiFormat::RGBA16Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case RhiFormat::RGBA32Float: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case RhiFormat::D32Float: return VK_FORMAT_D32_SFLOAT;
    case RhiFormat::D16Unorm: return VK_FORMAT_D16_UNORM;
    case RhiFormat::Undefined:
    default: return VK_FORMAT_UNDEFINED;
    }
}

RhiFormat fromVkFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM: return RhiFormat::R8Unorm;
    case VK_FORMAT_R16_SFLOAT: return RhiFormat::R16Float;
    case VK_FORMAT_R32_SFLOAT: return RhiFormat::R32Float;
    case VK_FORMAT_R32_UINT: return RhiFormat::R32Uint;
    case VK_FORMAT_R8G8_UNORM: return RhiFormat::RG8Unorm;
    case VK_FORMAT_R16G16_SFLOAT: return RhiFormat::RG16Float;
    case VK_FORMAT_R32G32_SFLOAT: return RhiFormat::RG32Float;
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB: return RhiFormat::BGRA8Unorm;
    case VK_FORMAT_R8G8B8A8_UNORM: return RhiFormat::RGBA8Unorm;
    case VK_FORMAT_R8G8B8A8_SRGB: return RhiFormat::RGBA8Srgb;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return RhiFormat::RGBA16Float;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return RhiFormat::RGBA32Float;
    case VK_FORMAT_D32_SFLOAT: return RhiFormat::D32Float;
    case VK_FORMAT_D16_UNORM: return RhiFormat::D16Unorm;
    default: return RhiFormat::Undefined;
    }
}

VkImageUsageFlags toVkImageUsage(RhiTextureUsage usage) {
    VkImageUsageFlags flags = 0;
    if ((usage & RhiTextureUsage::RenderTarget) != RhiTextureUsage::None) {
        flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if ((usage & RhiTextureUsage::ShaderRead) != RhiTextureUsage::None) {
        flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if ((usage & RhiTextureUsage::ShaderWrite) != RhiTextureUsage::None) {
        flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return flags;
}

VkFormat toVkVertexFormat(RhiVertexFormat format) {
    switch (format) {
    case RhiVertexFormat::Float2: return VK_FORMAT_R32G32_SFLOAT;
    case RhiVertexFormat::Float3: return VK_FORMAT_R32G32B32_SFLOAT;
    case RhiVertexFormat::Float4: return VK_FORMAT_R32G32B32A32_SFLOAT;
    default: return VK_FORMAT_UNDEFINED;
    }
}

VkFilter toVkFilter(RhiSamplerFilterMode filter) {
    return filter == RhiSamplerFilterMode::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
}

VkSamplerMipmapMode toVkMipFilter(RhiSamplerMipFilterMode filter) {
    return filter == RhiSamplerMipFilterMode::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR
                                                     : VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

VkSamplerAddressMode toVkAddressMode(RhiSamplerAddressMode mode) {
    return mode == RhiSamplerAddressMode::ClampToEdge ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                                                      : VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

bool isVkDepthFormat(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D16_UNORM ||
           format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

VkShaderModule createShaderModuleFromSpirv(VkDevice device, std::span<const uint32_t> spirv) {
    if (device == VK_NULL_HANDLE || spirv.empty()) {
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = spirv.size_bytes();
    createInfo.pCode = spirv.data();

    VkShaderModule module = VK_NULL_HANDLE;
    return vkCreateShaderModule(device, &createInfo, nullptr, &module) == VK_SUCCESS
               ? module
               : VK_NULL_HANDLE;
}

VkDescriptorType toVkDescriptorType(SlangShaderBindingType type) {
    switch (type) {
    case SlangShaderBindingType::UniformBuffer:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case SlangShaderBindingType::StorageBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case SlangShaderBindingType::SampledTexture:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case SlangShaderBindingType::StorageTexture:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case SlangShaderBindingType::Sampler:
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    case SlangShaderBindingType::AccelerationStructure:
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    default:
        return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
}

void destroyPipelineLayouts(VulkanPipelineResource& resource) {
    for (size_t setIndex = 0; setIndex < resource.setLayouts.size(); ++setIndex) {
        VkDescriptorSetLayout setLayout = resource.setLayouts[setIndex];
        if (setLayout != VK_NULL_HANDLE) {
            const bool ownsSetLayout =
                setIndex >= resource.setLayoutOwnership.size() || resource.setLayoutOwnership[setIndex] != 0;
            if (ownsSetLayout) {
                vkDestroyDescriptorSetLayout(resource.device, setLayout, nullptr);
            } else if (resource.device != VK_NULL_HANDLE &&
                       resource.bindlessSetIndex == static_cast<uint32_t>(setIndex)) {
                vulkanReleaseBindlessSetLayout(resource.device);
            }
        }
    }
    resource.setLayouts.clear();
    resource.setLayoutOwnership.clear();
    resource.bindlessSetIndex = UINT32_MAX;

    if (resource.ownsLayout && resource.layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(resource.device, resource.layout, nullptr);
        resource.layout = VK_NULL_HANDLE;
    }
}

bool assignBindingLocation(VulkanDescriptorBindingLocation& location,
                           uint32_t set,
                           uint32_t binding,
                           uint32_t arrayElement,
                           VkDescriptorType descriptorType) {
    location.set = set;
    location.binding = binding;
    location.arrayElement = arrayElement;
    location.descriptorType = descriptorType;
    return true;
}

uint32_t bindlessDescriptorLimit(uint32_t bindingIndex, VkDescriptorType descriptorType) {
    switch (bindingIndex) {
    case kVulkanBindlessSampledImageBinding:
        return descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ? kVulkanBindlessMaxSampledImages : 0;
    case kVulkanBindlessSamplerBinding:
        return descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ? kVulkanBindlessMaxSamplers : 0;
    case kVulkanBindlessStorageImageBinding:
        return descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? kVulkanBindlessMaxStorageImages : 0;
    case kVulkanBindlessStorageBufferBinding:
        return descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? kVulkanBindlessMaxStorageBuffers : 0;
    case kVulkanBindlessAccelerationStructureBinding:
        return descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
            ? kVulkanBindlessMaxAccelerationStructures
            : 0;
    default:
        return 0;
    }
}

bool validateBindlessBinding(const SlangShaderBindingDesc& binding,
                             VkDescriptorType descriptorType,
                             std::string& errorMessage) {
    const uint32_t maxDescriptorCount = bindlessDescriptorLimit(binding.bindingIndex, descriptorType);
    if (maxDescriptorCount == 0) {
        errorMessage =
            "Shader declares an unsupported bindless resource binding at set " +
            std::to_string(binding.bindingSpace) + ", binding " + std::to_string(binding.bindingIndex);
        return false;
    }

    if (binding.descriptorCount == 0 || binding.descriptorCount > maxDescriptorCount) {
        errorMessage =
            "Shader bindless resource at set " + std::to_string(binding.bindingSpace) +
            ", binding " + std::to_string(binding.bindingIndex) +
            " exceeds Metallic's shared bindless table capacity";
        return false;
    }

    return true;
}

bool buildPipelineResourceLayout(VkDevice device,
                                 const void* data,
                                 size_t size,
                                 VulkanPipelineResource& outResource,
                                 std::string& errorMessage,
                                 bool useDescriptorBuffer = false) {
    SlangShaderBindingLayout shaderLayout;
    if (!findSlangBindingLayoutForBinary(data, size, shaderLayout)) {
        errorMessage = "Missing cached Slang reflection for SPIR-V shader";
        return false;
    }

    uint32_t maxSetIndex = 0;
    for (const auto& binding : shaderLayout.bindings) {
        maxSetIndex = std::max(maxSetIndex, binding.bindingSpace);
    }

    std::vector<std::vector<VkDescriptorSetLayoutBinding>> perSetBindings(
        shaderLayout.bindings.empty() ? 0 : maxSetIndex + 1);
    std::vector<std::vector<VkDescriptorBindingFlags>> perSetBindingFlags(perSetBindings.size());

    uint32_t logicalBufferIndex = 0;
    uint32_t logicalTextureIndex = 0;
    uint32_t logicalSamplerIndex = 0;
    uint32_t logicalAccelerationStructureIndex = 0;
    bool bindlessSetPresent = false;

    for (const auto& binding : shaderLayout.bindings) {
        if (binding.type == SlangShaderBindingType::PushConstantBuffer) {
            ++logicalBufferIndex;
            continue;
        }

        const VkDescriptorType descriptorType = toVkDescriptorType(binding.type);
        if (descriptorType == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
            errorMessage = "Unsupported Slang resource type in Vulkan binding layout";
            destroyPipelineLayouts(outResource);
            return false;
        }

        if (vulkanIsBindlessSetIndex(binding.bindingSpace)) {
            if (!validateBindlessBinding(binding, descriptorType, errorMessage)) {
                destroyPipelineLayouts(outResource);
                return false;
            }
            bindlessSetPresent = true;
            continue;
        }

        if (binding.bindingSpace >= perSetBindings.size()) {
            errorMessage = "Invalid descriptor set index in Slang reflection";
            destroyPipelineLayouts(outResource);
            return false;
        }

        auto& setBindings = perSetBindings[binding.bindingSpace];
        auto existingBinding = std::find_if(setBindings.begin(),
                                            setBindings.end(),
                                            [&](const VkDescriptorSetLayoutBinding& vkBinding) {
                                                return vkBinding.binding == binding.bindingIndex;
                                            });
        if (existingBinding == setBindings.end()) {
            VkDescriptorSetLayoutBinding layoutBinding{};
            layoutBinding.binding = binding.bindingIndex;
            layoutBinding.descriptorType = descriptorType;
            layoutBinding.descriptorCount = binding.descriptorCount;
            layoutBinding.stageFlags = VK_SHADER_STAGE_ALL;
            setBindings.push_back(layoutBinding);
            perSetBindingFlags[binding.bindingSpace].push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
        }

        auto assignExpandedBinding = [&](auto& locations, uint32_t& logicalIndex) -> bool {
            for (uint32_t arrayElement = 0; arrayElement < binding.descriptorCount; ++arrayElement) {
                if (logicalIndex >= locations.size()) {
                    errorMessage = "Slang reflection exceeded Metallic's Vulkan binding slot limits";
                    return false;
                }
                assignBindingLocation(locations[logicalIndex],
                                      binding.bindingSpace,
                                      binding.bindingIndex,
                                      arrayElement,
                                      descriptorType);
                ++logicalIndex;
            }
            return true;
        };

        bool ok = true;
        switch (binding.type) {
        case SlangShaderBindingType::UniformBuffer:
        case SlangShaderBindingType::StorageBuffer:
            ok = assignExpandedBinding(outResource.bufferBindings, logicalBufferIndex);
            break;
        case SlangShaderBindingType::SampledTexture:
        case SlangShaderBindingType::StorageTexture:
            ok = assignExpandedBinding(outResource.textureBindings, logicalTextureIndex);
            break;
        case SlangShaderBindingType::Sampler:
            ok = assignExpandedBinding(outResource.samplerBindings, logicalSamplerIndex);
            break;
        case SlangShaderBindingType::AccelerationStructure:
            ok = assignExpandedBinding(outResource.accelerationStructureBindings,
                                       logicalAccelerationStructureIndex);
            break;
        }

        if (!ok) {
            destroyPipelineLayouts(outResource);
            return false;
        }
    }

    outResource.setLayouts.resize(perSetBindings.size(), VK_NULL_HANDLE);
    outResource.setLayoutOwnership.assign(perSetBindings.size(), 1);
    for (size_t setIndex = 0; setIndex < perSetBindings.size(); ++setIndex) {
        if (bindlessSetPresent && vulkanIsBindlessSetIndex(static_cast<uint32_t>(setIndex))) {
            VkDescriptorSetLayout bindlessSetLayout = VK_NULL_HANDLE;
            if (!vulkanRetainBindlessSetLayout(device, bindlessSetLayout, &errorMessage, useDescriptorBuffer)) {
                if (errorMessage.empty()) {
                    errorMessage = "Failed to acquire shared bindless descriptor set layout";
                }
                destroyPipelineLayouts(outResource);
                return false;
            }

            outResource.setLayouts[setIndex] = bindlessSetLayout;
            outResource.setLayoutOwnership[setIndex] = 0;
            outResource.bindlessSetIndex = static_cast<uint32_t>(setIndex);
            continue;
        }

        auto& bindings = perSetBindings[setIndex];
        if (!bindings.empty()) {
            std::vector<size_t> order(bindings.size());
            for (size_t index = 0; index < order.size(); ++index) {
                order[index] = index;
            }
            std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
                return bindings[lhs].binding < bindings[rhs].binding;
            });

            std::vector<VkDescriptorSetLayoutBinding> sortedBindings;
            std::vector<VkDescriptorBindingFlags> sortedFlags;
            sortedBindings.reserve(bindings.size());
            sortedFlags.reserve(bindings.size());
            for (size_t orderedIndex : order) {
                sortedBindings.push_back(bindings[orderedIndex]);
                sortedFlags.push_back(perSetBindingFlags[setIndex][orderedIndex]);
            }
            bindings = std::move(sortedBindings);
            perSetBindingFlags[setIndex] = std::move(sortedFlags);
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        if (!bindings.empty()) {
            flagsInfo.bindingCount = static_cast<uint32_t>(perSetBindingFlags[setIndex].size());
            flagsInfo.pBindingFlags = perSetBindingFlags[setIndex].data();
            layoutInfo.pNext = &flagsInfo;
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
        }
        if (useDescriptorBuffer) {
            layoutInfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }

        const VkResult result = vkCreateDescriptorSetLayout(device,
                                                            &layoutInfo,
                                                            nullptr,
                                                            &outResource.setLayouts[setIndex]);
        if (result != VK_SUCCESS) {
            errorMessage = "Failed to create Vulkan descriptor set layout (VkResult: " +
                           std::to_string(result) + ")";
            destroyPipelineLayouts(outResource);
            return false;
        }
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
    pushConstantRange.offset = 0;
    pushConstantRange.size = kPushConstantSize;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(outResource.setLayouts.size());
    pipelineLayoutInfo.pSetLayouts =
        outResource.setLayouts.empty() ? nullptr : outResource.setLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    const VkResult layoutResult = vkCreatePipelineLayout(device,
                                                         &pipelineLayoutInfo,
                                                         nullptr,
                                                         &outResource.layout);
    if (layoutResult != VK_SUCCESS) {
        errorMessage = "Failed to create Vulkan pipeline layout (VkResult: " +
                       std::to_string(layoutResult) + ")";
        destroyPipelineLayouts(outResource);
        return false;
    }

    outResource.ownsLayout = true;
    return true;
}

bool buildVertexInputDescriptions(const RhiVertexDescriptor* vertexDescriptor,
                                  std::vector<VkVertexInputBindingDescription>& outBindings,
                                  std::vector<VkVertexInputAttributeDescription>& outAttributes,
                                  std::string& errorMessage) {
    outBindings.clear();
    outAttributes.clear();

    if (!vertexDescriptor || !vertexDescriptor->nativeHandle()) {
        return true;
    }

    const auto* resource = getVulkanVertexDescriptorResource(*vertexDescriptor);
    if (!resource) {
        errorMessage = "Invalid Vulkan vertex descriptor handle";
        return false;
    }

    outBindings.reserve(resource->bindings.size());
    for (const auto& binding : resource->bindings) {
        if (!binding.valid) {
            continue;
        }

        VkVertexInputBindingDescription vkBinding{};
        vkBinding.binding = binding.binding;
        vkBinding.stride = binding.stride;
        vkBinding.inputRate = binding.inputRate;
        outBindings.push_back(vkBinding);
    }
    std::sort(outBindings.begin(),
              outBindings.end(),
              [](const VkVertexInputBindingDescription& lhs,
                 const VkVertexInputBindingDescription& rhs) {
                  return lhs.binding < rhs.binding;
              });

    outAttributes.reserve(resource->attributes.size());
    for (const auto& attribute : resource->attributes) {
        if (!attribute.valid) {
            continue;
        }

        if (attribute.format == VK_FORMAT_UNDEFINED) {
            errorMessage = "Unsupported Vulkan vertex format";
            return false;
        }

        const bool hasBinding = std::any_of(outBindings.begin(),
                                            outBindings.end(),
                                            [&](const VkVertexInputBindingDescription& binding) {
                                                return binding.binding == attribute.binding;
                                            });
        if (!hasBinding) {
            errorMessage = "Vertex attribute location " + std::to_string(attribute.location) +
                           " references missing vertex layout binding " +
                           std::to_string(attribute.binding);
            return false;
        }

        VkVertexInputAttributeDescription vkAttribute{};
        vkAttribute.location = attribute.location;
        vkAttribute.binding = attribute.binding;
        vkAttribute.format = attribute.format;
        vkAttribute.offset = attribute.offset;
        outAttributes.push_back(vkAttribute);
    }
    std::sort(outAttributes.begin(),
              outAttributes.end(),
              [](const VkVertexInputAttributeDescription& lhs,
                 const VkVertexInputAttributeDescription& rhs) {
                  return lhs.location < rhs.location;
              });

    return true;
}

std::string vkVersionString(uint32_t version) {
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "." +
           std::to_string(VK_API_VERSION_MINOR(version)) + "." +
           std::to_string(VK_API_VERSION_PATCH(version));
}

void checkVk(VkResult result, const char* message) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(message) + " (VkResult=" + std::to_string(result) + ")");
    }
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    std::optional<uint32_t> compute;   // dedicated compute (no graphics bit)
    std::optional<uint32_t> transfer;  // dedicated transfer (no graphics/compute bits)

    bool complete() const {
        return graphics.has_value() && present.has_value();
    }
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphics = i;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
        if (presentSupport == VK_TRUE) {
            indices.present = i;
        }

        // Dedicated compute: has compute but NOT graphics
        if ((queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            indices.compute = i;
        }

        // Dedicated transfer: has transfer but NOT graphics and NOT compute
        if ((queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
            !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            !(queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            indices.transfer = i;
        }
    }

    return indices;
}

SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface) {
    SwapchainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    if (formatCount > 0) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    if (presentModeCount > 0) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
    for (const auto& format : availableFormats) {
        if ((format.format == VK_FORMAT_B8G8R8A8_UNORM || format.format == VK_FORMAT_B8G8R8A8_SRGB) &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return availableFormats.front();
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& presentMode : availablePresentModes) {
        if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return presentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities,
                            uint32_t requestedWidth,
                            uint32_t requestedHeight) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }

    VkExtent2D actualExtent = {requestedWidth, requestedHeight};
    actualExtent.width = std::clamp(actualExtent.width,
                                    capabilities.minImageExtent.width,
                                    capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height,
                                     capabilities.minImageExtent.height,
                                     capabilities.maxImageExtent.height);
    return actualExtent;
}

bool hasExtension(std::span<const VkExtensionProperties> extensions, const char* name) {
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties& extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

bool hasLayer(std::span<const VkLayerProperties> layers, const char* name) {
    return std::any_of(layers.begin(), layers.end(), [name](const VkLayerProperties& layer) {
        return std::strcmp(layer.layerName, name) == 0;
    });
}

RhiSubgroupStage toRhiSubgroupStages(VkShaderStageFlags flags) {
    RhiSubgroupStage result = RhiSubgroupStage::None;
    if ((flags & VK_SHADER_STAGE_VERTEX_BIT) != 0) {
        result = result | RhiSubgroupStage::Vertex;
    }
    if ((flags & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) != 0) {
        result = result | RhiSubgroupStage::TessControl;
    }
    if ((flags & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT) != 0) {
        result = result | RhiSubgroupStage::TessEvaluation;
    }
    if ((flags & VK_SHADER_STAGE_GEOMETRY_BIT) != 0) {
        result = result | RhiSubgroupStage::Geometry;
    }
    if ((flags & VK_SHADER_STAGE_FRAGMENT_BIT) != 0) {
        result = result | RhiSubgroupStage::Fragment;
    }
    if ((flags & VK_SHADER_STAGE_COMPUTE_BIT) != 0) {
        result = result | RhiSubgroupStage::Compute;
    }
    if ((flags & VK_SHADER_STAGE_TASK_BIT_EXT) != 0) {
        result = result | RhiSubgroupStage::Task;
    }
    if ((flags & VK_SHADER_STAGE_MESH_BIT_EXT) != 0) {
        result = result | RhiSubgroupStage::Mesh;
    }
    if ((flags & VK_SHADER_STAGE_RAYGEN_BIT_KHR) != 0) {
        result = result | RhiSubgroupStage::RayGen;
    }
    if ((flags & VK_SHADER_STAGE_ANY_HIT_BIT_KHR) != 0) {
        result = result | RhiSubgroupStage::AnyHit;
    }
    if ((flags & VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) != 0) {
        result = result | RhiSubgroupStage::ClosestHit;
    }
    if ((flags & VK_SHADER_STAGE_MISS_BIT_KHR) != 0) {
        result = result | RhiSubgroupStage::Miss;
    }
    if ((flags & VK_SHADER_STAGE_INTERSECTION_BIT_KHR) != 0) {
        result = result | RhiSubgroupStage::Intersection;
    }
    if ((flags & VK_SHADER_STAGE_CALLABLE_BIT_KHR) != 0) {
        result = result | RhiSubgroupStage::Callable;
    }
    return result;
}

RhiSubgroupOperation toRhiSubgroupOperations(VkSubgroupFeatureFlags flags) {
    RhiSubgroupOperation result = RhiSubgroupOperation::None;
    if ((flags & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0) {
        result = result | RhiSubgroupOperation::Basic;
    }
    if ((flags & VK_SUBGROUP_FEATURE_VOTE_BIT) != 0) {
        result = result | RhiSubgroupOperation::Vote;
    }
    if ((flags & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0) {
        result = result | RhiSubgroupOperation::Arithmetic;
    }
    if ((flags & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0) {
        result = result | RhiSubgroupOperation::Ballot;
    }
    if ((flags & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0) {
        result = result | RhiSubgroupOperation::Shuffle;
    }
    if ((flags & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0) {
        result = result | RhiSubgroupOperation::ShuffleRelative;
    }
    if ((flags & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0) {
        result = result | RhiSubgroupOperation::Clustered;
    }
    if ((flags & VK_SUBGROUP_FEATURE_QUAD_BIT) != 0) {
        result = result | RhiSubgroupOperation::Quad;
    }
    if ((flags & VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV) != 0) {
        result = result | RhiSubgroupOperation::Partitioned;
    }
    if ((flags & VK_SUBGROUP_FEATURE_ROTATE_BIT) != 0) {
        result = result | RhiSubgroupOperation::Rotate;
    }
    if ((flags & VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT) != 0) {
        result = result | RhiSubgroupOperation::RotateClustered;
    }
    return result;
}

std::string formatSubgroupStages(RhiSubgroupStage stages) {
    if (stages == RhiSubgroupStage::None) {
        return "None";
    }

    struct StageName {
        RhiSubgroupStage stage;
        const char* name;
    };
    constexpr StageName kStageNames[] = {
        {RhiSubgroupStage::Vertex, "Vertex"},
        {RhiSubgroupStage::TessControl, "TessControl"},
        {RhiSubgroupStage::TessEvaluation, "TessEvaluation"},
        {RhiSubgroupStage::Geometry, "Geometry"},
        {RhiSubgroupStage::Fragment, "Fragment"},
        {RhiSubgroupStage::Compute, "Compute"},
        {RhiSubgroupStage::Task, "Task"},
        {RhiSubgroupStage::Mesh, "Mesh"},
        {RhiSubgroupStage::RayGen, "RayGen"},
        {RhiSubgroupStage::AnyHit, "AnyHit"},
        {RhiSubgroupStage::ClosestHit, "ClosestHit"},
        {RhiSubgroupStage::Miss, "Miss"},
        {RhiSubgroupStage::Intersection, "Intersection"},
        {RhiSubgroupStage::Callable, "Callable"},
    };

    std::string result;
    for (const auto& entry : kStageNames) {
        if ((stages & entry.stage) == RhiSubgroupStage::None) {
            continue;
        }
        if (!result.empty()) {
            result += "|";
        }
        result += entry.name;
    }
    return result.empty() ? "None" : result;
}

std::string formatSubgroupOperations(RhiSubgroupOperation operations) {
    if (operations == RhiSubgroupOperation::None) {
        return "None";
    }

    struct OperationName {
        RhiSubgroupOperation operation;
        const char* name;
    };
    constexpr OperationName kOperationNames[] = {
        {RhiSubgroupOperation::Basic, "Basic"},
        {RhiSubgroupOperation::Vote, "Vote"},
        {RhiSubgroupOperation::Arithmetic, "Arithmetic"},
        {RhiSubgroupOperation::Ballot, "Ballot"},
        {RhiSubgroupOperation::Shuffle, "Shuffle"},
        {RhiSubgroupOperation::ShuffleRelative, "ShuffleRelative"},
        {RhiSubgroupOperation::Clustered, "Clustered"},
        {RhiSubgroupOperation::Quad, "Quad"},
        {RhiSubgroupOperation::Partitioned, "Partitioned"},
        {RhiSubgroupOperation::Rotate, "Rotate"},
        {RhiSubgroupOperation::RotateClustered, "RotateClustered"},
    };

    std::string result;
    for (const auto& entry : kOperationNames) {
        if ((operations & entry.operation) == RhiSubgroupOperation::None) {
            continue;
        }
        if (!result.empty()) {
            result += "|";
        }
        result += entry.name;
    }
    return result.empty() ? "None" : result;
}

void logSubgroupProperties(const RhiSubgroupProperties& subgroupProperties) {
    spdlog::info(
        "Vulkan: subgroup properties "
        "(supported={}, subgroupSize={}, supportedStages={}, supportedOperations={}, quadAllStages={}, "
        "sizeControl={}, computeFullSubgroups={}, minSubgroupSize={}, maxSubgroupSize={}, "
        "maxComputeWorkgroupSubgroups={}, requiredSubgroupSizeStages={})",
        subgroupProperties.supported,
        subgroupProperties.subgroupSize,
        formatSubgroupStages(subgroupProperties.supportedStages),
        formatSubgroupOperations(subgroupProperties.supportedOperations),
        subgroupProperties.quadOperationsInAllStages,
        subgroupProperties.sizeControl,
        subgroupProperties.computeFullSubgroups,
        subgroupProperties.minSubgroupSize,
        subgroupProperties.maxSubgroupSize,
        subgroupProperties.maxComputeWorkgroupSubgroups,
        formatSubgroupStages(subgroupProperties.requiredSubgroupSizeStages));
}

class VulkanShaderModule final : public RhiShaderModule {
public:
    VulkanShaderModule(VkDevice device, VkShaderModule module)
        : m_device(device), m_module(module) {}

    ~VulkanShaderModule() override {
        if (m_module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_device, m_module, nullptr);
        }
    }

    VkShaderModule handle() const { return m_module; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkShaderModule m_module = VK_NULL_HANDLE;
};

class VulkanBuffer final : public RhiBuffer {
public:
    explicit VulkanBuffer(VulkanBufferResource resource)
        : m_resource(resource) {}

    ~VulkanBuffer() override {
        vmaDestroyBufferResource(m_resource);
    }

    size_t size() const override { return m_resource.size; }
    void* nativeHandle() const override { return const_cast<VulkanBufferResource*>(&m_resource); }
    void* mappedData() override {
        if (m_resource.mappedData == nullptr && m_resource.allocation != nullptr) {
            if (vmaMapMemory(m_resource.allocator, m_resource.allocation, &m_resource.mappedData) != VK_SUCCESS) {
                return nullptr;
            }
        }
        return m_resource.mappedData;
    }
    VkBuffer handle() const { return m_resource.buffer; }

private:
    VulkanBufferResource m_resource{};
};


class VulkanGraphicsPipeline final : public RhiGraphicsPipeline {
public:
    VulkanGraphicsPipeline(VkDevice device, VkPipelineLayout layout, VkPipeline pipeline)
        : m_device(device), m_layout(layout), m_pipeline(pipeline) {}

    ~VulkanGraphicsPipeline() override {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
        }
        if (m_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_layout, nullptr);
        }
    }

    void* nativeHandle() const override { return reinterpret_cast<void*>(m_pipeline); }
    VkPipeline pipeline() const { return m_pipeline; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

class VulkanContext final : public RhiContext {
public:
    explicit VulkanContext(const RhiCreateInfo& createInfo)
        : m_commandContext(*this) {
        if (!createInfo.window) {
            throw std::runtime_error("Vulkan RHI requires a valid GLFW window.");
        }

        m_vkGetDeviceProcAddrProxy =
            reinterpret_cast<PFN_vkGetDeviceProcAddr>(createInfo.vkGetDeviceProcAddrProxy);
        m_requestedWidth = createInfo.width;
        m_requestedHeight = createInfo.height;
        createInstance(createInfo);
        createSurface(createInfo.window);
        pickPhysicalDevice(createInfo.requireVulkan14);
        populateLimits();
        createLogicalDevice(createInfo);
        m_pipelineCache.load(m_device, m_physicalDevice, createInfo.pipelineCacheDir);
        resolveStreamlinePresentHooks();
        m_gpuProfiler.init(m_physicalDevice, m_device, kMaxFramesInFlight, m_features.meshShaders);
        createVmaAllocator();
        createCommandObjects();
        createDescriptorPool();
        recreateSwapchain();
        populateNativeHandles();

        // Transient memory subsystems (after VMA is ready)
        constexpr VkDeviceSize kUploadRingSize  = 64 * 1024 * 1024; // 64 MB per frame
        constexpr VkDeviceSize kReadbackHeapSize = 16 * 1024 * 1024; // 16 MB per frame
        m_uploadRing.init(m_device, m_allocator, kUploadRingSize, kMaxFramesInFlight);
        m_transientPool.init(128, 64);
        m_readbackHeap.init(m_device, m_allocator, kReadbackHeapSize, kMaxFramesInFlight);

        vulkanSetResourceContext(m_device,
                                 m_physicalDevice,
                                 m_allocator,
                                 m_graphicsQueue,
                                 m_queueFamilies.graphics.value_or(0),
                                 m_queueFamilies.transfer.value_or(UINT32_MAX),
                                 m_features.bufferDeviceAddress,
                                 m_features.externalHostMemory,
                                 m_features.rayTracing,
                                 m_toolingInfo.debugUtils,
                                 createInfo.vkGetDeviceProcAddrProxy);
        vulkanLoadMeshShaderFunctions(m_device);
        applyDebugObjectNames();
        logSubgroupProperties(m_subgroupProperties);
    }

    ~VulkanContext() override {
        waitIdle();
        cleanupSwapchain();
        vulkanClearResourceContext();

        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        }

        for (auto& frame : m_frames) {
            if (frame.computeCommandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(m_device, frame.computeCommandPool, nullptr);
            }
            if (frame.commandPool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(m_device, frame.commandPool, nullptr);
            }
            if (frame.imageAvailable != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_device, frame.imageAvailable, nullptr);
            }
            if (frame.inFlight != VK_NULL_HANDLE) {
                vkDestroyFence(m_device, frame.inFlight, nullptr);
            }
        }

        if (m_computeTimelineSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, m_computeTimelineSemaphore, nullptr);
        }
        if (m_transferTimelineSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, m_transferTimelineSemaphore, nullptr);
        }

        m_uploadRing.destroy();
        m_transientPool.destroy();
        m_readbackHeap.destroy();
        m_gpuProfiler.destroy();

        if (m_allocator != nullptr) {
            vmaDestroyAllocator(m_allocator);
        }

        m_pipelineCache.save();
        m_pipelineCache.destroy();

        if (m_device != VK_NULL_HANDLE) {
            vkDestroyDevice(m_device, nullptr);
        }
        if (m_debugMessenger != VK_NULL_HANDLE &&
            m_vkDestroyDebugUtilsMessengerEXT &&
            m_instance != VK_NULL_HANDLE) {
            m_vkDestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
            m_debugMessenger = VK_NULL_HANDLE;
        }
        if (m_surface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        }
        if (m_instance != VK_NULL_HANDLE) {
            vkDestroyInstance(m_instance, nullptr);
        }
    }

    RhiBackendType backendType() const override { return RhiBackendType::Vulkan; }
    const RhiFeatures& features() const override { return m_features; }
    const RhiLimits& limits() const override { return m_limits; }
    const RhiDeviceInfo& deviceInfo() const override { return m_deviceInfo; }
    const RhiSubgroupProperties& subgroupProperties() const override { return m_subgroupProperties; }
    const RhiRayTracingPipelineProperties& rayTracingPipelineProperties() const override { return m_rtPipelineProperties; }
    const RhiNativeHandles& nativeHandles() const override { return m_nativeHandles; }
    const VulkanGpuFrameDiagnostics& latestFrameDiagnostics() const { return m_gpuProfiler.latestFrame(); }
    const VulkanToolingInfo& toolingInfo() const { return m_toolingInfo; }
    VulkanPipelineCacheTelemetry pipelineCacheTelemetry() const {
        VulkanPipelineCacheTelemetry telemetry;
        telemetry.graphicsPipelinesCompiled = m_pipelineCache.graphicsPipelinesCompiled();
        telemetry.computePipelinesCompiled = m_pipelineCache.computePipelinesCompiled();
        telemetry.totalCompileMs = m_pipelineCache.totalCompileMs();
        return telemetry;
    }
    bool isDeviceLost() const { return m_deviceLost; }
    const std::string& deviceLostMessage() const { return m_deviceLostMessage; }
    VulkanGpuProfiler* gpuProfiler() { return &m_gpuProfiler; }

    bool beginFrame() override {
        if (m_deviceLost) {
            return false;
        }
        if (m_pendingResize) {
            recreateSwapchain();
            m_pendingResize = false;
        }

        FrameResources& frame = m_frames[m_frameIndex];
        if (!handleRuntimeResult(vkWaitForFences(m_device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX),
                                 "Failed to wait for in-flight frame fence")) {
            return false;
        }

        VkResult acquireResult = acquireNextImageKHR(
            m_device,
            m_swapchain,
            UINT64_MAX,
            frame.imageAvailable,
            VK_NULL_HANDLE,
            &m_imageIndex);

        if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return false;
        }
        if (acquireResult == VK_ERROR_DEVICE_LOST) {
            markDeviceLost("Failed to acquire Vulkan swapchain image");
            return false;
        }
        if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire Vulkan swapchain image.");
        }

        if (m_imagesInFlight[m_imageIndex] != VK_NULL_HANDLE) {
            if (!handleRuntimeResult(vkWaitForFences(m_device, 1, &m_imagesInFlight[m_imageIndex], VK_TRUE, UINT64_MAX),
                                     "Failed to wait for a busy swapchain image fence")) {
                return false;
            }
        }
        m_imagesInFlight[m_imageIndex] = frame.inFlight;

        if (!handleRuntimeResult(vkResetFences(m_device, 1, &frame.inFlight), "Failed to reset frame fence")) {
            return false;
        }
        if (!handleRuntimeResult(vkResetCommandPool(m_device, frame.commandPool, 0), "Failed to reset command pool")) {
            return false;
        }
        if (frame.computeCommandPool != VK_NULL_HANDLE) {
            if (!handleRuntimeResult(vkResetCommandPool(m_device, frame.computeCommandPool, 0),
                                     "Failed to reset async compute command pool")) {
                return false;
            }
        }

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!handleRuntimeResult(vulkanBeginCommandBufferHooked(frame.commandBuffer, &beginInfo),
                                 "Failed to begin command buffer")) {
            return false;
        }
        m_gpuProfiler.beginFrame(m_frameIndex, m_submittedFrameCounter);
        m_gpuProfiler.resetActiveFrameQueries(frame.commandBuffer);
        if (frame.computeCommandBuffer != VK_NULL_HANDLE) {
            if (!handleRuntimeResult(vkBeginCommandBuffer(frame.computeCommandBuffer, &beginInfo),
                                     "Failed to begin async compute command buffer")) {
                return false;
            }
        }

        m_insideRendering = false;
        m_pendingGraphicsTimelineWaits.clear();

        // Notify transient subsystems that this frame's resources are safe to reuse.
        m_uploadRing.beginFrame(m_frameIndex);
        m_transientPool.beginFrame();
        m_readbackHeap.beginFrame(m_frameIndex);

        return true;
    }

    void endFrame() override {
        if (m_deviceLost) {
            return;
        }
        FrameResources& frame = m_frames[m_frameIndex];
        VkSemaphore renderFinished = m_swapchainRenderFinished[m_imageIndex];
        transitionCurrentImage(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                               VK_ACCESS_2_NONE);

        if (!handleRuntimeResult(vkEndCommandBuffer(frame.commandBuffer), "Failed to end command buffer")) {
            return;
        }

        // Upgraded to VkQueueSubmit2 for timeline semaphore compatibility.
        VkCommandBufferSubmitInfo cmdSubmitInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cmdSubmitInfo.commandBuffer = frame.commandBuffer;

        VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        waitInfo.semaphore = frame.imageAvailable;
        waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

        VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        signalInfo.semaphore = renderFinished;
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        std::vector<VkSemaphoreSubmitInfo> waitInfos;
        waitInfos.reserve(1 + m_pendingGraphicsTimelineWaits.size());
        waitInfos.push_back(waitInfo);
        for (const QueuedSemaphoreWait& queuedWait : m_pendingGraphicsTimelineWaits) {
            if (queuedWait.semaphore == VK_NULL_HANDLE) {
                continue;
            }

            VkSemaphoreSubmitInfo queuedWaitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            queuedWaitInfo.semaphore = queuedWait.semaphore;
            queuedWaitInfo.value = queuedWait.value;
            queuedWaitInfo.stageMask = queuedWait.stageMask;
            waitInfos.push_back(queuedWaitInfo);
        }

        VkSubmitInfo2 submitInfo2{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo2.waitSemaphoreInfoCount = static_cast<uint32_t>(waitInfos.size());
        submitInfo2.pWaitSemaphoreInfos = waitInfos.data();
        submitInfo2.commandBufferInfoCount = 1;
        submitInfo2.pCommandBufferInfos = &cmdSubmitInfo;
        submitInfo2.signalSemaphoreInfoCount = 1;
        submitInfo2.pSignalSemaphoreInfos = &signalInfo;

        if (!handleRuntimeResult(vkQueueSubmit2(m_graphicsQueue, 1, &submitInfo2, frame.inFlight),
                                 "Failed to submit graphics queue")) {
            return;
        }
        ++m_submittedFrameCounter;
        m_pendingGraphicsTimelineWaits.clear();

        VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderFinished;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &m_swapchain;
        presentInfo.pImageIndices = &m_imageIndex;

        const VkResult presentResult = queuePresentKHR(m_presentQueue, &presentInfo);
        if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR || m_pendingResize) {
            recreateSwapchain();
            m_pendingResize = false;
        } else if (presentResult == VK_ERROR_DEVICE_LOST) {
            markDeviceLost("Failed to present Vulkan swapchain image");
        } else if (presentResult != VK_SUCCESS) {
            throw std::runtime_error("Failed to present Vulkan swapchain image.");
        }

        m_frameIndex = (m_frameIndex + 1) % kMaxFramesInFlight;
    }

    void resize(uint32_t width, uint32_t height) override {
        if (width > 0 && height > 0) {
            m_requestedWidth = width;
            m_requestedHeight = height;
            m_pendingResize = true;
        }
    }

    void waitIdle() override {
        if (m_device != VK_NULL_HANDLE && !m_deviceLost) {
            const VkResult result = vkDeviceWaitIdle(m_device);
            if (result == VK_ERROR_DEVICE_LOST) {
                markDeviceLost("vkDeviceWaitIdle reported device lost");
            }
        }
    }

    RhiCommandContext& commandContext() override { return m_commandContext; }
    uint32_t drawableWidth() const override { return m_swapchainExtent.width; }
    uint32_t drawableHeight() const override { return m_swapchainExtent.height; }
    RhiFormat colorFormat() const override { return fromVkFormat(m_swapchainFormat.format); }
    RhiBufferHandle createSharedBuffer(const void* initialData,
                                       size_t size,
                                       const char* debugName) override {
        if (size == 0) {
            return {};
        }

        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                   VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        usage = vulkanEnableBufferDeviceAddress(usage, m_features.bufferDeviceAddress);
        usage = vulkanEnableAccelerationStructureBuildInput(usage, m_features.rayTracing);

        VmaBufferCreateInfo vmaInfo{};
        vmaInfo.device = m_device;
        vmaInfo.allocator = m_allocator;
        vmaInfo.size = size;
        vmaInfo.usage = usage;
        vmaInfo.hostVisible = true;
        vmaInfo.externalMemoryHandleTypes =
            vulkanHostVisibleExternalMemoryHandleTypes(vmaInfo.hostVisible,
                                                       m_features.externalHostMemory);
        vmaInfo.debugName = debugName;

        const char* errorMsg = nullptr;
        auto resource = vmaCreateBufferResource(vmaInfo, &errorMsg);
        if (!resource) {
            (void)errorMsg;
            return {};
        }

        auto* buffer = new VulkanBufferResource(*resource);
        if (initialData && buffer->mappedData) {
            std::memcpy(buffer->mappedData, initialData, size);
        }
        return RhiBufferHandle(buffer, size);
    }

    RhiTextureHandle createTexture2D(uint32_t width,
                                     uint32_t height,
                                     RhiFormat format,
                                     bool /*mipmapped*/,
                                     uint32_t mipLevelCount,
                                     RhiTextureStorageMode /*storageMode*/,
                                     RhiTextureUsage usage) override {
        VkFormat vkFormat = toVkFormat(format);
        VkImageUsageFlags vkUsage = toVkImageUsage(usage);
        const bool depth = isVkDepthFormat(vkFormat);
        if (depth) {
            vkUsage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            vkUsage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        }

        VmaImageCreateInfo vmaInfo{};
        vmaInfo.device = m_device;
        vmaInfo.allocator = m_allocator;
        vmaInfo.depth = depth;
        vmaInfo.imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        vmaInfo.imageInfo.imageType = VK_IMAGE_TYPE_2D;
        vmaInfo.imageInfo.format = vkFormat;
        vmaInfo.imageInfo.extent = {width, height, 1};
        vmaInfo.imageInfo.mipLevels = std::max(mipLevelCount, 1u);
        vmaInfo.imageInfo.arrayLayers = 1;
        vmaInfo.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        vmaInfo.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        vmaInfo.imageInfo.usage = vkUsage;
        vmaInfo.imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vmaInfo.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        const char* errorMsg = nullptr;
        auto resource = vmaCreateImageResource(vmaInfo, &errorMsg);
        if (!resource) {
            (void)errorMsg;
            return {};
        }

        auto* texture = new VulkanTextureResource(*resource);
        texture->usage = usage;
        return RhiTextureHandle(texture, width, height);
    }

    RhiTextureHandle createTexture3D(uint32_t width,
                                     uint32_t height,
                                     uint32_t depth,
                                     RhiFormat format,
                                     RhiTextureStorageMode /*storageMode*/,
                                     RhiTextureUsage usage) override {
        VmaImageCreateInfo vmaInfo{};
        vmaInfo.device = m_device;
        vmaInfo.allocator = m_allocator;
        vmaInfo.imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        vmaInfo.imageInfo.imageType = VK_IMAGE_TYPE_3D;
        vmaInfo.imageInfo.format = toVkFormat(format);
        vmaInfo.imageInfo.extent = {width, height, depth};
        vmaInfo.imageInfo.mipLevels = 1;
        vmaInfo.imageInfo.arrayLayers = 1;
        vmaInfo.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        vmaInfo.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        vmaInfo.imageInfo.usage = toVkImageUsage(usage);
        vmaInfo.imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vmaInfo.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        const char* errorMsg = nullptr;
        auto resource = vmaCreateImageResource(vmaInfo, &errorMsg);
        if (!resource) {
            (void)errorMsg;
            return {};
        }

        auto* texture = new VulkanTextureResource(*resource);
        texture->usage = usage;
        return RhiTextureHandle(texture, width, height);
    }

    RhiSamplerHandle createSampler(const RhiSamplerDesc& desc) override {
        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.minFilter = toVkFilter(desc.minFilter);
        samplerInfo.magFilter = toVkFilter(desc.magFilter);
        samplerInfo.mipmapMode = toVkMipFilter(desc.mipFilter);
        samplerInfo.addressModeU = toVkAddressMode(desc.addressModeS);
        samplerInfo.addressModeV = toVkAddressMode(desc.addressModeT);
        samplerInfo.addressModeW = toVkAddressMode(desc.addressModeR);
        samplerInfo.maxLod = desc.mipFilter != RhiSamplerMipFilterMode::None ? VK_LOD_CLAMP_NONE : 0.0f;
        samplerInfo.maxAnisotropy = 1.0f;

        auto* sampler = new VulkanSamplerResource{};
        sampler->device = m_device;
        if (vkCreateSampler(m_device, &samplerInfo, nullptr, &sampler->sampler) != VK_SUCCESS) {
            delete sampler;
            return {};
        }
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_SAMPLER,
                                 vkObjectHandle(sampler->sampler),
                                 "Metallic Sampler");
        return RhiSamplerHandle(sampler);
    }

    RhiDepthStencilStateHandle createDepthStencilState(bool /*depthWriteEnabled*/,
                                                       bool /*reversedZ*/) override {
        return RhiDepthStencilStateHandle(nullptr);
    }

    RhiShaderLibraryHandle createShaderLibraryFromSource(const std::string& /*source*/,
                                                         const RhiShaderLibrarySourceDesc& /*desc*/,
                                                         std::string& errorMessage) override {
        errorMessage =
            "Shader libraries from source are not supported on Vulkan (use SPIR-V pipeline creation)";
        return {};
    }

    RhiComputePipelineHandle createComputePipelineFromLibrary(const RhiShaderLibrary& /*library*/,
                                                              const char* /*entryPoint*/,
                                                              std::string& errorMessage) override {
        errorMessage =
            "Compute pipeline from library is not supported on Vulkan (use SPIR-V pipeline creation)";
        return {};
    }

    RhiGraphicsPipelineHandle createRenderPipelineFromSource(const std::string& source,
                                                             const RhiRenderPipelineSourceDesc& desc,
                                                             std::string& errorMessage) override {
        if (source.empty()) {
            errorMessage = "Empty shader source";
            return {};
        }
        if ((source.size() % sizeof(uint32_t)) != 0) {
            errorMessage = "SPIR-V source size is not aligned to 4 bytes";
            return {};
        }

        std::vector<uint32_t> spirv(source.size() / sizeof(uint32_t));
        std::memcpy(spirv.data(), source.data(), source.size());

        VkShaderModule shaderModule = createShaderModuleFromSpirv(m_device, spirv);
        if (shaderModule == VK_NULL_HANDLE) {
            errorMessage = "Failed to create VkShaderModule from SPIR-V";
            return {};
        }

        auto* resource = new VulkanPipelineResource{};
        resource->device = m_device;
        if (!buildPipelineResourceLayout(m_device, source.data(), source.size(), *resource, errorMessage, m_features.descriptorBuffer)) {
            vkDestroyShaderModule(m_device, shaderModule, nullptr);
            delete resource;
            return {};
        }

        std::vector<VkPipelineShaderStageCreateInfo> stages;
        const bool isMeshShader = desc.meshEntry && desc.meshEntry[0] != '\0';
        if (isMeshShader) {
            VkPipelineShaderStageCreateInfo meshStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            meshStage.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
            meshStage.module = shaderModule;
            meshStage.pName = desc.meshEntry;
            stages.push_back(meshStage);
        } else if (desc.vertexEntry && desc.vertexEntry[0] != '\0') {
            VkPipelineShaderStageCreateInfo vertexStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            vertexStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vertexStage.module = shaderModule;
            vertexStage.pName = desc.vertexEntry;
            stages.push_back(vertexStage);
        }

        if (desc.fragmentEntry && desc.fragmentEntry[0] != '\0') {
            VkPipelineShaderStageCreateInfo fragmentStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            fragmentStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragmentStage.module = shaderModule;
            fragmentStage.pName = desc.fragmentEntry;
            stages.push_back(fragmentStage);
        }

        std::vector<VkVertexInputBindingDescription> vertexBindings;
        std::vector<VkVertexInputAttributeDescription> vertexAttributes;
        if (!isMeshShader &&
            !buildVertexInputDescriptions(desc.vertexDescriptor, vertexBindings, vertexAttributes, errorMessage)) {
            vkDestroyShaderModule(m_device, shaderModule, nullptr);
            destroyPipelineLayouts(*resource);
            delete resource;
            return {};
        }

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindings.size());
        vertexInputInfo.pVertexBindingDescriptions =
            vertexBindings.empty() ? nullptr : vertexBindings.data();
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes.size());
        vertexInputInfo.pVertexAttributeDescriptions =
            vertexAttributes.empty() ? nullptr : vertexAttributes.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        std::array<VkDynamicState, 4> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_CULL_MODE,
            VK_DYNAMIC_STATE_FRONT_FACE,
        };
        VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        const bool hasDepth = desc.depthFormat != RhiFormat::Undefined;
        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        if (hasDepth) {
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
        }

        VkFormat colorFormat = toVkFormat(desc.colorFormat);
        VkFormat depthFormat = hasDepth ? toVkFormat(desc.depthFormat) : VK_FORMAT_UNDEFINED;
        VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;
        renderingInfo.depthAttachmentFormat = depthFormat;

        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
        pipelineInfo.pStages = stages.data();
        pipelineInfo.pVertexInputState = isMeshShader ? nullptr : &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = isMeshShader ? nullptr : &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.pDepthStencilState = hasDepth ? &depthStencil : nullptr;
        if (m_features.descriptorBuffer) {
            pipelineInfo.flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }
        pipelineInfo.layout = resource->layout;
        pipelineInfo.renderPass = VK_NULL_HANDLE;

        VkPipeline pipeline = VK_NULL_HANDLE;
        const auto t0gfx1 = std::chrono::high_resolution_clock::now();
        const VkResult result = vkCreateGraphicsPipelines(m_device,
                                                          m_pipelineCache.handle(),
                                                          1,
                                                          &pipelineInfo,
                                                          nullptr,
                                                          &pipeline);
        const double msGfx1 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0gfx1).count();
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
        if (result != VK_SUCCESS) {
            errorMessage = "Failed to create Vulkan graphics pipeline (VkResult: " +
                           std::to_string(result) + ")";
            destroyPipelineLayouts(*resource);
            delete resource;
            return {};
        }
        m_pipelineCache.recordCompile(msGfx1, /*isGraphics=*/true);
        spdlog::debug("VulkanPipeline: graphics pipeline compiled in {:.1f} ms", msGfx1);

        resource->pipeline = pipeline;
        if (resource->layout != VK_NULL_HANDLE) {
            std::string layoutName = "Graphics Pipeline Layout";
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                     vkObjectHandle(resource->layout),
                                     layoutName.c_str());
        }
        std::string pipelineName =
            std::string("Graphics PSO [") +
            (isMeshShader ? (desc.meshEntry ? desc.meshEntry : "meshMain")
                          : (desc.vertexEntry ? desc.vertexEntry : "vertexMain")) +
            " -> " +
            (desc.fragmentEntry ? desc.fragmentEntry : "<none>") +
            "]";
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_PIPELINE,
                                 vkObjectHandle(pipeline),
                                 pipelineName.c_str());
        return RhiGraphicsPipelineHandle(resource);
    }

    RhiComputePipelineHandle createComputePipelineFromSource(const std::string& source,
                                                             const char* entryPoint,
                                                             std::string& errorMessage) override {
        (void)entryPoint;
        if (source.empty()) {
            errorMessage = "Empty shader source";
            return {};
        }
        if ((source.size() % sizeof(uint32_t)) != 0) {
            errorMessage = "SPIR-V source size is not aligned to 4 bytes";
            return {};
        }

        std::vector<uint32_t> spirv(source.size() / sizeof(uint32_t));
        std::memcpy(spirv.data(), source.data(), source.size());

        VkShaderModule shaderModule = createShaderModuleFromSpirv(m_device, spirv);
        if (shaderModule == VK_NULL_HANDLE) {
            errorMessage = "Failed to create VkShaderModule from SPIR-V";
            return {};
        }

        auto* resource = new VulkanPipelineResource{};
        resource->device = m_device;
        if (!buildPipelineResourceLayout(m_device, source.data(), source.size(), *resource, errorMessage, m_features.descriptorBuffer)) {
            vkDestroyShaderModule(m_device, shaderModule, nullptr);
            delete resource;
            return {};
        }

        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = resource->layout;
        if (m_features.descriptorBuffer) {
            pipelineInfo.flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }

        VkPipeline pipeline = VK_NULL_HANDLE;
        const auto t0cmp = std::chrono::high_resolution_clock::now();
        const VkResult result = vkCreateComputePipelines(m_device,
                                                         m_pipelineCache.handle(),
                                                         1,
                                                         &pipelineInfo,
                                                         nullptr,
                                                         &pipeline);
        const double msCmp = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0cmp).count();
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
        if (result != VK_SUCCESS) {
            errorMessage = "Failed to create Vulkan compute pipeline (VkResult: " +
                           std::to_string(result) + ")";
            destroyPipelineLayouts(*resource);
            delete resource;
            return {};
        }
        m_pipelineCache.recordCompile(msCmp, /*isGraphics=*/false);
        spdlog::debug("VulkanPipeline: compute pipeline compiled in {:.1f} ms", msCmp);

        resource->pipeline = pipeline;
        if (resource->layout != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                     vkObjectHandle(resource->layout),
                                     "Compute Pipeline Layout");
        }
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_PIPELINE,
                                 vkObjectHandle(pipeline),
                                 "Compute PSO [main]");
        return RhiComputePipelineHandle(resource);
    }

    VkImage currentSwapchainImage() const {
        return m_imageIndex < m_swapchainImages.size() ? m_swapchainImages[m_imageIndex] : VK_NULL_HANDLE;
    }
    VkPipelineCache pipelineCacheHandle() const { return m_pipelineCache.handle(); }
    const VkPhysicalDeviceDescriptorBufferPropertiesEXT& descriptorBufferProperties() const {
        return m_descriptorBufferProperties;
    }
    VulkanUploadRing& uploadRing() { return m_uploadRing; }
    VulkanTransientPool& transientPool() { return m_transientPool; }
    VulkanReadbackHeap& readbackHeap() { return m_readbackHeap; }
    VkImageView currentSwapchainImageView() const {
        return m_imageIndex < m_swapchainImageViews.size() ? m_swapchainImageViews[m_imageIndex] : VK_NULL_HANDLE;
    }
    VkExtent2D currentSwapchainExtent() const { return m_swapchainExtent; }
    VkImageLayout currentSwapchainLayout() const {
        return m_imageIndex < m_swapchainImageLayouts.size()
                   ? m_swapchainImageLayouts[m_imageIndex]
                   : VK_IMAGE_LAYOUT_UNDEFINED;
    }

    VkCommandBuffer currentComputeCommandBuffer() const {
        return m_frames[m_frameIndex].computeCommandBuffer;
    }

    // End and submit the async compute command buffer to the dedicated compute queue.
    // Uses a timeline semaphore signal if available; otherwise a simple submit.
    // Returns the timeline value to wait on (0 if no timeline semaphore).
    uint64_t scheduleAsyncComputeSubmit() {
        if (m_deviceLost) {
            return 0;
        }
        const FrameResources& frame = m_frames[m_frameIndex];
        if (frame.computeCommandBuffer == VK_NULL_HANDLE || m_computeQueue == VK_NULL_HANDLE) {
            return 0;
        }

        if (!handleRuntimeResult(vkEndCommandBuffer(frame.computeCommandBuffer),
                                 "Failed to end async compute command buffer")) {
            return 0;
        }

        VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cmdInfo.commandBuffer = frame.computeCommandBuffer;

        VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdInfo;

        uint64_t signalValue = 0;
        VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        if (m_computeTimelineSemaphore != VK_NULL_HANDLE) {
            ++m_computeTimelineValue;
            signalValue = m_computeTimelineValue;
            signalInfo.semaphore = m_computeTimelineSemaphore;
            signalInfo.value = signalValue;
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            submitInfo.signalSemaphoreInfoCount = 1;
            submitInfo.pSignalSemaphoreInfos = &signalInfo;
        }

        if (!handleRuntimeResult(vkQueueSubmit2(m_computeQueue, 1, &submitInfo, VK_NULL_HANDLE),
                                 "Failed to submit async compute queue")) {
            return 0;
        }
        return signalValue;
    }

    bool hasEnabledExtension(const char* extensionName) const {
        return std::any_of(m_enabledExtensions.begin(), m_enabledExtensions.end(),
                           [extensionName](const std::string& ext) { return ext == extensionName; });
    }

    void enqueueGraphicsTimelineWait(VkSemaphore semaphore,
                                     uint64_t value,
                                     VkPipelineStageFlags2 stageMask) {
        if (semaphore == VK_NULL_HANDLE) {
            return;
        }

        QueuedSemaphoreWait wait{};
        wait.semaphore = semaphore;
        wait.value = value;
        wait.stageMask = stageMask != 0 ? stageMask : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        m_pendingGraphicsTimelineWaits.push_back(wait);
    }

    std::unique_ptr<RhiShaderModule> createShaderModule(const RhiShaderModuleDesc& desc) override {
        if (desc.spirv.empty()) {
            throw std::runtime_error("Cannot create Vulkan shader module from an empty SPIR-V blob.");
        }

        VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        createInfo.codeSize = desc.spirv.size() * sizeof(uint32_t);
        createInfo.pCode = desc.spirv.data();

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        checkVk(vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule),
                "Failed to create Vulkan shader module");
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_SHADER_MODULE,
                                 vkObjectHandle(shaderModule),
                                 desc.debugName ? desc.debugName : "Shader Module");
        return std::make_unique<VulkanShaderModule>(m_device, shaderModule);
    }

    std::unique_ptr<RhiBuffer> createVertexBuffer(const RhiBufferDesc& desc) override {
        if (desc.size == 0) {
            throw std::runtime_error("Cannot create a zero-sized Vulkan buffer.");
        }

        VmaBufferCreateInfo vmaInfo{};
        vmaInfo.device = m_device;
        vmaInfo.allocator = m_allocator;
        vmaInfo.size = desc.size;
        vmaInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vmaInfo.usage = vulkanEnableBufferDeviceAddress(vmaInfo.usage, m_features.bufferDeviceAddress);
        vmaInfo.hostVisible = desc.hostVisible;
        vmaInfo.sharedWithTransferQueue = desc.sharedWithTransferQueue;
        vmaInfo.graphicsQueueFamily = m_queueFamilies.graphics.value_or(0);
        vmaInfo.transferQueueFamily = m_queueFamilies.transfer.value_or(UINT32_MAX);
        vmaInfo.externalMemoryHandleTypes =
            vulkanHostVisibleExternalMemoryHandleTypes(vmaInfo.hostVisible,
                                                       m_features.externalHostMemory);

        auto resource = vmaCreateBufferResource(vmaInfo);
        if (!resource) {
            throw std::runtime_error("Failed to create Vulkan vertex buffer");
        }

        if (desc.initialData) {
            if (!desc.hostVisible) {
                throw std::runtime_error("Device-local uploads are not implemented yet for Vulkan vertex buffers.");
            }
            if (!resource->mappedData) {
                throw std::runtime_error("Failed to obtain mapped memory for Vulkan vertex buffer.");
            }
            std::memcpy(resource->mappedData, desc.initialData, desc.size);
        }

        return std::make_unique<VulkanBuffer>(*resource);
    }

    std::unique_ptr<RhiGraphicsPipeline> createGraphicsPipeline(const RhiGraphicsPipelineDesc& desc) override {
        const auto* shaderModule = dynamic_cast<const VulkanShaderModule*>(desc.shaderModule);
        if (!shaderModule) {
            throw std::runtime_error("Vulkan graphics pipeline requires a Vulkan shader module.");
        }

        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        auto pushStage = [&](VkShaderStageFlagBits stage, const char* entryPoint) {
            VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stageInfo.stage = stage;
            stageInfo.module = shaderModule->handle();
            stageInfo.pName = entryPoint;
            shaderStages.push_back(stageInfo);
        };

        if (desc.enableMeshShaders) {
            if (!m_features.meshShaders) {
                throw std::runtime_error("The active Vulkan device does not support mesh shaders.");
            }
            if (desc.taskEntry && desc.taskEntry[0] != '\0') {
                pushStage(VK_SHADER_STAGE_TASK_BIT_EXT, desc.taskEntry);
            }
            pushStage(VK_SHADER_STAGE_MESH_BIT_EXT, desc.meshEntry ? desc.meshEntry : "meshMain");
        } else {
            pushStage(VK_SHADER_STAGE_VERTEX_BIT, desc.vertexEntry ? desc.vertexEntry : "vertexMain");
        }

        if (desc.fragmentEntry && desc.fragmentEntry[0] != '\0') {
            pushStage(VK_SHADER_STAGE_FRAGMENT_BIT, desc.fragmentEntry);
        }

        std::vector<VkVertexInputBindingDescription> bindings;
        bindings.reserve(desc.bindings.size());
        for (const auto& binding : desc.bindings) {
            bindings.push_back({binding.binding, binding.stride, VK_VERTEX_INPUT_RATE_VERTEX});
        }

        std::vector<VkVertexInputAttributeDescription> attributes;
        attributes.reserve(desc.attributes.size());
        for (const auto& attribute : desc.attributes) {
            attributes.push_back({attribute.location, attribute.binding, toVkVertexFormat(attribute.format), attribute.offset});
        }

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(bindings.size());
        vertexInputInfo.pVertexBindingDescriptions = bindings.data();
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributes.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = desc.enableDepth ? VK_TRUE : VK_FALSE;
        depthStencil.depthWriteEnable = desc.enableDepth ? VK_TRUE : VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        VkPipelineLayout layout = VK_NULL_HANDLE;
        checkVk(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &layout),
                "Failed to create Vulkan pipeline layout");

        VkFormat colorFormat = toVkFormat(desc.colorFormat);
        VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;

        VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipelineInfo.pNext = &renderingInfo;
        pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = desc.enableMeshShaders ? nullptr : &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = desc.enableMeshShaders ? nullptr : &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = layout;
        pipelineInfo.renderPass = VK_NULL_HANDLE;
        pipelineInfo.subpass = 0;
        pipelineInfo.pDepthStencilState = desc.enableDepth ? &depthStencil : nullptr;
        if (m_features.descriptorBuffer) {
            pipelineInfo.flags |= VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }

        VkPipeline pipeline = VK_NULL_HANDLE;
        const auto t0gfx2 = std::chrono::high_resolution_clock::now();
        const VkResult pipelineResult = vkCreateGraphicsPipelines(m_device, m_pipelineCache.handle(), 1, &pipelineInfo, nullptr, &pipeline);
        const double msGfx2 = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0gfx2).count();
        if (pipelineResult != VK_SUCCESS) {
            vkDestroyPipelineLayout(m_device, layout, nullptr);
            throw std::runtime_error("Failed to create Vulkan graphics pipeline.");
        }
        m_pipelineCache.recordCompile(msGfx2, /*isGraphics=*/true);
        spdlog::debug("VulkanPipeline: graphics pipeline compiled in {:.1f} ms", msGfx2);

        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                                 vkObjectHandle(layout),
                                 "Legacy Graphics Pipeline Layout");
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_PIPELINE,
                                 vkObjectHandle(pipeline),
                                 "Legacy Graphics Pipeline");

        return std::make_unique<VulkanGraphicsPipeline>(m_device, layout, pipeline);
    }

 private:
    void resolveStreamlinePresentHooks() {
        if (!m_vkGetDeviceProcAddrProxy || m_device == VK_NULL_HANDLE) {
            return;
        }

        m_vkAcquireNextImageKHRProxy =
            reinterpret_cast<PFN_vkAcquireNextImageKHR>(m_vkGetDeviceProcAddrProxy(m_device, "vkAcquireNextImageKHR"));
        m_vkQueuePresentKHRProxy =
            reinterpret_cast<PFN_vkQueuePresentKHR>(m_vkGetDeviceProcAddrProxy(m_device, "vkQueuePresentKHR"));
    }

    VkResult acquireNextImageKHR(VkDevice device,
                                 VkSwapchainKHR swapchain,
                                 uint64_t timeout,
                                 VkSemaphore semaphore,
                                 VkFence fence,
                                 uint32_t* imageIndex) const {
        if (m_vkAcquireNextImageKHRProxy) {
            return m_vkAcquireNextImageKHRProxy(device, swapchain, timeout, semaphore, fence, imageIndex);
        }
        return vkAcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, imageIndex);
    }

    VkResult queuePresentKHR(VkQueue queue, const VkPresentInfoKHR* presentInfo) const {
        if (m_vkQueuePresentKHRProxy) {
            return m_vkQueuePresentKHRProxy(queue, presentInfo);
        }
        return vkQueuePresentKHR(queue, presentInfo);
    }

    struct FrameResources {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        VkCommandPool computeCommandPool = VK_NULL_HANDLE;
        VkCommandBuffer computeCommandBuffer = VK_NULL_HANDLE;
    };

    struct QueuedSemaphoreWait {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        uint64_t value = 0;
        VkPipelineStageFlags2 stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    };

    class CommandContext final : public RhiCommandContext {
    public:
        explicit CommandContext(VulkanContext& parent)
            : m_parent(parent) {}

        void beginRendering(const RhiRenderTargetInfo& targetInfo) override {
            if (m_parent.m_insideRendering) {
                throw std::runtime_error("Vulkan rendering has already begun for this frame.");
            }

            VkImageLayout currentLayout = m_parent.m_swapchainImageLayouts[m_parent.m_imageIndex];
            m_parent.transitionCurrentImage(currentLayout,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                            VK_ACCESS_2_NONE,
                                            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

            VkClearValue clearValue{};
            clearValue.color = {{
                targetInfo.clearColor[0],
                targetInfo.clearColor[1],
                targetInfo.clearColor[2],
                targetInfo.clearColor[3],
            }};

            VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            colorAttachment.imageView = m_parent.m_swapchainImageViews[m_parent.m_imageIndex];
            colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorAttachment.loadOp = targetInfo.clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            colorAttachment.clearValue = clearValue;

            VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
            renderingInfo.renderArea.offset = {0, 0};
            renderingInfo.renderArea.extent = m_parent.m_swapchainExtent;
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.pColorAttachments = &colorAttachment;

            vkCmdBeginRendering(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer, &renderingInfo);
            m_parent.m_insideRendering = true;
        }

        void endRendering() override {
            if (!m_parent.m_insideRendering) {
                return;
            }

            vkCmdEndRendering(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer);
            m_parent.m_insideRendering = false;
        }

        void setViewport(float width, float height) override {
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = height;
            viewport.width = width;
            viewport.height = -height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer, 0, 1, &viewport);
        }

        void setScissor(uint32_t width, uint32_t height) override {
            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = {width, height};
            vkCmdSetScissor(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer, 0, 1, &scissor);
        }

        void bindGraphicsPipeline(const RhiGraphicsPipeline& pipeline) override {
            const auto& vkPipeline = dynamic_cast<const VulkanGraphicsPipeline&>(pipeline);
            vulkanCmdBindPipelineHooked(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer,
                                        VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        vkPipeline.pipeline());
        }

        void bindVertexBuffer(const RhiBuffer& buffer, uint64_t offset) override {
            const auto& vkBuffer = dynamic_cast<const VulkanBuffer&>(buffer);
            VkBuffer vertexBuffer = vkBuffer.handle();
            VkDeviceSize vertexOffset = offset;
            vkCmdBindVertexBuffers(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer,
                                   0, 1, &vertexBuffer, &vertexOffset);
        }

        void draw(uint32_t vertexCount,
                  uint32_t instanceCount,
                  uint32_t firstVertex,
                  uint32_t firstInstance) override {
            vkCmdDraw(m_parent.m_frames[m_parent.m_frameIndex].commandBuffer,
                      vertexCount, instanceCount, firstVertex, firstInstance);
        }

        void* nativeCommandBuffer() const override {
            return m_parent.m_frames[m_parent.m_frameIndex].commandBuffer;
        }

    private:
        VulkanContext& m_parent;
    };

    bool handleRuntimeResult(VkResult result, const char* message) {
        if (result == VK_SUCCESS) {
            return true;
        }
        if (result == VK_ERROR_DEVICE_LOST) {
            markDeviceLost(message);
            return false;
        }
        throw std::runtime_error(std::string(message) + " (VkResult: " + std::to_string(result) + ")");
    }

    void markDeviceLost(const char* message) {
        if (m_deviceLost) {
            return;
        }

        m_deviceLost = true;
        m_deviceLostMessage = message ? message : "Vulkan device lost";
        spdlog::critical("Vulkan device lost: {}", m_deviceLostMessage);
        if (m_toolingInfo.debugUtils) {
            spdlog::critical("Vulkan diagnostics: debug utils markers were enabled for this session.");
        }
        if (m_toolingInfo.pipelineStatistics) {
            spdlog::critical("Vulkan diagnostics: GPU timestamp/pipeline statistics capture was enabled.");
        }
    }

    void applyDebugObjectNames() {
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_DEVICE,
                                 vkObjectHandle(m_device),
                                 "Metallic Device");
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_QUEUE,
                                 vkObjectHandle(m_graphicsQueue),
                                 "Metallic Graphics Queue");
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_QUEUE,
                                 vkObjectHandle(m_presentQueue),
                                 "Metallic Present Queue");
        if (m_computeQueue != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_QUEUE,
                                     vkObjectHandle(m_computeQueue),
                                     "Metallic Async Compute Queue");
        }
        if (m_transferQueue != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_QUEUE,
                                     vkObjectHandle(m_transferQueue),
                                     "Metallic Transfer Queue");
        }
        if (m_descriptorPool != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_DESCRIPTOR_POOL,
                                     vkObjectHandle(m_descriptorPool),
                                     "Metallic Descriptor Pool");
        }
        if (m_pipelineCache.handle() != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_PIPELINE_CACHE,
                                     vkObjectHandle(m_pipelineCache.handle()),
                                     "Metallic Pipeline Cache");
        }
        for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
            const std::string poolName = "Frame " + std::to_string(i) + " Graphics Command Pool";
            const std::string cmdName = "Frame " + std::to_string(i) + " Graphics Command Buffer";
            const std::string imageAvailableName = "Frame " + std::to_string(i) + " Image Available";
            const std::string frameFenceName = "Frame " + std::to_string(i) + " In Flight Fence";
            if (m_frames[i].commandPool != VK_NULL_HANDLE) {
                vulkanSetObjectDebugName(m_device,
                                         VK_OBJECT_TYPE_COMMAND_POOL,
                                         vkObjectHandle(m_frames[i].commandPool),
                                         poolName.c_str());
            }
            if (m_frames[i].commandBuffer != VK_NULL_HANDLE) {
                vulkanSetObjectDebugName(m_device,
                                         VK_OBJECT_TYPE_COMMAND_BUFFER,
                                         vkObjectHandle(m_frames[i].commandBuffer),
                                         cmdName.c_str());
            }
            if (m_frames[i].imageAvailable != VK_NULL_HANDLE) {
                vulkanSetObjectDebugName(m_device,
                                         VK_OBJECT_TYPE_SEMAPHORE,
                                         vkObjectHandle(m_frames[i].imageAvailable),
                                         imageAvailableName.c_str());
            }
            if (m_frames[i].inFlight != VK_NULL_HANDLE) {
                vulkanSetObjectDebugName(m_device,
                                         VK_OBJECT_TYPE_FENCE,
                                         vkObjectHandle(m_frames[i].inFlight),
                                         frameFenceName.c_str());
            }
            if (m_frames[i].computeCommandPool != VK_NULL_HANDLE) {
                const std::string computePoolName =
                    "Frame " + std::to_string(i) + " Compute Command Pool";
                vulkanSetObjectDebugName(m_device,
                                         VK_OBJECT_TYPE_COMMAND_POOL,
                                         vkObjectHandle(m_frames[i].computeCommandPool),
                                         computePoolName.c_str());
            }
            if (m_frames[i].computeCommandBuffer != VK_NULL_HANDLE) {
                const std::string computeCmdName =
                    "Frame " + std::to_string(i) + " Compute Command Buffer";
                vulkanSetObjectDebugName(m_device,
                                         VK_OBJECT_TYPE_COMMAND_BUFFER,
                                         vkObjectHandle(m_frames[i].computeCommandBuffer),
                                         computeCmdName.c_str());
            }
        }
        if (m_swapchain != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_SWAPCHAIN_KHR,
                                     vkObjectHandle(m_swapchain),
                                     "Metallic Swapchain");
        }
        for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
            const std::string imageName = "Swapchain Image " + std::to_string(i);
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_IMAGE,
                                     vkObjectHandle(m_swapchainImages[i]),
                                     imageName.c_str());
            if (i < m_swapchainImageViews.size() && m_swapchainImageViews[i] != VK_NULL_HANDLE) {
                const std::string viewName = imageName + " View";
                vulkanSetObjectDebugName(m_device,
                                         VK_OBJECT_TYPE_IMAGE_VIEW,
                                         vkObjectHandle(m_swapchainImageViews[i]),
                                         viewName.c_str());
            }
            if (i < m_swapchainRenderFinished.size() && m_swapchainRenderFinished[i] != VK_NULL_HANDLE) {
                const std::string semaphoreName = imageName + " Render Finished";
                vulkanSetObjectDebugName(m_device,
                                         VK_OBJECT_TYPE_SEMAPHORE,
                                         vkObjectHandle(m_swapchainRenderFinished[i]),
                                         semaphoreName.c_str());
            }
        }
        if (m_computeTimelineSemaphore != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_SEMAPHORE,
                                     vkObjectHandle(m_computeTimelineSemaphore),
                                     "Async Compute Timeline");
        }
        if (m_transferTimelineSemaphore != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_SEMAPHORE,
                                     vkObjectHandle(m_transferTimelineSemaphore),
                                     "Transfer Timeline");
        }
    }

    void createInstance(const RhiCreateInfo& createInfo) {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        if (!glfwExtensions) {
            throw std::runtime_error("GLFW did not provide Vulkan instance extensions.");
        }

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        for (auto ext : createInfo.extraInstanceExtensions) {
            extensions.push_back(ext);
        }

        uint32_t availableExtensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount, availableExtensions.data());
        m_toolingInfo.debugUtils = hasExtension(availableExtensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        if (m_toolingInfo.debugUtils) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        // Deduplicate instance extensions
        std::sort(extensions.begin(), extensions.end(),
                  [](const char* a, const char* b) { return std::string_view(a) < std::string_view(b); });
        extensions.erase(
            std::unique(extensions.begin(), extensions.end(),
                [](const char* a, const char* b) { return std::string_view(a) == std::string_view(b); }),
            extensions.end());

        uint32_t availableLayerCount = 0;
        vkEnumerateInstanceLayerProperties(&availableLayerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(availableLayerCount);
        vkEnumerateInstanceLayerProperties(&availableLayerCount, availableLayers.data());

        std::vector<const char*> layers;
        const bool validationAvailable = hasLayer(availableLayers, kValidationLayerName);
        m_toolingInfo.renderDocLayerAvailable = hasLayer(availableLayers, kRenderDocLayerName);
        if (createInfo.enableValidation && validationAvailable) {
            layers.push_back(kValidationLayerName);
            m_features.validation = true;
        }

        VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        appInfo.pApplicationName = createInfo.applicationName;
        appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.pEngineName = "Metallic";
        appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
        appInfo.apiVersion = VK_API_VERSION_1_4;

        VkInstanceCreateInfo createInfoVk{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        createInfoVk.pApplicationInfo = &appInfo;
        createInfoVk.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfoVk.ppEnabledExtensionNames = extensions.data();
        createInfoVk.enabledLayerCount = static_cast<uint32_t>(layers.size());
        createInfoVk.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = makeDebugMessengerCreateInfo();
        if (m_features.validation && m_toolingInfo.debugUtils) {
            createInfoVk.pNext = &debugCreateInfo;
        }

        checkVk(vkCreateInstance(&createInfoVk, nullptr, &m_instance), "Failed to create Vulkan instance");

        if (m_toolingInfo.debugUtils) {
            m_vkCreateDebugUtilsMessengerEXT =
                reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));
            m_vkDestroyDebugUtilsMessengerEXT =
                reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
            if (m_features.validation &&
                m_vkCreateDebugUtilsMessengerEXT &&
                m_vkCreateDebugUtilsMessengerEXT(m_instance,
                                                 &debugCreateInfo,
                                                 nullptr,
                                                 &m_debugMessenger) == VK_SUCCESS) {
                m_toolingInfo.validationMessenger = true;
            }
        }
    }

    void createSurface(GLFWwindow* window) {
        checkVk(glfwCreateWindowSurface(m_instance, window, nullptr, &m_surface), "Failed to create Vulkan surface");
    }

    void pickPhysicalDevice(bool requireVulkan14) {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("No Vulkan physical devices were found.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

        for (VkPhysicalDevice device : devices) {
            if (isDeviceSuitable(device, requireVulkan14)) {
                m_physicalDevice = device;
                vkGetPhysicalDeviceProperties(device, &m_physicalDeviceProperties);
                m_queueFamilies = findQueueFamilies(device, m_surface);
                m_deviceInfo.adapterName = m_physicalDeviceProperties.deviceName;
                m_deviceInfo.driverName = vkVersionString(m_physicalDeviceProperties.driverVersion);
                m_deviceInfo.apiVersion = m_physicalDeviceProperties.apiVersion;
                return;
            }
        }

        throw std::runtime_error("No suitable Vulkan 1.4 device was found for Metallic.");
    }

    void populateLimits() {
        const VkPhysicalDeviceLimits& vkLimits = m_physicalDeviceProperties.limits;

        // Push constants
        m_limits.maxPushConstantSize = vkLimits.maxPushConstantsSize;

        // Uniform buffers
        m_limits.minUniformBufferOffsetAlignment = vkLimits.minUniformBufferOffsetAlignment;
        m_limits.maxUniformBufferRange = vkLimits.maxUniformBufferRange;

        // Storage buffers
        m_limits.maxStorageBufferRange = vkLimits.maxStorageBufferRange;

        // Compute
        m_limits.maxComputeWorkGroupCountX = vkLimits.maxComputeWorkGroupCount[0];
        m_limits.maxComputeWorkGroupCountY = vkLimits.maxComputeWorkGroupCount[1];
        m_limits.maxComputeWorkGroupCountZ = vkLimits.maxComputeWorkGroupCount[2];
        m_limits.maxComputeWorkGroupInvocations = vkLimits.maxComputeWorkGroupInvocations;
        m_limits.maxComputeWorkGroupSizeX = vkLimits.maxComputeWorkGroupSize[0];
        m_limits.maxComputeWorkGroupSizeY = vkLimits.maxComputeWorkGroupSize[1];
        m_limits.maxComputeWorkGroupSizeZ = vkLimits.maxComputeWorkGroupSize[2];

        // Mesh shaders (via VkPhysicalDeviceProperties2 chain)
        if (m_features.meshShaders) {
            VkPhysicalDeviceMeshShaderPropertiesEXT meshProps{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            props2.pNext = &meshProps;
            vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

            m_limits.maxMeshOutputVertices = meshProps.maxMeshOutputVertices;
            m_limits.maxMeshOutputPrimitives = meshProps.maxMeshOutputPrimitives;
            m_limits.maxMeshWorkGroupInvocations = meshProps.maxMeshWorkGroupInvocations;
            m_limits.maxTaskWorkGroupInvocations = meshProps.maxTaskWorkGroupInvocations;
        }

        // Descriptors
        m_limits.maxBoundDescriptorSets = vkLimits.maxBoundDescriptorSets;
        m_limits.maxPerStageDescriptorSamplers = vkLimits.maxPerStageDescriptorSamplers;
        m_limits.maxPerStageDescriptorUniformBuffers = vkLimits.maxPerStageDescriptorUniformBuffers;
        m_limits.maxPerStageDescriptorStorageBuffers = vkLimits.maxPerStageDescriptorStorageBuffers;
        m_limits.maxPerStageDescriptorSampledImages = vkLimits.maxPerStageDescriptorSampledImages;
        m_limits.maxPerStageDescriptorStorageImages = vkLimits.maxPerStageDescriptorStorageImages;
        m_limits.maxDescriptorSetSamplers = vkLimits.maxDescriptorSetSamplers;
        m_limits.maxDescriptorSetUniformBuffers = vkLimits.maxDescriptorSetUniformBuffers;
        m_limits.maxDescriptorSetStorageBuffers = vkLimits.maxDescriptorSetStorageBuffers;
        m_limits.maxDescriptorSetSampledImages = vkLimits.maxDescriptorSetSampledImages;
        m_limits.maxDescriptorSetStorageImages = vkLimits.maxDescriptorSetStorageImages;

        // Memory
        m_limits.nonCoherentAtomSize = vkLimits.nonCoherentAtomSize;

        // Textures / framebuffer
        m_limits.maxImageDimension2D = vkLimits.maxImageDimension2D;
        m_limits.maxFramebufferWidth = vkLimits.maxFramebufferWidth;
        m_limits.maxFramebufferHeight = vkLimits.maxFramebufferHeight;
        m_limits.maxColorAttachments = vkLimits.maxColorAttachments;

        // Timing
        m_limits.timestampPeriod = vkLimits.timestampPeriod;

        // Subgroup / wave properties
        m_subgroupProperties = {};
        {
            VkPhysicalDeviceSubgroupProperties subgroupProps{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
            VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeProps{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            props2.pNext = &subgroupProps;
            if (m_subgroupSizeControlSupported) {
                subgroupProps.pNext = &subgroupSizeProps;
            }
            vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

            m_subgroupProperties.subgroupSize = subgroupProps.subgroupSize;
            m_subgroupProperties.supportedStages =
                toRhiSubgroupStages(subgroupProps.supportedStages);
            m_subgroupProperties.supportedOperations =
                toRhiSubgroupOperations(subgroupProps.supportedOperations);
            m_subgroupProperties.quadOperationsInAllStages =
                subgroupProps.quadOperationsInAllStages == VK_TRUE;
            m_subgroupProperties.sizeControl = m_subgroupSizeControlSupported;
            m_subgroupProperties.computeFullSubgroups = m_computeFullSubgroupsSupported;
            if (m_subgroupSizeControlSupported) {
                m_subgroupProperties.minSubgroupSize = subgroupSizeProps.minSubgroupSize;
                m_subgroupProperties.maxSubgroupSize = subgroupSizeProps.maxSubgroupSize;
                m_subgroupProperties.maxComputeWorkgroupSubgroups =
                    subgroupSizeProps.maxComputeWorkgroupSubgroups;
                m_subgroupProperties.requiredSubgroupSizeStages =
                    toRhiSubgroupStages(subgroupSizeProps.requiredSubgroupSizeStages);
            }
            m_subgroupProperties.supported =
                m_subgroupProperties.subgroupSize != 0 &&
                m_subgroupProperties.supportedStages != RhiSubgroupStage::None;
        }

        // Descriptor buffer properties (VK_EXT_descriptor_buffer)
        if (m_features.descriptorBuffer) {
            m_descriptorBufferProperties = {
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            props2.pNext = &m_descriptorBufferProperties;
            vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

            m_limits.descriptorBufferOffsetAlignment =
                m_descriptorBufferProperties.descriptorBufferOffsetAlignment;
            m_limits.sampledImageDescriptorSize =
                m_descriptorBufferProperties.sampledImageDescriptorSize;
            m_limits.samplerDescriptorSize =
                m_descriptorBufferProperties.samplerDescriptorSize;
            m_limits.storageImageDescriptorSize =
                m_descriptorBufferProperties.storageImageDescriptorSize;
            m_limits.storageBufferDescriptorSize =
                m_descriptorBufferProperties.storageBufferDescriptorSize;
            m_limits.uniformBufferDescriptorSize =
                m_descriptorBufferProperties.uniformBufferDescriptorSize;
            m_limits.accelerationStructureDescriptorSize =
                m_descriptorBufferProperties.accelerationStructureDescriptorSize;

            spdlog::info("Vulkan: descriptor buffer enabled "
                         "(offsetAlign={}, sampledImg={}, sampler={}, storageImg={}, storageBuf={}, ubo={}, as={})",
                         m_limits.descriptorBufferOffsetAlignment,
                         m_limits.sampledImageDescriptorSize,
                         m_limits.samplerDescriptorSize,
                         m_limits.storageImageDescriptorSize,
                         m_limits.storageBufferDescriptorSize,
                         m_limits.uniformBufferDescriptorSize,
                         m_limits.accelerationStructureDescriptorSize);
        }

        // Validate push constant size
        if (m_limits.maxPushConstantSize < kPushConstantSize) {
            spdlog::warn("Device maxPushConstantsSize ({}) < required kPushConstantSize ({})",
                         m_limits.maxPushConstantSize, kPushConstantSize);
        }

        // Validate bindless limits against device capabilities
        if (m_limits.maxDescriptorSetSampledImages < kVulkanBindlessMaxSampledImages) {
            spdlog::warn("Device maxDescriptorSetSampledImages ({}) < METALLIC_BINDLESS_MAX_SAMPLED_IMAGES ({})",
                         m_limits.maxDescriptorSetSampledImages, kVulkanBindlessMaxSampledImages);
        }
        if (m_limits.maxDescriptorSetStorageBuffers < kVulkanBindlessMaxStorageBuffers) {
            spdlog::warn("Device maxDescriptorSetStorageBuffers ({}) < METALLIC_BINDLESS_MAX_STORAGE_BUFFERS ({})",
                         m_limits.maxDescriptorSetStorageBuffers, kVulkanBindlessMaxStorageBuffers);
        }
        if (m_limits.maxDescriptorSetStorageImages < kVulkanBindlessMaxStorageImages) {
            spdlog::warn("Device maxDescriptorSetStorageImages ({}) < METALLIC_BINDLESS_MAX_STORAGE_IMAGES ({})",
                         m_limits.maxDescriptorSetStorageImages, kVulkanBindlessMaxStorageImages);
        }
        if (m_limits.maxDescriptorSetSamplers < kVulkanBindlessMaxSamplers) {
            spdlog::warn("Device maxDescriptorSetSamplers ({}) < METALLIC_BINDLESS_MAX_SAMPLERS ({})",
                         m_limits.maxDescriptorSetSamplers, kVulkanBindlessMaxSamplers);
        }
    }

    bool isDeviceSuitable(VkPhysicalDevice device, bool requireVulkan14) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);
        if (requireVulkan14 && VK_API_VERSION_MAJOR(properties.apiVersion) == 1 && VK_API_VERSION_MINOR(properties.apiVersion) < 4) {
            return false;
        }
        if (requireVulkan14 && VK_API_VERSION_MAJOR(properties.apiVersion) < 1) {
            return false;
        }

        QueueFamilyIndices indices = findQueueFamilies(device, m_surface);
        if (!indices.complete()) {
            return false;
        }

        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

        if (!hasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            return false;
        }

        const bool meshShaderAvailable = hasExtension(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);
        const bool dynamicRenderingAvailable = hasExtension(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) ||
                                               properties.apiVersion >= VK_API_VERSION_1_3;
        const bool sync2Available = hasExtension(extensions, VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME) ||
                                    properties.apiVersion >= VK_API_VERSION_1_3;
        const bool bufferDeviceAddressAvailable =
            hasExtension(extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) ||
            properties.apiVersion >= VK_API_VERSION_1_2;
        const bool accelerationStructureAvailable =
            hasExtension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        const bool rayQueryAvailable = hasExtension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
        const bool rayTracingPipelineAvailable =
            hasExtension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        const bool deferredHostOperationsAvailable =
            hasExtension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        const bool externalMemoryHostAvailable =
            hasExtension(extensions, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        const bool descriptorBufferAvailable =
            hasExtension(extensions, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
        const bool subgroupSizeControlAvailable =
            hasExtension(extensions, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) ||
            properties.apiVersion >= VK_API_VERSION_1_3;
        const bool robustness2Available =
            hasExtension(extensions, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
        m_deviceFaultAvailable = hasExtension(extensions, VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        m_diagnosticCheckpointsAvailable =
            hasExtension(extensions, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
        if (!dynamicRenderingAvailable || !sync2Available) {
            return false;
        }

        SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device, m_surface);
        if (swapchainSupport.formats.empty() || swapchainSupport.presentModes.empty()) {
            return false;
        }

        VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        VkPhysicalDeviceSynchronization2Features sync2Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
        VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
        VkPhysicalDeviceVulkan11Features vulkan11Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceVulkan12Features vulkan12Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceSubgroupSizeControlFeatures subgroupSizeControlFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
        VkPhysicalDeviceRobustness2FeaturesEXT robustness2Features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.pNext = &vulkan11Features;
        vulkan11Features.pNext = &vulkan12Features;
        if (subgroupSizeControlAvailable) {
            vulkan12Features.pNext = &subgroupSizeControlFeatures;
            subgroupSizeControlFeatures.pNext = &dynamicRenderingFeatures;
        } else {
            vulkan12Features.pNext = &dynamicRenderingFeatures;
        }
        dynamicRenderingFeatures.pNext = &sync2Features;
        sync2Features.pNext = &meshShaderFeatures;
        meshShaderFeatures.pNext = &rayQueryFeatures;
        rayQueryFeatures.pNext = &rayTracingPipelineFeatures;
        rayTracingPipelineFeatures.pNext = &accelerationStructureFeatures;
        accelerationStructureFeatures.pNext = &descriptorBufferFeatures;
        descriptorBufferFeatures.pNext = robustness2Available ? &robustness2Features : nullptr;
        vkGetPhysicalDeviceFeatures2(device, &features2);

        if (dynamicRenderingFeatures.dynamicRendering != VK_TRUE ||
            sync2Features.synchronization2 != VK_TRUE ||
            vulkan11Features.shaderDrawParameters != VK_TRUE ||
            vulkan12Features.descriptorBindingPartiallyBound != VK_TRUE ||
            vulkan12Features.runtimeDescriptorArray != VK_TRUE) {
            return false;
        }

        m_toolingInfo.pipelineStatistics = features2.features.pipelineStatisticsQuery == VK_TRUE;
        m_toolingInfo.deviceFault = m_deviceFaultAvailable;
        m_toolingInfo.diagnosticCheckpoints = m_diagnosticCheckpointsAvailable;

        m_features.dynamicRendering = true;
        m_features.synchronization2 = true;
        m_features.shaderDrawParameters = true;
        m_features.descriptorIndexing = true;
        m_features.bufferDeviceAddress =
            bufferDeviceAddressAvailable &&
            vulkan12Features.bufferDeviceAddress == VK_TRUE;
        m_features.meshShaders = meshShaderAvailable && meshShaderFeatures.meshShader == VK_TRUE;
        m_features.taskShaders = meshShaderAvailable && meshShaderFeatures.taskShader == VK_TRUE;
        m_features.rayTracing =
            m_features.bufferDeviceAddress &&
            accelerationStructureAvailable &&
            rayQueryAvailable &&
            deferredHostOperationsAvailable &&
            accelerationStructureFeatures.accelerationStructure == VK_TRUE &&
            rayQueryFeatures.rayQuery == VK_TRUE;
        m_features.rayTracingPipeline =
            m_features.rayTracing &&
            rayTracingPipelineAvailable &&
            rayTracingPipelineFeatures.rayTracingPipeline == VK_TRUE;
        m_features.externalHostMemory = externalMemoryHostAvailable;
        m_features.descriptorBuffer =
            descriptorBufferAvailable &&
            descriptorBufferFeatures.descriptorBuffer == VK_TRUE;
        m_storageBuffer16BitAccessEnabled = vulkan11Features.storageBuffer16BitAccess == VK_TRUE;
        m_uniformAndStorageBuffer8BitAccessEnabled =
            vulkan12Features.uniformAndStorageBuffer8BitAccess == VK_TRUE;
        m_uniformAndStorageBuffer16BitAccessEnabled =
            vulkan11Features.uniformAndStorageBuffer16BitAccess == VK_TRUE;
        m_subgroupSizeControlAvailable = subgroupSizeControlAvailable;
        m_subgroupSizeControlSupported =
            subgroupSizeControlAvailable &&
            subgroupSizeControlFeatures.subgroupSizeControl == VK_TRUE;
        m_computeFullSubgroupsSupported =
            subgroupSizeControlAvailable &&
            subgroupSizeControlFeatures.computeFullSubgroups == VK_TRUE;
        m_timelineSemaphoreSupported = vulkan12Features.timelineSemaphore == VK_TRUE;
        m_nullDescriptorEnabled =
            robustness2Available &&
            robustness2Features.nullDescriptor == VK_TRUE;

        // Query RT pipeline properties when supported.
        if (m_features.rayTracingPipeline) {
            VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtPipelineProps{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
            VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            props2.pNext = &rtPipelineProps;
            vkGetPhysicalDeviceProperties2(device, &props2);
            m_rtPipelineProperties.shaderGroupHandleSize      = rtPipelineProps.shaderGroupHandleSize;
            m_rtPipelineProperties.shaderGroupHandleAlignment  = rtPipelineProps.shaderGroupHandleAlignment;
            m_rtPipelineProperties.shaderGroupBaseAlignment    = rtPipelineProps.shaderGroupBaseAlignment;
            m_rtPipelineProperties.maxRayRecursionDepth        = rtPipelineProps.maxRayRecursionDepth;
            m_rtPipelineProperties.maxRayDispatchInvocationCount = rtPipelineProps.maxRayDispatchInvocationCount;
        }

        return true;
    }

    void createLogicalDevice(const RhiCreateInfo& createInfo) {
        const bool enableValidation = createInfo.enableValidation;
        float queuePriority = 1.0f;
        std::set<uint32_t> uniqueQueueFamilies = {m_queueFamilies.graphics.value(), m_queueFamilies.present.value()};
        if (m_queueFamilies.compute.has_value())  uniqueQueueFamilies.insert(m_queueFamilies.compute.value());
        if (m_queueFamilies.transfer.has_value()) uniqueQueueFamilies.insert(m_queueFamilies.transfer.value());

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        for (uint32_t family : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            queueCreateInfo.queueFamilyIndex = family;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        if (m_physicalDeviceProperties.apiVersion < VK_API_VERSION_1_3) {
            deviceExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
            deviceExtensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
        }
        if (m_features.meshShaders) {
            deviceExtensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
        }
        if (m_features.bufferDeviceAddress &&
            m_physicalDeviceProperties.apiVersion < VK_API_VERSION_1_2) {
            deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        }
        if (m_features.rayTracing) {
            deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        }
        if (m_features.rayTracingPipeline) {
            deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        }
        if (m_features.externalHostMemory) {
            deviceExtensions.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        }
        if (m_features.descriptorBuffer) {
            deviceExtensions.push_back(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
        }
        if (m_subgroupSizeControlSupported &&
            m_subgroupSizeControlAvailable &&
            m_physicalDeviceProperties.apiVersion < VK_API_VERSION_1_3) {
            deviceExtensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
        }
        if (m_nullDescriptorEnabled) {
            deviceExtensions.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
        }
        if (m_deviceFaultAvailable) {
            deviceExtensions.push_back(VK_EXT_DEVICE_FAULT_EXTENSION_NAME);
        }
        if (m_diagnosticCheckpointsAvailable) {
            deviceExtensions.push_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
        }
        for (auto ext : createInfo.extraDeviceExtensions) {
            deviceExtensions.push_back(ext);
        }

        // Deduplicate and resolve conflicts:
        // VK_KHR_buffer_device_address supersedes VK_EXT_buffer_device_address (can't enable both).
        // On Vulkan 1.2+ bufferDeviceAddress is core, so both extension variants are unnecessary.
        {
            const bool bdaIsCore = m_physicalDeviceProperties.apiVersion >= VK_API_VERSION_1_2;
            bool hasKhr = false;
            for (auto ext : deviceExtensions) {
                if (std::string_view(ext) == VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) hasKhr = true;
            }
            if (hasKhr || bdaIsCore) {
                deviceExtensions.erase(
                    std::remove_if(deviceExtensions.begin(), deviceExtensions.end(),
                        [](const char* e) { return std::string_view(e) == "VK_EXT_buffer_device_address"; }),
                    deviceExtensions.end());
            }
            // Also remove plain duplicates
            std::sort(deviceExtensions.begin(), deviceExtensions.end(),
                      [](const char* a, const char* b) { return std::string_view(a) < std::string_view(b); });
            deviceExtensions.erase(
                std::unique(deviceExtensions.begin(), deviceExtensions.end(),
                    [](const char* a, const char* b) { return std::string_view(a) == std::string_view(b); }),
                deviceExtensions.end());
        }

        // Store enabled extensions for runtime queries
        m_enabledExtensions.clear();
        m_enabledExtensions.reserve(deviceExtensions.size());
        for (const char* ext : deviceExtensions) {
            m_enabledExtensions.emplace_back(ext);
        }

        VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        meshShaderFeatures.meshShader = m_features.meshShaders ? VK_TRUE : VK_FALSE;
        meshShaderFeatures.taskShader = m_features.taskShaders ? VK_TRUE : VK_FALSE;

        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        rayQueryFeatures.rayQuery = m_features.rayTracing ? VK_TRUE : VK_FALSE;
        rayQueryFeatures.pNext = nullptr;

        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
        rayTracingPipelineFeatures.rayTracingPipeline = m_features.rayTracingPipeline ? VK_TRUE : VK_FALSE;
        rayTracingPipelineFeatures.pNext = &rayQueryFeatures;

        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        accelerationStructureFeatures.accelerationStructure = m_features.rayTracing ? VK_TRUE : VK_FALSE;
        accelerationStructureFeatures.pNext = &rayTracingPipelineFeatures;

        VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferEnableFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
        descriptorBufferEnableFeatures.descriptorBuffer = m_features.descriptorBuffer ? VK_TRUE : VK_FALSE;

        VkPhysicalDeviceSubgroupSizeControlFeatures subgroupSizeControlFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        subgroupSizeControlFeatures.subgroupSizeControl =
            m_subgroupSizeControlSupported ? VK_TRUE : VK_FALSE;
        subgroupSizeControlFeatures.computeFullSubgroups =
            m_computeFullSubgroupsSupported ? VK_TRUE : VK_FALSE;

        VkPhysicalDeviceRobustness2FeaturesEXT robustness2Features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
        robustness2Features.nullDescriptor = m_nullDescriptorEnabled ? VK_TRUE : VK_FALSE;

        void* optionalFeatureChain = nullptr;
        if (m_features.rayTracing) {
            optionalFeatureChain = &accelerationStructureFeatures;
        }
        if (m_features.meshShaders) {
            meshShaderFeatures.pNext = optionalFeatureChain;
            optionalFeatureChain = &meshShaderFeatures;
        }

        VkPhysicalDeviceSynchronization2Features sync2Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
        sync2Features.synchronization2 = VK_TRUE;
        sync2Features.pNext = optionalFeatureChain;

        // Descriptor buffer (VK_EXT_descriptor_buffer) — chain only when enabled
        if (m_features.descriptorBuffer) {
            descriptorBufferEnableFeatures.pNext = sync2Features.pNext;
            sync2Features.pNext = &descriptorBufferEnableFeatures;
        }
        if (m_subgroupSizeControlSupported) {
            subgroupSizeControlFeatures.pNext = sync2Features.pNext;
            sync2Features.pNext = &subgroupSizeControlFeatures;
        }
        if (m_nullDescriptorEnabled) {
            robustness2Features.pNext = sync2Features.pNext;
            sync2Features.pNext = &robustness2Features;
        }

        VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
        dynamicRenderingFeatures.dynamicRendering = VK_TRUE;
        dynamicRenderingFeatures.pNext = &sync2Features;

        VkPhysicalDeviceVulkan11Features vulkan11Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        vulkan11Features.shaderDrawParameters = VK_TRUE;
        vulkan11Features.storageBuffer16BitAccess = m_storageBuffer16BitAccessEnabled ? VK_TRUE : VK_FALSE;
        vulkan11Features.uniformAndStorageBuffer16BitAccess =
            m_uniformAndStorageBuffer16BitAccessEnabled ? VK_TRUE : VK_FALSE;

        VkPhysicalDeviceVulkan12Features vulkan12Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        vulkan12Features.descriptorBindingPartiallyBound = VK_TRUE;
        vulkan12Features.runtimeDescriptorArray = VK_TRUE;
        vulkan12Features.bufferDeviceAddress = m_features.bufferDeviceAddress ? VK_TRUE : VK_FALSE;
        vulkan12Features.uniformAndStorageBuffer8BitAccess =
            m_uniformAndStorageBuffer8BitAccessEnabled ? VK_TRUE : VK_FALSE;
        vulkan12Features.timelineSemaphore =
            (createInfo.enableTimelineSemaphore && m_timelineSemaphoreSupported) ? VK_TRUE : VK_FALSE;
        vulkan11Features.pNext = &vulkan12Features;

        VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        features2.features.pipelineStatisticsQuery = m_toolingInfo.pipelineStatistics ? VK_TRUE : VK_FALSE;
        if (createInfo.enableTimelineSemaphore && m_timelineSemaphoreSupported) {
            m_features.timelineSemaphore = true;
        } else {
            m_features.timelineSemaphore = false;
            if (createInfo.enableTimelineSemaphore) {
                spdlog::warn("Vulkan: timeline semaphores were requested but are not supported by the selected device");
            }
        }
        vulkan12Features.pNext = &dynamicRenderingFeatures;
        features2.pNext = &vulkan11Features;

        std::vector<const char*> layers;
        if (enableValidation && m_features.validation) {
            layers.push_back(kValidationLayerName);
        }

        VkDeviceCreateInfo deviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
        deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
        deviceCreateInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        deviceCreateInfo.pNext = &features2;

        checkVk(vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device),
                "Failed to create Vulkan logical device");

        vkGetDeviceQueue(m_device, m_queueFamilies.graphics.value(), 0, &m_graphicsQueue);
        vkGetDeviceQueue(m_device, m_queueFamilies.present.value(), 0, &m_presentQueue);
        if (m_queueFamilies.compute.has_value()) {
            vkGetDeviceQueue(m_device, m_queueFamilies.compute.value(), 0, &m_computeQueue);
            spdlog::info("Vulkan: dedicated async compute queue (family {})", m_queueFamilies.compute.value());
        }
        if (m_queueFamilies.transfer.has_value()) {
            vkGetDeviceQueue(m_device, m_queueFamilies.transfer.value(), 0, &m_transferQueue);
            spdlog::info("Vulkan: dedicated transfer queue (family {})", m_queueFamilies.transfer.value());
        }

        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_DEVICE,
                                 vkObjectHandle(m_device),
                                 "Metallic Device");
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_QUEUE,
                                 vkObjectHandle(m_graphicsQueue),
                                 "Metallic Graphics Queue");
        vulkanSetObjectDebugName(m_device,
                                 VK_OBJECT_TYPE_QUEUE,
                                 vkObjectHandle(m_presentQueue),
                                 "Metallic Present Queue");
        if (m_computeQueue != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_QUEUE,
                                     vkObjectHandle(m_computeQueue),
                                     "Metallic Async Compute Queue");
        }
        if (m_transferQueue != VK_NULL_HANDLE) {
            vulkanSetObjectDebugName(m_device,
                                     VK_OBJECT_TYPE_QUEUE,
                                     vkObjectHandle(m_transferQueue),
                                     "Metallic Transfer Queue");
        }

        if (m_features.rayTracingPipeline) {
            spdlog::info("Vulkan: ray tracing pipeline enabled (handleSize={}, baseAlign={}, maxRecursion={})",
                         m_rtPipelineProperties.shaderGroupHandleSize,
                         m_rtPipelineProperties.shaderGroupBaseAlignment,
                         m_rtPipelineProperties.maxRayRecursionDepth);
        }
        if (m_nullDescriptorEnabled) {
            spdlog::info("Vulkan: null descriptors enabled via VK_EXT_robustness2");
        }
    }

    void createCommandObjects() {
        for (size_t i = 0; i < kMaxFramesInFlight; ++i) {
            VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            poolInfo.queueFamilyIndex = m_queueFamilies.graphics.value();
            checkVk(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_frames[i].commandPool),
                    "Failed to create Vulkan command pool");

            VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocInfo.commandPool = m_frames[i].commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
            checkVk(vkAllocateCommandBuffers(m_device, &allocInfo, &m_frames[i].commandBuffer),
                    "Failed to allocate Vulkan command buffer");

            VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            checkVk(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_frames[i].imageAvailable),
                    "Failed to create Vulkan image-available semaphore");
            checkVk(vkCreateFence(m_device, &fenceInfo, nullptr, &m_frames[i].inFlight),
                    "Failed to create Vulkan fence");

            // Async compute command pool + buffer (only if dedicated compute queue is available)
            if (m_computeQueue != VK_NULL_HANDLE) {
                VkCommandPoolCreateInfo computePoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                computePoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                computePoolInfo.queueFamilyIndex = m_queueFamilies.compute.value();
                checkVk(vkCreateCommandPool(m_device, &computePoolInfo, nullptr, &m_frames[i].computeCommandPool),
                        "Failed to create Vulkan async compute command pool");

                VkCommandBufferAllocateInfo computeAllocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                computeAllocInfo.commandPool = m_frames[i].computeCommandPool;
                computeAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                computeAllocInfo.commandBufferCount = 1;
                checkVk(vkAllocateCommandBuffers(m_device, &computeAllocInfo, &m_frames[i].computeCommandBuffer),
                        "Failed to allocate Vulkan async compute command buffer");
            }
        }

        // Create timeline semaphores for async compute/transfer synchronisation
        VkSemaphoreTypeCreateInfo timelineTypeInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timelineTypeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineTypeInfo.initialValue = 0;
        VkSemaphoreCreateInfo timelineSemaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        timelineSemaphoreInfo.pNext = &timelineTypeInfo;

        if (m_computeQueue != VK_NULL_HANDLE && m_features.timelineSemaphore) {
            checkVk(vkCreateSemaphore(m_device, &timelineSemaphoreInfo, nullptr, &m_computeTimelineSemaphore),
                    "Failed to create compute timeline semaphore");
        }
        if (m_transferQueue != VK_NULL_HANDLE && m_features.timelineSemaphore) {
            checkVk(vkCreateSemaphore(m_device, &timelineSemaphoreInfo, nullptr, &m_transferTimelineSemaphore),
                    "Failed to create transfer timeline semaphore");
        }
    }

    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize, 12> poolSizes = {{
            {VK_DESCRIPTOR_TYPE_SAMPLER, 1024},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1024},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1024},
            {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1024},
            {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 256},
        }};

        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 1024 * static_cast<uint32_t>(poolSizes.size());
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        checkVk(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool),
                "Failed to create Vulkan descriptor pool");
    }

    void recreateSwapchain() {
        if (m_requestedWidth == 0 || m_requestedHeight == 0) {
            return;
        }

        waitIdle();
        cleanupSwapchain();

        const SwapchainSupportDetails support = querySwapchainSupport(m_physicalDevice, m_surface);
        if (support.formats.empty() || support.presentModes.empty()) {
            throw std::runtime_error("Swapchain support disappeared while recreating the Vulkan swapchain.");
        }

        m_swapchainFormat = chooseSwapSurfaceFormat(support.formats);
        const VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
        m_swapchainExtent = chooseSwapExtent(support.capabilities, m_requestedWidth, m_requestedHeight);

        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
            imageCount = support.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
        createInfo.surface = m_surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = m_swapchainFormat.format;
        createInfo.imageColorSpace = m_swapchainFormat.colorSpace;
        createInfo.imageExtent = m_swapchainExtent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        const uint32_t queueFamilyIndices[] = {m_queueFamilies.graphics.value(), m_queueFamilies.present.value()};
        if (m_queueFamilies.graphics != m_queueFamilies.present) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        checkVk(vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain),
                "Failed to create Vulkan swapchain");

        uint32_t createdImageCount = 0;
        vkGetSwapchainImagesKHR(m_device, m_swapchain, &createdImageCount, nullptr);
        m_swapchainImages.resize(createdImageCount);
        vkGetSwapchainImagesKHR(m_device, m_swapchain, &createdImageCount, m_swapchainImages.data());

        m_swapchainImageViews.resize(createdImageCount);
        m_swapchainImageLayouts.assign(createdImageCount, VK_IMAGE_LAYOUT_UNDEFINED);
        m_swapchainRenderFinished.resize(createdImageCount, VK_NULL_HANDLE);
        m_imagesInFlight.assign(createdImageCount, VK_NULL_HANDLE);

        for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = m_swapchainImages[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = m_swapchainFormat.format;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            checkVk(vkCreateImageView(m_device, &viewInfo, nullptr, &m_swapchainImageViews[i]),
                    "Failed to create Vulkan swapchain image view");

            VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            checkVk(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_swapchainRenderFinished[i]),
                    "Failed to create Vulkan render-finished semaphore");
        }

        populateNativeHandles();
        applyDebugObjectNames();
    }

    void cleanupSwapchain() {
        for (VkImageView imageView : m_swapchainImageViews) {
            if (imageView != VK_NULL_HANDLE) {
                vkDestroyImageView(m_device, imageView, nullptr);
            }
        }
        for (VkSemaphore renderFinished : m_swapchainRenderFinished) {
            if (renderFinished != VK_NULL_HANDLE) {
                vkDestroySemaphore(m_device, renderFinished, nullptr);
            }
        }
        m_swapchainImageViews.clear();
        m_swapchainImages.clear();
        m_swapchainImageLayouts.clear();
        m_swapchainRenderFinished.clear();
        m_imagesInFlight.clear();

        if (m_swapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
            m_swapchain = VK_NULL_HANDLE;
        }
    }

    void populateNativeHandles() {
        m_nativeHandles.instance = m_instance;
        m_nativeHandles.physicalDevice = m_physicalDevice;
        m_nativeHandles.device = m_device;
        m_nativeHandles.queue = m_graphicsQueue;
        m_nativeHandles.descriptorPool = m_descriptorPool;
        m_nativeHandles.allocator = m_allocator;
        m_nativeHandles.graphicsQueueFamily = m_queueFamilies.graphics.value_or(0);
        m_nativeHandles.computeQueueFamily = m_queueFamilies.compute.value_or(UINT32_MAX);
        m_nativeHandles.transferQueueFamily = m_queueFamilies.transfer.value_or(UINT32_MAX);
        m_nativeHandles.swapchainImageCount = static_cast<uint32_t>(m_swapchainImages.size());
        m_nativeHandles.colorFormat = static_cast<uint32_t>(m_swapchainFormat.format);
        m_nativeHandles.apiVersion = m_physicalDeviceProperties.apiVersion;
        m_nativeHandles.computeQueue = m_computeQueue;
        m_nativeHandles.transferQueue = m_transferQueue;
        m_nativeHandles.computeTimelineSemaphore = m_computeTimelineSemaphore;
        m_nativeHandles.transferTimelineSemaphore = m_transferTimelineSemaphore;
    }

    void createVmaAllocator() {
        VmaAllocatorCreateInfo allocatorInfo{};
        // VMA 3.1.0 supports up to Vulkan 1.3; pass 1.3 even if the device supports 1.4
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
        allocatorInfo.physicalDevice = m_physicalDevice;
        allocatorInfo.device = m_device;
        allocatorInfo.instance = m_instance;
        if (m_features.bufferDeviceAddress) {
            allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        }
        checkVk(vmaCreateAllocator(&allocatorInfo, &m_allocator), "Failed to create VMA allocator");
    }

    void transitionCurrentImage(VkImageLayout oldLayout,
                                VkImageLayout newLayout,
                                VkPipelineStageFlags2 srcStage,
                                VkPipelineStageFlags2 dstStage,
                                VkAccessFlags2 srcAccess,
                                VkAccessFlags2 dstAccess) {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = srcStage;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask = dstStage;
        barrier.dstAccessMask = dstAccess;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = m_swapchainImages[m_imageIndex];
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkDependencyInfo dependencyInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependencyInfo.imageMemoryBarrierCount = 1;
        dependencyInfo.pImageMemoryBarriers = &barrier;

        vkCmdPipelineBarrier2(m_frames[m_frameIndex].commandBuffer, &dependencyInfo);
        m_swapchainImageLayouts[m_imageIndex] = newLayout;
    }

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties m_physicalDeviceProperties{};
    VkDevice m_device = VK_NULL_HANDLE;
    PFN_vkCreateDebugUtilsMessengerEXT m_vkCreateDebugUtilsMessengerEXT = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT m_vkDestroyDebugUtilsMessengerEXT = nullptr;
    PFN_vkGetDeviceProcAddr m_vkGetDeviceProcAddrProxy = nullptr;
    PFN_vkAcquireNextImageKHR m_vkAcquireNextImageKHRProxy = nullptr;
    PFN_vkQueuePresentKHR m_vkQueuePresentKHRProxy = nullptr;
    VmaAllocator m_allocator = nullptr;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_presentQueue = VK_NULL_HANDLE;
    VkQueue m_computeQueue = VK_NULL_HANDLE;
    VkQueue m_transferQueue = VK_NULL_HANDLE;
    VkSemaphore m_computeTimelineSemaphore = VK_NULL_HANDLE;
    VkSemaphore m_transferTimelineSemaphore = VK_NULL_HANDLE;
    uint64_t m_computeTimelineValue = 0;
    uint64_t m_transferTimelineValue = 0;
    QueueFamilyIndices m_queueFamilies;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    std::array<FrameResources, kMaxFramesInFlight> m_frames{};

    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkSurfaceFormatKHR m_swapchainFormat{};
    VkExtent2D m_swapchainExtent{};
    std::vector<VkImage> m_swapchainImages;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<VkImageLayout> m_swapchainImageLayouts;
    std::vector<VkSemaphore> m_swapchainRenderFinished;
    std::vector<VkFence> m_imagesInFlight;

    uint32_t m_frameIndex = 0;
    uint32_t m_imageIndex = 0;
    uint64_t m_submittedFrameCounter = 0;
    uint32_t m_requestedWidth = 0;
    uint32_t m_requestedHeight = 0;
    bool m_pendingResize = false;
    bool m_insideRendering = false;
    bool m_deviceLost = false;
    bool m_deviceFaultAvailable = false;
    bool m_diagnosticCheckpointsAvailable = false;
    bool m_storageBuffer16BitAccessEnabled = false;
    bool m_uniformAndStorageBuffer8BitAccessEnabled = false;
    bool m_uniformAndStorageBuffer16BitAccessEnabled = false;
    bool m_subgroupSizeControlAvailable = false;
    bool m_subgroupSizeControlSupported = false;
    bool m_computeFullSubgroupsSupported = false;
    bool m_timelineSemaphoreSupported = false;
    bool m_nullDescriptorEnabled = false;
    std::string m_deviceLostMessage;

    RhiFeatures m_features{};
    RhiLimits m_limits{};
    RhiDeviceInfo m_deviceInfo{};
    RhiSubgroupProperties m_subgroupProperties{};
    RhiRayTracingPipelineProperties m_rtPipelineProperties{};
    VkPhysicalDeviceDescriptorBufferPropertiesEXT m_descriptorBufferProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
    RhiNativeHandles m_nativeHandles{};
    VulkanToolingInfo m_toolingInfo{};
    std::vector<std::string> m_enabledExtensions;
    VulkanPipelineCacheManager m_pipelineCache;
    VulkanGpuProfiler m_gpuProfiler;
    VulkanUploadRing m_uploadRing;
    VulkanTransientPool m_transientPool;
    VulkanReadbackHeap m_readbackHeap;
    std::vector<QueuedSemaphoreWait> m_pendingGraphicsTimelineWaits;
    CommandContext m_commandContext;
};

} // namespace

// Accessor functions for VulkanContext internals (used by frame graph backend, resource utils, etc.)
VmaAllocator getVulkanAllocator(RhiContext& context) {
    return static_cast<VmaAllocator>(context.nativeHandles().allocator);
}

VkDevice getVulkanDevice(RhiContext& context) {
    return static_cast<VkDevice>(context.nativeHandles().device);
}

VkPhysicalDevice getVulkanPhysicalDevice(RhiContext& context) {
    return static_cast<VkPhysicalDevice>(context.nativeHandles().physicalDevice);
}

VkCommandBuffer getVulkanCurrentCommandBuffer(RhiContext& context) {
    return static_cast<VkCommandBuffer>(context.commandContext().nativeCommandBuffer());
}

VkQueue getVulkanGraphicsQueue(RhiContext& context) {
    return static_cast<VkQueue>(context.nativeHandles().queue);
}

uint32_t getVulkanGraphicsQueueFamily(RhiContext& context) {
    return context.nativeHandles().graphicsQueueFamily;
}

VkImage getVulkanCurrentBackbufferImage(RhiContext& context) {
    return static_cast<VulkanContext&>(context).currentSwapchainImage();
}

VkImageView getVulkanCurrentBackbufferImageView(RhiContext& context) {
    return static_cast<VulkanContext&>(context).currentSwapchainImageView();
}

VkExtent2D getVulkanCurrentBackbufferExtent(RhiContext& context) {
    return static_cast<VulkanContext&>(context).currentSwapchainExtent();
}

VkImageLayout getVulkanCurrentBackbufferLayout(RhiContext& context) {
    return static_cast<VulkanContext&>(context).currentSwapchainLayout();
}

VkQueue getVulkanComputeQueue(RhiContext& context) {
    return static_cast<VkQueue>(context.nativeHandles().computeQueue);
}

VkCommandBuffer getVulkanCurrentComputeCommandBuffer(RhiContext& context) {
    return static_cast<VulkanContext&>(context).currentComputeCommandBuffer();
}

uint64_t vulkanScheduleAsyncComputeSubmit(RhiContext& context) {
    return static_cast<VulkanContext&>(context).scheduleAsyncComputeSubmit();
}

void vulkanEnqueueGraphicsTimelineWait(RhiContext& context,
                                       VkSemaphore semaphore,
                                       uint64_t value,
                                       VkPipelineStageFlags2 stageMask) {
    static_cast<VulkanContext&>(context).enqueueGraphicsTimelineWait(semaphore, value, stageMask);
}

VkPipelineCache getVulkanPipelineCache(RhiContext& context) {
    return static_cast<VulkanContext&>(context).pipelineCacheHandle();
}

const VulkanGpuFrameDiagnostics& getVulkanLatestFrameDiagnostics(RhiContext& context) {
    return static_cast<VulkanContext&>(context).latestFrameDiagnostics();
}

const VulkanToolingInfo& getVulkanToolingInfo(RhiContext& context) {
    return static_cast<VulkanContext&>(context).toolingInfo();
}

VulkanPipelineCacheTelemetry getVulkanPipelineCacheTelemetry(RhiContext& context) {
    return static_cast<VulkanContext&>(context).pipelineCacheTelemetry();
}

bool vulkanIsDeviceLost(RhiContext& context) {
    return static_cast<VulkanContext&>(context).isDeviceLost();
}

const std::string& vulkanDeviceLostMessage(RhiContext& context) {
    return static_cast<VulkanContext&>(context).deviceLostMessage();
}

VulkanGpuProfiler* getVulkanGpuProfiler(RhiContext& context) {
    return static_cast<VulkanContext&>(context).gpuProfiler();
}

const VkPhysicalDeviceDescriptorBufferPropertiesEXT& getVulkanDescriptorBufferProperties(
    RhiContext& context) {
    return static_cast<VulkanContext&>(context).descriptorBufferProperties();
}

VulkanUploadRing& getVulkanUploadRing(RhiContext& context) {
    return static_cast<VulkanContext&>(context).uploadRing();
}

VulkanTransientPool& getVulkanTransientPool(RhiContext& context) {
    return static_cast<VulkanContext&>(context).transientPool();
}

VulkanReadbackHeap& getVulkanReadbackHeap(RhiContext& context) {
    return static_cast<VulkanContext&>(context).readbackHeap();
}

bool vulkanHasExtension(RhiContext& context, const char* extensionName) {
    return static_cast<VulkanContext&>(context).hasEnabledExtension(extensionName);
}

std::unique_ptr<RhiContext> createVulkanContext(const RhiCreateInfo& createInfo,
                                                std::string& errorMessage) {
    try {
        return std::make_unique<VulkanContext>(createInfo);
    } catch (const std::exception& e) {
        errorMessage = e.what();
        return {};
    }
}
