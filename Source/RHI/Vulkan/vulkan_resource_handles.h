#pragma once

#ifdef _WIN32

#include "../rhi_backend.h"
#include "../bindless_scene_constants.h"
#include "vulkan_descriptor_manager.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

enum class VulkanResourceType : uint32_t {
    Texture,
    Buffer,
    Sampler,
    Pipeline,
    VertexDescriptor,
    AccelerationStructure,
};

struct VulkanResourceHeader {
    VulkanResourceType type = VulkanResourceType::Texture;
};

struct VulkanTextureResource {
    VulkanResourceHeader header{VulkanResourceType::Texture};
    VkDevice device = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocator allocator = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mipLevels = 1;
    VkFormat format = VK_FORMAT_UNDEFINED;
    RhiTextureUsage usage = RhiTextureUsage::None;
    uint32_t refCount = 1;
    bool ownsImage = true;
    bool ownsImageView = true;
};

struct VulkanBufferResource {
    VulkanResourceHeader header{VulkanResourceType::Buffer};
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocator allocator = nullptr;
    void* mappedData = nullptr;
    size_t size = 0;
    VkBufferUsageFlags usageFlags = 0;
    VkDeviceAddress deviceAddress = 0;
    bool ownsBuffer = true;
};

struct VulkanSamplerResource {
    VulkanResourceHeader header{VulkanResourceType::Sampler};
    VkDevice device = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

struct VulkanDescriptorBindingLocation {
    uint32_t set = UINT32_MAX;
    uint32_t binding = UINT32_MAX;
    uint32_t arrayElement = 0;
    VkDescriptorType descriptorType = VK_DESCRIPTOR_TYPE_MAX_ENUM;

    bool valid() const {
        return set != UINT32_MAX && binding != UINT32_MAX &&
               descriptorType != VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
};

struct VulkanPipelineResource {
    VulkanResourceHeader header{VulkanResourceType::Pipeline};
    VkDevice device = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    bool ownsLayout = false;
    std::vector<VkDescriptorSetLayout> setLayouts;
    std::vector<uint8_t> setLayoutOwnership;
    uint32_t bindlessSetIndex = UINT32_MAX;
    std::array<VulkanDescriptorBindingLocation, kMaxBufferBindings> bufferBindings{};
    std::array<VulkanDescriptorBindingLocation, kMaxTextureBindings> textureBindings{};
    std::array<VulkanDescriptorBindingLocation, kMaxSamplerBindings> samplerBindings{};
    std::array<VulkanDescriptorBindingLocation, kMaxAccelerationStructureBindings> accelerationStructureBindings{};
};

struct VulkanAccelerationStructureResource {
    VulkanResourceHeader header{VulkanResourceType::AccelerationStructure};
    VkDevice device = VK_NULL_HANDLE;
    VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocator allocator = nullptr;
    VkDeviceAddress deviceAddress = 0;
};

struct VulkanVertexAttributeDesc {
    bool valid = false;
    uint32_t location = 0;
    uint32_t binding = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t offset = 0;
};

struct VulkanVertexBindingDesc {
    bool valid = false;
    uint32_t binding = 0;
    uint32_t stride = 0;
    VkVertexInputRate inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
};

struct VulkanVertexDescriptorResource {
    VulkanResourceHeader header{VulkanResourceType::VertexDescriptor};
    std::vector<VulkanVertexAttributeDesc> attributes;
    std::vector<VulkanVertexBindingDesc> bindings;
};

inline VulkanResourceHeader* getVulkanResourceHeader(void* handle) {
    return static_cast<VulkanResourceHeader*>(handle);
}

inline const VulkanResourceHeader* getVulkanResourceHeader(const void* handle) {
    return static_cast<const VulkanResourceHeader*>(handle);
}

inline VulkanTextureResource* getVulkanTextureResource(const RhiTexture* texture) {
    return texture ? static_cast<VulkanTextureResource*>(texture->nativeHandle()) : nullptr;
}

inline const VulkanTextureResource* getVulkanTextureResource(const RhiTexture& texture) {
    return static_cast<const VulkanTextureResource*>(texture.nativeHandle());
}

inline VkImage getVulkanImage(const RhiTexture* texture) {
    const VulkanTextureResource* resource = getVulkanTextureResource(texture);
    return resource ? resource->image : VK_NULL_HANDLE;
}

inline VkImageView getVulkanImageView(const RhiTexture* texture) {
    const VulkanTextureResource* resource = getVulkanTextureResource(texture);
    return resource ? resource->imageView : VK_NULL_HANDLE;
}

inline VulkanBufferResource* getVulkanBufferResource(const RhiBuffer* buffer) {
    return buffer ? static_cast<VulkanBufferResource*>(buffer->nativeHandle()) : nullptr;
}

inline const VulkanBufferResource* getVulkanBufferResource(const RhiBuffer& buffer) {
    return static_cast<const VulkanBufferResource*>(buffer.nativeHandle());
}

inline VulkanBufferResource* getVulkanBufferResource(RhiBuffer& buffer) {
    return static_cast<VulkanBufferResource*>(buffer.nativeHandle());
}

inline VkBuffer getVulkanBufferHandle(const RhiBuffer* buffer) {
    const VulkanBufferResource* resource = getVulkanBufferResource(buffer);
    return resource ? resource->buffer : VK_NULL_HANDLE;
}

inline VkDeviceAddress getVulkanBufferDeviceAddress(const RhiBuffer* buffer) {
    const VulkanBufferResource* resource = getVulkanBufferResource(buffer);
    return resource ? resource->deviceAddress : 0;
}

inline VkBufferUsageFlags vulkanEnableBufferDeviceAddress(VkBufferUsageFlags usage,
                                                          bool enableBufferDeviceAddress) {
    if (enableBufferDeviceAddress) {
        usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }
    return usage;
}

inline VkBufferUsageFlags vulkanEnableAccelerationStructureBuildInput(VkBufferUsageFlags usage,
                                                                      bool enableBuildInput) {
    if (enableBuildInput) {
        usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }
    return usage;
}

inline bool vulkanBufferUsesDeviceAddress(VkBufferUsageFlags usage) {
    return (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
}

inline VkExternalMemoryHandleTypeFlags vulkanHostVisibleExternalMemoryHandleTypes(
    bool hostVisible,
    bool externalHostMemoryEnabled) {
    return (hostVisible && externalHostMemoryEnabled)
        ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT
        : 0;
}

inline VulkanSamplerResource* getVulkanSamplerResource(const RhiSampler* sampler) {
    return sampler ? static_cast<VulkanSamplerResource*>(sampler->nativeHandle()) : nullptr;
}

inline const VulkanSamplerResource* getVulkanSamplerResource(const RhiSampler& sampler) {
    return static_cast<const VulkanSamplerResource*>(sampler.nativeHandle());
}

inline VkSampler getVulkanSamplerHandle(const RhiSampler* sampler) {
    const VulkanSamplerResource* resource = getVulkanSamplerResource(sampler);
    return resource ? resource->sampler : VK_NULL_HANDLE;
}

inline const VulkanPipelineResource* getVulkanPipelineResource(const RhiGraphicsPipeline& pipeline) {
    return static_cast<const VulkanPipelineResource*>(pipeline.nativeHandle());
}

inline const VulkanPipelineResource* getVulkanPipelineResource(const RhiComputePipeline& pipeline) {
    return static_cast<const VulkanPipelineResource*>(pipeline.nativeHandle());
}

inline VulkanAccelerationStructureResource* getVulkanAccelerationStructureResource(
    const RhiAccelerationStructure* accelerationStructure) {
    return accelerationStructure
        ? static_cast<VulkanAccelerationStructureResource*>(accelerationStructure->nativeHandle())
        : nullptr;
}

inline const VulkanAccelerationStructureResource* getVulkanAccelerationStructureResource(
    const RhiAccelerationStructure& accelerationStructure) {
    return static_cast<const VulkanAccelerationStructureResource*>(accelerationStructure.nativeHandle());
}

inline VulkanAccelerationStructureResource* getVulkanAccelerationStructureResource(
    RhiAccelerationStructure& accelerationStructure) {
    return static_cast<VulkanAccelerationStructureResource*>(accelerationStructure.nativeHandle());
}

inline VkAccelerationStructureKHR getVulkanAccelerationStructureHandle(
    const RhiAccelerationStructure* accelerationStructure) {
    const VulkanAccelerationStructureResource* resource =
        getVulkanAccelerationStructureResource(accelerationStructure);
    return resource ? resource->accelerationStructure : VK_NULL_HANDLE;
}

inline VkDeviceAddress getVulkanAccelerationStructureDeviceAddress(
    const RhiAccelerationStructure* accelerationStructure) {
    const VulkanAccelerationStructureResource* resource =
        getVulkanAccelerationStructureResource(accelerationStructure);
    return resource ? resource->deviceAddress : 0;
}

inline VulkanVertexDescriptorResource* getVulkanVertexDescriptorResource(const RhiVertexDescriptor* descriptor) {
    return descriptor ? static_cast<VulkanVertexDescriptorResource*>(descriptor->nativeHandle()) : nullptr;
}

inline VulkanVertexDescriptorResource* getVulkanVertexDescriptorResource(RhiVertexDescriptor& descriptor) {
    return static_cast<VulkanVertexDescriptorResource*>(descriptor.nativeHandle());
}

inline const VulkanVertexDescriptorResource* getVulkanVertexDescriptorResource(const RhiVertexDescriptor& descriptor) {
    return static_cast<const VulkanVertexDescriptorResource*>(descriptor.nativeHandle());
}

inline VkPipeline getVulkanPipelineHandle(const RhiGraphicsPipeline& pipeline) {
    const VulkanPipelineResource* resource = getVulkanPipelineResource(pipeline);
    return resource ? resource->pipeline : VK_NULL_HANDLE;
}

inline VkPipeline getVulkanPipelineHandle(const RhiComputePipeline& pipeline) {
    const VulkanPipelineResource* resource = getVulkanPipelineResource(pipeline);
    return resource ? resource->pipeline : VK_NULL_HANDLE;
}

inline VkPipelineLayout getVulkanPipelineLayout(const RhiGraphicsPipeline& pipeline) {
    const VulkanPipelineResource* resource = getVulkanPipelineResource(pipeline);
    return resource ? resource->layout : VK_NULL_HANDLE;
}

inline VkPipelineLayout getVulkanPipelineLayout(const RhiComputePipeline& pipeline) {
    const VulkanPipelineResource* resource = getVulkanPipelineResource(pipeline);
    return resource ? resource->layout : VK_NULL_HANDLE;
}

// --- VMA Resource Creation Helpers ---
// Unified VMA-backed resource creation/destruction used across vulkan_backend,
// vulkan_frame_graph, rhi_resource_utils, and rhi_raytracing_utils.

// Creates a VMA-backed buffer. On success, returns a fully populated VulkanBufferResource.
// On failure, returns std::nullopt and sets errorMessage (if provided).
struct VmaBufferCreateInfo {
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = nullptr;
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    bool hostVisible = false;
    VkExternalMemoryHandleTypeFlags externalMemoryHandleTypes = 0;
    const char* debugName = nullptr;
};

inline std::optional<VulkanBufferResource> vmaCreateBufferResource(const VmaBufferCreateInfo& info,
                                                                     const char** outErrorMessage = nullptr) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = info.size;
    bufferInfo.usage = info.usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkExternalMemoryBufferCreateInfo externalMemoryInfo{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    if (info.externalMemoryHandleTypes != 0) {
        externalMemoryInfo.handleTypes = info.externalMemoryHandleTypes;
        externalMemoryInfo.pNext = bufferInfo.pNext;
        bufferInfo.pNext = &externalMemoryInfo;
    }

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if (info.hostVisible) {
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VulkanBufferResource res{};
    res.device = info.device;
    res.allocator = info.allocator;
    res.size = static_cast<size_t>(info.size);
    res.usageFlags = info.usage;

    VmaAllocationInfo allocInfo{};
    VkResult result = vmaCreateBuffer(info.allocator, &bufferInfo, &allocCreateInfo,
                                      &res.buffer, &res.allocation, &allocInfo);
    if (result != VK_SUCCESS) {
        if (outErrorMessage) {
            static constexpr const char* kMsg = "vmaCreateBufferResource failed";
            *outErrorMessage = kMsg;
        }
        return std::nullopt;
    }

    res.mappedData = allocInfo.pMappedData;

    if (vulkanBufferUsesDeviceAddress(info.usage)) {
        VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addrInfo.buffer = res.buffer;
        res.deviceAddress = vkGetBufferDeviceAddress(info.device, &addrInfo);
    }

    if (info.debugName && info.debugName[0] != '\0') {
        vmaSetAllocationName(info.allocator, res.allocation, info.debugName);
    }

    return res;
}

struct VmaImageCreateInfo {
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = nullptr;
    VkImageCreateInfo imageInfo{};
    bool depth = false;
    bool dedicated = true;
    const char* debugName = nullptr;
};

inline std::optional<VulkanTextureResource> vmaCreateImageResource(const VmaImageCreateInfo& info,
                                                                     const char** outErrorMessage = nullptr) {
    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if (info.dedicated) {
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }

    VulkanTextureResource res{};
    res.device = info.device;
    res.allocator = info.allocator;
    res.width = info.imageInfo.extent.width;
    res.height = info.imageInfo.extent.height;
    res.depth = info.imageInfo.extent.depth;
    res.mipLevels = info.imageInfo.mipLevels;
    res.format = info.imageInfo.format;

    VkResult result = vmaCreateImage(info.allocator, &info.imageInfo, &allocCreateInfo,
                                     &res.image, &res.allocation, nullptr);
    if (result != VK_SUCCESS) {
        if (outErrorMessage) {
            static constexpr const char* kMsg = "vmaCreateImageResource: vmaCreateImage failed";
            *outErrorMessage = kMsg;
        }
        return std::nullopt;
    }

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = res.image;
    viewInfo.viewType = info.imageInfo.imageType == VK_IMAGE_TYPE_3D
                            ? VK_IMAGE_VIEW_TYPE_3D
                            : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = info.imageInfo.format;
    viewInfo.subresourceRange.aspectMask = info.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = info.imageInfo.mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = info.imageInfo.arrayLayers;

    result = vkCreateImageView(info.device, &viewInfo, nullptr, &res.imageView);
    if (result != VK_SUCCESS) {
        vmaDestroyImage(info.allocator, res.image, res.allocation);
        if (outErrorMessage) {
            static constexpr const char* kMsg = "vmaCreateImageResource: vkCreateImageView failed";
            *outErrorMessage = kMsg;
        }
        return std::nullopt;
    }

    if (info.debugName && info.debugName[0] != '\0') {
        vmaSetAllocationName(info.allocator, res.allocation, info.debugName);
    }

    return res;
}

inline void vmaDestroyBufferResource(VulkanBufferResource& res) {
    if (res.buffer != VK_NULL_HANDLE && res.allocator != nullptr) {
        vmaDestroyBuffer(res.allocator, res.buffer, res.allocation);
        res.buffer = VK_NULL_HANDLE;
        res.allocation = nullptr;
    }
}

inline void vmaDestroyImageResource(VulkanTextureResource& res) {
    if (res.imageView != VK_NULL_HANDLE && res.device != VK_NULL_HANDLE) {
        vkDestroyImageView(res.device, res.imageView, nullptr);
        res.imageView = VK_NULL_HANDLE;
    }
    if (res.image != VK_NULL_HANDLE && res.allocator != nullptr) {
        vmaDestroyImage(res.allocator, res.image, res.allocation);
        res.image = VK_NULL_HANDLE;
        res.allocation = nullptr;
    }
}

#endif // _WIN32
