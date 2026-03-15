#pragma once

#ifdef _WIN32

#include "../rhi_backend.h"

#include <cstddef>
#include <cstdint>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

enum class VulkanResourceType : uint32_t {
    Texture,
    Buffer,
    Sampler,
    Pipeline,
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
    bool ownsBuffer = true;
};

struct VulkanSamplerResource {
    VulkanResourceHeader header{VulkanResourceType::Sampler};
    VkDevice device = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

struct VulkanPipelineResource {
    VulkanResourceHeader header{VulkanResourceType::Pipeline};
    VkDevice device = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    bool ownsLayout = false;
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

inline VkBuffer getVulkanBufferHandle(const RhiBuffer* buffer) {
    const VulkanBufferResource* resource = getVulkanBufferResource(buffer);
    return resource ? resource->buffer : VK_NULL_HANDLE;
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

#endif // _WIN32
