#include "rhi_resource_utils.h"

#ifdef __APPLE__
#include "metal_resource_utils.h"

namespace {

MetalSamplerFilter toMetalFilter(RhiSamplerFilterMode filter) {
    switch (filter) {
    case RhiSamplerFilterMode::Nearest:
        return MetalSamplerFilter::Nearest;
    case RhiSamplerFilterMode::Linear:
    default:
        return MetalSamplerFilter::Linear;
    }
}

MetalSamplerMipFilter toMetalMipFilter(RhiSamplerMipFilterMode filter) {
    switch (filter) {
    case RhiSamplerMipFilterMode::Linear:
        return MetalSamplerMipFilter::Linear;
    case RhiSamplerMipFilterMode::None:
    default:
        return MetalSamplerMipFilter::None;
    }
}

MetalSamplerAddressMode toMetalAddressMode(RhiSamplerAddressMode mode) {
    switch (mode) {
    case RhiSamplerAddressMode::ClampToEdge:
        return MetalSamplerAddressMode::ClampToEdge;
    case RhiSamplerAddressMode::Repeat:
    default:
        return MetalSamplerAddressMode::Repeat;
    }
}

} // namespace

RhiBufferHandle rhiCreateSharedBuffer(const RhiDevice& device,
                                      const void* initialData,
                                      size_t size,
                                      const char* debugName) {
    return RhiBufferHandle(metalCreateSharedBuffer(device.nativeHandle(), initialData, size, debugName), size);
}

void* rhiBufferContents(const RhiBuffer& buffer) {
    return metalBufferContents(buffer.nativeHandle());
}

RhiTextureHandle rhiCreateTexture2D(const RhiDevice& device,
                                    uint32_t width,
                                    uint32_t height,
                                    RhiFormat format,
                                    bool mipmapped,
                                    uint32_t mipLevelCount,
                                    RhiTextureStorageMode storageMode,
                                    RhiTextureUsage usage) {
    return RhiTextureHandle(
        metalCreateTexture2D(device.nativeHandle(),
                             width,
                             height,
                             format,
                             mipmapped,
                             mipLevelCount,
                             storageMode,
                             usage),
        width,
        height);
}

RhiTextureHandle rhiCreateTexture3D(const RhiDevice& device,
                                    uint32_t width,
                                    uint32_t height,
                                    uint32_t depth,
                                    RhiFormat format,
                                    RhiTextureStorageMode storageMode,
                                    RhiTextureUsage usage) {
    return RhiTextureHandle(
        metalCreateTexture3D(device.nativeHandle(),
                             width,
                             height,
                             depth,
                             format,
                             storageMode,
                             usage),
        width,
        height);
}

void rhiUploadTexture2D(const RhiTexture& texture,
                        uint32_t width,
                        uint32_t height,
                        const void* data,
                        size_t bytesPerRow,
                        uint32_t mipLevel) {
    metalUploadTexture2D(texture.nativeHandle(), width, height, data, bytesPerRow, mipLevel);
}

void rhiUploadTexture3D(const RhiTexture& texture,
                        uint32_t width,
                        uint32_t height,
                        uint32_t depth,
                        const void* data,
                        size_t bytesPerRow,
                        size_t bytesPerImage,
                        uint32_t mipLevel) {
    metalUploadTexture3D(texture.nativeHandle(),
                         width,
                         height,
                         depth,
                         data,
                         bytesPerRow,
                         bytesPerImage,
                         mipLevel);
}

void rhiGenerateMipmaps(const RhiCommandQueue& commandQueue, const RhiTexture& texture) {
    metalGenerateMipmaps(commandQueue.nativeHandle(), texture.nativeHandle());
}

RhiSamplerHandle rhiCreateSampler(const RhiDevice& device, const RhiSamplerDesc& desc) {
    MetalSamplerDesc metalDesc;
    metalDesc.minFilter = toMetalFilter(desc.minFilter);
    metalDesc.magFilter = toMetalFilter(desc.magFilter);
    metalDesc.mipFilter = toMetalMipFilter(desc.mipFilter);
    metalDesc.addressModeS = toMetalAddressMode(desc.addressModeS);
    metalDesc.addressModeT = toMetalAddressMode(desc.addressModeT);
    metalDesc.addressModeR = toMetalAddressMode(desc.addressModeR);
    return RhiSamplerHandle(metalCreateSampler(device.nativeHandle(), metalDesc));
}

RhiDepthStencilStateHandle rhiCreateDepthStencilState(const RhiDevice& device,
                                                      bool depthWriteEnabled,
                                                      bool reversedZ) {
    return RhiDepthStencilStateHandle(
        metalCreateDepthStencilState(device.nativeHandle(), depthWriteEnabled, reversedZ));
}

bool rhiSupportsRaytracing(const RhiDevice& device) {
    return metalSupportsRaytracing(device.nativeHandle());
}

RhiTextureHandle rhiRetainTexture(const RhiTexture& texture) {
    return RhiTextureHandle(metalRetainHandle(texture.nativeHandle()),
                            texture.width(),
                            texture.height());
}

void rhiReleaseNativeHandle(void* handle) {
    metalReleaseHandle(handle);
}

#elif defined(_WIN32)

#include "vulkan_backend.h"
#include "vulkan_frame_graph.h"
#include "vulkan_resource_handles.h"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <spdlog/spdlog.h>

#include <cstring>
#include <algorithm>

namespace {

// Stored Vulkan context handles for resource creation (set during first call)
struct VulkanResourceContext {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VmaAllocator allocator = nullptr;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;
    bool initialized = false;
};

VulkanResourceContext g_vkResCtx;

void ensureResourceContext(const RhiDevice& device) {
    if (g_vkResCtx.initialized) return;
    // The device handle stores VkDevice on Vulkan
    g_vkResCtx.device = static_cast<VkDevice>(device.nativeHandle());
    g_vkResCtx.initialized = true;
}

void setResourceContext(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator,
                        VkQueue queue, uint32_t queueFamily) {
    g_vkResCtx.device = device;
    g_vkResCtx.physicalDevice = physicalDevice;
    g_vkResCtx.allocator = allocator;
    g_vkResCtx.graphicsQueue = queue;
    g_vkResCtx.graphicsQueueFamily = queueFamily;
    g_vkResCtx.initialized = true;
}

VkFilter toVkFilter(RhiSamplerFilterMode filter) {
    return filter == RhiSamplerFilterMode::Nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
}

VkSamplerMipmapMode toVkMipFilter(RhiSamplerMipFilterMode filter) {
    return filter == RhiSamplerMipFilterMode::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
}

VkSamplerAddressMode toVkAddressMode(RhiSamplerAddressMode mode) {
    return mode == RhiSamplerAddressMode::ClampToEdge ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                                                       : VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

bool isVkDepthFormat(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D16_UNORM ||
           format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat format,
                             VkImageViewType viewType, uint32_t mipLevels, uint32_t layerCount) {
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = viewType;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = isVkDepthFormat(format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = layerCount;

    VkImageView view = VK_NULL_HANDLE;
    vkCreateImageView(device, &viewInfo, nullptr, &view);
    return view;
}

// One-shot command buffer for staging uploads and mipmap generation
VkCommandBuffer beginOneTimeCommands(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = pool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

void endOneTimeCommands(VkDevice device, VkCommandPool pool, VkQueue queue, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

// Transient command pool for one-time uploads
VkCommandPool g_uploadPool = VK_NULL_HANDLE;

VkCommandPool getUploadPool() {
    if (g_uploadPool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = g_vkResCtx.graphicsQueueFamily;
        vkCreateCommandPool(g_vkResCtx.device, &poolInfo, nullptr, &g_uploadPool);
    }
    return g_uploadPool;
}

} // namespace

void vulkanSetResourceContext(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator,
                              VkQueue queue, uint32_t queueFamily) {
    setResourceContext(device, physicalDevice, allocator, queue, queueFamily);
}

RhiBufferHandle rhiCreateSharedBuffer(const RhiDevice& /*device*/,
                                      const void* initialData,
                                      size_t size,
                                      const char* debugName) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;

    auto* res = new VulkanBufferResource{};
    res->device = g_vkResCtx.device;
    res->allocator = g_vkResCtx.allocator;
    res->size = size;

    VmaAllocationInfo allocInfo{};
    VkResult result = vmaCreateBuffer(g_vkResCtx.allocator, &bufferInfo, &allocCreateInfo,
                                       &res->buffer, &res->allocation, &allocInfo);
    if (result != VK_SUCCESS) {
        spdlog::error("Failed to create Vulkan shared buffer (VkResult: {})", static_cast<int>(result));
        delete res;
        return {};
    }

    res->mappedData = allocInfo.pMappedData;
    if (initialData && res->mappedData) {
        std::memcpy(res->mappedData, initialData, size);
    }

    if (debugName) {
        vmaSetAllocationName(g_vkResCtx.allocator, res->allocation, debugName);
    }

    return RhiBufferHandle(res, size);
}

void* rhiBufferContents(const RhiBuffer& buffer) {
    auto* res = getVulkanBufferResource(buffer);
    return res ? res->mappedData : nullptr;
}

RhiTextureHandle rhiCreateTexture2D(const RhiDevice& /*device*/,
                                    uint32_t width,
                                    uint32_t height,
                                    RhiFormat format,
                                    bool /*mipmapped*/,
                                    uint32_t mipLevelCount,
                                    RhiTextureStorageMode /*storageMode*/,
                                    RhiTextureUsage usage) {
    VkFormat vkFormat = toVkFormat(format);
    VkImageUsageFlags vkUsage = toVkImageUsage(usage);
    if (isVkDepthFormat(vkFormat)) {
        vkUsage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        vkUsage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = vkFormat;
    imageInfo.extent = {width, height, 1};
    imageInfo.mipLevels = std::max(mipLevelCount, 1u);
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = vkUsage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    auto* res = new VulkanTextureResource{};
    res->device = g_vkResCtx.device;
    res->allocator = g_vkResCtx.allocator;
    res->width = width;
    res->height = height;
    res->mipLevels = imageInfo.mipLevels;
    res->format = vkFormat;
    res->usage = usage;

    VkResult result = vmaCreateImage(g_vkResCtx.allocator, &imageInfo, &allocCreateInfo,
                                      &res->image, &res->allocation, nullptr);
    if (result != VK_SUCCESS) {
        spdlog::error("Failed to create Vulkan 2D texture (VkResult: {})", static_cast<int>(result));
        delete res;
        return {};
    }

    res->imageView = createImageView(g_vkResCtx.device, res->image, vkFormat,
                                      VK_IMAGE_VIEW_TYPE_2D, imageInfo.mipLevels, 1);
    return RhiTextureHandle(res, width, height);
}

RhiTextureHandle rhiCreateTexture3D(const RhiDevice& /*device*/,
                                    uint32_t width,
                                    uint32_t height,
                                    uint32_t depth,
                                    RhiFormat format,
                                    RhiTextureStorageMode /*storageMode*/,
                                    RhiTextureUsage usage) {
    VkFormat vkFormat = toVkFormat(format);

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_3D;
    imageInfo.format = vkFormat;
    imageInfo.extent = {width, height, depth};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = toVkImageUsage(usage);
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    auto* res = new VulkanTextureResource{};
    res->device = g_vkResCtx.device;
    res->allocator = g_vkResCtx.allocator;
    res->width = width;
    res->height = height;
    res->depth = depth;
    res->format = vkFormat;
    res->usage = usage;

    VkResult result = vmaCreateImage(g_vkResCtx.allocator, &imageInfo, &allocCreateInfo,
                                      &res->image, &res->allocation, nullptr);
    if (result != VK_SUCCESS) {
        spdlog::error("Failed to create Vulkan 3D texture (VkResult: {})", static_cast<int>(result));
        delete res;
        return {};
    }

    res->imageView = createImageView(g_vkResCtx.device, res->image, vkFormat,
                                      VK_IMAGE_VIEW_TYPE_3D, 1, 1);
    return RhiTextureHandle(res, width, height);
}

void rhiUploadTexture2D(const RhiTexture& texture,
                        uint32_t width,
                        uint32_t height,
                        const void* data,
                        size_t bytesPerRow,
                        uint32_t mipLevel) {
    auto* res = static_cast<VulkanTextureResource*>(texture.nativeHandle());
    if (!res || !data) return;

    size_t imageSize = bytesPerRow * height;

    // Create staging buffer
    VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagingInfo.size = imageSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = nullptr;
    VmaAllocationInfo stagingData{};
    vmaCreateBuffer(g_vkResCtx.allocator, &stagingInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAlloc, &stagingData);

    std::memcpy(stagingData.pMappedData, data, imageSize);

    VkCommandPool pool = getUploadPool();
    VkCommandBuffer cmd = beginOneTimeCommands(g_vkResCtx.device, pool);

    // Transition to TRANSFER_DST
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_NONE;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = res->image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mipLevel, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    // Copy buffer to image
    VkBufferImageCopy region{};
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mipLevel, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(cmd, stagingBuffer, res->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to SHADER_READ
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    endOneTimeCommands(g_vkResCtx.device, pool, g_vkResCtx.graphicsQueue, cmd);
    vmaDestroyBuffer(g_vkResCtx.allocator, stagingBuffer, stagingAlloc);
}

void rhiUploadTexture3D(const RhiTexture& texture,
                        uint32_t width,
                        uint32_t height,
                        uint32_t depth,
                        const void* data,
                        size_t bytesPerRow,
                        size_t bytesPerImage,
                        uint32_t mipLevel) {
    auto* res = static_cast<VulkanTextureResource*>(texture.nativeHandle());
    if (!res || !data) return;

    size_t imageSize = bytesPerImage * depth;

    VkBufferCreateInfo stagingInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stagingInfo.size = imageSize;
    stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo stagingAllocInfo{};
    stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                             VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = nullptr;
    VmaAllocationInfo stagingData{};
    vmaCreateBuffer(g_vkResCtx.allocator, &stagingInfo, &stagingAllocInfo,
                    &stagingBuffer, &stagingAlloc, &stagingData);
    std::memcpy(stagingData.pMappedData, data, imageSize);

    VkCommandPool pool = getUploadPool();
    VkCommandBuffer cmd = beginOneTimeCommands(g_vkResCtx.device, pool);

    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_NONE;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = res->image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mipLevel, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mipLevel, 0, 1};
    region.imageExtent = {width, height, depth};
    vkCmdCopyBufferToImage(cmd, stagingBuffer, res->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    endOneTimeCommands(g_vkResCtx.device, pool, g_vkResCtx.graphicsQueue, cmd);
    vmaDestroyBuffer(g_vkResCtx.allocator, stagingBuffer, stagingAlloc);
}

void rhiGenerateMipmaps(const RhiCommandQueue& /*commandQueue*/, const RhiTexture& texture) {
    auto* res = static_cast<VulkanTextureResource*>(texture.nativeHandle());
    if (!res || res->mipLevels <= 1) return;

    VkCommandPool pool = getUploadPool();
    VkCommandBuffer cmd = beginOneTimeCommands(g_vkResCtx.device, pool);

    int32_t mipWidth = static_cast<int32_t>(res->width);
    int32_t mipHeight = static_cast<int32_t>(res->height);

    for (uint32_t i = 1; i < res->mipLevels; ++i) {
        // Transition level i-1 to TRANSFER_SRC
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = res->image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};

        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        // Blit from level i-1 to level i
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};

        vkCmdBlitImage(cmd,
                       res->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       res->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_LINEAR);

        // Transition level i-1 to SHADER_READ
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        if (mipWidth > 1) mipWidth /= 2;
        if (mipHeight > 1) mipHeight /= 2;
    }

    // Transition last mip level to SHADER_READ
    VkImageMemoryBarrier2 lastBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    lastBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    lastBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    lastBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    lastBarrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    lastBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    lastBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    lastBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    lastBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    lastBarrier.image = res->image;
    lastBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, res->mipLevels - 1, 1, 0, 1};

    VkDependencyInfo lastDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    lastDep.imageMemoryBarrierCount = 1;
    lastDep.pImageMemoryBarriers = &lastBarrier;
    vkCmdPipelineBarrier2(cmd, &lastDep);

    endOneTimeCommands(g_vkResCtx.device, pool, g_vkResCtx.graphicsQueue, cmd);
}

RhiSamplerHandle rhiCreateSampler(const RhiDevice& /*device*/, const RhiSamplerDesc& desc) {
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.minFilter = toVkFilter(desc.minFilter);
    samplerInfo.magFilter = toVkFilter(desc.magFilter);
    samplerInfo.mipmapMode = toVkMipFilter(desc.mipFilter);
    samplerInfo.addressModeU = toVkAddressMode(desc.addressModeS);
    samplerInfo.addressModeV = toVkAddressMode(desc.addressModeT);
    samplerInfo.addressModeW = toVkAddressMode(desc.addressModeR);
    samplerInfo.maxLod = (desc.mipFilter != RhiSamplerMipFilterMode::None) ? VK_LOD_CLAMP_NONE : 0.0f;
    samplerInfo.maxAnisotropy = 1.0f;

    auto* res = new VulkanSamplerResource{};
    res->device = g_vkResCtx.device;
    VkResult result = vkCreateSampler(g_vkResCtx.device, &samplerInfo, nullptr, &res->sampler);
    if (result != VK_SUCCESS) {
        spdlog::error("Failed to create Vulkan sampler (VkResult: {})", static_cast<int>(result));
        delete res;
        return {};
    }
    return RhiSamplerHandle(res);
}

RhiDepthStencilStateHandle rhiCreateDepthStencilState(const RhiDevice& /*device*/,
                                                      bool /*depthWriteEnabled*/,
                                                      bool /*reversedZ*/) {
    // Vulkan handles depth/stencil state at pipeline creation time.
    // Store params so we can query later if needed.
    return RhiDepthStencilStateHandle(nullptr);
}

bool rhiSupportsRaytracing(const RhiDevice& /*device*/) {
    // TODO: Check VK_KHR_acceleration_structure + VK_KHR_ray_query support
    return false;
}

RhiTextureHandle rhiRetainTexture(const RhiTexture& texture) {
    auto* res = static_cast<VulkanTextureResource*>(texture.nativeHandle());
    if (res) {
        res->refCount++;
    }
    return RhiTextureHandle(res, texture.width(), texture.height());
}

void rhiReleaseNativeHandle(void* handle) {
    if (!handle) {
        return;
    }

    VulkanResourceHeader* header = getVulkanResourceHeader(handle);
    switch (header->type) {
    case VulkanResourceType::Texture: {
        auto* texture = static_cast<VulkanTextureResource*>(handle);
        if (texture->refCount > 1) {
            texture->refCount--;
            return;
        }
        if (texture->ownsImageView && texture->imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(texture->device, texture->imageView, nullptr);
        }
        if (texture->ownsImage && texture->image != VK_NULL_HANDLE && texture->allocator != nullptr) {
            vmaDestroyImage(texture->allocator, texture->image, texture->allocation);
        }
        delete texture;
        break;
    }
    case VulkanResourceType::Buffer: {
        auto* buffer = static_cast<VulkanBufferResource*>(handle);
        if (buffer->ownsBuffer && buffer->buffer != VK_NULL_HANDLE && buffer->allocator != nullptr) {
            vmaDestroyBuffer(buffer->allocator, buffer->buffer, buffer->allocation);
        }
        delete buffer;
        break;
    }
    case VulkanResourceType::Sampler: {
        auto* sampler = static_cast<VulkanSamplerResource*>(handle);
        if (sampler->sampler != VK_NULL_HANDLE) {
            vkDestroySampler(sampler->device, sampler->sampler, nullptr);
        }
        delete sampler;
        break;
    }
    case VulkanResourceType::Pipeline: {
        auto* pipeline = static_cast<VulkanPipelineResource*>(handle);
        if (pipeline->pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(pipeline->device, pipeline->pipeline, nullptr);
        }
        for (VkDescriptorSetLayout setLayout : pipeline->setLayouts) {
            if (setLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(pipeline->device, setLayout, nullptr);
            }
        }
        if (pipeline->ownsLayout && pipeline->layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(pipeline->device, pipeline->layout, nullptr);
        }
        delete pipeline;
        break;
    }
    case VulkanResourceType::VertexDescriptor: {
        auto* vertexDescriptor = static_cast<VulkanVertexDescriptorResource*>(handle);
        delete vertexDescriptor;
        break;
    }
    }
}

#endif
