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
VulkanResourceContextInfo g_vkResCtx;

template <typename Fn>
Fn loadHookedDeviceProc(PFN_vkGetDeviceProcAddr deviceProcAddrProxy,
                        VkDevice device,
                        const char* name) {
    if (!deviceProcAddrProxy || device == VK_NULL_HANDLE || !name) {
        return nullptr;
    }
    return reinterpret_cast<Fn>(deviceProcAddrProxy(device, name));
}

void setResourceContext(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator,
                        VkQueue queue, uint32_t queueFamily, bool bufferDeviceAddressEnabled,
                        bool externalHostMemoryEnabled,
                        bool rayTracingEnabled,
                        void* vkGetDeviceProcAddrProxy) {
    PFN_vkGetDeviceProcAddr deviceProcAddrProxy =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(vkGetDeviceProcAddrProxy);

    g_vkResCtx.device = device;
    g_vkResCtx.physicalDevice = physicalDevice;
    g_vkResCtx.allocator = allocator;
    g_vkResCtx.graphicsQueue = queue;
    g_vkResCtx.graphicsQueueFamily = queueFamily;
    g_vkResCtx.bufferDeviceAddressEnabled = bufferDeviceAddressEnabled;
    g_vkResCtx.externalHostMemoryEnabled = externalHostMemoryEnabled;
    g_vkResCtx.rayTracingEnabled = rayTracingEnabled;
    g_vkResCtx.streamlineHooksEnabled = false;
    g_vkResCtx.vkBeginCommandBufferProxy =
        loadHookedDeviceProc<PFN_vkBeginCommandBuffer>(deviceProcAddrProxy,
                                                       device,
                                                       "vkBeginCommandBuffer");
    g_vkResCtx.vkCmdBindPipelineProxy =
        loadHookedDeviceProc<PFN_vkCmdBindPipeline>(deviceProcAddrProxy,
                                                    device,
                                                    "vkCmdBindPipeline");
    g_vkResCtx.vkCmdBindDescriptorSetsProxy =
        loadHookedDeviceProc<PFN_vkCmdBindDescriptorSets>(deviceProcAddrProxy,
                                                          device,
                                                          "vkCmdBindDescriptorSets");
    g_vkResCtx.initialized = true;
}

PFN_vkDestroyAccelerationStructureKHR loadDestroyAccelerationStructure(VkDevice device) {
    return reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
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
    vulkanBeginCommandBufferHooked(cmd, &beginInfo);
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

RhiContext* resolveOwningContext(const RhiDevice& device, const char* functionName) {
    RhiContext* context = device.ownerContext();
    if (!context) {
        spdlog::error("{} requires an owning RhiContext on Vulkan", functionName);
        return nullptr;
    }
    if (context->backendType() != RhiBackendType::Vulkan) {
        spdlog::error("{} received a non-Vulkan RhiContext", functionName);
        return nullptr;
    }
    return context;
}

} // namespace

void vulkanSetResourceContext(VkDevice device,
                              VkPhysicalDevice physicalDevice,
                              VmaAllocator allocator,
                              VkQueue queue,
                              uint32_t queueFamily,
                              bool bufferDeviceAddressEnabled,
                              bool externalHostMemoryEnabled,
                              bool rayTracingEnabled,
                              void* vkGetDeviceProcAddrProxy) {
    setResourceContext(device,
                       physicalDevice,
                       allocator,
                       queue,
                       queueFamily,
                       bufferDeviceAddressEnabled,
                       externalHostMemoryEnabled,
                       rayTracingEnabled,
                       vkGetDeviceProcAddrProxy);
}

const VulkanResourceContextInfo& vulkanGetResourceContext() {
    return g_vkResCtx;
}

void vulkanSetStreamlineHookedCommandsEnabled(bool enabled) {
    g_vkResCtx.streamlineHooksEnabled = enabled;
}

VkResult vulkanBeginCommandBufferHooked(VkCommandBuffer commandBuffer,
                                        const VkCommandBufferBeginInfo* beginInfo) {
    if (g_vkResCtx.streamlineHooksEnabled && g_vkResCtx.vkBeginCommandBufferProxy) {
        return g_vkResCtx.vkBeginCommandBufferProxy(commandBuffer, beginInfo);
    }
    return vkBeginCommandBuffer(commandBuffer, beginInfo);
}

void vulkanCmdBindPipelineHooked(VkCommandBuffer commandBuffer,
                                 VkPipelineBindPoint pipelineBindPoint,
                                 VkPipeline pipeline) {
    if (g_vkResCtx.streamlineHooksEnabled && g_vkResCtx.vkCmdBindPipelineProxy) {
        g_vkResCtx.vkCmdBindPipelineProxy(commandBuffer, pipelineBindPoint, pipeline);
        return;
    }
    vkCmdBindPipeline(commandBuffer, pipelineBindPoint, pipeline);
}

void vulkanCmdBindDescriptorSetsHooked(VkCommandBuffer commandBuffer,
                                       VkPipelineBindPoint pipelineBindPoint,
                                       VkPipelineLayout layout,
                                       uint32_t firstSet,
                                       uint32_t descriptorSetCount,
                                       const VkDescriptorSet* descriptorSets,
                                       uint32_t dynamicOffsetCount,
                                       const uint32_t* dynamicOffsets) {
    if (g_vkResCtx.streamlineHooksEnabled && g_vkResCtx.vkCmdBindDescriptorSetsProxy) {
        g_vkResCtx.vkCmdBindDescriptorSetsProxy(commandBuffer,
                                                pipelineBindPoint,
                                                layout,
                                                firstSet,
                                                descriptorSetCount,
                                                descriptorSets,
                                                dynamicOffsetCount,
                                                dynamicOffsets);
        return;
    }
    vkCmdBindDescriptorSets(commandBuffer,
                            pipelineBindPoint,
                            layout,
                            firstSet,
                            descriptorSetCount,
                            descriptorSets,
                            dynamicOffsetCount,
                            dynamicOffsets);
}

void vulkanClearResourceContext() {
    if (g_uploadPool != VK_NULL_HANDLE && g_vkResCtx.device != VK_NULL_HANDLE) {
        vkDestroyCommandPool(g_vkResCtx.device, g_uploadPool, nullptr);
        g_uploadPool = VK_NULL_HANDLE;
    }
    g_vkResCtx = {};
}

RhiBufferHandle rhiCreateSharedBuffer(const RhiDevice& device,
                                      const void* initialData,
                                      size_t size,
                                      const char* debugName) {
    RhiContext* context = resolveOwningContext(device, "rhiCreateSharedBuffer");
    return context ? context->createSharedBuffer(initialData, size, debugName) : RhiBufferHandle{};
}

void* rhiBufferContents(const RhiBuffer& buffer) {
    auto* res = getVulkanBufferResource(buffer);
    return res ? res->mappedData : nullptr;
}

RhiTextureHandle rhiCreateTexture2D(const RhiDevice& device,
                                    uint32_t width,
                                    uint32_t height,
                                    RhiFormat format,
                                    bool mipmapped,
                                    uint32_t mipLevelCount,
                                    RhiTextureStorageMode storageMode,
                                    RhiTextureUsage usage) {
    RhiContext* context = resolveOwningContext(device, "rhiCreateTexture2D");
    return context ? context->createTexture2D(width,
                                              height,
                                              format,
                                              mipmapped,
                                              mipLevelCount,
                                              storageMode,
                                              usage)
                   : RhiTextureHandle{};
}

RhiTextureHandle rhiCreateTexture3D(const RhiDevice& device,
                                    uint32_t width,
                                    uint32_t height,
                                    uint32_t depth,
                                    RhiFormat format,
                                    RhiTextureStorageMode storageMode,
                                    RhiTextureUsage usage) {
    RhiContext* context = resolveOwningContext(device, "rhiCreateTexture3D");
    return context ? context->createTexture3D(width,
                                              height,
                                              depth,
                                              format,
                                              storageMode,
                                              usage)
                   : RhiTextureHandle{};
}

void rhiUploadTexture2D(const RhiTexture& texture,
                        uint32_t width,
                        uint32_t height,
                        const void* data,
                        size_t bytesPerRow,
                        uint32_t mipLevel) {
    auto* res = static_cast<VulkanTextureResource*>(texture.nativeHandle());
    if (!res || !data) return;

    const bool deferShaderReadTransition = res->mipLevels > 1 && mipLevel == 0;
    size_t imageSize = bytesPerRow * height;

    // Create staging buffer
    VmaBufferCreateInfo stagingVmaInfo{};
    stagingVmaInfo.device = g_vkResCtx.device;
    stagingVmaInfo.allocator = g_vkResCtx.allocator;
    stagingVmaInfo.size = imageSize;
    stagingVmaInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingVmaInfo.hostVisible = true;
    stagingVmaInfo.externalMemoryHandleTypes =
        g_vkResCtx.externalHostMemoryEnabled
            ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT
            : 0;

    auto stagingResource = vmaCreateBufferResource(stagingVmaInfo);
    if (!stagingResource) {
        spdlog::error("Failed to create staging buffer for 2D texture upload");
        return;
    }
    VkBuffer stagingBuffer = stagingResource->buffer;
    VmaAllocation stagingAlloc = stagingResource->allocation;

    std::memcpy(stagingResource->mappedData, data, imageSize);

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

    if (!deferShaderReadTransition) {
        // Transition to SHADER_READ
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

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

    VmaBufferCreateInfo stagingVmaInfo{};
    stagingVmaInfo.device = g_vkResCtx.device;
    stagingVmaInfo.allocator = g_vkResCtx.allocator;
    stagingVmaInfo.size = imageSize;
    stagingVmaInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingVmaInfo.hostVisible = true;
    stagingVmaInfo.externalMemoryHandleTypes =
        g_vkResCtx.externalHostMemoryEnabled
            ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT
            : 0;

    auto stagingResource = vmaCreateBufferResource(stagingVmaInfo);
    if (!stagingResource) {
        spdlog::error("Failed to create staging buffer for 3D texture upload");
        return;
    }
    VkBuffer stagingBuffer = stagingResource->buffer;
    VmaAllocation stagingAlloc = stagingResource->allocation;
    std::memcpy(stagingResource->mappedData, data, imageSize);

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

    for (uint32_t i = 1; i < res->mipLevels; ++i) {
        VkImageMemoryBarrier2 initBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        initBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        initBarrier.srcAccessMask = VK_ACCESS_2_NONE;
        initBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        initBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        initBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        initBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        initBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initBarrier.image = res->image;
        initBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i, 1, 0, 1};

        VkDependencyInfo initDep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        initDep.imageMemoryBarrierCount = 1;
        initDep.pImageMemoryBarriers = &initBarrier;
        vkCmdPipelineBarrier2(cmd, &initDep);
    }

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

RhiSamplerHandle rhiCreateSampler(const RhiDevice& device, const RhiSamplerDesc& desc) {
    RhiContext* context = resolveOwningContext(device, "rhiCreateSampler");
    return context ? context->createSampler(desc) : RhiSamplerHandle{};
}

RhiDepthStencilStateHandle rhiCreateDepthStencilState(const RhiDevice& device,
                                                      bool depthWriteEnabled,
                                                      bool reversedZ) {
    RhiContext* context = resolveOwningContext(device, "rhiCreateDepthStencilState");
    return context ? context->createDepthStencilState(depthWriteEnabled, reversedZ)
                   : RhiDepthStencilStateHandle{};
}

bool rhiSupportsRaytracing(const RhiDevice& /*device*/) {
    return g_vkResCtx.rayTracingEnabled;
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
        vmaDestroyImageResource(*texture);
        delete texture;
        break;
    }
    case VulkanResourceType::Buffer: {
        auto* buffer = static_cast<VulkanBufferResource*>(handle);
        vmaDestroyBufferResource(*buffer);
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
            pipeline->pipeline = VK_NULL_HANDLE;
        }
        for (size_t setIndex = 0; setIndex < pipeline->setLayouts.size(); ++setIndex) {
            VkDescriptorSetLayout setLayout = pipeline->setLayouts[setIndex];
            if (setLayout != VK_NULL_HANDLE) {
                const bool ownsSetLayout =
                    setIndex >= pipeline->setLayoutOwnership.size() ||
                    pipeline->setLayoutOwnership[setIndex] != 0;
                if (ownsSetLayout) {
                    vkDestroyDescriptorSetLayout(pipeline->device, setLayout, nullptr);
                } else if (pipeline->device != VK_NULL_HANDLE &&
                           pipeline->bindlessSetIndex == static_cast<uint32_t>(setIndex)) {
                    vulkanReleaseBindlessSetLayout(pipeline->device);
                }
            }
        }
        pipeline->setLayouts.clear();
        pipeline->setLayoutOwnership.clear();
        pipeline->bindlessSetIndex = UINT32_MAX;
        if (pipeline->ownsLayout && pipeline->layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(pipeline->device, pipeline->layout, nullptr);
            pipeline->layout = VK_NULL_HANDLE;
        }
        delete pipeline;
        break;
    }
    case VulkanResourceType::VertexDescriptor: {
        auto* vertexDescriptor = static_cast<VulkanVertexDescriptorResource*>(handle);
        delete vertexDescriptor;
        break;
    }
    case VulkanResourceType::AccelerationStructure: {
        auto* accelerationStructure = static_cast<VulkanAccelerationStructureResource*>(handle);
        PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure =
            loadDestroyAccelerationStructure(accelerationStructure->device);
        if (destroyAccelerationStructure &&
            accelerationStructure->accelerationStructure != VK_NULL_HANDLE) {
            destroyAccelerationStructure(accelerationStructure->device,
                                         accelerationStructure->accelerationStructure,
                                         nullptr);
        }
        VulkanBufferResource asBuffer{};
        asBuffer.buffer = accelerationStructure->buffer;
        asBuffer.allocation = accelerationStructure->allocation;
        asBuffer.allocator = accelerationStructure->allocator;
        vmaDestroyBufferResource(asBuffer);
        delete accelerationStructure;
        break;
    }
    }
}

#endif
