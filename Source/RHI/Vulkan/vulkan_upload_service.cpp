#include "vulkan_upload_service.h"

#ifdef _WIN32

#include "vulkan_transient_allocator.h"

#include <vk_mem_alloc.h>
#include <spdlog/spdlog.h>

#include <cstring>

// =========================================================================
// VulkanUploadService
// =========================================================================

void VulkanUploadService::init(VkDevice device,
                               VmaAllocator allocator,
                               VkQueue graphicsQueue,
                               uint32_t graphicsQueueFamily,
                               VkQueue transferQueue,
                               uint32_t transferQueueFamily,
                               VkSemaphore transferTimelineSemaphore,
                               VulkanUploadRing* uploadRing) {
    m_device = device;
    m_allocator = allocator;
    m_graphicsQueue = graphicsQueue;
    m_graphicsQueueFamily = graphicsQueueFamily;
    m_transferQueue = transferQueue;
    m_transferQueueFamily = transferQueueFamily;
    m_transferTimelineSemaphore = transferTimelineSemaphore;
    m_transferTimelineValue = 0;
    m_uploadRing = uploadRing;

    // Command pool for immediate (blocking) uploads on the graphics queue
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = graphicsQueueFamily;
    vkCreateCommandPool(device, &poolInfo, nullptr, &m_immediateCommandPool);

    // Command pool for async transfer queue (if available)
    if (m_transferQueue != VK_NULL_HANDLE && transferQueueFamily != UINT32_MAX) {
        VkCommandPoolCreateInfo transferPoolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        transferPoolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        transferPoolInfo.queueFamilyIndex = transferQueueFamily;
        vkCreateCommandPool(device, &transferPoolInfo, nullptr, &m_transferCommandPool);
    }

    spdlog::info("VulkanUploadService: initialized (transferQueue={})",
                 m_transferQueue != VK_NULL_HANDLE ? "available" : "none");
}

void VulkanUploadService::destroy() {
    freeStandaloneStaging();
    m_pendingUploads.clear();

    if (m_immediateCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_immediateCommandPool, nullptr);
        m_immediateCommandPool = VK_NULL_HANDLE;
    }
    if (m_transferCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_transferCommandPool, nullptr);
        m_transferCommandPool = VK_NULL_HANDLE;
    }
    m_device = VK_NULL_HANDLE;
}

// --- Staging allocation ---

VulkanUploadService::StagingAlloc VulkanUploadService::allocateStaging(VkDeviceSize size) {
    // Try ring buffer first
    if (m_uploadRing && m_uploadRing->isValid()) {
        auto ringAlloc = m_uploadRing->allocate(size);
        if (ringAlloc.valid()) {
            return {ringAlloc.buffer, ringAlloc.offset, ringAlloc.mappedPtr, true, nullptr};
        }
    }

    // Fallback: standalone VMA staging buffer
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo resultInfo{};
    if (vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo,
                        &buffer, &allocation, &resultInfo) != VK_SUCCESS) {
        spdlog::error("VulkanUploadService: failed to allocate standalone staging buffer ({} bytes)", size);
        return {VK_NULL_HANDLE, 0, nullptr, false, nullptr};
    }

    m_standaloneStagingBuffers.push_back({buffer, allocation});
    return {buffer, 0, resultInfo.pMappedData, false, allocation};
}

void VulkanUploadService::freeStandaloneStaging() {
    for (auto& s : m_standaloneStagingBuffers) {
        if (s.buffer != VK_NULL_HANDLE && m_allocator) {
            vmaDestroyBuffer(m_allocator, s.buffer, s.allocation);
        }
    }
    m_standaloneStagingBuffers.clear();
}

// --- Deferred staging ---

bool VulkanUploadService::stageTexture2D(VkImage dstImage, uint32_t width, uint32_t height,
                                          const void* data, size_t dataSize, uint32_t mipLevel,
                                          bool deferShaderReadTransition) {
    if (!data || dataSize == 0) return false;

    auto staging = allocateStaging(static_cast<VkDeviceSize>(dataSize));
    if (!staging.mappedPtr) return false;

    std::memcpy(staging.mappedPtr, data, dataSize);

    DeferredUpload upload{};
    upload.srcBuffer = staging.buffer;
    upload.srcOffset = staging.offset;
    upload.size = static_cast<VkDeviceSize>(dataSize);
    upload.dstImage = dstImage;
    upload.width = width;
    upload.height = height;
    upload.depth = 1;
    upload.mipLevel = mipLevel;
    upload.deferShaderReadTransition = deferShaderReadTransition;
    upload.dstBuffer = VK_NULL_HANDLE;
    upload.dstBufferOffset = 0;
    upload.isTexture = true;
    m_pendingUploads.push_back(upload);
    return true;
}

bool VulkanUploadService::stageTexture3D(VkImage dstImage, uint32_t width, uint32_t height,
                                          uint32_t depth, const void* data, size_t dataSize,
                                          uint32_t mipLevel) {
    if (!data || dataSize == 0) return false;

    auto staging = allocateStaging(static_cast<VkDeviceSize>(dataSize));
    if (!staging.mappedPtr) return false;

    std::memcpy(staging.mappedPtr, data, dataSize);

    DeferredUpload upload{};
    upload.srcBuffer = staging.buffer;
    upload.srcOffset = staging.offset;
    upload.size = static_cast<VkDeviceSize>(dataSize);
    upload.dstImage = dstImage;
    upload.width = width;
    upload.height = height;
    upload.depth = depth;
    upload.mipLevel = mipLevel;
    upload.deferShaderReadTransition = false;
    upload.dstBuffer = VK_NULL_HANDLE;
    upload.dstBufferOffset = 0;
    upload.isTexture = true;
    m_pendingUploads.push_back(upload);
    return true;
}

bool VulkanUploadService::stageBuffer(VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                       const void* data, VkDeviceSize size) {
    if (!data || size == 0) return false;

    auto staging = allocateStaging(size);
    if (!staging.mappedPtr) return false;

    std::memcpy(staging.mappedPtr, data, static_cast<size_t>(size));

    DeferredUpload upload{};
    upload.srcBuffer = staging.buffer;
    upload.srcOffset = staging.offset;
    upload.size = size;
    upload.dstImage = VK_NULL_HANDLE;
    upload.width = 0;
    upload.height = 0;
    upload.depth = 0;
    upload.mipLevel = 0;
    upload.deferShaderReadTransition = false;
    upload.dstBuffer = dstBuffer;
    upload.dstBufferOffset = dstOffset;
    upload.isTexture = false;
    m_pendingUploads.push_back(upload);
    return true;
}

// --- Record copy commands ---

void VulkanUploadService::recordTextureCopy(VkCommandBuffer cmd, const DeferredUpload& upload,
                                             VkBuffer srcBuffer, VkDeviceSize srcOffset) {
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
    barrier.image = upload.dstImage;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, upload.mipLevel, 1, 0, 1};

    VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.imageMemoryBarrierCount = 1;
    depInfo.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    // Copy buffer to image
    VkBufferImageCopy region{};
    region.bufferOffset = srcOffset;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, upload.mipLevel, 0, 1};
    region.imageExtent = {upload.width, upload.height, upload.depth};
    vkCmdCopyBufferToImage(cmd, srcBuffer, upload.dstImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition to SHADER_READ (unless deferred for mipmap generation)
    if (!upload.deferShaderReadTransition) {
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
}

void VulkanUploadService::recordBufferCopy(VkCommandBuffer cmd, const DeferredUpload& upload,
                                            VkBuffer srcBuffer, VkDeviceSize srcOffset) {
    VkBufferCopy region{};
    region.srcOffset = srcOffset;
    region.dstOffset = upload.dstBufferOffset;
    region.size = upload.size;
    vkCmdCopyBuffer(cmd, srcBuffer, upload.dstBuffer, 1, &region);
}

void VulkanUploadService::recordPendingUploads(VkCommandBuffer cmd) {
    for (const auto& upload : m_pendingUploads) {
        if (upload.isTexture) {
            recordTextureCopy(cmd, upload, upload.srcBuffer, upload.srcOffset);
        } else {
            recordBufferCopy(cmd, upload, upload.srcBuffer, upload.srcOffset);
        }
    }
    m_pendingUploads.clear();
    // Note: standalone staging buffers freed after the command buffer completes (next frame's fence)
}

// --- One-shot command buffer helpers ---

VkCommandBuffer VulkanUploadService::beginOneTimeCommands(VkCommandPool pool) {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = pool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

void VulkanUploadService::endOneTimeCommands(VkCommandPool pool, VkQueue queue,
                                              VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(m_device, pool, 1, &cmd);
}

// --- Immediate uploads ---

void VulkanUploadService::immediateUploadTexture2D(VkImage dstImage, uint32_t width,
                                                    uint32_t height, const void* data,
                                                    size_t dataSize, uint32_t mipLevel,
                                                    bool deferShaderReadTransition) {
    if (!data || dataSize == 0 || m_device == VK_NULL_HANDLE) return;

    auto staging = allocateStaging(static_cast<VkDeviceSize>(dataSize));
    if (!staging.mappedPtr) return;

    std::memcpy(staging.mappedPtr, data, dataSize);

    DeferredUpload upload{};
    upload.srcBuffer = staging.buffer;
    upload.srcOffset = staging.offset;
    upload.size = static_cast<VkDeviceSize>(dataSize);
    upload.dstImage = dstImage;
    upload.width = width;
    upload.height = height;
    upload.depth = 1;
    upload.mipLevel = mipLevel;
    upload.deferShaderReadTransition = deferShaderReadTransition;
    upload.isTexture = true;

    VkCommandBuffer cmd = beginOneTimeCommands(m_immediateCommandPool);
    recordTextureCopy(cmd, upload, staging.buffer, staging.offset);
    endOneTimeCommands(m_immediateCommandPool, m_graphicsQueue, cmd);

    // Free standalone staging immediately (GPU is idle after vkQueueWaitIdle)
    freeStandaloneStaging();
}

void VulkanUploadService::immediateUploadTexture3D(VkImage dstImage, uint32_t width,
                                                    uint32_t height, uint32_t depth,
                                                    const void* data, size_t dataSize,
                                                    uint32_t mipLevel) {
    if (!data || dataSize == 0 || m_device == VK_NULL_HANDLE) return;

    auto staging = allocateStaging(static_cast<VkDeviceSize>(dataSize));
    if (!staging.mappedPtr) return;

    std::memcpy(staging.mappedPtr, data, dataSize);

    DeferredUpload upload{};
    upload.srcBuffer = staging.buffer;
    upload.srcOffset = staging.offset;
    upload.size = static_cast<VkDeviceSize>(dataSize);
    upload.dstImage = dstImage;
    upload.width = width;
    upload.height = height;
    upload.depth = depth;
    upload.mipLevel = mipLevel;
    upload.deferShaderReadTransition = false;
    upload.isTexture = true;

    VkCommandBuffer cmd = beginOneTimeCommands(m_immediateCommandPool);
    recordTextureCopy(cmd, upload, staging.buffer, staging.offset);
    endOneTimeCommands(m_immediateCommandPool, m_graphicsQueue, cmd);
    freeStandaloneStaging();
}

void VulkanUploadService::immediateUploadBuffer(VkBuffer dstBuffer, VkDeviceSize dstOffset,
                                                 const void* data, VkDeviceSize size) {
    if (!data || size == 0 || m_device == VK_NULL_HANDLE) return;

    auto staging = allocateStaging(size);
    if (!staging.mappedPtr) return;

    std::memcpy(staging.mappedPtr, data, static_cast<size_t>(size));

    DeferredUpload upload{};
    upload.srcBuffer = staging.buffer;
    upload.srcOffset = staging.offset;
    upload.size = size;
    upload.dstBuffer = dstBuffer;
    upload.dstBufferOffset = dstOffset;
    upload.isTexture = false;

    VkCommandBuffer cmd = beginOneTimeCommands(m_immediateCommandPool);
    recordBufferCopy(cmd, upload, staging.buffer, staging.offset);
    endOneTimeCommands(m_immediateCommandPool, m_graphicsQueue, cmd);
    freeStandaloneStaging();
}

// --- Async transfer queue ---

uint64_t VulkanUploadService::submitAsyncTransfer() {
    if (m_transferQueue == VK_NULL_HANDLE || m_transferCommandPool == VK_NULL_HANDLE ||
        m_transferTimelineSemaphore == VK_NULL_HANDLE || m_pendingUploads.empty()) {
        return 0;
    }

    VkCommandBuffer cmd = beginOneTimeCommands(m_transferCommandPool);
    for (const auto& upload : m_pendingUploads) {
        if (upload.isTexture) {
            recordTextureCopy(cmd, upload, upload.srcBuffer, upload.srcOffset);
        } else {
            recordBufferCopy(cmd, upload, upload.srcBuffer, upload.srcOffset);
        }
    }
    vkEndCommandBuffer(cmd);
    m_pendingUploads.clear();

    ++m_transferTimelineValue;

    VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signalInfo.semaphore = m_transferTimelineSemaphore;
    signalInfo.value = m_transferTimelineValue;
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;

    VkCommandBufferSubmitInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdInfo.commandBuffer = cmd;

    VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos = &signalInfo;

    vkQueueSubmit2(m_transferQueue, 1, &submitInfo, VK_NULL_HANDLE);

    // Note: command buffer freed on next submitAsyncTransfer or destroy
    // (pool is TRANSIENT, reset implicit)

    return m_transferTimelineValue;
}

// =========================================================================
// VulkanReadbackService
// =========================================================================

void VulkanReadbackService::init(VkDevice device, VulkanReadbackHeap* readbackHeap,
                                  uint32_t framesInFlight) {
    m_device = device;
    m_readbackHeap = readbackHeap;
    m_framesInFlight = framesInFlight;
    m_nextId = 1;
    spdlog::info("VulkanReadbackService: initialized (framesInFlight={})", framesInFlight);
}

void VulkanReadbackService::destroy() {
    m_pendingThisFrame.clear();
    m_inFlight.clear();
    m_device = VK_NULL_HANDLE;
    m_readbackHeap = nullptr;
}

void VulkanReadbackService::beginFrame(uint32_t frameIndex) {
    m_currentFrame = frameIndex;

    // Retire in-flight readbacks that are now safe to read
    // (their frame's fence has been waited on by the caller before beginFrame)
    // We keep them in m_inFlight — callers query via isReady/readData.
    // Remove entries older than framesInFlight to avoid unbounded growth.
    m_inFlight.erase(
        std::remove_if(m_inFlight.begin(), m_inFlight.end(),
            [&](const PendingReadback& r) {
                // If the readback's frame is more than framesInFlight old, discard it
                return (frameIndex - r.frameIndex) > m_framesInFlight * 2;
            }),
        m_inFlight.end());
}

VulkanReadbackService::ReadbackRequest
VulkanReadbackService::scheduleBufferReadback(VkBuffer srcBuffer, VkDeviceSize srcOffset,
                                               VkDeviceSize size) {
    if (!m_readbackHeap || !m_readbackHeap->isValid() || size == 0) {
        return {};
    }

    auto heapAlloc = m_readbackHeap->allocate(size);
    if (!heapAlloc.valid()) {
        spdlog::warn("VulkanReadbackService: readback heap full, dropping request");
        return {};
    }

    uint32_t id = m_nextId++;
    PendingReadback pending{};
    pending.id = id;
    pending.srcBuffer = srcBuffer;
    pending.srcOffset = srcOffset;
    pending.heapOffset = heapAlloc.offset;
    pending.dstBuffer = heapAlloc.buffer;
    pending.size = size;
    pending.frameIndex = m_currentFrame;
    m_pendingThisFrame.push_back(pending);

    ReadbackRequest req{};
    req.id = id;
    req.frameSubmitted = m_currentFrame;
    req.heapOffset = heapAlloc.offset;
    req.size = size;
    return req;
}

void VulkanReadbackService::recordPendingReadbacks(VkCommandBuffer cmd) {
    for (const auto& r : m_pendingThisFrame) {
        VkBufferCopy region{};
        region.srcOffset = r.srcOffset;
        region.dstOffset = r.heapOffset;
        region.size = r.size;
        vkCmdCopyBuffer(cmd, r.srcBuffer, r.dstBuffer, 1, &region);
    }

    // Move to in-flight
    for (auto& r : m_pendingThisFrame) {
        m_inFlight.push_back(r);
    }
    m_pendingThisFrame.clear();
}

bool VulkanReadbackService::isReady(const ReadbackRequest& request, uint32_t currentFrame) const {
    if (!request.valid()) return false;
    // Ready when at least framesInFlight frames have passed
    return (currentFrame - request.frameSubmitted) >= m_framesInFlight;
}

bool VulkanReadbackService::readData(const ReadbackRequest& request, void* destPtr,
                                      VkDeviceSize size) const {
    if (!request.valid() || !m_readbackHeap || !destPtr || size == 0) return false;
    if (size > request.size) size = request.size;
    return m_readbackHeap->read(request.frameSubmitted, request.heapOffset, size, destPtr);
}

#endif // _WIN32
