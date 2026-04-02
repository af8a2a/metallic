#include "vulkan_transient_allocator.h"

#ifdef _WIN32

#include "rhi_backend.h"

#include <vk_mem_alloc.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>

namespace {

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0) return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

// =========================================================================
// VulkanUploadRing
// =========================================================================

void VulkanUploadRing::init(VkDevice device, VmaAllocator allocator,
                            VkDeviceSize capacityPerFrame, uint32_t framesInFlight) {
    m_device = device;
    m_allocator = allocator;
    m_capacityPerFrame = capacityPerFrame;
    m_slices.resize(framesInFlight);

    for (uint32_t i = 0; i < framesInFlight; ++i) {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = capacityPerFrame;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

        VmaAllocationInfo resultInfo{};
        if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                            &m_slices[i].buffer, &m_slices[i].allocation,
                            &resultInfo) != VK_SUCCESS) {
            spdlog::error("VulkanUploadRing: failed to create ring buffer slice {}", i);
            destroy();
            return;
        }
        m_slices[i].mappedData = resultInfo.pMappedData;
        m_slices[i].head = 0;
    }

    spdlog::info("VulkanUploadRing: initialized with {} MB per frame, {} slices",
                 capacityPerFrame / (1024 * 1024), framesInFlight);
}

void VulkanUploadRing::destroy() {
    for (auto& slice : m_slices) {
        if (slice.buffer != VK_NULL_HANDLE && m_allocator) {
            vmaDestroyBuffer(m_allocator, slice.buffer, slice.allocation);
        }
        slice = {};
    }
    m_slices.clear();
}

void VulkanUploadRing::beginFrame(uint32_t frameIndex) {
    m_currentFrame = frameIndex % static_cast<uint32_t>(m_slices.size());
    if (!m_slices.empty()) {
        m_slices[m_currentFrame].head = 0;
    }
}

VulkanUploadRing::Allocation VulkanUploadRing::allocate(VkDeviceSize size, VkDeviceSize alignment) {
    if (m_slices.empty()) {
        return {};
    }

    auto& slice = m_slices[m_currentFrame];
    const VkDeviceSize alignedOffset = alignUp(slice.head, alignment);
    if (alignedOffset + size > m_capacityPerFrame) {
        return {}; // exhausted — caller should fall back to one-shot staging
    }

    Allocation result;
    result.buffer = slice.buffer;
    result.offset = alignedOffset;
    result.mappedPtr = static_cast<uint8_t*>(slice.mappedData) + alignedOffset;
    slice.head = alignedOffset + size;
    return result;
}

VkDeviceSize VulkanUploadRing::usedThisFrame() const {
    if (m_slices.empty()) return 0;
    return m_slices[m_currentFrame].head;
}

// =========================================================================
// VulkanTransientPool
// =========================================================================

void VulkanTransientPool::init(uint32_t maxPooledTextures, uint32_t maxPooledBuffers) {
    m_maxPooledTextures = maxPooledTextures;
    m_maxPooledBuffers  = maxPooledBuffers;
    spdlog::info("VulkanTransientPool: initialized (maxTextures={}, maxBuffers={})",
                 maxPooledTextures, maxPooledBuffers);
}

void VulkanTransientPool::destroy() {
    m_texturePool.clear();
    m_bufferPool.clear();
    m_totalPooledTextures = 0;
    m_totalPooledBuffers  = 0;
}

void VulkanTransientPool::beginFrame() {
    // Nothing to do — resources stay in the pool until acquired or evicted.
}

std::unique_ptr<RhiTexture> VulkanTransientPool::acquireTexture(const RhiTextureDesc& desc) {
    TextureKey key{desc.width, desc.height, static_cast<uint32_t>(desc.format),
                   static_cast<uint32_t>(desc.usage)};
    auto it = m_texturePool.find(key);
    if (it != m_texturePool.end() && !it->second.empty()) {
        auto texture = std::move(it->second.back());
        it->second.pop_back();
        --m_totalPooledTextures;
        ++m_cacheHits;
        return texture;
    }
    ++m_cacheMisses;
    return nullptr;
}

std::unique_ptr<RhiBuffer> VulkanTransientPool::acquireBuffer(const RhiBufferDesc& desc) {
    BufferKey key{desc.size, desc.hostVisible};
    auto it = m_bufferPool.find(key);
    if (it != m_bufferPool.end() && !it->second.empty()) {
        auto buffer = std::move(it->second.back());
        it->second.pop_back();
        --m_totalPooledBuffers;
        ++m_cacheHits;
        return buffer;
    }
    ++m_cacheMisses;
    return nullptr;
}

void VulkanTransientPool::releaseTexture(std::unique_ptr<RhiTexture> texture, const RhiTextureDesc& desc) {
    if (!texture) return;

    // Evict if pool is full
    if (m_totalPooledTextures >= m_maxPooledTextures) {
        return; // let the unique_ptr destroy it
    }

    TextureKey key{desc.width, desc.height, static_cast<uint32_t>(desc.format),
                   static_cast<uint32_t>(desc.usage)};
    m_texturePool[key].push_back(std::move(texture));
    ++m_totalPooledTextures;
}

void VulkanTransientPool::releaseBuffer(std::unique_ptr<RhiBuffer> buffer, const RhiBufferDesc& desc) {
    if (!buffer) return;

    if (m_totalPooledBuffers >= m_maxPooledBuffers) {
        return;
    }

    BufferKey key{desc.size, desc.hostVisible};
    m_bufferPool[key].push_back(std::move(buffer));
    ++m_totalPooledBuffers;
}

uint32_t VulkanTransientPool::pooledTextureCount() const {
    return m_totalPooledTextures;
}

uint32_t VulkanTransientPool::pooledBufferCount() const {
    return m_totalPooledBuffers;
}

// =========================================================================
// VulkanReadbackHeap
// =========================================================================

void VulkanReadbackHeap::init(VkDevice device, VmaAllocator allocator,
                              VkDeviceSize capacityPerFrame, uint32_t framesInFlight) {
    m_device = device;
    m_allocator = allocator;
    m_capacityPerFrame = capacityPerFrame;
    m_slices.resize(framesInFlight);

    for (uint32_t i = 0; i < framesInFlight; ++i) {
        VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = capacityPerFrame;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo allocInfo{};
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

        VmaAllocationInfo resultInfo{};
        if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                            &m_slices[i].buffer, &m_slices[i].allocation,
                            &resultInfo) != VK_SUCCESS) {
            spdlog::error("VulkanReadbackHeap: failed to create readback buffer slice {}", i);
            destroy();
            return;
        }
        m_slices[i].mappedData = resultInfo.pMappedData;
        m_slices[i].head = 0;
    }

    spdlog::info("VulkanReadbackHeap: initialized with {} KB per frame, {} slices",
                 capacityPerFrame / 1024, framesInFlight);
}

void VulkanReadbackHeap::destroy() {
    for (auto& slice : m_slices) {
        if (slice.buffer != VK_NULL_HANDLE && m_allocator) {
            vmaDestroyBuffer(m_allocator, slice.buffer, slice.allocation);
        }
        slice = {};
    }
    m_slices.clear();
}

void VulkanReadbackHeap::beginFrame(uint32_t frameIndex) {
    m_currentFrame = frameIndex % static_cast<uint32_t>(m_slices.size());
    if (!m_slices.empty()) {
        m_slices[m_currentFrame].head = 0;
    }
}

VulkanReadbackHeap::Allocation VulkanReadbackHeap::allocate(VkDeviceSize size, VkDeviceSize alignment) {
    if (m_slices.empty()) {
        return {};
    }

    auto& slice = m_slices[m_currentFrame];
    const VkDeviceSize alignedOffset = alignUp(slice.head, alignment);
    if (alignedOffset + size > m_capacityPerFrame) {
        spdlog::warn("VulkanReadbackHeap: readback buffer exhausted ({} + {} > {})",
                     alignedOffset, size, m_capacityPerFrame);
        return {};
    }

    Allocation result;
    result.buffer = slice.buffer;
    result.offset = alignedOffset;
    slice.head = alignedOffset + size;
    return result;
}

bool VulkanReadbackHeap::read(uint32_t frameIndex, VkDeviceSize offset,
                              VkDeviceSize size, void* destPtr) const {
    if (m_slices.empty() || !destPtr || size == 0) {
        return false;
    }

    const uint32_t sliceIndex = frameIndex % static_cast<uint32_t>(m_slices.size());
    const auto& slice = m_slices[sliceIndex];

    if (!slice.mappedData || offset + size > m_capacityPerFrame) {
        return false;
    }

    // Invalidate before reading to ensure CPU cache coherence
    if (m_allocator && slice.allocation) {
        vmaInvalidateAllocation(m_allocator, slice.allocation, offset, size);
    }

    std::memcpy(destPtr, static_cast<const uint8_t*>(slice.mappedData) + offset, size);
    return true;
}

#endif // _WIN32
