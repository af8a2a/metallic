#pragma once

#ifdef _WIN32

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

struct RhiTextureDesc;
struct RhiBufferDesc;
class RhiTexture;
class RhiBuffer;
class RhiFrameGraphBackend;

// =========================================================================
// VulkanUploadRing — per-frame ring buffer for staging uploads
// =========================================================================
//
// Replaces per-upload staging allocation with a persistent mapped ring.
// Double-buffered (one slice per frame-in-flight).

class VulkanUploadRing {
public:
    VulkanUploadRing() = default;
    ~VulkanUploadRing() = default;

    VulkanUploadRing(const VulkanUploadRing&) = delete;
    VulkanUploadRing& operator=(const VulkanUploadRing&) = delete;

    void init(VkDevice device, VmaAllocator allocator,
              VkDeviceSize capacityPerFrame, uint32_t framesInFlight);
    void destroy();

    // Call at the start of each frame (after fence wait) to reset the ring head.
    void beginFrame(uint32_t frameIndex);

    struct Allocation {
        VkBuffer  buffer    = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        void*     mappedPtr = nullptr;
        bool valid() const { return buffer != VK_NULL_HANDLE; }
    };

    // Sub-allocate from the current frame's ring.
    // Returns invalid allocation if the ring is exhausted.
    Allocation allocate(VkDeviceSize size, VkDeviceSize alignment = 16);

    bool isValid() const { return !m_slices.empty(); }
    VkDeviceSize capacityPerFrame() const { return m_capacityPerFrame; }
    VkDeviceSize usedThisFrame() const;

private:
    struct Slice {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        void* mappedData = nullptr;
        VkDeviceSize head = 0;
    };

    VkDevice     m_device    = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;
    VkDeviceSize m_capacityPerFrame = 0;
    uint32_t     m_currentFrame = 0;
    std::vector<Slice> m_slices;
};

// =========================================================================
// VulkanTransientPool — reusable cache for FrameGraph transient resources
// =========================================================================
//
// Transient textures/buffers that are created and destroyed each frame by the
// FrameGraph are instead returned to this pool and reused when a matching
// descriptor is requested on a future frame.

class VulkanTransientPool {
public:
    VulkanTransientPool() = default;
    ~VulkanTransientPool() = default;

    VulkanTransientPool(const VulkanTransientPool&) = delete;
    VulkanTransientPool& operator=(const VulkanTransientPool&) = delete;

    void init(uint32_t maxPooledTextures = 128, uint32_t maxPooledBuffers = 64);
    void destroy();

    // Call at the start of each frame.
    void beginFrame();

    // Acquire a resource matching `desc` from the pool.
    // Returns nullptr if none available — caller should create a fresh one.
    std::unique_ptr<RhiTexture> acquireTexture(const RhiTextureDesc& desc);
    std::unique_ptr<RhiBuffer>  acquireBuffer(const RhiBufferDesc& desc);

    // Return a resource to the pool for future reuse.
    void releaseTexture(std::unique_ptr<RhiTexture> texture, const RhiTextureDesc& desc);
    void releaseBuffer(std::unique_ptr<RhiBuffer> buffer, const RhiBufferDesc& desc);

    // Stats
    uint32_t pooledTextureCount() const;
    uint32_t pooledBufferCount() const;
    uint32_t cacheHits() const   { return m_cacheHits; }
    uint32_t cacheMisses() const { return m_cacheMisses; }
    void resetStats() { m_cacheHits = 0; m_cacheMisses = 0; }

private:
    // Hash key for texture descriptors
    struct TextureKey {
        uint32_t width;
        uint32_t height;
        uint32_t format;
        uint32_t usage;
        bool operator==(const TextureKey& other) const {
            return width == other.width && height == other.height &&
                   format == other.format && usage == other.usage;
        }
    };
    struct TextureKeyHash {
        size_t operator()(const TextureKey& k) const {
            size_t h = std::hash<uint32_t>{}(k.width);
            h ^= std::hash<uint32_t>{}(k.height) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(k.format) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(k.usage)  + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct BufferKey {
        uint64_t size;
        bool hostVisible;
        bool operator==(const BufferKey& other) const {
            return size == other.size && hostVisible == other.hostVisible;
        }
    };
    struct BufferKeyHash {
        size_t operator()(const BufferKey& k) const {
            size_t h = std::hash<uint64_t>{}(k.size);
            h ^= std::hash<bool>{}(k.hostVisible) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<TextureKey, std::vector<std::unique_ptr<RhiTexture>>, TextureKeyHash> m_texturePool;
    std::unordered_map<BufferKey, std::vector<std::unique_ptr<RhiBuffer>>, BufferKeyHash>   m_bufferPool;

    uint32_t m_maxPooledTextures = 128;
    uint32_t m_maxPooledBuffers  = 64;
    uint32_t m_totalPooledTextures = 0;
    uint32_t m_totalPooledBuffers  = 0;
    uint32_t m_cacheHits   = 0;
    uint32_t m_cacheMisses = 0;
};

// =========================================================================
// VulkanReadbackHeap — per-frame host-visible buffer for GPU→CPU transfers
// =========================================================================
//
// Double-buffered so frame N can read back results from frame N-2
// (after the corresponding fence has signaled).

class VulkanReadbackHeap {
public:
    VulkanReadbackHeap() = default;
    ~VulkanReadbackHeap() = default;

    VulkanReadbackHeap(const VulkanReadbackHeap&) = delete;
    VulkanReadbackHeap& operator=(const VulkanReadbackHeap&) = delete;

    void init(VkDevice device, VmaAllocator allocator,
              VkDeviceSize capacityPerFrame, uint32_t framesInFlight);
    void destroy();

    // Call at the start of each frame.
    void beginFrame(uint32_t frameIndex);

    struct Allocation {
        VkBuffer     buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        bool valid() const { return buffer != VK_NULL_HANDLE; }
    };

    // Allocate space for a GPU→CPU copy target in the current frame's slice.
    Allocation allocate(VkDeviceSize size, VkDeviceSize alignment = 16);

    // Read back data from the PREVIOUS frame's readback buffer.
    // Only valid after the previous frame's fence has been waited on.
    bool read(uint32_t frameIndex, VkDeviceSize offset, VkDeviceSize size, void* destPtr) const;

    bool isValid() const { return !m_slices.empty(); }
    VkDeviceSize capacityPerFrame() const { return m_capacityPerFrame; }

private:
    struct Slice {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        void* mappedData = nullptr;
        VkDeviceSize head = 0;
    };

    VkDevice     m_device    = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;
    VkDeviceSize m_capacityPerFrame = 0;
    uint32_t     m_currentFrame = 0;
    std::vector<Slice> m_slices;
};

#endif // _WIN32
