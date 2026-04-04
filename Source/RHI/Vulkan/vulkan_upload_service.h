#pragma once

#ifdef _WIN32

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

class VulkanUploadRing;
class VulkanReadbackHeap;

// =========================================================================
// VulkanUploadService — formal upload service layer
// =========================================================================
//
// Wraps VulkanUploadRing to provide:
// - Immediate (blocking) uploads for startup/asset loading
// - Deferred per-frame uploads recorded into a command buffer
// - Async transfer queue path (when available) with timeline semaphore sync
//
// For immediate uploads that exceed the ring capacity, falls back to a
// temporary VMA staging buffer (same as the old one-shot pattern but
// centralized).

class VulkanUploadService {
public:
    void init(VkDevice device,
              VmaAllocator allocator,
              VkQueue graphicsQueue,
              uint32_t graphicsQueueFamily,
              VkQueue transferQueue,                // VK_NULL_HANDLE if unavailable
              uint32_t transferQueueFamily,          // UINT32_MAX if unavailable
              VkSemaphore transferTimelineSemaphore, // VK_NULL_HANDLE if unavailable
              VulkanUploadRing* uploadRing);
    void destroy();

    // --- Deferred (per-frame) uploads ---

    // Stage texture data; copy commands recorded later via recordPendingUploads().
    bool stageTexture2D(VkImage dstImage, uint32_t width, uint32_t height,
                        const void* data, size_t dataSize, uint32_t mipLevel = 0,
                        bool deferShaderReadTransition = false);
    bool stageTexture3D(VkImage dstImage, uint32_t width, uint32_t height, uint32_t depth,
                        const void* data, size_t dataSize, uint32_t mipLevel = 0);
    bool stageBuffer(VkBuffer dstBuffer, VkDeviceSize dstOffset,
                     const void* data, VkDeviceSize size);

    // Record all pending copy commands (with layout transitions) into cmd.
    void recordPendingUploads(VkCommandBuffer cmd);

    // --- Immediate (blocking) uploads ---

    void immediateUploadTexture2D(VkImage dstImage, uint32_t width, uint32_t height,
                                  const void* data, size_t dataSize, uint32_t mipLevel = 0,
                                  bool deferShaderReadTransition = false);
    void immediateUploadTexture3D(VkImage dstImage, uint32_t width, uint32_t height,
                                  uint32_t depth, const void* data, size_t dataSize,
                                  uint32_t mipLevel = 0);
    void immediateUploadBuffer(VkBuffer dstBuffer, VkDeviceSize dstOffset,
                               const void* data, VkDeviceSize size);

    // --- Async transfer queue ---

    // Submit pending uploads to dedicated transfer queue.
    // Returns timeline semaphore value to wait on, or 0 if no transfer queue.
    uint64_t submitAsyncTransfer();

    bool hasPendingUploads() const { return !m_pendingUploads.empty(); }
    bool hasTransferQueue() const { return m_transferQueue != VK_NULL_HANDLE; }

private:
    struct DeferredUpload {
        VkBuffer srcBuffer;
        VkDeviceSize srcOffset;
        VkDeviceSize size;
        // Texture uploads
        VkImage dstImage;
        uint32_t width, height, depth;
        uint32_t mipLevel;
        bool deferShaderReadTransition;
        // Buffer uploads
        VkBuffer dstBuffer;
        VkDeviceSize dstBufferOffset;
        // Type tag
        bool isTexture;
    };

    // Staging helpers
    struct StagingAlloc {
        VkBuffer buffer;
        VkDeviceSize offset;
        void* mappedPtr;
        bool fromRing; // true = ring suballoc, false = standalone VMA buffer
        VmaAllocation standaloneAllocation; // only when fromRing==false
    };
    StagingAlloc allocateStaging(VkDeviceSize size);
    void freeStandaloneStaging();

    // One-shot command buffer helpers
    VkCommandBuffer beginOneTimeCommands(VkCommandPool pool);
    void endOneTimeCommands(VkCommandPool pool, VkQueue queue, VkCommandBuffer cmd);

    // Record copy commands for a single upload
    static void recordTextureCopy(VkCommandBuffer cmd, const DeferredUpload& upload,
                                  VkBuffer srcBuffer, VkDeviceSize srcOffset);
    static void recordBufferCopy(VkCommandBuffer cmd, const DeferredUpload& upload,
                                 VkBuffer srcBuffer, VkDeviceSize srcOffset);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamily = 0;
    VkQueue m_transferQueue = VK_NULL_HANDLE;
    uint32_t m_transferQueueFamily = UINT32_MAX;
    VkSemaphore m_transferTimelineSemaphore = VK_NULL_HANDLE;
    uint64_t m_transferTimelineValue = 0;
    VulkanUploadRing* m_uploadRing = nullptr;
    VkCommandPool m_immediateCommandPool = VK_NULL_HANDLE;
    VkCommandPool m_transferCommandPool = VK_NULL_HANDLE;
    std::vector<DeferredUpload> m_pendingUploads;

    // Standalone staging buffers that need cleanup after submit
    struct StandaloneStaging {
        VkBuffer buffer;
        VmaAllocation allocation;
    };
    std::vector<StandaloneStaging> m_standaloneStagingBuffers;
};

// =========================================================================
// VulkanReadbackService — GPU→CPU readback with fence-tracked visibility
// =========================================================================
//
// Wraps VulkanReadbackHeap to provide a request-based readback API.
// Schedule a readback, it gets recorded into the current frame's command
// buffer, and the data becomes readable after the frame's fence signals
// (typically 2 frames later with double-buffering).

class VulkanReadbackService {
public:
    void init(VkDevice device, VulkanReadbackHeap* readbackHeap, uint32_t framesInFlight);
    void destroy();

    struct ReadbackRequest {
        uint32_t id = 0;
        uint32_t frameSubmitted = 0;
        VkDeviceSize heapOffset = 0;
        VkDeviceSize size = 0;
        bool valid() const { return id != 0; }
    };

    // Schedule a GPU buffer readback. Returns a request handle.
    ReadbackRequest scheduleBufferReadback(VkBuffer srcBuffer, VkDeviceSize srcOffset,
                                            VkDeviceSize size);

    // Record pending readback copy commands into the command buffer.
    void recordPendingReadbacks(VkCommandBuffer cmd);

    // Check if readback data is ready (frame fence has signaled).
    bool isReady(const ReadbackRequest& request, uint32_t currentFrame) const;

    // Read the data into destPtr. Returns false if not ready or invalid.
    bool readData(const ReadbackRequest& request, void* destPtr, VkDeviceSize size) const;

    // Call at start of each frame to advance internal state.
    void beginFrame(uint32_t frameIndex);

private:
    struct PendingReadback {
        uint32_t id;
        VkBuffer srcBuffer;
        VkDeviceSize srcOffset;
        VkBuffer dstBuffer;      // readback heap buffer for the frame
        VkDeviceSize heapOffset;
        VkDeviceSize size;
        uint32_t frameIndex;
    };

    VkDevice m_device = VK_NULL_HANDLE;
    VulkanReadbackHeap* m_readbackHeap = nullptr;
    uint32_t m_framesInFlight = 2;
    uint32_t m_currentFrame = 0;
    uint32_t m_nextId = 1;
    std::vector<PendingReadback> m_pendingThisFrame;
    std::vector<PendingReadback> m_inFlight; // submitted, waiting for fence
};

#endif // _WIN32
