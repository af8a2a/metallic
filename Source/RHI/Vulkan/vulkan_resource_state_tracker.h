#pragma once

#ifdef _WIN32

#include <vulkan/vulkan.h>
#include <unordered_map>
#include <vector>
#include <cstdint>

// ---------------------------------------------------------------------------
// VulkanResourceStateTracker
//
// Replaces VulkanImageLayoutTracker with a precise hazard-aware barrier system.
//
// Improvements over the old tracker:
//   - Tracks precise stage + access masks per resource (not just VkImageLayout)
//   - Tracks per-buffer state (no buffer tracking existed before)
//   - Accumulates barriers and batch-submits via a single vkCmdPipelineBarrier2
//   - Read-after-read optimization: no barrier emitted when layout is unchanged
//     and the new access is read-only (WAR/RAR-safe)
//   - Queue ownership transfer support (srcQueueFamilyIndex / dstQueueFamilyIndex)
//   - Debug statistics: barriers emitted, redundant skips, flush calls per frame
//
// Usage pattern:
//   requireImageState(image, newLayout, dstStage, dstAccess, aspect);  // accumulate
//   requireBufferState(buffer, offset, size, dstStage, dstAccess);      // accumulate
//   flushBarriers(cmd);  // single vkCmdPipelineBarrier2 for all accumulated barriers
//
// Backward-compatible convenience:
//   transition(cmd, image, newLayout, aspect);  // immediate (requires+flush in one call)
// ---------------------------------------------------------------------------

// Returns true if accessMask contains only read bits (no write bits).
// Used for read-after-read optimization.
inline bool isReadOnlyAccess(VkAccessFlags2 access) {
    constexpr VkAccessFlags2 kWriteBits =
        VK_ACCESS_2_SHADER_WRITE_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_2_TRANSFER_WRITE_BIT |
        VK_ACCESS_2_HOST_WRITE_BIT |
        VK_ACCESS_2_MEMORY_WRITE_BIT |
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    return (access & kWriteBits) == 0;
}

// Derive canonical dst stage + access from a target image layout.
// Used when callers do not supply explicit stage/access (backward-compat path).
inline void deriveImageBarrierParams(VkImageLayout layout,
                                     VkPipelineStageFlags2& outStage,
                                     VkAccessFlags2& outAccess) {
    switch (layout) {
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        outStage  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        outAccess = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        break;
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        outStage  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        outAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        break;
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        outStage  = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT |
                    VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
        outAccess = VK_ACCESS_2_SHADER_READ_BIT;
        break;
    case VK_IMAGE_LAYOUT_GENERAL:
        outStage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        outAccess = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        outStage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        outAccess = VK_ACCESS_2_TRANSFER_READ_BIT;
        break;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        outStage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        outAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        break;
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        outStage  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        outAccess = VK_ACCESS_2_NONE;
        break;
    case VK_IMAGE_LAYOUT_UNDEFINED:
        outStage  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        outAccess = VK_ACCESS_2_NONE;
        break;
    default:
        outStage  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        outAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        break;
    }
}

class VulkanResourceStateTracker {
public:
    // -----------------------------------------------------------------------
    // Per-resource state
    // -----------------------------------------------------------------------

    struct ImageState {
        VkImageLayout        layout      = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2        accessMask = VK_ACCESS_2_NONE;
        uint32_t queueFamilyIndex        = VK_QUEUE_FAMILY_IGNORED;
    };

    struct BufferState {
        VkPipelineStageFlags2 stageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2        accessMask = VK_ACCESS_2_NONE;
        uint32_t queueFamilyIndex        = VK_QUEUE_FAMILY_IGNORED;
    };

    // -----------------------------------------------------------------------
    // Per-frame debug statistics
    // -----------------------------------------------------------------------

    struct FrameStats {
        uint32_t imageBarriers   = 0;  // image memory barriers emitted
        uint32_t bufferBarriers  = 0;  // buffer memory barriers emitted
        uint32_t memoryBarriers  = 0;  // global memory barriers emitted
        uint32_t redundantSkips  = 0;  // transitions skipped (no-op)
        uint32_t flushCalls      = 0;  // flushBarriers() calls that emitted work
        uint32_t emptyFlushCalls = 0;  // flushBarriers() calls with nothing pending
    };

    // -----------------------------------------------------------------------
    // State management
    // -----------------------------------------------------------------------

    void setImageState(VkImage image, const ImageState& state) {
        m_imageStates[image] = state;
    }

    // Backward-compat: set only layout (stage/access inferred from layout)
    void setLayout(VkImage image, VkImageLayout layout) {
        ImageState state;
        state.layout = layout;
        deriveImageBarrierParams(layout, state.stageMask, state.accessMask);
        m_imageStates[image] = state;
    }

    void setBufferState(VkBuffer buffer, const BufferState& state) {
        m_bufferStates[buffer] = state;
    }

    void removeImage(VkImage image) { m_imageStates.erase(image); }
    void removeBuffer(VkBuffer buffer) { m_bufferStates.erase(buffer); }

    void clear() {
        m_imageStates.clear();
        m_bufferStates.clear();
        m_pendingImageBarriers.clear();
        m_pendingBufferBarriers.clear();
        m_pendingMemoryBarriers.clear();
        resetStats();
    }

    VkImageLayout getLayout(VkImage image) const {
        auto it = m_imageStates.find(image);
        return (it != m_imageStates.end()) ? it->second.layout : VK_IMAGE_LAYOUT_UNDEFINED;
    }

    const ImageState* getImageState(VkImage image) const {
        auto it = m_imageStates.find(image);
        return (it != m_imageStates.end()) ? &it->second : nullptr;
    }

    const BufferState* getBufferState(VkBuffer buffer) const {
        auto it = m_bufferStates.find(buffer);
        return (it != m_bufferStates.end()) ? &it->second : nullptr;
    }

    // -----------------------------------------------------------------------
    // Barrier accumulation
    // -----------------------------------------------------------------------

    // Declare that 'image' needs to be in 'newLayout' accessed via dstStage/dstAccess.
    // If the current state already satisfies the requirement (RAR optimization), no
    // barrier is accumulated. Otherwise a VkImageMemoryBarrier2 is pushed to the
    // pending list and the stored state is updated.
    //
    // Queue ownership transfer: set dstQueueFamily != VK_QUEUE_FAMILY_IGNORED and
    // different from current queueFamily to insert a release/acquire pair.
    void requireImageState(VkImage                image,
                           VkImageLayout          newLayout,
                           VkPipelineStageFlags2  dstStage,
                           VkAccessFlags2         dstAccess,
                           VkImageAspectFlags     aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           uint32_t               dstQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
        if (image == VK_NULL_HANDLE) return;

        ImageState current;
        auto it = m_imageStates.find(image);
        if (it != m_imageStates.end()) {
            current = it->second;
        }

        // Determine effective queue family comparison
        const bool queueOwnershipTransfer =
            (dstQueueFamily != VK_QUEUE_FAMILY_IGNORED) &&
            (current.queueFamilyIndex != VK_QUEUE_FAMILY_IGNORED) &&
            (current.queueFamilyIndex != dstQueueFamily);

        // Read-after-read optimization: if layout is unchanged, the new access is
        // purely read-only, and no queue transfer is needed — only accumulate stage
        // bits without emitting a barrier.
        if (current.layout == newLayout &&
            !queueOwnershipTransfer &&
            isReadOnlyAccess(dstAccess)) {
            // Accumulate stage/access into current state (multiple readers)
            if (it != m_imageStates.end()) {
                it->second.stageMask  |= dstStage;
                it->second.accessMask |= dstAccess;
            } else {
                ImageState s;
                s.layout      = newLayout;
                s.stageMask   = dstStage;
                s.accessMask  = dstAccess;
                s.queueFamilyIndex = dstQueueFamily;
                m_imageStates[image] = s;
            }
            ++m_stats.redundantSkips;
            return;
        }

        // Emit barrier
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = current.stageMask;
        barrier.srcAccessMask = current.accessMask;
        barrier.dstStageMask  = dstStage;
        barrier.dstAccessMask = dstAccess;
        barrier.oldLayout     = current.layout;
        barrier.newLayout     = newLayout;
        barrier.srcQueueFamilyIndex = queueOwnershipTransfer ? current.queueFamilyIndex : VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = queueOwnershipTransfer ? dstQueueFamily           : VK_QUEUE_FAMILY_IGNORED;
        barrier.image         = image;
        barrier.subresourceRange.aspectMask     = aspectMask;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;

        m_pendingImageBarriers.push_back(barrier);
        ++m_stats.imageBarriers;

        // Update stored state
        ImageState newState;
        newState.layout           = newLayout;
        newState.stageMask        = dstStage;
        newState.accessMask       = dstAccess;
        newState.queueFamilyIndex = queueOwnershipTransfer ? dstQueueFamily : current.queueFamilyIndex;
        m_imageStates[image] = newState;
    }

    // Convenience: derive dstStage/dstAccess from target layout (backward-compat).
    void requireImageState(VkImage            image,
                           VkImageLayout      newLayout,
                           VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                           uint32_t           dstQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
        VkPipelineStageFlags2 dstStage;
        VkAccessFlags2        dstAccess;
        deriveImageBarrierParams(newLayout, dstStage, dstAccess);
        requireImageState(image, newLayout, dstStage, dstAccess, aspectMask, dstQueueFamily);
    }

    // Declare that 'buffer' (range [offset, offset+size)) needs to be in the given
    // access state. Whole-buffer variant: use offset=0, size=VK_WHOLE_SIZE.
    void requireBufferState(VkBuffer               buffer,
                            VkDeviceSize           offset,
                            VkDeviceSize           size,
                            VkPipelineStageFlags2  dstStage,
                            VkAccessFlags2         dstAccess,
                            uint32_t               dstQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
        if (buffer == VK_NULL_HANDLE) return;

        BufferState current;
        auto it = m_bufferStates.find(buffer);
        const bool hasTrackedState = it != m_bufferStates.end();
        if (hasTrackedState) {
            current = it->second;
        } else {
            // The tracker is cleared at frame start. Treat the first observed use of an
            // externally managed buffer as already synchronized and just seed its state.
            m_bufferStates[buffer] = {dstStage, dstAccess, dstQueueFamily};
            ++m_stats.redundantSkips;
            return;
        }

        const bool queueOwnershipTransfer =
            (dstQueueFamily != VK_QUEUE_FAMILY_IGNORED) &&
            (current.queueFamilyIndex != VK_QUEUE_FAMILY_IGNORED) &&
            (current.queueFamilyIndex != dstQueueFamily);

        // RAR: read-after-read never needs a barrier
        if (!queueOwnershipTransfer && isReadOnlyAccess(dstAccess) && isReadOnlyAccess(current.accessMask)) {
            if (it != m_bufferStates.end()) {
                it->second.stageMask  |= dstStage;
                it->second.accessMask |= dstAccess;
            } else {
                m_bufferStates[buffer] = {dstStage, dstAccess, dstQueueFamily};
            }
            ++m_stats.redundantSkips;
            return;
        }

        VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        barrier.srcStageMask        = current.stageMask;
        barrier.srcAccessMask       = current.accessMask;
        barrier.dstStageMask        = dstStage;
        barrier.dstAccessMask       = dstAccess;
        barrier.srcQueueFamilyIndex = queueOwnershipTransfer ? current.queueFamilyIndex : VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = queueOwnershipTransfer ? dstQueueFamily           : VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer              = buffer;
        barrier.offset              = offset;
        barrier.size                = size;

        m_pendingBufferBarriers.push_back(barrier);
        ++m_stats.bufferBarriers;

        m_bufferStates[buffer] = {dstStage, dstAccess,
                                  queueOwnershipTransfer ? dstQueueFamily : current.queueFamilyIndex};
    }

    // Whole-buffer convenience overload.
    void requireBufferState(VkBuffer              buffer,
                            VkPipelineStageFlags2 dstStage,
                            VkAccessFlags2        dstAccess,
                            uint32_t              dstQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
        requireBufferState(buffer, 0, VK_WHOLE_SIZE, dstStage, dstAccess, dstQueueFamily);
    }

    // Add a global memory barrier to the pending batch.
    void globalMemoryBarrier(VkPipelineStageFlags2 srcStage,
                             VkAccessFlags2        srcAccess,
                             VkPipelineStageFlags2 dstStage,
                             VkAccessFlags2        dstAccess) {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = srcStage;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask  = dstStage;
        barrier.dstAccessMask = dstAccess;
        m_pendingMemoryBarriers.push_back(barrier);
        ++m_stats.memoryBarriers;
    }

    // Dynamic rendering only permits memory barriers. Collapse any pending same-queue
    // buffer barriers into one conservative memory barrier and emit it in-place.
    void flushRenderPassBarriers(VkCommandBuffer cmd) {
        const bool hasImages = !m_pendingImageBarriers.empty();
        const bool hasBuffers = !m_pendingBufferBarriers.empty();
        const bool hasMemory = !m_pendingMemoryBarriers.empty();
        const bool hasWork = hasImages || hasBuffers || hasMemory;

        if (!hasWork) {
            ++m_stats.emptyFlushCalls;
            return;
        }

        if (hasImages) {
            // Render passes must declare image transitions before vkCmdBeginRendering.
            flushBarriers(cmd);
            return;
        }

        VkMemoryBarrier2 mergedBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        bool mergedBarrierValid = false;

        auto mergeBarrier = [&](VkPipelineStageFlags2 srcStage,
                                VkAccessFlags2 srcAccess,
                                VkPipelineStageFlags2 dstStage,
                                VkAccessFlags2 dstAccess) {
            if (!mergedBarrierValid) {
                mergedBarrier.srcStageMask = srcStage;
                mergedBarrier.srcAccessMask = srcAccess;
                mergedBarrier.dstStageMask = dstStage;
                mergedBarrier.dstAccessMask = dstAccess;
                mergedBarrierValid = true;
                return;
            }

            mergedBarrier.srcStageMask |= srcStage;
            mergedBarrier.srcAccessMask |= srcAccess;
            mergedBarrier.dstStageMask |= dstStage;
            mergedBarrier.dstAccessMask |= dstAccess;
        };

        for (const auto& barrier : m_pendingMemoryBarriers) {
            mergeBarrier(barrier.srcStageMask,
                         barrier.srcAccessMask,
                         barrier.dstStageMask,
                         barrier.dstAccessMask);
        }

        for (const auto& barrier : m_pendingBufferBarriers) {
            if (barrier.srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED ||
                barrier.dstQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED) {
                flushBarriers(cmd);
                return;
            }

            mergeBarrier(barrier.srcStageMask,
                         barrier.srcAccessMask,
                         barrier.dstStageMask,
                         barrier.dstAccessMask);
        }

        if (!mergedBarrierValid) {
            ++m_stats.emptyFlushCalls;
            return;
        }

        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &mergedBarrier;

        vkCmdPipelineBarrier2(cmd, &depInfo);

        m_pendingBufferBarriers.clear();
        m_pendingMemoryBarriers.clear();

        ++m_stats.flushCalls;
    }

    // -----------------------------------------------------------------------
    // Flush: submit all accumulated barriers in a single vkCmdPipelineBarrier2
    // -----------------------------------------------------------------------

    void flushBarriers(VkCommandBuffer cmd) {
        const bool hasWork = !m_pendingImageBarriers.empty() ||
                             !m_pendingBufferBarriers.empty() ||
                             !m_pendingMemoryBarriers.empty();

        if (!hasWork) {
            ++m_stats.emptyFlushCalls;
            return;
        }

        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.memoryBarrierCount       = static_cast<uint32_t>(m_pendingMemoryBarriers.size());
        depInfo.pMemoryBarriers          = m_pendingMemoryBarriers.empty() ? nullptr : m_pendingMemoryBarriers.data();
        depInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(m_pendingBufferBarriers.size());
        depInfo.pBufferMemoryBarriers    = m_pendingBufferBarriers.empty() ? nullptr : m_pendingBufferBarriers.data();
        depInfo.imageMemoryBarrierCount  = static_cast<uint32_t>(m_pendingImageBarriers.size());
        depInfo.pImageMemoryBarriers     = m_pendingImageBarriers.empty() ? nullptr : m_pendingImageBarriers.data();

        vkCmdPipelineBarrier2(cmd, &depInfo);

        m_pendingImageBarriers.clear();
        m_pendingBufferBarriers.clear();
        m_pendingMemoryBarriers.clear();

        ++m_stats.flushCalls;
    }

    // -----------------------------------------------------------------------
    // Backward-compatible immediate transition (requireImageState + flushBarriers)
    // -----------------------------------------------------------------------

    void transition(VkCommandBuffer    cmd,
                    VkImage            image,
                    VkImageLayout      newLayout,
                    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT) {
        requireImageState(image, newLayout, aspectMask);
        flushBarriers(cmd);
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    const FrameStats& stats() const { return m_stats; }

    void resetStats() {
        m_stats = {};
    }

    uint32_t pendingBarrierCount() const {
        return static_cast<uint32_t>(m_pendingImageBarriers.size() +
                                     m_pendingBufferBarriers.size() +
                                     m_pendingMemoryBarriers.size());
    }

private:
    std::unordered_map<VkImage,  ImageState>  m_imageStates;
    std::unordered_map<VkBuffer, BufferState> m_bufferStates;

    std::vector<VkImageMemoryBarrier2>  m_pendingImageBarriers;
    std::vector<VkBufferMemoryBarrier2> m_pendingBufferBarriers;
    std::vector<VkMemoryBarrier2>       m_pendingMemoryBarriers;

    FrameStats m_stats{};
};

#endif // _WIN32
