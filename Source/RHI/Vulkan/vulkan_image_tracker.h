#pragma once

#ifdef _WIN32

#include <vulkan/vulkan.h>
#include <unordered_map>

// Tracks VkImage layouts and inserts pipeline barriers for layout transitions.
// Uses VK_KHR_synchronization2 (Vulkan 1.3 core).
class VulkanImageLayoutTracker {
public:
    // Set the known layout for an image (e.g., after swapchain creation)
    void setLayout(VkImage image, VkImageLayout layout) {
        m_layouts[image] = layout;
    }

    // Get the current known layout (UNDEFINED if unknown)
    VkImageLayout getLayout(VkImage image) const {
        auto it = m_layouts.find(image);
        return (it != m_layouts.end()) ? it->second : VK_IMAGE_LAYOUT_UNDEFINED;
    }

    // Remove tracking for an image (e.g., when destroyed)
    void remove(VkImage image) {
        m_layouts.erase(image);
    }

    // Clear all tracked layouts
    void clear() {
        m_layouts.clear();
    }

    // Transition an image to a new layout, inserting a pipeline barrier.
    // Automatically determines src/dst stages and access masks from layouts.
    void transition(VkCommandBuffer cmd, VkImage image, VkImageLayout newLayout,
                    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT) {
        VkImageLayout oldLayout = getLayout(image);
        if (oldLayout == newLayout) return;

        VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkAccessFlags2 srcAccess = VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        VkAccessFlags2 dstAccess = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

        // Refine based on old layout
        switch (oldLayout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            srcAccess = VK_ACCESS_2_NONE;
            break;
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            srcStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            srcAccess = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            srcStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            srcAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            srcStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            srcAccess = VK_ACCESS_2_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL:
            srcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            srcAccess = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            srcStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            srcAccess = VK_ACCESS_2_TRANSFER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            srcStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            srcAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            srcStage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            srcAccess = VK_ACCESS_2_NONE;
            break;
        default:
            break;
        }

        // Refine based on new layout
        switch (newLayout) {
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            dstStage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            dstAccess = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            dstStage = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            dstAccess = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            dstStage = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            dstAccess = VK_ACCESS_2_SHADER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_GENERAL:
            dstStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            dstAccess = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            dstStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            dstAccess = VK_ACCESS_2_TRANSFER_READ_BIT;
            break;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            dstStage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            dstAccess = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            break;
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            dstStage = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            dstAccess = VK_ACCESS_2_NONE;
            break;
        default:
            break;
        }

        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask = srcStage;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask = dstStage;
        barrier.dstAccessMask = dstAccess;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspectMask;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        m_layouts[image] = newLayout;
    }

private:
    std::unordered_map<VkImage, VkImageLayout> m_layouts;
};

#endif // _WIN32
