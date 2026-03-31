#include "vulkan_frame_graph.h"

#ifdef _WIN32

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "imgui.h"
#include "imgui_impl_vulkan.h"

#include <array>
#include <cstring>
#include <stdexcept>

#include "../rhi_resource_utils.h"

// Mesh shader extension function pointers (loaded dynamically)
static PFN_vkCmdDrawMeshTasksEXT pfnCmdDrawMeshTasksEXT = nullptr;
static PFN_vkCmdDrawMeshTasksIndirectEXT pfnCmdDrawMeshTasksIndirectEXT = nullptr;

void vulkanLoadMeshShaderFunctions(VkDevice device) {
    if (!pfnCmdDrawMeshTasksEXT) {
        pfnCmdDrawMeshTasksEXT = reinterpret_cast<PFN_vkCmdDrawMeshTasksEXT>(
            vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksEXT"));
    }
    if (!pfnCmdDrawMeshTasksIndirectEXT) {
        pfnCmdDrawMeshTasksIndirectEXT = reinterpret_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(
            vkGetDeviceProcAddr(device, "vkCmdDrawMeshTasksIndirectEXT"));
    }
}

namespace {

// Helper to check Vulkan results
void checkVk(VkResult result, const char* message) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(message) + " (VkResult: " + std::to_string(result) + ")");
    }
}

VkImageAspectFlags imageAspectMask(const VulkanTextureResource* resource) {
    if (!resource) {
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }

    switch (resource->format) {
    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

} // namespace

// Owned texture with VMA allocation backed by a shared VulkanTextureResource wrapper.
class VulkanOwnedTexture final : public RhiTexture {
public:
    explicit VulkanOwnedTexture(VulkanTextureResource resource)
        : m_resource(resource) {}

    ~VulkanOwnedTexture() override {
        vmaDestroyImageResource(m_resource);
    }

    void* nativeHandle() const override { return const_cast<VulkanTextureResource*>(&m_resource); }
    uint32_t width() const override { return m_resource.width; }
    uint32_t height() const override { return m_resource.height; }

    VkImage image() const { return m_resource.image; }
    VkImageView imageView() const { return m_resource.imageView; }

private:
    VulkanTextureResource m_resource{};
};

namespace {

// Owned buffer with VMA allocation
class VulkanOwnedBuffer final : public RhiBuffer {
public:
    explicit VulkanOwnedBuffer(VulkanBufferResource resource)
        : m_resource(resource) {}

    ~VulkanOwnedBuffer() override {
        vmaDestroyBufferResource(m_resource);
    }

    size_t size() const override { return m_resource.size; }
    void* nativeHandle() const override { return const_cast<VulkanBufferResource*>(&m_resource); }
    void* mappedData() override {
        if (m_resource.mappedData == nullptr && m_resource.allocation != nullptr) {
            vmaMapMemory(m_resource.allocator, m_resource.allocation, &m_resource.mappedData);
        }
        return m_resource.mappedData;
    }

    VkBuffer buffer() const { return m_resource.buffer; }

private:
    VulkanBufferResource m_resource{};
};

// Render command encoder
class VulkanRenderCommandEncoder final : public RhiRenderCommandEncoder {
public:
    VulkanRenderCommandEncoder(VkCommandBuffer commandBuffer, VkDevice device,
                               VulkanDescriptorManager* descriptorManager,
                               VulkanResourceStateTracker* stateTracker)
        : m_commandBuffer(commandBuffer), m_device(device),
          m_descriptorManager(descriptorManager), m_stateTracker(stateTracker) {
        m_pendingBuffers.fill({});
        m_pendingTextures.fill({});
        m_pendingSamplers.fill({});
        m_pendingAccelerationStructures.fill({});
    }

    ~VulkanRenderCommandEncoder() override {
        if (m_commandBuffer != VK_NULL_HANDLE) {
            vkCmdEndRendering(m_commandBuffer);
        }
    }

    void* nativeHandle() const override { return m_commandBuffer; }

    void setViewport(float width, float height, bool flipY = true) override {
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.width = width;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        if (flipY) {
            viewport.y = height;
            viewport.height = -height;
        } else {
            viewport.y = 0.0f;
            viewport.height = height;
        }
        vkCmdSetViewport(m_commandBuffer, 0, 1, &viewport);
    }

    void setDepthStencilState(const RhiDepthStencilState* /*state*/) override {}

    void setFrontFacingWinding(RhiWinding winding) override {
        vkCmdSetFrontFace(m_commandBuffer,
            winding == RhiWinding::Clockwise ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE);
    }

    void setCullMode(RhiCullMode cullMode) override {
        VkCullModeFlags vkCull = VK_CULL_MODE_NONE;
        if (cullMode == RhiCullMode::Front) vkCull = VK_CULL_MODE_FRONT_BIT;
        else if (cullMode == RhiCullMode::Back) vkCull = VK_CULL_MODE_BACK_BIT;
        vkCmdSetCullMode(m_commandBuffer, vkCull);
    }

    void setRenderPipeline(const RhiGraphicsPipeline& pipeline) override {
        m_boundPipeline = getVulkanPipelineResource(pipeline);
        vulkanCmdBindPipelineHooked(m_commandBuffer,
                                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    getVulkanPipelineHandle(pipeline));
    }

    void setVertexBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) override {
        if (!buffer) return;
        VkBuffer vkBuf = getVulkanBufferHandle(buffer);
        VkDeviceSize vkOffset = offset;
        vkCmdBindVertexBuffers(m_commandBuffer, index, 1, &vkBuf, &vkOffset);
    }

    void setFragmentBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) override {
        if (!buffer || index >= kMaxBufferBindings) return;
        m_pendingBuffers[index] = {getVulkanBufferHandle(buffer), offset, buffer->size()};
    }

    void setMeshBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) override {
        if (!buffer || index >= kMaxBufferBindings) return;
        m_pendingBuffers[index] = {getVulkanBufferHandle(buffer), offset, buffer->size()};
    }

    void setVertexBytes(const void* data, size_t size, uint32_t index) override {
        if (index >= kMaxBufferBindings || !m_descriptorManager) {
            return;
        }
        m_pendingBuffers[index] = m_descriptorManager->uploadInlineUniformData(data, size);
    }

    void setFragmentBytes(const void* data, size_t size, uint32_t index) override {
        if (index >= kMaxBufferBindings || !m_descriptorManager) {
            return;
        }
        m_pendingBuffers[index] = m_descriptorManager->uploadInlineUniformData(data, size);
    }

    void setMeshBytes(const void* data, size_t size, uint32_t index) override {
        if (index >= kMaxBufferBindings || !m_descriptorManager) {
            return;
        }
        m_pendingBuffers[index] = m_descriptorManager->uploadInlineUniformData(data, size);
    }

    void setPushConstants(const void* data, size_t size) override {
        if (!m_boundPipeline || m_boundPipeline->layout == VK_NULL_HANDLE) return;
        vkCmdPushConstants(m_commandBuffer, m_boundPipeline->layout,
                           VK_SHADER_STAGE_ALL, 0,
                           static_cast<uint32_t>(size), data);
    }

    void setFragmentTexture(const RhiTexture* texture, uint32_t index) override {
        if (index >= kMaxTextureBindings) return;
        auto* resource = getVulkanTextureResource(texture);
        VkImageView view = getVulkanImageView(texture);
        m_pendingTextures[index] = {resource, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true, false};
    }

    void setFragmentTextures(const RhiTexture* const* textures, uint32_t startIndex, uint32_t count) override {
        for (uint32_t i = 0; i < count && (startIndex + i) < kMaxTextureBindings; ++i) {
            auto* resource = getVulkanTextureResource(textures[i]);
            VkImageView view = getVulkanImageView(textures[i]);
            m_pendingTextures[startIndex + i] = {resource, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true, false};
        }
    }

    void setMeshTextures(const RhiTexture* const* textures, uint32_t startIndex, uint32_t count) override {
        setFragmentTextures(textures, startIndex, count);
    }

    void setFragmentSampler(const RhiSampler* sampler, uint32_t index) override {
        if (!sampler || index >= kMaxSamplerBindings) return;
        m_pendingSamplers[index] = {getVulkanSamplerHandle(sampler), true};
    }

    void setMeshSampler(const RhiSampler* sampler, uint32_t index) override {
        setFragmentSampler(sampler, index);
    }

    void drawPrimitives(RhiPrimitiveType /*primitiveType*/, uint32_t vertexStart, uint32_t vertexCount) override {
        flushDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS);
        vkCmdDraw(m_commandBuffer, vertexCount, 1, vertexStart, 0);
    }

    void drawIndexedPrimitives(RhiPrimitiveType /*primitiveType*/,
                               uint32_t indexCount,
                               RhiIndexType indexType,
                               const RhiBuffer& indexBuffer,
                               uint64_t indexBufferOffset) override {
        VkIndexType vkIndexType = (indexType == RhiIndexType::UInt16) ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        vkCmdBindIndexBuffer(m_commandBuffer, getVulkanBufferHandle(&indexBuffer),
                             indexBufferOffset, vkIndexType);
        flushDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS);
        vkCmdDrawIndexed(m_commandBuffer, indexCount, 1, 0, 0, 0);
    }

    void drawMeshThreadgroups(RhiSize3D threadgroupsPerGrid,
                              RhiSize3D /*threadsPerObjectThreadgroup*/,
                              RhiSize3D /*threadsPerMeshThreadgroup*/) override {
        flushDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS);
        if (pfnCmdDrawMeshTasksEXT) {
            pfnCmdDrawMeshTasksEXT(m_commandBuffer,
                                   threadgroupsPerGrid.width,
                                   threadgroupsPerGrid.height,
                                   threadgroupsPerGrid.depth);
        }
    }

    void drawMeshThreadgroupsIndirect(const RhiBuffer& indirectBuffer,
                                      uint64_t indirectBufferOffset,
                                      RhiSize3D /*threadsPerObjectThreadgroup*/,
                                      RhiSize3D /*threadsPerMeshThreadgroup*/) override {
        flushDescriptors(VK_PIPELINE_BIND_POINT_GRAPHICS);
        if (pfnCmdDrawMeshTasksIndirectEXT) {
            pfnCmdDrawMeshTasksIndirectEXT(m_commandBuffer,
                                           getVulkanBufferHandle(&indirectBuffer),
                                           indirectBufferOffset, 1, 0);
        }
    }

    void renderImGuiDrawData() override {
        ImDrawData* drawData = ImGui::GetDrawData();
        if (!drawData) {
            return;
        }
        if (m_commandBuffer != VK_NULL_HANDLE) {
            ImGui_ImplVulkan_RenderDrawData(drawData, m_commandBuffer);
        }
    }

private:
    void transitionPendingTextures() {
        if (!m_stateTracker) {
            return;
        }

        for (const auto& texture : m_pendingTextures) {
            if (!texture.resource || texture.imageView == VK_NULL_HANDLE) {
                continue;
            }

            const uint32_t usageMask = static_cast<uint32_t>(texture.resource->usage);
            const bool needsTrackedLayout = texture.isStorage ||
                (usageMask & static_cast<uint32_t>(RhiTextureUsage::ShaderWrite)) != 0 ||
                (usageMask & static_cast<uint32_t>(RhiTextureUsage::RenderTarget)) != 0;
            if (!needsTrackedLayout) {
                continue;
            }

            m_stateTracker->requireImageState(texture.resource->image, texture.layout,
                                              imageAspectMask(texture.resource));
        }
    }

    void flushDescriptors(VkPipelineBindPoint bindPoint) {
        if (m_descriptorManager && m_boundPipeline) {
            transitionPendingTextures();
            // Batch-flush all accumulated barriers before binding descriptors and drawing
            if (m_stateTracker) {
                m_stateTracker->flushBarriers(m_commandBuffer);
            }
            m_descriptorManager->flushAndBind(m_commandBuffer,
                                              bindPoint,
                                              *m_boundPipeline,
                                              m_pendingBuffers,
                                              m_pendingTextures,
                                              m_pendingSamplers,
                                              m_pendingAccelerationStructures);
        }
    }

    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VulkanDescriptorManager* m_descriptorManager = nullptr;
    VulkanResourceStateTracker* m_stateTracker = nullptr;
    const VulkanPipelineResource* m_boundPipeline = nullptr;
    std::array<PendingBufferBinding, kMaxBufferBindings> m_pendingBuffers{};
    std::array<PendingTextureBinding, kMaxTextureBindings> m_pendingTextures{};
    std::array<PendingSamplerBinding, kMaxSamplerBindings> m_pendingSamplers{};
    std::array<PendingAccelerationStructureBinding, kMaxAccelerationStructureBindings>
        m_pendingAccelerationStructures{};
};

// Compute command encoder
class VulkanComputeCommandEncoder final : public RhiComputeCommandEncoder {
public:
    VulkanComputeCommandEncoder(VkCommandBuffer commandBuffer, VkDevice device,
                                VulkanDescriptorManager* descriptorManager,
                                VulkanResourceStateTracker* stateTracker)
        : m_commandBuffer(commandBuffer), m_device(device),
          m_descriptorManager(descriptorManager), m_stateTracker(stateTracker) {
        m_pendingBuffers.fill({});
        m_pendingTextures.fill({});
        m_pendingSamplers.fill({});
        m_pendingAccelerationStructures.fill({});
    }

    ~VulkanComputeCommandEncoder() override = default;

    void* nativeHandle() const override { return m_commandBuffer; }

    void setComputePipeline(const RhiComputePipeline& pipeline) override {
        m_boundPipeline = getVulkanPipelineResource(pipeline);
        vulkanCmdBindPipelineHooked(m_commandBuffer,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    getVulkanPipelineHandle(pipeline));
    }

    void setBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) override {
        if (!buffer || index >= kMaxBufferBindings) return;
        m_pendingBuffers[index] = {getVulkanBufferHandle(buffer), offset, buffer->size()};
    }

    void setBytes(const void* data, size_t size, uint32_t index) override {
        if (index >= kMaxBufferBindings || !m_descriptorManager) {
            return;
        }
        m_pendingBuffers[index] = m_descriptorManager->uploadInlineUniformData(data, size);
    }

    void setPushConstants(const void* data, size_t size) override {
        if (!m_boundPipeline || m_boundPipeline->layout == VK_NULL_HANDLE) return;
        vkCmdPushConstants(m_commandBuffer, m_boundPipeline->layout,
                           VK_SHADER_STAGE_ALL, 0,
                           static_cast<uint32_t>(size), data);
    }

    void setTexture(const RhiTexture* texture, uint32_t index) override {
        if (index >= kMaxTextureBindings) return;
        auto* resource = getVulkanTextureResource(texture);
        VkImageView view = getVulkanImageView(texture);
        m_pendingTextures[index] = {resource, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true, false};
    }

    void setStorageTexture(const RhiTexture* texture, uint32_t index) override {
        if (index >= kMaxTextureBindings) return;
        auto* resource = getVulkanTextureResource(texture);
        VkImageView view = getVulkanImageView(texture);
        m_pendingTextures[index] = {resource, view, VK_IMAGE_LAYOUT_GENERAL, true, true};
    }

    void setTextures(const RhiTexture* const* textures, uint32_t startIndex, uint32_t count) override {
        for (uint32_t i = 0; i < count && (startIndex + i) < kMaxTextureBindings; ++i) {
            auto* resource = getVulkanTextureResource(textures[i]);
            VkImageView view = getVulkanImageView(textures[i]);
            m_pendingTextures[startIndex + i] = {resource, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, true, false};
        }
    }

    void setSampler(const RhiSampler* sampler, uint32_t index) override {
        if (!sampler || index >= kMaxSamplerBindings) return;
        m_pendingSamplers[index] = {getVulkanSamplerHandle(sampler), true};
    }

    void setAccelerationStructure(const RhiAccelerationStructure* accelerationStructure, uint32_t index) override {
        if (!accelerationStructure) {
            return;
        }

        uint32_t resolvedIndex = index;
        if (resolvedIndex >= kMaxAccelerationStructureBindings) {
            resolvedIndex = findFallbackAccelerationStructureBindingIndex();
        } else if (m_boundPipeline &&
                   !m_boundPipeline->accelerationStructureBindings[resolvedIndex].valid()) {
            const uint32_t fallbackIndex = findFallbackAccelerationStructureBindingIndex();
            if (fallbackIndex < kMaxAccelerationStructureBindings) {
                resolvedIndex = fallbackIndex;
            }
        }

        if (resolvedIndex >= kMaxAccelerationStructureBindings) {
            return;
        }

        m_pendingAccelerationStructures[resolvedIndex] = {
            getVulkanAccelerationStructureHandle(accelerationStructure),
            true,
        };
    }

    void useResource(const RhiBuffer& /*resource*/, RhiResourceUsage /*usage*/) override {}
    void useResource(const RhiAccelerationStructure& resource, RhiResourceUsage usage) override {
        if ((static_cast<uint32_t>(usage) & static_cast<uint32_t>(RhiResourceUsage::Read)) == 0 ||
            !resource.nativeHandle()) {
            return;
        }

        // AS build → compute read barrier (always emitted regardless of state tracking)
        if (m_stateTracker) {
            m_stateTracker->globalMemoryBarrier(
                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);
            m_stateTracker->flushBarriers(m_commandBuffer);
        }
    }

    void memoryBarrier(RhiBarrierScope /*scope*/) override {
        // Emit a batched global memory barrier for compute→graphics/compute handoff.
        if (m_stateTracker) {
            m_stateTracker->globalMemoryBarrier(
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                    VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
                VK_ACCESS_2_SHADER_READ_BIT |
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
            m_stateTracker->flushBarriers(m_commandBuffer);
        }
    }

    void dispatchThreadgroups(RhiSize3D threadgroupsPerGrid, RhiSize3D /*threadsPerThreadgroup*/) override {
        flushDescriptors();
        vkCmdDispatch(m_commandBuffer,
                      threadgroupsPerGrid.width,
                      threadgroupsPerGrid.height,
                      threadgroupsPerGrid.depth);
    }

private:
    void transitionPendingTextures() {
        if (!m_stateTracker) {
            return;
        }

        for (const auto& texture : m_pendingTextures) {
            if (!texture.resource || texture.imageView == VK_NULL_HANDLE) {
                continue;
            }

            const uint32_t usageMask = static_cast<uint32_t>(texture.resource->usage);
            const bool needsTrackedLayout = texture.isStorage ||
                (usageMask & static_cast<uint32_t>(RhiTextureUsage::ShaderWrite)) != 0 ||
                (usageMask & static_cast<uint32_t>(RhiTextureUsage::RenderTarget)) != 0;
            if (!needsTrackedLayout) {
                continue;
            }

            m_stateTracker->requireImageState(texture.resource->image, texture.layout,
                                              imageAspectMask(texture.resource));
        }
    }

    void flushDescriptors() {
        if (m_descriptorManager && m_boundPipeline) {
            transitionPendingTextures();
            // Batch-flush all accumulated barriers before dispatch
            if (m_stateTracker) {
                m_stateTracker->flushBarriers(m_commandBuffer);
            }
            m_descriptorManager->flushAndBind(m_commandBuffer,
                                              VK_PIPELINE_BIND_POINT_COMPUTE,
                                              *m_boundPipeline,
                                              m_pendingBuffers,
                                              m_pendingTextures,
                                              m_pendingSamplers,
                                              m_pendingAccelerationStructures);
        }
    }

    uint32_t findFallbackAccelerationStructureBindingIndex() const {
        if (!m_boundPipeline) {
            return kMaxAccelerationStructureBindings;
        }

        uint32_t fallbackIndex = kMaxAccelerationStructureBindings;
        uint32_t validCount = 0;
        for (uint32_t index = 0; index < kMaxAccelerationStructureBindings; ++index) {
            if (m_boundPipeline->accelerationStructureBindings[index].valid()) {
                fallbackIndex = index;
                ++validCount;
            }
        }
        return validCount == 1 ? fallbackIndex : kMaxAccelerationStructureBindings;
    }

    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VulkanDescriptorManager* m_descriptorManager = nullptr;
    VulkanResourceStateTracker* m_stateTracker = nullptr;
    const VulkanPipelineResource* m_boundPipeline = nullptr;
    std::array<PendingBufferBinding, kMaxBufferBindings> m_pendingBuffers{};
    std::array<PendingTextureBinding, kMaxTextureBindings> m_pendingTextures{};
    std::array<PendingSamplerBinding, kMaxSamplerBindings> m_pendingSamplers{};
    std::array<PendingAccelerationStructureBinding, kMaxAccelerationStructureBindings>
        m_pendingAccelerationStructures{};
};

// Blit command encoder
class VulkanBlitCommandEncoder final : public RhiBlitCommandEncoder {
public:
    VulkanBlitCommandEncoder(VkCommandBuffer commandBuffer, VkDevice device,
                             VulkanResourceStateTracker* stateTracker = nullptr)
        : m_commandBuffer(commandBuffer), m_device(device), m_stateTracker(stateTracker) {}

    ~VulkanBlitCommandEncoder() override = default;

    void* nativeHandle() const override { return m_commandBuffer; }

    void copyTexture(const RhiTexture& source,
                     uint32_t /*sourceSlice*/,
                     uint32_t sourceLevel,
                     RhiOrigin3D sourceOrigin,
                     RhiSize3D sourceSize,
                     const RhiTexture& destination,
                     uint32_t /*destinationSlice*/,
                     uint32_t destinationLevel,
                     RhiOrigin3D destinationOrigin) override {
        // Ensure source and destination are in correct transfer layouts
        if (m_stateTracker) {
            m_stateTracker->requireImageState(getVulkanImage(&source),
                                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            m_stateTracker->requireImageState(getVulkanImage(&destination),
                                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            m_stateTracker->flushBarriers(m_commandBuffer);
        }

        VkImageCopy region{};
        region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.srcSubresource.mipLevel = sourceLevel;
        region.srcSubresource.baseArrayLayer = 0;
        region.srcSubresource.layerCount = 1;
        region.srcOffset = {static_cast<int32_t>(sourceOrigin.x),
                            static_cast<int32_t>(sourceOrigin.y),
                            static_cast<int32_t>(sourceOrigin.z)};
        region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.dstSubresource.mipLevel = destinationLevel;
        region.dstSubresource.baseArrayLayer = 0;
        region.dstSubresource.layerCount = 1;
        region.dstOffset = {static_cast<int32_t>(destinationOrigin.x),
                            static_cast<int32_t>(destinationOrigin.y),
                            static_cast<int32_t>(destinationOrigin.z)};
        region.extent = {sourceSize.width, sourceSize.height, sourceSize.depth};

        vkCmdCopyImage(m_commandBuffer,
                       getVulkanImage(&source),
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       getVulkanImage(&destination),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region);
    }

private:
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VulkanResourceStateTracker* m_stateTracker = nullptr;
};

bool isDepthFormat(RhiFormat format) {
    return format == RhiFormat::D32Float || format == RhiFormat::D16Unorm;
}

} // namespace

// --- Format conversion helpers (non-static, declared in header) ---

VkFormat toVkFormat(RhiFormat format) {
    switch (format) {
    case RhiFormat::R8Unorm:     return VK_FORMAT_R8_UNORM;
    case RhiFormat::R16Float:    return VK_FORMAT_R16_SFLOAT;
    case RhiFormat::R32Float:    return VK_FORMAT_R32_SFLOAT;
    case RhiFormat::R32Uint:     return VK_FORMAT_R32_UINT;
    case RhiFormat::RG8Unorm:    return VK_FORMAT_R8G8_UNORM;
    case RhiFormat::RG16Float:   return VK_FORMAT_R16G16_SFLOAT;
    case RhiFormat::RG32Float:   return VK_FORMAT_R32G32_SFLOAT;
    case RhiFormat::RGBA8Unorm:  return VK_FORMAT_R8G8B8A8_UNORM;
    case RhiFormat::RGBA8Srgb:   return VK_FORMAT_R8G8B8A8_SRGB;
    case RhiFormat::BGRA8Unorm:  return VK_FORMAT_B8G8R8A8_UNORM;
    case RhiFormat::RGBA16Float: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case RhiFormat::RGBA32Float: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case RhiFormat::D32Float:    return VK_FORMAT_D32_SFLOAT;
    case RhiFormat::D16Unorm:    return VK_FORMAT_D16_UNORM;
    case RhiFormat::Undefined:
    default:                     return VK_FORMAT_UNDEFINED;
    }
}

RhiFormat fromVkFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM:              return RhiFormat::R8Unorm;
    case VK_FORMAT_R16_SFLOAT:            return RhiFormat::R16Float;
    case VK_FORMAT_R32_SFLOAT:            return RhiFormat::R32Float;
    case VK_FORMAT_R32_UINT:              return RhiFormat::R32Uint;
    case VK_FORMAT_R8G8_UNORM:            return RhiFormat::RG8Unorm;
    case VK_FORMAT_R16G16_SFLOAT:         return RhiFormat::RG16Float;
    case VK_FORMAT_R32G32_SFLOAT:         return RhiFormat::RG32Float;
    case VK_FORMAT_R8G8B8A8_UNORM:        return RhiFormat::RGBA8Unorm;
    case VK_FORMAT_R8G8B8A8_SRGB:         return RhiFormat::RGBA8Srgb;
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:         return RhiFormat::BGRA8Unorm;
    case VK_FORMAT_R16G16B16A16_SFLOAT:   return RhiFormat::RGBA16Float;
    case VK_FORMAT_R32G32B32A32_SFLOAT:   return RhiFormat::RGBA32Float;
    case VK_FORMAT_D32_SFLOAT:            return RhiFormat::D32Float;
    case VK_FORMAT_D16_UNORM:             return RhiFormat::D16Unorm;
    default:                              return RhiFormat::Undefined;
    }
}

VkImageUsageFlags toVkImageUsage(RhiTextureUsage usage) {
    VkImageUsageFlags flags = 0;
    if ((usage & RhiTextureUsage::RenderTarget) != RhiTextureUsage::None) {
        flags |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if ((usage & RhiTextureUsage::ShaderRead) != RhiTextureUsage::None) {
        flags |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if ((usage & RhiTextureUsage::ShaderWrite) != RhiTextureUsage::None) {
        flags |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    // Always allow transfer for mipmap generation and staging uploads
    flags |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return flags;
}

// --- VulkanFrameGraphBackend ---

VulkanFrameGraphBackend::VulkanFrameGraphBackend(VkDevice device, VkPhysicalDevice physicalDevice, VmaAllocator allocator)
    : m_device(device), m_physicalDevice(physicalDevice), m_allocator(allocator) {}

std::unique_ptr<RhiTexture> VulkanFrameGraphBackend::createTexture(const RhiTextureDesc& desc) {
    VkFormat vkFormat = toVkFormat(desc.format);
    bool depth = isDepthFormat(desc.format);

    VkImageUsageFlags usage = toVkImageUsage(desc.usage);
    if (depth) {
        usage &= ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    }

    VmaImageCreateInfo vmaInfo{};
    vmaInfo.device = m_device;
    vmaInfo.allocator = m_allocator;
    vmaInfo.depth = depth;
    vmaInfo.imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    vmaInfo.imageInfo.imageType = VK_IMAGE_TYPE_2D;
    vmaInfo.imageInfo.format = vkFormat;
    vmaInfo.imageInfo.extent = {desc.width, desc.height, 1};
    vmaInfo.imageInfo.mipLevels = 1;
    vmaInfo.imageInfo.arrayLayers = 1;
    vmaInfo.imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    vmaInfo.imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    vmaInfo.imageInfo.usage = usage;
    vmaInfo.imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vmaInfo.imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    const char* errorMsg = nullptr;
    auto resource = vmaCreateImageResource(vmaInfo, &errorMsg);
    if (!resource) {
        checkVk(VK_ERROR_INITIALIZATION_FAILED, errorMsg ? errorMsg : "Failed to create frame graph texture");
    }

    resource->usage = desc.usage;
    return std::make_unique<VulkanOwnedTexture>(*resource);
}

std::unique_ptr<RhiBuffer> VulkanFrameGraphBackend::createBuffer(const RhiBufferDesc& desc) {
    const VulkanResourceContextInfo& resourceContext = vulkanGetResourceContext();
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    usage = vulkanEnableBufferDeviceAddress(usage, resourceContext.bufferDeviceAddressEnabled);

    VmaBufferCreateInfo vmaInfo{};
    vmaInfo.device = m_device;
    vmaInfo.allocator = m_allocator;
    vmaInfo.size = desc.size;
    vmaInfo.usage = usage;
    vmaInfo.hostVisible = desc.hostVisible;
    vmaInfo.externalMemoryHandleTypes =
        vulkanHostVisibleExternalMemoryHandleTypes(vmaInfo.hostVisible,
                                                   resourceContext.externalHostMemoryEnabled);
    vmaInfo.debugName = desc.debugName;

    const char* errorMsg = nullptr;
    auto resource = vmaCreateBufferResource(vmaInfo, &errorMsg);
    if (!resource) {
        checkVk(VK_ERROR_INITIALIZATION_FAILED, errorMsg ? errorMsg : "Failed to create frame graph buffer");
    }

    if (desc.initialData && resource->mappedData) {
        std::memcpy(resource->mappedData, desc.initialData, desc.size);
    }

    return std::make_unique<VulkanOwnedBuffer>(*resource);
}

// --- VulkanCommandBuffer ---

VulkanCommandBuffer::VulkanCommandBuffer(VkCommandBuffer commandBuffer, VkDevice device,
                                         VulkanDescriptorManager* descriptorManager,
                                         VulkanResourceStateTracker* stateTracker)
    : m_commandBuffer(commandBuffer), m_device(device),
      m_descriptorManager(descriptorManager), m_stateTracker(stateTracker) {}

void VulkanCommandBuffer::transitionTexture(const RhiTexture* texture, VkImageLayout layout) {
    if (!texture || !m_stateTracker) {
        return;
    }

    auto* resource = getVulkanTextureResource(texture);
    if (!resource || resource->image == VK_NULL_HANDLE) {
        return;
    }

    m_stateTracker->requireImageState(resource->image, layout, imageAspectMask(resource));
    m_stateTracker->flushBarriers(m_commandBuffer);
}

void VulkanCommandBuffer::prepareTextureForSampling(const RhiTexture* texture) {
    if (!texture || !m_stateTracker) return;
    auto* resource = getVulkanTextureResource(texture);
    if (!resource || resource->image == VK_NULL_HANDLE) return;
    // Accumulate without flushing — FrameGraph calls flushBarriers() after all reads
    m_stateTracker->requireImageState(resource->image,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      imageAspectMask(resource));
}

void VulkanCommandBuffer::flushBarriers() {
    if (m_stateTracker) {
        m_stateTracker->flushBarriers(m_commandBuffer);
    }
}

std::unique_ptr<RhiRenderCommandEncoder> VulkanCommandBuffer::beginRenderPass(const RhiRenderPassDesc& desc) {
    // Accumulate layout transitions for all attachments, then batch-flush
    if (m_stateTracker) {
        for (uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
            if (desc.colorAttachments[i].texture) {
                VkImage image = getVulkanImage(desc.colorAttachments[i].texture);
                m_stateTracker->requireImageState(image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            }
        }
        if (desc.depthAttachment.bound && desc.depthAttachment.texture) {
            VkImage image = getVulkanImage(desc.depthAttachment.texture);
            m_stateTracker->requireImageState(image, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                              VK_IMAGE_ASPECT_DEPTH_BIT);
        }
        m_stateTracker->flushBarriers(m_commandBuffer);
    }

    std::array<VkRenderingAttachmentInfo, 8> colorAttachments{};
    for (uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
        const auto& ca = desc.colorAttachments[i];
        auto& vkCA = colorAttachments[i];
        vkCA.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

        // Get the VkImageView from any RhiTexture type
        vkCA.imageView = getVulkanImageView(ca.texture);

        vkCA.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        vkCA.loadOp = (ca.loadAction == RhiLoadAction::Clear) ? VK_ATTACHMENT_LOAD_OP_CLEAR :
                       (ca.loadAction == RhiLoadAction::Load) ? VK_ATTACHMENT_LOAD_OP_LOAD :
                       VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        vkCA.storeOp = (ca.storeAction == RhiStoreAction::Store) ? VK_ATTACHMENT_STORE_OP_STORE :
                        VK_ATTACHMENT_STORE_OP_DONT_CARE;
        vkCA.clearValue.color = {{
            static_cast<float>(ca.clearColor.red),
            static_cast<float>(ca.clearColor.green),
            static_cast<float>(ca.clearColor.blue),
            static_cast<float>(ca.clearColor.alpha)
        }};
    }

    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea.offset = {0, 0};
    // Determine render area from first color attachment
    if (desc.colorAttachmentCount > 0 && desc.colorAttachments[0].texture) {
        renderingInfo.renderArea.extent.width = desc.colorAttachments[0].texture->width();
        renderingInfo.renderArea.extent.height = desc.colorAttachments[0].texture->height();
    } else if (desc.depthAttachment.bound && desc.depthAttachment.texture) {
        renderingInfo.renderArea.extent.width = desc.depthAttachment.texture->width();
        renderingInfo.renderArea.extent.height = desc.depthAttachment.texture->height();
    }
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = desc.colorAttachmentCount;
    renderingInfo.pColorAttachments = colorAttachments.data();

    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    if (desc.depthAttachment.bound && desc.depthAttachment.texture) {
        depthAttachment.imageView = getVulkanImageView(desc.depthAttachment.texture);
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = (desc.depthAttachment.loadAction == RhiLoadAction::Clear) ? VK_ATTACHMENT_LOAD_OP_CLEAR :
                                  (desc.depthAttachment.loadAction == RhiLoadAction::Load) ? VK_ATTACHMENT_LOAD_OP_LOAD :
                                  VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.storeOp = (desc.depthAttachment.storeAction == RhiStoreAction::Store) ? VK_ATTACHMENT_STORE_OP_STORE :
                                   VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue.depthStencil = {static_cast<float>(desc.depthAttachment.clearDepth), 0};
        renderingInfo.pDepthAttachment = &depthAttachment;
    }

    vkCmdBeginRendering(m_commandBuffer, &renderingInfo);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = static_cast<float>(renderingInfo.renderArea.extent.height);
    viewport.width = static_cast<float>(renderingInfo.renderArea.extent.width);
    viewport.height = -static_cast<float>(renderingInfo.renderArea.extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(m_commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = renderingInfo.renderArea.offset;
    scissor.extent = renderingInfo.renderArea.extent;
    vkCmdSetScissor(m_commandBuffer, 0, 1, &scissor);
    vkCmdSetCullMode(m_commandBuffer, VK_CULL_MODE_BACK_BIT);
    vkCmdSetFrontFace(m_commandBuffer, VK_FRONT_FACE_COUNTER_CLOCKWISE);

    return std::make_unique<VulkanRenderCommandEncoder>(m_commandBuffer, m_device,
                                                        m_descriptorManager, m_stateTracker);
}

std::unique_ptr<RhiComputeCommandEncoder> VulkanCommandBuffer::beginComputePass(const RhiComputePassDesc& /*desc*/) {
    return std::make_unique<VulkanComputeCommandEncoder>(m_commandBuffer, m_device,
                                                         m_descriptorManager, m_stateTracker);
}

std::unique_ptr<RhiBlitCommandEncoder> VulkanCommandBuffer::beginBlitPass(const RhiBlitPassDesc& /*desc*/) {
    return std::make_unique<VulkanBlitCommandEncoder>(m_commandBuffer, m_device, m_stateTracker);
}

#endif // _WIN32
