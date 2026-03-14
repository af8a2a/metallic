#include "vulkan_frame_graph.h"

#ifdef _WIN32

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <array>
#include <cstring>
#include <stdexcept>

namespace {

// Helper to check Vulkan results
void checkVk(VkResult result, const char* message) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(message) + " (VkResult: " + std::to_string(result) + ")");
    }
}

} // namespace

// Owned texture with VMA allocation — visible outside anonymous namespace for dynamic_cast in beginRenderPass
class VulkanOwnedTexture final : public RhiTexture {
public:
    VulkanOwnedTexture(VkDevice device, VkImage image, VkImageView imageView,
                       VmaAllocation allocation, VmaAllocator allocator,
                       uint32_t w, uint32_t h)
        : m_device(device), m_image(image), m_imageView(imageView),
          m_allocation(allocation), m_allocator(allocator),
          m_width(w), m_height(h) {}

    ~VulkanOwnedTexture() override {
        if (m_imageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_imageView, nullptr);
        }
        if (m_image != VK_NULL_HANDLE && m_allocator != nullptr) {
            vmaDestroyImage(m_allocator, m_image, m_allocation);
        }
    }

    void* nativeHandle() const override { return reinterpret_cast<void*>(m_image); }
    uint32_t width() const override { return m_width; }
    uint32_t height() const override { return m_height; }

    VkImage image() const { return m_image; }
    VkImageView imageView() const { return m_imageView; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkImage m_image = VK_NULL_HANDLE;
    VkImageView m_imageView = VK_NULL_HANDLE;
    VmaAllocation m_allocation = nullptr;
    VmaAllocator m_allocator = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
};

namespace {

// Owned buffer with VMA allocation
class VulkanOwnedBuffer final : public RhiBuffer {
public:
    VulkanOwnedBuffer(VkBuffer buffer, VmaAllocation allocation, VmaAllocator allocator, size_t byteSize)
        : m_buffer(buffer), m_allocation(allocation), m_allocator(allocator), m_size(byteSize) {}

    ~VulkanOwnedBuffer() override {
        if (m_buffer != VK_NULL_HANDLE && m_allocator != nullptr) {
            vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        }
    }

    size_t size() const override { return m_size; }
    void* nativeHandle() const override { return reinterpret_cast<void*>(m_buffer); }
    void* mappedData() override {
        if (m_mappedData == nullptr && m_allocation != nullptr) {
            vmaMapMemory(m_allocator, m_allocation, &m_mappedData);
        }
        return m_mappedData;
    }

    VkBuffer buffer() const { return m_buffer; }

private:
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = nullptr;
    VmaAllocator m_allocator = nullptr;
    size_t m_size = 0;
    void* m_mappedData = nullptr;
};

// Render command encoder
class VulkanRenderCommandEncoder final : public RhiRenderCommandEncoder {
public:
    VulkanRenderCommandEncoder(VkCommandBuffer commandBuffer, VkDevice device)
        : m_commandBuffer(commandBuffer), m_device(device) {}

    ~VulkanRenderCommandEncoder() override {
        if (m_commandBuffer != VK_NULL_HANDLE) {
            vkCmdEndRendering(m_commandBuffer);
        }
    }

    void* nativeHandle() const override { return m_commandBuffer; }

    void setDepthStencilState(const RhiDepthStencilState* /*state*/) override {
        // Vulkan handles depth/stencil state at pipeline creation time, not dynamically
    }

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
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          reinterpret_cast<VkPipeline>(pipeline.nativeHandle()));
    }

    void setVertexBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) override {
        if (!buffer) return;
        VkBuffer vkBuf = reinterpret_cast<VkBuffer>(buffer->nativeHandle());
        VkDeviceSize vkOffset = offset;
        vkCmdBindVertexBuffers(m_commandBuffer, index, 1, &vkBuf, &vkOffset);
    }

    void setFragmentBuffer(const RhiBuffer* /*buffer*/, uint64_t /*offset*/, uint32_t /*index*/) override {
        // Handled via descriptor sets in Vulkan - will be implemented with descriptor management
    }

    void setMeshBuffer(const RhiBuffer* /*buffer*/, uint64_t /*offset*/, uint32_t /*index*/) override {
        // Handled via descriptor sets in Vulkan
    }

    void setVertexBytes(const void* data, size_t size, uint32_t index) override {
        if (data && size > 0 && m_currentPipelineLayout != VK_NULL_HANDLE) {
            vkCmdPushConstants(m_commandBuffer, m_currentPipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, static_cast<uint32_t>(size), data);
        }
    }

    void setFragmentBytes(const void* data, size_t size, uint32_t /*index*/) override {
        if (data && size > 0 && m_currentPipelineLayout != VK_NULL_HANDLE) {
            vkCmdPushConstants(m_commandBuffer, m_currentPipelineLayout,
                               VK_SHADER_STAGE_FRAGMENT_BIT, 128, static_cast<uint32_t>(size), data);
        }
    }

    void setMeshBytes(const void* data, size_t size, uint32_t /*index*/) override {
        if (data && size > 0 && m_currentPipelineLayout != VK_NULL_HANDLE) {
            vkCmdPushConstants(m_commandBuffer, m_currentPipelineLayout,
                               VK_SHADER_STAGE_MESH_BIT_EXT, 0, static_cast<uint32_t>(size), data);
        }
    }

    void setFragmentTexture(const RhiTexture* /*texture*/, uint32_t /*index*/) override {
        // Handled via descriptor sets
    }

    void setFragmentTextures(const RhiTexture* const* /*textures*/, uint32_t /*startIndex*/, uint32_t /*count*/) override {
        // Handled via descriptor sets
    }

    void setMeshTextures(const RhiTexture* const* /*textures*/, uint32_t /*startIndex*/, uint32_t /*count*/) override {
        // Handled via descriptor sets
    }

    void setFragmentSampler(const RhiSampler* /*sampler*/, uint32_t /*index*/) override {
        // Handled via descriptor sets
    }

    void setMeshSampler(const RhiSampler* /*sampler*/, uint32_t /*index*/) override {
        // Handled via descriptor sets
    }

    void drawPrimitives(RhiPrimitiveType /*primitiveType*/, uint32_t vertexStart, uint32_t vertexCount) override {
        vkCmdDraw(m_commandBuffer, vertexCount, 1, vertexStart, 0);
    }

    void drawIndexedPrimitives(RhiPrimitiveType /*primitiveType*/,
                               uint32_t indexCount,
                               RhiIndexType indexType,
                               const RhiBuffer& indexBuffer,
                               uint64_t indexBufferOffset) override {
        VkIndexType vkIndexType = (indexType == RhiIndexType::UInt16) ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        vkCmdBindIndexBuffer(m_commandBuffer, reinterpret_cast<VkBuffer>(indexBuffer.nativeHandle()),
                             indexBufferOffset, vkIndexType);
        vkCmdDrawIndexed(m_commandBuffer, indexCount, 1, 0, 0, 0);
    }

    void drawMeshThreadgroups(RhiSize3D threadgroupsPerGrid,
                              RhiSize3D /*threadsPerObjectThreadgroup*/,
                              RhiSize3D /*threadsPerMeshThreadgroup*/) override {
        vkCmdDrawMeshTasksEXT(m_commandBuffer,
                              threadgroupsPerGrid.width,
                              threadgroupsPerGrid.height,
                              threadgroupsPerGrid.depth);
    }

    void drawMeshThreadgroupsIndirect(const RhiBuffer& indirectBuffer,
                                      uint64_t indirectBufferOffset,
                                      RhiSize3D /*threadsPerObjectThreadgroup*/,
                                      RhiSize3D /*threadsPerMeshThreadgroup*/) override {
        vkCmdDrawMeshTasksIndirectEXT(m_commandBuffer,
                                      reinterpret_cast<VkBuffer>(indirectBuffer.nativeHandle()),
                                      indirectBufferOffset, 1, 0);
    }

    void renderImGuiDrawData(const RhiNativeCommandBuffer& /*commandBuffer*/) override {
        // ImGui rendering is handled directly via ImGui_ImplVulkan_RenderDrawData
        // with the native VkCommandBuffer in main_vulkan.cpp
    }

    void setPipelineLayout(VkPipelineLayout layout) { m_currentPipelineLayout = layout; }

private:
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkPipelineLayout m_currentPipelineLayout = VK_NULL_HANDLE;
};

// Compute command encoder
class VulkanComputeCommandEncoder final : public RhiComputeCommandEncoder {
public:
    VulkanComputeCommandEncoder(VkCommandBuffer commandBuffer, VkDevice device)
        : m_commandBuffer(commandBuffer), m_device(device) {}

    ~VulkanComputeCommandEncoder() override = default;

    void* nativeHandle() const override { return m_commandBuffer; }

    void setComputePipeline(const RhiComputePipeline& pipeline) override {
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          reinterpret_cast<VkPipeline>(pipeline.nativeHandle()));
    }

    void setBuffer(const RhiBuffer* /*buffer*/, uint64_t /*offset*/, uint32_t /*index*/) override {
        // Handled via descriptor sets
    }

    void setBytes(const void* data, size_t size, uint32_t /*index*/) override {
        if (data && size > 0 && m_currentPipelineLayout != VK_NULL_HANDLE) {
            vkCmdPushConstants(m_commandBuffer, m_currentPipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(size), data);
        }
    }

    void setTexture(const RhiTexture* /*texture*/, uint32_t /*index*/) override {
        // Handled via descriptor sets
    }

    void setTextures(const RhiTexture* const* /*textures*/, uint32_t /*startIndex*/, uint32_t /*count*/) override {
        // Handled via descriptor sets
    }

    void setSampler(const RhiSampler* /*sampler*/, uint32_t /*index*/) override {
        // Handled via descriptor sets
    }

    void setAccelerationStructure(const RhiAccelerationStructure* /*accelerationStructure*/, uint32_t /*index*/) override {
        // Handled via descriptor sets with VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
    }

    void useResource(const RhiBuffer& /*resource*/, RhiResourceUsage /*usage*/) override {
        // Vulkan handles resource tracking via pipeline barriers, not explicit useResource calls
    }

    void useResource(const RhiAccelerationStructure& /*resource*/, RhiResourceUsage /*usage*/) override {
        // Same as above
    }

    void memoryBarrier(RhiBarrierScope scope) override {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;

        VkDependencyInfo depInfo{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(m_commandBuffer, &depInfo);
    }

    void dispatchThreadgroups(RhiSize3D threadgroupsPerGrid, RhiSize3D /*threadsPerThreadgroup*/) override {
        vkCmdDispatch(m_commandBuffer,
                      threadgroupsPerGrid.width,
                      threadgroupsPerGrid.height,
                      threadgroupsPerGrid.depth);
    }

    void setPipelineLayout(VkPipelineLayout layout) { m_currentPipelineLayout = layout; }

private:
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkPipelineLayout m_currentPipelineLayout = VK_NULL_HANDLE;
};

// Blit command encoder
class VulkanBlitCommandEncoder final : public RhiBlitCommandEncoder {
public:
    VulkanBlitCommandEncoder(VkCommandBuffer commandBuffer, VkDevice device)
        : m_commandBuffer(commandBuffer), m_device(device) {}

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
                       reinterpret_cast<VkImage>(source.nativeHandle()),
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       reinterpret_cast<VkImage>(destination.nativeHandle()),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region);
    }

private:
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
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
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:         return RhiFormat::RGBA8Unorm;
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

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = vkFormat;
    imageInfo.extent = {desc.width, desc.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    checkVk(vmaCreateImage(m_allocator, &imageInfo, &allocCreateInfo, &image, &allocation, nullptr),
            "Failed to create VMA image");

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = vkFormat;
    viewInfo.subresourceRange.aspectMask = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView = VK_NULL_HANDLE;
    VkResult viewResult = vkCreateImageView(m_device, &viewInfo, nullptr, &imageView);
    if (viewResult != VK_SUCCESS) {
        vmaDestroyImage(m_allocator, image, allocation);
        checkVk(viewResult, "Failed to create image view for frame graph texture");
    }

    return std::make_unique<VulkanOwnedTexture>(m_device, image, imageView, allocation, m_allocator, desc.width, desc.height);
}

std::unique_ptr<RhiBuffer> VulkanFrameGraphBackend::createBuffer(const RhiBufferDesc& desc) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = desc.size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    if (desc.hostVisible) {
        allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocInfo{};
    checkVk(vmaCreateBuffer(m_allocator, &bufferInfo, &allocCreateInfo, &buffer, &allocation, &allocInfo),
            "Failed to create VMA buffer");

    if (desc.initialData && allocInfo.pMappedData) {
        std::memcpy(allocInfo.pMappedData, desc.initialData, desc.size);
    }

    if (desc.debugName) {
        vmaSetAllocationName(m_allocator, allocation, desc.debugName);
    }

    return std::make_unique<VulkanOwnedBuffer>(buffer, allocation, m_allocator, desc.size);
}

// --- VulkanCommandBuffer ---

VulkanCommandBuffer::VulkanCommandBuffer(VkCommandBuffer commandBuffer, VkDevice device)
    : m_commandBuffer(commandBuffer), m_device(device) {}

std::unique_ptr<RhiRenderCommandEncoder> VulkanCommandBuffer::beginRenderPass(const RhiRenderPassDesc& desc) {
    std::array<VkRenderingAttachmentInfo, 8> colorAttachments{};
    for (uint32_t i = 0; i < desc.colorAttachmentCount; ++i) {
        const auto& ca = desc.colorAttachments[i];
        auto& vkCA = colorAttachments[i];
        vkCA.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

        // Get the VkImageView - for VulkanOwnedTexture we have it, for imported textures we need the view
        auto* ownedTex = dynamic_cast<const VulkanOwnedTexture*>(ca.texture);
        if (ownedTex) {
            vkCA.imageView = ownedTex->imageView();
        } else {
            // For imported textures (swapchain), the image view must be provided externally
            // This will be handled by the window runtime setting up the imported texture properly
            vkCA.imageView = VK_NULL_HANDLE;
        }

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
    }
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = desc.colorAttachmentCount;
    renderingInfo.pColorAttachments = colorAttachments.data();

    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    if (desc.depthAttachment.bound && desc.depthAttachment.texture) {
        auto* ownedDepth = dynamic_cast<const VulkanOwnedTexture*>(desc.depthAttachment.texture);
        if (ownedDepth) {
            depthAttachment.imageView = ownedDepth->imageView();
        }
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
    return std::make_unique<VulkanRenderCommandEncoder>(m_commandBuffer, m_device);
}

std::unique_ptr<RhiComputeCommandEncoder> VulkanCommandBuffer::beginComputePass(const RhiComputePassDesc& /*desc*/) {
    return std::make_unique<VulkanComputeCommandEncoder>(m_commandBuffer, m_device);
}

std::unique_ptr<RhiBlitCommandEncoder> VulkanCommandBuffer::beginBlitPass(const RhiBlitPassDesc& /*desc*/) {
    return std::make_unique<VulkanBlitCommandEncoder>(m_commandBuffer, m_device);
}

#endif // _WIN32
