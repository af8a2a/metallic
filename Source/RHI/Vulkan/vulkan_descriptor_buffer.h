#pragma once

#ifdef _WIN32

#include "vulkan_descriptor_manager.h" // IVulkanDescriptorBackend, Pending*Binding, constants

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>

struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

class RhiAccelerationStructure;
class RhiBuffer;
class RhiSampler;
class RhiTexture;

// Descriptor-buffer-based implementation of the descriptor backend.
// Uses VK_EXT_descriptor_buffer to write descriptors directly into GPU-visible
// buffers, avoiding VkDescriptorPool / vkUpdateDescriptorSets entirely.
class VulkanDescriptorBufferManager final : public IVulkanDescriptorBackend {
public:
    void init(VkDevice device,
              VkPhysicalDevice physicalDevice,
              VmaAllocator allocator,
              const VkPhysicalDeviceDescriptorBufferPropertiesEXT& props,
              VkDeviceSize minUniformBufferOffsetAlignment,
              VkDeviceSize nonCoherentAtomSize,
              VkDeviceSize maxUniformBufferRange);
    void destroy();

    // --- IVulkanDescriptorBackend ---
    void resetFrame() override;
    PendingBufferBinding uploadInlineUniformData(const void* data, size_t size) override;

    void updateBindlessSampledTextures(const RhiTexture* const* textures,
                                       uint32_t startIndex,
                                       uint32_t count) override;
    bool updateBindlessSampledTexture(uint32_t index, const RhiTexture* texture) override;
    bool updateBindlessSampler(uint32_t index, const RhiSampler* sampler) override;
    bool updateBindlessStorageImage(uint32_t index, const RhiTexture* texture) override;
    bool updateBindlessStorageBuffer(uint32_t index,
                                     const RhiBuffer* buffer,
                                     VkDeviceSize offset = 0,
                                     VkDeviceSize range = VK_WHOLE_SIZE) override;
    bool updateBindlessAccelerationStructure(
        uint32_t index,
        const RhiAccelerationStructure* accelerationStructure) override;

    void flushAndBind(
        VkCommandBuffer cmd,
        VkPipelineBindPoint bindPoint,
        const VulkanPipelineResource& pipeline,
        const std::array<PendingBufferBinding, kMaxBufferBindings>& buffers,
        const std::array<PendingTextureBinding, kMaxTextureBindings>& textures,
        const std::array<PendingSamplerBinding, kMaxSamplerBindings>& samplers,
        const std::array<PendingAccelerationStructureBinding,
                         kMaxAccelerationStructureBindings>& accelerationStructures) override;

private:
    // --- Bindless descriptor buffer (persistent, updated in-place) ---
    VkBuffer m_bindlessBuffer = VK_NULL_HANDLE;
    VmaAllocation m_bindlessAllocation = nullptr;
    void* m_bindlessMapped = nullptr;
    VkDeviceAddress m_bindlessBufferAddress = 0;
    VkDeviceSize m_bindlessBufferSize = 0;

    // Offsets within bindless buffer for each binding type (queried from layout)
    VkDeviceSize m_bindlessSampledImageOffset = 0;
    VkDeviceSize m_bindlessSamplerOffset = 0;
    VkDeviceSize m_bindlessStorageImageOffset = 0;
    VkDeviceSize m_bindlessStorageBufferOffset = 0;
    VkDeviceSize m_bindlessAccelerationStructureOffset = 0;

    VkDescriptorSetLayout m_bindlessSetLayout = VK_NULL_HANDLE;
    VkDeviceSize m_bindlessSetLayoutSize = 0;

    // --- Per-frame transient descriptor buffer (for set 0 per-draw descriptors) ---
    struct FrameState {
        VkBuffer descriptorBuffer = VK_NULL_HANDLE;
        VmaAllocation descriptorAllocation = nullptr;
        void* descriptorMapped = nullptr;
        VkDeviceAddress descriptorBufferAddress = 0;
        VkDeviceSize descriptorHead = 0;
        VkDeviceSize descriptorCapacity = 0;

        // Inline uniform upload buffer (same pattern as VulkanDescriptorManager)
        VkBuffer uniformBuffer = VK_NULL_HANDLE;
        VmaAllocation uniformAllocation = nullptr;
        void* uniformMapped = nullptr;
        VkDeviceSize uniformHead = 0;
        VkDeviceSize uniformCapacity = 0;
    };
    std::array<FrameState, 2> m_frames{};
    uint32_t m_frameIndex = 1;

    // Descriptor sizes from VkPhysicalDeviceDescriptorBufferPropertiesEXT
    VkDeviceSize m_sampledImageDescSize = 0;
    VkDeviceSize m_samplerDescSize = 0;
    VkDeviceSize m_storageImageDescSize = 0;
    VkDeviceSize m_storageBufferDescSize = 0;
    VkDeviceSize m_uniformBufferDescSize = 0;
    VkDeviceSize m_accelerationStructureDescSize = 0;
    VkDeviceSize m_descriptorBufferOffsetAlignment = 0;

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;
    VkDeviceSize m_uniformUploadAlignment = 16;
    VkDeviceSize m_nonCoherentAtomSize = 1;
    VkDeviceSize m_maxUniformBufferRange = 65536;

    // Dynamically loaded function pointers
    PFN_vkGetDescriptorSetLayoutSizeEXT m_vkGetDescriptorSetLayoutSizeEXT = nullptr;
    PFN_vkGetDescriptorSetLayoutBindingOffsetEXT m_vkGetDescriptorSetLayoutBindingOffsetEXT = nullptr;
    PFN_vkGetDescriptorEXT m_vkGetDescriptorEXT = nullptr;
    PFN_vkCmdBindDescriptorBuffersEXT m_vkCmdBindDescriptorBuffersEXT = nullptr;
    PFN_vkCmdSetDescriptorBufferOffsetsEXT m_vkCmdSetDescriptorBufferOffsetsEXT = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR m_vkGetAccelerationStructureDeviceAddressKHR = nullptr;

    // Helpers
    FrameState& currentFrame();
    void destroyFrameState(FrameState& frame);
    bool ensureFrameUniformUploadCapacity(FrameState& frame, VkDeviceSize requiredSize);
    VkDeviceSize allocateTransientDescriptorSpace(VkDeviceSize size);
    void writeImageDescriptor(void* dst, VkImageView imageView, VkImageLayout layout,
                              VkDescriptorType type, VkDeviceSize descSize);
    void writeSamplerDescriptor(void* dst, VkSampler sampler);
    void writeBufferDescriptor(void* dst, VkDeviceAddress address, VkDeviceSize range,
                               VkDescriptorType type, VkDeviceSize descSize);
    void writeAccelerationStructureDescriptor(void* dst, VkDeviceAddress address);
};

#endif // _WIN32
