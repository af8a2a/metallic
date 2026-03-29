#pragma once

#ifdef _WIN32

#include <vulkan/vulkan.h>

#include "../bindless_scene_constants.h"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

class RhiAccelerationStructure;
class RhiBuffer;
class RhiSampler;
class RhiTexture;

struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

struct VulkanPipelineResource;
struct VulkanTextureResource;

constexpr uint32_t kMaxBufferBindings = 32;
constexpr uint32_t kMaxTextureBindings = 128;
constexpr uint32_t kMaxSamplerBindings = 16;
constexpr uint32_t kMaxAccelerationStructureBindings = 8;

constexpr uint32_t kVulkanBindlessSetIndex = METALLIC_BINDLESS_SET;
constexpr uint32_t kVulkanBindlessSampledImageBinding = METALLIC_BINDLESS_SAMPLED_IMAGE_BINDING;
constexpr uint32_t kVulkanBindlessSamplerBinding = METALLIC_BINDLESS_SAMPLER_BINDING;
constexpr uint32_t kVulkanBindlessStorageImageBinding = METALLIC_BINDLESS_STORAGE_IMAGE_BINDING;
constexpr uint32_t kVulkanBindlessStorageBufferBinding = METALLIC_BINDLESS_STORAGE_BUFFER_BINDING;
constexpr uint32_t kVulkanBindlessAccelerationStructureBinding =
    METALLIC_BINDLESS_ACCELERATION_STRUCTURE_BINDING;

constexpr uint32_t kVulkanBindlessMaxSampledImages = METALLIC_BINDLESS_MAX_SAMPLED_IMAGES;
constexpr uint32_t kVulkanBindlessMaxSamplers = METALLIC_BINDLESS_MAX_SAMPLERS;
constexpr uint32_t kVulkanBindlessMaxStorageImages = METALLIC_BINDLESS_MAX_STORAGE_IMAGES;
constexpr uint32_t kVulkanBindlessMaxStorageBuffers = METALLIC_BINDLESS_MAX_STORAGE_BUFFERS;
constexpr uint32_t kVulkanBindlessMaxAccelerationStructures =
    METALLIC_BINDLESS_MAX_ACCELERATION_STRUCTURES;

struct PendingBufferBinding {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = VK_WHOLE_SIZE;
    bool dirty = false;
};

struct PendingTextureBinding {
    VulkanTextureResource* resource = nullptr;
    VkImageView imageView = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bool dirty = false;
    bool isStorage = false;
};

struct PendingSamplerBinding {
    VkSampler sampler = VK_NULL_HANDLE;
    bool dirty = false;
};

struct PendingAccelerationStructureBinding {
    VkAccelerationStructureKHR accelerationStructure = VK_NULL_HANDLE;
    bool dirty = false;
};

class VulkanDescriptorManager {
public:
    void init(VkDevice device,
              VkPhysicalDevice physicalDevice,
              VmaAllocator allocator,
              VkDeviceSize minUniformBufferOffsetAlignment,
              VkDeviceSize nonCoherentAtomSize,
              VkDeviceSize maxUniformBufferRange);
    void destroy();
    void resetFrame();

    PendingBufferBinding uploadInlineUniformData(const void* data, size_t size);
    void updateBindlessSampledTextures(const RhiTexture* const* textures,
                                       uint32_t startIndex,
                                       uint32_t count);
    bool updateBindlessSampledTexture(uint32_t index, const RhiTexture* texture);
    bool updateBindlessSampler(uint32_t index, const RhiSampler* sampler);
    bool updateBindlessStorageImage(uint32_t index, const RhiTexture* texture);
    bool updateBindlessStorageBuffer(uint32_t index,
                                     const RhiBuffer* buffer,
                                     VkDeviceSize offset = 0,
                                     VkDeviceSize range = VK_WHOLE_SIZE);
    bool updateBindlessAccelerationStructure(uint32_t index,
                                             const RhiAccelerationStructure* accelerationStructure);

    void flushAndBind(VkCommandBuffer cmd,
                      VkPipelineBindPoint bindPoint,
                      const VulkanPipelineResource& pipeline,
                      const std::array<PendingBufferBinding, kMaxBufferBindings>& buffers,
                      const std::array<PendingTextureBinding, kMaxTextureBindings>& textures,
                      const std::array<PendingSamplerBinding, kMaxSamplerBindings>& samplers,
                      const std::array<PendingAccelerationStructureBinding,
                                       kMaxAccelerationStructureBindings>& accelerationStructures);

private:
    struct FrameUniformUpload {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = nullptr;
        void* mappedData = nullptr;
        VkDeviceSize capacity = 0;
        VkDeviceSize head = 0;
    };

    struct FrameState {
        VkDescriptorPool pool = VK_NULL_HANDLE;
        FrameUniformUpload uniformUpload;
    };

    void createPools();
    void destroyFrameState(FrameState& frame);
    bool ensureFrameUniformUploadCapacity(FrameState& frame, VkDeviceSize requiredSize);
    FrameState& currentFrame();

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = nullptr;
    VkDescriptorPool m_bindlessPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_bindlessSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_bindlessSet = VK_NULL_HANDLE;
    std::array<FrameState, 2> m_frames{};
    uint32_t m_frameIndex = 1;
    VkDeviceSize m_uniformUploadAlignment = 16;
    VkDeviceSize m_nonCoherentAtomSize = 1;
    VkDeviceSize m_maxUniformBufferRange = 65536;
};

bool vulkanRetainBindlessSetLayout(VkDevice device,
                                   VkDescriptorSetLayout& outLayout,
                                   std::string* errorMessage = nullptr);
void vulkanReleaseBindlessSetLayout(VkDevice device);
bool vulkanIsBindlessSetIndex(uint32_t setIndex);

#endif // _WIN32
