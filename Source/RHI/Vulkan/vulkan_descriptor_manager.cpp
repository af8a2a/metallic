#include "vulkan_descriptor_manager.h"

#ifdef _WIN32

#include "../rhi_resource_utils.h"
#include "vulkan_resource_handles.h"

#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unordered_map>

namespace {

void checkVk(VkResult result, const char* message) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(message) + " (VkResult: " + std::to_string(result) + ")");
    }
}

constexpr uint32_t kMaxSetsPerFrame = 4096;
constexpr VkDeviceSize kDefaultInlineUniformUploadSize = 1u << 20;

struct SharedBindlessLayoutState {
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    uint32_t refCount = 0;
};

std::mutex g_bindlessLayoutMutex;
std::unordered_map<uint64_t, SharedBindlessLayoutState> g_bindlessLayouts;

VkDescriptorPool createDescriptorPool(VkDevice device) {
    std::array<VkDescriptorPoolSize, 6> poolSizes = {{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxSetsPerFrame * kMaxBufferBindings},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSetsPerFrame * kMaxBufferBindings},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxSetsPerFrame * kMaxTextureBindings},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxSetsPerFrame * kMaxTextureBindings},
        {VK_DESCRIPTOR_TYPE_SAMPLER, kMaxSetsPerFrame * kMaxSamplerBindings},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
         kMaxSetsPerFrame * kMaxAccelerationStructureBindings},
    }};

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = kMaxSetsPerFrame * 4;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VkDescriptorPool pool = VK_NULL_HANDLE;
    checkVk(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool),
            "Failed to create frame descriptor pool");
    return pool;
}

VkDescriptorPool createBindlessDescriptorPool(VkDevice device) {
    std::array<VkDescriptorPoolSize, 5> poolSizes = {{
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kVulkanBindlessMaxSampledImages},
        {VK_DESCRIPTOR_TYPE_SAMPLER, kVulkanBindlessMaxSamplers},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kVulkanBindlessMaxStorageImages},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kVulkanBindlessMaxStorageBuffers},
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, kVulkanBindlessMaxAccelerationStructures},
    }};

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VkDescriptorPool pool = VK_NULL_HANDLE;
    checkVk(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool),
            "Failed to create bindless descriptor pool");
    return pool;
}

VkDescriptorSetLayout createBindlessSetLayout(VkDevice device) {
    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
    bindings[0].binding = kVulkanBindlessSampledImageBinding;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = kVulkanBindlessMaxSampledImages;
    bindings[0].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[1].binding = kVulkanBindlessSamplerBinding;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = kVulkanBindlessMaxSamplers;
    bindings[1].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[2].binding = kVulkanBindlessStorageImageBinding;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = kVulkanBindlessMaxStorageImages;
    bindings[2].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[3].binding = kVulkanBindlessStorageBufferBinding;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = kVulkanBindlessMaxStorageBuffers;
    bindings[3].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[4].binding = kVulkanBindlessAccelerationStructureBinding;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[4].descriptorCount = kVulkanBindlessMaxAccelerationStructures;
    bindings[4].stageFlags = VK_SHADER_STAGE_ALL;

    std::array<VkDescriptorBindingFlags, 5> bindingFlags{};
    bindingFlags.fill(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    flagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    flagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.pNext = &flagsInfo;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    checkVk(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout),
            "Failed to create bindless descriptor set layout");
    return layout;
}

uint64_t bindlessLayoutKey(VkDevice device) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(device));
}

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0) {
        return value;
    }
    const VkDeviceSize mask = alignment - 1;
    return (value + mask) & ~mask;
}

VkDeviceSize alignDown(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0) {
        return value;
    }
    return value & ~(alignment - 1);
}

VkDeviceSize nextUploadCapacity(VkDeviceSize requiredSize) {
    VkDeviceSize capacity = std::max(requiredSize, kDefaultInlineUniformUploadSize);
    VkDeviceSize rounded = kDefaultInlineUniformUploadSize;
    while (rounded < capacity) {
        rounded <<= 1;
    }
    return rounded;
}

} // namespace

void VulkanDescriptorManager::init(VkDevice device,
                                   VkPhysicalDevice physicalDevice,
                                   VmaAllocator allocator,
                                   VkDeviceSize minUniformBufferOffsetAlignment,
                                   VkDeviceSize nonCoherentAtomSize,
                                   VkDeviceSize maxUniformBufferRange) {
    m_device = device;
    m_physicalDevice = physicalDevice;
    m_allocator = allocator;
    m_uniformUploadAlignment = std::max<VkDeviceSize>(minUniformBufferOffsetAlignment, 16);
    m_nonCoherentAtomSize = std::max<VkDeviceSize>(nonCoherentAtomSize, 1);
    m_maxUniformBufferRange = maxUniformBufferRange;
    createPools();

    std::string errorMessage;
    if (!vulkanRetainBindlessSetLayout(m_device, m_bindlessSetLayout, &errorMessage)) {
        throw std::runtime_error(errorMessage.empty()
                                     ? "Failed to acquire bindless descriptor set layout"
                                     : errorMessage);
    }

    m_bindlessPool = createBindlessDescriptorPool(m_device);

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = m_bindlessPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_bindlessSetLayout;
    checkVk(vkAllocateDescriptorSets(m_device, &allocInfo, &m_bindlessSet),
            "Failed to allocate bindless descriptor set");
}

void VulkanDescriptorManager::destroy() {
    for (auto& frame : m_frames) {
        destroyFrameState(frame);
    }

    if (m_bindlessPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_bindlessPool, nullptr);
        m_bindlessPool = VK_NULL_HANDLE;
    }
    m_bindlessSet = VK_NULL_HANDLE;
    if (m_bindlessSetLayout != VK_NULL_HANDLE) {
        vulkanReleaseBindlessSetLayout(m_device);
        m_bindlessSetLayout = VK_NULL_HANDLE;
    }
    m_allocator = nullptr;
    m_physicalDevice = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}

void VulkanDescriptorManager::resetFrame() {
    m_frameIndex = (m_frameIndex + 1) % static_cast<uint32_t>(m_frames.size());

    FrameState& frame = currentFrame();
    if (frame.pool != VK_NULL_HANDLE) {
        vkResetDescriptorPool(m_device, frame.pool, 0);
    }
    frame.uniformUpload.head = 0;
}

PendingBufferBinding VulkanDescriptorManager::uploadInlineUniformData(const void* data, size_t size) {
    PendingBufferBinding binding{};
    if (!data || size == 0 || m_device == VK_NULL_HANDLE || m_allocator == nullptr) {
        return binding;
    }

    if (size > m_maxUniformBufferRange) {
        spdlog::warn("Inline uniform upload of {} bytes exceeds maxUniformBufferRange {}; skipping bind",
                     size,
                     m_maxUniformBufferRange);
        return binding;
    }

    FrameState& frame = currentFrame();
    const VkDeviceSize alignedOffset = alignUp(frame.uniformUpload.head, m_uniformUploadAlignment);
    const VkDeviceSize requiredSize = alignedOffset + static_cast<VkDeviceSize>(size);
    if (!ensureFrameUniformUploadCapacity(frame, requiredSize) ||
        frame.uniformUpload.buffer == VK_NULL_HANDLE ||
        frame.uniformUpload.mappedData == nullptr) {
        return binding;
    }

    std::byte* dst = static_cast<std::byte*>(frame.uniformUpload.mappedData) + alignedOffset;
    std::memcpy(dst, data, size);

    const VkDeviceSize flushOffset = alignDown(alignedOffset, m_nonCoherentAtomSize);
    const VkDeviceSize flushSize =
        alignUp((alignedOffset - flushOffset) + static_cast<VkDeviceSize>(size), m_nonCoherentAtomSize);
    vmaFlushAllocation(m_allocator, frame.uniformUpload.allocation, flushOffset, flushSize);

    frame.uniformUpload.head = requiredSize;

    binding.buffer = frame.uniformUpload.buffer;
    binding.offset = alignedOffset;
    binding.range = size;
    binding.dirty = true;
    return binding;
}

void VulkanDescriptorManager::updateBindlessSampledTextures(const RhiTexture* const* textures,
                                                            uint32_t startIndex,
                                                            uint32_t count) {
    if (!textures || count == 0) {
        return;
    }

    for (uint32_t index = 0; index < count; ++index) {
        updateBindlessSampledTexture(startIndex + index, textures[index]);
    }
}

bool VulkanDescriptorManager::updateBindlessSampledTexture(uint32_t index, const RhiTexture* texture) {
    if (m_bindlessSet == VK_NULL_HANDLE || index >= kVulkanBindlessMaxSampledImages) {
        return false;
    }

    const VkImageView imageView = getVulkanImageView(texture);
    if (imageView == VK_NULL_HANDLE) {
        return false;
    }

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = m_bindlessSet;
    write.dstBinding = kVulkanBindlessSampledImageBinding;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    return true;
}

bool VulkanDescriptorManager::updateBindlessSampler(uint32_t index, const RhiSampler* sampler) {
    if (m_bindlessSet == VK_NULL_HANDLE || index >= kVulkanBindlessMaxSamplers) {
        return false;
    }

    const VkSampler vkSampler = getVulkanSamplerHandle(sampler);
    if (vkSampler == VK_NULL_HANDLE) {
        return false;
    }

    VkDescriptorImageInfo imageInfo{};
    imageInfo.sampler = vkSampler;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = m_bindlessSet;
    write.dstBinding = kVulkanBindlessSamplerBinding;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    return true;
}

bool VulkanDescriptorManager::updateBindlessStorageImage(uint32_t index, const RhiTexture* texture) {
    if (m_bindlessSet == VK_NULL_HANDLE || index >= kVulkanBindlessMaxStorageImages) {
        return false;
    }

    const VkImageView imageView = getVulkanImageView(texture);
    if (imageView == VK_NULL_HANDLE) {
        return false;
    }

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = m_bindlessSet;
    write.dstBinding = kVulkanBindlessStorageImageBinding;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    return true;
}

bool VulkanDescriptorManager::updateBindlessStorageBuffer(uint32_t index,
                                                          const RhiBuffer* buffer,
                                                          VkDeviceSize offset,
                                                          VkDeviceSize range) {
    if (m_bindlessSet == VK_NULL_HANDLE || index >= kVulkanBindlessMaxStorageBuffers) {
        return false;
    }

    const VulkanBufferResource* resource = getVulkanBufferResource(buffer);
    if (!resource || resource->buffer == VK_NULL_HANDLE || offset > resource->size) {
        return false;
    }

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = resource->buffer;
    bufferInfo.offset = offset;
    bufferInfo.range = range == VK_WHOLE_SIZE ? resource->size - offset : range;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = m_bindlessSet;
    write.dstBinding = kVulkanBindlessStorageBufferBinding;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    return true;
}

bool VulkanDescriptorManager::updateBindlessAccelerationStructure(
    uint32_t index,
    const RhiAccelerationStructure* accelerationStructure) {
    if (m_bindlessSet == VK_NULL_HANDLE || index >= kVulkanBindlessMaxAccelerationStructures) {
        return false;
    }

    const VkAccelerationStructureKHR vkAccelerationStructure =
        getVulkanAccelerationStructureHandle(accelerationStructure);
    if (vkAccelerationStructure == VK_NULL_HANDLE) {
        return false;
    }

    VkWriteDescriptorSetAccelerationStructureKHR accelerationInfo{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    accelerationInfo.accelerationStructureCount = 1;
    accelerationInfo.pAccelerationStructures = &vkAccelerationStructure;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.pNext = &accelerationInfo;
    write.dstSet = m_bindlessSet;
    write.dstBinding = kVulkanBindlessAccelerationStructureBinding;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
    return true;
}

void VulkanDescriptorManager::flushAndBind(
    VkCommandBuffer cmd,
    VkPipelineBindPoint bindPoint,
    const VulkanPipelineResource& pipeline,
    const std::array<PendingBufferBinding, kMaxBufferBindings>& buffers,
    const std::array<PendingTextureBinding, kMaxTextureBindings>& textures,
    const std::array<PendingSamplerBinding, kMaxSamplerBindings>& samplers,
    const std::array<PendingAccelerationStructureBinding,
                     kMaxAccelerationStructureBindings>& accelerationStructures) {

    if (pipeline.layout == VK_NULL_HANDLE || pipeline.setLayouts.empty()) {
        return;
    }

    FrameState& frame = currentFrame();

    std::vector<VkDescriptorSet> boundSets(pipeline.setLayouts.size(), VK_NULL_HANDLE);
    std::vector<VkDescriptorSetLayout> transientLayouts;
    std::vector<uint32_t> transientSetIndices;
    transientLayouts.reserve(pipeline.setLayouts.size());
    transientSetIndices.reserve(pipeline.setLayouts.size());

    const bool pipelineUsesBindlessSet =
        pipeline.bindlessSetIndex < pipeline.setLayouts.size() && m_bindlessSet != VK_NULL_HANDLE;
    if (pipeline.bindlessSetIndex < pipeline.setLayouts.size() && !pipelineUsesBindlessSet) {
        spdlog::warn("Pipeline expects bindless set {}, but the global bindless set is unavailable",
                     pipeline.bindlessSetIndex);
        return;
    }

    for (uint32_t setIndex = 0; setIndex < pipeline.setLayouts.size(); ++setIndex) {
        if (pipelineUsesBindlessSet && setIndex == pipeline.bindlessSetIndex) {
            boundSets[setIndex] = m_bindlessSet;
            continue;
        }

        transientLayouts.push_back(pipeline.setLayouts[setIndex]);
        transientSetIndices.push_back(setIndex);
    }

    if (!transientLayouts.empty()) {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = frame.pool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(transientLayouts.size());
        allocInfo.pSetLayouts = transientLayouts.data();

        std::vector<VkDescriptorSet> transientSets(transientLayouts.size(), VK_NULL_HANDLE);
        const VkResult allocResult = vkAllocateDescriptorSets(m_device, &allocInfo, transientSets.data());
        if (allocResult != VK_SUCCESS) {
            spdlog::warn("Failed to allocate descriptor sets, skipping bind");
            return;
        }

        for (size_t setIndex = 0; setIndex < transientSets.size(); ++setIndex) {
            boundSets[transientSetIndices[setIndex]] = transientSets[setIndex];
        }
    }

    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(kMaxBufferBindings + kMaxTextureBindings + kMaxSamplerBindings +
                   kMaxAccelerationStructureBindings);

    std::vector<VkDescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(kMaxBufferBindings);

    std::vector<VkDescriptorImageInfo> imageInfos;
    imageInfos.reserve(kMaxTextureBindings + kMaxSamplerBindings);

    std::vector<VkWriteDescriptorSetAccelerationStructureKHR> accelerationInfos;
    accelerationInfos.reserve(kMaxAccelerationStructureBindings);

    for (uint32_t logicalIndex = 0; logicalIndex < kMaxBufferBindings; ++logicalIndex) {
        const VulkanDescriptorBindingLocation& location = pipeline.bufferBindings[logicalIndex];
        if (!location.valid() || buffers[logicalIndex].buffer == VK_NULL_HANDLE ||
            location.set >= boundSets.size() || boundSets[location.set] == VK_NULL_HANDLE) {
            continue;
        }

        bufferInfos.push_back({buffers[logicalIndex].buffer,
                               buffers[logicalIndex].offset,
                               buffers[logicalIndex].range});

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = boundSets[location.set];
        write.dstBinding = location.binding;
        write.dstArrayElement = location.arrayElement;
        write.descriptorCount = 1;
        write.descriptorType = location.descriptorType;
        write.pBufferInfo = &bufferInfos.back();
        writes.push_back(write);
    }

    for (uint32_t logicalIndex = 0; logicalIndex < kMaxTextureBindings; ++logicalIndex) {
        const VulkanDescriptorBindingLocation& location = pipeline.textureBindings[logicalIndex];
        if (!location.valid() || textures[logicalIndex].imageView == VK_NULL_HANDLE ||
            location.set >= boundSets.size() || boundSets[location.set] == VK_NULL_HANDLE) {
            continue;
        }

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = textures[logicalIndex].imageView;
        imageInfo.imageLayout = location.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
            ? VK_IMAGE_LAYOUT_GENERAL
            : textures[logicalIndex].layout;
        imageInfos.push_back(imageInfo);

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = boundSets[location.set];
        write.dstBinding = location.binding;
        write.dstArrayElement = location.arrayElement;
        write.descriptorCount = 1;
        write.descriptorType = location.descriptorType;
        write.pImageInfo = &imageInfos.back();
        writes.push_back(write);
    }

    for (uint32_t logicalIndex = 0; logicalIndex < kMaxSamplerBindings; ++logicalIndex) {
        const VulkanDescriptorBindingLocation& location = pipeline.samplerBindings[logicalIndex];
        if (!location.valid() || samplers[logicalIndex].sampler == VK_NULL_HANDLE ||
            location.set >= boundSets.size() || boundSets[location.set] == VK_NULL_HANDLE) {
            continue;
        }

        VkDescriptorImageInfo samplerInfo{};
        samplerInfo.sampler = samplers[logicalIndex].sampler;
        imageInfos.push_back(samplerInfo);

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = boundSets[location.set];
        write.dstBinding = location.binding;
        write.dstArrayElement = location.arrayElement;
        write.descriptorCount = 1;
        write.descriptorType = location.descriptorType;
        write.pImageInfo = &imageInfos.back();
        writes.push_back(write);
    }

    int32_t fallbackLocationIndex = -1;
    uint32_t validLocationCount = 0;
    for (uint32_t logicalIndex = 0; logicalIndex < kMaxAccelerationStructureBindings; ++logicalIndex) {
        if (pipeline.accelerationStructureBindings[logicalIndex].valid()) {
            fallbackLocationIndex = static_cast<int32_t>(logicalIndex);
            ++validLocationCount;
        }
    }

    uint32_t pendingAccelerationStructureCount = 0;
    int32_t fallbackPendingIndex = -1;
    for (uint32_t logicalIndex = 0; logicalIndex < kMaxAccelerationStructureBindings; ++logicalIndex) {
        if (accelerationStructures[logicalIndex].accelerationStructure != VK_NULL_HANDLE) {
            fallbackPendingIndex = static_cast<int32_t>(logicalIndex);
            ++pendingAccelerationStructureCount;
        }
    }

    auto appendAccelerationStructureWrite = [&](uint32_t pendingIndex, uint32_t locationIndex) {
        const VulkanDescriptorBindingLocation& location =
            pipeline.accelerationStructureBindings[locationIndex];
        if (!location.valid() || location.set >= boundSets.size() ||
            boundSets[location.set] == VK_NULL_HANDLE) {
            return;
        }

        VkWriteDescriptorSetAccelerationStructureKHR accelerationInfo{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        accelerationInfo.accelerationStructureCount = 1;
        accelerationInfo.pAccelerationStructures =
            &accelerationStructures[pendingIndex].accelerationStructure;
        accelerationInfos.push_back(accelerationInfo);

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.pNext = &accelerationInfos.back();
        write.dstSet = boundSets[location.set];
        write.dstBinding = location.binding;
        write.dstArrayElement = location.arrayElement;
        write.descriptorCount = 1;
        write.descriptorType = location.descriptorType;
        writes.push_back(write);
    };

    bool wroteAnyAccelerationStructure = false;
    for (uint32_t logicalIndex = 0; logicalIndex < kMaxAccelerationStructureBindings; ++logicalIndex) {
        const VulkanDescriptorBindingLocation& location =
            pipeline.accelerationStructureBindings[logicalIndex];
        if (!location.valid() ||
            accelerationStructures[logicalIndex].accelerationStructure == VK_NULL_HANDLE) {
            continue;
        }

        appendAccelerationStructureWrite(logicalIndex, logicalIndex);
        wroteAnyAccelerationStructure = true;
    }

    if (!wroteAnyAccelerationStructure &&
        validLocationCount == 1 &&
        pendingAccelerationStructureCount == 1 &&
        fallbackLocationIndex >= 0 &&
        fallbackPendingIndex >= 0) {
        appendAccelerationStructureWrite(static_cast<uint32_t>(fallbackPendingIndex),
                                         static_cast<uint32_t>(fallbackLocationIndex));
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    vulkanCmdBindDescriptorSetsHooked(cmd,
                                      bindPoint,
                                      pipeline.layout,
                                      0,
                                      static_cast<uint32_t>(boundSets.size()),
                                      boundSets.data(),
                                      0,
                                      nullptr);
}

void VulkanDescriptorManager::createPools() {
    for (auto& frame : m_frames) {
        frame.pool = createDescriptorPool(m_device);
    }
}

void VulkanDescriptorManager::destroyFrameState(FrameState& frame) {
    if (frame.uniformUpload.buffer != VK_NULL_HANDLE && frame.uniformUpload.allocation != nullptr) {
        vmaDestroyBuffer(m_allocator,
                         frame.uniformUpload.buffer,
                         frame.uniformUpload.allocation);
        frame.uniformUpload = {};
    }

    if (frame.pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, frame.pool, nullptr);
        frame.pool = VK_NULL_HANDLE;
    }
}

bool VulkanDescriptorManager::ensureFrameUniformUploadCapacity(FrameState& frame,
                                                               VkDeviceSize requiredSize) {
    if (requiredSize == 0) {
        return true;
    }
    if (frame.uniformUpload.buffer != VK_NULL_HANDLE &&
        frame.uniformUpload.capacity >= requiredSize &&
        frame.uniformUpload.mappedData != nullptr) {
        return true;
    }

    if (frame.uniformUpload.buffer != VK_NULL_HANDLE && frame.uniformUpload.allocation != nullptr) {
        vmaDestroyBuffer(m_allocator,
                         frame.uniformUpload.buffer,
                         frame.uniformUpload.allocation);
        frame.uniformUpload = {};
    }

    VmaBufferCreateInfo bufferInfo{};
    bufferInfo.device = m_device;
    bufferInfo.allocator = m_allocator;
    bufferInfo.size = nextUploadCapacity(requiredSize);
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.hostVisible = true;
    bufferInfo.debugName = "InlineUniformUpload";

    const char* errorMessage = nullptr;
    auto resource = vmaCreateBufferResource(bufferInfo, &errorMessage);
    if (!resource) {
        spdlog::warn("Failed to allocate frame uniform upload buffer: {}",
                     errorMessage ? errorMessage : "unknown error");
        return false;
    }

    if (resource->mappedData == nullptr && resource->allocation != nullptr) {
        void* mappedData = nullptr;
        if (vmaMapMemory(m_allocator, resource->allocation, &mappedData) != VK_SUCCESS) {
            vmaDestroyBuffer(m_allocator, resource->buffer, resource->allocation);
            spdlog::warn("Failed to map frame uniform upload buffer");
            return false;
        }
        resource->mappedData = mappedData;
    }

    frame.uniformUpload.buffer = resource->buffer;
    frame.uniformUpload.allocation = resource->allocation;
    frame.uniformUpload.mappedData = resource->mappedData;
    frame.uniformUpload.capacity = bufferInfo.size;
    frame.uniformUpload.head = 0;
    return true;
}

VulkanDescriptorManager::FrameState& VulkanDescriptorManager::currentFrame() {
    return m_frames[m_frameIndex % static_cast<uint32_t>(m_frames.size())];
}

bool vulkanRetainBindlessSetLayout(VkDevice device,
                                   VkDescriptorSetLayout& outLayout,
                                   std::string* errorMessage) {
    if (device == VK_NULL_HANDLE) {
        if (errorMessage) {
            *errorMessage = "Cannot create bindless layout without a valid Vulkan device";
        }
        outLayout = VK_NULL_HANDLE;
        return false;
    }

    std::lock_guard<std::mutex> lock(g_bindlessLayoutMutex);
    SharedBindlessLayoutState& state = g_bindlessLayouts[bindlessLayoutKey(device)];
    if (state.layout == VK_NULL_HANDLE) {
        try {
            state.layout = createBindlessSetLayout(device);
        } catch (const std::exception& e) {
            g_bindlessLayouts.erase(bindlessLayoutKey(device));
            if (errorMessage) {
                *errorMessage = e.what();
            }
            outLayout = VK_NULL_HANDLE;
            return false;
        }
    }

    ++state.refCount;
    outLayout = state.layout;
    return true;
}

void vulkanReleaseBindlessSetLayout(VkDevice device) {
    if (device == VK_NULL_HANDLE) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_bindlessLayoutMutex);
    auto it = g_bindlessLayouts.find(bindlessLayoutKey(device));
    if (it == g_bindlessLayouts.end()) {
        return;
    }

    if (it->second.refCount > 0) {
        --it->second.refCount;
    }
    if (it->second.refCount == 0) {
        if (it->second.layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, it->second.layout, nullptr);
        }
        g_bindlessLayouts.erase(it);
    }
}

bool vulkanIsBindlessSetIndex(uint32_t setIndex) {
    return setIndex == kVulkanBindlessSetIndex;
}

#endif // _WIN32
