#include "vulkan_descriptor_manager.h"

#ifdef _WIN32

#include "../rhi_resource_utils.h"
#include "vulkan_resource_handles.h"

#include <vk_mem_alloc.h>

#include <cstring>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace {

void checkVk(VkResult result, const char* message) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(message) + " (VkResult: " + std::to_string(result) + ")");
    }
}

constexpr uint32_t kMaxSetsPerFrame = 4096;

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

} // namespace

void VulkanDescriptorManager::init(VkDevice device, VmaAllocator allocator) {
    m_device = device;
    m_allocator = allocator;
    createPools();
}

void VulkanDescriptorManager::destroy() {
    for (auto& frame : m_frames) {
        destroyFrameState(frame);
    }
    m_allocator = nullptr;
    m_device = VK_NULL_HANDLE;
}

void VulkanDescriptorManager::resetFrame() {
    m_frameIndex = (m_frameIndex + 1) % static_cast<uint32_t>(m_frames.size());

    FrameState& frame = currentFrame();
    if (frame.pool != VK_NULL_HANDLE) {
        vkResetDescriptorPool(m_device, frame.pool, 0);
    }

    for (const TransientBuffer& transient : frame.transientBuffers) {
        if (transient.buffer != VK_NULL_HANDLE && transient.allocation != nullptr) {
            vmaDestroyBuffer(m_allocator,
                             transient.buffer,
                             static_cast<VmaAllocation>(transient.allocation));
        }
    }
    frame.transientBuffers.clear();
}

PendingBufferBinding VulkanDescriptorManager::createTransientUniformBuffer(const void* data, size_t size) {
    PendingBufferBinding binding{};
    if (!data || size == 0 || m_device == VK_NULL_HANDLE || m_allocator == nullptr) {
        return binding;
    }

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VmaAllocationInfo allocationInfo{};
    const VkResult result = vmaCreateBuffer(m_allocator,
                                            &bufferInfo,
                                            &allocInfo,
                                            &buffer,
                                            &allocation,
                                            &allocationInfo);
    if (result != VK_SUCCESS) {
        spdlog::warn("Failed to allocate transient uniform buffer (VkResult: {})",
                     static_cast<int>(result));
        return binding;
    }

    std::memcpy(allocationInfo.pMappedData, data, size);
    currentFrame().transientBuffers.push_back({buffer, allocation});

    binding.buffer = buffer;
    binding.range = size;
    binding.dirty = true;
    return binding;
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

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = frame.pool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(pipeline.setLayouts.size());
    allocInfo.pSetLayouts = pipeline.setLayouts.data();

    std::vector<VkDescriptorSet> sets(pipeline.setLayouts.size(), VK_NULL_HANDLE);
    const VkResult allocResult = vkAllocateDescriptorSets(m_device, &allocInfo, sets.data());
    if (allocResult != VK_SUCCESS) {
        spdlog::warn("Failed to allocate descriptor sets, skipping bind");
        return;
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
            location.set >= sets.size()) {
            continue;
        }

        bufferInfos.push_back({buffers[logicalIndex].buffer,
                               buffers[logicalIndex].offset,
                               buffers[logicalIndex].range});

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = sets[location.set];
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
            location.set >= sets.size()) {
            continue;
        }

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageView = textures[logicalIndex].imageView;
        imageInfo.imageLayout = location.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
            ? VK_IMAGE_LAYOUT_GENERAL
            : textures[logicalIndex].layout;
        imageInfos.push_back(imageInfo);

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = sets[location.set];
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
            location.set >= sets.size()) {
            continue;
        }

        VkDescriptorImageInfo samplerInfo{};
        samplerInfo.sampler = samplers[logicalIndex].sampler;
        imageInfos.push_back(samplerInfo);

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = sets[location.set];
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
        if (!location.valid() || location.set >= sets.size()) {
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
        write.dstSet = sets[location.set];
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
                                      static_cast<uint32_t>(sets.size()),
                                      sets.data(),
                                      0,
                                      nullptr);
}

void VulkanDescriptorManager::createPools() {
    for (auto& frame : m_frames) {
        frame.pool = createDescriptorPool(m_device);
    }
}

void VulkanDescriptorManager::destroyFrameState(FrameState& frame) {
    for (const TransientBuffer& transient : frame.transientBuffers) {
        if (transient.buffer != VK_NULL_HANDLE && transient.allocation != nullptr) {
            vmaDestroyBuffer(m_allocator,
                             transient.buffer,
                             static_cast<VmaAllocation>(transient.allocation));
        }
    }
    frame.transientBuffers.clear();

    if (frame.pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, frame.pool, nullptr);
        frame.pool = VK_NULL_HANDLE;
    }
}

VulkanDescriptorManager::FrameState& VulkanDescriptorManager::currentFrame() {
    return m_frames[m_frameIndex % static_cast<uint32_t>(m_frames.size())];
}

#endif // _WIN32
