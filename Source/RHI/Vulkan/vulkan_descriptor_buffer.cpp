#include "vulkan_descriptor_buffer.h"

#ifdef _WIN32

#include "rhi_resource_utils.h"
#include "vulkan_resource_handles.h"

#include <vk_mem_alloc.h>

#include <algorithm>
#include <cstring>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace {

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0) return value;
    return (value + alignment - 1) & ~(alignment - 1);
}

VkDeviceSize alignDown(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment == 0) return value;
    return value & ~(alignment - 1);
}

constexpr VkDeviceSize kDefaultTransientDescriptorBufferSize = 4u << 20; // 4 MB per frame
constexpr VkDeviceSize kDefaultInlineUniformUploadSize = 1u << 20;       // 1 MB

VkDeviceSize nextPowerOf2(VkDeviceSize value) {
    VkDeviceSize result = kDefaultInlineUniformUploadSize;
    while (result < value) result <<= 1;
    return result;
}

VkBuffer createDescriptorBufferVma(VkDevice device, VmaAllocator allocator,
                                   VkDeviceSize size, VmaAllocation* outAllocation,
                                   void** outMapped, VkDeviceAddress* outAddress,
                                   const char* debugName) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                       VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocationInfo allocInfo{};
    VkResult result = vmaCreateBuffer(allocator, &bufferInfo, &allocCreateInfo,
                                      &buffer, outAllocation, &allocInfo);
    if (result != VK_SUCCESS) {
        spdlog::error("Failed to create descriptor buffer '{}' (VkResult: {})", debugName, static_cast<int>(result));
        return VK_NULL_HANDLE;
    }

    *outMapped = allocInfo.pMappedData;

    VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer = buffer;
    *outAddress = vkGetBufferDeviceAddress(device, &addrInfo);

    if (debugName) {
        vmaSetAllocationName(allocator, *outAllocation, debugName);
    }

    return buffer;
}

} // namespace

void VulkanDescriptorBufferManager::init(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VmaAllocator allocator,
    const VkPhysicalDeviceDescriptorBufferPropertiesEXT& props,
    VkDeviceSize minUniformBufferOffsetAlignment,
    VkDeviceSize nonCoherentAtomSize,
    VkDeviceSize maxUniformBufferRange) {

    m_device = device;
    m_allocator = allocator;
    m_uniformUploadAlignment = std::max<VkDeviceSize>(minUniformBufferOffsetAlignment, 16);
    m_nonCoherentAtomSize = std::max<VkDeviceSize>(nonCoherentAtomSize, 1);
    m_maxUniformBufferRange = maxUniformBufferRange;

    // Store descriptor sizes
    m_sampledImageDescSize = props.sampledImageDescriptorSize;
    m_samplerDescSize = props.samplerDescriptorSize;
    m_storageImageDescSize = props.storageImageDescriptorSize;
    m_storageBufferDescSize = props.storageBufferDescriptorSize;
    m_uniformBufferDescSize = props.uniformBufferDescriptorSize;
    m_accelerationStructureDescSize = props.accelerationStructureDescriptorSize;
    m_descriptorBufferOffsetAlignment = props.descriptorBufferOffsetAlignment;

    // Load function pointers
    m_vkGetDescriptorSetLayoutSizeEXT = reinterpret_cast<PFN_vkGetDescriptorSetLayoutSizeEXT>(
        vkGetDeviceProcAddr(device, "vkGetDescriptorSetLayoutSizeEXT"));
    m_vkGetDescriptorSetLayoutBindingOffsetEXT =
        reinterpret_cast<PFN_vkGetDescriptorSetLayoutBindingOffsetEXT>(
            vkGetDeviceProcAddr(device, "vkGetDescriptorSetLayoutBindingOffsetEXT"));
    m_vkGetDescriptorEXT = reinterpret_cast<PFN_vkGetDescriptorEXT>(
        vkGetDeviceProcAddr(device, "vkGetDescriptorEXT"));
    m_vkCmdBindDescriptorBuffersEXT = reinterpret_cast<PFN_vkCmdBindDescriptorBuffersEXT>(
        vkGetDeviceProcAddr(device, "vkCmdBindDescriptorBuffersEXT"));
    m_vkCmdSetDescriptorBufferOffsetsEXT = reinterpret_cast<PFN_vkCmdSetDescriptorBufferOffsetsEXT>(
        vkGetDeviceProcAddr(device, "vkCmdSetDescriptorBufferOffsetsEXT"));
    m_vkGetAccelerationStructureDeviceAddressKHR =
        reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
            vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));

    if (!m_vkGetDescriptorSetLayoutSizeEXT || !m_vkGetDescriptorSetLayoutBindingOffsetEXT ||
        !m_vkGetDescriptorEXT || !m_vkCmdBindDescriptorBuffersEXT ||
        !m_vkCmdSetDescriptorBufferOffsetsEXT) {
        throw std::runtime_error("Failed to load VK_EXT_descriptor_buffer function pointers");
    }

    // Acquire the shared bindless set layout (with descriptor buffer flag)
    std::string errorMessage;
    if (!vulkanRetainBindlessSetLayout(m_device, m_bindlessSetLayout, &errorMessage, true)) {
        throw std::runtime_error(errorMessage.empty()
                                     ? "Failed to acquire bindless descriptor set layout (descriptor buffer)"
                                     : errorMessage);
    }

    // Query bindless set layout size and per-binding offsets
    m_vkGetDescriptorSetLayoutSizeEXT(device, m_bindlessSetLayout, &m_bindlessSetLayoutSize);
    m_bindlessSetLayoutSize = alignUp(m_bindlessSetLayoutSize, m_descriptorBufferOffsetAlignment);

    m_vkGetDescriptorSetLayoutBindingOffsetEXT(
        device, m_bindlessSetLayout, kVulkanBindlessSampledImageBinding, &m_bindlessSampledImageOffset);
    m_vkGetDescriptorSetLayoutBindingOffsetEXT(
        device, m_bindlessSetLayout, kVulkanBindlessSamplerBinding, &m_bindlessSamplerOffset);
    m_vkGetDescriptorSetLayoutBindingOffsetEXT(
        device, m_bindlessSetLayout, kVulkanBindlessStorageImageBinding, &m_bindlessStorageImageOffset);
    m_vkGetDescriptorSetLayoutBindingOffsetEXT(
        device, m_bindlessSetLayout, kVulkanBindlessStorageBufferBinding, &m_bindlessStorageBufferOffset);
    m_vkGetDescriptorSetLayoutBindingOffsetEXT(
        device, m_bindlessSetLayout, kVulkanBindlessAccelerationStructureBinding,
        &m_bindlessAccelerationStructureOffset);

    // Create the persistent bindless descriptor buffer
    m_bindlessBuffer = createDescriptorBufferVma(
        device, allocator, m_bindlessSetLayoutSize,
        &m_bindlessAllocation, &m_bindlessMapped, &m_bindlessBufferAddress,
        "BindlessDescriptorBuffer");
    if (m_bindlessBuffer == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to create bindless descriptor buffer");
    }
    m_bindlessBufferSize = m_bindlessSetLayoutSize;
    std::memset(m_bindlessMapped, 0, static_cast<size_t>(m_bindlessBufferSize));

    // Create per-frame transient descriptor buffers
    for (auto& frame : m_frames) {
        frame.descriptorCapacity = kDefaultTransientDescriptorBufferSize;
        frame.descriptorBuffer = createDescriptorBufferVma(
            device, allocator, frame.descriptorCapacity,
            &frame.descriptorAllocation, &frame.descriptorMapped,
            &frame.descriptorBufferAddress, "TransientDescriptorBuffer");
        if (frame.descriptorBuffer == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to create transient descriptor buffer");
        }
        frame.descriptorHead = 0;
    }

    spdlog::info("Vulkan: descriptor buffer manager initialized "
                 "(bindlessSize={}, transientSize={}, alignment={})",
                 m_bindlessSetLayoutSize, kDefaultTransientDescriptorBufferSize,
                 m_descriptorBufferOffsetAlignment);
}

void VulkanDescriptorBufferManager::destroy() {
    for (auto& frame : m_frames) {
        destroyFrameState(frame);
    }

    if (m_bindlessBuffer != VK_NULL_HANDLE && m_allocator != nullptr) {
        vmaDestroyBuffer(m_allocator, m_bindlessBuffer, m_bindlessAllocation);
        m_bindlessBuffer = VK_NULL_HANDLE;
        m_bindlessAllocation = nullptr;
    }

    if (m_bindlessSetLayout != VK_NULL_HANDLE) {
        vulkanReleaseBindlessSetLayout(m_device);
        m_bindlessSetLayout = VK_NULL_HANDLE;
    }

    m_allocator = nullptr;
    m_device = VK_NULL_HANDLE;
}

void VulkanDescriptorBufferManager::resetFrame() {
    m_frameIndex = (m_frameIndex + 1) % static_cast<uint32_t>(m_frames.size());
    FrameState& frame = currentFrame();
    frame.descriptorHead = 0;
    frame.uniformHead = 0;
}

PendingBufferBinding VulkanDescriptorBufferManager::uploadInlineUniformData(const void* data,
                                                                            size_t size) {
    PendingBufferBinding binding{};
    if (!data || size == 0 || m_device == VK_NULL_HANDLE || m_allocator == nullptr) {
        return binding;
    }

    if (size > m_maxUniformBufferRange) {
        spdlog::warn("Inline uniform upload of {} bytes exceeds maxUniformBufferRange {}; skipping",
                     size, m_maxUniformBufferRange);
        return binding;
    }

    FrameState& frame = currentFrame();
    const VkDeviceSize alignedOffset = alignUp(frame.uniformHead, m_uniformUploadAlignment);
    const VkDeviceSize requiredSize = alignedOffset + static_cast<VkDeviceSize>(size);
    if (!ensureFrameUniformUploadCapacity(frame, requiredSize) ||
        frame.uniformBuffer == VK_NULL_HANDLE || frame.uniformMapped == nullptr) {
        return binding;
    }

    auto* dst = static_cast<std::byte*>(frame.uniformMapped) + alignedOffset;
    std::memcpy(dst, data, size);

    const VkDeviceSize flushOffset = alignDown(alignedOffset, m_nonCoherentAtomSize);
    const VkDeviceSize flushSize =
        alignUp((alignedOffset - flushOffset) + static_cast<VkDeviceSize>(size), m_nonCoherentAtomSize);
    vmaFlushAllocation(m_allocator, frame.uniformAllocation, flushOffset, flushSize);

    frame.uniformHead = requiredSize;

    binding.buffer = frame.uniformBuffer;
    binding.offset = alignedOffset;
    binding.range = size;
    binding.dirty = true;
    binding.trackState = false;
    return binding;
}

// --- Bindless updates (write directly into persistent bindless descriptor buffer) ---

void VulkanDescriptorBufferManager::updateBindlessSampledTextures(const RhiTexture* const* textures,
                                                                   uint32_t startIndex,
                                                                   uint32_t count) {
    if (!textures || count == 0) return;
    for (uint32_t i = 0; i < count; ++i) {
        updateBindlessSampledTexture(startIndex + i, textures[i]);
    }
}

bool VulkanDescriptorBufferManager::updateBindlessSampledTexture(uint32_t index,
                                                                  const RhiTexture* texture) {
    if (!m_bindlessMapped || index >= kVulkanBindlessMaxSampledImages) return false;
    const VkImageView imageView = getVulkanImageView(texture);
    if (imageView == VK_NULL_HANDLE) return false;

    void* dst = static_cast<std::byte*>(m_bindlessMapped) +
                m_bindlessSampledImageOffset +
                index * m_sampledImageDescSize;
    writeImageDescriptor(dst, imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, m_sampledImageDescSize);
    return true;
}

bool VulkanDescriptorBufferManager::updateBindlessSampler(uint32_t index,
                                                           const RhiSampler* sampler) {
    if (!m_bindlessMapped || index >= kVulkanBindlessMaxSamplers) return false;
    const VkSampler vkSampler = getVulkanSamplerHandle(sampler);
    if (vkSampler == VK_NULL_HANDLE) return false;

    void* dst = static_cast<std::byte*>(m_bindlessMapped) +
                m_bindlessSamplerOffset +
                index * m_samplerDescSize;
    writeSamplerDescriptor(dst, vkSampler);
    return true;
}

bool VulkanDescriptorBufferManager::updateBindlessStorageImage(uint32_t index,
                                                                const RhiTexture* texture) {
    if (!m_bindlessMapped || index >= kVulkanBindlessMaxStorageImages) return false;
    const VkImageView imageView = getVulkanImageView(texture);
    if (imageView == VK_NULL_HANDLE) return false;

    void* dst = static_cast<std::byte*>(m_bindlessMapped) +
                m_bindlessStorageImageOffset +
                index * m_storageImageDescSize;
    writeImageDescriptor(dst, imageView, VK_IMAGE_LAYOUT_GENERAL,
                         VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, m_storageImageDescSize);
    return true;
}

bool VulkanDescriptorBufferManager::updateBindlessStorageBuffer(uint32_t index,
                                                                 const RhiBuffer* buffer,
                                                                 VkDeviceSize offset,
                                                                 VkDeviceSize range) {
    if (!m_bindlessMapped || index >= kVulkanBindlessMaxStorageBuffers) return false;

    const VulkanBufferResource* resource = getVulkanBufferResource(buffer);
    if (!resource || resource->buffer == VK_NULL_HANDLE || offset > resource->size) return false;

    VkDeviceAddress address = resource->deviceAddress + offset;
    VkDeviceSize actualRange = (range == VK_WHOLE_SIZE) ? (resource->size - offset) : range;

    void* dst = static_cast<std::byte*>(m_bindlessMapped) +
                m_bindlessStorageBufferOffset +
                index * m_storageBufferDescSize;
    writeBufferDescriptor(dst, address, actualRange,
                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_storageBufferDescSize);
    return true;
}

bool VulkanDescriptorBufferManager::updateBindlessAccelerationStructure(
    uint32_t index,
    const RhiAccelerationStructure* accelerationStructure) {
    if (!m_bindlessMapped || index >= kVulkanBindlessMaxAccelerationStructures) return false;

    const VkDeviceAddress address =
        getVulkanAccelerationStructureDeviceAddress(accelerationStructure);
    if (address == 0) return false;

    void* dst = static_cast<std::byte*>(m_bindlessMapped) +
                m_bindlessAccelerationStructureOffset +
                index * m_accelerationStructureDescSize;
    writeAccelerationStructureDescriptor(dst, address);
    return true;
}

// --- flushAndBind ---

void VulkanDescriptorBufferManager::flushAndBind(
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

    // Determine which sets are in use
    const bool hasBindlessSet =
        pipeline.bindlessSetIndex < pipeline.setLayouts.size() && m_bindlessBuffer != VK_NULL_HANDLE;

    // --- Build transient descriptors for non-bindless sets ---
    // For each non-bindless set that has bindings, allocate space in the transient buffer
    // and write descriptors.
    struct SetEntry {
        uint32_t bufferIndex = 0; // index into the descriptor buffer binding array
        VkDeviceSize offset = 0;
        bool valid = false;
    };
    std::vector<SetEntry> setEntries(pipeline.setLayouts.size());

    // We bind up to 2 descriptor buffers: [0]=transient, [1]=bindless
    // The transient buffer is always index 0, bindless is always index 1
    uint32_t transientBufferIdx = 0;
    uint32_t bindlessBufferIdx = 1;

    for (uint32_t setIndex = 0; setIndex < pipeline.setLayouts.size(); ++setIndex) {
        if (hasBindlessSet && setIndex == pipeline.bindlessSetIndex) {
            // Bindless set: use the persistent bindless buffer
            setEntries[setIndex].bufferIndex = bindlessBufferIdx;
            setEntries[setIndex].offset = 0; // base of bindless buffer
            setEntries[setIndex].valid = true;
            continue;
        }

        if (pipeline.setLayouts[setIndex] == VK_NULL_HANDLE) {
            continue;
        }

        // Query this set layout's size
        VkDeviceSize setSize = 0;
        m_vkGetDescriptorSetLayoutSizeEXT(m_device, pipeline.setLayouts[setIndex], &setSize);
        setSize = alignUp(setSize, m_descriptorBufferOffsetAlignment);

        if (setSize == 0) continue;

        // Bump-allocate from the transient per-frame descriptor buffer
        VkDeviceSize allocOffset = allocateTransientDescriptorSpace(setSize);
        if (allocOffset == UINT64_MAX) {
            spdlog::warn("Transient descriptor buffer exhausted (set {})", setIndex);
            continue;
        }

        void* setBase = static_cast<std::byte*>(frame.descriptorMapped) + allocOffset;
        std::memset(setBase, 0, static_cast<size_t>(setSize));

        // Write buffer descriptors
        for (uint32_t logicalIndex = 0; logicalIndex < kMaxBufferBindings; ++logicalIndex) {
            const auto& location = pipeline.bufferBindings[logicalIndex];
            if (!location.valid() || location.set != setIndex ||
                buffers[logicalIndex].buffer == VK_NULL_HANDLE) {
                continue;
            }

            VkDeviceSize bindingOffset = 0;
            m_vkGetDescriptorSetLayoutBindingOffsetEXT(
                m_device, pipeline.setLayouts[setIndex], location.binding, &bindingOffset);

            VkDeviceSize descSize = (location.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                                        ? m_uniformBufferDescSize
                                        : m_storageBufferDescSize;

            // Get device address of the buffer
            VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            addrInfo.buffer = buffers[logicalIndex].buffer;
            VkDeviceAddress address = vkGetBufferDeviceAddress(m_device, &addrInfo);
            address += buffers[logicalIndex].offset;

            VkDeviceSize range = buffers[logicalIndex].range;

            void* dst = static_cast<std::byte*>(setBase) + bindingOffset +
                        location.arrayElement * descSize;
            writeBufferDescriptor(dst, address, range, location.descriptorType, descSize);
        }

        // Write texture descriptors
        for (uint32_t logicalIndex = 0; logicalIndex < kMaxTextureBindings; ++logicalIndex) {
            const auto& location = pipeline.textureBindings[logicalIndex];
            if (!location.valid() || location.set != setIndex ||
                textures[logicalIndex].imageView == VK_NULL_HANDLE) {
                continue;
            }

            VkDeviceSize bindingOffset = 0;
            m_vkGetDescriptorSetLayoutBindingOffsetEXT(
                m_device, pipeline.setLayouts[setIndex], location.binding, &bindingOffset);

            VkImageLayout layout = (location.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                       ? VK_IMAGE_LAYOUT_GENERAL
                                       : textures[logicalIndex].layout;

            VkDeviceSize descSize = (location.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                                        ? m_storageImageDescSize
                                        : m_sampledImageDescSize;

            void* dst = static_cast<std::byte*>(setBase) + bindingOffset +
                        location.arrayElement * descSize;
            writeImageDescriptor(dst, textures[logicalIndex].imageView, layout,
                                 location.descriptorType, descSize);
        }

        // Write sampler descriptors
        for (uint32_t logicalIndex = 0; logicalIndex < kMaxSamplerBindings; ++logicalIndex) {
            const auto& location = pipeline.samplerBindings[logicalIndex];
            if (!location.valid() || location.set != setIndex ||
                samplers[logicalIndex].sampler == VK_NULL_HANDLE) {
                continue;
            }

            VkDeviceSize bindingOffset = 0;
            m_vkGetDescriptorSetLayoutBindingOffsetEXT(
                m_device, pipeline.setLayouts[setIndex], location.binding, &bindingOffset);

            void* dst = static_cast<std::byte*>(setBase) + bindingOffset +
                        location.arrayElement * m_samplerDescSize;
            writeSamplerDescriptor(dst, samplers[logicalIndex].sampler);
        }

        // Write acceleration structure descriptors
        for (uint32_t logicalIndex = 0; logicalIndex < kMaxAccelerationStructureBindings; ++logicalIndex) {
            const auto& location = pipeline.accelerationStructureBindings[logicalIndex];
            if (!location.valid() || location.set != setIndex ||
                accelerationStructures[logicalIndex].accelerationStructure == VK_NULL_HANDLE) {
                continue;
            }

            VkDeviceSize bindingOffset = 0;
            m_vkGetDescriptorSetLayoutBindingOffsetEXT(
                m_device, pipeline.setLayouts[setIndex], location.binding, &bindingOffset);

            // Get device address of the acceleration structure
            VkAccelerationStructureDeviceAddressInfoKHR asAddrInfo{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            asAddrInfo.accelerationStructure =
                accelerationStructures[logicalIndex].accelerationStructure;
            VkDeviceAddress asAddress = 0;
            if (m_vkGetAccelerationStructureDeviceAddressKHR) {
                asAddress = m_vkGetAccelerationStructureDeviceAddressKHR(m_device, &asAddrInfo);
            }

            void* dst = static_cast<std::byte*>(setBase) + bindingOffset +
                        location.arrayElement * m_accelerationStructureDescSize;
            writeAccelerationStructureDescriptor(dst, asAddress);
        }

        setEntries[setIndex].bufferIndex = transientBufferIdx;
        setEntries[setIndex].offset = allocOffset;
        setEntries[setIndex].valid = true;
    }

    // --- Bind descriptor buffers ---
    std::array<VkDescriptorBufferBindingInfoEXT, 2> bufferBindings{};
    uint32_t bufferCount = 0;

    // Transient buffer always at index 0
    if (frame.descriptorBuffer != VK_NULL_HANDLE) {
        bufferBindings[0] = {VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT};
        bufferBindings[0].address = frame.descriptorBufferAddress;
        bufferBindings[0].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                                  VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
        bufferCount = 1;
    }

    // Bindless buffer at index 1
    if (hasBindlessSet && m_bindlessBuffer != VK_NULL_HANDLE) {
        bufferBindings[1] = {VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT};
        bufferBindings[1].address = m_bindlessBufferAddress;
        bufferBindings[1].usage = VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                                  VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT;
        bufferCount = 2;
    }

    if (bufferCount > 0) {
        m_vkCmdBindDescriptorBuffersEXT(cmd, bufferCount, bufferBindings.data());
    }

    // --- Set offsets for each set ---
    for (uint32_t setIndex = 0; setIndex < pipeline.setLayouts.size(); ++setIndex) {
        if (!setEntries[setIndex].valid) continue;

        uint32_t bufferIdx = setEntries[setIndex].bufferIndex;
        VkDeviceSize offset = setEntries[setIndex].offset;
        m_vkCmdSetDescriptorBufferOffsetsEXT(
            cmd, bindPoint, pipeline.layout,
            setIndex, 1, &bufferIdx, &offset);
    }
}

// --- Private helpers ---

VulkanDescriptorBufferManager::FrameState& VulkanDescriptorBufferManager::currentFrame() {
    return m_frames[m_frameIndex % static_cast<uint32_t>(m_frames.size())];
}

void VulkanDescriptorBufferManager::destroyFrameState(FrameState& frame) {
    if (frame.descriptorBuffer != VK_NULL_HANDLE && m_allocator != nullptr) {
        vmaDestroyBuffer(m_allocator, frame.descriptorBuffer, frame.descriptorAllocation);
        frame.descriptorBuffer = VK_NULL_HANDLE;
        frame.descriptorAllocation = nullptr;
    }
    if (frame.uniformBuffer != VK_NULL_HANDLE && m_allocator != nullptr) {
        vmaDestroyBuffer(m_allocator, frame.uniformBuffer, frame.uniformAllocation);
        frame.uniformBuffer = VK_NULL_HANDLE;
        frame.uniformAllocation = nullptr;
    }
}

bool VulkanDescriptorBufferManager::ensureFrameUniformUploadCapacity(FrameState& frame,
                                                                      VkDeviceSize requiredSize) {
    if (requiredSize == 0) return true;
    if (frame.uniformBuffer != VK_NULL_HANDLE && frame.uniformCapacity >= requiredSize &&
        frame.uniformMapped != nullptr) {
        return true;
    }

    if (frame.uniformBuffer != VK_NULL_HANDLE && frame.uniformAllocation != nullptr) {
        vmaDestroyBuffer(m_allocator, frame.uniformBuffer, frame.uniformAllocation);
        frame.uniformBuffer = VK_NULL_HANDLE;
        frame.uniformAllocation = nullptr;
        frame.uniformMapped = nullptr;
    }

    VmaBufferCreateInfo bufferInfo{};
    bufferInfo.device = m_device;
    bufferInfo.allocator = m_allocator;
    bufferInfo.size = nextPowerOf2(requiredSize);
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.hostVisible = true;
    bufferInfo.debugName = "DescBufInlineUniformUpload";

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

    frame.uniformBuffer = resource->buffer;
    frame.uniformAllocation = resource->allocation;
    frame.uniformMapped = resource->mappedData;
    frame.uniformCapacity = bufferInfo.size;
    frame.uniformHead = 0;
    return true;
}

VkDeviceSize VulkanDescriptorBufferManager::allocateTransientDescriptorSpace(VkDeviceSize size) {
    FrameState& frame = currentFrame();
    VkDeviceSize alignedHead = alignUp(frame.descriptorHead, m_descriptorBufferOffsetAlignment);
    if (alignedHead + size > frame.descriptorCapacity) {
        return UINT64_MAX; // Out of space
    }
    frame.descriptorHead = alignedHead + size;
    return alignedHead;
}

void VulkanDescriptorBufferManager::writeImageDescriptor(void* dst, VkImageView imageView,
                                                          VkImageLayout layout,
                                                          VkDescriptorType type,
                                                          VkDeviceSize descSize) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageView;
    imageInfo.imageLayout = layout;

    VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    getInfo.type = type;
    if (type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) {
        getInfo.data.pSampledImage = &imageInfo;
    } else {
        getInfo.data.pStorageImage = &imageInfo;
    }

    m_vkGetDescriptorEXT(m_device, &getInfo, static_cast<size_t>(descSize), dst);
}

void VulkanDescriptorBufferManager::writeSamplerDescriptor(void* dst, VkSampler sampler) {
    VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    getInfo.type = VK_DESCRIPTOR_TYPE_SAMPLER;
    getInfo.data.pSampler = &sampler;

    m_vkGetDescriptorEXT(m_device, &getInfo, static_cast<size_t>(m_samplerDescSize), dst);
}

void VulkanDescriptorBufferManager::writeBufferDescriptor(void* dst, VkDeviceAddress address,
                                                           VkDeviceSize range,
                                                           VkDescriptorType type,
                                                           VkDeviceSize descSize) {
    VkDescriptorAddressInfoEXT addressInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT};
    addressInfo.address = address;
    addressInfo.range = range;
    addressInfo.format = VK_FORMAT_UNDEFINED;

    VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    getInfo.type = type;
    if (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
        getInfo.data.pUniformBuffer = &addressInfo;
    } else {
        getInfo.data.pStorageBuffer = &addressInfo;
    }

    m_vkGetDescriptorEXT(m_device, &getInfo, static_cast<size_t>(descSize), dst);
}

void VulkanDescriptorBufferManager::writeAccelerationStructureDescriptor(void* dst,
                                                                          VkDeviceAddress address) {
    VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    getInfo.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    getInfo.data.accelerationStructure = address;

    m_vkGetDescriptorEXT(m_device, &getInfo, static_cast<size_t>(m_accelerationStructureDescSize), dst);
}

#endif // _WIN32
