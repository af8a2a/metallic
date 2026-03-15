#include "vulkan_descriptor_manager.h"

#ifdef _WIN32

#include <spdlog/spdlog.h>
#include <stdexcept>

namespace {

void checkVk(VkResult result, const char* message) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(message) + " (VkResult: " + std::to_string(result) + ")");
    }
}

// Maximum descriptor sets per frame (generous budget for complex pipelines)
constexpr uint32_t kMaxSetsPerFrame = 4096;

} // namespace

void VulkanDescriptorManager::init(VkDevice device) {
    m_device = device;
    createSetLayouts();
    createPipelineLayout();
    createPool();
}

void VulkanDescriptorManager::destroy() {
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    for (auto& layout : m_setLayouts) {
        if (layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
            layout = VK_NULL_HANDLE;
        }
    }
}

void VulkanDescriptorManager::resetFrame() {
    if (m_pool != VK_NULL_HANDLE) {
        vkResetDescriptorPool(m_device, m_pool, 0);
    }
}

void VulkanDescriptorManager::createSetLayouts() {
    // Set 0: Storage buffers (binding 0..kMaxBufferBindings-1)
    {
        std::array<VkDescriptorSetLayoutBinding, kMaxBufferBindings> bindings{};
        for (uint32_t i = 0; i < kMaxBufferBindings; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
        }

        std::array<VkDescriptorBindingFlags, kMaxBufferBindings> bindingFlags{};
        bindingFlags.fill(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        flagsInfo.bindingCount = kMaxBufferBindings;
        flagsInfo.pBindingFlags = bindingFlags.data();

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.pNext = &flagsInfo;
        layoutInfo.bindingCount = kMaxBufferBindings;
        layoutInfo.pBindings = bindings.data();
        checkVk(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_setLayouts[kDescriptorSetBuffers]),
                "Failed to create buffer descriptor set layout");
    }

    // Set 1: Sampled/storage images (binding 0..kMaxTextureBindings-1)
    {
        std::array<VkDescriptorSetLayoutBinding, kMaxTextureBindings> bindings{};
        for (uint32_t i = 0; i < kMaxTextureBindings; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
        }

        std::array<VkDescriptorBindingFlags, kMaxTextureBindings> bindingFlags{};
        bindingFlags.fill(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        flagsInfo.bindingCount = kMaxTextureBindings;
        flagsInfo.pBindingFlags = bindingFlags.data();

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.pNext = &flagsInfo;
        layoutInfo.bindingCount = kMaxTextureBindings;
        layoutInfo.pBindings = bindings.data();
        checkVk(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_setLayouts[kDescriptorSetTextures]),
                "Failed to create texture descriptor set layout");
    }

    // Set 2: Samplers (binding 0..kMaxSamplerBindings-1)
    {
        std::array<VkDescriptorSetLayoutBinding, kMaxSamplerBindings> bindings{};
        for (uint32_t i = 0; i < kMaxSamplerBindings; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_ALL;
        }

        std::array<VkDescriptorBindingFlags, kMaxSamplerBindings> bindingFlags{};
        bindingFlags.fill(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        flagsInfo.bindingCount = kMaxSamplerBindings;
        flagsInfo.pBindingFlags = bindingFlags.data();

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.pNext = &flagsInfo;
        layoutInfo.bindingCount = kMaxSamplerBindings;
        layoutInfo.pBindings = bindings.data();
        checkVk(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_setLayouts[kDescriptorSetSamplers]),
                "Failed to create sampler descriptor set layout");
    }
}

void VulkanDescriptorManager::createPipelineLayout() {
    // Push constant range: 256 bytes for all stages (covers vertex/fragment/mesh/compute bytes)
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_ALL;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 256;

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = kDescriptorSetCount;
    layoutInfo.pSetLayouts = m_setLayouts.data();
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    checkVk(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout),
            "Failed to create shared pipeline layout");
}

void VulkanDescriptorManager::createPool() {
    std::array<VkDescriptorPoolSize, 4> poolSizes = {{
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSetsPerFrame * kMaxBufferBindings},
        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kMaxSetsPerFrame * kMaxTextureBindings},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxSetsPerFrame * 32},
        {VK_DESCRIPTOR_TYPE_SAMPLER, kMaxSetsPerFrame * kMaxSamplerBindings},
    }};

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = kMaxSetsPerFrame * kDescriptorSetCount;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    checkVk(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_pool),
            "Failed to create frame descriptor pool");
}

void VulkanDescriptorManager::flushAndBind(
    VkCommandBuffer cmd, VkPipelineBindPoint bindPoint,
    const std::array<PendingBufferBinding, kMaxBufferBindings>& buffers,
    const std::array<PendingTextureBinding, kMaxTextureBindings>& textures,
    const std::array<PendingSamplerBinding, kMaxSamplerBindings>& samplers) {

    // Allocate 3 descriptor sets
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = m_pool;
    allocInfo.descriptorSetCount = kDescriptorSetCount;
    allocInfo.pSetLayouts = m_setLayouts.data();

    std::array<VkDescriptorSet, kDescriptorSetCount> sets{};
    VkResult result = vkAllocateDescriptorSets(m_device, &allocInfo, sets.data());
    if (result != VK_SUCCESS) {
        spdlog::warn("Failed to allocate descriptor sets, skipping bind");
        return;
    }

    // Collect writes
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(32);

    // Buffer info storage (must outlive vkUpdateDescriptorSets)
    std::vector<VkDescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(kMaxBufferBindings);

    for (uint32_t i = 0; i < kMaxBufferBindings; ++i) {
        if (buffers[i].buffer == VK_NULL_HANDLE) continue;

        bufferInfos.push_back({buffers[i].buffer, buffers[i].offset, buffers[i].range});

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = sets[kDescriptorSetBuffers];
        write.dstBinding = i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfos.back();
        writes.push_back(write);
    }

    // Image info storage
    std::vector<VkDescriptorImageInfo> imageInfos;
    imageInfos.reserve(kMaxTextureBindings);

    for (uint32_t i = 0; i < kMaxTextureBindings; ++i) {
        if (textures[i].imageView == VK_NULL_HANDLE) continue;

        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView = textures[i].imageView;
        imgInfo.imageLayout = textures[i].isStorage
            ? VK_IMAGE_LAYOUT_GENERAL
            : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfos.push_back(imgInfo);

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = sets[kDescriptorSetTextures];
        write.dstBinding = i;
        write.descriptorCount = 1;
        write.descriptorType = textures[i].isStorage
            ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
            : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &imageInfos.back();
        writes.push_back(write);
    }

    // Sampler info storage
    std::vector<VkDescriptorImageInfo> samplerInfos;
    samplerInfos.reserve(kMaxSamplerBindings);

    for (uint32_t i = 0; i < kMaxSamplerBindings; ++i) {
        if (samplers[i].sampler == VK_NULL_HANDLE) continue;

        VkDescriptorImageInfo samplerInfo{};
        samplerInfo.sampler = samplers[i].sampler;
        samplerInfos.push_back(samplerInfo);

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = sets[kDescriptorSetSamplers];
        write.dstBinding = i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
        write.pImageInfo = &samplerInfos.back();
        writes.push_back(write);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    vkCmdBindDescriptorSets(cmd, bindPoint, m_pipelineLayout,
                            0, kDescriptorSetCount, sets.data(), 0, nullptr);
}

#endif // _WIN32
