#include "vulkan_rt_pipeline.h"

#ifdef _WIN32

#include "vulkan_backend.h"
#include "vulkan_descriptor_manager.h"
#include "rhi_backend.h"

#include <vk_mem_alloc.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace {

// -------------------------------------------------------------------------
// Dynamically loaded RT pipeline function pointers.
// -------------------------------------------------------------------------

struct RTPipelineFunctions {
    PFN_vkCreateRayTracingPipelinesKHR       createRayTracingPipelines = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR getShaderGroupHandles    = nullptr;
    PFN_vkCmdTraceRaysKHR                    cmdTraceRays             = nullptr;
    PFN_vkGetBufferDeviceAddress             getBufferDeviceAddress   = nullptr;

    bool valid() const {
        return createRayTracingPipelines && getShaderGroupHandles &&
               cmdTraceRays && getBufferDeviceAddress;
    }
};

RTPipelineFunctions loadRTPipelineFunctions(VkDevice device) {
    RTPipelineFunctions f{};
    f.createRayTracingPipelines =
        reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
            vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
    f.getShaderGroupHandles =
        reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
            vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR"));
    f.cmdTraceRays =
        reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
            vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));
    f.getBufferDeviceAddress =
        reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
            vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddress"));
    if (!f.getBufferDeviceAddress) {
        f.getBufferDeviceAddress =
            reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
                vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR"));
    }
    return f;
}

// Cached per-device (populated on first use).
RTPipelineFunctions g_rtFunctions{};
VkDevice            g_rtFunctionsDevice = VK_NULL_HANDLE;

const RTPipelineFunctions& ensureRTFunctions(VkDevice device) {
    if (g_rtFunctionsDevice != device) {
        g_rtFunctions = loadRTPipelineFunctions(device);
        g_rtFunctionsDevice = device;
    }
    return g_rtFunctions;
}

uint32_t alignUp(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

} // namespace

// -------------------------------------------------------------------------
// VulkanRayTracingPipeline lifetime
// -------------------------------------------------------------------------

VulkanRayTracingPipeline::~VulkanRayTracingPipeline() {
    if (m_sbtBuffer != VK_NULL_HANDLE && m_allocator) {
        vmaDestroyBuffer(m_allocator, m_sbtBuffer, m_sbtAllocation);
    }
    if (m_pipeline != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
    }
    if (m_layout != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_layout, nullptr);
    }
}

VulkanRayTracingPipeline::VulkanRayTracingPipeline(VulkanRayTracingPipeline&& other) noexcept
    : m_device(other.m_device)
    , m_pipeline(other.m_pipeline)
    , m_layout(other.m_layout)
    , m_sbtBuffer(other.m_sbtBuffer)
    , m_sbtAllocation(other.m_sbtAllocation)
    , m_allocator(other.m_allocator)
    , m_raygenRegion(other.m_raygenRegion)
    , m_missRegion(other.m_missRegion)
    , m_hitRegion(other.m_hitRegion)
    , m_callableRegion(other.m_callableRegion) {
    other.m_device = VK_NULL_HANDLE;
    other.m_pipeline = VK_NULL_HANDLE;
    other.m_layout = VK_NULL_HANDLE;
    other.m_sbtBuffer = VK_NULL_HANDLE;
    other.m_sbtAllocation = nullptr;
    other.m_allocator = nullptr;
    other.m_raygenRegion = {};
    other.m_missRegion = {};
    other.m_hitRegion = {};
    other.m_callableRegion = {};
}

VulkanRayTracingPipeline& VulkanRayTracingPipeline::operator=(VulkanRayTracingPipeline&& other) noexcept {
    if (this != &other) {
        this->~VulkanRayTracingPipeline();
        new (this) VulkanRayTracingPipeline(std::move(other));
    }
    return *this;
}

// -------------------------------------------------------------------------
// Pipeline + SBT creation
// -------------------------------------------------------------------------

std::unique_ptr<VulkanRayTracingPipeline>
createVulkanRayTracingPipelineImpl(VkDevice device,
                                   VkPhysicalDevice /*physicalDevice*/,
                                   VmaAllocator allocator,
                                   VkPipelineCache pipelineCache,
                                   const RtPipelineDesc& desc,
                                   uint32_t handleSize,
                                   uint32_t handleAlignment,
                                   uint32_t baseAlignment,
                                   std::string& errorMessage) {
    const auto& rtFn = ensureRTFunctions(device);
    if (!rtFn.valid()) {
        errorMessage = "Failed to load RT pipeline function pointers";
        return nullptr;
    }

    if (!desc.shaderCode || desc.shaderCodeSize == 0) {
        errorMessage = "RT pipeline requires a non-empty SPIR-V blob";
        return nullptr;
    }

    // --- 0. Build pipeline layout (bindless set + push constants) ---

    VkDescriptorSetLayout bindlessLayout = VK_NULL_HANDLE;
    if (!vulkanRetainBindlessSetLayout(device, bindlessLayout, &errorMessage)) {
        return nullptr;
    }

    constexpr uint32_t kPushConstantSize = 256;
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_ALL;
    pushRange.offset = 0;
    pushRange.size = kPushConstantSize;

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        vulkanReleaseBindlessSetLayout(device);
        errorMessage = "Failed to create RT pipeline layout";
        return nullptr;
    }

    // --- 1. Create shader module ---

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = desc.shaderCodeSize;
    moduleInfo.pCode = static_cast<const uint32_t*>(desc.shaderCode);

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &moduleInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        errorMessage = "Failed to create VkShaderModule for RT pipeline";
        return nullptr;
    }

    // --- 2. Build shader stages + groups ---

    // Collect all unique entry points → stage array
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    auto addStage = [&](VkShaderStageFlagBits stage, const char* entry) -> uint32_t {
        VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stageInfo.stage = stage;
        stageInfo.module = shaderModule;
        stageInfo.pName = entry;
        uint32_t index = static_cast<uint32_t>(stages.size());
        stages.push_back(stageInfo);
        return index;
    };

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups;

    // Raygen (exactly one)
    {
        uint32_t rgenStageIdx = addStage(VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                                          desc.rayGenEntry.c_str());
        VkRayTracingShaderGroupCreateInfoKHR group{
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        group.generalShader = rgenStageIdx;
        group.closestHitShader = VK_SHADER_UNUSED_KHR;
        group.anyHitShader = VK_SHADER_UNUSED_KHR;
        group.intersectionShader = VK_SHADER_UNUSED_KHR;
        groups.push_back(group);
    }

    const uint32_t missGroupOffset = static_cast<uint32_t>(groups.size());

    // Miss groups
    for (const auto& missEntry : desc.missEntries) {
        uint32_t missStageIdx = addStage(VK_SHADER_STAGE_MISS_BIT_KHR, missEntry.c_str());
        VkRayTracingShaderGroupCreateInfoKHR group{
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        group.generalShader = missStageIdx;
        group.closestHitShader = VK_SHADER_UNUSED_KHR;
        group.anyHitShader = VK_SHADER_UNUSED_KHR;
        group.intersectionShader = VK_SHADER_UNUSED_KHR;
        groups.push_back(group);
    }

    const uint32_t hitGroupOffset = static_cast<uint32_t>(groups.size());

    // Hit groups
    for (const auto& hg : desc.hitGroups) {
        VkRayTracingShaderGroupCreateInfoKHR group{
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};

        if (hg.type == RtShaderGroupType::ProceduralHit) {
            group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        } else {
            group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        }

        group.generalShader = VK_SHADER_UNUSED_KHR;
        group.closestHitShader = VK_SHADER_UNUSED_KHR;
        group.anyHitShader = VK_SHADER_UNUSED_KHR;
        group.intersectionShader = VK_SHADER_UNUSED_KHR;

        if (!hg.closestHitEntry.empty()) {
            group.closestHitShader = addStage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                               hg.closestHitEntry.c_str());
        }
        if (!hg.anyHitEntry.empty()) {
            group.anyHitShader = addStage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                                           hg.anyHitEntry.c_str());
        }
        if (!hg.intersectionEntry.empty()) {
            group.intersectionShader = addStage(VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                                                 hg.intersectionEntry.c_str());
        }

        groups.push_back(group);
    }

    const uint32_t callableGroupOffset = static_cast<uint32_t>(groups.size());

    // Callable groups
    for (const auto& callableEntry : desc.callableEntries) {
        uint32_t callableStageIdx = addStage(VK_SHADER_STAGE_CALLABLE_BIT_KHR,
                                              callableEntry.c_str());
        VkRayTracingShaderGroupCreateInfoKHR group{
            VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        group.generalShader = callableStageIdx;
        group.closestHitShader = VK_SHADER_UNUSED_KHR;
        group.anyHitShader = VK_SHADER_UNUSED_KHR;
        group.intersectionShader = VK_SHADER_UNUSED_KHR;
        groups.push_back(group);
    }

    // --- 3. Create RT pipeline ---

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{
        VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(groups.size());
    pipelineInfo.pGroups = groups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = desc.maxRecursionDepth;
    pipelineInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto t0 = std::chrono::high_resolution_clock::now();
    const VkResult result = rtFn.createRayTracingPipelines(
        device, VK_NULL_HANDLE, pipelineCache, 1, &pipelineInfo, nullptr, &pipeline);
    const double compileMs = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    vkDestroyShaderModule(device, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        errorMessage = "vkCreateRayTracingPipelinesKHR failed (VkResult: " +
                       std::to_string(result) + ")";
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        return nullptr;
    }

    spdlog::debug("VulkanRTPipeline: compiled in {:.1f} ms ({} stages, {} groups)",
                  compileMs, stages.size(), groups.size());

    // --- 4. Retrieve shader group handles ---

    const uint32_t totalGroupCount = static_cast<uint32_t>(groups.size());
    const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);
    std::vector<uint8_t> handleData(totalGroupCount * handleSize);

    if (rtFn.getShaderGroupHandles(device, pipeline, 0, totalGroupCount,
                                    handleData.size(), handleData.data()) != VK_SUCCESS) {
        errorMessage = "vkGetRayTracingShaderGroupHandlesKHR failed";
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        return nullptr;
    }

    // --- 5. Build SBT buffer ---

    // Region sizes (rounded up to shaderGroupBaseAlignment)
    const uint32_t raygenCount    = 1;
    const uint32_t missCount      = static_cast<uint32_t>(desc.missEntries.size());
    const uint32_t hitCount       = static_cast<uint32_t>(desc.hitGroups.size());
    const uint32_t callableCount  = static_cast<uint32_t>(desc.callableEntries.size());

    const uint32_t raygenRegionSize   = alignUp(handleSizeAligned * raygenCount, baseAlignment);
    const uint32_t missRegionSize     = missCount > 0 ? alignUp(handleSizeAligned * missCount, baseAlignment) : 0;
    const uint32_t hitRegionSize      = hitCount > 0 ? alignUp(handleSizeAligned * hitCount, baseAlignment) : 0;
    const uint32_t callableRegionSize = callableCount > 0 ? alignUp(handleSizeAligned * callableCount, baseAlignment) : 0;

    const VkDeviceSize sbtSize = raygenRegionSize + missRegionSize + hitRegionSize + callableRegionSize;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = sbtSize;
    bufferInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                      VMA_ALLOCATION_CREATE_MAPPED_BIT;
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VkBuffer sbtBuffer = VK_NULL_HANDLE;
    VmaAllocation sbtAllocation = nullptr;
    VmaAllocationInfo sbtAllocInfo{};
    if (vmaCreateBuffer(allocator, &bufferInfo, &allocInfo,
                        &sbtBuffer, &sbtAllocation, &sbtAllocInfo) != VK_SUCCESS) {
        errorMessage = "Failed to allocate SBT buffer";
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        return nullptr;
    }

    // --- 6. Fill SBT ---

    auto* mapped = static_cast<uint8_t*>(sbtAllocInfo.pMappedData);
    std::memset(mapped, 0, sbtSize);

    auto copyHandles = [&](uint32_t groupStart, uint32_t groupCount, uint32_t regionOffset) {
        for (uint32_t i = 0; i < groupCount; ++i) {
            const uint8_t* src = handleData.data() + (groupStart + i) * handleSize;
            uint8_t* dst = mapped + regionOffset + i * handleSizeAligned;
            std::memcpy(dst, src, handleSize);
        }
    };

    uint32_t offset = 0;
    copyHandles(0, raygenCount, offset);
    offset += raygenRegionSize;

    if (missCount > 0) {
        copyHandles(missGroupOffset, missCount, offset);
    }
    offset += missRegionSize;

    if (hitCount > 0) {
        copyHandles(hitGroupOffset, hitCount, offset);
    }
    offset += hitRegionSize;

    if (callableCount > 0) {
        copyHandles(callableGroupOffset, callableCount, offset);
    }

    vmaFlushAllocation(allocator, sbtAllocation, 0, sbtSize);

    // --- 7. Compute SBT device address regions ---

    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = sbtBuffer;
    const VkDeviceAddress sbtAddress = rtFn.getBufferDeviceAddress(device, &addressInfo);

    auto result_pipeline = std::make_unique<VulkanRayTracingPipeline>();
    result_pipeline->m_device = device;
    result_pipeline->m_pipeline = pipeline;
    result_pipeline->m_layout = pipelineLayout;
    result_pipeline->m_sbtBuffer = sbtBuffer;
    result_pipeline->m_sbtAllocation = sbtAllocation;
    result_pipeline->m_allocator = allocator;

    VkDeviceAddress regionAddr = sbtAddress;

    result_pipeline->m_raygenRegion.deviceAddress = regionAddr;
    result_pipeline->m_raygenRegion.stride = handleSizeAligned;
    result_pipeline->m_raygenRegion.size = raygenRegionSize;
    regionAddr += raygenRegionSize;

    result_pipeline->m_missRegion.deviceAddress = missCount > 0 ? regionAddr : 0;
    result_pipeline->m_missRegion.stride = missCount > 0 ? handleSizeAligned : 0;
    result_pipeline->m_missRegion.size = missRegionSize;
    regionAddr += missRegionSize;

    result_pipeline->m_hitRegion.deviceAddress = hitCount > 0 ? regionAddr : 0;
    result_pipeline->m_hitRegion.stride = hitCount > 0 ? handleSizeAligned : 0;
    result_pipeline->m_hitRegion.size = hitRegionSize;
    regionAddr += hitRegionSize;

    result_pipeline->m_callableRegion.deviceAddress = callableCount > 0 ? regionAddr : 0;
    result_pipeline->m_callableRegion.stride = callableCount > 0 ? handleSizeAligned : 0;
    result_pipeline->m_callableRegion.size = callableRegionSize;

    spdlog::info("VulkanRTPipeline: SBT {} bytes (raygen={}, miss={}, hit={}, callable={})",
                 sbtSize, raygenRegionSize, missRegionSize, hitRegionSize, callableRegionSize);

    return result_pipeline;
}

// -------------------------------------------------------------------------
// Public API
// -------------------------------------------------------------------------

std::unique_ptr<VulkanRayTracingPipeline> createVulkanRayTracingPipeline(
    RhiContext& context,
    const RtPipelineDesc& desc,
    std::string& errorMessage) {

    if (!context.features().rayTracingPipeline) {
        errorMessage = "RT pipeline feature not available on this device";
        return nullptr;
    }

    const auto& rtProps = context.rayTracingPipelineProperties();
    if (rtProps.shaderGroupHandleSize == 0) {
        errorMessage = "RT pipeline properties not populated";
        return nullptr;
    }

    VkDevice device = getVulkanDevice(context);
    VkPhysicalDevice physicalDevice = getVulkanPhysicalDevice(context);
    VmaAllocator allocator = getVulkanAllocator(context);
    VkPipelineCache pipelineCache = getVulkanPipelineCache(context);

    return createVulkanRayTracingPipelineImpl(
        device, physicalDevice, allocator, pipelineCache,
        desc, rtProps.shaderGroupHandleSize, rtProps.shaderGroupHandleAlignment,
        rtProps.shaderGroupBaseAlignment, errorMessage);
}

void vulkanTraceRays(RhiContext& context,
                     const VulkanRayTracingPipeline& pipeline,
                     uint32_t width,
                     uint32_t height,
                     uint32_t depth) {
    if (!pipeline.isValid()) {
        return;
    }

    VkDevice device = getVulkanDevice(context);
    const auto& rtFn = ensureRTFunctions(device);
    if (!rtFn.cmdTraceRays) {
        return;
    }

    VkCommandBuffer cmdBuf = getVulkanCurrentCommandBuffer(context);

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.pipeline());

    rtFn.cmdTraceRays(cmdBuf,
                      &pipeline.raygenRegion(),
                      &pipeline.missRegion(),
                      &pipeline.hitRegion(),
                      &pipeline.callableRegion(),
                      width, height, depth);
}

#endif // _WIN32
