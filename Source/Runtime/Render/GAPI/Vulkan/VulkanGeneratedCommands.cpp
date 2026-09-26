#include "Runtime/Render/GAPI/Vulkan/VulkanGeneratedCommands.h"

#include <algorithm>
#include <limits>

namespace metallic::render::vulkan {
namespace {

Result<> convertResult(VkResult result)
{
    if (result == VK_SUCCESS) { return {}; }
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        return makeError(Error::OutOfMemory);
    }
    if (result == VK_ERROR_DEVICE_LOST) { return makeError(Error::DeviceLost); }
    if (result == VK_ERROR_FEATURE_NOT_PRESENT || result == VK_ERROR_EXTENSION_NOT_PRESENT) {
        return makeError(Error::Unsupported);
    }
    return makeError(Error::Failure);
}

} // namespace

Result<VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT> queryGeneratedCommandsProperties(Device& device)
{
    VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT};
    const auto native = nativeDevice(device);
    if (!native.device) { return makeError(Error::InvalidArgument); }
    if (!device.capabilities().deviceGeneratedCommands) { return makeError(Error::Unsupported); }
    VkPhysicalDeviceProperties2 query{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &properties};
    vkGetPhysicalDeviceProperties2(native.physicalDevice, &query);
    return properties;
}

struct GeneratedCommands::Impl {
    NativeDevice device;
    // Keep per-device entry points; other RHI devices may change volk's globals.
    VolkDeviceTable vk{};
    VkIndirectCommandsLayoutEXT layout = VK_NULL_HANDLE;
    VkIndirectExecutionSetEXT executionSet = VK_NULL_HANDLE;
    VkIndirectExecutionSetInfoTypeEXT setType = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
    uint32_t setCapacity = 0;
    VkShaderStageFlags stages = 0;
    uint32_t stride = 0;
    uint32_t maxSequences = 0;
    uint32_t maxDraws = 0;
    bool explicitPreprocess = false;
    bool ready = false;
    VkGeneratedCommandsPipelineInfoEXT pipelineInfo{.sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT};
    std::vector<VkShaderEXT> shaders;
    VkGeneratedCommandsShaderInfoEXT shaderInfo{.sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_SHADER_INFO_EXT};
    VkMemoryRequirements requirements{};
    VkBuffer scratch = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;

    ~Impl()
    {
        releaseScratch();
        if (executionSet) { vk.vkDestroyIndirectExecutionSetEXT(device.device, executionSet, nullptr); }
        if (layout) { vk.vkDestroyIndirectCommandsLayoutEXT(device.device, layout, nullptr); }
    }

    void releaseScratch()
    {
        if (scratch) { vk.vkDestroyBuffer(device.device, scratch, nullptr); }
        if (memory) { vk.vkFreeMemory(device.device, memory, nullptr); }
        scratch = VK_NULL_HANDLE;
        memory = VK_NULL_HANDLE;
        address = 0;
        ready = false;
    }

    const void* fixedState() const
    {
        if (executionSet) { return nullptr; }
        return pipelineInfo.pipeline ? static_cast<const void*>(&pipelineInfo) : &shaderInfo;
    }

    bool owns(CommandBuffer& commands) const
    {
        return nativeCommandBufferDevice(commands) == device.device;
    }

    bool supportsQueue(CommandBuffer& commands) const
    {
        const auto capabilities = commands.queueCapabilities();
        constexpr VkShaderStageFlags computeOrRayTracing = VK_SHADER_STAGE_COMPUTE_BIT |
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
            VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CALLABLE_BIT_KHR;
        return (stages & computeOrRayTracing) != 0
            ? hasFlag(capabilities, QueueAccessBits::Compute)
            : hasFlag(capabilities, QueueAccessBits::Graphics);
    }

    Result<> fillInfo(CommandBuffer& commands, const GeneratedCommandsArguments& args, VkGeneratedCommandsInfoEXT& info)
    {
        if (!ready || !owns(commands) || !supportsQueue(commands) || !args.commands || args.sequenceCount == 0 ||
            args.sequenceCount > maxSequences || (args.offset & 3) != 0 ||
            !hasFlag(args.commands->desc().usage, BufferUsageBits::Indirect)) {
            return makeError(Error::InvalidArgument);
        }
        const auto buffer = nativeBuffer(*args.commands);
        if (buffer.device != device.device || !buffer.address || args.offset >= buffer.size) {
            return makeError(Error::InvalidArgument);
        }
        const uint64_t size = args.size ? args.size : buffer.size - args.offset;
        if (size > buffer.size - args.offset || size < uint64_t(stride) * args.sequenceCount) {
            return makeError(Error::InvalidArgument);
        }
        VkDeviceAddress countAddress = 0;
        if (args.countBuffer) {
            const auto count = nativeBuffer(*args.countBuffer);
            if (!hasFlag(args.countBuffer->desc().usage, BufferUsageBits::Indirect) ||
                count.device != device.device || !count.address || (args.countOffset & 3) != 0 ||
                args.countOffset > count.size || sizeof(uint32_t) > count.size - args.countOffset) {
                return makeError(Error::InvalidArgument);
            }
            countAddress = count.address + args.countOffset;
        }
        info = {
            .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT,
            .pNext = fixedState(),
            .shaderStages = stages,
            .indirectExecutionSet = executionSet,
            .indirectCommandsLayout = layout,
            .indirectAddress = buffer.address + args.offset,
            .indirectAddressSize = size,
            .preprocessAddress = address,
            .preprocessSize = requirements.size,
            .maxSequenceCount = args.sequenceCount,
            .sequenceCountAddress = countAddress,
            .maxDrawCount = maxDraws,
        };
        return {};
    }
};

GeneratedCommands::GeneratedCommands() = default;
GeneratedCommands::~GeneratedCommands() = default;
GeneratedCommands::GeneratedCommands(GeneratedCommands&&) noexcept = default;
GeneratedCommands& GeneratedCommands::operator=(GeneratedCommands&&) noexcept = default;

void GeneratedCommands::reset()
{
    impl_.reset();
}

Result<> GeneratedCommands::initialize(Device& device, const GeneratedCommandsDesc& desc)
{
    reset();
    const auto queriedProperties = queryGeneratedCommandsProperties(device);
    if (!queriedProperties) { return makeError(queriedProperties.error()); }
    const auto& properties = *queriedProperties;
    Result<> result;
    if (!desc.layout.pTokens || desc.layout.tokenCount == 0 || desc.maxSequenceCount == 0 ||
        desc.layout.shaderStages == 0 || desc.layout.indirectStride == 0 || (desc.layout.indirectStride & 3) != 0 ||
        desc.layout.tokenCount > properties.maxIndirectCommandsTokenCount ||
        desc.layout.indirectStride > properties.maxIndirectCommandsIndirectStride ||
        desc.maxSequenceCount > properties.maxIndirectSequenceCount) {
        return makeError(Error::InvalidArgument);
    }
    if ((desc.layout.shaderStages & properties.supportedIndirectCommandsShaderStages) != desc.layout.shaderStages) {
        return makeError(Error::Unsupported);
    }
    bool hasExecutionSet = false;
    bool hasCountToken = false;
    for (uint32_t i = 0; i < desc.layout.tokenCount; ++i) {
        const auto& token = desc.layout.pTokens[i];
        if (token.offset > properties.maxIndirectCommandsTokenOffset || token.offset >= desc.layout.indirectStride ||
            (token.offset & 3) != 0) { return makeError(Error::InvalidArgument); }
        if (token.type == VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT) {
            if (i != 0 || !token.data.pExecutionSet || !desc.executionSet ||
                token.data.pExecutionSet->type != desc.executionSet->type) {
                return makeError(Error::InvalidArgument);
            }
            hasExecutionSet = true;
        }
        hasCountToken |= token.type == VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_COUNT_EXT ||
            token.type == VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_INDEXED_COUNT_EXT ||
            token.type == VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_COUNT_EXT ||
            token.type == VK_INDIRECT_COMMANDS_TOKEN_TYPE_DRAW_MESH_TASKS_COUNT_NV_EXT;
    }
    if (hasCountToken && (!desc.maxDrawCount || uint64_t(desc.maxDrawCount) * desc.maxSequenceCount >= (1u << 24))) {
        return makeError(Error::InvalidArgument);
    }
    if (hasCountToken && !properties.deviceGeneratedCommandsMultiDrawIndirectCount) {
        return makeError(Error::Unsupported);
    }
    if (hasExecutionSet != (desc.executionSet != nullptr) ||
        (hasExecutionSet && (desc.pipeline || !desc.shaders.empty())) ||
        (!hasExecutionSet && (bool(desc.pipeline) == !desc.shaders.empty()))) {
        return makeError(Error::InvalidArgument);
    }
    auto impl = std::make_unique<Impl>();
    impl->device = nativeDevice(device);
    volkLoadDeviceTable(&impl->vk, impl->device.device);
    if (!impl->vk.vkCreateIndirectCommandsLayoutEXT || !impl->vk.vkCreateIndirectExecutionSetEXT ||
        !impl->vk.vkGetGeneratedCommandsMemoryRequirementsEXT || !impl->vk.vkCmdExecuteGeneratedCommandsEXT ||
        !impl->vk.vkCmdPreprocessGeneratedCommandsEXT) { return makeError(Error::Unsupported); }
    impl->stages = desc.layout.shaderStages;
    impl->stride = desc.layout.indirectStride;
    impl->maxSequences = desc.maxSequenceCount;
    impl->maxDraws = desc.maxDrawCount;
    impl->explicitPreprocess = (desc.layout.flags & VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT) != 0;
    impl->pipelineInfo.pipeline = desc.pipeline;
    impl->shaders.assign(desc.shaders.begin(), desc.shaders.end());
    impl->shaderInfo.shaderCount = static_cast<uint32_t>(impl->shaders.size());
    impl->shaderInfo.pShaders = impl->shaders.data();
    if (desc.executionSet) {
        impl->setType = desc.executionSet->type;
        if (impl->setType == VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT) {
            const auto* info = desc.executionSet->info.pPipelineInfo;
            if (!info || !info->initialPipeline || !info->maxPipelineCount ||
                info->maxPipelineCount > properties.maxIndirectPipelineCount) { return makeError(Error::InvalidArgument); }
            impl->setCapacity = info->maxPipelineCount;
        } else if (impl->setType == VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT) {
            const auto* info = desc.executionSet->info.pShaderInfo;
            if (!info || !info->pInitialShaders || !info->shaderCount || info->shaderCount > info->maxShaderCount ||
                info->maxShaderCount > properties.maxIndirectShaderObjectCount) { return makeError(Error::InvalidArgument); }
            impl->setCapacity = info->maxShaderCount;
        } else { return makeError(Error::InvalidArgument); }
        result = convertResult(impl->vk.vkCreateIndirectExecutionSetEXT(
            impl->device.device, desc.executionSet, nullptr, &impl->executionSet));
        if (!result) { return result; }
    }
    result = convertResult(impl->vk.vkCreateIndirectCommandsLayoutEXT(impl->device.device, &desc.layout, nullptr, &impl->layout));
    if (!result) { return result; }
    impl_ = std::move(impl);
    result = prepare();
    if (!result) { reset(); }
    return result;
}

Result<> GeneratedCommands::prepare()
{
    if (!impl_) { return makeError(Error::InvalidArgument); }
    auto& impl = *impl_;
    impl.releaseScratch();
    const VkGeneratedCommandsMemoryRequirementsInfoEXT query{
        .sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT,
        .pNext = impl.fixedState(),
        .indirectExecutionSet = impl.executionSet,
        .indirectCommandsLayout = impl.layout,
        .maxSequenceCount = impl.maxSequences,
        .maxDrawCount = impl.maxDraws,
    };
    VkMemoryRequirements2 requirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    impl.vk.vkGetGeneratedCommandsMemoryRequirementsEXT(impl.device.device, &query, &requirements);
    impl.requirements = requirements.memoryRequirements;
    if (!impl.requirements.size) { impl.ready = true; return {}; }
    const uint64_t alignment = std::max<uint64_t>(1, impl.requirements.alignment);
    if (impl.requirements.size > std::numeric_limits<uint64_t>::max() - alignment + 1) {
        return makeError(Error::OutOfMemory);
    }
    const VkBufferUsageFlags2CreateInfo usage{
        .sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
        .usage = VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
    };
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &usage,
        .size = impl.requirements.size + alignment - 1, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    auto result = convertResult(impl.vk.vkCreateBuffer(impl.device.device, &bufferInfo, nullptr, &impl.scratch));
    if (!result) { return result; }
    VkMemoryDedicatedRequirements dedicated{.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 bufferRequirements{.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
    const VkBufferMemoryRequirementsInfo2 bufferQuery{.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2, .buffer = impl.scratch};
    impl.vk.vkGetBufferMemoryRequirements2(impl.device.device, &bufferQuery, &bufferRequirements);
    const uint32_t allowedTypes = bufferRequirements.memoryRequirements.memoryTypeBits & impl.requirements.memoryTypeBits;
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(impl.device.physicalDevice, &memoryProperties);
    uint32_t memoryType = UINT32_MAX;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((allowedTypes & (1u << i)) == 0) { continue; }
        if ((memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_PROTECTED_BIT) != 0) { continue; }
        memoryType = i;
        if ((memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) { break; }
    }
    if (memoryType == UINT32_MAX) { return makeError(Error::Unsupported); }
    VkMemoryDedicatedAllocateInfo dedicatedAllocation{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = impl.scratch};
    const VkMemoryAllocateFlagsInfo flags{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
        .pNext = dedicated.requiresDedicatedAllocation ? &dedicatedAllocation : nullptr,
        .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    const VkMemoryAllocateInfo allocation{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &flags,
        .allocationSize = bufferRequirements.memoryRequirements.size, .memoryTypeIndex = memoryType};
    result = convertResult(impl.vk.vkAllocateMemory(impl.device.device, &allocation, nullptr, &impl.memory));
    if (!result) { return result; }
    result = convertResult(impl.vk.vkBindBufferMemory(impl.device.device, impl.scratch, impl.memory, 0));
    if (!result) { return result; }
    const VkBufferDeviceAddressInfo addressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = impl.scratch};
    const uint64_t base = impl.vk.vkGetBufferDeviceAddress(impl.device.device, &addressInfo);
    if (!base) { return makeError(Error::Failure); }
    impl.address = base + (alignment - base % alignment) % alignment;
    impl.ready = true;
    return {};
}

VkMemoryRequirements GeneratedCommands::memoryRequirements() const
{
    return impl_ ? impl_->requirements : VkMemoryRequirements{};
}

Result<> GeneratedCommands::updatePipelines(std::span<const VkWriteIndirectExecutionSetPipelineEXT> writes)
{
    if (!impl_ || !impl_->executionSet || impl_->setType != VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT ||
        writes.empty() || writes.size() > impl_->setCapacity) { return makeError(Error::InvalidArgument); }
    for (size_t i = 0; i < writes.size(); ++i) {
        if (!writes[i].pipeline || writes[i].index >= impl_->setCapacity) { return makeError(Error::InvalidArgument); }
        for (size_t j = 0; j < i; ++j) {
            if (writes[i].index == writes[j].index) { return makeError(Error::InvalidArgument); }
        }
    }
    impl_->vk.vkUpdateIndirectExecutionSetPipelineEXT(impl_->device.device, impl_->executionSet,
        static_cast<uint32_t>(writes.size()), writes.data());
    impl_->ready = false;
    return {};
}

Result<> GeneratedCommands::updateShaders(std::span<const VkWriteIndirectExecutionSetShaderEXT> writes)
{
    if (!impl_ || !impl_->executionSet || impl_->setType != VK_INDIRECT_EXECUTION_SET_INFO_TYPE_SHADER_OBJECTS_EXT ||
        writes.empty() || writes.size() > impl_->setCapacity) { return makeError(Error::InvalidArgument); }
    for (size_t i = 0; i < writes.size(); ++i) {
        if (!writes[i].shader || writes[i].index >= impl_->setCapacity) { return makeError(Error::InvalidArgument); }
        for (size_t j = 0; j < i; ++j) {
            if (writes[i].index == writes[j].index) { return makeError(Error::InvalidArgument); }
        }
    }
    impl_->vk.vkUpdateIndirectExecutionSetShaderEXT(impl_->device.device, impl_->executionSet,
        static_cast<uint32_t>(writes.size()), writes.data());
    impl_->ready = false;
    return {};
}

Result<> GeneratedCommands::preprocess(CommandBuffer& commands, const GeneratedCommandsArguments& args, CommandBuffer& state)
{
    if (!impl_ || !impl_->explicitPreprocess || !impl_->owns(state) || !impl_->supportsQueue(state)) {
        return makeError(Error::InvalidArgument);
    }
    VkGeneratedCommandsInfoEXT info{};
    auto result = impl_->fillInfo(commands, args, info);
    if (!result) { return result; }
    impl_->vk.vkCmdPreprocessGeneratedCommandsEXT(nativeCommandBuffer(commands), &info, nativeCommandBuffer(state));
    return {};
}

Result<> GeneratedCommands::preprocessBarrier(CommandBuffer& commands)
{
    if (!impl_ || !impl_->explicitPreprocess || !impl_->owns(commands) || !impl_->supportsQueue(commands)) {
        return makeError(Error::InvalidArgument);
    }
    const VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT,
        .srcAccessMask = VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_EXT,
        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
    };
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
    impl_->vk.vkCmdPipelineBarrier2(nativeCommandBuffer(commands), &dependency);
    return {};
}

Result<> GeneratedCommands::execute(CommandBuffer& commands, const GeneratedCommandsArguments& args, bool isPreprocessed)
{
    if (!impl_ || isPreprocessed != impl_->explicitPreprocess) { return makeError(Error::InvalidArgument); }
    VkGeneratedCommandsInfoEXT info{};
    auto result = impl_->fillInfo(commands, args, info);
    if (!result) { return result; }
    impl_->vk.vkCmdExecuteGeneratedCommandsEXT(nativeCommandBuffer(commands), isPreprocessed, &info);
    notifyGeneratedCommandsExecution(commands);
    return {};
}

} // namespace metallic::render::vulkan
