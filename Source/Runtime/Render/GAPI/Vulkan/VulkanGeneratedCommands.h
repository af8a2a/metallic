#pragma once

#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

namespace metallic::render::vulkan {

// Native token descriptions intentionally preserve all EXT token types, including
// mesh draws and descriptor-heap push data. Vulkan layout compatibility rules apply.
struct GeneratedCommandsDesc {
    VkIndirectCommandsLayoutCreateInfoEXT layout{
        .sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT};
    const VkIndirectExecutionSetCreateInfoEXT* executionSet = nullptr;
    // Without an execution-set token, supply exactly one fixed pipeline or shader list.
    VkPipeline pipeline = VK_NULL_HANDLE;
    std::span<const VkShaderEXT> shaders;
    uint32_t maxSequenceCount = 1;
    uint32_t maxDrawCount = 1;
};

struct GeneratedCommandsArguments {
    Buffer* commands = nullptr;
    uint64_t offset = 0;
    uint64_t size = 0; // Zero uses the remaining buffer range.
    uint32_t sequenceCount = 1; // Upper bound when countBuffer is provided.
    Buffer* countBuffer = nullptr;
    uint64_t countOffset = 0;
};

Result queryGeneratedCommandsProperties(Device& device, VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT& properties);

// Device, referenced pipelines/shaders/layouts, and argument buffers must outlive
// submitted work. One instance owns one scratch allocation: synchronize reuse or
// use one instance per frame in flight. Updates/reset/destruction require idle use.
class GeneratedCommands {
public:
    GeneratedCommands();
    ~GeneratedCommands();
    GeneratedCommands(GeneratedCommands&&) noexcept;
    GeneratedCommands& operator=(GeneratedCommands&&) noexcept;
    GeneratedCommands(const GeneratedCommands&) = delete;
    GeneratedCommands& operator=(const GeneratedCommands&) = delete;

    Result initialize(Device& device, const GeneratedCommandsDesc& desc);
    void reset();
    Result updatePipelines(std::span<const VkWriteIndirectExecutionSetPipelineEXT> writes);
    Result updateShaders(std::span<const VkWriteIndirectExecutionSetShaderEXT> writes);
    // Requery and reallocate after execution-set updates, before recording commands.
    Result prepare();
    VkMemoryRequirements memoryRequirements() const;

    // Explicit preprocessing must be outside rendering. state must be recording
    // with the state that will be used for execution. Synchronize GPU-written
    // arguments to COMMAND_PREPROCESS/COMMAND_PREPROCESS_READ before this call.
    Result preprocess(CommandBuffer& commands, const GeneratedCommandsArguments& args, CommandBuffer& state);
    // Same-queue dependency between explicit preprocessing and execution.
    Result preprocessBarrier(CommandBuffer& commands);
    // Bind initial pipeline/shaders and all non-token state first. For graphics,
    // execute inside rendering. Rebind affected state after execution.
    Result execute(CommandBuffer& commands, const GeneratedCommandsArguments& args, bool isPreprocessed = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render::vulkan
