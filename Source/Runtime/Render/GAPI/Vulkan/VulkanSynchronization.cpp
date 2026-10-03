#include "VulkanSynchronization.h"

namespace metallic::render::vulkan {

VkImageLayout imageLayout(TextureLayout usage, bool unified)
{
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    switch (usage) {
    case TextureLayout::Undefined: layout = VK_IMAGE_LAYOUT_UNDEFINED; break;
    case TextureLayout::Present: layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; break;
    case TextureLayout::ColorAttachment: layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; break;
    case TextureLayout::DepthStencilAttachment: layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; break;
    case TextureLayout::ShaderRead: layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; break;
    case TextureLayout::TransferSource: layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; break;
    case TextureLayout::TransferDestination: layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; break;
    case TextureLayout::General: layout = VK_IMAGE_LAYOUT_GENERAL; break;
    default: break;
    }
    return unified && layout != VK_IMAGE_LAYOUT_UNDEFINED && layout != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        ? VK_IMAGE_LAYOUT_GENERAL : layout;
}

VkPipelineStageFlags2 toVkPipelineStages(PipelineStageBits stages)
{
    VkPipelineStageFlags2 flags = VK_PIPELINE_STAGE_2_NONE;
    const auto value = static_cast<uint64_t>(stages);
    if ((value & static_cast<uint64_t>(PipelineStageBits::TopOfPipe)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::DrawIndirect)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::VertexShader)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::FragmentShader)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::ComputeShader)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::ColorAttachment)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::Transfer)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::BottomOfPipe)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    }
    if ((value & static_cast<uint64_t>(PipelineStageBits::AllCommands)) != 0) {
        flags |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
    if (value & uint64_t(PipelineStageBits::DepthStencil)) { flags |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT; }
    if (value & uint64_t(PipelineStageBits::PreRasterization)) { flags |= VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT; }
    if (value & uint64_t(PipelineStageBits::AccelerationStructureBuild)) { flags |= VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR; }
    if (value & uint64_t(PipelineStageBits::RayTracingShader)) { flags |= VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR; }
    if (value & uint64_t(PipelineStageBits::MemoryDecompression)) { flags |= VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT; }
    if (value & uint64_t(PipelineStageBits::Host)) { flags |= VK_PIPELINE_STAGE_2_HOST_BIT; }
    return flags;
}

VkAccessFlags2 accessFlags(AccessBits access)
{
    VkAccessFlags2 flags = 0;
    const auto value = uint64_t(access);
    if (value & uint64_t(AccessBits::ShaderRead)) { flags |= VK_ACCESS_2_SHADER_READ_BIT; }
    if (value & uint64_t(AccessBits::ShaderWrite)) { flags |= VK_ACCESS_2_SHADER_WRITE_BIT; }
    if (value & uint64_t(AccessBits::UniformRead)) { flags |= VK_ACCESS_2_UNIFORM_READ_BIT; }
    if (value & uint64_t(AccessBits::IndirectRead)) { flags |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT; }
    if (value & uint64_t(AccessBits::TransferRead)) { flags |= VK_ACCESS_2_TRANSFER_READ_BIT; }
    if (value & uint64_t(AccessBits::TransferWrite)) { flags |= VK_ACCESS_2_TRANSFER_WRITE_BIT; }
    if (value & uint64_t(AccessBits::ColorRead)) { flags |= VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT; }
    if (value & uint64_t(AccessBits::ColorWrite)) { flags |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT; }
    if (value & uint64_t(AccessBits::DepthStencilRead)) { flags |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT; }
    if (value & uint64_t(AccessBits::DepthStencilWrite)) { flags |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT; }
    if (value & uint64_t(AccessBits::AccelerationStructureRead)) { flags |= VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR; }
    if (value & uint64_t(AccessBits::AccelerationStructureWrite)) { flags |= VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR; }
    if (value & uint64_t(AccessBits::DecompressionRead)) { flags |= VK_ACCESS_2_MEMORY_DECOMPRESSION_READ_BIT_EXT; }
    if (value & uint64_t(AccessBits::DecompressionWrite)) { flags |= VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT; }
    if (value & uint64_t(AccessBits::HostRead)) { flags |= VK_ACCESS_2_HOST_READ_BIT; }
    if (value & uint64_t(AccessBits::HostWrite)) { flags |= VK_ACCESS_2_HOST_WRITE_BIT; }
    if (value & uint64_t(AccessBits::MemoryRead)) { flags |= VK_ACCESS_2_MEMORY_READ_BIT; }
    if (value & uint64_t(AccessBits::MemoryWrite)) { flags |= VK_ACCESS_2_MEMORY_WRITE_BIT; }
    if (value & uint64_t(AccessBits::DescriptorRead)) { flags |= VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT | VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT; }
    return flags;
}

VulkanSyncScope scopeInfo(SyncScope scope)
{
    return {toVkPipelineStages(scope.stages),
        accessFlags(scope.access)};
}

bool validScope(SyncScope scope, SyncSupport support)
{
    const auto stages = uint64_t(scope.stages);
    const auto access = uint64_t(scope.access);
    if ((stages & ~((1ull << 15) - 1)) || (access & ~((1ull << 19) - 1)) || (!stages && access)) { return false; }
    const uint64_t graphics = uint64_t(PipelineStageBits::VertexShader) | uint64_t(PipelineStageBits::FragmentShader) |
        uint64_t(PipelineStageBits::ColorAttachment) | uint64_t(PipelineStageBits::DepthStencil) | uint64_t(PipelineStageBits::PreRasterization);
    if ((stages & graphics) && !(support.queues & VK_QUEUE_GRAPHICS_BIT)) { return false; }
    if ((stages & uint64_t(PipelineStageBits::ComputeShader)) && !(support.queues & VK_QUEUE_COMPUTE_BIT)) { return false; }
    if ((stages & uint64_t(PipelineStageBits::AccelerationStructureBuild)) && !support.rayTracingAS) { return false; }
    if ((stages & uint64_t(PipelineStageBits::MemoryDecompression)) && !support.decompression) { return false; }
    if ((stages & uint64_t(PipelineStageBits::RayTracingShader)) && !support.rayTracingPipeline) { return false; }
    const uint64_t computeQueueStages = uint64_t(PipelineStageBits::AccelerationStructureBuild) |
        uint64_t(PipelineStageBits::RayTracingShader) | uint64_t(PipelineStageBits::MemoryDecompression);
    if ((stages & computeQueueStages) && !(support.queues & VK_QUEUE_COMPUTE_BIT)) { return false; }
    if ((stages & uint64_t(PipelineStageBits::DrawIndirect)) && !(support.queues & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) { return false; }
    if ((access & uint64_t(AccessBits::DescriptorRead)) && !support.bindless) { return false; }
    if ((access & uint64_t(AccessBits::AccelerationStructureRead | AccessBits::AccelerationStructureWrite)) &&
        !support.rayTracingAS) { return false; }
    if ((access & uint64_t(AccessBits::DecompressionRead | AccessBits::DecompressionWrite)) &&
        !support.decompression) { return false; }
    const auto supports = [&](AccessBits mask, PipelineStageBits required, bool allCommands = true) {
        return !(access & uint64_t(mask)) || (stages & uint64_t(required)) ||
            (allCommands && (stages & uint64_t(PipelineStageBits::AllCommands)));
    };
    const auto shaders = PipelineStageBits::VertexShader | PipelineStageBits::FragmentShader |
        PipelineStageBits::PreRasterization | PipelineStageBits::ComputeShader | PipelineStageBits::RayTracingShader;
    return supports(AccessBits::ShaderRead, shaders | PipelineStageBits::AccelerationStructureBuild) &&
        supports(AccessBits::ShaderWrite | AccessBits::UniformRead | AccessBits::DescriptorRead, shaders) &&
        supports(AccessBits::IndirectRead, PipelineStageBits::DrawIndirect) &&
        supports(AccessBits::TransferRead | AccessBits::TransferWrite, PipelineStageBits::Transfer) &&
        supports(AccessBits::ColorRead | AccessBits::ColorWrite, PipelineStageBits::ColorAttachment) &&
        supports(AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite, PipelineStageBits::DepthStencil) &&
        supports(AccessBits::AccelerationStructureRead, shaders | PipelineStageBits::AccelerationStructureBuild) &&
        supports(AccessBits::AccelerationStructureWrite, PipelineStageBits::AccelerationStructureBuild) &&
        supports(AccessBits::DecompressionRead | AccessBits::DecompressionWrite, PipelineStageBits::MemoryDecompression) &&
        supports(AccessBits::HostRead | AccessBits::HostWrite, PipelineStageBits::Host, false);
}

} // namespace metallic::render::vulkan
