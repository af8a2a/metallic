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

namespace {
using Stage = PipelineStageBits;
using Access = AccessBits;
using Feature = bool SyncSupport::*;
constexpr auto kGraphics = VK_QUEUE_GRAPHICS_BIT;
constexpr auto kCompute = VK_QUEUE_COMPUTE_BIT;
constexpr auto kDrawQueues = kGraphics | kCompute;
constexpr auto kTransferQueues = kDrawQueues | VK_QUEUE_TRANSFER_BIT;
constexpr auto kShaders = Stage::VertexShader | Stage::FragmentShader | Stage::PreRasterization |
    Stage::ComputeShader | Stage::RayTracingShader;

struct StageRule {
    Stage bit;
    VkPipelineStageFlags2 native;
    VkQueueFlags queues = 0; // Any listed queue capability suffices; zero is unrestricted.
    Feature feature = nullptr;
    bool inAllCommands = true;
    bool deviceCommand = true;
};
constexpr StageRule kStages[]{
    {Stage::TopOfPipe, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, nullptr, false},
    {Stage::DrawIndirect, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, kDrawQueues},
    {Stage::VertexShader, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT, kGraphics},
    {Stage::FragmentShader, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, kGraphics},
    {Stage::ComputeShader, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kCompute},
    {Stage::ColorAttachment, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, kGraphics},
    {Stage::Transfer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, kTransferQueues},
    {Stage::BottomOfPipe, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0, nullptr, false},
    {Stage::AllCommands, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, nullptr, false},
    {Stage::DepthStencil, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, kGraphics},
    {Stage::PreRasterization, VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT, kGraphics},
    {Stage::AccelerationStructureBuild, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, kCompute, &SyncSupport::rayTracingAS},
    {Stage::RayTracingShader, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, kCompute, &SyncSupport::rayTracingPipeline},
    {Stage::MemoryDecompression, VK_PIPELINE_STAGE_2_MEMORY_DECOMPRESSION_BIT_EXT, kCompute, &SyncSupport::decompression},
    {Stage::Host, VK_PIPELINE_STAGE_2_HOST_BIT, 0, nullptr, false, false},
};

struct AccessRule {
    Access bit;
    VkAccessFlags2 native;
    Stage stages; // At least one compatible stage; None means any nonempty scope.
    Feature feature = nullptr;
};
constexpr AccessRule kAccesses[]{
    {Access::ShaderRead, VK_ACCESS_2_SHADER_READ_BIT, kShaders | Stage::AccelerationStructureBuild},
    {Access::ShaderWrite, VK_ACCESS_2_SHADER_WRITE_BIT, kShaders},
    {Access::UniformRead, VK_ACCESS_2_UNIFORM_READ_BIT, kShaders},
    {Access::IndirectRead, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, Stage::DrawIndirect},
    {Access::TransferRead, VK_ACCESS_2_TRANSFER_READ_BIT, Stage::Transfer},
    {Access::TransferWrite, VK_ACCESS_2_TRANSFER_WRITE_BIT, Stage::Transfer},
    {Access::ColorRead, VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT, Stage::ColorAttachment},
    {Access::ColorWrite, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, Stage::ColorAttachment},
    {Access::DepthStencilRead, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, Stage::DepthStencil},
    {Access::DepthStencilWrite, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, Stage::DepthStencil},
    {Access::AccelerationStructureRead, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        kShaders | Stage::AccelerationStructureBuild, &SyncSupport::rayTracingAS},
    {Access::AccelerationStructureWrite, VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        Stage::AccelerationStructureBuild, &SyncSupport::rayTracingAS},
    {Access::DecompressionRead, VK_ACCESS_2_MEMORY_DECOMPRESSION_READ_BIT_EXT, Stage::MemoryDecompression, &SyncSupport::decompression},
    {Access::DecompressionWrite, VK_ACCESS_2_MEMORY_DECOMPRESSION_WRITE_BIT_EXT, Stage::MemoryDecompression, &SyncSupport::decompression},
    {Access::DescriptorRead, VK_ACCESS_2_RESOURCE_HEAP_READ_BIT_EXT | VK_ACCESS_2_SAMPLER_HEAP_READ_BIT_EXT, kShaders, &SyncSupport::bindless},
    {Access::HostRead, VK_ACCESS_2_HOST_READ_BIT, Stage::Host},
    {Access::HostWrite, VK_ACCESS_2_HOST_WRITE_BIT, Stage::Host},
    {Access::MemoryRead, VK_ACCESS_2_MEMORY_READ_BIT, Stage::None},
    {Access::MemoryWrite, VK_ACCESS_2_MEMORY_WRITE_BIT, Stage::None},
};

template<class Rule, size_t N>
constexpr uint64_t knownBits(const Rule (&rules)[N])
{
    uint64_t bits = 0;
    for (const auto& rule : rules) { bits |= uint64_t(rule.bit); }
    return bits;
}

template<class Rule, size_t N>
consteval bool validRules(const Rule (&rules)[N])
{
    uint64_t seen = 0;
    for (const auto& rule : rules) {
        const auto bit = uint64_t(rule.bit);
        if (!bit || (bit & (bit - 1)) || (seen & bit) || !rule.native) { return false; }
        seen |= bit;
    }
    return true;
}
static_assert(validRules(kStages) && validRules(kAccesses));
static_assert([] {
    for (const auto& rule : kAccesses) {
        if (uint64_t(rule.stages) & ~knownBits(kStages)) { return false; }
    }
    return true;
}());

template<class Bit, class Rule, size_t N>
uint64_t nativeFlags(Bit bits, const Rule (&rules)[N])
{
    uint64_t flags = 0;
    for (const auto& rule : rules) {
        if (uint64_t(bits) & uint64_t(rule.bit)) { flags |= rule.native; }
    }
    return flags;
}

bool supported(const StageRule& rule, const SyncSupport& support)
{
    return (!rule.queues || (rule.queues & support.queues)) &&
        (!rule.feature || support.*rule.feature);
}
} // namespace

VkPipelineStageFlags2 toVkPipelineStages(PipelineStageBits stages)
{
    return nativeFlags(stages, kStages);
}

VkAccessFlags2 accessFlags(AccessBits access)
{
    return nativeFlags(access, kAccesses);
}

VulkanSyncScope scopeInfo(SyncScope scope)
{
    return {toVkPipelineStages(scope.stages), accessFlags(scope.access)};
}

bool validScope(SyncScope scope, SyncSupport support)
{
    const auto stages = uint64_t(scope.stages);
    const auto access = uint64_t(scope.access);
    if ((stages & ~knownBits(kStages)) || (access & ~knownBits(kAccesses)) || (!stages && access)) { return false; }
    uint64_t effectiveStages = stages;
    for (const auto& rule : kStages) {
        const bool available = supported(rule, support);
        if ((stages & uint64_t(rule.bit)) && !available) { return false; }
        // AllCommands covers only stages supported by this queue and device.
        // Host accesses still require an explicit Host stage.
        if ((stages & uint64_t(Stage::AllCommands)) && rule.inAllCommands && available) {
            effectiveStages |= uint64_t(rule.bit);
        }
    }
    for (const auto& rule : kAccesses) {
        if (!(access & uint64_t(rule.bit))) { continue; }
        if (rule.feature && !(support.*rule.feature)) { return false; }
        if (rule.stages != Stage::None && !(effectiveStages & uint64_t(rule.stages))) { return false; }
    }
    return true;
}

bool validDeviceStages(PipelineStageBits stages, SyncSupport support)
{
    if (!validScope({stages, AccessBits::None}, support)) { return false; }
    for (const auto& rule : kStages) {
        if (!rule.deviceCommand && (uint64_t(stages) & uint64_t(rule.bit))) { return false; }
    }
    return true;
}

} // namespace metallic::render::vulkan
