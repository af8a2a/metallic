#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanGeneratedCommands.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <gtest/gtest.h>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

namespace rv = render::vulkan;

TEST(DeviceGeneratedCommands, RejectsUninitializedObjects)
{
    rv::GeneratedCommands generated;
    render::Device device;
    render::CommandBuffer commands;
    EXPECT_TRUE(render::hasError(generated.initialize(device, {}), render::Error::InvalidArgument));
    EXPECT_TRUE(render::hasError(generated.prepare(), render::Error::InvalidArgument));
    EXPECT_TRUE(render::hasError(generated.execute(commands, {}), render::Error::InvalidArgument));
    EXPECT_TRUE(render::hasError(generated.preprocess(commands, {}, commands), render::Error::InvalidArgument));
    EXPECT_TRUE(render::hasError(generated.preprocessBarrier(commands), render::Error::InvalidArgument));
    EXPECT_TRUE(render::hasError(generated.updatePipelines({}), render::Error::InvalidArgument));
    EXPECT_TRUE(render::hasError(generated.updateShaders({}), render::Error::InvalidArgument));
}

TEST(DeviceGeneratedCommands, IndirectBindingChangesPipelineHash)
{
    render::ComputePipelineDesc compute;
    const auto computeHash = render::detail::computePipelineStateHash(compute);
    compute.indirectBindable = true;
    EXPECT_NE(computeHash, render::detail::computePipelineStateHash(compute));
    render::GraphicsPipelineDesc graphics;
    const auto graphicsHash = render::detail::graphicsPipelineStateHash(graphics);
    graphics.indirectBindable = true;
    EXPECT_NE(graphicsHash, render::detail::graphicsPipelineStateHash(graphics));
}

TEST(PipelineStateHash, ColorAttachmentArray)
{
    render::GraphicsPipelineDesc desc{.colorFormats = {render::Format::RGBA8Unorm, render::Format::R32Uint,
        render::Format::RGBA16Sfloat, render::Format::RGBA32Sfloat}, .colorAttachmentCount = 4};
    const auto hash = render::detail::graphicsPipelineStateHash(desc);
    auto changed = desc;
    changed.colorFormats[3] = render::Format::RGBA8Unorm;
    EXPECT_NE(hash, render::detail::graphicsPipelineStateHash(changed));
    changed = desc;
    std::swap(changed.colorFormats[0], changed.colorFormats[1]);
    EXPECT_NE(hash, render::detail::graphicsPipelineStateHash(changed));
    changed = desc;
    changed.colorAttachmentCount = 3;
    EXPECT_NE(hash, render::detail::graphicsPipelineStateHash(changed));
    changed = desc;
    changed.colorFormats[7] = render::Format::R32Uint;
    EXPECT_EQ(hash, render::detail::graphicsPipelineStateHash(changed));
}

TEST(DeviceGeneratedCommands, ProbeShaderCompiles)
{
    render::ShaderCompileResult shader;
    const auto result = render::compileSlangShaderToSpirv({.moduleName = "GeneratedCommandsProbe",
        .entryPointName = "generatedCommandsProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
    EXPECT_TRUE(result.has_value()) << shader.diagnostics;
    EXPECT_FALSE(shader.spirv.empty());
}

struct ProbeResources {
    VkDevice device;
    const VolkDeviceTable& functions;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::array<VkPipeline, 2> pipelines{};
    std::array<VkShaderModule, 2> shaders{};

    ~ProbeResources()
    {
        // Also protect cleanup after a failed submission/wait assertion.
        functions.vkDeviceWaitIdle(device);
        for (auto pipeline : pipelines) { if (pipeline) { functions.vkDestroyPipeline(device, pipeline, nullptr); } }
        for (auto shader : shaders) { if (shader) { functions.vkDestroyShaderModule(device, shader, nullptr); } }
        if (layout) { functions.vkDestroyPipelineLayout(device, layout, nullptr); }
        if (pool) { functions.vkDestroyDescriptorPool(device, pool, nullptr); }
        if (setLayout) { functions.vkDestroyDescriptorSetLayout(device, setLayout, nullptr); }
    }
};

class GeneratedCommandsComputeTest final : public RHITest {
public:
    GeneratedCommandsComputeTest()
    {
        type = RHITestType::Command;
        name = "device_generated_commands_compute_readback";
    }

    std::optional<bench::Metadata> metadata() const override
    {
        return bench::comparisonMetadata({"dgc.backend.compute.pipelineSet.push.count.preprocess.rebind"}, bench::Layer::Backend,
            "binding", {"binding-dgc", "dgc", bench::Capability::GeneratedCommands});
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto& device = context.device;
        const bool useGenerated = !context.deviceDesc || context.deviceDesc->enableDeviceGeneratedCommands;
        if (useGenerated && !device.capabilities().deviceGeneratedCommands) { return RHITestResult::skip("DGC unsupported"); }
        const auto properties = rv::queryGeneratedCommandsProperties(device);
        if (useGenerated && (!properties ||
            !(properties->supportedIndirectCommandsShaderStagesPipelineBinding & VK_SHADER_STAGE_COMPUTE_BIT))) {
            return RHITestResult::skip("DGC compute pipeline binding unsupported");
        }
        const auto native = rv::nativeDevice(device);
        ProbeResources resources{native.device, *native.functions};
#define DGC_VK(expr) if ((expr) != VK_SUCCESS) { return RHITestResult::fail(#expr); }
#define DGC_RHI(expr) if (!(expr)) { return RHITestResult::fail(#expr); }
        const VkDescriptorSetLayoutBinding binding{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
        const VkDescriptorSetLayoutCreateInfo setInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1, .pBindings = &binding};
        DGC_VK(native.functions->vkCreateDescriptorSetLayout(native.device, &setInfo, nullptr, &resources.setLayout));
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        const VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1, .pSetLayouts = &resources.setLayout, .pushConstantRangeCount = 1, .pPushConstantRanges = &push};
        DGC_VK(native.functions->vkCreatePipelineLayout(native.device, &layoutInfo, nullptr, &resources.layout));
        for (uint32_t i = 0; i < 2; ++i) {
            const render::SlangMacroDefine macro{"DGC_ADD", i ? "1000" : "0"};
            render::ShaderCompileResult shader;
            DGC_RHI(render::compileSlangShaderToSpirv({
                .moduleName = "GeneratedCommandsProbe",
                .entryPointName = "generatedCommandsProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .macroDefines = {&macro, 1},
            }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); }));
            const VkShaderModuleCreateInfo shaderInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = shader.spirv.size() * 4, .pCode = shader.spirv.data()};
            DGC_VK(native.functions->vkCreateShaderModule(native.device, &shaderInfo, nullptr, &resources.shaders[i]));
            const VkPipelineCreateFlags2CreateInfo flags{.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
                .flags = useGenerated ? VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT : 0u};
            const VkComputePipelineCreateInfo pipelineInfo{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = &flags, .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = resources.shaders[i], .pName = "main"},
                .layout = resources.layout};
            DGC_VK(native.functions->vkCreateComputePipelines(native.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &resources.pipelines[i]));
        }
        std::unique_ptr<render::Buffer> output, arguments;
        DGC_RHI(device.createBuffer({.size = 16, .usage = render::BufferUsageBits::Storage,
            .memoryLocation = render::MemoryLocation::HostReadback}).transform([&](auto rhiValue) { output = std::move(rhiValue); }));
        DGC_RHI(device.createBuffer({.size = 80, .usage = render::BufferUsageBits::Indirect,
            .memoryLocation = render::MemoryLocation::HostUpload}).transform([&](auto rhiValue) { arguments = std::move(rhiValue); }));
        const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        const VkDescriptorPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &poolSize};
        DGC_VK(native.functions->vkCreateDescriptorPool(native.device, &poolInfo, nullptr, &resources.pool));
        VkDescriptorSet set = VK_NULL_HANDLE;
        const VkDescriptorSetAllocateInfo setAllocation{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = resources.pool, .descriptorSetCount = 1, .pSetLayouts = &resources.setLayout};
        DGC_VK(native.functions->vkAllocateDescriptorSets(native.device, &setAllocation, &set));
        const VkDescriptorBufferInfo bufferInfo{rv::nativeBuffer(*output).buffer, 0, 16};
        const VkWriteDescriptorSet write{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
            .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bufferInfo};
        native.functions->vkUpdateDescriptorSets(native.device, 1, &write, 0, nullptr);

        std::vector<uint32_t> observations;
        // Exercise fixed state, pipeline switching, and explicit preprocessing.
        for (uint32_t mode = 0; mode < 3; ++mode) {
            const VkIndirectCommandsExecutionSetTokenEXT executionToken{VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT, VK_SHADER_STAGE_COMPUTE_BIT};
            const VkIndirectCommandsPushConstantTokenEXT pushToken{push};
            const VkIndirectCommandsLayoutTokenEXT tokens[] = {
                {.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT, .type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT,
                    .data = {.pExecutionSet = &executionToken}, .offset = 0},
                {.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT, .type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT,
                    .data = {.pPushConstant = &pushToken}, .offset = 4},
                {.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT, .type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT, .offset = 12}};
            const VkIndirectExecutionSetPipelineInfoEXT pipelineSet{.sType = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT,
                .initialPipeline = resources.pipelines[0], .maxPipelineCount = 2};
            const VkIndirectExecutionSetCreateInfoEXT executionSet{.sType = VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT,
                .type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT, .info = {.pPipelineInfo = &pipelineSet}};
            rv::GeneratedCommands generated;
            if (useGenerated) {
                DGC_RHI(generated.initialize(device, {
                    .layout = {.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT,
                        .flags = mode == 2 ? VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT : 0u,
                        .shaderStages = VK_SHADER_STAGE_COMPUTE_BIT, .indirectStride = 24, .pipelineLayout = resources.layout,
                        .tokenCount = mode ? 3u : 2u, .pTokens = mode ? tokens : tokens + 1},
                    .executionSet = mode ? &executionSet : nullptr, .pipeline = mode ? VK_NULL_HANDLE : resources.pipelines[0],
                    .maxSequenceCount = 3}));
                if (mode) {
                    const VkWriteIndirectExecutionSetPipelineEXT update{.sType = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT,
                        .index = 1, .pipeline = resources.pipelines[1]};
                    DGC_RHI(generated.updatePipelines(std::span(&update, 1)));
                    DGC_RHI(generated.prepare());
                }
            }
            // Header holds a GPU-readable count and padding. The third sequence must not execute.
            const uint32_t stream[] = {2, 0, 0, 0, 7, 1, 1, 1, 1, 1, 9, 1, 1, 1, 0, 2, 999, 1, 1, 1};
            auto* mappedArguments = arguments->map();
            auto* mappedOutput = output->map();
            if (!mappedArguments || !mappedOutput) { return RHITestResult::fail("Map failed"); }
            std::memcpy(mappedArguments, stream, sizeof(stream));
            std::memset(mappedOutput, 0, 16);
            arguments->flush(); output->flush();
            arguments->unmap(); output->unmap();
            std::unique_ptr<render::CommandPool> pool;
            std::unique_ptr<render::CommandBuffer> commands;
            std::unique_ptr<render::Fence> fence;
            DGC_RHI(device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
            DGC_RHI(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
            DGC_RHI(device.createFence(false).transform([&](auto rhiValue) { fence = std::move(rhiValue); }));
            struct Drain { render::Queue& queue; ~Drain() { (void)queue.waitIdle(); } } drain{context.graphicsQueue};
            DGC_RHI(commands->begin());
            {
                rv::ExternalCommandScope scope(*commands);
                const auto cmd = scope.commandBuffer();
                native.functions->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipelines[0]);
                native.functions->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resources.layout, 0, 1, &set, 0, nullptr);
                const rv::GeneratedCommandsArguments args{.commands = arguments.get(), .offset = 8, .sequenceCount = 3,
                    .countBuffer = arguments.get()};
                if (useGenerated) {
                    auto invalid = args;
                    invalid.offset = 1;
                    if (!render::hasError(generated.execute(*commands, invalid, mode == 2), render::Error::InvalidArgument)) {
                        return RHITestResult::fail("Misaligned indirect stream was accepted");
                    }
                    if (mode == 2) {
                        DGC_RHI(generated.preprocess(*commands, args, *commands));
                        DGC_RHI(generated.preprocessBarrier(*commands));
                    }
                    DGC_RHI(generated.execute(*commands, args, mode == 2));
                } else {
                    for (uint32_t index = 0; index < 2; ++index) {
                        native.functions->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipelines[mode && index ? 1 : 0]);
                        const uint32_t pushValues[]{index, index ? 9u : 7u};
                        native.functions->vkCmdPushConstants(cmd, resources.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushValues), pushValues);
                        native.functions->vkCmdDispatch(cmd, 1, 1, 1);
                    }
                }
                // Generated execution invalidates native binding state; explicitly establish all state again.
                native.functions->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resources.pipelines[0]);
                native.functions->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, resources.layout, 0, 1, &set, 0, nullptr);
                const uint32_t rebound[]{3, 777};
                native.functions->vkCmdPushConstants(cmd, resources.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rebound), rebound);
                native.functions->vkCmdDispatch(cmd, 1, 1, 1);
                const VkMemoryBarrier2 barrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
                const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
                native.functions->vkCmdPipelineBarrier2(cmd, &dependency);
            }
            DGC_RHI(commands->end());
            render::CommandBuffer* submitted[] = {commands.get()};
            DGC_RHI(context.graphicsQueue.submit({.commandBuffers = {submitted, 1}, .signalFence = fence.get()}));
            const auto waitResult = fence->wait(5'000'000'000ull);
            DGC_RHI(waitResult);
            output->invalidate();
            const auto* values = static_cast<const uint32_t*>(output->map());
            const bool correct = values && values[0] == 7 && values[1] == (mode ? 1009u : 9u) && values[2] == 0 && values[3] == 777;
            if (values) {
                bench::readbackEvidence(context, "readback.bin", std::span<const uint32_t>(values, 4));
                observations.insert(observations.end(), values, values + 4);
            }
            output->unmap();
            if (!correct) { return RHITestResult::fail("DGC pipeline/push/count readback mismatch"); }
        }
        bench::comparisonEvidence(context, {{"modes", {"fixed", "pipelineSet", "preprocess"}}, {"count", 2},
            {"maximumCount", 3}, {"values", {7, 9, 999}}, {"rebind", 777}}, observations, useGenerated);
#undef DGC_RHI
#undef DGC_VK
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(GeneratedCommandsComputeTest);

} // namespace
} // namespace metallic::tests
