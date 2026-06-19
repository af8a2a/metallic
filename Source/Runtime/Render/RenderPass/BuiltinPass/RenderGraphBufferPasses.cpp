#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

class RenderGraphBufferWritePass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addBufferOutput("data", "Known test byte pattern")
            .buffer(kRenderGraphBufferByteSize)
            .storageReadWrite()
            .bindlessBuffer();
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().bindlessDescriptorHeap) {
            log = "RenderGraphBufferWritePass requires DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        Result result = createSlangShaderModule(
            *context.device,
            kRenderGraphBufferShaderModuleName,
            kRenderGraphBufferWriteEntryPoint,
            shader_,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createComputePipeline(
            ComputePipelineDesc{
                .computeShader = shader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(RenderGraphBufferUserPush),
            },
            pipeline_);
        if (!result) {
            log += resultMessage("createComputePipeline(RenderGraphBufferWritePass)", result);
            log += '\n';
        }
        return result;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        BufferHandle data = context.outputBuffer("data");
        if (!data.valid() || !data.bindlessHandle().valid() || pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const RenderGraphBufferUserPush push{
            .inputBuffer = 0,
            .outputBuffer = data.bindlessHandle().index,
            .passIndex = 0,
            .padding = 0,
        };
        context.commandBuffer().pushBindlessData(&push, sizeof(push));
        context.commandBuffer().bindComputePipeline(*pipeline_);
        context.commandBuffer().dispatch(1, 1, 1);
        return {};
    }

private:
    std::unique_ptr<ShaderModule> shader_;
    std::unique_ptr<ComputePipeline> pipeline_;
};

class RenderGraphBufferCopyPass final : public ComputePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addBufferInput("source", "Source byte buffer")
            .buffer(kRenderGraphBufferByteSize)
            .storageReadWrite()
            .bindlessBuffer();
        reflection.addBufferOutput("data", "Copied byte buffer")
            .buffer(kRenderGraphBufferByteSize)
            .storageReadWrite()
            .bindlessBuffer();
        return reflection;
    }

    Result compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().bindlessDescriptorHeap) {
            log = "RenderGraphBufferCopyPass requires DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }
        if (pipeline_ != nullptr) {
            return {};
        }

        Result result = createSlangShaderModule(
            *context.device,
            kRenderGraphBufferShaderModuleName,
            kRenderGraphBufferCopyEntryPoint,
            shader_,
            log);
        if (!result) {
            return result;
        }

        result = context.device->createComputePipeline(
            ComputePipelineDesc{
                .computeShader = shader_.get(),
                .computeEntryPoint = "main",
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(RenderGraphBufferUserPush),
            },
            pipeline_);
        if (!result) {
            log += resultMessage("createComputePipeline(RenderGraphBufferCopyPass)", result);
            log += '\n';
        }
        return result;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        BufferHandle source = context.inputBuffer("source");
        BufferHandle data = context.outputBuffer("data");
        if (!source.valid() ||
            !source.bindlessHandle().valid() ||
            !data.valid() ||
            !data.bindlessHandle().valid() ||
            pipeline_ == nullptr) {
            return makeError(Error::InvalidArgument);
        }

        const RenderGraphBufferUserPush push{
            .inputBuffer = source.bindlessHandle().index,
            .outputBuffer = data.bindlessHandle().index,
            .passIndex = 0,
            .padding = 0,
        };
        context.commandBuffer().pushBindlessData(&push, sizeof(push));
        context.commandBuffer().bindComputePipeline(*pipeline_);
        context.commandBuffer().dispatch(1, 1, 1);
        return {};
    }

private:
    std::unique_ptr<ShaderModule> shader_;
    std::unique_ptr<ComputePipeline> pipeline_;
};

} // namespace

std::unique_ptr<RenderGraphPass> createRenderGraphBufferWritePass()
{
    return std::make_unique<RenderGraphBufferWritePass>();
}

std::unique_ptr<RenderGraphPass> createRenderGraphBufferCopyPass()
{
    return std::make_unique<RenderGraphBufferCopyPass>();
}

} // namespace metallic::render::builtin_pass
