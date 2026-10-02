#include "Runtime/Render/Core/RenderGraphBufferParameters.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPasses.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

namespace metallic::render::builtin_pass {
namespace {

class RenderGraphBufferWritePass final : public ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    CPURecordingPolicy cpuRecordingPolicy() const override { return CPURecordingPolicy::ParallelJoined; }
    bool supportsPipelinedSubmission() const override { return true; }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addBufferOutput("data", "Known test byte pattern")
            .buffer(kRenderGraphBufferByteSize)
            .storageWrite()
            .transient(RenderGraphInitialization::FullOverwrite);
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().bindlessDescriptorHeap) {
            log = "RenderGraphBufferWritePass requires DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }
        if (kernel_.valid()) {
            return {};
        }

        device_ = context.device;
        ShaderCompileResult shader;
        auto result = compileSlangShaderToSpirv({.moduleName = kRenderGraphBufferShaderModuleName,
            .entryPointName = kRenderGraphBufferWriteEntryPoint, .searchPath = kTriangleShaderSearchPath},
            shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        return kernel_.initialize(*context.device, {.spirv = shader.spirv,
            .parameters = parameterAbi<RenderGraphBufferParams>(kRenderGraphBufferABI, ParameterTransport::InlinePush),
            .debugName = "RenderGraphBuffer"}, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        BufferHandle data = context.outputBuffer("data");
        if (!data.valid() || !kernel_.valid()) {
            return makeError(Error::InvalidArgument);
        }

        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, commands.frameContext());
        const RenderGraphBufferParams params{
            .output = writer.dataBuffer(data.buffer(), 4, 4),
        };
        auto encoded = writer.encode(params, kRenderGraphBufferABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return kernel_.dispatch(commands, *encoded, 1);
    }

private:
    Device* device_ = nullptr;
    ComputeKernel kernel_;
};

class RenderGraphBufferCopyPass final : public ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    CPURecordingPolicy cpuRecordingPolicy() const override { return CPURecordingPolicy::ParallelJoined; }
    bool supportsPipelinedSubmission() const override { return true; }
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addBufferInput("source", "Source byte buffer")
            .buffer(kRenderGraphBufferByteSize)
            .storageRead();
        reflection.addBufferOutput("data", "Copied byte buffer")
            .buffer(kRenderGraphBufferByteSize)
            .storageWrite()
            .transient(RenderGraphInitialization::FullOverwrite);
        return reflection;
    }

    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        if (context.device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (!context.device->capabilities().bindlessDescriptorHeap) {
            log = "RenderGraphBufferCopyPass requires DeviceCapabilities::bindlessDescriptorHeap";
            return makeError(Error::Unsupported);
        }
        if (kernel_.valid()) {
            return {};
        }

        device_ = context.device;
        ShaderCompileResult shader;
        auto result = compileSlangShaderToSpirv({.moduleName = kRenderGraphBufferShaderModuleName,
            .entryPointName = kRenderGraphBufferCopyEntryPoint, .searchPath = kTriangleShaderSearchPath},
            shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        return kernel_.initialize(*context.device, {.spirv = shader.spirv,
            .parameters = parameterAbi<RenderGraphBufferParams>(kRenderGraphBufferABI, ParameterTransport::InlinePush),
            .debugName = "RenderGraphBuffer"}, log);
    }

    Result<> execute(RenderGraphExecutionContext& context) override
    {
        BufferHandle source = context.inputBuffer("source");
        BufferHandle data = context.outputBuffer("data");
        if (!source.valid() ||
            !data.valid() ||
            !kernel_.valid()) {
            return makeError(Error::InvalidArgument);
        }

        auto registry = device_->resourceRegistry();
        if (!registry) { return makeError(registry.error()); }
        auto& commands = context.commandBuffer();
        ParameterWriter writer(*device_, **registry, commands.frameContext());
        const RenderGraphBufferParams params{
            .source = writer.dataBuffer(source.buffer(), 4, 4),
            .output = writer.dataBuffer(data.buffer(), 4, 4),
        };
        auto encoded = writer.encode(params, kRenderGraphBufferABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return kernel_.dispatch(commands, *encoded, 1);
    }

private:
    Device* device_ = nullptr;
    ComputeKernel kernel_;
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
