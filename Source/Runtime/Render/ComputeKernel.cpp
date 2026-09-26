#include "Runtime/Render/ComputeKernel.h"

namespace metallic::render {

struct ComputeKernel::Impl {
    const void* device = nullptr;
    ParameterAbi parameters;
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
};

Result ComputeKernel::initialize(Device& device, const ComputeKernelDesc& desc, std::string& log)
{
    clear();
    if (desc.spirv.empty() || !desc.parameters.id || !desc.parameters.size || !desc.parameters.alignment) {
        return makeError(Error::InvalidArgument);
    }
    auto impl = std::make_shared<Impl>();
    auto result = device.createShaderModule({.code = desc.spirv.data(), .byteSize = desc.spirv.size_bytes(),
        .debugName = desc.debugName}, impl->shader);
    if (result) {
        result = device.createComputePipeline({.computeShader = impl->shader.get(), .computeEntryPoint = "main",
            .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(uint64_t),
            .pipelineCache = desc.pipelineCache}, impl->pipeline);
    }
    if (!result) {
        log += "ComputeKernel creation failed: "; log += resultToString(result); return result;
    }
    impl->device = device.identity(); impl->parameters = desc.parameters;
    impl_ = std::move(impl);
    return {};
}

Result ComputeKernel::bind(CommandBuffer& commands, const EncodedParameters& params) const
{
    if (!impl_ || commands.deviceIdentity() != impl_->device || !params.compatible(commands, impl_->parameters)) {
        return makeError(Error::InvalidArgument);
    }
    commands.frameContext()->retain(impl_);
    // The packet binds its heap before this prepared pipeline is selected.
    auto result = params.bindResources(commands);
    if (!result) { return result; }
    const uint64_t root = params.address();
    commands.bindComputePipeline(*impl_->pipeline, &root, sizeof(root));
    return {};
}

Result ComputeKernel::dispatch(CommandBuffer& commands, const EncodedParameters& params,
    uint32_t x, uint32_t y, uint32_t z) const
{
    if (!x || !y || !z || (uint32_t(commands.queueCapabilities()) & uint32_t(QueueAccessBits::Compute)) == 0) {
        return makeError(Error::InvalidArgument);
    }
    auto result = bind(commands, params);
    if (!result) { return result; }
    commands.dispatch(x, y, z);
    return {};
}

Result ComputeKernel::dispatchIndirect(CommandBuffer& commands, const EncodedParameters& params,
    Buffer& arguments, uint64_t offset) const
{
    BufferSlice slice;
    auto result = arguments.slice(slice, offset, 12);
    return result ? dispatchIndirect(commands, params, slice) : result;
}

Result ComputeKernel::dispatchIndirect(CommandBuffer& commands, const EncodedParameters& params,
    const BufferSlice& arguments) const
{
    if (!arguments.validate(commands.deviceIdentity(), BufferUsageBits::Indirect, 4, 12) ||
        (uint32_t(commands.queueCapabilities()) & uint32_t(QueueAccessBits::Compute)) == 0) {
        return makeError(Error::InvalidArgument);
    }
    auto result = bind(commands, params);
    return result ? commands.dispatchIndirect(arguments) : result;
}

} // namespace metallic::render
