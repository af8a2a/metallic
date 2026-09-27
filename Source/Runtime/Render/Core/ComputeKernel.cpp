#include "Runtime/Render/Core/ComputeKernel.h"

#include <bit>
#include <vector>

namespace metallic::render {

struct ComputeKernel::Impl {
    const void* device = nullptr;
    ParameterAbi parameters;
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    PreparedExecution execution;
};

Result<> ComputeKernel::initialize(Device& device, const ComputeKernelDesc& desc, std::string& log)
{
    clear();
    log.clear();
    if (desc.spirv.size() < 5 || desc.spirv[0] != 0x07230203u || !desc.parameters.id || !desc.parameters.size || !std::has_single_bit(desc.parameters.alignment) ||
        desc.parameters.alignment > 4096) {
        return makeError(Error::InvalidArgument);
    }
    for (size_t word = 5; word < desc.spirv.size();) {
        const uint32_t count = desc.spirv[word] >> 16;
        if (count == 0 || count > desc.spirv.size() - word) { return makeError(Error::InvalidArgument); }
        word += count;
    }
    auto impl = std::make_shared<Impl>();
    auto result = device.createShaderModule({
        .spirv = desc.spirv,
        .debugName = desc.debugName,
    }).transform([&](auto rhiValue) { impl->shader = std::move(rhiValue); });
    if (result) {
        result = device.createComputePipeline({
            .computeShader = {impl->shader.get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(uint64_t),
            .pipelineCache = desc.pipelineCache,
        }).transform([&](auto rhiValue) { impl->pipeline = std::move(rhiValue); });
    }
    if (!result) {
        log += "ComputeKernel creation failed: "; log += resultToString(result); return result;
    }
    impl->execution = impl->pipeline->execution();
    impl->device = device.identity(); impl->parameters = desc.parameters;
    impl_ = std::move(impl);
    return {};
}

struct PreparedComputeDispatch::Impl {
    struct Item {
        PreparedExecution execution;
        EncodedParameters parameters;
        BufferSlice arguments;
        std::shared_ptr<void> executableOwner;
    };
    std::vector<Item> items;
    uint32_t x = 1, y = 1, z = 1;
};

Result<PreparedComputeDispatch> ComputeKernel::prepareDispatch(const EncodedParameters& params,
    uint32_t x, uint32_t y, uint32_t z) const
{
    if (!impl_ || !x || !y || !z || params.deviceIdentity() != impl_->device || params.abi() != impl_->parameters) {
        return makeError(Error::InvalidArgument);
    }
    auto packet = std::make_shared<PreparedComputeDispatch::Impl>();
    packet->items.push_back({impl_->execution, params, {}, impl_});
    packet->x = x; packet->y = y; packet->z = z;
    PreparedComputeDispatch out;
    out.impl_ = std::move(packet);
    return out;
}

Result<PreparedComputeDispatch> ComputeKernel::prepareIndirectBatch(
    std::span<const ComputeIndirectParameters> dispatches) const
{
    if (!impl_ || dispatches.empty()) { return makeError(Error::InvalidArgument); }
    auto packet = std::make_shared<PreparedComputeDispatch::Impl>();
    packet->items.reserve(dispatches.size());
    for (const auto& dispatch : dispatches) {
        const auto& kernel = dispatch.kernel ? dispatch.kernel->impl_ : impl_;
        if (!kernel || kernel->device != impl_->device || kernel->parameters != impl_->parameters ||
            dispatch.parameters.deviceIdentity() != impl_->device || dispatch.parameters.abi() != impl_->parameters ||
            !dispatch.arguments.validate(impl_->device, BufferUsageBits::Indirect, 4, 12)) {
            return makeError(Error::InvalidArgument);
        }
        packet->items.push_back({kernel->execution, dispatch.parameters, dispatch.arguments, kernel});
    }
    PreparedComputeDispatch out;
    out.impl_ = std::move(packet);
    return out;
}

Result<> PreparedComputeDispatch::record(CommandBuffer& commands, const BarrierDesc& betweenDispatches) const
{
    if (!impl_ || !commands.recording() ||
        (uint32_t(commands.queueCapabilities()) & uint32_t(QueueAccessBits::Compute)) == 0) {
        return makeError(Error::InvalidArgument);
    }
    // Validate the entire batch before recording: a later stale packet must not
    // leave an earlier dispatch in the command buffer.
    for (const auto& item : impl_->items) {
        if (!item.parameters.compatible(commands, item.parameters.abi()) ||
            commands.deviceIdentity() != item.execution.deviceIdentity()) {
            return makeError(Error::InvalidArgument);
        }
    }
    auto result = commands.retainResource(std::const_pointer_cast<Impl>(impl_));
    if (!result) { return result; }
    for (size_t i = 0; i < impl_->items.size(); ++i) {
        const auto& item = impl_->items[i];
        result = item.parameters.bindResources(commands);
        if (!result) { return result; }
        const uint64_t root = item.parameters.address();
        result = commands.bindExecution(item.execution, &root, sizeof(root));
        if (!result) { return result; }
        if (item.arguments.valid()) { result = commands.dispatchIndirect(item.arguments); }
        else { commands.dispatch(impl_->x, impl_->y, impl_->z); }
        if (!result) { return result; }
        if (i + 1 < impl_->items.size()) {
            result = commands.synchronize(betweenDispatches);
            if (!result) { return result; }
        }
    }
    return {};
}

Result<> ComputeKernel::dispatch(CommandBuffer& commands, const EncodedParameters& params,
    uint32_t x, uint32_t y, uint32_t z) const
{
    auto prepared = prepareDispatch(params, x, y, z);
    return prepared ? prepared->record(commands) : makeError(prepared.error());
}

Result<> ComputeKernel::dispatchIndirect(CommandBuffer& commands, const EncodedParameters& params,
    Buffer& arguments, uint64_t offset) const
{
    auto slice = arguments.slice({offset, 12});
    return slice ? dispatchIndirect(commands, params, *slice) : makeError(slice.error());
}

Result<> ComputeKernel::dispatchIndirect(CommandBuffer& commands, const EncodedParameters& params,
    const BufferSlice& arguments) const
{
    const ComputeIndirectParameters dispatch{params, arguments};
    auto prepared = prepareIndirectBatch({&dispatch, 1});
    return prepared ? prepared->record(commands) : makeError(prepared.error());
}

} // namespace metallic::render
