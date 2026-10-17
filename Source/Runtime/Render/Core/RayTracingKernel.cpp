#include "Runtime/Render/Core/RayTracingKernel.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <array>
#include <bit>

namespace metallic::render {
namespace {

bool validRayTracingSPIRV(std::span<const uint32_t> spirv)
{
    if (spirv.size() < 5 || spirv[0] != 0x07230203u) { return false; }
    for (size_t word = 5; word < spirv.size();) {
        const uint32_t count = spirv[word] >> 16;
        if (count == 0 || count > spirv.size() - word) { return false; }
        word += count;
    }
    return true;
}

} // namespace

struct RayTracingKernel::Impl {
    const void* device = nullptr;
    ParameterABI parameters;
    std::array<std::unique_ptr<ShaderModule>, 4> shaders;
    std::unique_ptr<RayTracingPipeline> pipeline;
    PreparedExecution execution;
};

ParameterABI RayTracingKernel::parameterABI() const
{
    return impl_ ? impl_->parameters : ParameterABI{};
}

Result<> RayTracingKernel::initialize(Device& device, const RayTracingKernelDesc& desc, std::string& log)
{
    clear();
    log.clear();
    const std::array stages{desc.rayGenerationSpirv, desc.missSpirv, desc.closestHitSpirv, desc.anyHitSpirv};
    for (const auto stage : stages) {
        if (!validRayTracingSPIRV(stage)) {
            log = "RayTracingKernel requires four valid SPIR-V modules";
            return makeError(Error::InvalidArgument);
        }
    }
    if (!desc.parameters.id || !desc.parameters.size || !std::has_single_bit(desc.parameters.alignment) ||
        desc.parameters.alignment > 4096 ||
        (desc.parameters.transport != ParameterTransport::DescriptorBuffer && desc.parameters.transport != ParameterTransport::InlinePush) ||
        (desc.parameters.size & 3u)) {
        log = "RayTracingKernel requires a valid parameter ABI";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().rayTracingPipeline || !device.capabilities().rayTracingAccelerationStructure ||
        !device.capabilities().bindlessDescriptorHeap) {
        log = "RayTracingKernel requires unavailable device capabilities";
        return makeError(Error::Unsupported);
    }
    auto impl = std::make_shared<Impl>();
    Result<> result;
    for (size_t index = 0; index < stages.size(); ++index) {
        result = ShaderRegistry::instance().getShaderModule(device, {
            .spirv = stages[index], .debugName = desc.debugName,
        }).transform([&](auto rhiValue) { impl->shaders[index] = std::move(rhiValue); });
        if (!result) { break; }
    }
    if (result) {
        result = ShaderRegistry::instance().getRayTracingPipeline(device, {
            .rayGenerationShader = {impl->shaders[0].get(), "main", desc.specializationConstants},
            .missShader = {impl->shaders[1].get(), "main"},
            .closestHitShader = {impl->shaders[2].get(), "main"},
            .anyHitShader = {impl->shaders[3].get(), "main"},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = desc.parameters.transport == ParameterTransport::InlinePush
                ? desc.parameters.size : uint32_t(sizeof(GPUBufferSpan)),
            .pipelineCache = desc.pipelineCache,
            .maxRecursionDepth = 1,
            .clusterAccelerationStructure = true,
            .enableInvocationReorder = desc.enableInvocationReorder,
        }).transform([&](auto rhiValue) { impl->pipeline = std::move(rhiValue); });
    }
    if (!result) {
        log += "RayTracingKernel creation failed: "; log += resultToString(result); return result;
    }
    impl->execution = impl->pipeline->execution();
    impl->device = device.identity();
    impl->parameters = desc.parameters;
    impl_ = std::move(impl);
    return {};
}

struct PreparedRayTracingDispatch::Impl {
    PreparedExecution execution;
    EncodedParameters parameters;
    std::shared_ptr<void> executableOwner;
    uint32_t width = 1, height = 1, depth = 1;
};

Result<PreparedRayTracingDispatch> RayTracingKernel::prepareDispatch(const EncodedParameters& params,
    uint32_t width, uint32_t height, uint32_t depth) const
{
    if (!impl_ || !width || !height || !depth || params.deviceIdentity() != impl_->device || params.abi() != impl_->parameters) {
        return makeError(Error::InvalidArgument);
    }
    auto packet = std::make_shared<PreparedRayTracingDispatch::Impl>();
    packet->execution = impl_->execution;
    packet->parameters = params;
    packet->executableOwner = impl_;
    packet->width = width; packet->height = height; packet->depth = depth;
    PreparedRayTracingDispatch out;
    out.impl_ = std::move(packet);
    return out;
}

Result<> PreparedRayTracingDispatch::record(CommandBuffer& commands) const
{
    if (!impl_ || !commands.recording() ||
        (uint32_t(commands.queueCapabilities()) & uint32_t(QueueAccessBits::Compute)) == 0 ||
        !impl_->parameters.compatible(commands, impl_->parameters.abi()) ||
        commands.deviceIdentity() != impl_->execution.deviceIdentity()) {
        return makeError(Error::InvalidArgument);
    }
    // Retain the packet before recording its binding. This includes the native
    // pipeline/SBT and stage modules even if the owning kernel is then cleared.
    auto result = commands.retainResource(std::const_pointer_cast<Impl>(impl_));
    if (!result) { return result; }
    result = impl_->parameters.bindResources(commands);
    if (!result) { return result; }
    const auto root = impl_->parameters.root();
    const auto bytes = impl_->parameters.inlineData();
    result = bytes.empty() ? commands.bindExecution(impl_->execution, &root, sizeof(root))
        : commands.bindExecution(impl_->execution, bytes.data(), uint32_t(bytes.size()));
    if (!result) { return result; }
    return commands.traceRays(impl_->width, impl_->height, impl_->depth);
}

Result<> RayTracingKernel::dispatch(CommandBuffer& commands, const EncodedParameters& params,
    uint32_t width, uint32_t height, uint32_t depth) const
{
    auto prepared = prepareDispatch(params, width, height, depth);
    return prepared ? prepared->record(commands) : makeError(prepared.error());
}

} // namespace metallic::render
