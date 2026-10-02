#include "Runtime/Render/Core/ResidentLODParameters.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/ResidentMeshletLOD.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include <algorithm>

namespace metallic::render {
Result<> ResidentMeshletLOD::initialize(Device& device, uint32_t capacity, std::string& log)
{
    if (capacity == 0 || !visibilityRecordCapacityFitsId(capacity)) { return makeError(Error::InvalidArgument); }
    device_ = &device;
    capacity_ = capacity;
    Result<> result = device.createBuffer({.size = (uint64_t(capacity) + 1u) * 16u, .structureStride = 16,
        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { selections_ = std::move(rhiValue); });
    if (result) {
        result = device.createBuffer({.size = 36, .structureStride = 4,
            .usage = BufferUsageBits::Storage | BufferUsageBits::Indirect | BufferUsageBits::TransferSource,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { arguments_ = std::move(rhiValue); });
    }
    if (!result) { return result; }
    result = device.createBuffer({.size = (uint64_t(capacity) + (capacity + 63u) / 64u) * 4u,
        .structureStride = 4, .usage = BufferUsageBits::Storage}).transform([&](auto rhiValue) { scratch_ = std::move(rhiValue); });
    if (!result) { return result; }
    const char* entries[] = {"residentLodResetMain", "residentLodSelectMain", "residentLodArgumentsMain", "residentLodScatterMain"};
    for (size_t i = 0; i < kernels_.size(); ++i) {
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/GPUDriven/ResidentMeshletLOD",
            .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log += shader.diagnostics; return result; }
        result = kernels_[i].initialize(device, {.spirv = shader.spirv,
            .parameters = parameterAbi<ResidentLODParameters>(kResidentLODABI, ParameterTransport::InlinePush),
            .debugName = entries[i]}, log);
        if (!result) { return result; }
    }
    return {};
}

Result<> ResidentMeshletLOD::record(CommandBuffer& commands, ResourceRegistry& registry,
    const GPUSceneGlobalBufferViews& inputs, const MeshletLODView& view,
    GPUSceneRasterDrawRange candidates, uint32_t instanceCount, uint32_t groupCount, uint32_t manualLevel)
{
    // Provision for every input record. Selection can never overflow, even when
    // the camera crosses the near plane and requests the finest entire scene.
    if (candidates.count > capacity_) { return makeError(Error::InvalidArgument); }
    ParameterWriter writer(*device_, registry, commands.frameContext());
    ResidentLODParameters params{.eye = view.eye, .forward = view.forward, .projection = view.projection,
        .offset = candidates.offset, .count = candidates.count, .capacity = capacity_,
        .instanceCount = instanceCount, .groupCount = groupCount, .manualLevel = manualLevel};
    const GPUSceneBufferView* views[] = {&inputs.meshlets, &inputs.meshletDraws, &inputs.instances, &inputs.lodGroups};
    ShaderDataSpan* spans[] = {&params.clusters, &params.records, &params.instances, &params.groups};
    const uint32_t strides[] = {sizeof(GPUSceneGPUMeshletRecord), sizeof(GPUSceneGPUMeshletDrawRecord),
        sizeof(GPUSceneGPUInstanceRecord), sizeof(MeshletLODGroupRecord)};
    // A streaming-only scene has no resident inputs. Reset and argument generation
    // still run to clear results from a previously populated frame.
    if (candidates.count == 0) {
        params.offset = params.instanceCount = params.groupCount = 0;
    }
    for (size_t i = 0; candidates.count != 0 && i < std::size(views); ++i) {
        const auto& input = *views[i];
        if (input.buffer == nullptr) { return makeError(Error::InvalidArgument); }
        auto slice = input.buffer->slice({input.offset, input.size});
        if (!slice) { return makeError(slice.error()); }
        *spans[i] = writer.dataBuffer(*slice, strides[i], 16);
    }
    if (params.offset > params.records.count || params.count > params.records.count - params.offset ||
        params.instanceCount > params.instances.count || params.groupCount > params.groups.count) { return makeError(Error::InvalidArgument); }
    params.output = writer.dataBuffer(selections_.get(), 16, 16);
    params.arguments = writer.dataBuffer(arguments_.get(), 4, 4);
    params.scratch = writer.dataBuffer(scratch_.get(), 4, 4);
    auto encoded = writer.encode(params, kResidentLODABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    std::array<PreparedComputeDispatch, 4> dispatches;
    for (uint32_t stage = 0; stage < dispatches.size(); ++stage) {
        const uint32_t groups = (stage == 1 || stage == 3) ? (candidates.count + 63u) / 64u : 1u;
        if (groups == 0) { continue; }
        auto dispatch = kernels_[stage].prepareDispatch(*encoded, std::min(groups, 65535u), (groups + 65534u) / 65535u);
        if (!dispatch) { return makeError(dispatch.error()); }
        dispatches[stage] = std::move(*dispatch);
    }
    Result<> result;
    using namespace detail;
    using Access = RenderGraphResourceAccess;
    constexpr auto compute = RenderGraphPassKind::Compute;
    constexpr auto consumer = RenderGraphPassKind::Unsafe;
    Buffer* buffers[] = {selections_.get(), arguments_.get(), scratch_.get()};
    std::array<GraphAccessResource, 3> resources;
    std::array<GraphAccessBinding, 3> resourcesBound;
    for (size_t i = 0; i < resources.size(); ++i) {
        resources[i] = {.type = RenderGraphResourceType::Buffer,
            .state = initialized_ ? ResourceState::General : ResourceState::Undefined,
            .scope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
        auto slice = buffers[i]->slice();
        if (!slice) { return makeError(slice.error()); }
        resourcesBound[i].buffer = std::move(*slice);
    }
    const GraphAccessPass phases[] = {
        {.uses = {declaredGraphAccess(0, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(2, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferStorageWrite, compute),
            declaredGraphAccess(1, Access::BufferStorageWrite, compute),
            declaredGraphAccess(2, Access::BufferStorageReadWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferStorageWrite, compute),
            declaredGraphAccess(2, Access::BufferStorageRead, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferShaderRead, consumer),
            declaredGraphAccess(1, Access::BufferIndirectRead, consumer)}},
    };
    auto plan = buildGraphAccessPlan(resources, phases);
    if (!plan) { return makeError(plan.error()); }
    commands.beginDebugLabel({.name = "Resident adaptive meshlet LOD"});
    for (uint32_t stage = 0; stage < 4; ++stage) {
        result = recordGraphAccessBarriers(commands, plan->passes[stage], resourcesBound);
        if (!result) { commands.endDebugLabel(); return result; }
        if (dispatches[stage].valid()) {
            result = dispatches[stage].record(commands);
            if (!result) { commands.endDebugLabel(); return result; }
        }
    }
    result = recordGraphAccessBarriers(commands, plan->passes.back(), resourcesBound);
    commands.endDebugLabel();
    if (!result) { return result; }
    initialized_ = true;
    return {};
}

} // namespace metallic::render
